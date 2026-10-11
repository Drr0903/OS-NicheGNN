"""Stage 14 (plan v2) — patient-invariant OS-NicheGNN on all 26 discovery sections (19 patients, five sites).

  --lopo   leave-one-patient-out region probe (19 folds) for PCA-50, BANKSY-20, plain OS-NicheGNN and the patient-invariant
           OS-NicheGNN (n_seeds each), plus training-embedding patient/region ARI of balanced-Leiden communities (K = v2.k_fixed);
           supports --patients / --tag parts and --merge tags.
  --final  train the frozen v2 model on all patients (epochs = median of the invariant LOPO runs), five seeds; write
           data/processed/frozen_v2/{model_seed*.pt, genes.txt, scaler.npz, meta.json}, data/processed/embeddings/v2/<seed>/<section>.npy
           and the patient-identity diagnostics of all representations on all spots.
Outputs results/14_model/{lopo_metrics.tsv, lopo_summary.tsv, lopo_patient_level.tsv, paired_tests.tsv, identity_diagnostics.tsv}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.banksy import BanksyEmbedder  # noqa: E402
from lib.evalprobe import evaluate_heldout  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.v2 import compute_anchors, harmony_correct, identity_index, load_cohort, make_graphs, train_and_embed  # noqa: E402

STAGE = "14_model"
torch.set_num_threads(8)  # four folds run in parallel on this 256-core host; unlimited intra-op threads oversubscribe the CPU by ~10×


def paired(a: pd.Series, b: pd.Series, seed: int) -> dict:
    d = (a - b).dropna(); rng = np.random.default_rng(seed)
    if len(d) < 3:
        return {"n_patients": len(d), "mean_diff": float(d.mean()) if len(d) else np.nan}
    boots = [rng.choice(d.to_numpy(), len(d), replace=True).mean() for _ in range(5000)]
    from scipy import stats
    return {"n_patients": len(d), "mean_diff": float(d.mean()), "ci95_lo": float(np.percentile(boots, 2.5)), "ci95_hi": float(np.percentile(boots, 97.5)),
            "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum()), "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if (d != 0).any() else 1.0}


def chosen_method() -> str:
    """LOPO method name of the variant selected by amendments A9/A10 (results/14_model/variant_choice.json), or the default."""
    f = stage_dir(STAGE) / "variant_choice.json"
    return "nichegnn_" + json.load(open(f))["choice"] if f.exists() else "nichegnn_invariant"


def run_lopo(cfg, adatas, labels, patient_of, site_of, patients, seeds, tag, methods=("pca", "banksy", "nichegnn", "nichegnn_invariant")):
    secs = list(adatas); all_pats = sorted(set(patient_of.values())); K = cfg["v2"]["k_fixed"]; rows = []; sd = stage_dir(STAGE)
    for i, P in enumerate(patients):
        test = [s for s in secs if patient_of[s] == P]; train = [s for s in secs if patient_of[s] != P]
        rng = np.random.default_rng(cfg["seed"] + all_pats.index(P)); tr_pats = sorted({patient_of[s] for s in train}); val_pat = tr_pats[int(rng.integers(len(tr_pats)))]
        val = [s for s in train if patient_of[s] == val_pat]; log(f"[fold {i + 1}/{len(patients)}] held-out {P} {test} ({[site_of[s] for s in test]}); inner val {val_pat}")
        G, genes, _, _ = make_graphs(cfg, adatas, labels, patient_of, train, train + test)
        tr_g = [G[s] for s in train if s not in val]; va_g = [G[s] for s in val]; te_g = [G[s] for s in test]; all_tr = tr_g + va_g; Gtr = {s: G[s] for s in train}
        def rec(method, seed, emb, info=None, t=0.0):
            m = evaluate_heldout(emb, all_tr, te_g, cfg, seed); m.pop("per_section_boundary", None)
            idx = identity_index({s: emb[s] for s in train}, Gtr, K, seed) if (method.startswith("nichegnn") or method.startswith("mlp") or "harmony" in method or method == "raw") else {}
            rows.append({"method": method, "seed": seed, "heldout_patient": P, "heldout_site": "+".join(sorted({site_of[s] for s in test})), "n_test_sections": len(test), "run_time_s": round(t, 1),
                         **m, **{f"train_{k}": v for k, v in idx.items()}, "epochs_run": info.get("epochs_run") if info else None})
            log(f"    {method} seed {seed}: F1 {m['macro_f1']:.3f} bF1 {m['boundary_f1']:.3f}" + (f" | train ARI pat {idx['ari_vs_patient']:.2f} reg {idx['ari_vs_region']:.2f}" if idx else ""))
        with threadpool_limits(limits=8):
            if "pca" in methods:
                t0 = time.time(); pca = PCA(cfg["baselines"]["pca_dim"], random_state=seeds[0]).fit(np.vstack([g.x for g in all_tr])); rec("pca", seeds[0], {s: pca.transform(G[s].x) for s in train + test}, t=time.time() - t0)
            if "banksy" in methods:
                b = cfg["baselines"]["banksy"]; t0 = time.time(); be = BanksyEmbedder(b["lambda"], b["k_geom"], b["n_pcs"], seeds[0]).fit(all_tr); rec("banksy", seeds[0], {s: be.transform(G[s]) for s in train + test}, t=time.time() - t0)
            if "raw" in methods:
                rec("raw", seeds[0], {s: G[s].x for s in train + test})
            if "pca_harmony" in methods:  # Harmony on PCA-50, patient as batch, train + held-out corrected jointly (transductive)
                t0 = time.time(); pca = PCA(cfg["baselines"]["pca_dim"], random_state=seeds[0]).fit(np.vstack([g.x for g in all_tr]))
                for seed in seeds[:3]:
                    rec("pca_harmony", seed, harmony_correct({s: pca.transform(G[s].x) for s in train + test}, patient_of, seed), t=time.time() - t0)
            if "banksy_harmony" in methods:
                b = cfg["baselines"]["banksy"]; t0 = time.time(); be = BanksyEmbedder(b["lambda"], b["k_geom"], b["n_pcs"], seeds[0]).fit(all_tr); emb0 = {s: be.transform(G[s]) for s in train + test}
                for seed in seeds[:3]:
                    rec("banksy_harmony", seed, harmony_correct(emb0, patient_of, seed), t=time.time() - t0)
        t0 = time.time(); anchors = compute_anchors(cfg, tr_g); log(f"    anchors: {sum(len(v[0]) for v in anchors.values()) // 2} mutual pairs ({time.time() - t0:.0f}s)")
        for variant, name in [("gnn" if n == "nichegnn" else n[len("nichegnn_"):], n) for n in methods if n.startswith("nichegnn")] + [(n, n) for n in methods if n.startswith("mlp")]:  # nichegnn_<variant> → variant; mlp_<variant> (A11 encoder control)
            for seed in seeds:
                t0 = time.time(); emb, info = train_and_embed(cfg, variant, tr_g, va_g, [G[s] for s in train + test], seed, cfg["device"], anchors=anchors); rec(name, seed, emb, info, time.time() - t0)
        pd.DataFrame(rows).to_csv(sd / f"lopo_metrics{tag}.partial.tsv", sep="\t", index=False)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, cfg: dict) -> None:
    sd = stage_dir(STAGE); pat = df.groupby(["method", "heldout_patient"])[["macro_f1", "boundary_f1", "accuracy", "balanced_accuracy"]].mean().reset_index()
    summ = pat.groupby("method").agg(macro_f1_patient_mean=("macro_f1", "mean"), macro_f1_patient_sd=("macro_f1", "std"), boundary_f1_patient_mean=("boundary_f1", "mean"), boundary_f1_patient_sd=("boundary_f1", "std"),
                                     accuracy_patient_mean=("accuracy", "mean"), n_patients=("heldout_patient", "nunique"), n_patients_macro=("macro_f1", "count")).reset_index()
    tr = df.groupby("method")[[c for c in df.columns if c.startswith("train_") and c != "train_K"]].mean().reset_index(); summ = summ.merge(tr, on="method", how="left")
    write_table(summ, sd / "lopo_summary.tsv"); write_table(pat, sd / "lopo_patient_level.tsv"); write_table(df, sd / "lopo_metrics.tsv")
    ref_name = chosen_method(); ref_name = ref_name if ref_name in set(pat["method"]) else "nichegnn_invariant"
    ref = pat[pat["method"] == ref_name].set_index("heldout_patient"); rows = []
    for m in sorted(set(pat["method"]) - {ref_name}):
        sub = pat[pat["method"] == m].set_index("heldout_patient")
        for metric in ["macro_f1", "boundary_f1", "accuracy"]:
            rows.append({"comparison": f"{m} − {ref_name}", "metric": metric, **paired(sub[metric], ref.reindex(sub.index)[metric], cfg["seed"])})
    write_table(pd.DataFrame(rows), sd / "paired_tests.tsv"); log(f"\n{summ.round(3).to_string()}\n{pd.DataFrame(rows).round(3).to_string()}")


def run_final(cfg, adatas, labels, patient_of, site_of):
    secs = list(adatas); sd = stage_dir(STAGE); K = cfg["v2"]["k_fixed"]
    G, genes, mu, sd_ = make_graphs(cfg, adatas, labels, patient_of, secs, secs)
    lopo = pd.read_csv(sd / "lopo_metrics.tsv", sep="\t"); m = chosen_method(); m = m if m in set(lopo["method"]) else "nichegnn_invariant"
    ep = lopo[lopo["method"] == m]["epochs_run"].dropna()
    n_epochs = int(cfg["v2"].get("frozen_epochs") or (int(np.median(ep)) if len(ep) else cfg["train"]["max_epochs"]))
    log(f"frozen epochs: {n_epochs}" + ("" if cfg["v2"].get("frozen_epochs") else f" (median of the {m} folds)"))
    cfg_final = json.loads(json.dumps(cfg)); cfg_final["train"]["max_epochs"] = n_epochs; cfg_final["train"]["patience"] = 10**9
    fro = ROOT / cfg["paths"]["processed"] / "frozen_v2"; fro.mkdir(exist_ok=True); open(fro / "genes.txt", "w").write("\n".join(genes)); np.savez(fro / "scaler.npz", mu=mu, sd=sd_)
    rows, infos = [], []
    with threadpool_limits(limits=8):
        pca = PCA(cfg["baselines"]["pca_dim"], random_state=cfg["seed"]).fit(np.vstack([G[s].x for s in secs])); rows.append({"representation": "PCA-50", **identity_index({s: pca.transform(G[s].x) for s in secs}, G, K, cfg["seed"])})
        b = cfg["baselines"]["banksy"]; be = BanksyEmbedder(b["lambda"], b["k_geom"], b["n_pcs"], cfg["seed"]).fit([G[s] for s in secs]); rows.append({"representation": "BANKSY-20", **identity_index({s: be.transform(G[s]) for s in secs}, G, K, cfg["seed"])})
    rows.append({"representation": "standardised expression", **identity_index({s: G[s].x for s in secs}, G, K, cfg["seed"])})
    with threadpool_limits(limits=8):  # A11 comparators: Harmony on PCA-50 / BANKSY-20 (three seeds; Harmony is stochastic)
        pca_e = {s: pca.transform(G[s].x) for s in secs}; ban_e = {s: be.transform(G[s]) for s in secs}
        for i in range(3):
            seed = cfg["seed"] + i
            rows.append({"representation": f"Harmony on PCA-50 seed {i + 1}", **identity_index(harmony_correct(pca_e, patient_of, seed), G, K, seed)})
            rows.append({"representation": f"Harmony on BANKSY-20 seed {i + 1}", **identity_index(harmony_correct(ban_e, patient_of, seed), G, K, seed)})
    choice = json.load(open(sd / "variant_choice.json"))["choice"] if (sd / "variant_choice.json").exists() else "invariant"  # amendment A9 selection
    anchors = compute_anchors(cfg, [G[s] for s in secs], gate=choice.endswith("_gated")) if "invariant" in choice else None; log(f"final variant: {choice}")
    for i in range(cfg["n_seeds"]):
        seed = cfg["seed"] + i
        emb, info = train_and_embed(cfg, "gnn", [G[s] for s in secs], [], [G[s] for s in secs], seed, cfg["device"], cfg_final); rows.append({"representation": f"OS-NicheGNN (plain) seed {i + 1}", **identity_index(emb, G, K, seed)})
        emb, info = train_and_embed(cfg, choice, [G[s] for s in secs], [], [G[s] for s in secs], seed, cfg["device"], cfg_final, anchors=anchors); rows.append({"representation": f"OS-NicheGNN ({choice.replace('_', ' ')}) seed {i + 1}", **identity_index(emb, G, K, seed)})
        if i < 3:  # A11 encoder control: MLP (no graph) with the same anchor-only objective
            emb_m, _ = train_and_embed(cfg, "mlp_" + choice, [G[s] for s in secs], [], [G[s] for s in secs], seed, cfg["device"], cfg_final, anchors=anchors); rows.append({"representation": f"MLP ({choice.replace('_', ' ')}) seed {i + 1}", **identity_index(emb_m, G, K, seed)})
        torch.save(info["model"].state_dict(), fro / f"model_seed{seed}.pt"); d = ROOT / cfg["paths"]["processed"] / "embeddings" / "v2" / str(seed); d.mkdir(parents=True, exist_ok=True)
        for s in secs:
            np.save(d / f"{s}.npy", emb[s])
        infos.append({k: v for k, v in info.items() if k not in ("history", "model")}); log(f"  seed {seed}: {rows[-1]}")
    write_table(pd.DataFrame(rows), sd / "identity_diagnostics.tsv")
    meta = {"plan": "v2", "variant": choice, "kind": info.get("kind", "nichegnn"), "K": K, "frozen_seed": cfg["seed"], "n_epochs": n_epochs, "n_genes": len(genes), "sections": secs, "patients": sorted(set(patient_of.values())), "sites": site_of, "model": cfg["v2"]["model"], "train_infos": infos}
    json.dump(meta, open(fro / "meta.json", "w"), indent=2); return meta


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--lopo", action="store_true"); ap.add_argument("--final", action="store_true"); ap.add_argument("--patients", default=""); ap.add_argument("--tag", default=""); ap.add_argument("--n_seeds", type=int, default=0); ap.add_argument("--merge", default=""); ap.add_argument("--methods", default="pca,banksy,nichegnn,nichegnn_invariant")
    args = ap.parse_args(); t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE)
    if args.merge:
        df = pd.concat([pd.read_csv(sd / f"lopo_metrics{t}.tsv", sep="\t") for t in args.merge.split(",")], ignore_index=True); summarise(df, cfg)
        write_provenance(STAGE + "_lopo", [sd / f"lopo_metrics{t}.tsv" for t in args.merge.split(",")], [sd / "lopo_summary.tsv", sd / "lopo_patient_level.tsv", sd / "lopo_metrics.tsv", sd / "paired_tests.tsv"], cfg, extra={"plan": "v2", "merged": args.merge}); return
    adatas, labels, patient_of, site_of = load_cohort(cfg); log(f"v2 cohort: {len(adatas)} sections, {len(set(patient_of.values()))} patients, sites {pd.Series(site_of).value_counts().to_dict()}")
    if args.lopo:
        pats = sorted(set(patient_of.values())); pats = [p for p in pats if not args.patients or p in args.patients.split(",")]; seeds = [cfg["seed"] + i for i in range(args.n_seeds or cfg["n_seeds"])]
        df = run_lopo(cfg, adatas, labels, patient_of, site_of, pats, seeds, args.tag, tuple(args.methods.split(","))); write_table(df, sd / f"lopo_metrics{args.tag}.tsv")
        if not args.tag:
            summarise(df, cfg)
        write_provenance(STAGE + f"_lopo{args.tag}", [ROOT / cfg["paths"]["processed"] / "qc"], [sd / f"lopo_metrics{args.tag}.tsv"], cfg, extra={"plan": "v2", "patients": pats, "seeds": seeds}, t_start=t0)
    if args.final:
        meta = run_final(cfg, adatas, labels, patient_of, site_of)
        write_provenance(STAGE + "_final", [sd / "lopo_metrics.tsv"], [ROOT / cfg["paths"]["processed"] / "frozen_v2" / "meta.json", sd / "identity_diagnostics.tsv"], cfg, extra={k: v for k, v in meta.items() if k != "train_infos"}, t_start=t0)


if __name__ == "__main__":
    with threadpool_limits(limits=8):
        main()
