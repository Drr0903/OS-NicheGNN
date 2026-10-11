"""Stage 07 — leave-one-patient-out evaluation of every representation on the discovery lung cohort.

For each held-out patient P (all sections of P together):
  • HVGs (n_hvg, excluding the T-cell hold-out set in the confirmatory configuration) and the gene
    scaler are fitted on the remaining patients only; one remaining patient is the inner validation set.
  • Representations: PCA · MLP-AE · NeighMean-AE · BANKSY · STAGATE (transductive) · OS-NicheGNN ·
    ablations (no_outer, no_diff, no_aux, no_edges, rewired).  Stochastic models: n_seeds seeds.
  • A single linear probe per representation (fitted on training patients' image-derived labels) gives
    patient-level macro-F1 / balanced accuracy and boundary F1 on P.
Outputs
  results/07_lopo/lopo_metrics.tsv        one row per (method, ablation, seed, held-out patient)
  results/07_lopo/lopo_summary.tsv        mean ± sd over patients and seeds, failures, run time
  results/07_lopo/training_curves.tsv
  data/processed/embeddings/lopo/<method>_<seed>/<section>.npy   held-out embeddings (for stage 08 stability)
Usage: python 07_lopo_cv.py [--methods a,b,c] [--config gene_holdout|full] [--patients P01,P02]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.banksy import BanksyEmbedder  # noqa: E402
from lib.data import build_graph, fit_scaler, holdout_genes, load_labels, load_qc_sections, select_hvgs  # noqa: E402
from lib.evalprobe import evaluate_heldout  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.train import embed_sections, train_model  # noqa: E402

STAGE = "07_lopo"
DEEP = {"nichegnn": [None, "no_outer", "no_diff", "no_aux", "no_edges", "rewired"], "mlp_ae": [None], "neigh_mean_ae": [None]}


def fold_graphs(cfg, adatas, labels, train_secs, test_secs, patient_of, exclude):
    tr_ad = [adatas[s] for s in train_secs]
    genes = select_hvgs(tr_ad, cfg["genes"]["n_hvg"], exclude)
    mu, sd = fit_scaler(tr_ad, genes)
    G = {s: build_graph(adatas[s], genes, mu, sd, cfg, patient_of[s], labels[s]) for s in train_secs + test_secs}
    return G, genes


def run_fold(cfg, G, genes, train_secs, val_secs, test_secs, methods, seeds, device, emb_root, fold_tag):
    rows, curves = [], []
    tr_g = [G[s] for s in train_secs if s not in val_secs]; va_g = [G[s] for s in val_secs]; te_g = [G[s] for s in test_secs]
    all_tr = tr_g + va_g
    def record(method, ablation, seed, emb, info=None, t=0.0, failed=False, err=""):
        base = {"method": method, "ablation": ablation or "full", "seed": seed, "heldout_patient": G[test_secs[0]].patient, "fold": fold_tag,
                "n_train_sections": len(all_tr), "n_test_sections": len(te_g), "run_time_s": round(t, 1), "failed": failed, "error": err}
        if failed:
            rows.append(base); return
        m = evaluate_heldout(emb, all_tr, te_g, cfg, seed)
        per = m.pop("per_section_boundary")
        rows.append({**base, **m, "epochs_run": info.get("epochs_run") if info else None, "n_params": info.get("n_params") if info else None,
                     "best_val_masked_mse": info.get("best_val_masked_mse") if info else None, "per_section_boundary": json.dumps(per)})
        d = emb_root / f"{method}_{ablation or 'full'}_{seed}"; d.mkdir(parents=True, exist_ok=True)
        for s in test_secs:
            np.save(d / f"{s}.npy", emb[s])
    if "pca" in methods:
        t0 = time.time(); pca = PCA(cfg["baselines"]["pca_dim"], random_state=seeds[0]).fit(np.vstack([g.x for g in all_tr]))
        record("pca", None, seeds[0], {s: pca.transform(G[s].x).astype(np.float32) for s in train_secs + test_secs}, t=time.time() - t0)
    if "banksy" in methods:
        b = cfg["baselines"]["banksy"]; t0 = time.time()
        be = BanksyEmbedder(b["lambda"], b["k_geom"], b["n_pcs"], seeds[0]).fit(all_tr)
        record("banksy", None, seeds[0], {s: be.transform(G[s]) for s in train_secs + test_secs}, t=time.time() - t0)
    if "banksy_50pc" in methods:
        b = cfg["baselines"]["banksy_50pc"]; t0 = time.time()
        be = BanksyEmbedder(b["lambda"], b["k_geom"], b["n_pcs"], seeds[0]).fit(all_tr)
        record("banksy_50pc", None, seeds[0], {s: be.transform(G[s]) for s in train_secs + test_secs}, t=time.time() - t0)
    if "stagate" in methods:
        from lib.stagate_wrap import stagate_embed
        for seed in seeds:
            t0 = time.time()
            try:
                emb = {s: stagate_embed(G[s], cfg, seed, device) for s in train_secs + test_secs}
                record("stagate_transductive", None, seed, emb, t=time.time() - t0)
            except Exception as e:  # noqa: BLE001
                record("stagate_transductive", None, seed, None, t=time.time() - t0, failed=True, err=repr(e)[:200])
    for kind, ablations in DEEP.items():
        if kind not in methods:
            continue
        for abl in ablations:
            if kind == "nichegnn" and abl is not None and "ablations" not in methods:
                continue
            for seed in seeds:
                t0 = time.time()
                try:
                    model, info = train_model(kind, tr_g, va_g, cfg, seed, ablation=abl, device=device)
                    emb = embed_sections(model, [G[s] for s in train_secs + test_secs], kind, cfg, device, abl, seed)
                    for h in info["history"]:
                        curves.append({"method": kind, "ablation": abl or "full", "seed": seed, "fold": fold_tag, **h})
                    record(kind, abl, seed, emb, info, time.time() - t0)
                    log(f"    {kind}/{abl or 'full'} seed {seed}: F1 {rows[-1]['macro_f1']:.3f} bF1 {rows[-1]['boundary_f1']:.3f} ({info['epochs_run']} ep, {time.time() - t0:.0f}s)")
                except Exception as e:  # noqa: BLE001
                    log(f"    {kind}/{abl} seed {seed} FAILED: {e!r}")
                    record(kind, abl, seed, None, t=time.time() - t0, failed=True, err=repr(e)[:200])
    return rows, curves


def summarise(df: pd.DataFrame, sd: Path, tag: str) -> None:
    ok = df[~df["failed"]]
    summ = ok.groupby(["method", "ablation"]).agg(n_runs=("macro_f1", "size"), accuracy_mean=("accuracy", "mean"), macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
                                                  bacc_mean=("balanced_accuracy", "mean"), boundary_f1_mean=("boundary_f1", "mean"), boundary_f1_sd=("boundary_f1", "std"),
                                                  run_time_s_mean=("run_time_s", "mean"), n_params=("n_params", "first")).reset_index()
    fails = df.groupby(["method", "ablation"])["failed"].sum().reset_index().rename(columns={"failed": "n_failed"})
    summ = summ.merge(fails, on=["method", "ablation"], how="left")
    # patient-level means first (average seeds within patient), then across patients
    pat = ok.groupby(["method", "ablation", "heldout_patient"])[["macro_f1", "boundary_f1", "accuracy", "balanced_accuracy"]].mean().reset_index()
    pat_summ = pat.groupby(["method", "ablation"]).agg(macro_f1_patient_mean=("macro_f1", "mean"), macro_f1_patient_sd=("macro_f1", "std"),
                                                       boundary_f1_patient_mean=("boundary_f1", "mean"), boundary_f1_patient_sd=("boundary_f1", "std"), accuracy_patient_mean=("accuracy", "mean"),
                                                       n_patients=("heldout_patient", "nunique"), n_patients_macro=("macro_f1", "count")).reset_index()
    summ = summ.merge(pat_summ, on=["method", "ablation"])
    write_table(summ.sort_values("macro_f1_patient_mean", ascending=False), sd / f"lopo_summary_{tag}.tsv")
    write_table(pat, sd / f"lopo_patient_level_{tag}.tsv")
    log(f"\n{summ[['method', 'ablation', 'n_runs', 'accuracy_patient_mean', 'macro_f1_patient_mean', 'boundary_f1_patient_mean', 'n_failed', 'run_time_s_mean']].round(3).to_string()}")


def merge_parts(cfg: dict, config: str, tags: list[str]) -> None:
    sd = stage_dir(STAGE)
    df = pd.concat([pd.read_csv(sd / f"lopo_metrics_{config}{t}.tsv", sep="\t") for t in tags], ignore_index=True)
    write_table(df, sd / f"lopo_metrics_{config}.tsv")
    cur = [sd / f"training_curves_{config}{t}.tsv" for t in tags if (sd / f"training_curves_{config}{t}.tsv").exists()]
    if cur:
        write_table(pd.concat([pd.read_csv(c, sep="\t") for c in cur], ignore_index=True), sd / f"training_curves_{config}.tsv")
    summarise(df, sd, config)
    write_provenance(STAGE + f"_{config}", [sd / f"lopo_metrics_{config}{t}.tsv" for t in tags], [sd / f"lopo_metrics_{config}.tsv", sd / f"lopo_summary_{config}.tsv"], cfg,
                     extra={"merged_parts": tags, "n_patients": int(df["heldout_patient"].nunique())})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="pca,banksy,stagate,mlp_ae,neigh_mean_ae,nichegnn,ablations")
    ap.add_argument("--config", default="gene_holdout", choices=["gene_holdout", "full"])
    ap.add_argument("--patients", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--n_seeds", type=int, default=0, help="override config n_seeds (smoke tests)")
    ap.add_argument("--merge", default="", help="comma-separated tags of partial runs to merge into the untagged result")
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_config(); set_global_seed(cfg["seed"])
    if args.merge:
        merge_parts(cfg, args.config, args.merge.split(",")); return
    disc = cfg["cohorts"]["discovery"]
    secs = disc["lung_sections"]; patient_of = disc["patient_groups"]
    man = pd.read_csv(ROOT / "results/01_manifest/sample_manifest.tsv", sep="\t", dtype=str).set_index("section_id")
    secs = [s for s in secs if (ROOT / cfg["paths"]["processed"] / "qc" / f"{s}.h5ad").exists()]
    adatas = load_qc_sections(cfg, secs)
    labels = {s: load_labels(cfg, s) for s in secs}
    exclude = holdout_genes(cfg) if args.config == "gene_holdout" else set()
    patients = sorted({patient_of[s] for s in secs})
    if args.patients:
        patients = [p for p in patients if p in args.patients.split(",")]
    methods = args.methods.split(",")
    seeds = [cfg["seed"] + i for i in range(args.n_seeds or cfg["n_seeds"])]
    emb_root = ROOT / cfg["paths"]["processed"] / "embeddings" / f"lopo_{args.config}{args.tag}"
    rows, curves = [], []
    all_pats = sorted({patient_of[s] for s in secs})
    for i, P in enumerate(patients):
        test_secs = [s for s in secs if patient_of[s] == P]
        train_secs = [s for s in secs if patient_of[s] != P]
        # inner validation patient: deterministic per held-out patient (independent of how folds are partitioned across processes)
        tr_pats = sorted({patient_of[s] for s in train_secs})
        rng = np.random.default_rng(cfg["seed"] + all_pats.index(P))
        val_pat = tr_pats[int(rng.integers(len(tr_pats)))]
        val_secs = [s for s in train_secs if patient_of[s] == val_pat]
        log(f"[fold {i + 1}/{len(patients)}] held-out {P} {test_secs}; inner val {val_pat}")
        G, genes = fold_graphs(cfg, adatas, labels, train_secs, test_secs, patient_of, exclude)
        r, c = run_fold(cfg, G, genes, train_secs, val_secs, test_secs, methods, seeds, cfg["device"], emb_root, f"LOPO_{P}")
        rows += r; curves += c
        pd.DataFrame(rows).to_csv(stage_dir(STAGE) / f"lopo_metrics_{args.config}{args.tag}.partial.tsv", sep="\t", index=False)
    sd = stage_dir(STAGE)
    df = pd.DataFrame(rows); write_table(df, sd / f"lopo_metrics_{args.config}{args.tag}.tsv")
    if curves:
        write_table(pd.DataFrame(curves), sd / f"training_curves_{args.config}{args.tag}.tsv")
    summarise(df, sd, f"{args.config}{args.tag}")
    write_provenance(STAGE + f"_{args.config}{args.tag}", [ROOT / cfg["paths"]["processed"] / "qc"], [sd / f"lopo_metrics_{args.config}{args.tag}.tsv", sd / f"lopo_summary_{args.config}{args.tag}.tsv"], cfg,
                     extra={"methods": methods, "config": args.config, "patients": patients, "seeds": seeds}, t_start=t0)


if __name__ == "__main__":
    main()
