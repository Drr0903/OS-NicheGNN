"""Stage 14b (plan v2, amendment A9) — choose the representation for the niche atlas by niche stability and purity.

Variants (all patients, three seeds each): plain OS-NicheGNN; invariant; invariant with gated anchors; hybrid-invariant;
hybrid-invariant with gated anchors. Each seed's embedding is clustered exactly as in stage 15 (patient-balanced Leiden,
K = v2.k_fixed, centroid assignment) and the partition is scored: ARI vs patient, ARI vs region, spot-weighted niche purity,
max lung-parenchyma fraction of any niche, number of niches present in ≥ half of the patients, and the pairwise
seed-stability ARI. The pre-specified rule of A9 picks the variant; the choice is written to config-independent
results/14_model/variant_selection.tsv and variant_choice.json (stage 14 --final then uses it).
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.cluster import assign, cluster_prototypes  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.v2 import CLASSES, compute_anchors, load_cohort, make_graphs, train_and_embed  # noqa: E402

STAGE = "14_model"; VARIANTS = ["gnn", "invariant", "invariant_gated", "hybrid_invariant", "hybrid_invariant_gated"]


COMPARATORS = ["raw", "pca", "banksy", "pca_harmony", "banksy_harmony"]  # reference representations, scored but never selected
TRANSDUCTIVE_SUFFIX = "_harmonised"  # Harmony on a trained embedding — reported, but not a candidate for the atlas (it cannot place an unseen patient without being re-run)


def partition(emb: dict, G: dict, K: int, seed: int, cfg: dict) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed); Z, P = [], []
    for pat in sorted({g.patient for g in G.values()}):
        zs = np.vstack([emb[s] for s in G if G[s].patient == pat]); idx = rng.choice(len(zs), min(1500, len(zs)), replace=False); Z.append(zs[idx]); P += [pat] * len(idx)
    _, protos, info = cluster_prototypes(np.vstack(Z), K, seed, batch=np.array(P), k_within=cfg["niches"]["k_within"])
    lab = np.concatenate([assign(emb[s], protos)[0] for s in G]); pat = np.concatenate([[G[s].patient] * G[s].n for s in G]); reg = np.concatenate([G[s].labels for s in G]); ev = np.isin(reg, CLASSES)
    ct = pd.crosstab(lab, reg, normalize="index"); frac = pd.crosstab(pat, lab, normalize="index")
    return lab, {"ari_vs_patient": float(adjusted_rand_score(pat, lab)), "ari_vs_region": float(adjusted_rand_score(reg[ev], lab[ev])), "purity": float(np.average(ct.max(1), weights=pd.Series(lab).value_counts().reindex(ct.index).fillna(0))),
                 "max_lung_frac": float(ct.get("lung_parenchyma", pd.Series(0, index=ct.index)).max()), "n_present_half_patients": int(((frac >= 0.02).sum(0) >= frac.shape[0] / 2).sum()), "centroid_agreement": info["centroid_agreement_ari"]}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--variants", default=",".join(VARIANTS)); ap.add_argument("--tag", default=""); ap.add_argument("--reuse-metrics", action="store_true", dest="reuse_metrics", help="rebuild the summary, the choice and the provenance from the recorded per-seed metrics instead of re-embedding")
    args = ap.parse_args(); variants = args.variants.split(",")
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); v2 = cfg["v2"]; K = v2["k_fixed"]; sd = stage_dir(STAGE)
    if args.reuse_metrics:  # the per-seed metrics are recorded stage outputs; rebuilding the summary from them is deterministic
        df = pd.read_csv(sd / "variant_seed_metrics.tsv", sep="\t"); df = df[df["variant"].isin(variants)].copy()
        prev_sel = pd.read_csv(sd / "variant_selection.tsv", sep="\t"); stab = prev_sel[["variant", "seed_stability_ari"]].to_dict("records")
        lopo = pd.read_csv(sd / "lopo_metrics.tsv", sep="\t"); ep = lopo[lopo["method"] == "nichegnn_invariant"]["epochs_run"].dropna(); n_epochs = int(np.median(ep)) if len(ep) else 70
    else:
        adatas, labels, patient_of, site_of = load_cohort(cfg); secs = list(adatas); G, genes, mu, sd_ = make_graphs(cfg, adatas, labels, patient_of, secs, secs); gl = [G[s] for s in secs]
        lopo = pd.read_csv(sd / "lopo_metrics.tsv", sep="\t"); ep = lopo[lopo["method"] == "nichegnn_invariant"]["epochs_run"].dropna(); n_epochs = int(np.median(ep)) if len(ep) else 70
        cfg_final = json.loads(json.dumps(cfg)); cfg_final["train"]["max_epochs"] = n_epochs; cfg_final["train"]["patience"] = 10**9
        anchors = {False: compute_anchors(cfg, gl, gate=False), True: compute_anchors(cfg, gl, gate=True)}; log(f"anchors: {sum(len(v[0]) for v in anchors[False].values()) // 2} ungated, {sum(len(v[0]) for v in anchors[True].values()) // 2} gated pairs")
        rows, parts = [], {}
        base_emb = {}
        if set(variants) & set(COMPARATORS):  # A11 comparators are not trained; they are fitted once on all sections
            from sklearn.decomposition import PCA
            from lib.banksy import BanksyEmbedder
            from lib.v2 import harmony_correct
            base_emb["raw"] = {s: G[s].x for s in secs}
            pca = PCA(cfg["baselines"]["pca_dim"], random_state=cfg["seed"]).fit(np.vstack([G[s].x for s in secs])); base_emb["pca"] = {s: pca.transform(G[s].x) for s in secs}
            b = cfg["baselines"]["banksy"]; be = BanksyEmbedder(b["lambda"], b["k_geom"], b["n_pcs"], cfg["seed"]).fit(gl); base_emb["banksy"] = {s: be.transform(G[s]) for s in secs}
        for var in variants:
            for i in range(3):
                seed = cfg["seed"] + i; gate = "_gated" in var; anc = anchors[gate] if ("invariant" in var and "nbronly" not in var) else None
                if var in COMPARATORS:
                    emb = base_emb[var] if var in base_emb else harmony_correct(base_emb[var.split("_harmony")[0]], patient_of, seed); info = {"epochs_run": None}
                else:
                    base_var = var[:-len("_harmonised")] if var.endswith("_harmonised") else var  # Harmony applied to a trained embedding
                    emb, info = train_and_embed(cfg, base_var, gl, [], gl, seed, cfg["device"], cfg_final, anchors=anchors["_gated" in base_var] if ("invariant" in base_var and "nbronly" not in base_var) else None)
                    if var.endswith("_harmonised"):
                        from lib.v2 import harmony_correct as _hc
                        emb = _hc(emb, patient_of, seed)
                lab, m = partition(emb, G, K, seed, cfg); parts[(var, seed)] = lab; rows.append({"variant": var, "seed": seed, "comparator": var in COMPARATORS or var.endswith(TRANSDUCTIVE_SUFFIX), "epochs": info["epochs_run"], **m}); log(f"  {var} seed {i + 1}: {m}")
        df = pd.DataFrame(rows); stab = []
        for var in variants:
            ss = [s for (v, s) in parts if v == var]; aris = [adjusted_rand_score(parts[(var, a)], parts[(var, b)]) for a, b in itertools.combinations(ss, 2)]; stab.append({"variant": var, "seed_stability_ari": float(np.mean(aris))})
    summ = df.groupby("variant")[["ari_vs_patient", "ari_vs_region", "purity", "max_lung_frac", "n_present_half_patients", "centroid_agreement"]].mean().reset_index().merge(pd.DataFrame(stab), on="variant")
    summ["comparator"] = summ["variant"].isin(COMPARATORS) | summ["variant"].str.endswith(TRANSDUCTIVE_SUFFIX)  # transductive representations are reported, never selected
    prev = sd / "variant_selection.tsv"
    if args.tag and prev.exists():  # accumulate rounds (A9 → A10)
        old = pd.read_csv(prev, sep="\t"); summ = pd.concat([old[~old["variant"].isin(summ["variant"])].drop(columns=["eligible", "selected"], errors="ignore"), summ], ignore_index=True)
        summ["comparator"] = summ["variant"].isin(COMPARATORS) | summ["variant"].str.endswith(TRANSDUCTIVE_SUFFIX)
        df = pd.concat([pd.read_csv(sd / "variant_seed_metrics.tsv", sep="\t"), df], ignore_index=True).drop_duplicates(["variant", "seed"], keep="last")
    base_pur = float(summ.loc[summ["variant"] == "gnn", "purity"].iloc[0])
    summ["eligible"] = (summ["ari_vs_patient"] <= 0.10) & (summ["purity"] >= base_pur - 0.05) & (summ["max_lung_frac"] >= 0.5) & ~summ["comparator"]  # comparators are reported, never selected
    elig = summ[summ["eligible"]].sort_values("seed_stability_ari", ascending=False)
    fallback = summ[(summ["ari_vs_patient"] <= 0.10) & ~summ["comparator"]].sort_values("seed_stability_ari", ascending=False)  # A10 fallback: most stable patient-invariant variant
    choice = elig["variant"].iloc[0] if len(elig) else (fallback["variant"].iloc[0] if len(fallback) else "invariant")
    summ["selected"] = summ["variant"] == choice; write_table(df, sd / "variant_seed_metrics.tsv"); write_table(summ, sd / "variant_selection.tsv")
    ctrl = summ[summ["variant"].str.startswith("mlp") & ~summ["eligible"]]  # A11 encoder control: reported, ineligible by the same rule
    choice_doc = {"choice": choice, "rule": "A9/A10 + A13 (inductive representations only)", "eligible_rule_met": bool(len(elig)), "n_epochs": n_epochs,
                  "eligible": elig["variant"].tolist(), "reported_not_selected": summ.loc[summ["comparator"], "variant"].tolist(),
                  "ineligible_encoder_control": (f"{ctrl['variant'].iloc[0]} (patient index {ctrl['ari_vs_patient'].iloc[0]:.3f} > 0.10)" if len(ctrl) else None)}
    json.dump(choice_doc, open(sd / "variant_choice.json", "w"), indent=2); log(f"\n{summ.round(3).to_string()}\nchoice: {choice}")
    write_provenance(STAGE + "_variant_select", [sd / "lopo_metrics.tsv"], [sd / "variant_selection.tsv", sd / "variant_seed_metrics.tsv", sd / "variant_choice.json"], cfg, extra={"amendment": "A9", "choice": choice, "reused_metrics": bool(args.reuse_metrics)}, t_start=t0)


if __name__ == "__main__":
    with threadpool_limits(limits=8):
        main()
