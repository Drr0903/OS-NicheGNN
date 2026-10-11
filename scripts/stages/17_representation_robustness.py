"""Stage 17 (plan v2, ) — does the niche atlas depend on the representation?

A second atlas is built on Harmony-corrected PCA-50 (patient as batch; transductive by construction) with the identical protocol
used for the frozen atlas: consensus of the seeds on a patient-balanced sample, Leiden with K = v2.k_fixed, centroid prototypes,
nearest-prototype assignment and the 97.5 % unmatched rule. The two atlases are then compared on the spot partition, the niche
correspondence, the composition of the matched niches, the boundary odds ratios, the T-cell ordering and the pre-specified classes.

Outputs (results/17_robustness/): spot_niches_harmony.tsv, atlas_agreement.json, niche_correspondence.tsv,
niche_composition_harmony.tsv, niche_boundary_enrichment_harmony.tsv, niche_tcell_harmony.tsv, niche_classification_harmony.tsv.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.cluster import assign, atlas_batch, consensus_prototypes  # noqa: E402
from lib.data import build_graph, ring_neighbors  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.v2 import harmony_correct, load_cohort  # noqa: E402

STAGE = "17_robustness"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

s15 = import_module("15_niche_atlas")


def build_atlas(emb: dict, patient_of: dict, cfg: dict, K: int, n_seeds: int, seed0: int, harmonise: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    """Consensus partition over n_seeds Harmony runs (or of one fixed embedding), prototypes and per-spot assignment."""
    Zs, P = [], None
    for i in range(n_seeds):
        e = harmony_correct(emb, patient_of, seed0 + i) if harmonise else emb
        Z, Pi = s15.balanced_sample(e, patient_of, 1500, np.random.default_rng(seed0))
        Zs.append(Z); P = Pi if P is None else P
        if i == 0:
            e0 = e
    lab, protos, info = consensus_prototypes(Zs, K, seed0, batch=atlas_batch(cfg, P), k_within=cfg["niches"]["k_within"])
    return lab, protos, {"info": info, "emb0": e0}


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); v2 = cfg["v2"]; K = v2["k_fixed"]; sd = stage_dir(STAGE)
    adatas, labels, patient_of, site_of = load_cohort(cfg); secs = list(adatas); proc = ROOT / cfg["paths"]["processed"]
    fro = proc / cfg["atlas"]["frozen_dir"]; genes = open(fro / "genes.txt").read().split(); sc_ = np.load(fro / "scaler.npz")
    G = {s: build_graph(adatas[s], genes, sc_["mu"], sc_["sd"], cfg, patient_of[s], labels[s]) for s in secs}
    with threadpool_limits(limits=8):
        pca = PCA(cfg["baselines"]["pca_dim"], random_state=cfg["seed"]).fit(np.vstack([G[s].x for s in secs]))
        emb0 = {s: pca.transform(G[s].x).astype(np.float32) for s in secs}
        lab_c, protos, extra = build_atlas(emb0, patient_of, cfg, K, cfg["n_seeds"], cfg["seed"], harmonise=True)
    log(f"Harmony-PCA atlas: per-seed agreement with consensus {np.round(extra['info']['seed_agreement_with_consensus'], 2).tolist()}")
    emb = extra["emb0"]; rows = []
    for s in secs:
        lab, dist = assign(emb[s], protos); g = G[s]
        rows.append(pd.DataFrame({"section": s, "patient": patient_of[s], "site": site_of[s], "barcode": g.barcodes, "niche": lab, "dist_to_prototype": dist,
                                  "histology_label": g.labels, "tumor_boundary": g.boundary, "x": g.xy[:, 0], "y": g.xy[:, 1]}))
    sp = pd.concat(rows, ignore_index=True); thr = float(np.quantile(sp["dist_to_prototype"], cfg["niches"]["unmatched_quantile"])); sp["unmatched"] = sp["dist_to_prototype"] > thr
    # frozen atlas for comparison
    fz = pd.read_csv(ROOT / "results/15_niches/spot_niches.tsv", sep="\t"); meta_f = json.load(open(ROOT / "results/15_niches/meta.json")); names_f = {int(k): v for k, v in meta_f["names"].items()}
    key = ["section", "barcode"]; m = fz[key + ["niche", "unmatched"]].merge(sp[key + ["niche", "unmatched"]], on=key, suffixes=("_frozen", "_harmony"))
    both = m[~m["unmatched_frozen"].astype(bool) & ~m["unmatched_harmony"].astype(bool)]
    ari_all = float(adjusted_rand_score(m["niche_frozen"], m["niche_harmony"])); ari_matched = float(adjusted_rand_score(both["niche_frozen"], both["niche_harmony"]))
    ct = pd.crosstab(m["niche_frozen"], m["niche_harmony"], normalize="index")
    ct_rev = ct / ct.sum(axis=0)  # a high forward fraction can also mean the partner simply swallowed a larger region
    corr_rows = [{"niche_frozen": int(k), "name_frozen": names_f[int(k)], "best_match_harmony": int(ct.loc[k].idxmax()),
                  "frac_of_frozen_niche": float(ct.loc[k].max()),
                  "frac_of_harmony_niche": float(ct_rev.loc[k, ct.loc[k].idxmax()]),
                  "jaccard": float(ct.loc[k, ct.loc[k].idxmax()] / (ct.loc[k].sum() + ct[ct.loc[k].idxmax()].sum() - ct.loc[k, ct.loc[k].idxmax()]))}
                 for k in ct.index]
    # characterise the Harmony atlas the same way
    w_dir = ROOT / "results/05_deconvolution/weights"; W = []
    for s in secs:
        w = pd.read_csv(w_dir / f"{s}_rctd_weights.tsv", sep="\t").set_index("barcode"); sub = sp[sp["section"] == s].set_index("barcode"); w = w.reindex(sub.index); w["niche"] = sub["niche"].values; W.append(w)
    W = pd.concat(W); types = [c for c in W.columns if c != "niche"]; comp_h = W.groupby("niche")[types].mean(); write_table(comp_h.reset_index(), sd / "niche_composition_harmony.tsv")
    comp_f = pd.read_csv(ROOT / "results/15_niches/niche_composition_rctd.tsv", sep="\t").set_index("niche")
    for r in corr_rows:
        a = comp_f.loc[r["niche_frozen"], types].to_numpy(float); b = comp_h.loc[r["best_match_harmony"], types].to_numpy(float)
        r["composition_pearson_r"] = float(np.corrcoef(a, b)[0, 1])
    # the correlation of composition between any two niches is high (fibroblast dominates every profile), so the matched
    # values are reported against the null of all 8 x 8 pairings and with the rank of the matched partner among the eight.
    Rm = np.array([[float(np.corrcoef(comp_f.loc[a, types].to_numpy(float), comp_h.loc[b, types].to_numpy(float))[0, 1]) for b in comp_h.index] for a in comp_f.index])
    for i, rr in enumerate(corr_rows):
        j = list(comp_h.index).index(rr["best_match_harmony"]); rr["rank_of_match_among_8"] = int((Rm[i] > Rm[i, j]).sum() + 1)
    null = {"median_r_all_pairings": float(np.median(Rm)), "frac_all_pairings_r_ge_0.9": float((Rm >= 0.9).mean()), "n_pairings": int(Rm.size),
            "n_matched_rank1": int(sum(rr["rank_of_match_among_8"] == 1 for rr in corr_rows)), "max_matched_rank": int(max(rr["rank_of_match_among_8"] for rr in corr_rows))}
    write_table(pd.DataFrame(corr_rows), sd / "niche_correspondence.tsv"); log(f"\n{pd.DataFrame(corr_rows).round(3).to_string()}\nnull: {null}")
    hist_h = pd.crosstab(sp["niche"], sp["histology_label"], normalize="index")
    near_b = np.zeros(len(sp), bool); off = 0
    for s in secs:
        g = G[s]; nb = ring_neighbors(g.xy, g.pitch, rings=(1,)); b = g.boundary; near_b[off:off + g.n] = [b[i] or (len(nb[i]) > 0 and b[nb[i]].any()) for i in range(g.n)]; off += g.n
    matched = ~sp["unmatched"].astype(bool).to_numpy(); enr = []
    for k in range(K):
        mm = (sp["niche"] == k).to_numpy() & matched; a_, b_ = (mm & near_b).sum(), (~mm & matched & near_b).sum(); c_, d_ = (mm & ~near_b).sum(), (~mm & matched & ~near_b).sum()
        enr.append({"niche": k, "n_spots": int(mm.sum()), "frac_near_boundary": a_ / max(a_ + c_, 1), "odds_ratio_near_boundary": (a_ * d_) / max(b_ * c_, 1), "tumor_frac_histology": float(hist_h.loc[k].get("tumor_cellular", 0))})
    enr = pd.DataFrame(enr); write_table(enr, sd / "niche_boundary_enrichment_harmony.tsv")
    by_sec = pd.crosstab(sp[matched]["section"], sp[matched]["niche"], normalize="index").reindex(columns=range(K), fill_value=0.0)
    cls_h = s15.classify(by_sec, site_of, K); write_table(cls_h, sd / "niche_classification_harmony.tsv")
    cls_f = pd.read_csv(ROOT / "results/15_niches/niche_classification.tsv", sep="\t").set_index("niche")
    cls_agree = sum(1 for r in corr_rows if cls_f.loc[r["niche_frozen"], "class"] == cls_h.set_index("niche").loc[r["best_match_harmony"], "class"])
    # T-cell score per Harmony niche (patient level)
    import anndata as ad
    from lib.interface import t_score
    ts_rows = []
    for s in secs:
        a = ad.read_h5ad(proc / "qc" / f"{s}.h5ad"); ts = t_score(a, cfg["genes"]["t_cell_holdout"]); sub = sp[sp["section"] == s].set_index("barcode").reindex(a.obs_names)
        for k in range(K):
            mm = (sub["niche"] == k).to_numpy() & (~sub["unmatched"].fillna(True).to_numpy(bool))
            if mm.sum() >= 20:
                ts_rows.append({"section": s, "patient": patient_of[s], "niche": k, "n_spots": int(mm.sum()), "t_score_mean": float(ts[mm].mean())})
    tsd = pd.DataFrame(ts_rows).groupby(["patient", "niche"])["t_score_mean"].mean().groupby("niche").agg(["mean", "std", "count"]).reset_index(); write_table(tsd, sd / "niche_tcell_harmony.tsv")
    ts_f = pd.read_csv(ROOT / "results/15_niches/niche_tcell_patient_level.tsv", sep="\t").set_index("niche")
    from scipy.stats import spearmanr
    rho_t = float(spearmanr([ts_f.loc[r["niche_frozen"], "mean"] for r in corr_rows],
                            [float(tsd.set_index("niche")["mean"].get(r["best_match_harmony"], np.nan)) for r in corr_rows], nan_policy="omit")[0])
    agree = {"ari_all_spots": ari_all, "ari_matched_spots": ari_matched, "n_matched_niches_r_ge_0.9": int(sum(r["composition_pearson_r"] >= 0.9 for r in corr_rows)),
             "n_distinct_harmony_partners": int(len({r["best_match_harmony"] for r in corr_rows})), "mean_composition_r": float(np.mean([r["composition_pearson_r"] for r in corr_rows])),
             "class_agreement_of_matched": cls_agree, "spearman_tcell_order": rho_t, "boundary_niche_frozen": meta_f["interface_niche"],
             "boundary_niche_harmony_best_match": int([r["best_match_harmony"] for r in corr_rows if r["niche_frozen"] == meta_f["interface_niche"]][0]),
             "boundary_or_frozen": float(pd.read_csv(ROOT / "results/15_niches/niche_boundary_enrichment.tsv", sep="\t").set_index("niche").loc[meta_f["interface_niche"], "odds_ratio_near_boundary"]),
             "boundary_or_harmony_best_match": float(enr.set_index("niche").loc[[r["best_match_harmony"] for r in corr_rows if r["niche_frozen"] == meta_f["interface_niche"]][0], "odds_ratio_near_boundary"]),
             "unmatched_frac_harmony": float(sp["unmatched"].mean()), "seed_agreement_with_consensus": extra["info"]["seed_agreement_with_consensus"],
             "criterion_met": bool(ari_all >= 0.5 and sum(r["composition_pearson_r"] >= 0.9 for r in corr_rows) >= 6), **null}
    json.dump(agree, open(sd / "atlas_agreement.json", "w"), indent=2); log(json.dumps(agree, indent=2))
    write_table(sp, sd / "spot_niches_harmony.tsv")
    write_provenance(STAGE, [ROOT / "results/15_niches/spot_niches.tsv"], [sd / "atlas_agreement.json", sd / "niche_correspondence.tsv"], cfg, extra={"plan": "v2", "amendment": "A12", "K": K}, t_start=t0)


if __name__ == "__main__":
    main()
