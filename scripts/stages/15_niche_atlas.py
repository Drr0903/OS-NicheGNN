"""Stage 15 (plan v2, Aim 1) — metastatic niche atlas across all sites from the frozen v2 patient-invariant model.

Niches = balanced-Leiden communities (K = v2.k_fixed) of a patient-balanced sample of the frozen embedding, prototypes = centroids,
all spots assigned by nearest prototype (97.5 % unmatched rule). Characterisation: RCTD composition, region-label composition,
boundary enrichment, T-cell score, per-section / per-site distribution and the shared / site-specific / primary-only classification
of the plan, patient-level pseudobulk markers, niche–niche adjacency enrichment, K = 6/10 sensitivity, seed stability, and the
lung-interface sub-analysis (v1 machinery) for the fibroblast-rich boundary niche.
Outputs results/15_niches/*.tsv, niche_names.json, meta.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.v2 import CLASSES  # noqa: E402
from lib.cluster import assign, atlas_batch, integrate, cluster_prototypes, consensus_prototypes, leiden_fixed_k  # noqa: E402
from lib.inference import pooled_models  # noqa: E402
from lib.interface import run_interface_cohort, t_score  # noqa: E402
from lib.niches_v2 import adjacency_enrichment, common_genes, name_niches, niche_markers, pseudobulk  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.spatial import ring_neighbors  # noqa: E402
from lib.v2 import load_cohort, make_graphs  # noqa: E402

STAGE = "15_niches"
RETRAIN = os.environ.get("PROJECT018_ATLAS_RETRAIN") == "1"
VARIANT = os.environ.get("PROJECT018_ATLAS_VARIANT", "")  # A19 candidate encoder (stage 24 --variant)  # stage 19: the same atlas from the re-trained encoder


def atlas_dirs(cfg: dict, sd: Path) -> tuple[Path, Path, Path]:
    """Frozen encoder directory, embedding directory and output directory of the atlas (A19: config `atlas`)."""
    a = cfg["atlas"]; proc = ROOT / cfg["paths"]["processed"]
    if VARIANT:  # a candidate encoder's atlas, written outside results/
        out = proc / f"atlas_{VARIANT}" / STAGE; out.mkdir(parents=True, exist_ok=True)
        return proc / f"frozen_{VARIANT}", proc / "embeddings" / VARIANT, out
    if RETRAIN:
        out = ROOT / a["retrain_results"] / STAGE; out.mkdir(parents=True, exist_ok=True)
        return proc / a["retrain_frozen_dir"], proc / "embeddings" / a["retrain_emb_dir"], out
    return proc / a["frozen_dir"], proc / "embeddings" / a["emb_dir"], sd


def balanced_sample(emb: dict, patient_of: dict, per_patient: int, rng) -> tuple[np.ndarray, np.ndarray]:
    Z, P = [], []
    for pat in sorted(set(patient_of.values())):
        zs = np.vstack([emb[s] for s in emb if patient_of[s] == pat]); idx = rng.choice(len(zs), min(per_patient, len(zs)), replace=False); Z.append(zs[idx]); P += [pat] * len(idx)
    return np.vstack(Z), np.array(P)


def classify(by_sec: pd.DataFrame, site_of: dict, K: int) -> pd.DataFrame:
    met = [s for s in by_sec.index if site_of[s] != "bone"]; prim = [s for s in by_sec.index if site_of[s] == "bone"]; rows = []
    for k in range(K):
        col = str(k) if str(k) in by_sec.columns else k; pres = by_sec[col] >= 0.02
        n_met = int(pres.loc[met].sum()); sites_met = {site_of[s] for s in met if pres.loc[s]}; n_prim = int(pres.loc[prim].sum())
        per_site = {site: float(np.mean([pres.loc[s] for s in by_sec.index if site_of[s] == site])) for site in sorted(set(site_of.values()))}
        if n_met >= len(met) / 2 and len(sites_met) >= 2:
            cls = "shared metastatic"
        elif any(v >= 0.5 for st, v in per_site.items() if st != "bone") and sum(v >= 0.5 for st, v in per_site.items()) == 1:
            cls = "site-specific (" + [st for st, v in per_site.items() if v >= 0.5][0] + ")"
        elif n_prim >= len(prim) / 2 and n_met < len(met) / 2:
            cls = "primary-predominant"
        else:
            cls = "sporadic"
        rows.append({"niche": k, "n_metastatic_sections_present": n_met, "n_metastatic_sites_present": len(sites_met), "n_primary_sections_present": n_prim, **{f"present_frac_{st}": v for st, v in per_site.items()}, "class": cls})
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); v2 = cfg["v2"]; K = v2["k_fixed"]; sd = stage_dir(STAGE)
    adatas, labels, patient_of, site_of = load_cohort(cfg); secs = list(adatas); proc = ROOT / cfg["paths"]["processed"]
    fro, emb_dir, sd = atlas_dirs(cfg, sd)
    genes = open(fro / "genes.txt").read().split(); sc_ = np.load(fro / "scaler.npz"); meta_m = json.load(open(fro / "meta.json"))
    from lib.data import build_graph
    G = {s: build_graph(adatas[s], genes, sc_["mu"], sc_["sd"], cfg, patient_of[s], labels[s]) for s in secs}
    embs = {seed: integrate({s: np.load(emb_dir / str(seed) / f"{s}.npy") for s in secs}, patient_of, cfg["seed"], cfg) for seed in [cfg["seed"] + i for i in range(cfg["n_seeds"])]}
    emb = embs[cfg["seed"]]; rng = np.random.default_rng(cfg["seed"]); Z, P = balanced_sample(emb, patient_of, 1500, rng)
    # the same balanced rows in every seed's embedding → consensus partition → prototypes in the seed-1 embedding
    Zs = [Z] + [balanced_sample(embs[sd_], patient_of, 1500, np.random.default_rng(cfg["seed"]))[0] for sd_ in embs if sd_ != cfg["seed"]]
    lab_c, protos, cinfo = consensus_prototypes(Zs, K, cfg["seed"], batch=atlas_batch(cfg, P), k_within=cfg["niches"]["k_within"]); np.save(sd / "prototypes.npy", protos)
    log(f"K = {K}: consensus of {cinfo['n_seeds']} seeds, per-seed agreement ARI {np.round(cinfo['seed_agreement_with_consensus'], 2).tolist()}, centroid agreement ARI {cinfo['centroid_agreement_ari']:.2f}")
    rows = []
    for s in secs:
        lab, dist = assign(emb[s], protos); g = G[s]
        rows.append(pd.DataFrame({"section": s, "patient": patient_of[s], "site": site_of[s], "barcode": g.barcodes, "niche": lab, "dist_to_prototype": dist, "histology_label": g.labels, "tumor_boundary": g.boundary, "tumor_lung_interface": g.interface, "x": g.xy[:, 0], "y": g.xy[:, 1]}))
    spots = pd.concat(rows, ignore_index=True); thr = float(np.quantile(spots["dist_to_prototype"], cfg["niches"]["unmatched_quantile"])); spots["unmatched"] = spots["dist_to_prototype"] > thr
    # K sensitivity and seed stability
    ksens = []
    for k in [K] + list(v2["k_sensitivity"]):
        lk, _ = leiden_fixed_k(Z, k, cfg["seed"], batch=atlas_batch(cfg, P), k_within=cfg["niches"]["k_within"]); frac = pd.crosstab(P, lk, normalize="index"); ksens.append({"k": k, "n_present_half_patients": int(((frac >= 0.02).sum(0) >= frac.shape[0] / 2).sum()), "ari_vs_fixed_k": float(adjusted_rand_score(lab_c, lk))})
    stab = []
    for seed, e in embs.items():
        if seed == cfg["seed"]:
            continue
        Zs, Ps = balanced_sample(e, patient_of, 1500, np.random.default_rng(cfg["seed"])); _, pr, _ = cluster_prototypes(Zs, K, cfg["seed"], batch=atlas_batch(cfg, Ps), k_within=cfg["niches"]["k_within"])
        ls = np.concatenate([assign(e[s], pr)[0] for s in secs]); stab.append({"seed": seed, "ari_vs_frozen_seed": float(adjusted_rand_score(spots["niche"], ls)), "ari_with_consensus_sample": cinfo["seed_agreement_with_consensus"][list(embs).index(seed)]})
    write_table(pd.DataFrame(ksens), sd / "k_sensitivity.tsv"); write_table(pd.DataFrame(stab), sd / "seed_stability.tsv")
    # characterisation
    w_dir = ROOT / "results/05_deconvolution/weights"; W = []
    for s in secs:
        w = pd.read_csv(w_dir / f"{s}_rctd_weights.tsv", sep="\t").set_index("barcode"); sub = spots[spots["section"] == s].set_index("barcode"); w = w.reindex(sub.index); w["niche"] = sub["niche"].values; w["section"] = s; W.append(w)
    W = pd.concat(W); types = [c for c in W.columns if c not in ("niche", "section")]; comp = W.groupby("niche")[types].mean(); write_table(comp.reset_index(), sd / "niche_composition_rctd.tsv")
    spots = spots.merge(W.reset_index()[["barcode", "section"] + types], on=["barcode", "section"], how="left")
    hist = pd.crosstab(spots["niche"], spots["histology_label"], normalize="index"); write_table(hist.reset_index(), sd / "niche_histology.tsv")
    by_sec = pd.crosstab(spots["section"], spots["niche"], normalize="index"); write_table(by_sec.reset_index().assign(site=lambda d: d["section"].map(site_of), patient=lambda d: d["section"].map(patient_of)), sd / "niche_by_section.tsv")
    by_site = by_sec.groupby(by_sec.index.map(site_of)).mean(); write_table(by_site.reset_index().rename(columns={"index": "site", "section": "site"}), sd / "niche_by_site.tsv")
    near_b = np.zeros(len(spots), bool); off = 0
    for s in secs:
        g = G[s]; nb = ring_neighbors(g.xy, g.pitch, rings=(1,)); b = g.boundary; near_b[off:off + g.n] = [b[i] or (len(nb[i]) > 0 and b[nb[i]].any()) for i in range(g.n)]; off += g.n
    spots["near_boundary"] = near_b; enr = []
    matched = ~spots["unmatched"].astype(bool).to_numpy()
    for k in range(K):
        m = (spots["niche"] == k).to_numpy() & matched; a_, b_ = (m & near_b).sum(), (~m & matched & near_b).sum(); c_, d_ = (m & ~near_b).sum(), (~m & matched & ~near_b).sum()  # unmatched spots excluded
        enr.append({"niche": k, "n_spots": int(m.sum()), "frac_near_boundary": a_ / max(a_ + c_, 1), "odds_ratio_near_boundary": (a_ * d_) / max(b_ * c_, 1), "tumor_frac_histology": float(hist.loc[k].get("tumor_cellular", 0))})
    enr = pd.DataFrame(enr); cand = enr[enr["tumor_frac_histology"] < 0.6]; inter = int(cand.sort_values("odds_ratio_near_boundary", ascending=False)["niche"].iloc[0]); enr["is_interface_niche"] = enr["niche"] == inter
    write_table(enr, sd / "niche_boundary_enrichment.tsv"); names = name_niches(comp, hist, enr, inter); json.dump({str(k): v for k, v in names.items()}, open(sd / "niche_names.json", "w"), indent=2)
    cls = classify(by_sec, site_of, K); cls["name"] = cls["niche"].map(names); write_table(cls, sd / "niche_classification.tsv"); log(f"\n{cls.to_string()}")
    # T-cell score per niche (patient-level means)
    ts_rows = []
    for s in secs:
        import anndata as ad
        a = ad.read_h5ad(proc / "qc" / f"{s}.h5ad"); ts = t_score(a, cfg["genes"]["t_cell_holdout"]); sub = spots[spots["section"] == s].set_index("barcode").reindex(a.obs_names)
        for k in range(K):
            m = (sub["niche"] == k).to_numpy() & (~sub["unmatched"].fillna(True).to_numpy(bool))
            if m.sum() >= 20:
                ts_rows.append({"section": s, "patient": patient_of[s], "site": site_of[s], "niche": k, "n_spots": int(m.sum()), "t_score_mean": float(ts[m].mean())})
    tsd = pd.DataFrame(ts_rows); write_table(tsd, sd / "niche_tcell_by_section.tsv"); write_table(tsd.groupby(["patient", "niche"])["t_score_mean"].mean().groupby("niche").agg(["mean", "std", "count"]).reset_index(), sd / "niche_tcell_patient_level.tsv")
    # markers (patient-level pseudobulk)
    genes_c = common_genes(cfg, secs); l2, npb = pseudobulk(cfg, spots, secs, genes_c, "patient", v2["min_spots_per_patient_niche"]); mk = niche_markers(l2, K)
    write_table(mk[mk["fdr"] < 0.1].groupby("niche").head(200) if len(mk) else mk, sd / "niche_markers.tsv"); log(f"markers: {mk[mk['fdr'] < 0.05].groupby('niche').size().to_dict() if len(mk) else 'none'} genes at FDR < 0.05")
    # adjacency
    adj = adjacency_enrichment(spots, G, K, v2["adjacency_permutations"], cfg["seed"]); write_table(adj, sd / "niche_adjacency_by_section.tsv")
    padj = adj.groupby(["patient", "niche_a", "niche_b"])[["log2_ratio", "z"]].mean().groupby(["niche_a", "niche_b"]).agg(log2_ratio_mean=("log2_ratio", "mean"), z_mean=("z", "mean"), n_patients=("z", "count"), n_patients_positive=("log2_ratio", lambda x: int((x > 0).sum()))).reset_index()
    write_table(padj, sd / "niche_adjacency_patient_level.tsv")
    # lung interface sub-analysis (v1 machinery, fixed rules)
    man = pd.read_csv(ROOT / "results/01_manifest/sample_manifest.tsv", sep="\t", dtype=str); elig = man[(man["role"] == "discovery_lung") & (man["boundary_eligible"] == "True")]["section_id"].tolist()
    sec, seg, con = run_interface_cohort(spots, proc / "qc", w_dir, cfg, inter, elig, patient_of); write_table(sec, sd / "interface_section_metrics.tsv"); write_table(seg, sd / "interface_segments.tsv"); write_table(con, sd / "interface_matched_contrast.tsv")
    pm = pooled_models(seg, cluster="patient", n_perm=cfg["interface"]["n_permutations"], seed=cfg["seed"]); write_table(pm, sd / "interface_pooled_models.tsv")
    from scipy.stats import spearmanr  # patient-level Spearman of coverage with each readout (sections averaged within patient)
    ev = sec[sec["evaluable_T"]]; plv = ev.groupby("patient")[["coverage", "E_s", "E_s_noniche", "E_s_distal"]].mean(); rows_ = []
    for col in ["E_s", "E_s_noniche", "E_s_distal"]:
        rho, pv = spearmanr(plv["coverage"], plv[col]); rows_.append({"readout": col, "n_patients": len(plv), "spearman_rho": float(rho), "p": float(pv)})
    write_table(pd.DataFrame(rows_), sd / "interface_patient_level.tsv")
    dl = spots[~spots["unmatched"].astype(bool)]; evd = dl["histology_label"].isin(CLASSES)  # the delivered partition, not only the balanced sample
    delivered = {"n_matched_spots": int(len(dl)), "ari_vs_patient": float(adjusted_rand_score(dl["patient"], dl["niche"])), "ari_vs_region": float(adjusted_rand_score(dl.loc[evd, "histology_label"], dl.loc[evd, "niche"]))}
    write_table(pd.DataFrame([delivered]), sd / "delivered_partition_identity.tsv"); log(f"delivered partition: {delivered}")
    write_table(spots, sd / "spot_niches.tsv")
    meta = {"plan": "v2", "K": K, "unmatched_threshold": thr, "interface_niche": inter, "names": names, "centroid_agreement_ari": cinfo["centroid_agreement_ari"], "consensus_seed_agreement": cinfo["seed_agreement_with_consensus"], "frozen_epochs": meta_m["n_epochs"]}
    json.dump(meta, open(sd / "meta.json", "w"), indent=2); log(json.dumps({k: v for k, v in meta.items() if k != "names"}))
    write_provenance(STAGE + ("_retrain" if RETRAIN else "") + (f"_{VARIANT}" if VARIANT else ""), [fro / "meta.json"], [sd / "spot_niches.tsv", sd / "niche_classification.tsv"], cfg, extra=meta, t_start=t0)


if __name__ == "__main__":
    main()
