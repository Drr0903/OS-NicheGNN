"""Stage 16 (plan v2, Aim 2) — microenvironment remodelling between primary and metastatic osteosarcoma.

(a) paired primary → lung metastasis (PT1–4 vs MR1–4): niche and RCTD composition change; within-niche programme change
    (pseudobulk per section × niche, log2FC per pair, consistent in ≥ 3 of 4 pairs), Hallmark enrichment
(b) unpaired primary vs metastasis: patient-level niche fractions, Mann–Whitney + Cliff's delta
(c) metastatic sites: niche repertoire per site (from stage 15) — descriptive
(d) longitudinal MR10A → MR10B — descriptive
(e) treatment before relapse (regimen family) — descriptive
(f) adjacency remodelling: niche-pair contact z in primaries vs metastases (paired + unpaired)
(g) ligand–receptor co-expression at boundary-niche/tumour contacts, primaries vs metastases
Outputs results/16_remodelling/*.tsv
"""
from __future__ import annotations

import json
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.niches_v2 import common_genes, pseudobulk  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.v2 import load_cohort  # noqa: E402

STAGE = "16_remodelling"; s12 = import_module("12_niche_biology")


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b); return float((np.sign(a[:, None] - b[None, :])).mean()) if len(a) and len(b) else np.nan


def regimen_family(t: str) -> str:
    t = str(t).upper()
    if t == "NAN":
        return "primary"
    if "MEI" in t:
        return "MEI-based"
    if "MAP" in t:
        return "MAP"
    return "other"


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); v2 = cfg["v2"]; sd = stage_dir(STAGE); N = ROOT / "results/15_niches"
    spots = pd.read_csv(N / "spot_niches.tsv", sep="\t"); meta = json.load(open(N / "meta.json")); K = meta["K"]; names = {int(k): v for k, v in meta["names"].items()}; inter = meta["interface_niche"]
    site_of = v2["site_of"]; patient_of = v2["patient_of"]; secs = sorted(spots["section"].unique()); man = pd.read_csv(ROOT / "results/01_manifest/sample_manifest.tsv", sep="\t", dtype=str).set_index("section_id")
    by_sec = pd.crosstab(spots[~spots["unmatched"].astype(bool)]["section"], spots[~spots["unmatched"].astype(bool)]["niche"], normalize="index").reindex(columns=range(K), fill_value=0.0)
    types = [c for c in spots.columns if c in ["Alveolar_macrophage", "B_plasma", "Endothelial", "Fibroblast", "Lung_epithelium", "Mast", "Myeloid", "Osteoclast", "Osteosarcoma", "Pericyte_SMC", "T_NK"]]
    comp_sec = spots.groupby("section")[types].mean()
    # (a) paired
    pairs = v2["paired"]; rows = []
    for pt, mr in list(pairs.items()) + list(v2.get("paired_sensitivity", {}).items()):  # PT5 → LR5 (soft tissue) as a sensitivity pair
        for k in range(K):
            rows.append({"pair": f"{pt}→{mr}", "patient": patient_of[pt], "variable": f"niche {k} ({names[k]})", "primary": float(by_sec.loc[pt, k]), "metastasis": float(by_sec.loc[mr, k]), "delta": float(by_sec.loc[mr, k] - by_sec.loc[pt, k])})
        for t in types:
            rows.append({"pair": f"{pt}→{mr}", "patient": patient_of[pt], "variable": f"RCTD {t}", "primary": float(comp_sec.loc[pt, t]), "metastasis": float(comp_sec.loc[mr, t]), "delta": float(comp_sec.loc[mr, t] - comp_sec.loc[pt, t])})
    paired = pd.DataFrame(rows); paired["sensitivity_pair"] = paired["pair"].isin([f"{a}→{b}" for a, b in v2.get("paired_sensitivity", {}).items()]); write_table(paired, sd / "paired_composition_change.tsv")
    ps = paired[~paired["sensitivity_pair"]].groupby("variable")["delta"].agg(mean_delta="mean", n_pairs="count", n_pairs_up=lambda x: int((x > 0).sum()), n_pairs_down=lambda x: int((x < 0).sum())).reset_index()
    ps5 = paired.groupby("variable")["delta"].agg(mean_delta_with_PT5_LR5="mean", n_pairs_up_with_PT5_LR5=lambda x: int((x > 0).sum())).reset_index(); ps = ps.merge(ps5, on="variable"); write_table(ps, sd / "paired_composition_summary.tsv"); log(f"\n{ps.round(3).to_string()}")
    genes = common_genes(cfg, secs); gmt = ROOT / "data/raw/msigdb/h.all.v2023.2.Hs.symbols.gmt"
    # Programme change is computed twice: on all matched spots of a niche, and restricted to tumour-labelled spots
    # (copy-number-based region label, v2.tumour_scope_label) so that lung-parenchyma admixture in metastases cannot masquerade as remodelling.
    prog, hall = [], []
    for scope, sp in [("all_spots", spots), ("tumour_dominant", spots[spots["histology_label"] == v2["tumour_scope_label"]])]:
        l2, npb = pseudobulk(cfg, sp, secs, genes, "section", v2["min_spots_per_patient_niche"])
        pr, ha = programme_change(l2, pairs, K, names, gmt)
        for d in pr + ha:
            d.insert(0, "scope", scope)
        prog += pr; hall += ha
    write_table(pd.concat(prog, ignore_index=True) if prog else pd.DataFrame(), sd / "paired_programme_change.tsv"); write_table(pd.concat(hall, ignore_index=True) if hall else pd.DataFrame(), sd / "paired_programme_hallmark.tsv")
    log(f"programme change: {pd.concat(prog).groupby(['scope', 'niche']).size().to_dict() if prog else 'none'} consistent genes per scope/niche")
    rest_a(cfg, v2, sd, N, pairs, spots, by_sec, comp_sec, types, K, names, inter, site_of, patient_of, secs, man, genes, meta, t0)


IEG = {"FOS", "FOSB", "JUN", "JUNB", "JUND", "EGR1", "EGR2", "EGR3", "ATF3", "NR4A1", "NR4A2", "NR4A3", "IER2", "IER3", "DUSP1", "DUSP2", "KLF2", "KLF4", "KLF6", "ZFP36", "PPP1R15A", "BTG2", "CYR61", "CCN1", "SGK1", "RHOB", "SOCS3", "GADD45B"}
"""Immediate-early / handling-responsive genes; their induction tracks ischaemia time and tissue processing rather than tumour biology."""


def programme_change(l2: pd.DataFrame, pairs: dict, K: int, names: dict, gmt: Path) -> tuple[list, list]:
    """Genes changing consistently (same sign in >= 3 of the pairs, |mean log2FC| >= 1) within each niche and over all matched spots (-1)."""
    prog, hall = [], []
    for k in list(range(K)) + [-1]:
        ok = [(pt, mr) for pt, mr in pairs.items() if (pt, k) in l2.columns and (mr, k) in l2.columns]
        if len(ok) < 3:
            continue
        D = np.column_stack([l2[(mr, k)] - l2[(pt, k)] for pt, mr in ok]); expr = np.column_stack([l2[(mr, k)] for _, mr in ok] + [l2[(pt, k)] for pt, _ in ok]).mean(1) > 1
        cons_up = (D > 0).sum(1) >= 3; cons_dn = (D < 0).sum(1) >= 3; mean = D.mean(1)
        df = pd.DataFrame({"niche": k, "name": names.get(k, "all matched spots"), "gene": l2.index, "n_pairs": len(ok), "mean_log2fc_met_vs_primary": mean, "n_pairs_up": (D > 0).sum(1), "n_pairs_down": (D < 0).sum(1)})
        for c_, (pt_, mr_) in enumerate(ok):  # per-pair values, so a figure can show consistency instead of only the mean
            df[f"log2fc_{pt_}_{mr_}"] = D[:, c_]
        df = df[expr & ((cons_up & (mean >= 1)) | (cons_dn & (mean <= -1)))].sort_values("mean_log2fc_met_vs_primary"); prog.append(df)
        universe = l2.index[expr].tolist()
        for direction, sel in [("up_in_metastasis", df[df["mean_log2fc_met_vs_primary"] > 0]), ("down_in_metastasis", df[df["mean_log2fc_met_vs_primary"] < 0])]:
            if len(sel) >= 10 and gmt.exists():
                h = s12.hallmark(sel["gene"].tolist(), universe, gmt)  # whole table kept, so every set is reported in both directions
                h["n_selected"] = len(set(sel["gene"]) & set(universe)); h["n_universe"] = len(universe)
                h["ieg_in_overlap"] = [sum(g in IEG for g in (gs.split(",") if gs else [])) for gs in h["genes"]]
                # A11 sensitivity: the same test with immediate-early genes removed from the selection and the universe
                u2 = [g for g in universe if g not in IEG]; s2 = [g for g in sel["gene"] if g not in IEG]
                h2 = s12.hallmark(s2, u2, gmt).set_index("gene_set") if len(s2) >= 10 else None
                h["p_no_ieg"] = [float(h2.loc[g, "p"]) if h2 is not None and g in h2.index else np.nan for g in h["gene_set"]]
                h["overlap_no_ieg"] = [int(h2.loc[g, "overlap"]) if h2 is not None and g in h2.index else -1 for g in h["gene_set"]]
                h.insert(0, "direction", direction); h.insert(0, "niche", k); hall.append(h)
    return prog, hall


def rest_a(cfg, v2, sd, N, pairs, spots, by_sec, comp_sec, types, K, names, inter, site_of, patient_of, secs, man, genes, meta, t0) -> None:
    # (b) unpaired primary vs metastasis (patient-level)
    pl = by_sec.copy(); pl["patient"] = pl.index.map(patient_of); pl["group"] = ["primary" if site_of[s] == "bone" else "metastasis" for s in pl.index]; pl = pl.groupby(["group", "patient"]).mean().reset_index(); rows = []
    for k in range(K):
        a = pl[pl["group"] == "metastasis"][k].to_numpy(); b = pl[pl["group"] == "primary"][k].to_numpy()
        rows.append({"niche": k, "name": names[k], "primary_mean": float(b.mean()), "metastasis_mean": float(a.mean()), "n_primary_patients": len(b), "n_metastasis_patients": len(a), "cliffs_delta_met_minus_primary": cliffs_delta(a, b), "p_mannwhitney": float(stats.mannwhitneyu(a, b).pvalue)})
    write_table(pd.DataFrame(rows), sd / "unpaired_primary_vs_metastasis.tsv"); log(f"\n{pd.DataFrame(rows).round(3).to_string()}")
    # (b') A11 sensitivity: non-overlapping patients only (primaries of patients without a metastasis vs metastases of patients without a primary)
    both = {p_ for p_ in set(patient_of.values()) if len({site_of[s_] == "bone" for s_ in secs if patient_of[s_] == p_}) == 2}; rows2 = []
    for k in range(K):
        a = pl[(pl["group"] == "metastasis") & ~pl["patient"].isin(both)][k].to_numpy(); b = pl[(pl["group"] == "primary") & ~pl["patient"].isin(both)][k].to_numpy()
        rows2.append({"niche": k, "name": names[k], "primary_mean": float(b.mean()), "metastasis_mean": float(a.mean()), "n_primary_patients": len(b), "n_metastasis_patients": len(a), "cliffs_delta_met_minus_primary": cliffs_delta(a, b), "p_mannwhitney": float(stats.mannwhitneyu(a, b).pvalue)})
    write_table(pd.DataFrame(rows2), sd / "unpaired_primary_vs_metastasis_nonoverlap.tsv")
    # (b'') A14: lung against non-lung metastases — both groups are post-treatment relapses of soft tissue, so decalcification,
    # elapsed time and the primary/metastasis treatment gap do not separate them
    mets = [s_ for s_ in secs if site_of[s_] != "bone"]
    pm_ = by_sec.loc[mets].copy(); pm_["patient"] = [patient_of[s_] for s_ in mets]; pm_["grp"] = ["lung" if site_of[s_] == "lung" else "non-lung" for s_ in mets]
    pmp = pm_.groupby(["grp", "patient"]).mean(numeric_only=True).reset_index(); rows3 = []
    for k in range(K):
        a = pmp[pmp["grp"] == "lung"][k].to_numpy(); b = pmp[pmp["grp"] == "non-lung"][k].to_numpy()
        rows3.append({"niche": k, "name": names[k], "lung_mean": float(a.mean()), "nonlung_mean": float(b.mean()), "n_lung_patients": len(a), "n_nonlung_patients": len(b),
                      "cliffs_delta_lung_minus_nonlung": cliffs_delta(a, b), "p_mannwhitney": float(stats.mannwhitneyu(a, b).pvalue)})
    write_table(pd.DataFrame(rows3), sd / "lung_vs_nonlung_metastases.tsv"); log(f"\nlung vs non-lung metastases\n{pd.DataFrame(rows3).round(3).to_string()}")
    l2m, _ = pseudobulk(cfg, spots[spots["histology_label"] == v2["tumour_scope_label"]], mets, genes, "section", v2["min_spots_per_patient_niche"])
    prog3 = []
    for k in list(range(K)) + [-1]:
        lu = [c for c in l2m.columns if c[1] == k and site_of[c[0]] == "lung"]; nl = [c for c in l2m.columns if c[1] == k and site_of[c[0]] != "lung"]
        if len(lu) < 3 or len(nl) < 3:
            continue
        d_ = l2m[lu].mean(1) - l2m[nl].mean(1); expr = pd.concat([l2m[lu], l2m[nl]], axis=1).mean(1) > 1
        df3 = pd.DataFrame({"niche": k, "name": names.get(k, "all matched spots"), "gene": l2m.index, "n_lung_sections": len(lu), "n_nonlung_sections": len(nl), "mean_log2fc_lung_vs_nonlung": d_})
        prog3.append(df3[expr & (d_.abs() >= 1)].sort_values("mean_log2fc_lung_vs_nonlung"))
    write_table(pd.concat(prog3, ignore_index=True) if prog3 else pd.DataFrame(), sd / "lung_vs_nonlung_programme.tsv")
    log(f"lung vs non-lung programme: {pd.concat(prog3).groupby('niche').size().to_dict() if prog3 else 'no niche with >= 3 sections per group'}")
    # (c) sites (descriptive) — carried from stage 15; (d) longitudinal; (e) treatment
    lon = v2["longitudinal"]; ifs = pd.read_csv(N / "interface_section_metrics.tsv", sep="\t").set_index("section")
    rows = [{"section": s, "timepoint": man.loc[s, "timepoint"], "treatment_before_relapse": man.loc[s, "treatment_before_relapse"], **{f"niche_{k}": float(by_sec.loc[s, k]) for k in range(K)}, **{f"rctd_{t}": float(comp_sec.loc[s, t]) for t in types},
             "interface_coverage": float(ifs.loc[s, "coverage"]) if s in ifs.index else np.nan, "E_s": float(ifs.loc[s, "E_s"]) if s in ifs.index else np.nan} for s in lon]
    write_table(pd.DataFrame(rows), sd / "longitudinal_MR10.tsv")
    tr = by_sec.copy(); tr["patient"] = tr.index.map(patient_of); tr["regimen_family"] = [regimen_family(man.loc[s, "treatment_before_relapse"]) for s in tr.index]; tr = tr[tr["regimen_family"] != "primary"]
    trp = tr.groupby(["regimen_family", "patient"]).mean(numeric_only=True).reset_index().drop(columns="patient"); write_table(trp.groupby("regimen_family").agg(["mean", "count"]).reset_index(), sd / "treatment_family_niche_fractions.tsv")
    # (f) adjacency remodelling
    adj = pd.read_csv(N / "niche_adjacency_by_section.tsv", sep="\t"); adj["group"] = ["primary" if site_of[s] == "bone" else "metastasis" for s in adj["section"]]
    # the size-invariant log2(observed / expected) ratio is the primary statistic (the permutation z grows with section size); z is kept for reference
    ap = adj.groupby(["group", "patient", "niche_a", "niche_b"])[["log2_ratio", "z"]].mean().reset_index(); rows = []
    for (a_, b_), g in ap.groupby(["niche_a", "niche_b"]):
        gm, gp = g[g["group"] == "metastasis"], g[g["group"] == "primary"]; rm, rp = gm["log2_ratio"].to_numpy(), gp["log2_ratio"].to_numpy()
        pr = [(gp[gp["patient"] == p]["log2_ratio"].mean(), gm[gm["patient"] == p]["log2_ratio"].mean()) for p in sorted(set(patient_of[pt] for pt in pairs))]
        pr = [(x, y) for x, y in pr if x == x and y == y]
        rows.append({"niche_a": a_, "niche_b": b_, "name_a": names[a_], "name_b": names[b_], "log2_ratio_primary_mean": float(rp.mean()) if len(rp) else np.nan, "log2_ratio_metastasis_mean": float(rm.mean()) if len(rm) else np.nan,
                     "n_primary": len(rp), "n_metastasis": len(rm), "cliffs_delta": cliffs_delta(rm, rp) if len(rm) and len(rp) else np.nan, "p_mannwhitney": float(stats.mannwhitneyu(rm, rp).pvalue) if len(rm) > 1 and len(rp) > 1 else np.nan,
                     "paired_n": len(pr), "paired_mean_delta": float(np.mean([y - x for x, y in pr])) if pr else np.nan, "paired_n_up": int(sum(y > x for x, y in pr)),
                     "z_primary_mean": float(gp["z"].mean()) if len(gp) else np.nan, "z_metastasis_mean": float(gm["z"].mean()) if len(gm) else np.nan})
    write_table(pd.DataFrame(rows), sd / "adjacency_remodelling.tsv")
    n_sec = spots.groupby("section").size(); hh = adj[adj["niche_a"] == adj["niche_b"]].assign(n=lambda d: d["section"].map(n_sec))
    json.dump({"spearman_z_vs_section_size": float(hh[["z", "n"]].corr("spearman").iloc[0, 1]), "spearman_log2_ratio_vs_section_size": float(hh[["log2_ratio", "n"]].corr("spearman").iloc[0, 1]),
               "mean_spots_primary": float(n_sec[[s_ for s_ in secs if site_of[s_] == "bone"]].mean()), "mean_spots_metastasis": float(n_sec[[s_ for s_ in secs if site_of[s_] != "bone"]].mean())}, open(sd / "adjacency_statistic_check.json", "w"), indent=2)
    # (g) ligand–receptor at boundary-niche / tumour contacts
    tum = [k for k in range(K) if k != inter and float(pd.read_csv(N / "niche_histology.tsv", sep="\t").set_index("niche").loc[k].get("tumor_cellular", 0)) >= 0.6]
    prim = [s for s in secs if site_of[s] == "bone"]; mets = [s for s in secs if site_of[s] != "bone"]
    lr_p = s12.lr_adjacency(cfg, spots, prim, inter, tum, n_perm=200); lr_m = s12.lr_adjacency(cfg, spots, mets, inter, tum, n_perm=200)
    if len(lr_p) and len(lr_m):
        lr = lr_p.merge(lr_m, on=["ligand", "receptor"], suffixes=("_primary", "_metastasis")); lr["delta_log2_ratio"] = lr["log2_ratio_metastasis"] - lr["log2_ratio_primary"]; write_table(lr, sd / "lr_primary_vs_metastasis.tsv")
    write_provenance(STAGE, [N / "spot_niches.tsv"], [sd / "paired_composition_change.tsv", sd / "unpaired_primary_vs_metastasis.tsv", sd / "adjacency_remodelling.tsv"], cfg, extra={"plan": "v2", "tumour_niches": tum, "interface_niche": inter}, t_start=t0)


if __name__ == "__main__":
    main()
