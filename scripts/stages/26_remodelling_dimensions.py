"""Stage 26 () — remodelling in three dimensions on the delivered atlas.

composition      niche fractions per section, over all matched spots and over tumour-labelled spots only (so that the
                 amount of host tissue in a section cannot drive the change)
spatial          ring-1 contact enrichment between niches, log2(observed / expected) against a within-section label
                 permutation that keeps every niche's abundance (stage 15), so a change in contact is not a change in amount
internal state   within each niche: mean T-cell programme score (44 genes never used by the model) and PROGENy pathway
                 activity per section
For every variable: the four matched primary → lung-metastasis pairs (mean change, number of pairs in each direction), the
soft-tissue relapse pair PT5 → LR5 separately, and the unpaired patient-level comparison (metastasis vs primary; Cliff's delta,
Mann–Whitney), restricted to patients that contribute only one of the two groups.
Reads results/15_niches/{spot_niches, niche_tcell_by_section, niche_adjacency_by_section}.tsv,
      results/18_signalling/pathway_by_section_niche.tsv
Outputs results/26_remodelling_dimensions/{paired_dimensions.tsv, unpaired_dimensions.tsv, dimension_summary.tsv}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "26_remodelling_dimensions"
PATHWAYS = ["TGFb", "WNT", "TNFa", "NFkB", "JAK-STAT", "Hypoxia", "VEGF", "PI3K", "MAPK", "p53"]


def cliffs(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b); return float(np.sign(a[:, None] - b[None, :]).mean()) if len(a) and len(b) else np.nan


def long_table(cfg: dict, K: int) -> pd.DataFrame:
    """One row per (section, dimension, variable, value)."""
    N = ROOT / "results/15_niches"; v2 = cfg["v2"]; rows = []
    sp = pd.read_csv(N / "spot_niches.tsv", sep="\t"); sp = sp[~sp["unmatched"].astype(bool)]
    for scope, d in [("all spots", sp), ("tumour-labelled spots", sp[sp["histology_label"] == v2["tumour_scope_label"]])]:
        fr = pd.crosstab(d["section"], d["niche"], normalize="index").reindex(columns=range(K), fill_value=0.0)
        rows += [{"section": s, "dimension": "composition", "variable": f"niche {k} fraction ({scope})", "niche": k, "value": float(fr.loc[s, k])} for s in fr.index for k in range(K)]
    adj = pd.read_csv(N / "niche_adjacency_by_section.tsv", sep="\t")
    rows += [{"section": r.section, "dimension": "spatial", "variable": f"contact {r.niche_a}–{r.niche_b}", "niche": int(r.niche_a), "value": float(r.log2_ratio)} for r in adj.itertuples()]
    tc = pd.read_csv(N / "niche_tcell_by_section.tsv", sep="\t"); tc = tc[tc["n_spots"] >= v2["min_spots_per_patient_niche"]]
    rows += [{"section": r.section, "dimension": "internal state", "variable": f"niche {r.niche} T-cell programme", "niche": int(r.niche), "value": float(r.t_score_mean)} for r in tc.itertuples()]
    pw = ROOT / "results/18_signalling/pathway_by_section_niche.tsv"
    if pw.exists():
        p = pd.read_csv(pw, sep="\t"); p = p[p["source"].isin(PATHWAYS)]
        rows += [{"section": r.section, "dimension": "internal state", "variable": f"niche {r.niche} {r.source}", "niche": int(r.niche), "value": float(r.activity)} for r in p.itertuples()]
    return pd.DataFrame(rows)


def paired(L: pd.DataFrame, pairs: dict, extra: dict, patient_of: dict) -> pd.DataFrame:
    W = L.pivot_table(index=["dimension", "variable", "niche"], columns="section", values="value"); rows = []
    for (dim, var, k), r in W.iterrows():
        d = {pt: r.get(mr) - r.get(pt) for pt, mr in pairs.items() if pd.notna(r.get(pt)) and pd.notna(r.get(mr))}
        x5 = {pt: r.get(mr) - r.get(pt) for pt, mr in extra.items() if pd.notna(r.get(pt)) and pd.notna(r.get(mr))}
        if not d:
            continue
        v = np.array(list(d.values()))
        rows.append({"dimension": dim, "variable": var, "niche": k, "n_pairs": len(v), "mean_change": float(v.mean()), "n_up": int((v > 0).sum()), "n_down": int((v < 0).sum()),
                     "consistent_all_pairs": bool(len(v) == len(pairs) and (np.all(v > 0) or np.all(v < 0))),
                     **{f"change_{patient_of[pt]}": float(x) for pt, x in d.items()}, "change_PT5_LR5": float(next(iter(x5.values()))) if x5 else np.nan})
    return pd.DataFrame(rows)


def unpaired(L: pd.DataFrame, site_of: dict, patient_of: dict) -> pd.DataFrame:
    L = L.assign(patient=L["section"].map(patient_of), group=np.where(L["section"].map(site_of) == "bone", "primary", "metastasis"))
    groups = L.groupby("patient")["group"].nunique(); only = set(groups[groups == 1].index)  # patients contributing one group only
    P = L[L["patient"].isin(only)].groupby(["dimension", "variable", "niche", "group", "patient"])["value"].mean().reset_index(); rows = []
    for (dim, var, k), g in P.groupby(["dimension", "variable", "niche"]):
        a = g[g["group"] == "metastasis"]["value"].to_numpy(); b = g[g["group"] == "primary"]["value"].to_numpy()
        if len(a) < 3 or len(b) < 3:
            continue
        rows.append({"dimension": dim, "variable": var, "niche": k, "n_metastasis_patients": len(a), "n_primary_patients": len(b), "metastasis_mean": float(a.mean()), "primary_mean": float(b.mean()),
                     "cliffs_delta": cliffs(a, b), "p_mannwhitney": float(stats.mannwhitneyu(a, b).pvalue)})
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); v2 = cfg["v2"]; sd = stage_dir(STAGE)
    meta = json.load(open(ROOT / "results/15_niches/meta.json")); K = meta["K"]
    L = long_table(cfg, K)
    pr = paired(L, v2["paired"], v2.get("paired_sensitivity", {}), v2["patient_of"]); up = unpaired(L, v2["site_of"], v2["patient_of"])
    both = pr.merge(up[["variable", "cliffs_delta", "p_mannwhitney"]], on="variable", how="left")
    both["same_direction_unpaired"] = np.sign(both["mean_change"]) == np.sign(both["cliffs_delta"])
    summ = both.groupby("dimension").agg(n_variables=("variable", "size"), n_consistent_all_pairs=("consistent_all_pairs", "sum"),
                                         n_consistent_and_unpaired_same_direction=("same_direction_unpaired", lambda s: int((s & both.loc[s.index, "consistent_all_pairs"]).sum()))).reset_index()
    summ["expected_consistent_by_chance"] = summ["n_variables"] * 2 * 0.5 ** len(v2["paired"])  # all pairs share a sign under a symmetric null
    out = [sd / "paired_dimensions.tsv", sd / "unpaired_dimensions.tsv", sd / "dimension_summary.tsv"]
    write_table(both, out[0]); write_table(up, out[1]); write_table(summ, out[2]); log(f"\n{summ.to_string()}")
    log(f"\n{both[both['consistent_all_pairs']].sort_values(['dimension', 'mean_change']).round(3).to_string()}")
    write_provenance(STAGE, [ROOT / "results/15_niches/spot_niches.tsv"], out, cfg, t_start=t0)


if __name__ == "__main__":
    main()
