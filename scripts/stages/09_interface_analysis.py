"""Stage 09 — interface coverage / continuity and the inner–outer T-cell readout (discovery cohort).

Uses the frozen niche map (stage 08), image-derived boundaries (stage 04), RCTD weights (stage 05) and the
locked T-cell gene set scored on log-normalised expression (genes never seen by the model in the gene-holdout
configuration). Section-level metrics, segment-level composition-matched contrasts, restricted permutation
nulls, and patient-level aggregation with a set in advance representative-section rule
(earliest lung-metastasis time point; ties → more interface spots).
Outputs results/09_interface/{section_metrics.tsv, segments.tsv, matched_contrast.tsv, patient_level.tsv, summary.json}
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
from lib.inference import pooled_models  # noqa: E402
from lib.interface import run_interface_cohort  # noqa: E402
from lib.repro import ROOT, frozen_name, load_config, log, results_path, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "09_interface"


def representative_sections(man: pd.DataFrame, sec_metrics: pd.DataFrame) -> dict[str, str]:
    """patient → representative section: earliest time from diagnosis (Mxx), tie → more boundary spots."""
    m = man.set_index("section_id")
    rep = {}
    for pat, grp in sec_metrics[sec_metrics["evaluable"]].groupby("patient"):
        g = grp.copy()
        g["t_months"] = [int(str(m.loc[s, "time_from_diagnosis"]).lstrip("M")) if str(m.loc[s, "time_from_diagnosis"]).startswith("M") else 9999 for s in g["section"]]
        g = g.sort_values(["t_months", "n_boundary"], ascending=[True, False])
        rep[pat] = g["section"].iloc[0]
    return rep


def patient_level(sec: pd.DataFrame, con: pd.DataFrame, rep: dict[str, str], seed: int) -> tuple[pd.DataFrame, dict]:
    rows = []
    for pat, s in rep.items():
        r = sec[sec["section"] == s].iloc[0]; c = con[con["section"] == s]
        rows.append({"patient": pat, "representative_section": s, "coverage": r["coverage"], "continuity_mean_run": r["continuity_mean_run"], "fragmentation": r["fragmentation"],
                     "E_s": r["E_s"], "evaluable_T": r["evaluable_T"], "matched_diff_E": c["matched_diff_E"].iloc[0] if len(c) else np.nan,
                     "unmatched_diff_E": c["unmatched_diff_E"].iloc[0] if len(c) else np.nan, "n_matched": c["n_matched"].iloc[0] if len(c) else 0,
                     "spearman_cov_E": c["spearman_cov_E"].iloc[0] if len(c) else np.nan, "perm_p": c["perm_p"].iloc[0] if len(c) else np.nan})
    pl = pd.DataFrame(rows)
    summ = {"n_patients": len(pl), "n_patients_evaluable_T": int(pl["evaluable_T"].sum())}
    ev = pl[pl["evaluable_T"]]
    if len(ev) >= 4:
        rho = stats.spearmanr(ev["coverage"], ev["E_s"]); summ["spearman_coverage_vs_E_patient"] = float(rho.correlation); summ["spearman_p"] = float(rho.pvalue)
        md = ev["matched_diff_E"].dropna()
        if len(md) >= 4:
            rng = np.random.default_rng(seed)
            boots = [rng.choice(md, len(md), replace=True).mean() for _ in range(5000)]
            summ.update({"matched_diff_E_mean": float(md.mean()), "matched_diff_E_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                         "matched_diff_E_n_positive": int((md > 0).sum()), "matched_diff_E_n": int(len(md)), "wilcoxon_p": float(stats.wilcoxon(md).pvalue) if (md != 0).any() else 1.0})
        um = ev["unmatched_diff_E"].dropna()
        if len(um) >= 4:
            summ.update({"unmatched_diff_E_mean": float(um.mean()), "unmatched_diff_E_n_positive": int((um > 0).sum())})
    return pl, summ


def main() -> None:
    t0 = time.time()
    cfg = load_config(); set_global_seed(cfg["seed"])
    meta = json.load(open(ROOT / cfg["paths"]["processed"] / frozen_name() / "meta.json"))
    spots = pd.read_csv(results_path("08_niches/spot_niches.tsv"), sep="\t")
    man = pd.read_csv(ROOT / "results/01_manifest/sample_manifest.tsv", sep="\t", dtype=str)
    elig = man[(man["role"] == "discovery_lung") & (man["boundary_eligible"] == "True")]["section_id"].tolist()
    patient_of = cfg["cohorts"]["discovery"]["patient_groups"]
    log(f"eligible sections: {elig}; interface niche {meta['interface_niche']}")
    sec, seg, con = run_interface_cohort(spots, ROOT / cfg["paths"]["processed"] / "qc", ROOT / "results/05_deconvolution/weights", cfg, meta["interface_niche"], elig, patient_of)
    sd = stage_dir(STAGE)
    write_table(sec, sd / "section_metrics.tsv"); write_table(seg, sd / "segments.tsv"); write_table(con, sd / "matched_contrast.tsv")
    rep = representative_sections(man, sec)
    pl, summ = patient_level(sec, con, rep, cfg["seed"])
    write_table(pl, sd / "patient_level.tsv")
    # sensitivity: all sections, patient-weighted
    allv = sec[sec["evaluable_T"]].merge(con[["section", "matched_diff_E"]], on="section", how="left")
    if len(allv):
        pw = allv.groupby("patient")[["coverage", "E_s", "matched_diff_E"]].mean()
        summ["all_sections_patient_weighted"] = {"n_patients": len(pw), "matched_diff_E_mean": float(pw["matched_diff_E"].mean()),
                                                 "spearman_coverage_vs_E": float(stats.spearmanr(pw["coverage"], pw["E_s"]).correlation) if len(pw) >= 4 else None}
    # pooled within-section models (secondary analysis; small-cluster-honest inference; readout variants) — lib.inference
    pm = pooled_models(seg, cluster="patient", n_perm=cfg["interface"]["n_permutations"], seed=cfg["seed"])
    write_table(pm, sd / "pooled_models.tsv")
    log(f"\n{pm[['outcome', 'covariate_set', 'n_segments', 'n_sections', 'n_clusters', 'coef', 'ci95_lo', 'ci95_hi', 'p_t_gminus1', 'p_perm_restricted', 'loo_coef_min', 'loo_coef_max']].round(3).to_string()}")
    key = pm.set_index(["outcome", "covariate_set"])
    summ["segment_model"] = {"note": "see pooled_models.tsv; all pooled models are post hoc / exploratory ; the reported estimate is E_seg_noniche with 11-type adjustment",
                             **{f"{o}__{c}": {k: (None if (not isinstance(v, str) and pd.isna(v)) else (v if isinstance(v, str) else float(v))) for k, v in key.loc[(o, c)].drop(["covariates"]).items()} for o in key.index.get_level_values(0).unique() for c in ["none", "3type", "full"] if (o, c) in key.index}}
    ev = pl[pl["evaluable_T"]]
    summ["E_variants_patient_level"] = {v: {"spearman_coverage": float(stats.spearmanr(ev["coverage"], sec.set_index("section").loc[ev["representative_section"], v]).correlation)} for v in ["E_s", "E_s_noniche", "E_s_distal"] if len(ev) >= 4}
    json.dump(summ, open(sd / "summary.json", "w"), indent=2)
    log(json.dumps(summ, indent=1))
    write_provenance(STAGE, [results_path("08_niches/spot_niches.tsv")], [sd / "section_metrics.tsv", sd / "patient_level.tsv"], cfg, extra=summ, t_start=t0)


if __name__ == "__main__":
    main()
