"""The README sheet of the supplementary workbook: every sheet with its full name and table, and the meaning of the
statistical columns that recur across sheets."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

GLOSSARY = {
    "patient / heldout_patient": "patient identifier (P01–P19); patients are the unit of every test",
    "section / section_id": "Visium section (PT primaries, MR lung and other metastases, LR soft-tissue relapses)",
    "niche / name": "atlas niche index (0–7) and its rule-based name",
    "n_patients, n_sections, n_spots, n_segments": "numbers of patients, sections, spots or interface segments entering the estimate",
    "mean_diff": "mean over patients of the paired difference (first minus second named method or condition)",
    "ci95_lo, ci95_hi": "95 % confidence interval: bootstrap over patients (5,000 resamples) for paired differences; t distribution with G − 1 degrees of freedom and patient-clustered standard errors for regression coefficients",
    "n_better, n_worse": "patients in which the first method scored higher / lower",
    "wilcoxon_p": "two-sided Wilcoxon signed-rank P over patients",
    "p_mannwhitney": "two-sided Mann–Whitney P between patient groups",
    "coef, se_cluster": "regression coefficient and its standard error clustered by patient",
    "p_t_gminus1": "P from the t distribution with G − 1 degrees of freedom, G = number of patients (clusters)",
    "p_normal_ref": "P from the normal reference distribution (for comparison only)",
    "p_perm_restricted, p_perm_cyclic, n_perm": "P from permutations within section (restricted) or cyclic shifts along the boundary, and the number of permutations",
    "loo_coef_min, loo_coef_max, loo_min_cluster, loo_max_cluster": "range of the coefficient when each patient is left out in turn, and the patient whose removal gives each end",
    "covariate_set / covariates": "none = unadjusted; 3type = tumour, fibroblast and myeloid fractions; full = all eleven deconvolved cell-type fractions",
    "fdr": "Benjamini–Hochberg false discovery rate",
    "tcell_rho, tcell_nbhd_rho, comp_rho, comp_nbhd_rho": "Spearman correlation between predicted and observed T-cell score or cell composition of the spot or of its neighbourhood (held-out patient)",
    "macro_f1, boundary_f1, chaos": "region recovery (macro-F1), boundary F1 with one-ring tolerance and spatial coherence (CHAOS, lower is more coherent)",
    "ari, nmi, ari_vs_patient": "adjusted Rand index and normalized mutual information with the reference classes; ARI of niches with patient identity",
    "odds_ratio_near_boundary": "odds of lying within one ring of the tumour boundary, niche against all other spots",
}


def readme(sheets: list[Path], legends: Path) -> pd.DataFrame:
    titles = dict(re.findall(r"^## Supplementary Table (S\d+)\. (.+)$", legends.read_text(encoding="utf-8"), flags=re.M))
    rows = []
    for p in sheets:
        t = re.match(r"Table(S\d+)_", p.stem).group(1)
        rows.append({"sheet": p.stem[:31], "full name": p.stem, "table": "Supplementary Table " + t, "table title": titles.get(t, "")})
    rows += [{}] + [{"sheet": "column", "full name": k, "table": "", "table title": v} for k, v in GLOSSARY.items()]
    return pd.DataFrame(rows)
