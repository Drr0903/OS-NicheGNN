"""Pooled within-section models for the segment-level interface analysis, with small-cluster-honest inference.

pooled_models(seg, cluster, ...) fits, for each outcome (E_seg, E_seg_noniche, E_seg_distal) and each covariate set
    "none"      E ~ coverage + C(section)
    "3type"     + w_tumor_frac + w_fibro_frac + w_myeloid_frac                       (primary adjustment)
    "full"      + every RCTD type weight (11) — includes endothelium, alveolar macrophage, lung epithelium and T/NK
and reports the coverage coefficient with
    • cluster-robust SE (clusters = patients or specimens) and P from t(G−1), not the normal reference
    • a 95 % CI from t(G−1)
    • a restricted permutation P: coverage is permuted among the segments of each section (2,000 draws), the model is
      refitted and the coefficient compared with the observed one — anti-conservative when adjacent segments are
      autocorrelated, so it is reported next to
    • a cyclic-shift permutation P: within each section the coverage sequence (segments ordered along the boundary chains)
      is rotated by a random offset and randomly reversed, which preserves its autocorrelation structure
    • the leave-one-cluster-out range of the coefficient and the clusters whose removal gives the extremes
    • E_seg_noniche_min5: the niche-excluded readout restricted to segments with ≥ 5 non-niche outer spots (sensitivity to
      selection on the exposure: fully covered segments have few or no non-niche outer spots)
    • the number of informative segments (sections with coverage variance) and sections
Only segments from sections with an evaluable T readout enter the models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

OUTCOMES = ["E_seg", "E_seg_noniche", "E_seg_noniche_min5", "E_seg_distal"]
TYPES = ["Alveolar_macrophage", "B_plasma", "Endothelial", "Fibroblast", "Lung_epithelium", "Mast", "Myeloid", "Osteoclast", "Osteosarcoma", "Pericyte_SMC", "T_NK"]


def _fit(d: pd.DataFrame, outcome: str, covs: list[str], cluster: str):
    f = f"{outcome} ~ coverage" + ("".join(f" + {c}" for c in covs)) + " + C(section)"
    return smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": d[cluster]})


def one_model(seg: pd.DataFrame, outcome: str, covs: list[str], cluster: str, n_perm: int, seed: int) -> dict:
    d = seg.dropna(subset=[outcome] + covs).copy()
    d = d[d.groupby("section")["coverage"].transform("nunique") > 1]  # informative sections only (fixed effects absorb the rest)
    G = d[cluster].nunique()
    if len(d) < 10 or G < 3:
        return {"outcome": outcome, "covariates": ",".join(covs) or "none", "n_segments": len(d), "n_sections": d["section"].nunique(), "n_clusters": G, "coef": np.nan}
    m = _fit(d, outcome, covs, cluster)
    b, se = float(m.params["coverage"]), float(m.bse["coverage"])
    tcrit = stats.t.ppf(0.975, G - 1); p_t = float(2 * stats.t.sf(abs(b / se), G - 1))
    f = f"{outcome} ~ coverage" + "".join(f" + {c}" for c in covs) + " + C(section)"
    d = d.sort_values(["section", "segment"]) if "segment" in d.columns else d
    rng = np.random.default_rng(seed); null, null_cyc = [], []
    for _ in range(n_perm):
        dp = d.copy(); dp["coverage"] = dp.groupby("section")["coverage"].transform(lambda x: rng.permutation(x.to_numpy()))
        dc = d.copy(); dc["coverage"] = dc.groupby("section")["coverage"].transform(lambda x: _cyclic_shift(x.to_numpy(), rng))
        for dd_, store in [(dp, null), (dc, null_cyc)]:
            try:
                store.append(float(smf.ols(f, dd_).fit().params["coverage"]))
            except Exception:  # noqa: BLE001
                continue
    null = np.array(null); p_perm = float((np.sum(np.abs(null) >= abs(b)) + 1) / (len(null) + 1)) if len(null) else np.nan
    null_cyc = np.array(null_cyc); p_cyc = float((np.sum(np.abs(null_cyc) >= abs(b)) + 1) / (len(null_cyc) + 1)) if len(null_cyc) else np.nan
    loo = {}
    for g in d[cluster].unique():
        dd = d[d[cluster] != g]
        if dd["section"].nunique() >= 2 and dd.groupby("section")["coverage"].transform("nunique").gt(1).any():
            try:
                loo[str(g)] = float(smf.ols(f, dd).fit().params["coverage"])
            except Exception:  # noqa: BLE001
                pass
    return {"outcome": outcome, "covariates": ",".join(covs) or "none", "n_segments": len(d), "n_sections": d["section"].nunique(), "n_clusters": G,
            "coef": b, "se_cluster": se, "ci95_lo": b - tcrit * se, "ci95_hi": b + tcrit * se, "p_t_gminus1": p_t, "p_normal_ref": float(m.pvalues["coverage"]),
            "p_perm_restricted": p_perm, "p_perm_cyclic": p_cyc, "n_perm": len(null),
            "loo_coef_min": min(loo.values()) if loo else np.nan, "loo_coef_max": max(loo.values()) if loo else np.nan,
            "loo_min_cluster": min(loo, key=loo.get) if loo else "", "loo_max_cluster": max(loo, key=loo.get) if loo else ""}


def _cyclic_shift(x: np.ndarray, rng) -> np.ndarray:
    """Rotate a sequence by a random offset (and reverse it with probability 1/2): keeps its autocorrelation."""
    if len(x) < 2:
        return x
    y = np.roll(x, int(rng.integers(len(x))))
    return y[::-1] if rng.random() < 0.5 else y


def pooled_models(seg: pd.DataFrame, cluster: str, n_perm: int, seed: int, outcomes: list[str] | None = None) -> pd.DataFrame:
    seg = seg[seg["evaluable_T"].astype(bool)].copy() if "evaluable_T" in seg else seg.copy()
    if "E_seg_noniche" in seg.columns and "n_outer_noniche" in seg.columns:
        seg["E_seg_noniche_min5"] = seg["E_seg_noniche"].where(seg["n_outer_noniche"] >= 5)
    full = [f"w_{t}" for t in TYPES if f"w_{t}" in seg.columns]
    sets = {"none": [], "3type": [c for c in ["w_tumor_frac", "w_fibro_frac", "w_myeloid_frac"] if c in seg.columns], "full": full}
    rows = []
    for out in outcomes or OUTCOMES:
        if out not in seg.columns:
            continue
        for name, covs in sets.items():
            r = one_model(seg, out, covs, cluster, n_perm, seed); r["covariate_set"] = name; rows.append(r)
    return pd.DataFrame(rows)
