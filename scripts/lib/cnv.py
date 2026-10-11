"""Expression-based copy-number signal per spot (inferCNV-style, Tirosh et al. 2016 heuristics).

Given log-normalised expression, a table of gene genomic positions and a reference mask (non-malignant spots,
here lung-parenchyma spots of the same cohort/platform), the smoothed relative expression along each
chromosome is computed and summarised per spot as
    cnv_signal       mean of squared smoothed deviations (magnitude of aneuploidy-like structure)
    cnv_correlation  Pearson correlation of the spot's profile with the mean profile of the top-5 % signal spots
                     of its own section (self-consistency of the section's dominant CNV pattern)
Osteosarcoma is among the most aneuploid human tumours, so this signal separates malignant from reactive
fibroblastic tissue far better than differentiation markers, which fail in poorly differentiated lesions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

CHROMS = [str(i) for i in range(1, 23)] + ["X"]


def ordered_genes(var_names: pd.Index, positions: pd.DataFrame, min_frac_expr: np.ndarray | None = None) -> pd.DataFrame:
    pos = positions.drop_duplicates("gene_name").set_index("gene_name")
    g = pos.loc[pos.index.intersection(var_names)]
    g = g[g["chrom"].isin(CHROMS)].copy()
    g["chrom_order"] = g["chrom"].map({c: i for i, c in enumerate(CHROMS)})
    return g.sort_values(["chrom_order", "start"])


def cnv_profiles(X, var_names: pd.Index, genes: pd.DataFrame, ref_mask: np.ndarray, ref_mean: np.ndarray | None = None,
                 window: int = 100, clip: float = 3.0, step: int = 10) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return (spots × windows) smoothed relative-expression matrix, window chromosome labels, and the reference
    mean used (so an external mean can be passed in for sections without reference spots)."""
    idx = [var_names.get_loc(g) for g in genes.index]
    E = X[:, idx]
    E = np.asarray(E.todense() if sp.issparse(E) else E, dtype=np.float32)
    # centre on the reference spots, clip extreme values (as in inferCNV)
    if ref_mean is None:
        ref_mean = E[ref_mask].mean(0) if ref_mask.sum() >= 20 else E.mean(0)
    R = np.clip(E - ref_mean, -clip, clip)
    chrom = genes["chrom"].to_numpy()
    prof, labels = [], []
    for c in CHROMS:
        cols = np.where(chrom == c)[0]
        if len(cols) < window // 2:
            continue
        Rc = R[:, cols]
        cs = np.cumsum(np.pad(Rc, ((0, 0), (1, 0))), axis=1)
        w = min(window, Rc.shape[1])
        starts = np.arange(0, Rc.shape[1] - w + 1, step)
        sm = (cs[:, starts + w] - cs[:, starts]) / w
        prof.append(sm); labels += [c] * sm.shape[1]
    P = np.hstack(prof)
    # re-centre each spot's profile on its own median (removes global depth effects), as inferCNV does
    P = P - np.median(P, axis=1, keepdims=True)
    return P, labels, ref_mean


def cnv_scores(P: np.ndarray, top_frac: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    signal = (P**2).mean(1)
    k = max(10, int(len(signal) * top_frac))
    top = np.argsort(signal)[-k:]
    ref_profile = P[top].mean(0)
    Pc = P - P.mean(1, keepdims=True); rc = ref_profile - ref_profile.mean()
    corr = (Pc @ rc) / (np.linalg.norm(Pc, axis=1) * np.linalg.norm(rc) + 1e-9)
    return signal, corr
