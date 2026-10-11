"""BANKSY embedding (Singhal et al., Nat Genet 2024) implemented directly on our SectionGraphs.

For every spot the BANKSY matrix concatenates
    sqrt(1-λ) · x_i                     own expression
    sqrt(λ/2) · m_i                     Gaussian-weighted mean expression of the k_geom nearest neighbours
    sqrt(λ/2) · |g_i|                   magnitude of the azimuthal Gabor filter (order 1) over the same neighbours
followed by PCA. λ = 0.8 is the authors' recommended value for domain segmentation, k_geom = 15, 20 PCs.
Being a fixed feature transform + PCA, BANKSY is inductive: the PCA is fitted on training patients and
applied unchanged to held-out / external sections, exactly like every other frozen representation here.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA


def banksy_features(x: np.ndarray, xy: np.ndarray, lam: float = 0.8, k_geom: int = 15) -> np.ndarray:
    tree = cKDTree(xy)
    d, idx = tree.query(xy, k=k_geom + 1)
    d, idx = d[:, 1:], idx[:, 1:]
    sigma = np.median(d[:, -1]) / 1.5
    w = np.exp(-(d**2) / (2 * sigma**2)); w = w / w.sum(1, keepdims=True)
    mean_nb = np.einsum("ik,ikg->ig", w, x[idx])
    # azimuthal Gabor filter, order 1: sum_k w_k x_k exp(i·θ_k)
    vec = xy[idx] - xy[:, None, :]
    theta = np.arctan2(vec[..., 1], vec[..., 0])
    re = np.einsum("ik,ikg->ig", w * np.cos(theta), x[idx])
    im = np.einsum("ik,ikg->ig", w * np.sin(theta), x[idx])
    agf = np.sqrt(re**2 + im**2)
    return np.hstack([np.sqrt(1 - lam) * x, np.sqrt(lam / 2) * mean_nb, np.sqrt(lam / 2) * agf]).astype(np.float32)


class BanksyEmbedder:
    def __init__(self, lam: float = 0.8, k_geom: int = 15, n_pcs: int = 20, seed: int = 0):
        self.lam, self.k_geom, self.n_pcs, self.seed = lam, k_geom, n_pcs, seed
        self.pca: PCA | None = None

    def fit(self, graphs) -> "BanksyEmbedder":
        F = np.vstack([banksy_features(g.x, g.xy, self.lam, self.k_geom) for g in graphs])
        self.pca = PCA(self.n_pcs, random_state=self.seed).fit(F)
        return self

    def transform(self, g) -> np.ndarray:
        return self.pca.transform(banksy_features(g.x, g.xy, self.lam, self.k_geom)).astype(np.float32)
