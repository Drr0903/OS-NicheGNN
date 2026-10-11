"""helpers of the pathologist-annotated clustering benchmark (stage 27b).

Cohort loading (stage-27a files), profile construction ([expression PCA-50, H&E PCA-50] fitted on a given set of sections,
each block divided by one global s.d., as in stages 23/24), fixed-K clustering (Leiden with the resolution searched for
exactly K on a plain kNN graph, and k-means), nearest-centroid transfer, and ARI / NMI against the pathologist classes.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from .cluster import assign, centroids, leiden_fixed_k

UNSCORED = "unscored"


def load(cohort_dir: Path) -> tuple[dict[str, ad.AnnData], dict[str, np.ndarray], dict[str, str]]:
    A, F, P = {}, {}, {}
    for f in sorted(cohort_dir.glob("*.h5ad")):
        s = f.stem; a = ad.read_h5ad(f); a.obs["region"] = a.obs["truth"].astype(str); a.obs["section"] = s  # lib.generality reads "region", "section"
        A[s] = a; F[s] = np.load(cohort_dir / f"{s}_img.npy"); P[s] = str(a.obs["patient"].iloc[0])
        assert len(F[s]) == a.n_obs, f"{s}: image rows differ from spots"
    return A, F, P


def profiles(G: dict, F: dict, fit: list[str], dim: int, seed: int) -> dict:
    """Graphs whose x = [expression PCA, H&E PCA], both PCA fitted on the sections `fit` only."""
    pe = PCA(dim, random_state=seed).fit(np.vstack([G[s].x for s in fit])); se = float(np.vstack([pe.transform(G[s].x) for s in fit]).std())
    pi = PCA(dim, random_state=seed).fit(np.vstack([F[s] for s in fit])); si = float(np.vstack([pi.transform(F[s]) for s in fit]).std())
    return {s: dataclasses.replace(g, x=np.hstack([pe.transform(g.x) / se, pi.transform(F[s]) / si]).astype(np.float32)) for s, g in G.items()}


def kmeans_fixed(Z: np.ndarray, k: int, seed: int) -> np.ndarray:
    return KMeans(k, n_init=10, random_state=seed).fit_predict(Z)


def reduce(Z: np.ndarray, seed: int, n_pc: int = 30, batch: np.ndarray | None = None) -> np.ndarray:
    """A21 addendum 5: every representation is standardised and reduced to its first 30 principal components before
    clustering (the same for all methods); with `batch`, Harmony (patient) is applied to those components."""
    Zs = StandardScaler().fit_transform(Z); Zp = PCA(min(n_pc, Zs.shape[1]), random_state=seed).fit_transform(Zs)
    if batch is None:
        return Zp
    import harmonypy as hm
    import pandas as pd
    ho = hm.run_harmony(Zp, pd.DataFrame({"b": batch}), "b", random_state=seed, verbose=False); Zc = np.asarray(ho.Z_corr)
    return Zc.T if Zc.shape[0] != Zp.shape[0] else Zc


def gmm_fixed(Z: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Gaussian mixture with one covariance shared by all components (the EEE model of mclust used by STAGATE and GraphST)."""
    from sklearn.mixture import GaussianMixture
    return GaussianMixture(k, covariance_type="tied", n_init=3, random_state=seed).fit_predict(Z)


def refine(lab: np.ndarray, xy: np.ndarray, pitch: float) -> np.ndarray:
    """Label refinement used by spatial-domain methods: each spot takes the majority label of itself and its ring-1 neighbours."""
    from .spatial import ring_neighbors
    nb = ring_neighbors(xy, pitch, rings=(1,)); out = lab.copy()
    for i, v in enumerate(nb):
        vals, cnt = np.unique(np.append(lab[v], lab[i]), return_counts=True); out[i] = vals[cnt.argmax()]
    return out


def cluster(Z: np.ndarray, k: int, seed: int, how: str, prep: str = "raw", batch: np.ndarray | None = None) -> np.ndarray:
    Zs = StandardScaler().fit_transform(Z) if prep == "raw" else reduce(Z, seed, batch=batch if prep == "pc30_harmony" else None)
    return leiden_fixed_k(Zs, k, seed)[0] if how == "leiden" else gmm_fixed(Zs, k, seed) if how == "gmm" else kmeans_fixed(Zs, k, seed)


def transfer(emb: dict, train: list[str], test: list[str], k: int, seed: int, how: str, per_section: int = 3000, prep: str = "raw") -> dict[str, np.ndarray]:
    """Cluster (a per-section subsample of) the training sections, prototypes = centroids, held-out spots → nearest prototype.
    prep "pc30": standardisation and 30 principal components fitted on the training sample, applied to the held-out spots."""
    rng = np.random.default_rng(seed); Z = []
    for s in train:
        e = emb[s]; Z.append(e[rng.choice(len(e), min(per_section, len(e)), replace=False)])
    Z = np.vstack(Z); sc0 = StandardScaler().fit(Z)
    if prep == "pc30":
        pca = PCA(min(30, Z.shape[1]), random_state=seed).fit(sc0.transform(Z))
        class _T:
            def transform(self, X):
                return pca.transform(sc0.transform(X))
        sc = _T()
    else:
        sc = sc0
    Zs = sc.transform(Z)
    lab = leiden_fixed_k(Zs, k, seed)[0] if how == "leiden" else kmeans_fixed(Zs, k, seed)
    protos = centroids(Zs, lab)
    return {s: assign(sc.transform(emb[s]), protos)[0] for s in test}


def scores(lab: np.ndarray, truth: np.ndarray) -> dict:
    ok = truth != UNSCORED
    return {"ari": float(adjusted_rand_score(truth[ok], lab[ok])), "nmi": float(normalized_mutual_info_score(truth[ok], lab[ok])), "n_scored": int(ok.sum())}
