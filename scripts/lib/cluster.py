"""Graph-based niche clustering with a fixed number of clusters .

Leiden community detection on a *batch-balanced* k-nearest-neighbour graph of the embedding (BBKNN-style: every spot is
linked to its k_within nearest spots of every patient, so communities must span patients), with the resolution searched so
that exactly K communities result (binary search; if no resolution gives exactly K, the closest solution with ≥ K
communities has its smallest communities merged into the nearest centroid). Prototypes are community centroids, so frozen
transfer and the unmatched rule work exactly as before (nearest-prototype assignment); the agreement between community
labels and nearest-centroid labels is returned so that the loss of information in that step is measurable.
A plain kNN graph of a patient-dominated embedding is essentially a union of per-patient graphs and Leiden then returns the
patients themselves at any resolution — that is why the graph is balanced across patients here.
"""
from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits


def balanced_knn_graph(Z: np.ndarray, batch: np.ndarray, k_within: int = 3) -> sp.csr_matrix:
    """Symmetric binary adjacency linking every row of Z to its k_within nearest rows of each batch (self excluded)."""
    n = len(Z); rows, cols = [], []
    for b in np.unique(batch):
        idx = np.where(batch == b)[0]
        kq = min(k_within + 1, len(idx))
        nn = NearestNeighbors(n_neighbors=kq).fit(Z[idx]); _, nb = nn.kneighbors(Z)
        for j in range(kq):
            tgt = idx[nb[:, j]]; keep = tgt != np.arange(n)
            rows.append(np.arange(n)[keep]); cols.append(tgt[keep])
    r = np.concatenate(rows); c = np.concatenate(cols)
    A = sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(n, n)); A = A.maximum(A.T); A.setdiag(0); A.eliminate_zeros()
    return A


def leiden_fixed_k(Z: np.ndarray, k: int, seed: int, batch: np.ndarray | None = None, n_neighbors: int = 15, k_within: int = 3, max_iter: int = 40) -> tuple[np.ndarray, float]:
    """Leiden labels (0..k−1) for the rows of Z and the resolution used. `batch` (e.g. patient) → balanced graph."""
    with threadpool_limits(limits=8), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = ad.AnnData(np.asarray(Z, dtype=np.float32))
        if batch is not None:
            A = balanced_knn_graph(Z, np.asarray(batch), k_within); a.obsp["connectivities"] = A; a.uns["neighbors"] = {"connectivities_key": "connectivities", "params": {"method": "bbknn_like"}}
        else:
            sc.pp.neighbors(a, n_neighbors=n_neighbors, use_rep="X", random_state=seed)
        lo, hi, best, lab = 1e-4, 5.0, None, None
        for _ in range(max_iter):
            res = float(np.sqrt(lo * hi))
            sc.tl.leiden(a, resolution=res, flavor="igraph", n_iterations=-1, directed=False, random_state=seed, key_added="l", obsp="connectivities")
            lab = a.obs["l"].astype(int).to_numpy(); n = lab.max() + 1
            if n == k:
                return _relabel(lab), res
            if n > k:
                hi = res
                if best is None or n < best[1]:
                    best = (lab, n, res)
            else:
                lo = res
            if hi / lo < 1.02:
                break
    lab, n, res = best if best is not None else (lab, lab.max() + 1, res)
    return _merge_to_k(Z, _relabel(lab), k), res


def _relabel(lab: np.ndarray) -> np.ndarray:
    _, inv = np.unique(lab, return_inverse=True); return inv


def _merge_to_k(Z: np.ndarray, lab: np.ndarray, k: int) -> np.ndarray:
    lab = lab.copy()
    while lab.max() + 1 > k:
        sizes = np.bincount(lab); small = int(np.argmin(sizes)); cents = np.vstack([Z[lab == c].mean(0) for c in range(lab.max() + 1)])
        d = np.linalg.norm(cents - cents[small], axis=1); d[small] = np.inf; target = int(np.argmin(d))
        lab[lab == small] = target; lab = _relabel(lab)
    while lab.max() + 1 < k:  # too few communities: split the largest along its principal axis
        sizes = np.bincount(lab); big = int(np.argmax(sizes)); idx = np.where(lab == big)[0]; X = Z[idx] - Z[idx].mean(0)
        u = np.linalg.svd(X, full_matrices=False)[2][0]; lab[idx[(X @ u) > 0]] = lab.max() + 1
    return lab


def integrate(emb: dict, patient_of: dict, seed: int, cfg: dict) -> dict:
    """A20 addendum 5: atlas space = standardised latent → first 30 principal components → Harmony with patient as batch,
    over all spots of all sections (label-free; config `niches.integration`, "none" keeps the latent as it is)."""
    if cfg["niches"].get("integration", "none") == "none":
        return emb
    import harmonypy as hm
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    secs = list(emb); Z = np.vstack([emb[s] for s in secs]); Zp = PCA(30, random_state=seed).fit_transform(StandardScaler().fit_transform(Z))
    ho = hm.run_harmony(Zp, pd.DataFrame({"p": np.concatenate([[patient_of[s]] * len(emb[s]) for s in secs])}), "p", random_state=seed, verbose=False, ncores=int(__import__("os").environ.get("OMP_NUM_THREADS", 8)))
    Zc = np.asarray(ho.Z_corr); Zc = Zc.T if Zc.shape[0] != Zp.shape[0] else Zc; out, o = {}, 0
    for s in secs:
        out[s] = Zc[o:o + len(emb[s])].astype(np.float32); o += len(emb[s])
    return out


def atlas_batch(cfg: dict, P: np.ndarray) -> np.ndarray | None:
    """A20: the atlas graph is a plain kNN graph unless config `niches.graph` asks for the patient-balanced one."""
    return np.asarray(P) if cfg["niches"].get("graph", "balanced") == "balanced" else None


def centroids(Z: np.ndarray, lab: np.ndarray) -> np.ndarray:
    return np.vstack([Z[lab == c].mean(0) for c in range(lab.max() + 1)]).astype(np.float32)


def assign(Z: np.ndarray, protos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.linalg.norm(Z[:, None, :] - protos[None], axis=2)
    return d.argmin(1), d.min(1)


def cluster_prototypes(Z: np.ndarray, k: int, seed: int, batch: np.ndarray | None = None, n_neighbors: int = 15, k_within: int = 3) -> tuple[np.ndarray, np.ndarray, dict]:
    """Leiden with fixed K on Z → (community labels, prototypes = centroids, info{resolution, centroid_agreement_ari})."""
    lab, res = leiden_fixed_k(Z, k, seed, batch, n_neighbors, k_within)
    protos = centroids(Z, lab); lab_c, _ = assign(Z, protos)
    return lab, protos, {"resolution": res, "centroid_agreement_ari": float(adjusted_rand_score(lab, lab_c))}


def consensus_partition(labels: list[np.ndarray], k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Consensus of several partitions of the same rows (A10): average-linkage hierarchical clustering of the co-association
    matrix into k groups. Returns consensus labels and the per-partition ARI with the consensus."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    n = len(labels[0]); co = np.zeros((n, n), dtype=np.float32)
    for lab in labels:
        co += (lab[:, None] == lab[None, :])
    co /= len(labels); Z = linkage(squareform(1.0 - co, checks=False), method="average"); cons = fcluster(Z, k, criterion="maxclust") - 1
    cons = _relabel(cons)
    if cons.max() + 1 != k:  # co-association rows act as the feature space for merging/splitting
        cons = _merge_to_k(co, cons, k)
    return cons, np.array([adjusted_rand_score(cons, lab) for lab in labels])


def consensus_prototypes(Zs: list[np.ndarray], k: int, seed: int, batch: np.ndarray | None = None, k_within: int = 3) -> tuple[np.ndarray, np.ndarray, dict]:
    """A10: balanced Leiden in every seed's embedding of the SAME rows, consensus partition, prototypes = consensus-group
    centroids in the first embedding. Returns (consensus labels, prototypes, info)."""
    labs = [leiden_fixed_k(Z, k, seed, batch, k_within=k_within)[0] for Z in Zs]
    cons, agree = consensus_partition(labs, k, seed); protos = centroids(Zs[0], cons); lab_c, _ = assign(Zs[0], protos)
    return cons, protos, {"seed_agreement_with_consensus": agree.tolist(), "centroid_agreement_ari": float(adjusted_rand_score(cons, lab_c)), "n_seeds": len(Zs)}
