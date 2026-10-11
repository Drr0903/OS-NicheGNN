"""Spatial helpers for Visium hexagonal arrays (both cohorts), built on coordinates + spot pitch only.

Ring k of a hexagonal lattice sits at centre distances (in pitch units):
    ring 1: 1.0            ring 2: 1.732, 2.0            ring 3: 2.646, 3.0
so rings are separated by the mid-points 1.4, 2.3, 3.3 (pitch units). Edges are created only between
spots that are both present (in tissue, QC-passed); nothing bridges tissue gaps or tears because
missing spots simply have no node.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

RING_BOUNDS = {1: (0.5, 1.4), 2: (1.4, 2.3), 3: (2.3, 3.3)}


def ring_neighbors(xy: np.ndarray, pitch: float, rings=(1,)) -> list[np.ndarray]:
    """Indices of neighbours in the given rings for every spot (list of int arrays, self excluded)."""
    lo = min(RING_BOUNDS[r][0] for r in rings) * pitch
    hi = max(RING_BOUNDS[r][1] for r in rings) * pitch
    tree = cKDTree(xy)
    pairs = tree.query_pairs(hi, output_type="ndarray")
    if len(pairs) == 0:
        return [np.array([], dtype=int) for _ in range(len(xy))]
    d = np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)
    keep = np.zeros(len(pairs), bool)
    for r in rings:
        a, b = RING_BOUNDS[r]
        keep |= (d >= a * pitch) & (d < b * pitch)
    keep &= d >= lo
    pairs = pairs[keep]
    out = [[] for _ in range(len(xy))]
    for i, j in pairs:
        out[i].append(j); out[j].append(i)
    return [np.asarray(sorted(o), dtype=int) for o in out]


def ring_edge_index(xy: np.ndarray, pitch: float, rings=(1,)) -> np.ndarray:
    """(2, E) directed edge index (both directions) for the given rings."""
    nb = ring_neighbors(xy, pitch, rings)
    src = np.concatenate([np.full(len(n), i) for i, n in enumerate(nb)]) if len(nb) else np.array([], int)
    dst = np.concatenate(nb) if len(nb) else np.array([], int)
    return np.vstack([src, dst]).astype(np.int64)


def estimate_pitch(xy: np.ndarray) -> float:
    tree = cKDTree(xy)
    d, _ = tree.query(xy, k=2)
    return float(np.median(d[:, 1]))


def neighbor_mean(X, nb: list[np.ndarray]):
    """Mean of X over each spot's neighbour set (rows with no neighbours get their own value)."""
    import scipy.sparse as sp

    n = len(nb)
    rows = np.concatenate([np.full(len(v), i) for i, v in enumerate(nb)])
    cols = np.concatenate(nb)
    w = np.concatenate([np.full(len(v), 1.0 / len(v)) for v in nb if len(v)])
    A = sp.csr_matrix((w, (rows, cols)), shape=(n, n))
    empty = np.array([len(v) == 0 for v in nb])
    if empty.any():
        A = A + sp.csr_matrix((np.ones(empty.sum()), (np.where(empty)[0], np.where(empty)[0])), shape=(n, n))
    return A @ X
