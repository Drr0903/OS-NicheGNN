"""fold helpers shared by stage 23: fold-fitted PCA input, inductive anchor-direction projection, and the
paired patient-level bootstrap used for every comparison."""
from __future__ import annotations

import dataclasses

import numpy as np
from sklearn.decomposition import PCA

from .upgrade import mnn_anchors


def pca_graphs(G: dict, train: list[str], dim: int, seed: int) -> dict:
    """PCA fitted on the training patients' standardised expression; scores divided by one global training s.d. (= stage 22)."""
    pca = PCA(dim, random_state=seed).fit(np.vstack([G[s].x for s in train])); sc = float(np.vstack([pca.transform(G[s].x) for s in train]).std())
    return {s: dataclasses.replace(g, x=(pca.transform(g.x) / sc).astype(np.float32)) for s, g in G.items()}


def mnn_projection(emb: dict[str, np.ndarray], G: dict, train: list[str], rank: int, seed: int) -> dict[str, np.ndarray]:
    """Project out the top `rank` principal directions of cross-patient MNN anchor differences, estimated on training patients
    only (fastMNN-style batch vectors), from every section's representation. Centred on the training mean, not rescaled, so the
geometry the probe sees is unchanged apart from the removed directions."""
    Ztr = np.vstack([emb[s] for s in train]); mu = Ztr.mean(0)
    std = {s: (e - mu).astype(np.float32) for s, e in emb.items()}
    graphs = [dataclasses.replace(G[s], x=std[s]) for s in train]
    anc = mnn_anchors(graphs, n_pcs=min(50, Ztr.shape[1]), k=10, seed=seed)
    diffs = np.vstack([std[a][ia] - std[b][ib] for (a, b), (ia, ib) in anc.items() if a < b])
    V = PCA(rank, random_state=seed).fit(diffs).components_          # (rank, dim), orthonormal rows
    return {s: (e - (e @ V.T) @ V).astype(np.float32) for s, e in std.items()}


def paired(a, b, seed: int) -> dict:
    from scipy import stats
    d = (a - b).dropna(); rng = np.random.default_rng(seed); boots = [rng.choice(d.to_numpy(), len(d)).mean() for _ in range(5000)]
    return {"n_patients": len(d), "mean_diff": float(d.mean()), "ci95_lo": float(np.percentile(boots, 2.5)), "ci95_hi": float(np.percentile(boots, 97.5)),
            "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum()), "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if (d != 0).any() else 1.0}


def image_graphs(G: dict, Gp: dict, train: list[str], dim: int, seed: int, feat_dir) -> tuple[dict, dict]:
    """Phikon morphology features (stage 04c) → PCA-`dim` fitted on the training patients, scaled like the expression PCA.
    Returns (image-only graphs, multimodal graphs with x = [expression PCA, image PCA])."""
    F = {s: np.load(feat_dir / f"{s}.npy") for s in G}
    for s in G:
        assert len(F[s]) == G[s].n, f"{s}: image rows {len(F[s])} != spots {G[s].n}"
    pca = PCA(dim, random_state=seed).fit(np.vstack([F[s] for s in train])); sc = float(np.vstack([pca.transform(F[s]) for s in train]).std())
    I = {s: (pca.transform(F[s]) / sc).astype(np.float32) for s in G}
    return ({s: dataclasses.replace(G[s], x=I[s]) for s in G}, {s: dataclasses.replace(Gp[s], x=np.hstack([Gp[s].x, I[s]])) for s in G})
