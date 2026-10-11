"""Osteosarcoma-specific blocks of the OS-NicheGNN spot profile.

cnv_block         inferred copy-number profile of every spot: log-normalized expression ordered along the genome, centred on
                  the mean of the training patients' lung-parenchyma spots, smoothed in windows of 100 genes (lib.cnv), then
                  reduced to `dim` principal components fitted on the training patients and scaled like the other blocks.
                  Osteosarcoma is among the most aneuploid human tumours, so this block separates tumour subclones and the
                  tumour from its host tissue. With cnv_target the block is a target only (placed before the lineage block,
                  both at the end of the profile): the encoder sees expression and morphology and must reconstruct it.
programme_block   osteoblastic, chondroblastic and fibroblastic differentiation scores (mean z-score of marker genes, gene
                  means and s.d. from the training patients), the three histological lineages of conventional osteosarcoma.
                  Appended as target-only columns: the encoder never sees them, the decoder must reconstruct them from z.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .cnv import cnv_profiles, ordered_genes
from .repro import ROOT

PROGRAMMES = {
    "osteoblastic": ["RUNX2", "SP7", "ALPL", "IBSP", "SPP1", "BGLAP", "COL1A1", "PHEX", "DMP1", "SATB2"],
    "chondroblastic": ["SOX9", "COL2A1", "ACAN", "COL9A1", "COL11A2", "MATN3", "COMP", "SOX5", "SOX6"],
    "fibroblastic": ["POSTN", "COL3A1", "FAP", "PDGFRB", "DCN", "LUM", "COL6A3", "THY1"],
}


def _rows(a, g) -> np.ndarray:
    ix = pd.Index(a.obs_names).get_indexer(g.barcodes); assert (ix >= 0).all(), g.section; return ix


def cnv_block(adatas: dict, labels: dict, G: dict, train: list[str], dim: int, seed: int, max_fit: int = 20000) -> dict[str, np.ndarray]:
    positions = pd.read_csv(ROOT / "metadata/gene_positions_grch38_ens113.tsv", sep="\t")
    genes = ordered_genes(adatas[train[0]].var_names, positions)
    for s in G:
        genes = genes.loc[genes.index.intersection(adatas[s].var_names)]
    genes = genes.sort_values(["chrom_order", "start"]); idx = {s: [adatas[s].var_names.get_loc(x) for x in genes.index] for s in G}
    ref = []
    for s in train:
        a = adatas[s]; r = _rows(a, G[s]); lab = labels[s]["label"].reindex(G[s].barcodes).to_numpy()
        m = lab == "lung_parenchyma"
        if m.sum() >= 20:
            E = a.layers["lognorm"][r[m]][:, idx[s]]; ref.append(np.asarray(E.todense() if hasattr(E, "todense") else E))
    ref_mean = np.vstack(ref).mean(0).astype(np.float32)
    P = {}
    for s in G:
        a = adatas[s]; X = a.layers["lognorm"][_rows(a, G[s])]
        P[s], _, _ = cnv_profiles(X, a.var_names, genes, np.zeros(X.shape[0], bool), ref_mean=ref_mean)
    rng = np.random.default_rng(seed); fit = np.vstack([P[s] for s in train]); fit = fit[rng.choice(len(fit), min(max_fit, len(fit)), replace=False)]
    pca = PCA(dim, random_state=seed).fit(fit); sc = float(pca.transform(fit).std())
    return {s: (pca.transform(P[s]) / sc).astype(np.float32) for s in G}


def programme_block(adatas: dict, G: dict, train: list[str], weight: float) -> dict[str, np.ndarray]:
    def expr(s, gs):
        a = adatas[s]; cols = [a.var_names.get_loc(x) for x in gs]; E = a.layers["lognorm"][_rows(a, G[s])][:, cols]
        return np.asarray(E.todense() if hasattr(E, "todense") else E, dtype=np.float32)
    out = {s: [] for s in G}
    for gs in PROGRAMMES.values():
        gs = [x for x in gs if all(x in adatas[s].var_names for s in G)]
        tr = np.vstack([expr(s, gs) for s in train]); mu, sd = tr.mean(0), tr.std(0) + 1e-6
        sc = {s: ((expr(s, gs) - mu) / sd).mean(1) for s in G}; m_, s_ = np.concatenate([sc[s] for s in train]).mean(), np.concatenate([sc[s] for s in train]).std() + 1e-6
        for s in G:
            out[s].append(weight * (sc[s] - m_) / s_)
    return {s: np.column_stack(v).astype(np.float32) for s, v in out.items()}


def with_os_blocks(Gm: dict, adatas: dict, labels: dict, train: list[str], ac: dict, seed: int) -> dict:
    """Append the copy-number block (input) and the lineage block (target-only, last columns) to every graph."""
    extra = {s: [g.x] for s, g in Gm.items()}
    if ac.get("cnv_dim", 0) > 0:
        C = cnv_block(adatas, labels, Gm, train, ac["cnv_dim"], seed)
        for s in Gm:
            extra[s].append(C[s])
    if ac.get("programme_weight", 0) > 0:
        L = programme_block(adatas, Gm, train, ac["programme_weight"])
        for s in Gm:
            extra[s].append(L[s])
    return {s: dataclasses.replace(g, x=np.hstack(extra[s]).astype(np.float32)) for s, g in Gm.items()}
