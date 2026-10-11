"""Section → graph data for training/evaluation.

A SectionGraph holds, for one Visium section:
  x          (n_spots × n_genes) float32 log-normalised expression restricted to the *training* gene list
  edge_near  (2, E1) ring-1 edges          edge_outer (2, E2) ring-2–3 edges
  labels     image-derived region label per spot (string), boundary flags
  meta       section id, patient id, cohort, pitch
Gene lists (HVGs) are chosen by the caller on training patients only and passed in, so this module never
looks at held-out expression to decide features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from .repro import ROOT
from .spatial import ring_edge_index, ring_neighbors


@dataclass
class SectionGraph:
    section: str
    patient: str
    cohort: str
    x: np.ndarray
    edge_near: np.ndarray
    edge_outer: np.ndarray
    xy: np.ndarray
    pitch: float
    barcodes: np.ndarray
    labels: np.ndarray | None = None
    boundary: np.ndarray | None = None
    interface: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.x.shape[0]


def load_qc_sections(cfg: dict, sections: list[str]) -> dict[str, ad.AnnData]:
    proc = ROOT / cfg["paths"]["processed"] / "qc"
    return {s: ad.read_h5ad(proc / f"{s}.h5ad") for s in sections}


def load_labels(cfg: dict, section: str) -> pd.DataFrame | None:
    p = ROOT / cfg["paths"]["processed"] / "histology" / f"{section}_labels.tsv"
    return pd.read_csv(p, sep="\t").set_index("barcode") if p.exists() else None


def holdout_genes(cfg: dict) -> set[str]:
    """Genes excluded from model inputs in the confirmatory configuration: the T-cell readout set and the
    region-label marker panel (so neither the T readout nor the evaluation labels are visible to the model)."""
    g = set(cfg["genes"]["t_cell_holdout"])
    for v in cfg["genes"].get("region_label_panel", {}).values():
        g |= set(v)
    return g


def select_hvgs(adatas: list[ad.AnnData], n_hvg: int, exclude: set[str] | None = None, universe: bool = True) -> list[str]:
    """Seurat-flavour HVGs on the concatenated training sections (batch = section), drawn only from the cross-platform
    gene universe (var["in_universe"]) and excluding held-out genes."""
    cat = ad.concat([ad.AnnData(X=a.layers["lognorm"], obs=pd.DataFrame({"section": a.obs["section"].values}, index=a.obs_names), var=a.var[[]]) for a in adatas], join="inner")
    pool = set(adatas[0].var_names[adatas[0].var["in_universe"].to_numpy(bool)]) if (universe and "in_universe" in adatas[0].var) else set(cat.var_names)
    keep = [g for g in cat.var_names if g in pool and not (exclude and g in exclude)]
    cat = cat[:, keep].copy()
    sc.pp.highly_variable_genes(cat, n_top_genes=n_hvg, batch_key="section", flavor="seurat")
    return sorted(cat.var_names[cat.var["highly_variable"]].tolist())


def fit_scaler(adatas: list[ad.AnnData], genes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = sp.vstack([a[:, genes].layers["lognorm"] for a in adatas]).tocsr()
    mu = np.asarray(X.mean(0)).ravel()
    sq = np.asarray(X.multiply(X).mean(0)).ravel()
    sd = np.sqrt(np.maximum(sq - mu**2, 1e-8))
    return mu.astype(np.float32), sd.astype(np.float32)


def build_graph(a: ad.AnnData, genes: list[str], mu: np.ndarray, sd: np.ndarray, cfg: dict, patient: str, labels: pd.DataFrame | None) -> SectionGraph:
    present = [g for g in genes if g in a.var_names]
    Xp = a[:, present].layers["lognorm"]
    Xp = np.asarray(Xp.todense() if sp.issparse(Xp) else Xp, dtype=np.float32)
    if len(present) < len(genes):  # genes absent from this object (lymph-node objects only) → zero expression
        X = np.zeros((a.n_obs, len(genes)), dtype=np.float32)
        X[:, [i for i, g in enumerate(genes) if g in a.var_names]] = Xp
    else:
        X = Xp
    X = np.clip((X - mu) / sd, -10, 10)
    cohort = a.obs["cohort"].iloc[0]
    if cohort == "discovery":
        xy = a.obsm["spatial"].astype(float); pitch = float(a.uns["spot_pitch_fullres_px"])
    else:
        xy = a.obsm["spatial"].astype(float); pitch = float(a.uns["spot_pitch_img_px"])
    g = cfg["graph"]
    e_near = ring_edge_index(xy, pitch, rings=(g["near_rings"],))
    e_outer = ring_edge_index(xy, pitch, rings=tuple(g["outer_rings"]))
    sg = SectionGraph(section=a.obs["section"].iloc[0], patient=patient, cohort=cohort, x=X, edge_near=e_near, edge_outer=e_outer,
                      xy=xy, pitch=pitch, barcodes=a.obs_names.to_numpy())
    if labels is not None:
        lab = labels.reindex(a.obs_names)
        sg.labels = lab["label"].fillna("unevaluable").to_numpy(dtype=object)
        sg.boundary = lab["tumor_boundary"].fillna(False).to_numpy(bool)
        sg.interface = lab["tumor_lung_interface"].fillna(False).to_numpy(bool)
    return sg


def neighbor_means(sg: SectionGraph) -> tuple[np.ndarray, np.ndarray]:
    """Ring-1 and ring-2–3 mean expression per spot (for the neighbourhood-mean baseline and the aux target)."""
    from .spatial import neighbor_mean

    nb1 = ring_neighbors(sg.xy, sg.pitch, rings=(1,))
    nb23 = ring_neighbors(sg.xy, sg.pitch, rings=(2, 3))
    return np.asarray(neighbor_mean(sg.x, nb1), dtype=np.float32), np.asarray(neighbor_mean(sg.x, nb23), dtype=np.float32)
