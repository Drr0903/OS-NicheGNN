"""Stage 05a — coarse single-cell reference for spot deconvolution.

Sources
  GSE162454  6 primary osteosarcoma lesions (10x 3')         → tumour / stromal / immune reference
  GSE152048  11 osteosarcoma lesions (10x 3')                → same, adds lung-metastasis lesions
  HLCA core  normal lung parenchyma cells (CELLxGENE h5ad)   → lung epithelium, alveolar macrophage, lung endothelium

Procedure (all in this script, seeded)
  1. load, basic cell QC, subsample per sample, concatenate osteosarcoma cells
  2. Leiden clustering on the osteosarcoma cells; clusters named by marker-score majority into coarse types
     (Osteosarcoma, Fibroblast, Endothelial, Pericyte_SMC, Myeloid, Osteoclast, T_NK, B_plasma, Mast);
     clusters whose top marker score is ambiguous (margin < 0.15) are dropped
  3. HLCA parenchyma cells mapped from ann_level_3 to Lung_epithelium (AT1/AT2/Secretory/Multiciliated),
     Alveolar_macrophage (Macrophages with MARCO/FABP4 high), Endothelial → merged as Endothelial
  4. cap cells per type, write data/processed/reference/reference_counts.mtx + cell_types.tsv (RCTD input)
     and results/05_deconvolution/reference_composition.tsv, reference_markers.tsv
"""
from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "05_deconvolution"
MARKERS = {
    "Osteosarcoma": ["COL1A1", "COL1A2", "COL3A1", "RUNX2", "SP7", "ALPL", "IBSP", "SATB2", "SPP1", "MKI67", "TOP2A", "CDK1"],
    "Fibroblast": ["DCN", "LUM", "PDGFRA", "CFD", "APOD", "C7", "FBLN1"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "PLVAP", "EGFL7"],
    "Pericyte_SMC": ["RGS5", "ACTA2", "MYH11", "TAGLN", "NOTCH3", "PDGFRB"],
    "Myeloid": ["CD68", "CD163", "LYZ", "CD14", "C1QA", "C1QB", "FCGR3A", "S100A8", "S100A9", "MS4A7"],
    "Osteoclast": ["ACP5", "CTSK", "MMP9", "ATP6V0D2", "DCSTAMP", "OCSTAMP"],
    "T_NK": ["CD3D", "CD3E", "CD2", "CD8A", "IL7R", "NKG7", "GNLY", "KLRD1", "TRAC"],
    "B_plasma": ["MS4A1", "CD79A", "CD79B", "MZB1", "JCHAIN", "IGKC", "IGHG1"],
    "Mast": ["TPSAB1", "TPSB2", "CPA3", "KIT", "MS4A2"],
}
HLCA_MAP = {"AT1": "Lung_epithelium", "AT2": "Lung_epithelium", "Secretory": "Lung_epithelium", "Multiciliated lineage": "Lung_epithelium",
            "Basal": "Lung_epithelium", "EC capillary": "Endothelial", "EC arterial": "Endothelial", "EC venous": "Endothelial"}


def read_10x(dirpath: Path, prefix: str = "") -> ad.AnnData:
    m = scipy.io.mmread(glob.glob(str(dirpath / f"{prefix}matrix.mtx*"))[0]).T.tocsr().astype(np.float32)
    feats = pd.read_csv(glob.glob(str(dirpath / f"{prefix}features.tsv*"))[0], sep="\t", header=None)
    bcs = pd.read_csv(glob.glob(str(dirpath / f"{prefix}barcodes.tsv*"))[0], sep="\t", header=None)[0].astype(str)
    a = ad.AnnData(X=m, obs=pd.DataFrame(index=bcs), var=pd.DataFrame({"gene_id": feats[0].values}, index=feats[1].astype(str).values))
    a.var_names_make_unique()
    return a


def load_osteosarcoma(cfg: dict, max_cells_per_sample: int, seed: int) -> ad.AnnData:
    parts = []
    for d in sorted(glob.glob(str(ROOT / cfg["paths"]["raw_gse162454"] / "*_matrix.mtx.gz"))):
        prefix = Path(d).name.replace("matrix.mtx.gz", "")
        a = read_10x(Path(d).parent, prefix)
        a.obs["sample"] = prefix.split("_OS_")[0] + "_OS" + prefix.split("_OS")[1].rstrip("_")
        a.obs["dataset"] = "GSE162454"
        parts.append(a)
    for d in sorted(glob.glob(str(ROOT / cfg["paths"]["raw_gse152048"] / "extracted" / "*" / "*" / "matrix.mtx.gz"))):
        a = read_10x(Path(d).parent)
        a.obs["sample"] = Path(d).parent.name
        a.obs["dataset"] = "GSE152048"
        parts.append(a)
    out = []
    rng = np.random.default_rng(seed)
    for a in parts:
        sc.pp.filter_cells(a, min_genes=300)
        a.obs["pct_mito"] = np.asarray(a[:, a.var_names.str.startswith("MT-")].X.sum(1)).ravel() / np.maximum(np.asarray(a.X.sum(1)).ravel(), 1) * 100
        a = a[a.obs["pct_mito"] < 20].copy()
        if a.n_obs > max_cells_per_sample:
            a = a[rng.choice(a.n_obs, max_cells_per_sample, replace=False)].copy()
        a.obs_names = [f"{a.obs['sample'].iloc[0]}:{b}" for b in a.obs_names]
        out.append(a)
        log(f"  {a.obs['sample'].iloc[0]} ({a.obs['dataset'].iloc[0]}): {a.n_obs} cells")
    common = set.intersection(*[set(a.var_names) for a in out])
    out = [a[:, sorted(common)].copy() for a in out]
    return ad.concat(out, join="inner", merge="same")


def annotate_clusters(a: ad.AnnData, seed: int) -> ad.AnnData:
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=2000, batch_key="sample")
    sc.pp.pca(a, n_comps=40, mask_var="highly_variable", random_state=seed)
    sc.pp.neighbors(a, n_neighbors=15, random_state=seed)
    sc.tl.leiden(a, resolution=1.5, random_state=seed, flavor="igraph", n_iterations=2, directed=False)
    for t, genes in MARKERS.items():
        sc.tl.score_genes(a, [g for g in genes if g in a.var_names], score_name=f"score_{t}", random_state=seed)
    S = a.obs[[f"score_{t}" for t in MARKERS]].copy(); S.columns = list(MARKERS)
    S = (S - S.mean()) / S.std()
    cl_mean = S.groupby(a.obs["leiden"].to_numpy()).mean()
    top = cl_mean.idxmax(1); srt = np.sort(cl_mean.to_numpy(), axis=1); margin = pd.Series(srt[:, -1] - srt[:, -2], index=cl_mean.index)
    names = {cl: (top[cl] if margin[cl] >= 0.15 else "ambiguous") for cl in cl_mean.index}
    a.obs["coarse_type"] = a.obs["leiden"].map(names).astype(str)
    a.uns["cluster_annotation"] = pd.DataFrame({"leiden": cl_mean.index, "top": top.values, "margin": margin.values, "n_cells": a.obs["leiden"].value_counts().reindex(cl_mean.index).values}).to_dict("list")
    return a


def load_hlca(cfg: dict, n_per_type: int, seed: int) -> ad.AnnData:
    path = ROOT / cfg["paths"]["raw_hlca"]
    rng = np.random.default_rng(seed)
    with h5py.File(path) as f:
        obs = f["obs"]
        def cat(k):
            g = obs[k]; return g["categories"][:].astype(str)[g["codes"][:]]
        tissue, lvl3, cond = cat("tissue"), cat("ann_level_3"), cat("lung_condition")
        finest = cat("ann_finest_level")
        keep_mask = (tissue == "lung parenchyma") & (cond == "Healthy")
        target = np.array([HLCA_MAP.get(x, "") for x in lvl3], dtype=object)
        # alveolar macrophages by finest annotation
        target[np.isin(finest, ["Alveolar macrophages", "Alveolar Mφ CCL3+", "Alveolar Mφ MT-positive", "Alveolar Mφ proliferating", "Alveolar macrophages CCL3+"])] = "Alveolar_macrophage"
        target[np.char.find(finest.astype(str), "Alveolar M") >= 0] = "Alveolar_macrophage"
        idx = []
        for t in ["Lung_epithelium", "Alveolar_macrophage", "Endothelial"]:
            ii = np.where(keep_mask & (target == t))[0]
            if len(ii) > n_per_type:
                ii = rng.choice(ii, n_per_type, replace=False)
            idx.append(ii); log(f"  HLCA {t}: {len(ii)} cells (of {(keep_mask & (target == t)).sum()})")
        idx = np.sort(np.concatenate(idx))
        var = f["var"]; gene_names = var["feature_name"]["categories"][:].astype(str)[var["feature_name"]["codes"][:]] if isinstance(var["feature_name"], h5py.Group) else var["feature_name"][:].astype(str)
        rx = f["raw"]["X"]
        indptr = rx["indptr"][:]
        rows = []
        for i in idx:
            s, e = indptr[i], indptr[i + 1]
            rows.append((rx["indices"][s:e], rx["data"][s:e]))
        n_genes = f["raw"]["X"].attrs["shape"][1]
        data = np.concatenate([r[1] for r in rows]); indices = np.concatenate([r[0] for r in rows])
        ptr = np.concatenate([[0], np.cumsum([len(r[0]) for r in rows])])
        X = sp.csr_matrix((data.astype(np.float32), indices, ptr), shape=(len(idx), n_genes))
        cell_ids = obs["_index"][:].astype(str)[idx] if "_index" in obs else np.array([f"hlca_{i}" for i in idx])
    a = ad.AnnData(X=X, obs=pd.DataFrame({"coarse_type": target[idx], "sample": "HLCA", "dataset": "HLCA"}, index=cell_ids), var=pd.DataFrame(index=gene_names))
    a.var_names_make_unique()
    a.layers["counts"] = a.X.copy()
    return a


def main() -> None:
    t0 = time.time()
    cfg = load_config(); set_global_seed(cfg["seed"])
    dc = cfg["deconvolution"]
    out_dir = ROOT / cfg["paths"]["processed"] / "reference"; out_dir.mkdir(parents=True, exist_ok=True)
    log("loading osteosarcoma scRNA-seq")
    os_ref = load_osteosarcoma(cfg, max_cells_per_sample=4000, seed=cfg["seed"])
    log(f"osteosarcoma cells: {os_ref.n_obs} × {os_ref.n_vars} genes; clustering")
    os_ref = annotate_clusters(os_ref, cfg["seed"])
    log(f"coarse types:\n{os_ref.obs['coarse_type'].value_counts().to_string()}")
    log("loading HLCA parenchyma subset")
    hl = load_hlca(cfg, n_per_type=dc["reference_max_cells_per_type"], seed=cfg["seed"])
    common = sorted(set(os_ref.var_names) & set(hl.var_names))
    ref = ad.concat([ad.AnnData(X=os_ref.layers["counts"], obs=os_ref.obs[["sample", "dataset", "coarse_type"]], var=os_ref.var[[]])[:, common],
                     ad.AnnData(X=hl.layers["counts"], obs=hl.obs[["sample", "dataset", "coarse_type"]], var=hl.var[[]])[:, common]], join="inner")
    ref = ref[ref.obs["coarse_type"] != "ambiguous"].copy()
    # cap per type
    rng = np.random.default_rng(cfg["seed"]); keep = []
    for t, idx in ref.obs.groupby("coarse_type").indices.items():
        if len(idx) < dc["reference_min_cells_per_type"]:
            log(f"  dropping {t}: only {len(idx)} cells"); continue
        keep.append(rng.choice(idx, min(len(idx), dc["reference_max_cells_per_type"]), replace=False))
    ref = ref[np.sort(np.concatenate(keep))].copy()
    ref.write_h5ad(out_dir / "reference.h5ad")
    scipy.io.mmwrite(out_dir / "reference_counts.mtx", sp.csr_matrix(ref.X).T.tocoo())  # genes × cells for R
    pd.Series(ref.var_names).to_csv(out_dir / "reference_genes.tsv", index=False, header=False)
    ref.obs[["coarse_type", "sample", "dataset"]].to_csv(out_dir / "reference_cell_types.tsv", sep="\t")
    sd = stage_dir(STAGE)
    comp = ref.obs.groupby(["coarse_type", "dataset"]).size().unstack(fill_value=0); comp["total"] = comp.sum(1)
    write_table(comp.reset_index(), sd / "reference_composition.tsv")
    write_table(pd.DataFrame(os_ref.uns["cluster_annotation"]), sd / "reference_cluster_annotation.tsv")
    # marker table: mean lognorm per type for the marker genes
    tmp = ref.copy(); sc.pp.normalize_total(tmp, target_sum=1e4); sc.pp.log1p(tmp)
    genes = [g for gs in MARKERS.values() for g in gs if g in tmp.var_names] + [g for g in ["SFTPC", "AGER", "MARCO", "FABP4", "SCGB1A1"] if g in tmp.var_names]
    mk = pd.DataFrame(np.asarray(tmp[:, genes].X.todense()), columns=genes, index=tmp.obs_names).groupby(tmp.obs["coarse_type"].to_numpy()).mean().round(3)
    write_table(mk.reset_index().rename(columns={"index": "coarse_type"}), sd / "reference_markers.tsv")
    write_provenance(STAGE + "_reference", [ROOT / cfg["paths"]["raw_gse162454"], ROOT / cfg["paths"]["raw_hlca"]],
                     [out_dir / "reference.h5ad", sd / "reference_composition.tsv"], cfg,
                     extra={"n_cells": int(ref.n_obs), "n_genes": int(ref.n_vars), "types": comp["total"].to_dict()}, t_start=t0)


if __name__ == "__main__":
    main()
