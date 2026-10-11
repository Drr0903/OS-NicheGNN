"""Stage 01 — sample manifest + load GSE293065 Visium sections into per-section AnnData.

Outputs
  results/01_manifest/sample_manifest.tsv     one row per section (both cohorts; external rows filled by stage 02)
  results/01_manifest/gse293065_section_qc.tsv
  data/processed/gse293065/<section>.h5ad     raw counts, obs = array/pixel coords, var = gene ids
"""
from __future__ import annotations

import glob
import gzip
import json
import re
import sys
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "01_manifest"


def read_10x_h5(path: str) -> ad.AnnData:
    with h5py.File(path) as f:
        g = f["matrix"]
        shape = tuple(g["shape"][:])  # (n_genes, n_barcodes)
        X = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape).T.tocsr()
        barcodes = g["barcodes"][:].astype(str)
        feats = g["features"]
        var = pd.DataFrame(
            {
                "gene_id": feats["id"][:].astype(str),
                "feature_type": feats["feature_type"][:].astype(str),
                "genome": feats["genome"][:].astype(str),
            },
            index=feats["name"][:].astype(str),
        )
    a = ad.AnnData(X=X.astype(np.float32), obs=pd.DataFrame(index=barcodes), var=var)
    a.var_names_make_unique()
    return a


def read_positions(path: str) -> pd.DataFrame:
    cols = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    if path.endswith("positions_list.csv.gz"):
        df = pd.read_csv(path, header=None, names=cols)
    else:
        df = pd.read_csv(path)
        df.columns = cols
    return df.set_index("barcode")


def canonical_symbols(raw_dir: Path) -> dict[str, str]:
    """gene_id → most frequent gene symbol across all sections. Sections processed with different probe-set /
    Space Ranger versions carry different HGNC symbols for the same Ensembl id (e.g. CYR61 vs CCN1); aligning on
    gene_id makes every section's var_names identical."""
    from collections import Counter, defaultdict
    votes: dict[str, Counter] = defaultdict(Counter)
    for h5 in glob.glob(str(raw_dir / "*_filtered_feature_bc_matrix.h5")):
        with h5py.File(h5) as f:
            ids = f["matrix/features/id"][:].astype(str); names = f["matrix/features/name"][:].astype(str)
        for i, n in zip(ids, names):
            votes[i][n] += 1
    return {i: c.most_common(1)[0][0] for i, c in votes.items()}


def load_section(raw_dir: Path, section: str, canon: dict[str, str] | None = None) -> tuple[ad.AnnData, dict]:
    h5 = glob.glob(str(raw_dir / f"*_{section}_filtered_feature_bc_matrix.h5"))[0]
    gsm = Path(h5).name.split("_")[0]
    pos_file = glob.glob(str(raw_dir / f"*_{section}_tissue_positions*.csv.gz"))[0]
    sf_file = glob.glob(str(raw_dir / f"*_{section}_scalefactors_json.json.gz"))[0]
    img_file = glob.glob(str(raw_dir / f"*_{section}_tissue_hires_image.png.gz"))[0]

    a = read_10x_h5(h5)
    if canon is not None:
        a.var["symbol_original"] = a.var_names
        a.var_names = pd.Index([canon.get(g, n) for g, n in zip(a.var["gene_id"], a.var_names)])
        a.var_names_make_unique()
    pos = read_positions(pos_file)
    pos = pos.loc[pos.index.intersection(a.obs_names)]
    a = a[pos.index].copy()
    a.obs = a.obs.join(pos)
    sf = json.load(gzip.open(sf_file))
    a.uns["scalefactors"] = sf
    a.uns["files"] = {"h5": Path(h5).name, "positions": Path(pos_file).name, "hires_image": Path(img_file).name}
    a.obsm["spatial"] = a.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=float)
    a.obs["section"] = section
    a.obs["gsm"] = gsm
    a.obs["n_counts"] = np.asarray(a.X.sum(1)).ravel()
    a.obs["n_genes"] = np.asarray((a.X > 0).sum(1)).ravel()
    mt = a.var_names.str.startswith("MT-")
    a.obs["pct_mito"] = np.asarray(a[:, mt].X.sum(1)).ravel() / np.maximum(a.obs["n_counts"], 1) * 100

    # spot pitch in full-res pixels from array-adjacent spots (used later to convert px → µm: pitch = 100 µm)
    r, c = a.obs["array_row"].to_numpy(), a.obs["array_col"].to_numpy()
    xy = a.obsm["spatial"]
    key = {(int(rr), int(cc)): i for i, (rr, cc) in enumerate(zip(r, c))}
    d = [np.linalg.norm(xy[i] - xy[key[(int(rr), int(cc) + 2)]]) for i, (rr, cc) in enumerate(zip(r, c)) if (int(rr), int(cc) + 2) in key]
    pitch = float(np.median(d)) if d else float("nan")
    a.uns["spot_pitch_fullres_px"] = pitch
    qc = {
        "section": section, "gsm": gsm, "n_spots_in_tissue": int(a.n_obs), "n_genes_detected": int((a.X.sum(0) > 0).sum()),
        "median_counts": float(np.median(a.obs["n_counts"])), "median_genes": float(np.median(a.obs["n_genes"])),
        "median_pct_mito": float(np.median(a.obs["pct_mito"])), "spot_pitch_fullres_px": pitch,
        "um_per_fullres_px": 100.0 / pitch if pitch else np.nan, "tissue_hires_scalef": sf["tissue_hires_scalef"],
        "spot_diameter_fullres": sf["spot_diameter_fullres"],
    }
    return a, qc


def build_manifest(cfg: dict, qc_df: pd.DataFrame) -> pd.DataFrame:
    clin = pd.read_csv(ROOT / cfg["paths"]["clinical_table"], sep="\t", dtype=str)
    sr = pd.read_csv(ROOT / cfg["paths"]["spaceranger_table"], sep="\t", dtype=str)
    disc = cfg["cohorts"]["discovery"]
    rows = []
    for _, r in clin.iterrows():
        s = r["sample"]
        organ = r["localization"].lower()
        role = (
            "discovery_lung" if s in disc["lung_sections"] else
            "paired_primary" if s in disc["paired_primary"] else
            "excluded_non_lung"
        )
        q = qc_df.set_index("section").loc[s] if s in qc_df["section"].values else None
        rows.append({
            "cohort": "discovery", "source": "GSE293065", "section_id": s, "specimen_id": s.rstrip("AB") if s in ("MR11A", "MR11B") else s,
            "patient_id": disc["patient_groups"].get(s, f"P{int(r['patient_id']):02d}"), "gsm": q["gsm"] if q is not None else "",
            "organ": organ, "role": role, "timepoint": r["timepoint_detail"], "time_from_diagnosis": r["time_from_diagnosis"],
            "treatment_before_relapse": r["drugs_before_relapse"], "sample_type": r["sample_type"], "sex": r["sex"], "age_band": r["age_band"],
            "survival": r["survival"], "rna_assay": "raw_counts_10x_h5", "coordinate_file": "tissue_positions.csv.gz",
            "image_file": "tissue_hires_image.png.gz", "image_scale_hires": q["tissue_hires_scalef"] if q is not None else np.nan,
            "n_spots_geo_h5": int(q["n_spots_in_tissue"]) if q is not None else np.nan,
            "n_spots_paper_table": sr.set_index("sample").loc[s, "n_spots"] if s in sr["sample"].values else "",
            "boundary_eligible": "TBD_stage04", "exclusion_reason": "" if role != "excluded_non_lung" else f"non-lung site ({organ})",
        })
    for s in cfg["cohorts"]["external"]["lung_sections"] + cfg["cohorts"]["external"]["lymph_node_sections"]:
        rows.append({"cohort": "external", "source": "Zenodo_14594702", "section_id": s, "specimen_id": s, "patient_id": "TBD_stage02",
                     "organ": "lung" if s in cfg["cohorts"]["external"]["lung_sections"] else "hilar lymph node",
                     "role": "external_lung" if s in cfg["cohorts"]["external"]["lung_sections"] else "external_lymph_node",
                     "rna_assay": "TBD_stage02", "boundary_eligible": "TBD_stage04",
                     "exclusion_reason": "" if s in cfg["cohorts"]["external"]["lung_sections"] else "lymph node, not lung parenchyma"})
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    set_global_seed(cfg["seed"])
    raw = ROOT / cfg["paths"]["raw_gse293065"]
    out_dir = ROOT / cfg["paths"]["processed"] / "gse293065"
    out_dir.mkdir(parents=True, exist_ok=True)
    sections = sorted({re.sub(r"^GSM\d+_", "", Path(p).name).replace("_filtered_feature_bc_matrix.h5", "") for p in glob.glob(str(raw / "*_filtered_feature_bc_matrix.h5"))})
    log(f"{len(sections)} sections: {sections}")
    canon = canonical_symbols(raw)
    qcs, outputs = [], []
    for s in sections:
        a, qc = load_section(raw, s, canon)
        qc["n_symbols_renamed"] = int((a.var["symbol_original"] != a.var_names).sum())
        a.write_h5ad(out_dir / f"{s}.h5ad")
        outputs.append(out_dir / f"{s}.h5ad")
        qcs.append(qc)
        log(f"{s}: {a.n_obs} spots × {a.n_vars} genes, median counts {qc['median_counts']:.0f}, pitch {qc['spot_pitch_fullres_px']:.1f}px")
    qc_df = pd.DataFrame(qcs)
    sd = stage_dir(STAGE)
    write_table(qc_df, sd / "gse293065_section_qc.tsv")
    man = build_manifest(cfg, qc_df)
    write_table(man, sd / "sample_manifest.tsv")
    summary = {
        "n_sections_total": len(sections), "n_discovery_lung_sections": int((man["role"] == "discovery_lung").sum()),
        "n_discovery_lung_patients": int(man.loc[man["role"] == "discovery_lung", "patient_id"].nunique()),
        "n_paired_primary": int((man["role"] == "paired_primary").sum()),
        "n_genes": int(a.n_vars), "spots_geo_vs_paper_max_abs_diff": int((pd.to_numeric(man["n_spots_geo_h5"], errors="coerce") - pd.to_numeric(man["n_spots_paper_table"], errors="coerce")).abs().max()),
    }
    json.dump(summary, open(sd / "summary.json", "w"), indent=2)
    log(f"summary: {summary}")
    write_provenance(STAGE, [raw / "GSE293065_RAW.tar" if (raw / "GSE293065_RAW.tar").exists() else raw, ROOT / cfg["paths"]["clinical_table"]],
                     outputs + [sd / "sample_manifest.tsv", sd / "gse293065_section_qc.tsv"], cfg, extra=summary, t_start=t0)


if __name__ == "__main__":
    main()
