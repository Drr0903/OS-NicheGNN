"""Stage 02b — turn the R export of the external cohort into per-section AnnData + manifest rows.

Reads  data/processed/external_export/<OS>/{counts.mtx,genes.tsv,barcodes.tsv,meta.tsv,coords.tsv,scalefactors.json,image.png}
Writes data/processed/external/<OS>.h5ad, results/02_external_export/external_section_qc.tsv,
       and fills the external rows of results/01_manifest/sample_manifest.tsv.

Patient mapping: the Zenodo record and the source paper give specimen-level identities only
(nine specimens from seven patients; seven lung-parenchyma specimens). No public specimen→patient
key exists, so each lung specimen is treated as its own patient-level unit and the manifest
records `patient_mapping = "unavailable_public"`; this is stated in the manuscript.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "02_external_export"


def load_export(d: Path, section: str) -> tuple[ad.AnnData, dict]:
    X = scipy.io.mmread(d / "counts.mtx").T.tocsr().astype(np.float32)  # spots × genes
    genes = [g.strip() for g in open(d / "genes.tsv")]
    bcs = [b.strip() for b in open(d / "barcodes.tsv")]
    meta = pd.read_csv(d / "meta.tsv", sep="\t").set_index("barcode").loc[bcs]
    co = pd.read_csv(d / "coords.tsv", sep="\t").set_index("barcode").loc[bcs]
    sf = json.load(open(d / "scalefactors.json"))
    a = ad.AnnData(X=X, obs=meta, var=pd.DataFrame(index=genes))
    a.var_names_make_unique()
    a.obs["section"] = section
    # coordinates are stored at the scale of the carried image (Seurat imagerow/imagecol)
    a.obs["img_row"] = co["imagerow"].to_numpy(float)
    a.obs["img_col"] = co["imagecol"].to_numpy(float)
    a.obsm["spatial"] = a.obs[["img_col", "img_row"]].to_numpy(float)
    a.uns["scalefactors"] = sf
    a.uns["files"] = {"image": "image.png"}
    a.obs["n_counts"] = np.asarray(a.X.sum(1)).ravel()
    a.obs["n_genes"] = np.asarray((a.X > 0).sum(1)).ravel()
    mt = a.var_names.str.startswith("MT-")
    a.obs["pct_mito"] = np.asarray(a[:, mt].X.sum(1)).ravel() / np.maximum(a.obs["n_counts"], 1) * 100
    # spot pitch (image px) = median nearest-neighbour distance
    tree = cKDTree(a.obsm["spatial"])
    dd, _ = tree.query(a.obsm["spatial"], k=2)
    pitch = float(np.median(dd[:, 1]))
    a.uns["spot_pitch_img_px"] = pitch
    qc = {"section": section, "n_spots": int(a.n_obs), "n_genes_in_object": int(a.n_vars), "n_genes_detected": int((a.X.sum(0) > 0).sum()),
          "median_counts": float(np.median(a.obs["n_counts"])), "median_genes": float(np.median(a.obs["n_genes"])),
          "median_pct_mito": float(np.median(a.obs["pct_mito"])), "counts_integer": bool(sf.get("counts_integer", False)),
          "median_contamination_rate": float(np.median(a.obs["contamination_rate"])) if "contamination_rate" in a.obs else np.nan,
          "spot_pitch_img_px": pitch, "image_rows": sf["image_dim"][0], "image_cols": sf["image_dim"][1], "scale_hires": sf["hires"],
          "img_px_per_spot_pitch": pitch}
    return a, qc


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    set_global_seed(cfg["seed"])
    exp_root = ROOT / cfg["paths"]["processed"] / "external_export"
    out_dir = ROOT / cfg["paths"]["processed"] / "external"
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = cfg["cohorts"]["external"]
    qcs, outputs = [], []
    for s in ext["lung_sections"] + ext["lymph_node_sections"]:
        d = exp_root / s
        if not (d / "counts.mtx").exists():
            log(f"{s}: export missing, skipped")
            continue
        a, qc = load_export(d, s)
        a.write_h5ad(out_dir / f"{s}.h5ad")
        outputs.append(out_dir / f"{s}.h5ad")
        qcs.append(qc)
        log(f"{s}: {a.n_obs} spots × {a.n_vars} genes, median counts {qc['median_counts']:.0f}, pitch {qc['spot_pitch_img_px']:.1f}px, image {qc['image_rows']}×{qc['image_cols']}")
    qc_df = pd.DataFrame(qcs)
    sd = stage_dir(STAGE)
    write_table(qc_df, sd / "external_section_qc.tsv")
    # fill manifest rows
    man_path = ROOT / "results/01_manifest/sample_manifest.tsv"
    man = pd.read_csv(man_path, sep="\t", dtype=str)
    for _, q in qc_df.iterrows():
        m = man["section_id"] == q["section"]
        man.loc[m, "patient_id"] = f"EXT_{q['section']}"
        man.loc[m, "rna_assay"] = "SpotClean_decontaminated_counts_nonint"
        man.loc[m, "coordinate_file"] = "coords.tsv (Seurat imagerow/imagecol)"
        man.loc[m, "image_file"] = f"image.png ({q['image_rows']}x{q['image_cols']})"
        man.loc[m, "image_scale_hires"] = str(q["scale_hires"])
        man.loc[m, "n_spots_geo_h5"] = str(q["n_spots"])
    man["patient_mapping"] = np.where(man["cohort"] == "external", "unavailable_public (specimen treated as unit)", "Supplementary Table 1 of source paper")
    write_table(man, man_path)
    write_provenance(STAGE, [exp_root], outputs + [sd / "external_section_qc.tsv", man_path], cfg,
                     extra={"n_external_sections": len(qcs)}, t_start=t0)


if __name__ == "__main__":
    main()
