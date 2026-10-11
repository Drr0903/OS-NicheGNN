"""Stage 27f () — human DLPFC (12 Visium sections, 3 donors) in the stage-27 benchmark format.

Counts and hires H&E of the spatialLIBD release, manual layer map (L1–L6, WM); unannotated spots are "unscored". Stage-03
QC rule; Phikon features of the hires image by the stage-04c procedure. Donor = patient unit.
Outputs <benchmark_dir>/dlpfc/<section>.h5ad (+ _img.npy); results/27_benchmark/dlpfc_manifest.tsv
"""
from __future__ import annotations

import json
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.generality import qc_and_lognorm  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "27_benchmark"; D = (ROOT / "data/dlpfc"); OUT = (ROOT / "data/benchmark/dlpfc")
DONOR = {**{s: "donor1" for s in ("151507", "151508", "151509", "151510")}, **{s: "donor2" for s in ("151669", "151670", "151671", "151672")},
         **{s: "donor3" for s in ("151673", "151674", "151675", "151676")}}
s04c = import_module("04c_image_features"); Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    from transformers import ViTModel
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); ic = cfg["image"]; sd = stage_dir(STAGE); OUT.mkdir(parents=True, exist_ok=True)
    model = ViTModel.from_pretrained(ic["model"], revision=ic["revision"], add_pooling_layer=False).to(cfg["device"]).eval()
    lay = pd.read_csv(D / "barcode_level_layer_map.tsv", sep="\t", header=None, names=["barcode", "section", "layer"], dtype=str); rows = []
    for s, donor in DONOR.items():
        a = sc.read_10x_h5(D / s / "filtered_feature_bc_matrix.h5"); a.var_names_make_unique()
        pos = pd.read_csv(D / s / "tissue_positions_list.txt", header=None, index_col=0); pos.columns = ["in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
        pos = pos.reindex(a.obs_names); a = a[pos["in_tissue"].fillna(0).to_numpy() == 1].copy(); pos = pos.reindex(a.obs_names)
        a.obsm["spatial"] = pos[["pxl_col", "pxl_row"]].to_numpy(float)
        l = lay[lay["section"] == s].set_index("barcode")["layer"]
        a.obs["raw_label"] = l.reindex(a.obs_names).fillna("unscored").to_numpy(); a.obs["truth"] = a.obs["raw_label"]; a.obs["patient"] = donor
        a = qc_and_lognorm(a); sf = json.load(open(D / s / "scalefactors_json.json")); a.uns["scalefactors"] = sf
        img = np.asarray(Image.open(D / s / "tissue_hires_image.png").convert("RGB")); side = int(round(ic["patch_spots"] * sf["spot_diameter_fullres"] * sf["tissue_hires_scalef"]))
        F = s04c.section_features(model, a, img, side, cfg["device"]).astype(np.float32)
        a.write_h5ad(OUT / f"{s}.h5ad"); np.save(OUT / f"{s}_img.npy", F)
        vc = a.obs["truth"].value_counts(); rows.append({"section": s, "donor": donor, "n_spots_qc": a.n_obs, "patch_px": side, **{f"n_{k}": int(v) for k, v in vc.items()}})
        log(rows[-1])
    write_table(pd.DataFrame(rows).fillna(0), sd / "dlpfc_manifest.tsv")
    write_provenance(STAGE + "_dlpfc_prep", [D / "CHECKSUMS.md5"], [sd / "dlpfc_manifest.tsv"], cfg, extra={"amendment": "A22"}, t_start=t0)


if __name__ == "__main__":
    main()
