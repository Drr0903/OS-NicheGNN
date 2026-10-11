"""Stage 27d (exploratory, requested) — the external osteosarcoma sections (OS1–OS9) with the original authors' Seurat
clusters as per-section labels, in the stage-27 benchmark format.

The Seurat clusters were computed by the authors on each section separately (expression only, no spatial information), so
they are compared only with per-section clusterings (K = number of Seurat clusters of the section); joint scores are not
meaningful for this cohort. H&E features: Phikon on the distributed image (stage-04c procedure; the image is low
resolution, about 5 px per spot pitch).
Outputs <benchmark_dir>/os_seurat/<section>.h5ad (+ _img.npy); results/27_benchmark/os_seurat_manifest.tsv
"""
from __future__ import annotations

import sys
import time
from importlib import import_module
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.generality import qc_and_lognorm  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "27_benchmark"; OUT = (ROOT / "data/benchmark/os_seurat"); s04c = import_module("04c_image_features")


def main() -> None:
    from transformers import ViTModel
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); ic = cfg["image"]; sd = stage_dir(STAGE); OUT.mkdir(parents=True, exist_ok=True)
    model = ViTModel.from_pretrained(ic["model"], revision=ic["revision"], add_pooling_layer=False).to(cfg["device"]).eval()
    proc = ROOT / cfg["paths"]["processed"]; rows = []
    for f in sorted((proc / "external").glob("OS*.h5ad")):
        s = f.stem; a = ad.read_h5ad(f); a.X = np.round(a.X) if not hasattr(a.X, "toarray") else a.X
        a = qc_and_lognorm(a); a.obs["patient"] = s; a.obs["truth"] = "c" + a.obs["seurat_clusters"].astype(str); a.obs["raw_label"] = a.obs["truth"]
        pitch = float(a.uns["spot_pitch_img_px"]); a.uns["scalefactors"] = {"spot_diameter_fullres": 0.55 * pitch, "tissue_hires_scalef": 1.0}
        img = np.asarray(Image.open(proc / "external_export" / s / "image.png").convert("RGB")); side = max(8, int(round(ic["patch_spots"] * 0.55 * pitch)))
        F = s04c.section_features(model, a, img, side, cfg["device"]).astype(np.float32)
        for c in [c for c in a.obs.columns if c not in ("patient", "truth", "raw_label")]:
            a.obs.pop(c)
        a.write_h5ad(OUT / f"{s}.h5ad"); np.save(OUT / f"{s}_img.npy", F)
        rows.append({"section": s, "n_spots": a.n_obs, "n_seurat_clusters": a.obs["truth"].nunique(), "patch_px": side}); log(rows[-1])
    write_table(pd.DataFrame(rows), sd / "os_seurat_manifest.tsv"); write_provenance(STAGE + "_os_seurat", [], [sd / "os_seurat_manifest.tsv"], cfg, t_start=t0)


if __name__ == "__main__":
    main()
