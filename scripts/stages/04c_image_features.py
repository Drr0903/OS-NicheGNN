"""Stage 04c (second set) — H&E morphology features per spot.

For every QC spot of the 26 sections, the hires H&E image is cropped around the spot centre (side = `patch_spots` spot
diameters), resized to 224 px and passed through Phikon (Filiot et al. 2023; ViT-B/16 self-supervised on 40 M TCGA
histology tiles, weights owkin/phikon); the CLS token (768-d) is the spot's morphology feature.
The image is never used to make the region reference labels (those come from expression, stage 04); it enters the
representation as a second modality, and the stage-04 image checks stay as reported.
Outputs data/processed/image_features/<sec>.npy (rows aligned to the QC h5ad), results/04c_image_features/{summary.tsv, PROVENANCE.json}
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "04c_image_features"
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def patches(img: np.ndarray, xy: np.ndarray, side: int) -> np.ndarray:
    pad = side; im = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=255); h = side // 2; out = []
    for x, y in np.round(xy).astype(int) + pad:
        p = Image.fromarray(im[y - h:y - h + side, x - h:x - h + side]).resize((224, 224), Image.BICUBIC)
        out.append((np.asarray(p, np.float32) / 255.0 - MEAN) / STD)
    return np.stack(out).transpose(0, 3, 1, 2)


@torch.no_grad()
def section_features(model, a: ad.AnnData, img: np.ndarray, side: int, device: str, bs: int = 256) -> np.ndarray:
    xy = a.obsm["spatial"] * a.uns["scalefactors"]["tissue_hires_scalef"]; feats = []
    for i in range(0, len(xy), bs):
        x = torch.tensor(patches(img, xy[i:i + bs], side), device=device)
        with torch.autocast("cuda", dtype=torch.float16):
            feats.append(model(pixel_values=x).last_hidden_state[:, 0].float().cpu().numpy())
    return np.vstack(feats)


def main():
    from transformers import ViTModel
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); ic = cfg["image"]; sd = stage_dir(STAGE)
    proc = ROOT / cfg["paths"]["processed"]; od = proc / "image_features"; od.mkdir(parents=True, exist_ok=True)
    model = ViTModel.from_pretrained(ic["model"], revision=ic["revision"], add_pooling_layer=False).to(cfg["device"]).eval()
    rows, ins, outs = [], [], []
    for sec in cfg["v2"]["sections"]:
        a = ad.read_h5ad(proc / "qc" / f"{sec}.h5ad"); ip = proc / "qc_images" / f"{sec}_hires.png"
        img = np.asarray(Image.open(ip).convert("RGB")); side = int(round(ic["patch_spots"] * a.uns["scalefactors"]["spot_diameter_fullres"] * a.uns["scalefactors"]["tissue_hires_scalef"]))
        F = section_features(model, a, img, side, cfg["device"]); np.save(od / f"{sec}.npy", F.astype(np.float32))
        rows.append({"section": sec, "n_spots": len(F), "patch_px": side, "feat_dim": F.shape[1]}); ins += [proc / "qc" / f"{sec}.h5ad", ip]; outs.append(od / f"{sec}.npy")
        log(f"{sec}: {len(F)} spots, patch {side}px")
    write_table(pd.DataFrame(rows), sd / "summary.tsv")
    write_provenance(STAGE, ins, outs + [sd / "summary.tsv"], cfg, extra={"model": ic["model"], "revision": ic["revision"]}, t_start=t0)


if __name__ == "__main__":
    main()
