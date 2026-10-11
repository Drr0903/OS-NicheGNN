"""Stage 04b — export the image panels of Figure 1 and Supplementary Figure S1 as plotting inputs.

The H&E panels are the only ones that would otherwise need the gitignored data/ tree to be redrawn. This stage
reads it once and writes, per panel section, the cropped hires image and a spot table already in cropped-image
coordinates, plus one cohort-wide table of the label features behind Supplementary Figure S1c, d. The figure
scripts then read results/ only, as the reproducibility contract requires.

Inputs : data/processed/{qc/<sec>.h5ad, qc_images/<sec>_hires.png, histology/<sec>_labels.tsv}
Outputs: results/04b_panel_exports/{<sec>_hires_crop.png, <sec>_spots.tsv, label_features.tsv, PROVENANCE.json}
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "04b_panel_exports"
PANELS = ["MR11B", "PT2", "LR17"]  # Figure 1c uses MR11B; Supplementary Figure S1a uses all three
FEATURES = ["section", "label", "cnv_signal", "cnv_corr", "score_lung", "img_tissue_frac"]


def export_section(cfg: dict, sec: str, sd: Path) -> pd.DataFrame:
    proc = ROOT / cfg["paths"]["processed"]
    img = np.asarray(Image.open(proc / "qc_images" / f"{sec}_hires.png").convert("RGB"))
    lab = pd.read_csv(proc / "histology" / f"{sec}_labels.tsv", sep="\t").set_index("barcode")
    a = ad.read_h5ad(proc / "qc" / f"{sec}.h5ad", backed="r")
    xy = a.obsm["spatial"] * a.uns["scalefactors"]["tissue_hires_scalef"]
    lab = lab.reindex(a.obs_names)
    ys, xs = np.where(img.mean(-1) < 225)  # crop to the tissue, the same rule the v1 panel used
    pad = 30
    r0, r1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad, img.shape[0])
    c0, c1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad, img.shape[1])
    Image.fromarray(img[r0:r1, c0:c1]).save(sd / f"{sec}_hires_crop.png")
    out = pd.DataFrame({"barcode": list(a.obs_names), "x": xy[:, 0] - c0, "y": xy[:, 1] - r0,
                        "label": lab["label"].to_numpy(),
                        "interface": lab["tumor_lung_interface"].fillna(False).to_numpy(bool)})
    write_table(out, sd / f"{sec}_spots.tsv")
    log(f"  {sec}: {len(out)} spots, crop {c1 - c0} x {r1 - r0} px")
    return out


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE)
    proc = ROOT / cfg["paths"]["processed"]; secs = cfg["v2"]["sections"]
    ins: list[Path] = []
    for sec in PANELS:
        export_section(cfg, sec, sd)
        ins += [proc / "qc" / f"{sec}.h5ad", proc / "qc_images" / f"{sec}_hires.png"]
    lung = {s for s, site in cfg["v2"]["site_of"].items() if site == "lung"}
    feats = []
    for s in secs:
        d = pd.read_csv(proc / "histology" / f"{s}_labels.tsv", sep="\t")
        d["section"] = s
        if s not in lung:  # the lung-marker class has no meaning outside lung sections
            d["label"] = d["label"].replace({"lung_parenchyma": "paucicellular"})
        feats.append(d[[c for c in FEATURES if c in d.columns]])
        ins.append(proc / "histology" / f"{s}_labels.tsv")
    fe = pd.concat(feats, ignore_index=True); write_table(fe, sd / "label_features.tsv")
    log(f"label features: {len(fe)} spots over {fe['section'].nunique()} sections")
    outs = [sd / "label_features.tsv"] + [sd / f"{s}_spots.tsv" for s in PANELS] + [sd / f"{s}_hires_crop.png" for s in PANELS]
    write_provenance(STAGE, ins, outs, cfg, extra={"panel_sections": PANELS}, t_start=t0)


if __name__ == "__main__":
    main()
