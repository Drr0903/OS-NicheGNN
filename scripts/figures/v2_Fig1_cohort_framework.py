"""v2 Figure 1 — study framework, cohort and region reference labels.
a  the network: inputs, masking, neighbour tokens, spot encoder, cross-attention, latent with linear skip, decoder and
   linear neighbourhood heads with the two loss terms, and the use of the latent (atlas, new patients, applications)
b  example section (MR11B): H&E and the algorithmic region reference labels with the tumour–lung interface
The patient → section hierarchy of the cohort is Supplementary Figure S1.
Reads results/04b_panel_exports/.
"""
from __future__ import annotations

import json

import sys
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.plotting import REGION_COL, REGION_NAME, panel_label, save_fig, save_source, set_style  # noqa: E402
from lib.repro import ROOT  # noqa: E402

NAME = "Fig1"
arch = import_module("v2_Fig1_architecture")  # the network panel is drawn by its own module


def panel_b(ax_img, ax_lab, section: str = "MR11B"):
    """H&E and the algorithmic region labels, both read from the stage-04b export (no raw data needed)"""
    sd = ROOT / "results/04b_panel_exports"
    img = np.asarray(Image.open(sd / f"{section}_hires_crop.png"))
    d = pd.read_csv(sd / f"{section}_spots.tsv", sep="\t")
    ax_img.imshow(img); ax_img.axis("off"); ax_img.set_title(f"{section} — H&E", fontsize=7)
    ax_lab.imshow(img, alpha=0.25); ax_lab.axis("off"); ax_lab.set_title("region reference + interface", fontsize=7)
    for k, c in REGION_COL.items():
        m = (d["label"] == k).to_numpy()
        if m.any():
            ax_lab.scatter(d.loc[m, "x"], d.loc[m, "y"], s=3, c=c, linewidths=0, label=REGION_NAME.get(k, k.replace("_", " ")))
    ifc = d["interface"].fillna(False).to_numpy(bool)
    ax_lab.scatter(d.loc[ifc, "x"], d.loc[ifc, "y"], s=6, facecolors="none", edgecolors="black", linewidths=0.4, label="tumour–lung interface")
    save_source(d, NAME, "b")


def main() -> None:
    set_style(); FW, FH = 7.2, 7.25
    fig = plt.figure(figsize=(FW, FH))
    ax_a = fig.add_axes([0, 1 - (arch.H + 0.05) / FH, arch.W / FW, arch.H / FH]); arch.draw(ax_a); arch.mark(ax_a)
    ax_c = fig.add_axes([0, 1 - (arch.H + 0.07 + arch.SH) / FH, arch.W / FW, arch.SH / FH]); arch.contributions(ax_c)
    fig.text(0.006, 1 - 0.02 / FH, "a", fontsize=9, fontweight="bold", va="top")
    hb = 2.62; yb = 0.36 / FH
    ax_b1 = fig.add_axes([0.03, yb, 0.455, hb / FH]); ax_b2 = fig.add_axes([0.525, yb, 0.455, hb / FH])
    panel_b(ax_b1, ax_b2, "MR11B"); fig.text(0.006, yb + (hb + 0.27) / FH, "b", fontsize=9, fontweight="bold", va="top")
    fig.legend(*ax_b2.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.004), ncol=4,
               fontsize=6, markerscale=2.0, frameon=False, handletextpad=0.4, columnspacing=2.0)
    save_fig(fig, f"{NAME}_cohort_framework")


if __name__ == "__main__":
    main()
