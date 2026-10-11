"""v2 Figure 1 — study framework, cohort and region reference labels.
a  framework: cohort → dual-scale neighbourhood → encoder → patient-invariant objective → consensus niches → the two aims
b  example section (MR11B): H&E and the algorithmic region reference labels with the tumour–lung interface
The patient → section hierarchy of the cohort is Supplementary Figure S1.
Reads results/04b_panel_exports/ and figures/source_data/Fig1_a_spec.json.
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
schematic = import_module("v2_Fig1_schematic")  # the framework panel is drawn by its own module


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
    set_style()
    spec = json.loads((ROOT / "figures/source_data/Fig1_a_spec.json").read_text())
    sp = pd.read_csv(ROOT / "results/15_niches/spot_niches.tsv", sep="\t", usecols=["section", "x", "y", "niche", "unmatched"])
    maps = {s_: d.loc[~d["unmatched"].astype(bool)].sample(n=min(1500, int((~d["unmatched"].astype(bool)).sum())), random_state=0)
            for s_, d in sp.groupby("section") if s_ in {"PT3", "MR3", "MR14", "MR16", "LR17"}}
    fig = plt.figure(figsize=(7.2, 5.6))
    ax_a = fig.add_axes([0.018, 0.690, 0.964, 0.298]); schematic.draw(ax_a, spec, maps); panel_label(ax_a, "a")
    save_source(pd.concat(maps.values()).assign(panel="a_maps"), NAME, "a_maps")
    ax_b1 = fig.add_axes([0.018, 0.075, 0.462, 0.584]); ax_b2 = fig.add_axes([0.520, 0.075, 0.462, 0.584])  # full width, wide gutter
    panel_b(ax_b1, ax_b2, "MR11B"); panel_label(ax_b1, "b")
    fig.legend(*ax_b2.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.004), ncol=4,
               fontsize=6, markerscale=2.0, frameon=False, handletextpad=0.4, columnspacing=2.0)
    save_fig(fig, f"{NAME}_cohort_framework")


if __name__ == "__main__":
    main()
