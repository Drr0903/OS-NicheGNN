"""Graphical abstract — four steps: cohort → OS-NicheGNN → niche atlas → findings, in the style of Figure 1.

Drawn in millimetres (168.6 × 67.5 mm, the 1328 × 531 px proportion); illustrated icons from figures/source_data/Fig1_icons,
tissue maps are real spots of section MR2 (atlas niches and router weights, results/35_representative_interface).
Writes figures/Graphical_abstract.{png,pdf}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig1_draw as D  # noqa: E402
from fig1_draw import GREY, arrow, box, icon, txt  # noqa: E402
from lib.plotting import PALETTE, set_style  # noqa: E402
from lib.repro import ROOT  # noqa: E402
from v2_Fig1_overview import C  # noqa: E402

W, H = 168.6, 67.5


def spots(ax, img, d, col, x0, y0, w, colors=None):
    h = w * img.shape[0] / img.shape[1]
    ax.add_patch(Rectangle((x0, y0), w, h, fc="white", ec="#c9d1d8", lw=0.5, zorder=4)); ax.imshow(img, extent=[x0, x0 + w, y0, y0 + h], alpha=0.2, zorder=5)
    c = d[col].map(colors) if colors else d[col]
    ax.scatter(x0 + d["x"] / img.shape[1] * w, y0 + h - d["y"] / img.shape[0] * h, s=0.35, c=c, cmap="coolwarm_r", vmin=0, vmax=1, linewidths=0, zorder=6)
    return h


def main() -> None:
    set_style(); fig = plt.figure(figsize=(W / 25.4, H / 25.4)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    sd = ROOT / "results/35_representative_interface"; img = np.asarray(Image.open(sd / "MR2_hires_crop.png").convert("RGB"))
    d = pd.read_csv(sd / "MR2_spots.tsv", sep="\t"); d = d[~d["unmatched"].fillna(True).astype(bool)].copy(); d["niche_c"] = d["niche"].astype(int)
    K = json.load(open(ROOT / "results/15_niches/meta.json"))["K"]; cols = {k: PALETTE[k % len(PALETTE)] for k in range(K)}
    panels = [(2, 36, C["fa"], "Osteosarcoma cohort"), (41, 44, C["fb"], "OS-NicheGNN"), (88.5, 34, "#eaf3ec", "Niche atlas"), (126, 40.6, C["fd"], "Findings")]
    fr = []
    for x, w, fc, title in panels:
        fr.append(box(ax, x, 2, x + w, H - 2, fc, "none", lw=0, r=2.0, z=1)); txt(ax, x + w / 2, H - 6.5, title, fr[-1], fs=7.6, fw="bold")
    for (x, w, *_), (x2, *_) in zip(panels[:-1], panels[1:]):
        arrow(ax, (x + w + 0.3, H / 2 - 2), (x2 - 0.3, H / 2 - 2), col="#5f6f7a", lw=1.2)
    # 1 cohort
    icon(ax, "femur", 9.5, 42, h=15); arrow(ax, (14.5, 42), (19.5, 42), lw=0.8); icon(ax, "lungs", 27.5, 42, h=14, w=14)
    icon(ax, "lymph_node", 9, 23.5, h=9); icon(ax, "heart", 20, 23.5, h=9); icon(ax, "soft_tissue", 30.5, 23.5, h=8.5, w=9.5)
    txt(ax, 20, 12.5, "19 patients, 26 Visium\nsections, five sites", fr[0], fs=6.0); txt(ax, 20, 6.5, "diagnosis → relapse", fr[0], fs=5.6, col=GREY, style="italic")
    # 2 network
    ya, yt = 24, 46
    D.trapezoid(ax, 45, 57, yt, 12, 6.5, C["ring"], "ring-\nmean", fs=5.4); D.trapezoid(ax, 45, 57, ya, 12, 6.5, C["att"], "graph\nattention", fs=5.4)
    rb = box(ax, 61, 30, 70, 40, "white", C["router"], lw=0.8, r=1.0, z=4); txt(ax, 65.5, 37.5, "router", rb, fs=5.4, fw="bold", col=C["router"])
    for k, (v, c) in enumerate([(0.3, C["host"]), (0.85, C["tumour"])]):
        ax.add_patch(Rectangle((63 + k * 3.4, 31.2), 2.4, 4.6 * v, fc=c, ec="none", zorder=6))
    arrow(ax, (57.3, yt), (60.8, 38.5), lw=0.7); arrow(ax, (57.3, ya), (60.8, 31.5), lw=0.7)
    for k, (c, lab) in enumerate([(C["host"], "expert 1"), (C["tumour"], "expert 2")]):
        e = box(ax, 72.5, 37 - k * 7.5, 82, 42 - k * 7.5, c, c, r=0.8, z=5); txt(ax, 77.25, 39.5 - k * 7.5, lab, e, fs=5.0, col="white", fw="bold")
    arrow(ax, (70.2, 35), (72.3, 35), lw=0.7)
    txt(ax, 63, 12.5, "neighbourhood composition and\nindividual neighbours, fused by a\nmixture of experts", fr[1], fs=5.6)
    # 3 atlas
    tw = 14.5; h = spots(ax, img, d, "niche_c", 91, 28, tw, cols); spots(ax, img, d, "host_expert_weight", 107.5, 28, tw)
    txt(ax, 91 + tw / 2, 28 + h + 2.2, "8 niches", fr[2], fs=5.6); txt(ax, 107.5 + tw / 2, 28 + h + 2.2, "router", fr[2], fs=5.6)
    txt(ax, 105.5, 18.5, "shared across sites;", fr[2], fs=5.8); txt(ax, 105.5, 14.0, "new patients mapped:", fr[2], fs=5.8)
    txt(ax, 105.5, 9.0, "97 % of spots assigned", fr[2], fs=5.8, fw="bold")
    txt(ax, 105.5, 4.6, "tumour vs host tissue routed", fr[2], fs=5.0, col=GREY)
    # 4 findings
    c1 = box(ax, 128.5, 34.5, 164.5, 58.5, "white", "#c8b88f", lw=0.7, r=1.2, z=3)
    icon(ax, "t_cell", 136, 46.5, h=10)
    txt(ax, 152.5, 50.5, "At the interface:", c1, fs=5.6, fw="bold"); txt(ax, 152.5, 44.8, "T cells rise across\nthe stromal band", c1, fs=5.4)
    txt(ax, 152.5, 38.5, "router switches experts", c1, fs=5.0, col=GREY)
    c2 = box(ax, 128.5, 5, 164.5, 31.5, "white", "#c8b88f", lw=0.7, r=1.2, z=3)
    icon(ax, "tumour_cell", 136, 23.5, h=9); arrow(ax, (140.6, 23.5), (143.2, 23.5), lw=0.7); icon(ax, "osteoclast", 148.5, 23.5, h=7.5, w=9.5)
    txt(ax, 146.5, 13.8, "Relapse: matrix-forming →\nboundary tumour niche", c2, fs=5.4, fw="bold"); txt(ax, 146.5, 8.0, "router shifts to host expert", c2, fs=5.0, col=GREY)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"figures/Graphical_abstract.{ext}", dpi=400 if ext == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
