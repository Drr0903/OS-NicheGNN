"""Graphical abstract — a simple four-step summary: cohort → OS-NicheGNN → niche atlas → osteosarcoma findings.

Shapes are drawn in inches (W × H = 1328 × 531 px at 200 dpi); the two tissue maps are real spots of section MR11B
(region reference labels, and atlas niches). Reads results/15_niches/{spot_niches.tsv, meta.json}.
Writes figures/Graphical_abstract.{png,pdf}.
"""
from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.plotting import REGION_COL, set_style  # noqa: E402
from lib.repro import ROOT  # noqa: E402

A = import_module("v2_Fig1_architecture")
W, H = 6.64, 2.656
NICHE = A.NICHE


def tissue(ax, d, col, x0, y0, w, h, s=0.9):
    px, py = d["x"].to_numpy(), -d["y"].to_numpy(); sx = (px - px.min()) / np.ptp(px); sy = (py - py.min()) / np.ptp(py)
    ar = np.ptp(px) / np.ptp(py); hh = min(h, w / ar); ww = hh * ar
    ax.scatter(x0 + (w - ww) / 2 + sx * ww, y0 + (h - hh) / 2 + sy * hh, s=s, c=col, linewidths=0, zorder=3)


def main() -> None:
    set_style(); fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    sp = pd.read_csv(ROOT / "results/15_niches/spot_niches.tsv", sep="\t"); meta = json.load(open(ROOT / "results/15_niches/meta.json"))
    d = sp[(sp["section"] == "MR11B") & ~sp["unmatched"].astype(bool)]
    t = lambda x, y, s, fs=7.0, **k: A.text(ax, x, y, s, fs=fs, **k)  # noqa: E731
    cols = [(0.08, 1.42, "#eef3f7", "Osteosarcoma cohort"), (1.70, 1.42, "#eceffa", "OS-NicheGNN"), (3.32, 1.42, "#eaf3ec", "Niche atlas"), (4.94, 1.62, "#fbeeea", "Findings")]
    for x, w, fc, title in cols:
        A.box(ax, x, 0.10, w, H - 0.22, fc, r=0.08); t(x + w / 2, H - 0.30, title, fs=8.2, fontweight="bold")
    for (x, w, *_), (x2, *_) in zip(cols[:-1], cols[1:]):
        A.arrow(ax, x + w + 0.02, H / 2 - 0.05, x2 - 0.02, H / 2 - 0.05, "#5f6f7a", lw=1.4)
    # 1 cohort
    tissue(ax, d, [REGION_COL.get(v, "#bbbbbb") for v in d["histology_label"]], 0.22, 0.95, 1.14, 1.12)
    t(0.79, 0.68, "19 patients, 26 sections", fs=6.6); t(0.79, 0.48, "bone, lung, lymph node,", fs=6.0); t(0.79, 0.33, "heart, soft tissue", fs=6.0)
    # 2 network
    A.lattice(ax, 2.18, 1.62, 0.072)
    for ang in (20, 75, 140, 200, 260, 320):
        r = 0.36; ax.plot([2.18, 2.18 + r * np.cos(np.deg2rad(ang))], [1.62, 1.62 + r * np.sin(np.deg2rad(ang))], color=A.C["att"], lw=0.8 + (ang % 3) * 0.3, alpha=0.8, zorder=4)
    A.vec(ax, 2.68, 1.30, 1, 0.12, 0.66, A.C["z"], horizontal=False); A.vec(ax, 2.68, 1.30, 8, 0.12, 0.66, A.C["z"], horizontal=False)
    t(2.74, 2.04, r"$\mathbf{z}_i$", fs=7.0, color=A.C["z"])
    t(2.41, 0.98, "expression + H&E", fs=6.4); t(2.41, 0.78, "attention over rings 1–3", fs=6.4)
    t(2.41, 0.50, "reconstructs the spot", fs=6.0); t(2.41, 0.35, "and its surroundings", fs=6.0)
    # 3 atlas
    tissue(ax, d, [NICHE[int(v)] for v in d["niche"]], 3.46, 0.95, 1.14, 1.12)
    t(4.03, 0.68, "eight niches, six shared", fs=6.6); t(4.03, 0.48, "new patients mapped onto", fs=6.0); t(4.03, 0.33, "the fixed atlas (98 %)", fs=6.0)
    # 4 findings
    fx = 5.02; fw = 1.46
    A.box(ax, fx, 1.26, fw, 0.86, "white", ec="#c0564f", r=0.06)
    ax.add_patch(plt.Circle((fx + 0.26, 1.70), 0.15, fc="#7b2cbf", ec=NICHE[3], lw=3.0, zorder=4))
    for k, ang in enumerate(np.linspace(0, 2 * np.pi, 9)[:-1]):
        ax.add_patch(plt.Circle((fx + 0.26 + 0.27 * np.cos(ang), 1.70 + 0.27 * np.sin(ang)), 0.025 if k % 3 else 0.04, fc="#2a9d8f", lw=0, zorder=4))
    t(fx + 1.02, 1.94, "stromal band", fs=6.4, fontweight="bold"); t(fx + 1.02, 1.76, "covers more of", fs=5.8)
    t(fx + 1.02, 1.62, "the interface:", fs=5.8); t(fx + 1.02, 1.48, "fewer T cells", fs=5.8); t(fx + 1.02, 1.34, "beyond it", fs=5.8)
    A.box(ax, fx, 0.22, fw, 0.92, "white", ec="#c0564f", r=0.06)
    t(fx + fw / 2, 0.98, "diagnosis → relapse", fs=6.4, fontweight="bold")
    for k, (c, lab) in enumerate([(NICHE[7], "matrix-forming"), (NICHE[6], "boundary state")]):
        xx = fx + 0.14 + k * 0.72; A.box(ax, xx, 0.52, 0.46, 0.26, c, ec="white", r=0.04); t(xx + 0.23, 0.40, lab, fs=5.6)
    A.arrow(ax, fx + 0.64, 0.65, fx + 0.84, 0.65, "#5f6f7a", lw=1.2)
    t(fx + 0.37, 0.65, "↓", fs=8, color="white", fontweight="bold"); t(fx + 1.09, 0.65, "↑", fs=8, color="white", fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"figures/Graphical_abstract.{ext}", dpi=200 if ext == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
