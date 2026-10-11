"""v2 Figure 6 — remodelling between primary tumours and metastases .
a  niche fractions in the four matched primary → lung-metastasis pairs (one line per pair; the soft-tissue relapse pair dashed)
b  deconvolved composition change per matched pair (mean, pairs as points)
c  patient-level niche fractions of the patients contributing only a primary or only a metastasis (Cliff's delta, Mann–Whitney P)
d  remodelling in three dimensions: variables changing in the same direction in all four pairs, against the number expected
   by chance, and how many of them also point the same way in the unpaired patients
Reads results/16_remodelling/{paired_composition_change, unpaired_primary_vs_metastasis_nonoverlap}.tsv,
results/26_remodelling_dimensions/dimension_summary.tsv, results/15_niches/meta.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.plotting import PALETTE, bold_axes, panel_label, save_fig, save_source, set_style  # noqa: E402
from lib.repro import ROOT  # noqa: E402

NAME = "Fig6"


def paired(ax, pc, K, names, cols):
    nk = pc[pc["variable"].str.startswith("niche")].copy(); nk["niche"] = nk["variable"].str.extract(r"niche (\d+)").astype(int)
    pairs = list(nk["pair"].unique()); w = 0.8 / len(pairs)
    for j, pr in enumerate(pairs):
        d = nk[nk["pair"] == pr].sort_values("niche"); dashed = bool(d["sensitivity_pair"].iloc[0])
        for _, r in d.iterrows():
            x0 = r["niche"] + j * w - 0.4
            ax.plot([x0, x0 + w * 0.8], [r["primary"], r["metastasis"]], color=cols[r["niche"]], lw=0.7 if dashed else 1.1, ls="--" if dashed else "-", marker="o", ms=2.3, mfc="white")
    ax.set_xticks(range(K)); ax.set_xticklabels([f"{k}" for k in range(K)], fontsize=6)
    bold_axes(ax, x="niche (primary → metastasis per pair;\ndashed: soft-tissue relapse PT5 → LR5)", y="fraction of spots"); panel_label(ax, "a"); save_source(nk, NAME, "a")


def unpaired(ax, up, K, cols):
    for _, r in up.iterrows():
        k = int(r["niche"]); ax.bar(k - 0.18, r["primary_mean"], 0.34, color="white", edgecolor=cols[k], lw=1)
        ax.bar(k + 0.18, r["metastasis_mean"], 0.34, color=cols[k])
        if r["p_mannwhitney"] < 0.05:
            ax.text(k, max(r["primary_mean"], r["metastasis_mean"]) + 0.03, f"δ {r['cliffs_delta_met_minus_primary']:+.2f}\nP = {r['p_mannwhitney']:.3f}", ha="center", fontsize=4.8)
    from matplotlib.patches import Patch
    hs = [Patch(facecolor="white", edgecolor="black", label="primary"), Patch(facecolor="#777777", label="metastasis")]
    ax.set_xticks(range(K)); ax.set_xticklabels([str(k) for k in range(K)], fontsize=6); ax.set_ylim(0, 0.95)
    ax.legend(handles=hs, fontsize=5.5, loc="upper left", frameon=True, fancybox=False, edgecolor="#CCCCCC")
    bold_axes(ax, x="niche (patients with one lesion type only)", y="mean fraction of spots"); panel_label(ax, "c"); save_source(up, NAME, "c")


def rctd(ax, pc):
    rc = pc[pc["variable"].str.startswith("RCTD") & ~pc["sensitivity_pair"]].copy(); rc["type"] = rc["variable"].str.replace("RCTD ", "")
    g = rc.groupby("type")["delta"].mean().sort_values(); ax.barh(range(len(g)), g, color="#bbbbbb")
    for i, t in enumerate(g.index):
        v = rc[rc["type"] == t]["delta"]; ax.scatter(v, np.full(len(v), i), s=10, color=PALETTE[0], edgecolor="black", linewidths=0.3, zorder=3)
    ax.axvline(0, color="#888888", lw=0.6); ax.set_yticks(range(len(g))); ax.set_yticklabels([t.replace("_", " ") for t in g.index], fontsize=6)
    bold_axes(ax, x="Δ deconvolved weight\n(metastasis − primary)"); panel_label(ax, "b"); save_source(rc, NAME, "b")


def dimensions(ax, ds):
    d = ds.set_index("dimension").loc[["composition", "spatial", "internal state"]]; x = np.arange(len(d))
    ax.bar(x - 0.25, d["n_consistent_all_pairs"], 0.25, color=PALETTE[0], label="same direction in 4/4 pairs")
    ax.bar(x, d["n_consistent_and_unpaired_same_direction"], 0.25, color=PALETTE[2], label="… and in unpaired patients")
    ax.bar(x + 0.25, d["expected_consistent_by_chance"], 0.25, color="white", edgecolor="black", label="expected by chance (4/4)")
    for i, n in enumerate(d["n_variables"]):
        ax.text(i, -1.3, f"{n} variables", ha="center", fontsize=5.2)
    ax.set_xticks(x); ax.set_xticklabels(["composition", "spatial\ncontacts", "within-niche\nstate"], fontsize=6); ax.set_ylim(-2, 12)
    ax.legend(fontsize=5.2, loc="upper left", frameon=True, fancybox=False, edgecolor="#CCCCCC"); bold_axes(ax, y="number of variables"); panel_label(ax, "d")
    save_source(d.reset_index(), NAME, "d")


def main() -> None:
    set_style(); R = ROOT / "results/16_remodelling"; meta = json.load(open(ROOT / "results/15_niches/meta.json"))
    names = {int(k): v for k, v in meta["names"].items()}; K = meta["K"]; cols = {k: PALETTE[k % len(PALETTE)] for k in range(K)}
    pc = pd.read_csv(R / "paired_composition_change.tsv", sep="\t"); up = pd.read_csv(R / "unpaired_primary_vs_metastasis_nonoverlap.tsv", sep="\t")
    ds = pd.read_csv(ROOT / "results/26_remodelling_dimensions/dimension_summary.tsv", sep="\t")
    fig = plt.figure(figsize=(7.2, 7.2)); gs = fig.add_gridspec(3, 2, hspace=0.62, wspace=0.5, height_ratios=[1, 1, 0.42], width_ratios=[1.3, 1], left=0.1, right=0.97, top=0.95, bottom=0.03)
    paired(fig.add_subplot(gs[0, 0]), pc, K, names, cols); rctd(fig.add_subplot(gs[0, 1]), pc)
    unpaired(fig.add_subplot(gs[1, 0]), up, K, cols); dimensions(fig.add_subplot(gs[1, 1]), ds)
    key = fig.add_subplot(gs[2, :]); key.axis("off")
    for k in range(K):
        x0, y0 = (k // 4) * 0.5, 1 - (k % 4) * 0.28
        key.scatter([x0], [y0], color=cols[k], s=18); key.text(x0 + 0.015, y0, f"{k}: {names[k]}", va="center", fontsize=6)
    key.set_xlim(-0.02, 1.0); key.set_ylim(0.05, 1.12)
    save_fig(fig, f"{NAME}_paired_remodelling")


if __name__ == "__main__":
    main()
