"""v2 Supplementary Figure S6 — signalling identity of the niches and the gradient across the tumour–lung interface .
a  PROGENy pathway activity per niche (patient means, z-scored across niches); all 14 pathways differ between niches
b  contrasts that survive adjustment for the eleven deconvolved cell-type fractions (patient fixed effects, cluster-robust CI)
c  signed ring distance to the tumour–lung interface: T-cell score and the pathways with a monotone gradient (patient means)
Reads results/18_signalling/*
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.plotting import PALETTE, bold_axes, panel_label, save_fig, save_source, set_style  # noqa: E402
from lib.repro import ROOT  # noqa: E402

NAME = "FigS6"
GRAD = ["t_score", "TGFb", "Hypoxia", "TNFa", "JAK-STAT"]
LAB = {"t_score": "T-cell score", "TGFb": "TGF-β", "Hypoxia": "hypoxia", "TNFa": "TNF-α", "JAK-STAT": "JAK–STAT"}


def main() -> None:
    set_style(); R = ROOT / "results/18_signalling"
    bp = pd.read_csv(R / "pathway_by_patient_niche.tsv", sep="\t"); tests = pd.read_csv(R / "pathway_niche_tests.tsv", sep="\t")
    adj = pd.read_csv(R / "pathway_composition_adjusted.tsv", sep="\t"); prof = pd.read_csv(R / "interface_distance_profile.tsv", sep="\t")
    fig = plt.figure(figsize=(7.2, 9.0)); gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.75, 1.05], hspace=0.45, wspace=0.5)
    # a: per-niche pathway profile, z-scored across niches so that the pattern, not the scale, is shown
    ax = fig.add_subplot(gs[0, :]); m = bp.groupby(["niche", "source"])["activity"].mean().unstack()
    names = bp.drop_duplicates("niche").set_index("niche")["name"]
    z = (m - m.mean()) / m.std(ddof=0)
    order = tests.sort_values("p")["source"].tolist(); z = z[[c for c in order if c in z.columns]]
    v = float(np.nanmax(np.abs(z.to_numpy()))); im = ax.imshow(z.to_numpy(), cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_yticks(range(len(z))); ax.set_yticklabels([f"{k}: {names[k]}" for k in z.index], fontsize=5.6)
    ax.set_xticks(range(z.shape[1])); ax.set_xticklabels([LAB.get(c, c) for c in z.columns], rotation=55, ha="right", fontsize=6)
    cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.015); cb.ax.tick_params(labelsize=5.5); cb.set_label("activity, z across niches", fontsize=6)
    panel_label(ax, "a"); save_source(m.reset_index(), NAME, "a")
    # b: composition-adjusted contrasts that survive
    ax = fig.add_subplot(gs[1, :]); s = adj[(adj["adjustment"] == "composition") & (adj["fdr"] < 0.1)].sort_values("coef_vs_reference")
    s = s.assign(label=lambda d: d["source"].map(lambda x: LAB.get(x, x)) + "  ·  " + d["niche"].astype(str) + " " + d["name"].str.replace(" (boundary)", "", regex=False))
    y = np.arange(len(s)); col = [PALETTE[3] if c < 0 else PALETTE[0] for c in s["coef_vs_reference"]]
    ax.errorbar(s["coef_vs_reference"], y, xerr=[s["coef_vs_reference"] - s["ci95_lo"], s["ci95_hi"] - s["coef_vs_reference"]], fmt="none", ecolor="#888888", lw=0.8)
    ax.scatter(s["coef_vs_reference"], y, s=16, c=col, edgecolor="black", linewidths=0.3, zorder=3)
    ax.axvline(0, color="#888888", lw=0.6, ls="--"); ax.set_yticks(y); ax.set_yticklabels(s["label"], fontsize=5.4); ax.set_ylim(-0.8, len(s) - 0.2)
    ref = int(s["reference_niche"].iloc[0]) if len(s) else 0
    bold_axes(ax, x=f"activity difference from niche {ref} after adjustment for eleven cell-type fractions (95 % CI)")
    panel_label(ax, "b"); save_source(s.drop(columns="label"), NAME, "b")
    # c: gradient across the interface
    ax = fig.add_subplot(gs[2, :]); g = prof.groupby("signed_ring")[GRAD].agg(["mean", "sem"])
    x = g.index.to_numpy(); ax2 = ax.twinx()
    ax.errorbar(x, g[("t_score", "mean")], yerr=g[("t_score", "sem")], color="black", lw=1.4, marker="o", ms=3, label=LAB["t_score"], zorder=5)
    for i, c in enumerate([p for p in GRAD if p != "t_score"]):
        ax2.errorbar(x, g[(c, "mean")], yerr=g[(c, "sem")], color=PALETTE[i], lw=1.0, marker="s", ms=2.5, alpha=0.85, label=LAB[c])
    ax.axvline(0, color="#888888", lw=0.8, ls="--"); ax.set_xticks(x)
    bold_axes(ax, x="signed ring distance to the tumour–lung interface (negative = inside the tumour)", y="T-cell score")
    ax2.set_ylabel("pathway activity", fontsize=7)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False)
    ax.text(0.99, 0.04, f"{prof['patient'].nunique()} patients", transform=ax.transAxes, ha="right", fontsize=5.4)
    panel_label(ax, "c"); save_source(prof, NAME, "c")
    save_fig(fig, f"{NAME}_signalling")


if __name__ == "__main__":
    main()
