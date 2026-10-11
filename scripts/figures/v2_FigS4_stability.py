"""v2 Supplementary Figure S4 — robustness of the niche atlas and of the paired programme change.
a  K sensitivity: number of niches present in at least half of the patients and ARI with the K = 8 partition for K = 6, 8, 10
b  seed stability: agreement of every training seed with the consensus partition (balanced sample) and with the delivered niche map (all spots)
c  paired programme change, all-spots scope vs tumour-dominant scope: number of consistent genes per niche and the overlap
Reads results/15_niches/{k_sensitivity, seed_stability}.tsv, results/16_remodelling/paired_programme_change.tsv
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

NAME = "FigS4"


def main() -> None:
    set_style(); N = ROOT / "results/15_niches"; meta = json.load(open(N / "meta.json")); K = meta["K"]
    ks = pd.read_csv(N / "k_sensitivity.tsv", sep="\t"); st = pd.read_csv(N / "seed_stability.tsv", sep="\t"); prog = pd.read_csv(ROOT / "results/16_remodelling/paired_programme_change.tsv", sep="\t")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    ax = axes[0]; x = np.arange(len(ks)); ax.bar(x - 0.2, ks["n_present_half_patients"], 0.4, color=PALETTE[0], label="niches present in ≥ half of patients")
    ax2 = ax.twinx(); ax2.bar(x + 0.2, ks["ari_vs_fixed_k"], 0.4, color="#bbbbbb", label=f"ARI with K = {K}"); ax2.set_ylim(0, 1.05); ax2.tick_params(labelsize=6)
    ax.set_xticks(x); ax.set_xticklabels([f"K = {k}" for k in ks["k"]], fontsize=6); ax.set_ylim(0, max(ks["k"]) * 1.5); bold_axes(ax, y="niches present (count)"); ax2.set_ylabel("ARI", fontsize=7)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=4.6, loc="upper left", bbox_to_anchor=(0.0, 1.02), frameon=False, handlelength=1.0, handletextpad=0.4); panel_label(ax, "a"); save_source(ks, NAME, "a")
    ax = axes[1]; y = np.arange(len(st)); cons = [c for c in ["ari_with_consensus_sample"] if c in st.columns]
    ax.barh(y - 0.2, st["ari_vs_frozen_seed"], 0.4, color=PALETTE[0], label="vs delivered niche map (all spots)")
    if cons:
        ax.barh(y + 0.2, st[cons[0]], 0.4, color="#bbbbbb", label="vs consensus (balanced sample)")
    ax.set_yticks(y); ax.set_yticklabels([f"seed {i + 2}" for i in range(len(st))], fontsize=6); ax.set_xlim(0, 1); bold_axes(ax, x="ARI"); ax.legend(fontsize=5, loc="upper right", frameon=False, handlelength=1.0, handletextpad=0.4); panel_label(ax, "b"); save_source(st, NAME, "b")
    ax = axes[2]
    if "scope" in prog.columns and len(prog):
        rows = []
        for k in sorted(prog["niche"].unique()):
            a = set(prog[(prog["scope"] == "all_spots") & (prog["niche"] == k)]["gene"]); t = set(prog[(prog["scope"] == "tumour_dominant") & (prog["niche"] == k)]["gene"])
            rows.append({"niche": k, "n_all_spots": len(a), "n_tumour_dominant": len(t), "n_shared": len(a & t)})
        ov = pd.DataFrame(rows); x = np.arange(len(ov)); ax.bar(x - 0.25, ov["n_all_spots"], 0.25, color="#bbbbbb", label="all matched spots"); ax.bar(x, ov["n_tumour_dominant"], 0.25, color=PALETTE[0], label="tumour-dominant spots"); ax.bar(x + 0.25, ov["n_shared"], 0.25, color=PALETTE[2], label="shared")
        ax.set_xticks(x); ax.set_xticklabels(["all" if k == -1 else str(k) for k in ov["niche"]], fontsize=6); bold_axes(ax, x="niche", y="consistent genes (≥ 3 of 4 pairs)"); ax.set_ylim(0, ax.get_ylim()[1] * 1.35); ax.legend(fontsize=5, loc="upper right", frameon=False); save_source(ov, NAME, "c")
    else:
        ax.axis("off"); ax.text(0.5, 0.5, "no consistent genes", ha="center", fontsize=7)
    panel_label(ax, "c"); fig.tight_layout(w_pad=1.5, rect=(0, 0.01, 1, 1)); save_fig(fig, f"{NAME}_stability")


if __name__ == "__main__":
    main()
