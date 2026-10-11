"""v2 Figure 5 — spatial organisation of the niches.
a  niche–niche adjacency enrichment (patient-level mean z of ring-1 contacts vs within-section permutation null)
b  patient-level T-cell score per niche
c  boundary enrichment (odds ratio within one ring of the tumour boundary) per niche
d  lung sections: interface coverage by the fibroblast-rich boundary niche vs E_s (per section, colour = patient)
e  pooled within-section coefficients of coverage on the three T-cell readouts (v1 machinery, lung sections)
Reads results/15_niches/*
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.plotting import COVSET_NAME, OUTCOME_NAME, PALETTE, bold_axes, forest, panel_label, save_fig, save_source, set_style  # noqa: E402
from lib.repro import ROOT  # noqa: E402

NAME = "Fig5"


def main() -> None:
    set_style(); N = ROOT / "results/15_niches"; meta = json.load(open(N / "meta.json")); names = {int(k): v for k, v in meta["names"].items()}; K = meta["K"]; inter = meta["interface_niche"]; cols = {k: PALETTE[k % len(PALETTE)] for k in range(K)}
    adj = pd.read_csv(N / "niche_adjacency_patient_level.tsv", sep="\t"); ts = pd.read_csv(N / "niche_tcell_patient_level.tsv", sep="\t"); enr = pd.read_csv(N / "niche_boundary_enrichment.tsv", sep="\t")
    sec = pd.read_csv(N / "interface_section_metrics.tsv", sep="\t"); pm = pd.read_csv(N / "interface_pooled_models.tsv", sep="\t")
    fig = plt.figure(figsize=(7.2, 7.4)); gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.85, width_ratios=[1.3, 1, 1], height_ratios=[1.0, 1.45, 0.34], left=0.135, right=0.975, top=0.965, bottom=0.065)
    ax = fig.add_subplot(gs[0, 0]); M = np.full((K, K), np.nan)
    for _, r in adj.iterrows():
        M[int(r["niche_a"]), int(r["niche_b"])] = r["z_mean"]; M[int(r["niche_b"]), int(r["niche_a"])] = r["z_mean"]
    v = np.nanmax(np.abs(M)); im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto"); ax.set_xticks(range(K)); ax.set_yticks(range(K)); ax.set_xticklabels([str(k) for k in range(K)], fontsize=6); ax.set_yticklabels([names[k] for k in range(K)], fontsize=5.5)  # aspect="auto" fills the cell, so the panel label stays level with b and c
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02); cb.ax.tick_params(labelsize=5.5); cb.set_label("contact z", fontsize=6); panel_label(ax, "a"); save_source(adj, NAME, "a")
    ax = fig.add_subplot(gs[0, 1]); ts = ts.sort_values("niche"); ax.bar(ts["niche"], ts["mean"], yerr=ts["std"], color=[cols[int(k)] for k in ts["niche"]], capsize=2, edgecolor="black", linewidth=0.3)
    ax.set_xticks(ts["niche"]); ax.set_xticklabels([str(int(k)) for k in ts["niche"]], fontsize=6); bold_axes(ax, x="niche", y="T-cell score (patient mean)"); panel_label(ax, "b"); save_source(ts, NAME, "b")
    ax = fig.add_subplot(gs[0, 2]); ax.bar(enr["niche"], enr["odds_ratio_near_boundary"], color=[cols[int(k)] for k in enr["niche"]], edgecolor=["red" if b else "black" for b in enr["is_interface_niche"]], linewidth=[1.2 if b else 0.3 for b in enr["is_interface_niche"]])
    ax.axhline(1, color="#888888", lw=0.6, ls="--"); ax.set_xticks(enr["niche"]); ax.set_xticklabels([str(int(k)) for k in enr["niche"]], fontsize=6); bold_axes(ax, x="niche", y="odds ratio, within 1 ring\nof tumour boundary"); ax.set_ylim(0, ax.get_ylim()[1] * 1.18); panel_label(ax, "c"); save_source(enr, NAME, "c")
    ax = fig.add_subplot(gs[1, :2]); ev = sec[sec["evaluable_T"]]; pats = sorted(ev["patient"].unique()); pc = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(pats)}
    ax.scatter(ev["coverage"], ev["E_s"], s=26, c=[pc[p] for p in ev["patient"]], edgecolor="black", linewidths=0.3)
    for _, r in ev.iterrows():
        ax.annotate(r["section"], (r["coverage"], r["E_s"]), xytext=(3, 3), textcoords="offset points", fontsize=5.5)
    ax.axhline(0, color="#888888", lw=0.6, ls="--"); ax.set_xlim(-0.06, 1.15); bold_axes(ax, x="coverage of the tumour–lung interface by the band (per section)", y="E_s = log(T outer / T inner)"); panel_label(ax, "d"); save_source(ev[["section", "patient", "coverage", "E_s", "E_s_noniche", "E_s_distal", "n_segments"]], NAME, "d")
    ax = fig.add_subplot(gs[1:, 2]); pmi = pm.set_index(["outcome", "covariate_set"]); rows = [pmi.loc[(o, c)].to_dict() | {"outcome": o, "covariate_set": c, "label": f"{OUTCOME_NAME[o].replace('E (', '').replace(')', '')}\n{COVSET_NAME[c]}, n = {int(pmi.loc[(o, c), 'n_segments'])}"} for o in ["E_seg", "E_seg_noniche", "E_seg_distal"] for c in ["none", "full"] if (o, c) in pmi.index and pmi.loc[(o, c), "coef"] == pmi.loc[(o, c), "coef"]]
    ft = pd.DataFrame(rows); forest(ax, ft, "label", color_col="outcome", label_fs=5, group_col="outcome", xlabel="coverage coefficient (95 % CI, t with G − 1 d.f.)"); panel_label(ax, "e"); save_source(ft.drop(columns="label"), NAME, "e")
    ax = fig.add_subplot(gs[2, :2]); ax.axis("off"); half = (K + 1) // 2; ax.text(0.0, 1.0, "niche key\n" + "\n".join(f"{k}: {names[k]}" for k in range(half)), fontsize=6, va="top", transform=ax.transAxes); ax.text(0.5, 1.0, "\n" + "\n".join(f"{k}: {names[k]}" for k in range(half, K)), fontsize=6, va="top", transform=ax.transAxes)
    save_fig(fig, f"{NAME}_organisation")


if __name__ == "__main__":
    main()
