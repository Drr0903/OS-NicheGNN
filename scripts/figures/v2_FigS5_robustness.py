"""v2 Supplementary Figure S5 — does the niche atlas depend on the representation? 
a  correspondence of the delivered atlas (OS-NicheGNN latent, inductive) with an atlas built by the identical protocol on
   Harmony-corrected PCA-50 (transductive): fraction of each delivered niche falling in its best-matching Harmony niche, and the
   Pearson correlation of their RCTD composition
b  niche-level agreement: boundary odds ratio and patient-level T-cell score of each delivered niche against its best match
c  contact-remodelling statistic: the permutation z-score grows with section size, the log2(observed/expected) ratio does not
Reads results/17_robustness/*, results/15_niches/*, results/16_remodelling/adjacency_statistic_check.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.plotting import PALETTE, bold_axes, label_points, panel_label, save_fig, save_source, set_style  # noqa: E402
from lib.repro import ROOT  # noqa: E402

NAME = "FigS5"


def main() -> None:
    set_style(); R = ROOT / "results/17_robustness"; N = ROOT / "results/15_niches"
    co = pd.read_csv(R / "niche_correspondence.tsv", sep="\t"); ag = json.load(open(R / "atlas_agreement.json"))
    enr_f = pd.read_csv(N / "niche_boundary_enrichment.tsv", sep="\t").set_index("niche"); enr_h = pd.read_csv(R / "niche_boundary_enrichment_harmony.tsv", sep="\t").set_index("niche")
    ts_f = pd.read_csv(N / "niche_tcell_patient_level.tsv", sep="\t").set_index("niche"); ts_h = pd.read_csv(R / "niche_tcell_harmony.tsv", sep="\t").set_index("niche")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.5, 1.1, 1.0]})
    ax = axes[0]; y = np.arange(len(co))
    ax.barh(y - 0.2, co["frac_of_frozen_niche"], 0.4, color=PALETTE[0], label="spots in best-matching niche")
    ax.barh(y + 0.2, co["composition_pearson_r"], 0.4, color="#bbbbbb", label="composition correlation r")
    for yi, (_, r) in zip(y, co.iterrows()):
        ax.text(1.01, yi, f"→ H{int(r['best_match_harmony'])}", va="center", fontsize=5)
    ax.set_yticks(y); ax.set_yticklabels([f"{int(r['niche_frozen'])}: {r['name_frozen']}" for _, r in co.iterrows()], fontsize=5.2); ax.set_xlim(0, 1.12); ax.invert_yaxis()
    bold_axes(ax, x="fraction / correlation"); ax.legend(fontsize=5, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=1, frameon=False, handlelength=1.0, columnspacing=1.2, handletextpad=0.4); panel_label(ax, "a"); save_source(co, NAME, "a")
    ax = axes[1]; rows = []
    for _, r in co.iterrows():
        kf, kh = int(r["niche_frozen"]), int(r["best_match_harmony"])
        rows.append({"niche_frozen": kf, "name": r["name_frozen"], "or_frozen": float(enr_f.loc[kf, "odds_ratio_near_boundary"]), "or_harmony": float(enr_h.loc[kh, "odds_ratio_near_boundary"]),
                     "t_frozen": float(ts_f.loc[kf, "mean"]) if kf in ts_f.index else np.nan, "t_harmony": float(ts_h.loc[kh, "mean"]) if kh in ts_h.index else np.nan})
    d = pd.DataFrame(rows)
    ax.scatter(d["t_frozen"], d["t_harmony"], s=26, color=PALETTE[0], edgecolor="black", linewidths=0.3)
    lim = [0, max(d[["t_frozen", "t_harmony"]].max()) * 1.15]; ax.plot(lim, lim, color="#888888", lw=0.6, ls="--")
    ds = d.sort_values("t_frozen"); label_points(ax, ds["t_frozen"], ds["t_harmony"], [int(k) for k in ds["niche_frozen"]])
    ax.set_xlim(lim); ax.set_ylim(lim); bold_axes(ax, x="T-cell score, atlas", y="T-cell score, Harmony atlas")
    ax.set_title(f"Spearman ρ = {ag['spearman_tcell_order']:.2f}, atlas ARI = {ag['ari_all_spots']:.2f}\n{ag['n_distinct_harmony_partners']} distinct partners for 8 niches", fontsize=5.6)
    panel_label(ax, "b"); save_source(d, NAME, "b")
    ax = axes[2]; ck = json.load(open(ROOT / "results/16_remodelling/adjacency_statistic_check.json"))
    ax.bar([0, 1], [ck["spearman_z_vs_section_size"], ck["spearman_log2_ratio_vs_section_size"]], color=[PALETTE[3], PALETTE[2]], width=0.6)
    ax.axhline(0, color="#888888", lw=0.6); ax.set_xticks([0, 1]); ax.set_xticklabels(["permutation z", "log2 obs/exp"], fontsize=6)
    bold_axes(ax, y="Spearman with spots per section\n(homotypic contacts)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.28)
    ax.text(0.5, 1.0, f"mean spots: primary {ck['mean_spots_primary']:.0f}, metastasis {ck['mean_spots_metastasis']:.0f}", transform=ax.transAxes, ha="center", va="bottom", fontsize=5.2)
    panel_label(ax, "c"); save_source(pd.DataFrame([ck]), NAME, "c")
    fig.tight_layout(w_pad=1.6); save_fig(fig, f"{NAME}_robustness")


if __name__ == "__main__":
    main()
