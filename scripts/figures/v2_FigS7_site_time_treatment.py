"""v2 Supplementary Figure S7 — remodelling beyond the paired analysis: metastatic sites, time, treatment, contacts and signalling.
a  niche repertoire per site (mean fraction per section; primaries for reference)
b  unpaired primary vs metastasis: patient-level niche fractions with Cliff's delta
c  longitudinal MR10A → MR10B (relapse 1 → 2): niche fractions and interface coverage
d  treatment-before-relapse regimen family: patient-level niche fractions (descriptive)
e  adjacency remodelling: niche-pair contact z, primaries vs metastases (paired delta where available)
f  ligand–receptor co-expression at boundary-niche / tumour contacts: primaries vs metastases
Reads results/16_remodelling/*, results/15_niches/{niche_by_site.tsv, meta.json}
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

NAME = "FigS7"; SITES = ["bone", "lung", "lymph_node", "heart", "soft_tissue"]


def stacked(ax, df, cols, K, xlabels, ylabel):
    bottom = np.zeros(len(df))
    for k in range(K):
        c = str(k) if str(k) in df.columns else (k if k in df.columns else f"niche_{k}"); ax.bar(range(len(df)), df[c], bottom=bottom, color=cols[k], edgecolor="white", linewidth=0.3); bottom += df[c].to_numpy()
    ax.set_xticks(range(len(df))); ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=6); bold_axes(ax, y=ylabel)


def main() -> None:
    set_style(); R = ROOT / "results/16_remodelling"; N = ROOT / "results/15_niches"; meta = json.load(open(N / "meta.json")); names = {int(k): v for k, v in meta["names"].items()}; K = meta["K"]; cols = {k: PALETTE[k % len(PALETTE)] for k in range(K)}
    fig = plt.figure(figsize=(7.2, 8.4))  # three bands placed explicitly: only the rows with rotated tick labels get deep gutters
    kw = dict(left=0.115, right=0.985, wspace=0.40)
    g0 = fig.add_gridspec(1, 2, top=0.965, bottom=0.772, **kw); g1 = fig.add_gridspec(1, 2, top=0.615, bottom=0.385, **kw); g2 = fig.add_gridspec(1, 2, top=0.315, bottom=0.075, **kw)
    axes = {(0, 0): fig.add_subplot(g0[0, 0]), (0, 1): fig.add_subplot(g0[0, 1]), (1, 0): fig.add_subplot(g1[0, 0]),
            (1, 1): fig.add_subplot(g1[0, 1]), (2, 0): fig.add_subplot(g2[0, 0]), (2, 1): fig.add_subplot(g2[0, 1])}
    from matplotlib.patches import Patch
    bs = pd.read_csv(N / "niche_by_site.tsv", sep="\t"); bs = bs.set_index("site").reindex([s for s in SITES if s in set(bs["site"])]).reset_index(); stacked(axes[0, 0], bs, cols, K, [s.replace("_", " ") for s in bs["site"]], "mean niche fraction per section"); panel_label(axes[0, 0], "a"); save_source(bs, NAME, "a")
    axl = fig.add_axes([0.06, 0.676, 0.92, 0.040]); axl.axis("off")
    axl.legend(handles=[Patch(color=cols[k], label=f"{k}: {names[k]}") for k in range(K)], fontsize=5.0, loc="center", ncol=4, frameon=False, handlelength=1.0, columnspacing=1.0, handletextpad=0.5)
    ax = axes[0, 1]; un = pd.read_csv(R / "unpaired_primary_vs_metastasis.tsv", sep="\t"); x = np.arange(K); ax.bar(x - 0.2, un["primary_mean"], 0.4, color="#bbbbbb", label="primary"); ax.bar(x + 0.2, un["metastasis_mean"], 0.4, color=PALETTE[0], label="metastasis")
    for i, r in un.iterrows():
        ax.text(i, max(r["primary_mean"], r["metastasis_mean"]) + 0.01, f"δ {r['cliffs_delta_met_minus_primary']:+.2f}", ha="center", fontsize=4.8)
    ax.set_xticks(x); ax.set_xticklabels([str(k) for k in range(K)], fontsize=6); bold_axes(ax, x="niche", y="fraction of spots (patient means)"); ax.set_ylim(0, float(np.nanmax([un["primary_mean"].max(), un["metastasis_mean"].max()])) * 1.18)
    ax.legend(fontsize=5.5, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False); panel_label(ax, "b"); save_source(un, NAME, "b")
    ax = axes[1, 0]; lon = pd.read_csv(R / "longitudinal_MR10.tsv", sep="\t"); stacked(ax, lon, cols, K, [f"{s}\n({t})" for s, t in zip(lon["section"], lon["timepoint"])], "fraction of spots")
    for i, r in lon.iterrows():
        ax.text(i, 1.02, f"coverage {r['interface_coverage']:.2f}, E_s {r['E_s']:+.2f}" if r["interface_coverage"] == r["interface_coverage"] else "", ha="center", fontsize=5)
    ax.set_ylim(0, 1.12); ax.set_title("patient P10: relapse 1 → relapse 2", fontsize=7); panel_label(ax, "c"); save_source(lon, NAME, "c")
    ax = axes[1, 1]; tr = pd.read_csv(R / "treatment_family_niche_fractions.tsv", sep="\t", header=[0, 1]); fam = tr.iloc[:, 0].astype(str).tolist(); means = pd.DataFrame({str(k): tr[(str(k), "mean")].to_numpy() if (str(k), "mean") in tr.columns else tr[(k, "mean")].to_numpy() for k in range(K)}) if len(tr) else pd.DataFrame()
    if len(means):
        cnt = tr[(str(0), "count")] if (str(0), "count") in tr.columns else tr[(0, "count")]; stacked(ax, means, cols, K, [f"{f}\n(n = {int(c)})" for f, c in zip(fam, cnt)], "fraction of spots (patient means)")
    ax.set_title("treatment before relapse (metastases)", fontsize=7); panel_label(ax, "d"); save_source(tr, NAME, "d")
    ax = axes[2, 0]; ad_ = pd.read_csv(R / "adjacency_remodelling.tsv", sep="\t"); ad_ = ad_.dropna(subset=["log2_ratio_primary_mean", "log2_ratio_metastasis_mean"]); ad_["label"] = ad_["niche_a"].astype(str) + "–" + ad_["niche_b"].astype(str) + ad_["niche_a"].eq(ad_["niche_b"]).map({True: " (self)", False: ""})
    ad_["abs_change"] = (ad_["log2_ratio_metastasis_mean"] - ad_["log2_ratio_primary_mean"]).abs(); ad_ = ad_.sort_values("abs_change", ascending=False).head(12).sort_values("cliffs_delta")
    y = np.arange(len(ad_)); ax.scatter(ad_["log2_ratio_primary_mean"], y, s=14, color="#bbbbbb", label="primary"); ax.scatter(ad_["log2_ratio_metastasis_mean"], y, s=14, color=PALETTE[0], label="metastasis")
    for yi, (_, r) in zip(y, ad_.iterrows()):
        ax.plot([r["log2_ratio_primary_mean"], r["log2_ratio_metastasis_mean"]], [yi, yi], color="#888888", lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels(ad_["label"], fontsize=5); ax.axvline(0, color="#888888", lw=0.6, ls="--"); bold_axes(ax, x="contact log2(observed / expected), patient means\n12 pairs with the largest change"); ax.legend(fontsize=5.5, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False); panel_label(ax, "e"); save_source(ad_.drop(columns=["label", "abs_change"]), NAME, "e")
    ax = axes[2, 1]; f = R / "lr_primary_vs_metastasis.tsv"
    if f.exists() and f.stat().st_size > 5:
        lr = pd.read_csv(f, sep="\t"); lr["pair"] = lr["ligand"] + "–" + lr["receptor"]; lr = lr.sort_values("delta_log2_ratio"); y = np.arange(len(lr))
        ax.scatter(lr["log2_ratio_primary"], y, s=14, color="#bbbbbb", label="primary"); ax.scatter(lr["log2_ratio_metastasis"], y, s=14, color=PALETTE[0], label="metastasis")
        for yi, (_, r) in zip(y, lr.iterrows()):
            ax.plot([r["log2_ratio_primary"], r["log2_ratio_metastasis"]], [yi, yi], color="#888888", lw=0.6)
        ax.set_yticks(y); ax.set_yticklabels(lr["pair"], fontsize=5); ax.axvline(0, color="#888888", lw=0.6, ls="--"); bold_axes(ax, x="log2(observed / null) co-expression\nat boundary–tumour contacts"); ax.legend(fontsize=5.5, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False); save_source(lr, NAME, "f")
    else:
        ax.axis("off"); ax.text(0.5, 0.5, "ligand–receptor comparison not estimable", ha="center", fontsize=7)
    panel_label(ax, "f"); save_fig(fig, f"{NAME}_site_time_treatment")


if __name__ == "__main__":
    main()
