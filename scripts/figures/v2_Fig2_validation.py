"""v2 Figure 2 — OS-NicheGNN on patients it has never seen .
a  leave-one-patient-out region macro-F1 per held-out patient (lines: patients, seed-averaged; colour: held-out site)
b  T-cell programme of the neighbourhood (Spearman ρ), same layout
c  paired patient-level differences of OS-NicheGNN against BANKSY-20 and PCA-50 + H&E (bootstrap 95 % CI)
d  the same encoder on the true and on a rewired neighbour graph (patient means ± s.e.m.)
e  niche reuse: agreement of a held-out patient's niches (prototypes from the other 18 patients) with the full-cohort atlas,
   against the same agreement on the training patients (re-clustering alone)
f  neighbourhood T-cell read-out against the patient separation of each representation's niches (patient means ± s.e.m.)
Reads results/23_arch_search/{arch_patient_level, arch_paired, arch_metrics}.tsv, results/25_niche_reuse/reuse_patient.tsv,
results/23e_patient_mixing/mixing_patient_level.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.plotting import PALETTE, bold_axes, panel_label, save_fig, save_source, set_style  # noqa: E402
from lib.repro import ROOT, load_config  # noqa: E402

NAME = "Fig2"
OURS = load_config()["atlas"]["encoder"]; REW = OURS + "_rewired"
ORDER = ["pca", "pca+img", "banksy", OURS]
LABEL = {"pca": "PCA-50", "pca+img": "PCA-50 + H&E", "banksy": "BANKSY-20", OURS: "OS-NicheGNN", REW: "rewired graph"}
SITE_COL = {"bone": PALETTE[5], "lung": PALETTE[0], "lymph_node": PALETTE[4], "heart": PALETTE[3], "soft_tissue": PALETTE[2], "bone+lung": PALETTE[1]}
METRIC = [("macro_f1", "region macro-F1"), ("boundary_f1", "boundary F1"), ("tcell_rho", "T-cell programme, spot"),
          ("tcell_nbhd_rho", "T-cell programme, neighbourhood"), ("comp_rho", "cell composition, spot"),
          ("comp_nbhd_rho", "cell composition, neighbourhood")]


def lines(ax, pat, metric, ylabel, panel, ylim):
    w = pat.pivot_table(index=["heldout_patient", "heldout_site"], columns="method", values=metric).reindex(columns=ORDER).dropna()
    for (_, st), r_ in w.iterrows():
        ax.plot(range(len(ORDER)), r_.to_numpy(), color=SITE_COL.get(st, "#888888"), lw=0.6, alpha=0.6, marker="o", ms=2.5, zorder=2)
    mu = w.mean(); ax.plot(range(len(ORDER)), mu.to_numpy(), color="black", lw=1.6, marker="o", ms=4, zorder=5)
    for i, v in enumerate(mu):
        ax.text(i, ylim[1] * 0.985, f"{v:.3f}", ha="center", va="top", fontsize=5.5)
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([LABEL[m] for m in ORDER], fontsize=6, rotation=25, ha="right")
    ax.set_xlim(-0.35, len(ORDER) - 0.65); ax.set_ylim(*ylim); bold_axes(ax, y=ylabel); panel_label(ax, panel)
    save_source(w.reset_index(), NAME, panel)


def forest(ax, pr):
    rows = []
    for j, (ref, col) in enumerate([("banksy", PALETTE[1]), ("pca+img", PALETTE[7])]):
        for i, (m, lab) in enumerate(METRIC):
            r = pr[(pr["comparison"] == f"{OURS} − {ref}") & (pr["metric"] == m)].iloc[0]; y = i + (j - 0.5) * 0.32
            sig = r["ci95_lo"] > 0 or r["ci95_hi"] < 0
            ax.plot([r["ci95_lo"], r["ci95_hi"]], [y, y], color=col, lw=1.1)
            ax.scatter([r["mean_diff"]], [y], color=col if sig else "white", edgecolor=col, s=16, zorder=3, lw=0.9)
            ax.text(0.33, y, f"{int(r['n_better'])}/{int(r['n_patients'])}", fontsize=5, va="center", color=col)
            rows.append({"reference": ref, "metric": m, **r[["n_patients", "mean_diff", "ci95_lo", "ci95_hi", "n_better", "n_worse"]].to_dict()})
        ax.scatter([], [], color=col, s=16, label=f"vs {LABEL[ref]}")
    ax.axvline(0, color="black", lw=0.6, ls="--"); ax.set_yticks(range(len(METRIC))); ax.set_yticklabels([l for _, l in METRIC], fontsize=6)
    ax.invert_yaxis(); ax.set_xlim(-0.1, 0.37); ax.text(0.33, -0.75, "patients\nbetter", fontsize=5, ha="left", va="bottom")
    bold_axes(ax, x="OS-NicheGNN − reference (patient-level mean difference, bootstrap 95 % CI; filled = CI excludes 0)")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.6, 0.99), frameon=True, fancybox=False, edgecolor="#CCCCCC"); panel_label(ax, "c")
    save_source(pd.DataFrame(rows), NAME, "c")


def rewired(ax, pat):
    ms = [("macro_f1", "region macro-F1"), ("tcell_nbhd_rho", "T cells, neighbourhood"), ("comp_nbhd_rho", "composition, neighbourhood")]; out = []
    for i, (m, lab) in enumerate(ms):
        for j, (meth, col) in enumerate([(OURS, PALETTE[0]), (REW, PALETTE[7])]):
            v = pat.loc[pat["method"] == meth, m].dropna(); se = v.std(ddof=1) / np.sqrt(len(v))
            ax.bar(i + (j - 0.5) * 0.36, v.mean(), 0.34, yerr=se, color=col, error_kw={"lw": 0.7, "capsize": 1.5}, label=LABEL[meth] if i == 0 else None)
            out.append({"metric": m, "graph": "true" if meth == OURS else "rewired", "mean": v.mean(), "sem": se, "n": len(v)})
    ax.set_xticks(range(len(ms))); ax.set_xticklabels([l for _, l in ms], fontsize=5.5, rotation=25, ha="right"); ax.set_ylim(0, 0.95)
    ax.legend(fontsize=5.8, loc="upper right", frameon=True, fancybox=False, edgecolor="#CCCCCC"); bold_axes(ax, y="held-out score"); panel_label(ax, "d")
    save_source(pd.DataFrame(out), NAME, "d")


def reuse(ax, ru, site):
    w = ru.pivot(index="heldout_patient", columns="scope", values="agreement")
    for p, r_ in w.iterrows():
        ax.plot([0, 1], [r_["training"], r_["held_out"]], color=SITE_COL.get(site.get(p), "#888888"), lw=0.6, alpha=0.7, marker="o", ms=2.5)
    mu = w[["training", "held_out"]].mean(); ax.plot([0, 1], mu.to_numpy(), color="black", lw=1.6, marker="o", ms=4)
    for i, v in enumerate(mu):
        ax.text(i, 1.0, f"{v:.3f}", ha="center", va="top", fontsize=5.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["training patients\n(re-clustering)", "held-out patient\n(niches transferred)"], fontsize=6)
    ax.set_xlim(-0.35, 1.35); ax.set_ylim(0.4, 1.02); bold_axes(ax, y="agreement with the\nfull-cohort atlas"); panel_label(ax, "e")
    save_source(w.reset_index(), NAME, "e")


MIX = {"pca": "PCA-50", "pca+img": "PCA-50\n+ H&E", "propagated_pca": "propagated\nPCA-50", "propagated_pca+img": "propagated\nPCA-50 + H&E",
       "banksy": "BANKSY-20", "banksy+img": "BANKSY-20\n+ H&E", OURS: "OS-NicheGNN"}
TXT = {OURS: (0.100, 0.615, "left"), "propagated_pca+img": (0.150, 0.805, "center"), "propagated_pca": (0.232, 0.815, "center"),
       "banksy": (0.250, 0.640, "center"), "banksy+img": (0.252, 0.455, "center"), "pca": (0.212, 0.375, "center"), "pca+img": (0.122, 0.400, "center")}


def mixing(ax, mx):
    g = mx.groupby("method")[["train_ari_vs_patient", "tcell_nbhd_rho"]]; mu = g.mean(); se = g.std(ddof=1) / np.sqrt(g.count())
    for m in MIX:
        col = PALETTE[0] if m == OURS else PALETTE[7]
        ax.errorbar(mu.loc[m, "train_ari_vs_patient"], mu.loc[m, "tcell_nbhd_rho"], xerr=se.loc[m, "train_ari_vs_patient"], yerr=se.loc[m, "tcell_nbhd_rho"],
                    fmt="o", ms=4 if m == OURS else 3, color=col, ecolor=col, elinewidth=0.6, capsize=1.2, zorder=3)
        tx, ty, ha = TXT[m]
        ax.annotate(MIX[m], (mu.loc[m, "train_ari_vs_patient"], mu.loc[m, "tcell_nbhd_rho"]), xytext=(tx, ty), textcoords="data", fontsize=5, ha=ha, va="center",
                    color=col, fontweight="bold" if m == OURS else "normal", arrowprops={"arrowstyle": "-", "lw": 0.4, "color": col, "shrinkA": 1, "shrinkB": 3})
    ax.set_xlim(0.07, 0.28); ax.set_ylim(0.33, 0.87)
    bold_axes(ax, x="patient separation of niches\n(ARI with patient; lower is better)", y="T-cell programme,\nneighbourhood (ρ)"); panel_label(ax, "f")
    save_source(mu.join(se, rsuffix="_sem").reset_index(), NAME, "f")


def main() -> None:
    set_style(); A = ROOT / "results/23_arch_search"
    pat = pd.read_csv(A / "arch_patient_level.tsv", sep="\t"); pr = pd.read_csv(A / "arch_paired.tsv", sep="\t"); m = pd.read_csv(A / "arch_metrics.tsv", sep="\t")
    site = m.groupby("heldout_patient")["heldout_site"].first(); pat["heldout_site"] = pat["heldout_patient"].map(site)
    ru = pd.read_csv(ROOT / "results/25_niche_reuse/reuse_patient.tsv", sep="\t")
    mx = pd.read_csv(ROOT / "results/23e_patient_mixing/mixing_patient_level.tsv", sep="\t")
    fig = plt.figure(figsize=(7.2, 8.4)); gs = GridSpec(3, 6, figure=fig, height_ratios=[1, 0.95, 0.9], hspace=0.62, wspace=1.6)
    a = fig.add_subplot(gs[0, :3]); b = fig.add_subplot(gs[0, 3:]); c = fig.add_subplot(gs[1, :]); d = fig.add_subplot(gs[2, :2]); e = fig.add_subplot(gs[2, 2:4])
    f = fig.add_subplot(gs[2, 4:])
    lines(a, pat, "macro_f1", "held-out region macro-F1", "a", (0, 1.1)); lines(b, pat, "tcell_nbhd_rho", "held-out T-cell programme,\nneighbourhood (ρ)", "b", (-0.15, 1.05))
    for s_, c_ in SITE_COL.items():
        b.plot([], [], color=c_, lw=1, label=s_.replace("_", " "))
    b.legend(fontsize=5.5, loc="upper left", bbox_to_anchor=(1.02, 1.0), title="held-out site", title_fontsize=5.5, frameon=False)
    forest(c, pr); rewired(d, pat); reuse(e, ru, site); mixing(f, mx)
    save_fig(fig, f"{NAME}_validation")


if __name__ == "__main__":
    main()
