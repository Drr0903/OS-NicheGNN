"""v2 Figure 2 — OS-NicheGNN on patients it has never seen .
a  leave-one-patient-out region macro-F1 per held-out patient (lines: patients, seed-averaged; colour: held-out site)
b  T-cell programme of the neighbourhood (Spearman ρ), same layout
c  paired patient-level differences of OS-NicheGNN against BANKSY-20 and PCA-50 + H&E (bootstrap 95 % CI)
d  the same encoder on the true and on a rewired neighbour graph (patient means ± s.e.m.)
e  niche reuse: agreement of a held-out patient's niches (prototypes from the other 18 patients) with the full-cohort atlas,
   against the same agreement on the training patients (re-clustering alone)
Reads results/23_arch_search/{arch_patient_level, arch_paired, arch_metrics}.tsv, results/25_niche_reuse/reuse_patient.tsv
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
        ax.text(i, ylim[1] * 0.985, f"{v:.3f}", ha="center", va="top", fontsize=6)
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([LABEL[m] for m in ORDER], fontsize=6, rotation=25, ha="right")
    ax.set_xlim(-0.35, len(ORDER) - 0.65); ax.set_ylim(*ylim); bold_axes(ax, y=ylabel); panel_label(ax, panel)
    save_source(w.reset_index(), NAME, panel)


def forest(ax, pr, nd):
    rows = []; sets = [("banksy", PALETTE[1], pr, "vs BANKSY-20"), ("banksy_k36", PALETTE[2], nd, "vs BANKSY-36"), ("pca+img", PALETTE[7], pr, "vs PCA-50 + H&E")]
    for j, (ref, col, tab, leg) in enumerate(sets):
        for i, (m, lab) in enumerate(METRIC):
            r = tab[(tab["comparison"] == f"{OURS} − {ref}") & (tab["metric"] == m)].iloc[0]; y = i + (j - 1) * 0.27
            sig = r["ci95_lo"] > 0 or r["ci95_hi"] < 0
            ax.plot([r["ci95_lo"], r["ci95_hi"]], [y, y], color=col, lw=1.1)
            ax.scatter([r["mean_diff"]], [y], color=col if sig else "white", edgecolor=col, s=16, zorder=3, lw=0.9)
            ax.text(0.300 + j * 0.033, i, f"{int(r['n_better'])}/{int(r['n_patients'])}", fontsize=5.2, va="center", ha="left", color=col)
            rows.append({"reference": leg, "metric": m, **r[["n_patients", "mean_diff", "ci95_lo", "ci95_hi", "n_better", "n_worse"]].to_dict()})
        ax.scatter([], [], color=col, s=16, label=leg)
    ax.axvline(0, color="black", lw=0.6, ls="--"); ax.set_yticks(range(len(METRIC))); ax.set_yticklabels([l for _, l in METRIC], fontsize=6)
    ax.invert_yaxis(); ax.set_xlim(-0.1, 0.40); ax.text(0.300, -0.62, "patients better", fontsize=6, ha="left", va="bottom")
    bold_axes(ax, x="OS-NicheGNN − reference (difference in patient means, 95 % CI)")
    ax.legend(fontsize=6, loc="lower center", bbox_to_anchor=(0.45, 1.01), ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.2); panel_label(ax, "c")
    save_source(pd.DataFrame(rows), NAME, "c")


def rewired(ax, pat):
    ms = [("macro_f1", "region macro-F1"), ("tcell_nbhd_rho", "T cells, neighbourhood"), ("comp_nbhd_rho", "composition, neighbourhood")]; out = []
    for i, (m, lab) in enumerate(ms):
        for j, (meth, col) in enumerate([(OURS, PALETTE[0]), (REW, PALETTE[7])]):
            v = pat.loc[pat["method"] == meth, m].dropna(); se = v.std(ddof=1) / np.sqrt(len(v))
            ax.bar(i + (j - 0.5) * 0.36, v.mean(), 0.34, yerr=se, color=col, error_kw={"lw": 0.7, "capsize": 1.5}, label=LABEL[meth] if i == 0 else None)
            out.append({"metric": m, "graph": "true" if meth == OURS else "rewired", "mean": v.mean(), "sem": se, "n": len(v)})
    ax.set_xticks(range(len(ms))); ax.set_xticklabels([l for _, l in ms], fontsize=6, rotation=25, ha="right"); ax.set_ylim(0, 1.12)
    ax.legend(fontsize=5.5, loc="upper center", ncol=2, frameon=False, handlelength=1.2, columnspacing=0.8); bold_axes(ax, y="held-out score"); panel_label(ax, "d")
    save_source(pd.DataFrame(out), NAME, "d")


def reuse(ax, ru, site):
    w = ru.pivot(index="heldout_patient", columns="scope", values="agreement")
    for p, r_ in w.iterrows():
        ax.plot([0, 1], [r_["training"], r_["held_out"]], color=SITE_COL.get(site.get(p), "#888888"), lw=0.6, alpha=0.7, marker="o", ms=2.5)
    mu = w[["training", "held_out"]].mean(); ax.plot([0, 1], mu.to_numpy(), color="black", lw=1.6, marker="o", ms=4)
    for i, v in enumerate(mu):
        ax.text(i, 1.0, f"{v:.3f}", ha="center", va="top", fontsize=6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["training\npatients", "held-out\npatient"], fontsize=6)
    ax.set_xlim(-0.35, 1.35); ax.set_ylim(0.4, 1.02); bold_axes(ax, y="agreement with the\nfull-cohort atlas"); panel_label(ax, "e")
    save_source(w.reset_index(), NAME, "e")


REP_COL = {"OS-NicheGNN": "#3B5BA5", "BANKSY-20": "#DD8452", "spot profile": "#8C8C8C", "profile + ring means": "#55A868",
           "without the attention branch": "#3B5BA5", "without the ring-mean branch": "#3B5BA5", "single fusion expert": "#9A4F8C",
           "branches concatenated": "#9A4F8C"}
REP_LAB = {"without the attention branch": "− attention branch", "without the ring-mean branch": "− ring-mean branch", "single fusion expert": "single expert",
           "branches concatenated": "concatenated"}
REP_LS = {"without the attention branch": "--", "without the ring-mean branch": ":", "single fusion expert": "--", "branches concatenated": ":"}


def coverage(ax, cv):
    """Agreement among the held-out spots closest to a niche, against the fraction of spots assigned (fixed reference)."""
    d = cv[(cv["mode"] == "mapped") & (cv["reference"] == "same-protocol atlas")]
    for rep, col in REP_COL.items():
        r = d[d.representation == rep].sort_values("coverage")
        ax.plot(r["coverage"], r["agreement_mean"], color=col, lw=1.4 if rep == "OS-NicheGNN" else 1.0, ls=REP_LS.get(rep, "-"), label=REP_LAB.get(rep, rep))
    ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0.4, 0.6, 0.8, 1.0]); ax.set_ylim(0.3, 1.42); ax.legend(fontsize=5.0, loc="upper center", ncol=2, frameon=False, handlelength=1.6, columnspacing=0.8)
    bold_axes(ax, x="held-out spots\nassigned (fraction)", y="agreement\nwith the atlas"); panel_label(ax, "f")
    save_source(d, NAME, "f")


def main() -> None:
    set_style(); A = ROOT / "results/23_arch_search"
    pat = pd.read_csv(A / "arch_patient_level.tsv", sep="\t"); pr = pd.read_csv(A / "arch_paired.tsv", sep="\t"); m = pd.read_csv(A / "arch_metrics.tsv", sep="\t")
    site = m.groupby("heldout_patient")["heldout_site"].first(); pat["heldout_site"] = pat["heldout_patient"].map(site)
    ru = pd.read_csv(ROOT / "results/25_niche_reuse/reuse_patient.tsv", sep="\t")
    fig = plt.figure(figsize=(7.2, 8.4)); gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 0.95, 0.9], hspace=0.62, wspace=0.55)
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1]); c = fig.add_subplot(gs[1, :])
    bot = gs[2, :].subgridspec(1, 3, wspace=0.62); d = fig.add_subplot(bot[0]); e = fig.add_subplot(bot[1]); f = fig.add_subplot(bot[2])
    lines(a, pat, "macro_f1", "held-out region macro-F1", "a", (0, 1.1)); lines(b, pat, "tcell_nbhd_rho", "held-out T-cell programme,\nneighbourhood (ρ)", "b", (-0.15, 1.05))
    for s_, c_ in SITE_COL.items():
        b.plot([], [], color=c_, lw=1, label=s_.replace("_", " "))
    b.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0), title="held-out site", title_fontsize=6, frameon=False)
    nd = pd.read_csv(ROOT / "results/23d_banksy_k36/banksy_k36_paired.tsv", sep="\t"); nd = nd[nd["patients"] == "all"]
    forest(c, pr, nd); rewired(d, pat); reuse(e, ru, site)
    coverage(f, pd.read_csv(ROOT / "results/25c_reference_mapping/coverage_curve.tsv", sep="\t"))
    save_fig(fig, f"{NAME}_validation")


if __name__ == "__main__":
    main()
