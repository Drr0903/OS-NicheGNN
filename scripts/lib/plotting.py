"""Figure conventions (workspace style guide: Arial 8 pt, hidden top/right spines, lowercase bold panel labels,
PDF + 400-dpi PNG, source data written next to every panel). Figure scripts read results/ only."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .repro import FIGURES  # noqa: E402

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
REGION_COL = {"tumor_cellular": "#7b2cbf", "lung_parenchyma": "#2a9d8f", "paucicellular": "#e9c46a", "unevaluable": "#bbbbbb"}
REGION_NAME = {"tumor_cellular": "tumour", "lung_parenchyma": "lung parenchyma", "paucicellular": "non-tumour stroma/other", "unevaluable": "unevaluable"}  # wording used in the manuscript
METHOD_COL = {"pca": "#8C8C8C", "mlp_ae": "#937860", "neigh_mean_ae": "#CCB974", "banksy": "#55A868", "banksy_50pc": "#2E7D4F", "stagate_transductive": "#DD8452",
              "nichegnn": "#4C72B0", "no_outer": "#64B5CD", "no_diff": "#8172B3", "no_aux": "#DA8BC3", "no_edges": "#C44E52", "rewired": "#bbbbbb"}
METHOD_NAME = {"pca": "PCA", "mlp_ae": "MLP-AE", "neigh_mean_ae": "NeighMean-AE", "banksy": "BANKSY (20 PCs)", "banksy_50pc": "BANKSY (50 PCs)", "stagate_transductive": "STAGATE (transductive)",
               "nichegnn": "OS-NicheGNN", "no_outer": "− outer branch", "no_diff": "− difference term", "no_aux": "− context head", "no_edges": "− all edges", "rewired": "rewired edges"}


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
        "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE), "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": "#CCCCCC",
    })


def panel_label(ax, letter: str) -> None:
    ax.set_title(letter, fontweight="bold", fontsize=11, loc="left", pad=8)


def label_points(ax, xs, ys, texts, fontsize=5.5, colour="black"):
    """Annotate scattered points, taking the first candidate offset that does not touch a label already placed."""
    cands = [(4, 4), (4, -9), (-11, 4), (-11, -9), (0, 8), (0, -13)]
    ax.figure.canvas.draw(); r = ax.figure.canvas.get_renderer(); placed = []
    for x, y, s in zip(xs, ys, texts):
        for dx, dy in cands:
            a = ax.annotate(str(s), (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize, color=colour)
            bb = a.get_window_extent(renderer=r).expanded(1.15, 1.25)
            if not any(bb.overlaps(p) for p in placed):
                placed.append(bb); break
            a.remove()
        else:
            placed.append(ax.annotate(str(s), (x, y), xytext=cands[0], textcoords="offset points", fontsize=fontsize, color=colour).get_window_extent(renderer=r))


def save_fig(fig, name: str, dpi: int = 400) -> None:  # 400 dpi keeps every panel above 300 dpi at the 170 mm print width
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=dpi)
    plt.close(fig)


def save_source(df: pd.DataFrame, name: str, panel: str) -> Path:
    d = FIGURES / "source_data"; d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}_{panel}.tsv"; df.to_csv(p, sep="\t", index=False); return p


def fmt_p(p: float) -> str:
    """P-value text that never prints 0.000."""
    if p != p:
        return "P = n/a"
    return "P < 0.001" if p < 0.001 else f"P = {p:.3f}"


def bold_axes(ax, x: str | None = None, y: str | None = None) -> None:
    if x is not None:
        ax.set_xlabel(x, fontweight="bold")
    if y is not None:
        ax.set_ylabel(y, fontweight="bold")


def box(ax, xy, w, h, text, fc="#f4f4f4", ec="#555555", fs=7, weight="normal", lw=0.8, r=0.02):
    """Rounded text box in axes coordinates (for schematics)."""
    from matplotlib.patches import FancyBboxPatch
    p = FancyBboxPatch(xy, w, h, boxstyle=f"round,pad=0.002,rounding_size={r}", fc=fc, ec=ec, lw=lw, transform=ax.transAxes, clip_on=False)
    ax.add_patch(p)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, fontweight=weight, transform=ax.transAxes, wrap=True)
    return p


def arrow(ax, p0, p1, color="#555555", lw=0.8, style="-|>"):
    from matplotlib.patches import FancyArrowPatch
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=7, color=color, lw=lw, transform=ax.transAxes, clip_on=False)
    ax.add_patch(a)
    return a


OUTCOME_NAME = {"E_seg": "E (outer band, primary readout)", "E_seg_noniche": "E (outer band without niche spots)", "E_seg_distal": "E (distal band, rings 3–5)"}
COVSET_NAME = {"none": "unadjusted", "3type": "3 fractions", "full": "11 RCTD types"}
OUTCOME_COL = {"E_seg": "#8C8C8C", "E_seg_noniche": "#4C72B0", "E_seg_distal": "#DD8452"}


def forest(ax, df: pd.DataFrame, label_col: str, color_col: str | None = None, p_col: str = "p_t_gminus1", perm_col: str | None = "p_perm_restricted",
           xlabel: str = "coverage coefficient (95 % CI, t with G − 1 d.f.)", label_fs: float = 5.5, group_col: str | None = None) -> None:
    """Horizontal forest plot of pooled coefficients: one row per record of `df` (columns coef, ci95_lo, ci95_hi)."""
    df = df.reset_index(drop=True); y = np.arange(len(df))[::-1]
    for i, r in df.iterrows():
        c = OUTCOME_COL.get(r[color_col], "#4C72B0") if color_col else "#4C72B0"
        ax.plot([r["ci95_lo"], r["ci95_hi"]], [y[i], y[i]], color=c, lw=1.1, solid_capstyle="butt")
        ax.scatter(r["coef"], y[i], s=16, color=c, edgecolor="black", linewidths=0.3, zorder=3)
        txt = fmt_p(r[p_col]).replace("P", "P") + (f"; perm {fmt_p(r[perm_col]).replace('P ', '')}" if perm_col and perm_col in df.columns else "")
        ax.text(1.01, y[i], txt, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=4.8)
    if group_col:
        prev = None
        for i, g in enumerate(df[group_col]):
            if g != prev and i > 0:
                ax.axhline(y[i] + 0.5, color="#dddddd", lw=0.5)
            prev = g
    ax.axvline(0, color="#888888", lw=0.6, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(df[label_col], fontsize=label_fs); ax.set_ylim(-0.7, len(df) - 0.3)
    bold_axes(ax, x=xlabel); ax.tick_params(axis="y", length=0)
