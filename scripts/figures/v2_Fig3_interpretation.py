"""v2 Figure 3 — how OS-NicheGNN reads the tissue.
a  router weight of the host-tissue expert on one section per site (PT2 bone, MR2 lung, MR14 lymph node, LR17 soft tissue),
   with the H&E of each section
b  router weight of the host-tissue expert per niche (patient-level mean ± s.d.)
c  attention of two MR2 spots on their 36 neighbours: a peritumoural-band spot on the tumour–lung interface and a tumour spot
   far from it (last layer, heads averaged, mean over the five networks)
d  last-layer attention on ring-1/2/3 neighbours against each ring's share of the neighbours
e  gradient attribution of the most attributed genes of each niche, relative to the niche's top gene
Reads results/36_interpretation_maps/, results/34_expert_routing/frozen_by_niche_name_patient.tsv,
results/28_interpretation/{attention, gene_attribution}.tsv, results/15_niches/niche_names.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.plotting import PALETTE, REGION_COL, bold_axes, panel_label, save_fig, save_source, set_style  # noqa: E402
from lib.repro import ROOT  # noqa: E402

NAME = "Fig3"
M = ROOT / "results/36_interpretation_maps"
SITES = [("PT2", "bone primary (PT2)"), ("MR2", "lung (MR2)"), ("MR14", "lymph node (MR14)"), ("LR17", "soft tissue (LR17)")]
NAMES = {int(k): v for k, v in json.load(open(ROOT / "results/15_niches/niche_names.json")).items()}


def router_maps(fig, gs) -> None:
    """a: H&E (top) and host-tissue router weight (bottom) for one section per site."""
    sub = gs.subgridspec(2, 4, hspace=0.06, wspace=0.03); src = []
    for k, (sec, title) in enumerate(SITES):
        img = np.asarray(Image.open(M / f"{sec}_hires_crop.png").convert("RGB")); d = pd.read_csv(M / f"{sec}_spots.tsv", sep="\t").dropna(subset=["host_expert_weight"])
        a1, a2 = fig.add_subplot(sub[0, k]), fig.add_subplot(sub[1, k])
        a1.imshow(img); a1.set_title(title, fontsize=6.5); a2.imshow(img, alpha=0.2)
        sc = a2.scatter(d["x"], d["y"], c=d["host_expert_weight"], cmap="coolwarm_r", vmin=0, vmax=1, s=0.6, linewidths=0)
        for a in (a1, a2):
            a.set_xticks([]); a.set_yticks([]); [s.set_visible(False) for s in a.spines.values()]
        if k == 0:
            panel_label(a1, "a"); a1.set_ylabel("H&E", fontsize=6.5); a2.set_ylabel("router weight", fontsize=6.5)
        src.append(d[["barcode", "x", "y", "niche", "histology_label", "host_expert_weight"]].assign(section=sec))
    cax = fig.add_axes([0.80, a2.get_position().y0 - 0.022, 0.15, 0.008]); cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1]); cb.set_ticklabels(["tumour expert", "host-tissue expert"]); cb.ax.tick_params(labelsize=5.5, length=0)
    save_source(pd.concat(src), NAME, "a")


def routing(ax) -> None:
    """b: host-tissue expert weight per niche (patients as points, mean ± s.d.), niches ordered by the weight."""
    pt = pd.read_csv(ROOT / "results/34_expert_routing/frozen_by_niche_name_patient.tsv", sep="\t"); idx = {v: k for k, v in NAMES.items()}
    g = pt.groupby("niche_name")["gate_0"].agg(["mean", "std"]).sort_values("mean"); y = np.arange(len(g))
    ax.barh(y, g["mean"], xerr=g["std"], color=[PALETTE[idx[n] % len(PALETTE)] for n in g.index], edgecolor="black", lw=0.3, error_kw={"lw": 0.6, "capsize": 1.5})
    ax.axvline(0.5, color="#888888", lw=0.6, ls="--"); ax.set_yticks(y); ax.set_yticklabels(g.index, fontsize=5.5); ax.set_xlim(0, 1)
    ax.text(0.02, len(g) - 0.2, "tumour\nexpert", fontsize=5.5, ha="left", va="bottom"); ax.text(0.98, len(g) - 0.2, "host-tissue\nexpert", fontsize=5.5, ha="right", va="bottom")
    ax.set_ylim(-0.6, len(g) + 0.9); bold_axes(ax, x="router weight of the host-tissue expert"); panel_label(ax, "b"); save_source(g.reset_index(), NAME, "b")


def attention_example(axs) -> None:
    """c: each neighbour of the two example spots on the H&E, with a line whose width and colour give its attention weight."""
    img = np.asarray(Image.open(M / "MR2_hires_crop.png").convert("RGB")); d = pd.read_csv(M / "MR2_spots.tsv", sep="\t").set_index("barcode")
    a = pd.read_csv(M / "attention_example.tsv", sep="\t"); xy = d[["x", "y"]].to_numpy()
    pitch = np.median(np.sort(np.sqrt(((xy[:400, None] - xy[None]) ** 2).sum(-1)), 1)[:, 1]); out = []
    for ax, (ex, title) in zip(axs, [("interface band spot", "band spot at the interface"), ("tumour spot", "tumour spot")]):
        g = a[a["example"] == ex]; c = d.loc[g["centre_barcode"].iloc[0]]; nb = d.loc[g["neighbour_barcode"]]; r = 4.3 * pitch
        ax.imshow(img); ax.set_xlim(c["x"] - r, c["x"] + r); ax.set_ylim(c["y"] + r, c["y"] - r)
        w = g["alpha_mean"].to_numpy(); order = np.argsort(w)
        for i in order:
            ax.plot([c["x"], nb["x"].iloc[i]], [c["y"], nb["y"].iloc[i]], color=plt.cm.Purples(0.35 + 0.65 * w[i] / w.max()), lw=0.3 + 3.2 * w[i] / w.max(), zorder=3, solid_capstyle="round")
        ax.scatter(nb["x"], nb["y"], s=16, c=[REGION_COL.get(v, "#bbbbbb") for v in nb["histology_label"]], edgecolors="white", linewidths=0.4, zorder=4)
        ax.scatter([c["x"]], [c["y"]], s=34, marker="*", c="black", zorder=5)
        tum = float(g.loc[nb["histology_label"].to_numpy() == "tumor_cellular", "alpha_mean"].sum())
        ax.set_title(f"{title}\n{tum * 100:.0f} % of attention on tumour spots", fontsize=6.0)
        ax.set_xticks([]); ax.set_yticks([]); out.append(g.assign(neighbour_label=nb["histology_label"].to_numpy(), x=nb["x"].to_numpy(), y=nb["y"].to_numpy()))
    axs[1].scatter([], [], s=12, c=REGION_COL["tumor_cellular"], label="tumour"); axs[1].scatter([], [], s=12, c=REGION_COL["lung_parenchyma"], label="lung parenchyma")
    axs[1].legend(fontsize=5.5, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, handletextpad=0.2, borderaxespad=0)
    panel_label(axs[0], "c"); save_source(pd.concat(out), NAME, "c")


def attention_rings(ax, att) -> None:
    d = att[(att["what"] == "ring") & (att["scope"].isin(["all", "tumour boundary"]))]; out = []
    for j, (sc, col) in enumerate([("all", PALETTE[0]), ("tumour boundary", PALETTE[3])]):
        g = d[d["scope"] == sc].set_index("ring")
        ax.bar(np.arange(3) + (j - 0.5) * 0.36, g.loc[[1, 2, 3], "attention"], 0.34, color=col, label="all spots" if sc == "all" else "boundary spots")
        out.append(g.reset_index().assign(scope=sc))
    share = d[d["scope"] == "all"].set_index("ring").loc[[1, 2, 3], "share_of_neighbours"]
    ax.plot(np.arange(3), share, color="black", marker="_", ms=14, mew=1.4, ls="none", label="share of\nneighbours")
    ax.set_xticks(range(3)); ax.set_xticklabels(["ring 1", "ring 2", "ring 3"], fontsize=6); ax.set_ylim(0, 1.32); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(fontsize=5.3, loc="upper right", bbox_to_anchor=(1.04, 1.02), frameon=False, handlelength=1.0, borderaxespad=0.1, labelspacing=0.3); bold_axes(ax, y="fraction of attention"); panel_label(ax, "d")
    save_source(pd.concat(out), NAME, "d")


def attribution(ax, ga, top: int = 3) -> None:
    """e: genes × niches; each niche's top genes, attribution relative to its top gene (blank: outside the niche's 30 most attributed genes)."""
    order = [k for k in NAMES]; genes = []
    for k in order:
        for gname in ga[(ga["niche"] == k) & (ga["rank"] <= top)].sort_values("rank")["gene"]:
            if gname not in genes:
                genes.append(gname)
    rel = ga.assign(rel=ga["attribution"] / ga.groupby("niche")["attribution"].transform("max"))
    W = rel.pivot_table(index="gene", columns="niche", values="rel").reindex(index=genes, columns=order)
    im = ax.imshow(W.T.to_numpy(), cmap="Purples", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(genes))); ax.set_xticklabels(genes, fontsize=5.5, rotation=60, ha="right", fontstyle="italic")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([NAMES[k] for k in order], fontsize=5.5)
    ax.set_xlim(-0.5, len(genes) - 0.5)
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.015); cb.set_label("attribution\n(relative)", fontsize=5.5); cb.ax.tick_params(labelsize=5.5)
    panel_label(ax, "e"); save_source(W.reset_index(), NAME, "e")


def main() -> None:
    set_style(); I = ROOT / "results/28_interpretation"
    fig = plt.figure(figsize=(7.2, 9.2)); gs = GridSpec(3, 1, figure=fig, height_ratios=[1.45, 0.95, 0.85], hspace=0.34)
    router_maps(fig, gs[0])
    mid = gs[1].subgridspec(1, 3, width_ratios=[1.0, 0.55, 0.55], wspace=0.12)
    routing(fig.add_subplot(mid[0])); attention_example([fig.add_subplot(mid[1]), fig.add_subplot(mid[2])])
    bot = gs[2].subgridspec(1, 2, width_ratios=[0.34, 1.0], wspace=1.05)
    attention_rings(fig.add_subplot(bot[0]), pd.read_csv(I / "attention.tsv", sep="\t")); attribution(fig.add_subplot(bot[1]), pd.read_csv(I / "gene_attribution.tsv", sep="\t"))
    save_fig(fig, f"{NAME}_interpretation")


if __name__ == "__main__":
    main()
