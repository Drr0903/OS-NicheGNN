"""v2 Figure 4 — metastatic niche atlas across sites.
a  niche maps of one section per site (bone primary, lung, lymph node, heart, soft tissue)
b  RCTD composition per niche (heat-map)
c  region-label composition per niche
d  niche fraction per section grouped by site, with the shared / site-specific / primary-predominant class
e  top patient-level marker genes per niche (log2FC vs rest, n patients agreeing)
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
from lib.plotting import PALETTE, REGION_COL, REGION_NAME, bold_axes, panel_label, save_fig, save_source, set_style  # noqa: E402
from lib.repro import ROOT  # noqa: E402

NAME = "Fig4"; SITE_ORDER = ["bone", "lung", "lymph_node", "heart", "soft_tissue"]
SITE_COL = {"bone": PALETTE[5], "lung": PALETTE[0], "lymph_node": PALETTE[4], "heart": PALETTE[3], "soft_tissue": PALETTE[2]}
SITE_LAB = {"bone": "bone primaries", "lung": "lung", "lymph_node": "lymph node", "heart": "heart", "soft_tissue": "soft tissue"}
SHOW = ["PT3", "MR3", "MR14", "MR16", "LR17"]; TYPE_LAB = {"Alveolar_macrophage": "alv. macrophage", "B_plasma": "B/plasma", "Endothelial": "endothelial", "Fibroblast": "fibroblast", "Lung_epithelium": "lung epithelium", "Mast": "mast", "Myeloid": "myeloid", "Osteoclast": "osteoclast", "Osteosarcoma": "osteosarcoma", "Pericyte_SMC": "pericyte/SMC", "T_NK": "T/NK"}


def main() -> None:
    set_style(); N = ROOT / "results/15_niches"; spots = pd.read_csv(N / "spot_niches.tsv", sep="\t"); meta = json.load(open(N / "meta.json")); names = {int(k): v for k, v in meta["names"].items()}; K = meta["K"]; inter = meta["interface_niche"]
    cols = {k: PALETTE[k % len(PALETTE)] for k in range(K)}; comp = pd.read_csv(N / "niche_composition_rctd.tsv", sep="\t").set_index("niche"); hist = pd.read_csv(N / "niche_histology.tsv", sep="\t").set_index("niche")
    by_sec = pd.read_csv(N / "niche_by_section.tsv", sep="\t"); cls = pd.read_csv(N / "niche_classification.tsv", sep="\t").set_index("niche"); mk = pd.read_csv(N / "niche_markers.tsv", sep="\t")
    order = [s_ for st in SITE_ORDER for s_ in sorted(by_sec[by_sec["site"] == st]["section"])]
    fig = plt.figure(figsize=(7.2, 10.1))  # bands placed explicitly: only the rows with rotated tick labels get a deep gutter
    kw = dict(left=0.155, right=0.965)
    ga = fig.add_gridspec(1, 1, top=0.985, bottom=0.600, left=0.02, right=0.99); gb = fig.add_gridspec(1, 5, top=0.573, bottom=0.440, wspace=0.55, **kw)
    gd = fig.add_gridspec(1, 1, top=0.338, bottom=0.238, **kw); ge = fig.add_gridspec(1, 1, top=0.175, bottom=0.075, **kw)
    sub = ga[0, 0].subgridspec(6, 5, wspace=0.04, hspace=0.30)  # every section of the cohort, not a selection
    ax = None
    for i, s_ in enumerate(order):
        ax = fig.add_subplot(sub[i // 5, i % 5]); d = spots[spots["section"] == s_]
        ax.scatter(d["x"], -d["y"], s=0.5, c="#dedede", linewidths=0)
        for k in range(K):
            m = (d["niche"] == k) & (~d["unmatched"].astype(bool)); ax.scatter(d.loc[m, "x"], -d.loc[m, "y"], s=0.5, c=cols[k], linewidths=0, label=names[k] if i == 0 else None)
        ax.set_aspect("equal"); ax.axis("off")
        ax.text(0.5, 1.01, s_, ha="center", va="bottom", fontsize=5.6, color=SITE_COL[d["site"].iloc[0]], fontweight="bold", transform=ax.transAxes)
        if i == 0:
            panel_label(ax, "a")  # placed with the other letters (left column, above the first map)
    from matplotlib.lines import Line2D
    axl = fig.add_subplot(sub[5, 1:5]); axl.axis("off")  # the two cells the 26 sections leave free carry the key
    hs = [Line2D([], [], marker="o", ls="", ms=3.2, color=cols[k]) for k in range(K)] + [Line2D([], [], marker="o", ls="", ms=3.2, color="#dedede")]
    ls_ = [f"{k}: {names[k]}" for k in range(K)] + ["unmatched"]
    hs += [Line2D([], [], marker="s", ls="", ms=3.2, color=SITE_COL[st]) for st in SITE_ORDER]
    ls_ += [SITE_LAB[st] for st in SITE_ORDER]
    axl.legend(hs, ls_, loc="upper left", bbox_to_anchor=(-0.02, 1.25), ncol=3, fontsize=5.0, frameon=False, handletextpad=0.35, labelspacing=0.3, columnspacing=1.0)
    save_source(spots[["section", "site", "patient", "barcode", "x", "y", "niche", "unmatched"]], NAME, "a")
    ax = fig.add_subplot(gb[0, :3]); M = comp[[c for c in comp.columns if c in TYPE_LAB]]; im = ax.imshow(M.to_numpy(), cmap="Blues", aspect="auto", vmin=0, vmax=max(0.5, float(M.to_numpy().max())))
    ax.set_yticks(range(len(M))); ax.set_yticklabels([names[int(k)] for k in M.index], fontsize=6); ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([TYPE_LAB[c] for c in M.columns], rotation=90, ha="center", va="top", fontsize=5.6)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M.iloc[i, j] >= 0.1:
                ax.text(j, i, f"{M.iloc[i, j]:.2f}", ha="center", va="center", fontsize=5, color="white" if M.iloc[i, j] > 0.35 else "black")
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.ax.tick_params(labelsize=5.5); cb.set_label("RCTD weight", fontsize=6); panel_label(ax, "b"); save_source(comp.reset_index(), NAME, "b")
    ax = fig.add_subplot(gb[0, 3:]); left = np.zeros(len(hist))
    for lab in ["tumor_cellular", "lung_parenchyma", "paucicellular"]:
        if lab in hist.columns:
            ax.barh(range(len(hist)), hist[lab], left=left, color=REGION_COL[lab], edgecolor="white", linewidth=0.3, label=REGION_NAME[lab]); left += hist[lab].to_numpy()
    ax.set_yticks(range(len(hist))); ax.set_yticklabels([str(k) for k in hist.index], fontsize=6); ax.invert_yaxis(); ax.yaxis.tick_right(); ax.tick_params(axis="y", length=0); ax.set_xlim(0, 1); bold_axes(ax, x="fraction of niche spots"); ax.legend(fontsize=5, loc="upper right", bbox_to_anchor=(1.02, -0.36), ncol=2, frameon=False, handlelength=1.0, columnspacing=1.0, handletextpad=0.4); panel_label(ax, "c"); save_source(hist.reset_index(), NAME, "c")  # panel c reads top to bottom in the same order as panel b
    ax = fig.add_subplot(gd[0, 0]); site_order = ["bone", "lung", "lymph_node", "heart", "soft_tissue"]; bs = by_sec.copy(); bs["site_rank"] = bs["site"].map({s: i for i, s in enumerate(site_order)}); bs = bs.sort_values(["site_rank", "section"]).reset_index(drop=True); bottom = np.zeros(len(bs))
    for k in range(K):
        col = str(k) if str(k) in bs.columns else k; ax.bar(range(len(bs)), bs[col], bottom=bottom, color=cols[k], edgecolor="white", linewidth=0.3); bottom += bs[col].to_numpy()
    ax.set_xticks(range(len(bs))); ax.set_xticklabels(bs["section"], rotation=90, ha="center", va="top", fontsize=5.2); bold_axes(ax, y="fraction of spots")
    for site in site_order:
        idx = np.where(bs["site"] == site)[0]
        if len(idx):
            ax.text(idx.mean(), 1.03, {"lymph_node": "LN", "soft_tissue": "soft tissue"}.get(site, site), ha="center", fontsize=5.5); ax.axvline(idx.max() + 0.5, color="#bbbbbb", lw=0.5)
    ax.set_ylim(0, 1.12); panel_label(ax, "d"); save_source(bs.drop(columns="site_rank"), NAME, "d")
    ax = fig.add_subplot(ge[0, 0]); rows = []; xt = []; x = 0
    for k in range(K):
        top = mk[(mk["niche"] == k) & (mk["fdr"] < 0.1) & (mk["log2fc_mean"] > 0)].sort_values("log2fc_mean", ascending=False).head(6)
        for _, r in top.iterrows():
            ax.bar(x, r["log2fc_mean"], color=cols[k], width=0.8); ax.text(x, r["log2fc_mean"] + 0.04, f"{int(r['n_patients_up'])}/{int(r['n_patients'])}", ha="center", va="bottom", rotation=90, fontsize=4.2); xt.append((x, r["gene"])); rows.append({"niche": k, **r.to_dict()}); x += 1
        x += 1
    ax.set_xticks([p for p, _ in xt]); ax.set_xticklabels([g for _, g in xt], rotation=90, fontsize=5); bold_axes(ax, y="log2FC niche vs rest\n(per patient)"); ax.set_ylim(min(0, ax.get_ylim()[0]), ax.get_ylim()[1] * 1.25); panel_label(ax, "e"); save_source(pd.DataFrame(rows), NAME, "e")
    save_fig(fig, f"{NAME}_niche_atlas")


if __name__ == "__main__":
    main()
