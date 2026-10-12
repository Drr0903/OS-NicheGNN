"""v2 Supplementary Figure S2 — region reference labels (algorithmic, no pathologist review) and their orthogonal checks, all 26 sections.
a  three sections (lung metastasis MR11B, bone primary PT2, soft-tissue metastasis LR17): H&E with the final label overlay
b  per-section AUC of the H&E air-space fraction (lung vs rest; lung sections) and of the atlas cancer-ICA weight (tumour vs paucicellular)
c  CNV signal vs correlation scatter coloured by final label (the two-component rule), all sections
d  lung marker score by label, lung sections only (in non-lung sections the lung-marker class is merged into paucicellular)
Reads results/04_histology/image_label_agreement.tsv and results/04b_panel_exports/* only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.plotting import REGION_COL, REGION_NAME, bold_axes, panel_label, save_fig, save_source, set_style  # noqa: E402
from lib.repro import ROOT, load_config  # noqa: E402

NAME = "FigS2"


def overlay(ax, section: str):
    """H&E with the final label overlay, read from the stage-04b export (no raw data needed)"""
    sd = ROOT / "results/04b_panel_exports"
    img = np.asarray(Image.open(sd / f"{section}_hires_crop.png"))
    d = pd.read_csv(sd / f"{section}_spots.tsv", sep="\t")
    ax.imshow(img, alpha=0.5); ax.axis("off")
    for k, c in REGION_COL.items():
        m = (d["label"] == k).to_numpy()
        if m.any():
            ax.scatter(d.loc[m, "x"], d.loc[m, "y"], s=3, c=c, linewidths=0, label=REGION_NAME.get(k, k))
    ax.set_title(section, fontsize=7)
    return d.assign(section=section)


def main() -> None:
    set_style(); cfg = load_config(); v2 = cfg["v2"]; lung_secs = [s for s, site in v2["site_of"].items() if site == "lung"]
    fig = plt.figure(figsize=(7.2, 7.6)); gs = fig.add_gridspec(3, 3, height_ratios=[0.85, 1, 1], hspace=0.62, wspace=0.4)
    src = []
    for i, s in enumerate(["MR11B", "PT2", "LR17"]):
        ax = fig.add_subplot(gs[0, i]); src.append(overlay(ax, s))
        if i == 0:
            pos = gs[0, 0].get_position(fig); fig.text(pos.x0, pos.y1 + 0.012, "a", fontweight="bold", fontsize=11, va="bottom")
            ax.legend(fontsize=5, loc="upper center", bbox_to_anchor=(1.72, -0.02), markerscale=3, ncol=3, frameon=False)
    save_source(pd.concat(src), NAME, "a")
    ag = pd.read_csv(ROOT / "results/04_histology/image_label_agreement.tsv", sep="\t"); ag = ag[ag["section"].isin(v2["sections"])].set_index("section").reindex(v2["sections"]).reset_index()
    ag.loc[~ag["section"].isin(lung_secs), "auc_airspace_lung_vs_rest"] = np.nan
    ax = fig.add_subplot(gs[1, :]); x = np.arange(len(ag))
    ax.bar(x - 0.2, ag["auc_airspace_lung_vs_rest"], 0.4, color=REGION_COL["lung_parenchyma"], label="H&E air-space fraction: lung vs rest (lung sections)", edgecolor="white", linewidth=0.3)
    ax.bar(x + 0.2, ag["auc_ica_cancer_tumor_vs_pauci"], 0.4, color=REGION_COL["tumor_cellular"], label="atlas cancer-ICA weight: tumour vs non-tumour", edgecolor="white", linewidth=0.3)
    ax.axhline(0.5, color="#888888", lw=0.6, ls="--"); ax.set_xticks(x); ax.set_xticklabels(ag["section"], rotation=90, ha="center", va="top", fontsize=5.4); ax.set_ylim(0, 1.05)
    bold_axes(ax, y="AUC"); ax.legend(fontsize=5.5, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False); panel_label(ax, "b"); save_source(ag, NAME, "b")
    lab = pd.read_csv(ROOT / "results/04b_panel_exports/label_features.tsv", sep="\t")  # the lung-marker class is already merged outside lung sections
    ax = fig.add_subplot(gs[2, 0:2]); d = lab.sample(frac=min(1.0, 25000 / len(lab)), random_state=0)
    for k, c in REGION_COL.items():
        m = d["label"] == k
        if m.any():
            ax.scatter(np.log(d.loc[m, "cnv_signal"] + 1e-6), d.loc[m, "cnv_corr"], s=2, c=c, alpha=0.4, linewidths=0, label=REGION_NAME.get(k, k))
    bold_axes(ax, x="log CNV signal", y="CNV correlation"); ax.set_title("all 26 sections", fontsize=7, pad=14); panel_label(ax, "c"); ax.legend(fontsize=5, markerscale=4, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False, borderaxespad=0.1)
    ax = fig.add_subplot(gs[2, 2]); d = lab[lab["section"].isin(lung_secs)]
    for k, c in REGION_COL.items():
        m = d["label"] == k
        if m.any():
            ax.hist(d.loc[m, "score_lung"], bins=60, range=(0, 3), color=c, alpha=0.6, label=k.replace("_", " "), density=True)
    bold_axes(ax, x="lung marker score\n(lung sections)", y="density"); panel_label(ax, "d")
    save_source(lab.sample(frac=min(1.0, 30000 / len(lab)), random_state=0), NAME, "cd")
    save_fig(fig, f"{NAME}_label_qc")


if __name__ == "__main__":
    main()
