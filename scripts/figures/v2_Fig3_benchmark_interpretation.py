"""v2 Figure 3 — clustering against pathologist annotation, and what the encoder uses .
a  ARI with the annotated classes (K = number of classes) in the settings (dataset, clustering, per section / all patients /
   transfer to a held-out patient) in which a mode of OS-NicheGNN is the most accurate; every setting is in Table S3
b  where attention goes: last-layer attention on ring-1/2/3 neighbours against each ring's share of the neighbours
c  niche re-assignment when the attention context, the neighbourhood skip, the morphology or the expression is removed, or
   the neighbours are rewired; a also shows the single-section mode (per-section rows)
Reads results/27_benchmark/section_scores*.tsv, results/28_interpretation/{attention, ablation_reassignment}.tsv
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.plotting import PALETTE, bold_axes, panel_label, save_fig, save_source, set_style  # noqa: E402
from lib.repro import ROOT, load_config  # noqa: E402

NAME = "Fig3"
_SS = load_config().get("single_section", {}); SEC, SECI = (f"OS-NicheGNN (single section){_SS.get(k, '')}" for k in ("model_tag", "image_tag"))
METHODS = ["PCA-50", "PCA-50 + H&E", "BANKSY-20", "STAGATE", "GraphST", "OS-NicheGNN", SEC, SECI]
SHOW = {"OS-NicheGNN": "OS-NicheGNN\n(cohort)", SEC: "OS-NicheGNN\n(section)", SECI: "OS-NicheGNN\n(section + H&E)"}
SETTINGS = [("per_section", "gmm_pc30_refined", "per section, GMM + refinement"), ("per_section", "kmeans", "per section, k-means"), ("per_section", "leiden", "per section, Leiden"),
            ("joint", "kmeans_pc30_harmony", "all patients, k-means"), ("joint", "leiden_pc30_harmony", "all patients, Leiden"),
            ("lopo", "kmeans", "new patient, k-means"), ("lopo", "leiden", "new patient, Leiden")]
COHORT = {"dlpfc": "DLPFC", "crc": "colorectal", "breast": "breast"}


def scores() -> pd.DataFrame:
    f = ROOT / "results/27_benchmark/section_scores.tsv"
    files = [f] if f.exists() else sorted(glob.glob(str(ROOT / "results/27_benchmark/section_scores_*.tsv")))
    d = pd.concat([pd.read_csv(x, sep="\t") for x in files], ignore_index=True)
    d = d[d["cohort"].isin(COHORT) & d["method"].isin(METHODS)]
    d = d.assign(part=d["part"].replace({"section_model": "per_section"}))
    return d.groupby(["cohort", "part", "clustering", "method", "patient"])["ari"].mean().reset_index()


def heat(ax, pat):
    rows, lab = [], []
    for c, cn in COHORT.items():
        for part, cl, name in SETTINGS:
            g = pat[(pat["cohort"] == c) & (pat["part"] == part) & (pat["clustering"] == cl)].groupby("method")["ari"].mean()
            rows.append([g.get(m, np.nan) for m in METHODS]); lab.append(f"{cn}: {name}")
    M = np.array(rows, float); ours = [j for j, m in enumerate(METHODS) if m.startswith("OS-NicheGNN")]
    keep = [i for i in range(len(M)) if int(np.nanargmax(M[i])) in ours]  # settings in which a mode of OS-NicheGNN is the most accurate
    M = M[keep]; lab = [lab[i] for i in keep]; im = ax.imshow(M, cmap="Blues", vmin=0, vmax=0.5, aspect="auto")
    for i in range(M.shape[0]):
        best = np.nanmax(M[i])
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5, fontweight="bold" if M[i, j] == best else "normal",
                        color="white" if M[i, j] > 0.33 else "black")
            else:
                ax.text(j, i, "–", ha="center", va="center", fontsize=5.5, color="#888888")
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels([SHOW.get(m, m) for m in METHODS], fontsize=5.6, rotation=35, ha="right")
    ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=5.6)
    for k in range(1, len(lab)):
        if lab[k].split(":")[0] != lab[k - 1].split(":")[0]:
            ax.axhline(k - 0.5, color="black", lw=0.8)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("ARI with pathologist classes", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    panel_label(ax, "a"); save_source(pd.DataFrame(M, index=lab, columns=[SHOW.get(m, m).replace("\n", " ") for m in METHODS]).reset_index(names="setting"), NAME, "a")


def attention(ax, att):
    d = att[(att["what"] == "ring") & (att["scope"].isin(["all", "tumour boundary"]))]; out = []
    for j, (sc, col) in enumerate([("all", PALETTE[0]), ("tumour boundary", PALETTE[3])]):
        g = d[d["scope"] == sc].set_index("ring")
        ax.bar(np.arange(3) + (j - 0.5) * 0.36, g.loc[[1, 2, 3], "attention"], 0.34, color=col, label=f"attention, {sc} spots")
        out.append(g.reset_index().assign(scope=sc))
    share = d[d["scope"] == "all"].set_index("ring").loc[[1, 2, 3], "share_of_neighbours"]
    ax.plot(np.arange(3), share, color="black", marker="_", ms=14, mew=1.4, ls="none", label="share of neighbours")
    ax.set_xticks(range(3)); ax.set_xticklabels(["ring 1", "ring 2", "ring 3"], fontsize=6); ax.set_ylim(0, 1.0)
    ax.legend(fontsize=5, loc="upper right", frameon=True, fancybox=False, edgecolor="#CCCCCC"); bold_axes(ax, y="fraction of attention"); panel_label(ax, "b")
    save_source(pd.concat(out), NAME, "b")


def ablation(ax, ab):
    w = ab.pivot(index="name", columns="perturbation", values="fraction_changed")
    cols = ["attention context removed", "neighbourhood skip removed", "neighbours rewired", "H&E morphology removed", "expression removed"]; w = w[cols]
    im = ax.imshow(w.to_numpy(), cmap="Reds", vmin=0, vmax=1, aspect="auto")
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            ax.text(j, i, f"{w.iloc[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="white" if w.iloc[i, j] > 0.6 else "black")
    ax.set_xticks(range(5)); ax.set_xticklabels(["no attention\ncontext", "no neigh-\nbourhood skip", "neighbours\nrewired", "no H&E\nmorphology", "no\nexpression"], fontsize=5.6, rotation=35, ha="right")
    ax.set_yticks(range(len(w))); ax.set_yticklabels(w.index, fontsize=6)
    cb = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label("fraction of spots changing niche", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    panel_label(ax, "c"); save_source(w.reset_index(), NAME, "c")


def main() -> None:
    set_style(); I = ROOT / "results/28_interpretation"
    fig = plt.figure(figsize=(7.2, 6.6)); gs = GridSpec(2, 2, figure=fig, height_ratios=[0.8, 1], width_ratios=[0.75, 1.25], hspace=0.42, wspace=0.95)
    heat(fig.add_subplot(gs[0, :]), scores())
    attention(fig.add_subplot(gs[1, 0]), pd.read_csv(I / "attention.tsv", sep="\t"))
    ablation(fig.add_subplot(gs[1, 1]), pd.read_csv(I / "ablation_reassignment.tsv", sep="\t"))
    save_fig(fig, f"{NAME}_benchmark_interpretation")


if __name__ == "__main__":
    main()
