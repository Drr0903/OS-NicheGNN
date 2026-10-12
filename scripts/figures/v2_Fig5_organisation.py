"""v2 Figure 5 — spatial organisation of the niches.
a  two representative tumour–lung interfaces (MR2, MR18): H&E, the peritumoural band with interface spots outlined, and the
   per-spot macrophage score, T-cell score and router weight of the host-tissue expert (stage 35)
b  niche–niche adjacency enrichment (patient-level mean z of ring-1 contacts vs within-section permutation null)
c  patient-level T-cell score per niche
d  boundary enrichment (odds ratio within one ring of the tumour boundary) per niche
e  lung sections: pooled within-section coefficients of band coverage of the tumour–lung interface on the outer-to-inner
   macrophage and T-cell ratios (beyond the band, with the band, distal control), unadjusted and adjusted for eleven cell types
f  across the interface (signed ring distance, negative inside the tumour): T-cell score and the router weight of the host-tissue
   expert, patient-level means ± s.e.m.
Reads results/15_niches, results/30_osteoclast_myeloid, results/30b_macrophage_outside_input, results/18_signalling, results/34_expert_routing, results/35_representative_interface
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


MAPS = [("H&E", None), ("peritumoural band", "band"), ("macrophage score", "macrophage_score"), ("T-cell score", "t_score"), ("host-tissue expert", "host_expert_weight")]


def maps(fig, cell, band: int, cols: dict) -> None:
    """a: two representative tumour–lung interfaces (stage 35): H&E, the band (interface spots outlined) and per-spot scores,
    each score shown between its 2nd and 98th percentile in the section (colour bars: low to high)."""
    from PIL import Image
    R = ROOT / "results/35_representative_interface"; secs = ["MR2", "MR18"]
    sub = cell.subgridspec(3, len(MAPS), hspace=0.06, wspace=0.04, height_ratios=[1, 1, 0.06]); first = None; out = []; sc = {}
    CMAP = {"macrophage_score": "viridis", "t_score": "viridis", "host_expert_weight": "coolwarm"}
    for i, sec in enumerate(secs):
        img = np.asarray(Image.open(R / f"{sec}_hires_crop.png")); d = pd.read_csv(R / f"{sec}_spots.tsv", sep="\t"); out.append(d.assign(section=sec))
        for j, (title, col) in enumerate(MAPS):
            ax = fig.add_subplot(sub[i, j]); ax.axis("off"); first = first or ax
            if col is None:
                ax.imshow(img)
            elif col == "band":
                ax.imshow(img, alpha=0.35); ax.scatter(d["x"], d["y"], s=0.6, c="#d9d9d9", linewidths=0)
                b = d["band"].astype(bool); ax.scatter(d.loc[b, "x"], d.loc[b, "y"], s=0.9, c=cols[band], linewidths=0)
                f = d["tumor_lung_interface"].fillna(False).astype(bool); ax.scatter(d.loc[f, "x"], d.loc[f, "y"], s=1.4, facecolors="none", edgecolors="black", linewidths=0.2)
            else:
                v = d[col]; lo, hi = (0.0, 1.0) if col == "host_expert_weight" else np.nanpercentile(v, [2, 98]); ax.imshow(img, alpha=0.2)
                sc[col] = ax.scatter(d["x"], d["y"], s=0.7, c=v.clip(lo, hi), cmap=CMAP[col], vmin=lo, vmax=hi, linewidths=0)
            ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0)
            if i == 0:
                ax.set_title(title, fontsize=6)
            if j == 0:
                ax.text(-0.04, 0.5, sec, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=6, fontweight="bold")
    for j, (title, col) in enumerate(MAPS):
        if col in sc:
            cax = fig.add_subplot(sub[2, j]); cb = fig.colorbar(sc[col], cax=cax, orientation="horizontal"); cb.set_ticks([])
            lab = ("tumour", "host tissue") if col == "host_expert_weight" else ("low", "high")
            cax.text(0, -1.2, lab[0], transform=cax.transAxes, ha="left", va="top", fontsize=5); cax.text(1, -1.2, lab[1], transform=cax.transAxes, ha="right", va="top", fontsize=5)
        elif col == "band":
            ax = fig.add_subplot(sub[2, j]); ax.axis("off"); ax.scatter([0.05], [0.5], s=8, c=cols[band]); ax.text(0.12, 0.5, "band", fontsize=5, va="center")
            ax.scatter([0.5], [0.5], s=8, facecolors="none", edgecolors="black", linewidths=0.4); ax.text(0.57, 0.5, "interface", fontsize=5, va="center"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    panel_label(first, "a"); save_source(pd.concat(out, ignore_index=True), NAME, "a")


def main() -> None:
    set_style(); N = ROOT / "results/15_niches"; meta = json.load(open(N / "meta.json")); names = {int(k): v for k, v in meta["names"].items()}; K = meta["K"]; inter = meta["interface_niche"]; cols = {k: PALETTE[k % len(PALETTE)] for k in range(K)}
    adj = pd.read_csv(N / "niche_adjacency_patient_level.tsv", sep="\t"); ts = pd.read_csv(N / "niche_tcell_patient_level.tsv", sep="\t"); enr = pd.read_csv(N / "niche_boundary_enrichment.tsv", sep="\t")
    sec = pd.read_csv(N / "interface_section_metrics.tsv", sep="\t"); pm = pd.read_csv(N / "interface_pooled_models.tsv", sep="\t")
    fig = plt.figure(figsize=(7.2, 10.6)); top = fig.add_gridspec(1, 1, left=0.05, right=0.975, top=0.975, bottom=0.715)
    maps(fig, top[0], inter, cols)
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=1.35, width_ratios=[1.3, 1, 1.1], height_ratios=[1.0, 1.45, 0.34], left=0.135, right=0.975, top=0.655, bottom=0.05)
    ax = fig.add_subplot(gs[0, 0]); M = np.full((K, K), np.nan)
    for _, r in adj.iterrows():
        M[int(r["niche_a"]), int(r["niche_b"])] = r["log2_ratio_mean"]; M[int(r["niche_b"]), int(r["niche_a"])] = r["log2_ratio_mean"]
    v = np.nanmax(np.abs(M)); im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto"); ax.set_xticks(range(K)); ax.set_yticks(range(K)); ax.set_xticklabels([str(k) for k in range(K)], fontsize=6); ax.set_yticklabels([names[k] for k in range(K)], fontsize=5.5)  # aspect="auto" fills the cell, so the panel label stays level with b and c
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02); cb.ax.tick_params(labelsize=5.5); cb.set_label("log$_2$ observed/expected contacts", fontsize=6); panel_label(ax, "b"); save_source(adj, NAME, "b")
    ax = fig.add_subplot(gs[0, 1]); ts = ts.sort_values("niche"); ax.bar(ts["niche"], ts["mean"], yerr=ts["std"], color=[cols[int(k)] for k in ts["niche"]], capsize=2, edgecolor="black", linewidth=0.3)
    ax.set_xticks(ts["niche"]); ax.set_xticklabels([str(int(k)) for k in ts["niche"]], fontsize=6); bold_axes(ax, x="niche", y="T-cell score (patient mean)"); panel_label(ax, "c"); save_source(ts, NAME, "c")
    ax = fig.add_subplot(gs[0, 2]); ax.bar(enr["niche"], enr["odds_ratio_near_boundary"], color=[cols[int(k)] for k in enr["niche"]], edgecolor=["red" if b else "black" for b in enr["is_interface_niche"]], linewidth=[1.2 if b else 0.3 for b in enr["is_interface_niche"]])
    ax.axhline(1, color="#888888", lw=0.6, ls="--"); ax.set_xticks(enr["niche"]); ax.set_xticklabels([str(int(k)) for k in enr["niche"]], fontsize=6); bold_axes(ax, x="niche", y="odds ratio, within 1 ring\nof tumour boundary"); ax.set_ylim(0, ax.get_ylim()[1] * 1.18); panel_label(ax, "d"); save_source(enr, NAME, "d")
    mac = pd.read_csv(ROOT / "results/30_osteoclast_myeloid/interface_myeloid_pooled_models.tsv", sep="\t").set_index(["outcome", "covariate_set"]); pmi = pm.set_index(["outcome", "covariate_set"])
    mac_out = pd.read_csv(ROOT / "results/30b_macrophage_outside_input/interface_pooled_models.tsv", sep="\t").set_index(["outcome", "covariate_set"])
    LAB = {("mac", "E_seg"): "macrophages, outer/inner", ("macout", "E_seg"): "macrophage genes outside\nthe network input", ("T", "E_seg_noniche"): "T cells, outer/inner,\nbeyond the band", ("T", "E_seg"): "T cells, outer/inner,\nincluding the band",
           ("T", "E_seg_distal"): "T cells, distal\nrings 3–5 (control)"}
    rows = []
    for (what, o), lab in LAB.items():
        src = mac if what == "mac" else mac_out if what == "macout" else pmi
        for c in ["none", "full"]:
            if (o, c) in src.index and src.loc[(o, c), "coef"] == src.loc[(o, c), "coef"]:
                r = src.loc[(o, c)]; rows.append({**{k: r[k] for k in ["coef", "ci95_lo", "ci95_hi", "p_t_gminus1", "p_perm_restricted", "n_segments", "n_sections"]},
                                                  "readout": f"{what}:{o}", "covariate_set": c, "label": f"{lab}, {COVSET_NAME[c]}"})
    ft = pd.DataFrame(rows); ft["grp"] = ft["readout"]
    ax = fig.add_subplot(gs[1, :2]); forest(ax, ft, "label", label_fs=5.2, group_col="grp", xlabel="coefficient of band coverage (95 % CI, t with G − 1 d.f.)")
    for i, r in ft.reset_index(drop=True).iterrows():  # macrophage rows in a distinct colour
        if r["readout"].startswith("mac"):
            y = len(ft) - 1 - i; ax.plot([r["ci95_lo"], r["ci95_hi"]], [y, y], color="#C44E52", lw=1.1); ax.scatter(r["coef"], y, s=16, color="#C44E52", edgecolor="black", linewidths=0.3, zorder=4)
    panel_label(ax, "e"); save_source(ft.drop(columns="label"), NAME, "e")
    prof = pd.read_csv(ROOT / "results/18_signalling/interface_distance_profile.tsv", sep="\t"); rt = pd.read_csv(ROOT / "results/34_expert_routing/frozen_interface_profile_patient.tsv", sep="\t")
    ax = fig.add_subplot(gs[1:, 2]); ax2 = ax.twinx(); out = []
    for d_, col, a_, lab in [(prof, "t_score", ax, "T-cell score"), (rt, "gate_0", ax2, "host-tissue expert weight")]:
        g = d_.groupby("signed_ring")[col].agg(["mean", "sem", "count"]).reset_index(); c = "#4C72B0" if a_ is ax else "#9a4f8c"
        a_.errorbar(g["signed_ring"], g["mean"], yerr=g["sem"], color=c, marker="o", ms=2.5, lw=0.9, capsize=1.5, label=lab); a_.tick_params(axis="y", colors=c, labelsize=5.5)
        out.append(g.assign(readout=lab))
    ax.axvline(0, color="#888888", lw=0.6, ls="--"); ax.set_xlabel("ring distance to interface\n(− tumour side, + lung side)", fontsize=6, fontweight="bold")
    ax.set_ylabel("T-cell score", fontsize=6, fontweight="bold", color="#4C72B0"); ax2.set_ylabel("host-tissue expert weight", fontsize=6, fontweight="bold", color="#9a4f8c")
    ax.tick_params(axis="x", labelsize=5.5); panel_label(ax, "f"); save_source(pd.concat(out), NAME, "f")
    ax = fig.add_subplot(gs[2, :2]); ax.axis("off"); half = (K + 1) // 2; ax.text(0.0, 1.0, "niche key\n" + "\n".join(f"{k}: {names[k]}" for k in range(half)), fontsize=6, va="top", transform=ax.transAxes); ax.text(0.5, 1.0, "\n" + "\n".join(f"{k}: {names[k]}" for k in range(half, K)), fontsize=6, va="top", transform=ax.transAxes)
    save_fig(fig, f"{NAME}_organisation")


if __name__ == "__main__":
    main()
