"""v2 Supplementary Figure S3 — what each component of OS-NicheGNN contributes (–A20).
Rows build the latent up one step at a time from the spot profile [expression PCA-50, H&E PCA-50]: attention encoder with a
64-d latent; cross-patient anchors and a 256-d latent; niche targets; linear target heads and the linear neighbourhood skip;
without the anchor term (= OS-NicheGNN); + graph infomax; + morphology input weight 0; + CLIP alignment and + mixture of
experts (not adopted; the last three separate patients beyond the 0.10 limit, dashed line in f).
a–f  held-out region macro-F1, boundary F1, T-cell programme of the neighbourhood, cell composition of the neighbourhood,
     spatial coherence of held-out niches (CHAOS, lower is better), and the training-embedding ARI against patient
     (means ± s.e.m. over the eight development patients; ARI: mean ± s.d.)
Reads results/23_arch_search/{arch_patient_level, arch_metrics}.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.plotting import PALETTE, bold_axes, panel_label, save_fig, save_source, set_style  # noqa: E402
from lib.repro import ROOT, load_config  # noqa: E402

NAME = "FigS3"
F = "mm_xattn_anchor_d256z256_ntrod_lin_ps"
ROWS = [("pca+img", "spot profile\n(PCA-50 + H&E)"), ("mm_xattn_z", "attention encoder,\n64-d latent"), ("mm_xattn_anchor_d256z256_z", "+ anchors,\n256-d latent"),
        ("mm_xattn_anchor_d256z256_ntro_z", "+ niche targets"), (F + "_z", "+ linear heads,\nneighbourhood skip"), (F + "_la0_z", "no anchor term\n= OS-NicheGNN"),
        (F + "_dgi_la0_z", "+ graph infomax\n(not adopted)"), (F + "_dgi_img0_la0_z", "+ morphology weight 0\n(separates patients)"),
        (F + "_clip_dgi_img0_la0_z", "+ CLIP alignment\n(separates patients)"), (F + "_moe_clip_dgi_img0_la0_z", "+ mixture of experts\n(separates patients)")]
PANELS = [("macro_f1", "region macro-F1"), ("boundary_f1", "boundary F1"), ("tcell_nbhd_rho", "T-cell programme,\nneighbourhood (ρ)"),
          ("comp_nbhd_rho", "cell composition,\nneighbourhood (ρ)"), ("chaos", "CHAOS (lower = more\ncoherent niches)"), ("train_ari_vs_patient", "ARI vs patient,\ntraining embeddings")]
COL = [PALETTE[7], PALETTE[5], PALETTE[4], PALETTE[2], PALETTE[9], PALETTE[6], PALETTE[3], PALETTE[0], PALETTE[8], PALETTE[1]]


def main() -> None:
    set_style(); A = ROOT / "results/23_arch_search"
    pat = pd.read_csv(A / "arch_patient_level.tsv", sep="\t"); m = pd.read_csv(A / "arch_metrics.tsv", sep="\t")
    dev = load_config()["arch"]["dev_patients"]  # every row on the same eight development patients (some variants were only run there)
    pat = pat[pat["heldout_patient"].isin(dev)]; m = m[m["heldout_patient"].isin(dev)]
    fold = m.groupby(["method", "heldout_patient"])["train_ari_vs_patient"].mean().reset_index()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 8.2)); out = []
    for k, (ax, (metric, lab)) in enumerate(zip(axes.ravel(), PANELS)):
        src = fold if metric == "train_ari_vs_patient" else pat
        lo = 1.0 if metric == "chaos" else 0
        rng_ = max(src.loc[src["method"].isin([r for r, _ in ROWS])].groupby("method")[metric].mean()) - lo
        for i, (meth, _) in enumerate(ROWS):
            v = src.loc[src["method"] == meth, metric].dropna()
            err = v.std(ddof=1) if metric == "train_ari_vs_patient" else v.std(ddof=1) / np.sqrt(len(v))
            ax.barh(i, v.mean(), xerr=err, color=COL[i], edgecolor="white", lw=0.3, error_kw={"lw": 0.7, "capsize": 1.5})
            ax.text(v.mean() + err + 0.04 * rng_, i, f"{v.mean():.3f}", va="center", fontsize=5.5)
            out.append({"panel": "abcdef"[k], "metric": metric, "method": meth, "mean": v.mean(), "error": err, "n": len(v)})
        hi = max(r["mean"] + r["error"] for r in out if r["metric"] == metric)
        ax.set_xlim(lo, lo + (hi - lo) * 1.4); ax.set_yticks(range(len(ROWS)))
        ax.set_yticklabels([r for _, r in ROWS] if k % 3 == 0 else [], fontsize=6); ax.invert_yaxis()
        if metric == "train_ari_vs_patient":
            ax.axvline(0.10, color="black", lw=0.7, ls="--")
        bold_axes(ax, x=lab); panel_label(ax, "abcdef"[k])
    fig.tight_layout(w_pad=0.8, h_pad=1.2); save_source(pd.DataFrame(out), NAME, "a-f"); save_fig(fig, f"{NAME}_variant_selection")


if __name__ == "__main__":
    main()
