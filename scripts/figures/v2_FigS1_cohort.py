"""v2 Supplementary Figure S1 — the cohort as a patient → section hierarchy.
Every patient of GSE293065 with their sections, coloured by anatomical site; sections of one patient are stacked and are
never separated in any split. Reads results/01_manifest/sample_manifest.tsv and config/run.yaml only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.plotting import PALETTE, save_fig, save_source, set_style  # noqa: E402
from lib.repro import ROOT, load_config  # noqa: E402

NAME = "FigS1"
SITE_COL = {"bone": PALETTE[5], "lung": PALETTE[0], "lymph_node": PALETTE[4], "heart": PALETTE[3], "soft_tissue": PALETTE[2]}
SITE_LAB = {"bone": "bone (primary)", "lung": "lung", "lymph_node": "lymph node", "heart": "heart", "soft_tissue": "soft tissue"}


def main() -> None:
    set_style(); cfg = load_config(); man = pd.read_csv(ROOT / "results/01_manifest/sample_manifest.tsv", sep="\t", dtype=str).set_index("section_id")
    v2 = cfg["v2"]; pat_of = v2["patient_of"]; site_of = v2["site_of"]; secs_all = v2["sections"]; site = pd.Series(site_of)
    fig = plt.figure(figsize=(7.2, 1.05)); ax = fig.add_axes([0.02, 0.10, 0.96, 0.86]); ax.axis("off")
    pats = sorted(set(pat_of.values())); x = np.linspace(0.03, 0.97, len(pats)); rows = []
    for i, p in enumerate(pats):
        secs = [s for s in secs_all if pat_of[s] == p]
        ax.text(x[i], 0.90, p, ha="center", va="bottom", fontsize=6.0, fontweight="bold", transform=ax.transAxes)
        for j, s in enumerate(secs):
            y = 0.72 - j * 0.26; n = int(float(man.loc[s, "n_spots_geo_h5"]))
            ax.add_patch(plt.Rectangle((x[i] - 0.022, y - 0.08), 0.044, 0.16, transform=ax.transAxes, fc=SITE_COL[site_of[s]], ec="black", lw=0.4))
            ax.text(x[i], y, s, ha="center", va="center", fontsize=5.0, color="white" if site_of[s] != "bone" else "black", transform=ax.transAxes)
            rows.append({"patient": p, "section": s, "site": site_of[s], "n_spots": n})
            if j > 0:
                ax.plot([x[i], x[i]], [y + 0.08, y + 0.18], color="black", lw=0.5, transform=ax.transAxes)
    hs = [plt.Rectangle((0, 0), 1, 1, fc=SITE_COL[k], ec="black", lw=0.4) for k in SITE_COL]
    ax.legend(hs, [SITE_LAB[k] for k in SITE_COL], loc="upper left", bbox_to_anchor=(0.0, 0.22), ncol=5, fontsize=5.5,
              frameon=False, handlelength=1.0, handleheight=0.8, handletextpad=0.4, columnspacing=1.4)
    save_source(pd.DataFrame(rows), NAME, "a")
    save_fig(fig, f"{NAME}_cohort")


if __name__ == "__main__":
    main()
