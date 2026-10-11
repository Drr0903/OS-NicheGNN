"""Stage 23e — how strongly each representation's niches separate patients, next to its neighbourhood read-outs.

Reads the stage-23 fold rows: in every leave-one-patient-out fold, balanced Leiden communities (K = 8) on 800 spots per training
patient were compared with patient identity (ARI; lower = niches shared across patients). Patient-level means (seeds
averaged) for OS-NicheGNN and every reference representation, and paired patient-level bootstrap differences of OS-NicheGNN
against each reference, over all patients and the patients not used to select the network.
Outputs results/23e_patient_mixing/{mixing_patient_level, mixing_summary, mixing_paired}.tsv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.arch_eval import paired  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "23e_patient_mixing"
REFS = ["banksy", "banksy+img", "propagated_pca", "propagated_pca+img", "pca", "pca+img"]
COLS = ["train_ari_vs_patient", "train_ari_vs_region", "macro_f1", "tcell_nbhd_rho", "comp_nbhd_rho"]


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE); O = cfg["atlas"]["encoder"]
    src = ROOT / "results/23_arch_search/arch_metrics.tsv"; df = pd.read_csv(src, sep="\t")
    pat = df[df.method.isin([O] + REFS)].groupby(["method", "heldout_patient"])[COLS].mean().reset_index()
    summ = pat.groupby("method")[COLS].mean().reset_index().sort_values("train_ari_vs_patient")
    S = pat[pat.method == O].set_index("heldout_patient"); rows = []
    for ref in REFS:
        R = pat[pat.method == ref].set_index("heldout_patient"); nondev = [p for p in R.index if p not in cfg["arch"]["dev_patients"]]
        for split, pts in (("all", list(R.index)), ("nondev", nondev)):
            ix = S.index.intersection(R.index).intersection(pts)
            for m in COLS:
                rows.append({"comparison": f"{O} − {ref}", "patients": split, "metric": m, **paired(S.loc[ix, m], R.loc[ix, m], cfg["seed"])})
    out = [sd / "mixing_patient_level.tsv", sd / "mixing_summary.tsv", sd / "mixing_paired.tsv"]
    write_table(pat, out[0]); write_table(summ, out[1]); write_table(pd.DataFrame(rows), out[2]); log(summ.round(3).to_string())
    write_provenance(STAGE, [src], out, cfg, t_start=t0)


if __name__ == "__main__":
    main()
