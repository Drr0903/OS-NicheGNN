"""Stage 31 — the two sides of the interface ratio and the effect in each patient.

The outer-to-inner T-cell ratio of stage 15 is split into its parts: ln(T_outer + ε) and ln(T_inner + ε) are each regressed
on band coverage with the same section fixed effects, covariate sets, patient-clustered errors and permutations; and the
coverage coefficient of the ratio is estimated within each patient (section fixed effects, patients with ≥ 10 segments).
Outputs results/31_interface_components/{side_models, per_patient}.tsv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.inference import pooled_models  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "31_interface_components"


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE); eps = 0.01
    src = ROOT / "results/15_niches/interface_segments.tsv"; seg = pd.read_csv(src, sep="\t")
    seg["ln_t_outer"] = np.log(seg["t_outer"] + eps); seg["ln_t_inner"] = np.log(seg["t_inner"] + eps)
    side = pooled_models(seg, cluster="patient", n_perm=cfg["interface"]["n_permutations"], seed=cfg["seed"], outcomes=["E_seg", "ln_t_outer", "ln_t_inner"])
    ev = seg[seg["evaluable_T"].astype(bool)]; rows = []
    for p, d in ev.groupby("patient"):
        d = d[d.groupby("section")["coverage"].transform("nunique") > 1]
        if len(d) >= 10:
            m = smf.ols("E_seg ~ coverage + C(section)", d).fit()
            rows.append({"patient": p, "n_sections": d["section"].nunique(), "n_segments": len(d), "coef": float(m.params["coverage"]), "se": float(m.bse["coverage"])})
    pp = pd.DataFrame(rows); log(f"\n{side.drop(columns=['covariates']).round(3).to_string()}\n{pp.round(3).to_string()}")
    out = [sd / "side_models.tsv", sd / "per_patient.tsv"]; write_table(side, out[0]); write_table(pp, out[1])
    write_provenance(STAGE, [src], out, cfg, t_start=t0)


if __name__ == "__main__":
    main()
