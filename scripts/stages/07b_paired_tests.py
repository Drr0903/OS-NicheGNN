"""Stage 07b — patient-paired comparisons of representations from the LOPO results.

For each metric (region macro-F1, boundary F1, accuracy) and each method/ablation, the seed-averaged value per
held-out patient is paired with OS-NicheGNN (full) on the same patient; we report the mean paired difference,
a patient-bootstrap 95 % CI (5,000 resamples), the number of patients favouring each side and an exact
Wilcoxon signed-rank P (n ≤ 10 patients — descriptive, not a claim of superiority either way).
Also: gene hold-out vs full-gene configuration for OS-NicheGNN and PCA (sensitivity).
Outputs results/07_lopo/paired_tests.tsv, results/07_lopo/config_sensitivity.tsv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "07_lopo"


def paired(a: pd.Series, b: pd.Series, seed: int) -> dict:
    d = (a - b).dropna()
    if len(d) < 3:
        return {"n_patients": len(d), "mean_diff": float(d.mean()) if len(d) else np.nan}
    rng = np.random.default_rng(seed)
    boots = [rng.choice(d.to_numpy(), len(d), replace=True).mean() for _ in range(5000)]
    return {"n_patients": len(d), "mean_diff": float(d.mean()), "ci95_lo": float(np.percentile(boots, 2.5)), "ci95_hi": float(np.percentile(boots, 97.5)),
            "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum()), "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if (d != 0).any() else 1.0}


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"])
    sd = stage_dir(STAGE)
    pat = pd.read_csv(sd / "lopo_patient_level_gene_holdout.tsv", sep="\t")
    if (sd / "lopo_patient_level_gene_holdout_banksy50.tsv").exists():  # dimension-matched BANKSY sensitivity run (review round 1)
        pat = pd.concat([pat, pd.read_csv(sd / "lopo_patient_level_gene_holdout_banksy50.tsv", sep="\t")], ignore_index=True)
    if (sd / "lopo_patient_level_gene_holdout_upgrade.tsv").exists():  #  upgraded model (selected configuration per fold)
        pat = pd.concat([pat, pd.read_csv(sd / "lopo_patient_level_gene_holdout_upgrade.tsv", sep="\t")], ignore_index=True)
    if (sd / "lopo_patient_level_gene_holdout_invariant.tsv").exists():  #  patient-invariant variant (selected configuration per fold)
        pat = pd.concat([pat, pd.read_csv(sd / "lopo_patient_level_gene_holdout_invariant.tsv", sep="\t")], ignore_index=True)
    pat["key"] = np.where(pat["ablation"] == "full", pat["method"], pat["ablation"])
    ref = pat[pat["key"] == "nichegnn"].set_index("heldout_patient")
    rows = []
    for k in sorted(set(pat["key"]) - {"nichegnn"}):
        sub = pat[pat["key"] == k].set_index("heldout_patient")
        for metric in ["macro_f1", "boundary_f1", "accuracy"]:
            r = paired(sub[metric], ref.reindex(sub.index)[metric], cfg["seed"])
            rows.append({"comparison": f"{k} − nichegnn", "metric": metric, **r})
    out = pd.DataFrame(rows); write_table(out, sd / "paired_tests.tsv")
    log(out.round(3).to_string())
    full = pd.read_csv(sd / "lopo_patient_level_full.tsv", sep="\t"); full["key"] = np.where(full["ablation"] == "full", full["method"], full["ablation"])
    rows = []
    for k in ["nichegnn", "pca"]:
        a = full[full["key"] == k].set_index("heldout_patient"); b = pat[pat["key"] == k].set_index("heldout_patient")
        for metric in ["macro_f1", "boundary_f1", "accuracy"]:
            rows.append({"method": k, "metric": metric, "gene_holdout_mean": float(b[metric].mean()), "full_gene_mean": float(a[metric].mean()), **paired(a[metric], b.reindex(a.index)[metric], cfg["seed"])})
    sens = pd.DataFrame(rows); write_table(sens, sd / "config_sensitivity.tsv"); log(sens.round(3).to_string())
    write_provenance(STAGE + "_paired", [sd / "lopo_patient_level_gene_holdout.tsv", sd / "lopo_patient_level_full.tsv"], [sd / "paired_tests.tsv", sd / "config_sensitivity.tsv"], cfg, t_start=t0)


if __name__ == "__main__":
    main()
