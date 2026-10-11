"""Stage 09b — EXPLORATORY: a model-free interface definition as a comparator for the niche-based one.

Because the frozen niches turned out to be largely patient-specific (stage 08 / 08b), coverage by the learned
interface niche is nearly all-or-none per section. Here the same interface machinery (lib/interface.py, identical
band/segment/T-readout rules) is run with an interface definition that uses no learned representation:
    a non-tumour spot belongs to the "stromal–myeloid interface" if its RCTD fibroblast + pericyte + myeloid +
    osteoclast weight is ≥ 0.5 (composition rule)
and, as a second comparator, the *paucicellular* reference label itself. Results are labelled exploratory in the
manuscript and are not part of the set in advance confirmatory analysis.
Outputs results/09_interface/modelfree_{composition,paucicellular}_{section_metrics,segments,matched_contrast,patient_level}.tsv + summary json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.inference import pooled_models  # noqa: E402
from lib.interface import run_interface_cohort  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "09_interface"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402


def make_spots(base: pd.DataFrame, rule: str, w_dir: Path) -> pd.DataFrame:
    """Return a spots table whose `niche` column is 1 for interface spots under the rule, 0 otherwise."""
    spots = base.copy(); spots["niche"] = 0; spots["unmatched"] = False
    if rule == "paucicellular":
        spots.loc[spots["histology_label"] == "paucicellular", "niche"] = 1
        return spots
    parts = []
    for s in spots["section"].unique():
        w = pd.read_csv(w_dir / f"{s}_rctd_weights.tsv", sep="\t").set_index("barcode")
        sub = spots[spots["section"] == s].set_index("barcode")
        stro = w.reindex(sub.index)[[c for c in ["Fibroblast", "Pericyte_SMC", "Myeloid", "Osteoclast"] if c in w.columns]].sum(1).fillna(0)
        sub = sub.copy(); sub["niche"] = ((stro >= 0.5) & (sub["histology_label"] != "tumor_cellular")).astype(int).to_numpy(); parts.append(sub.reset_index())
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"])
    s09 = import_module("09_interface_analysis")
    base = pd.read_csv(ROOT / "results/08_niches/spot_niches.tsv", sep="\t")
    man = pd.read_csv(ROOT / "results/01_manifest/sample_manifest.tsv", sep="\t", dtype=str)
    elig = man[(man["role"] == "discovery_lung") & (man["boundary_eligible"] == "True")]["section_id"].tolist()
    patient_of = cfg["cohorts"]["discovery"]["patient_groups"]; w_dir = ROOT / "results/05_deconvolution/weights"
    sd = stage_dir(STAGE); summaries = {}
    for rule in ["composition", "paucicellular"]:
        spots = make_spots(base, rule, w_dir)
        sec, seg, con = run_interface_cohort(spots, ROOT / cfg["paths"]["processed"] / "qc", w_dir, cfg, 1, elig, patient_of)
        rep = s09.representative_sections(man, sec); pl, summ = s09.patient_level(sec, con, rep, cfg["seed"])
        if len(seg) >= 20:
            pm = pooled_models(seg, cluster="patient", n_perm=cfg["interface"]["n_permutations"], seed=cfg["seed"]); write_table(pm, sd / f"modelfree_{rule}_pooled_models.tsv")
            key = pm.set_index(["outcome", "covariate_set"])
            summ["segment_model"] = {f"{o}__{c}": {k: (None if (not isinstance(v, str) and pd.isna(v)) else (v if isinstance(v, str) else float(v))) for k, v in key.loc[(o, c)].drop(["covariates"]).items()} for o in key.index.get_level_values(0).unique() for c in ["none", "3type", "full"] if (o, c) in key.index}
        for name, df in [("section_metrics", sec), ("segments", seg), ("matched_contrast", con), ("patient_level", pl)]:
            write_table(df, sd / f"modelfree_{rule}_{name}.tsv")
        summaries[rule] = summ; log(f"{rule}: {json.dumps(summ, indent=1)}")
    json.dump(summaries, open(sd / "modelfree_summary.json", "w"), indent=2)
    write_provenance(STAGE + "_modelfree", [ROOT / "results/08_niches/spot_niches.tsv"], [sd / "modelfree_summary.json"], cfg, extra={"exploratory": True}, t_start=t0)


if __name__ == "__main__":
    main()
