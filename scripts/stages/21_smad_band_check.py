"""Stage 21 — is the peritumoural band's low TGF-beta activity an artefact of one scoring method?

The band being TGF-beta-low runs against the fibroblast barrier described in carcinomas, so it is the claim a reader
will challenge first. It is scored in stage 18 with PROGENy, a pathway-footprint method. This stage asks the same
question with a different method and a different prior knowledge base: the activity of the SMAD regulons from
CollecTRI, whose targets are the transcription factors that transduce the TGF-beta (SMAD2/3) and BMP (SMAD1/5)
signals. The test is patient-level and paired: for every patient the band is compared against the mean of that
patient's other niches, by Wilcoxon signed-rank.

Inputs : results/18_signalling/tf_by_patient_niche.tsv, results/15_niches/meta.json
Outputs: results/21_smad_band/{smad_band_vs_rest.tsv, PROVENANCE.json}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "21_smad_band"
SMADS = ["SMAD1", "SMAD2", "SMAD3", "SMAD4", "SMAD5", "SMAD6", "SMAD7"]
MIN_NICHES = 4  # a patient counts towards "band lowest" only when it carries enough niches for the ranking to mean anything


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE)
    meta = json.load(open(ROOT / "results/15_niches/meta.json"))
    band = int(meta["interface_niche"]); names = {int(k): v for k, v in meta["names"].items()}
    tf = pd.read_csv(ROOT / "results/18_signalling/tf_by_patient_niche.tsv", sep="\t")
    log(f"band niche {band}: {names[band]}")
    rows = []
    for s in SMADS:
        d = tf[tf["source"] == s].pivot_table(index="patient", columns="niche", values="activity")
        if band not in d.columns:
            continue
        d = d.dropna(subset=[band])
        others = d.drop(columns=[band]).mean(axis=1)
        both = pd.concat([d[band], others], axis=1).dropna(); both.columns = ["band", "others"]
        evaluable = [p for p in d.index if d.loc[p].notna().sum() >= MIN_NICHES]
        lowest = sum(d.loc[p].idxmin() == band for p in evaluable)
        stat, p = wilcoxon(both["band"], both["others"])
        rows.append({"regulon": s, "n_patients": len(both), "band_mean": both["band"].mean(),
                     "other_niche_mean": both["others"].mean(), "delta": (both["band"] - both["others"]).mean(),
                     "p_wilcoxon": p, "n_patients_band_lowest": lowest, "n_patients_evaluable": len(evaluable)})
        log(f"  {s}: band {both['band'].mean():+.2f} vs rest {both['others'].mean():+.2f} "
            f"(delta {(both['band'] - both['others']).mean():+.2f}, P = {p:.4f}, lowest in {lowest}/{len(evaluable)})")
    out = pd.DataFrame(rows)
    out["fdr"] = out["p_wilcoxon"] * len(out) / out["p_wilcoxon"].rank()  # Benjamini-Hochberg over the seven regulons
    out["fdr"] = out["fdr"][::-1].cummin()[::-1].clip(upper=1.0)
    write_table(out, sd / "smad_band_vs_rest.tsv")
    write_provenance(STAGE, [ROOT / "results/18_signalling/tf_by_patient_niche.tsv"], [sd / "smad_band_vs_rest.tsv"],
                     cfg, extra={"band_niche": band, "regulons": SMADS, "min_niches_for_ranking": MIN_NICHES}, t_start=t0)


if __name__ == "__main__":
    main()
