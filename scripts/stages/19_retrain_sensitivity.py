"""Stage 19 — how much of the atlas survives re-training the encoder?

The niche partition depends on the encoder that produced it. Re-training the encoder from the committed
configuration, with the same seed and the same data, gives a second encoder and a second atlas; this stage measures
how far the two agree, so the manuscript can state the run-to-run sensitivity of the partition instead of assuming
it away. The provenance of the delivered encoder is recorded in notes/DEVIATIONS.md.

Inputs : results/15_niches/{spot_niches.tsv, niche_boundary_enrichment.tsv, niche_histology.tsv, meta.json}
         data/processed/<atlas.retrain_results>/15_niches/* (the same tables from the re-trained encoder)
Outputs: results/19_retrain_sensitivity/{agreement.json, boundary_odds_comparison.tsv, PROVENANCE.json}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "19_retrain_sensitivity"
from lib.repro import load_config as _lc  # noqa: E402
RET = f"{_lc()['atlas']['retrain_results']}/15_niches"  # config atlas.retrain_results


def side(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / RET / path, sep="\t")


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE)
    cols = ["section", "barcode", "niche", "unmatched"]
    a = pd.read_csv(ROOT / "results/15_niches/spot_niches.tsv", sep="\t", usecols=cols)
    b = side("spot_niches.tsv")[cols]
    m = a.merge(b, on=["section", "barcode"], suffixes=("_delivered", "_retrained"))
    ok = (~m["unmatched_delivered"].astype(bool)) & (~m["unmatched_retrained"].astype(bool))
    ari = float(adjusted_rand_score(m.loc[ok, "niche_delivered"], m.loc[ok, "niche_retrained"]))

    rows = []
    for tag, meta_p, be, hi in [("delivered", "results/15_niches/meta.json", "results/15_niches/niche_boundary_enrichment.tsv", "results/15_niches/niche_histology.tsv"),
                                ("retrained", f"{RET}/meta.json", f"{RET}/niche_boundary_enrichment.tsv", f"{RET}/niche_histology.tsv")]:
        meta = json.load(open(ROOT / meta_p))
        d = pd.read_csv(ROOT / be, sep="\t").merge(pd.read_csv(ROOT / hi, sep="\t")[["niche", "tumor_cellular"]], on="niche")
        d["atlas"] = tag; d["name"] = d["niche"].astype(str).map(meta["names"])
        d["tumour_dominant"] = d["tumor_cellular"] >= 0.6  # = the stage-15 band rule (candidates below 60 % tumour)
        d["is_band"] = d["niche"] == meta["interface_niche"]
        rows.append(d.sort_values("odds_ratio_near_boundary", ascending=False))
    cmp_ = pd.concat(rows, ignore_index=True); write_table(cmp_, sd / "boundary_odds_comparison.tsv")

    def margin(tag: str) -> dict:
        d = cmp_[(cmp_["atlas"] == tag) & (~cmp_["tumour_dominant"])].sort_values("odds_ratio_near_boundary", ascending=False)
        return {"band_niche": int(d["niche"].iloc[0]), "band_name": str(d["name"].iloc[0]),
                "odds_first": float(d["odds_ratio_near_boundary"].iloc[0]), "odds_second": float(d["odds_ratio_near_boundary"].iloc[1])}

    out = {"n_spots_matched_in_both": int(ok.sum()), "ari_delivered_vs_retrained": round(ari, 3),
           "n_niches_delivered": int(a.loc[~a["unmatched"].astype(bool), "niche"].nunique()),
           "n_niches_retrained": int(b.loc[~b["unmatched"].astype(bool), "niche"].nunique()),
           "delivered": margin("delivered"), "retrained": margin("retrained")}
    json.dump(out, open(sd / "agreement.json", "w"), indent=2)
    log(json.dumps(out, indent=2))
    write_provenance(STAGE, [ROOT / "results/15_niches/spot_niches.tsv", ROOT / RET / "spot_niches.tsv"],
                     [sd / "agreement.json", sd / "boundary_odds_comparison.tsv"], cfg, t_start=t0)


if __name__ == "__main__":
    main()
