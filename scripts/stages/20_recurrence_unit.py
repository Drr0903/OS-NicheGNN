"""Stage 20 — niche recurrence counted by patient rather than by section.

The pre-specified class rule counts metastatic *sections*, so the two patients who contributed two sections each
(P10 and P11) weigh twice as much as a patient with one section, which sits badly with the patient being the unit
of analysis everywhere else. This stage recomputes presence and the classes with the patient as the unit: a niche
is present in a patient when its mean fraction over that patient's sections reaches the same 2 % threshold. Both
counts are reported; the section-level rule is kept as the pre-specified sensitivity.

Inputs : results/15_niches/{niche_by_section.tsv, niche_classification.tsv, meta.json}
Outputs: results/20_recurrence_unit/{recurrence_by_unit.tsv, PROVENANCE.json}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "20_recurrence_unit"
THR = 0.02


def classes(pres: pd.DataFrame, is_met: pd.Series, site: pd.Series, K: int) -> pd.DataFrame:
    """the pre-specified rule, applied to whatever unit `pres` is indexed by"""
    met, prim = pres.index[is_met], pres.index[~is_met]
    rows = []
    for k in range(K):
        col = str(k) if str(k) in pres.columns else k
        p = pres[col]
        n_met, n_prim = int(p.loc[met].sum()), int(p.loc[prim].sum())
        sites = {site[u] for u in met if p.loc[u]}
        per_site = {s: float(np.mean([p.loc[u] for u in pres.index if site[u] == s])) for s in sorted(set(site))}
        if n_met >= len(met) / 2 and len(sites) >= 2:
            cls = "shared metastatic"
        elif any(v >= 0.5 for s, v in per_site.items() if s != "bone") and sum(v >= 0.5 for v in per_site.values()) == 1:
            cls = "site-specific (" + [s for s, v in per_site.items() if v >= 0.5][0] + ")"
        elif n_prim >= len(prim) / 2 and n_met < len(met) / 2:
            cls = "primary-predominant"
        else:
            cls = "sporadic"
        rows.append({"niche": k, "n_metastatic_present": n_met, "n_metastatic_total": len(met),
                     "n_metastatic_sites_present": len(sites), "n_primary_present": n_prim, "n_primary_total": len(prim), "class": cls})
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE)
    v2 = cfg["v2"]; meta = json.load(open(ROOT / "results/15_niches/meta.json")); K = meta["K"]
    by_sec = pd.read_csv(ROOT / "results/15_niches/niche_by_section.tsv", sep="\t").set_index("section")
    cols = [c for c in by_sec.columns if c.isdigit() or (isinstance(c, str) and c.strip().isdigit())]
    sec_site = pd.Series({s: v2["site_of"][s] for s in by_sec.index})
    sec_pres = (by_sec[cols] >= THR)
    sec_cls = classes(sec_pres, sec_site != "bone", sec_site, K).assign(unit="section")

    pat = pd.Series({s: v2["patient_of"][s] for s in by_sec.index})
    # a patient contributes primaries and metastases separately, so the unit is the patient within a role
    role = np.where(sec_site == "bone", "primary", "metastasis")
    key = pd.Series([f"{p}|{r}" for p, r in zip(pat, role)], index=by_sec.index)
    mean_frac = by_sec[cols].groupby(key).mean()
    unit_site = pd.Series({k: sec_site[key[key == k].index].mode().iloc[0] for k in mean_frac.index})
    pat_cls = classes(mean_frac >= THR, unit_site != "bone", unit_site, K).assign(unit="patient")

    out = pd.concat([pat_cls, sec_cls], ignore_index=True)
    out["name"] = out["niche"].astype(str).map(meta["names"])
    write_table(out, sd / "recurrence_by_unit.tsv")
    agree = int((pat_cls["class"].to_numpy() == sec_cls["class"].to_numpy()).sum())
    log(f"classes identical for {agree} of {K} niches between the patient-level and the section-level rule")
    for _, r in pat_cls.iterrows():
        log(f"  niche {int(r['niche'])} {out.loc[out['niche'] == r['niche'], 'name'].iloc[0]:34s} patients {int(r['n_metastatic_present'])}/{int(r['n_metastatic_total'])}  {r['class']}")
    write_provenance(STAGE, [ROOT / "results/15_niches/niche_by_section.tsv"], [sd / "recurrence_by_unit.tsv"], cfg,
                     extra={"threshold": THR, "classes_identical": agree, "K": K}, t_start=t0)


if __name__ == "__main__":
    main()
