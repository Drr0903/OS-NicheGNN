"""Stage 30 — two checks on the atlas niches.

(a) Osteoclast against macrophage programme per niche: patient-level mean log-normalized expression of osteoclast-specific
    genes (fusion and differentiation: DCSTAMP, OCSTAMP, CALCR, NFATC1, TNFRSF11A), genes shared by osteoclasts and
    macrophages (ACP5, CTSK, MMP9) and macrophage genes (CD68, CD163, C1QA, C1QB, MRC1, CSF1R); each programme score is the
    mean over its genes, z-scored across niches.
(b) The tumour–lung interface model of stage 15 with the myeloid (macrophage) score in place of the T-cell score, on the same
    sections, segments and band.
Outputs results/30_osteoclast_myeloid/{niche_programmes_patient, niche_programmes, interface_myeloid_pooled_models}.tsv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.inference import pooled_models  # noqa: E402
from lib.interface import run_interface_cohort, t_score  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "30_osteoclast_myeloid"
PROGRAMMES = {"osteoclast_specific": ["DCSTAMP", "OCSTAMP", "CALCR", "NFATC1", "TNFRSF11A"],
              "osteoclast_macrophage_shared": ["ACP5", "CTSK", "MMP9"],
              "macrophage": ["CD68", "CD163", "C1QA", "C1QB", "MRC1", "CSF1R"]}


def niche_programmes(spots: pd.DataFrame, qc: Path, patient_of: dict, K: int) -> pd.DataFrame:
    rows = []
    for s in sorted(spots["section"].unique()):
        a = ad.read_h5ad(qc / f"{s}.h5ad"); sub = spots[spots["section"] == s].set_index("barcode").reindex(a.obs_names)
        sc = {p: t_score(a, g) for p, g in PROGRAMMES.items()}
        for k in range(K):
            m = (sub["niche"] == k).to_numpy() & (~sub["unmatched"].fillna(True).to_numpy(bool))
            if m.sum() >= 20:
                rows.append({"section": s, "patient": patient_of[s], "niche": k, "n_spots": int(m.sum()), **{p: float(v[m].mean()) for p, v in sc.items()}})
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE); proc = ROOT / cfg["paths"]["processed"]
    N = ROOT / "results/15_niches"; meta = json.load(open(N / "meta.json")); K = meta["K"]; names = {int(k): v for k, v in meta["names"].items()}
    spots = pd.read_csv(N / "spot_niches.tsv", sep="\t"); patient_of = cfg["v2"]["patient_of"]
    by_sec = niche_programmes(spots, proc / "qc", patient_of, K)
    pat = by_sec.groupby(["patient", "niche"])[list(PROGRAMMES)].mean().reset_index()
    summ = pat.groupby("niche")[list(PROGRAMMES)].mean()
    for p in PROGRAMMES:
        summ[f"{p}_z"] = (summ[p] - summ[p].mean()) / summ[p].std(ddof=0)
    summ["n_patients"] = pat.groupby("niche").size(); summ = summ.reset_index(); summ.insert(1, "name", summ["niche"].map(names))
    log(f"\n{summ.round(3).to_string()}")
    # (b) myeloid score at the interface: same sections and rules as stage 15
    man = pd.read_csv(ROOT / "results/01_manifest/sample_manifest.tsv", sep="\t", dtype=str)
    elig = man[(man["role"] == "discovery_lung") & (man["boundary_eligible"] == "True")]["section_id"].tolist()
    c = json.loads(json.dumps(cfg)); c["genes"]["t_cell_holdout"] = PROGRAMMES["macrophage"]
    sec, seg, _ = run_interface_cohort(spots, proc / "qc", ROOT / "results/05_deconvolution/weights", c, meta["interface_niche"], elig, patient_of)
    pm = pooled_models(seg, cluster="patient", n_perm=cfg["interface"]["n_permutations"], seed=cfg["seed"], outcomes=["E_seg", "E_seg_noniche", "E_seg_distal"])
    pm.insert(0, "score", "myeloid"); log(f"\n{pm.round(3).to_string()}")
    out = [sd / "niche_programmes_patient.tsv", sd / "niche_programmes.tsv", sd / "interface_myeloid_pooled_models.tsv", sd / "interface_myeloid_sections.tsv"]
    write_table(pat, out[0]); write_table(summ, out[1]); write_table(pm, out[2]); write_table(sec, out[3])
    write_provenance(STAGE, [N / "spot_niches.tsv", N / "meta.json"], out, cfg, t_start=t0)


if __name__ == "__main__":
    main()
