"""Stage 30b — the interface macrophage model with a macrophage score built only from genes that are not network inputs.

Five of the six macrophage genes of stage 30 (CD68, CD163, C1QA, C1QB, CSF1R) are among the 2,000 genes whose principal
components feed the network. Here the score is the mean log-normalized expression of a predefined panel of canonical
macrophage markers, restricted to genes absent from the network's gene list (data/processed/<frozen_dir>/genes.txt) and
detected in every lung section; the interface model of stage 30 (same sections, segments, band, covariate sets and
permutations) is then refitted on this score.
Outputs results/30b_macrophage_outside_input/{panel.tsv, interface_pooled_models.tsv, interface_sections.tsv, PROVENANCE.json}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import anndata as ad
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.inference import pooled_models  # noqa: E402
from lib.interface import run_interface_cohort  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "30b_macrophage_outside_input"
PANEL = ["CD68", "CD163", "C1QA", "C1QB", "C1QC", "MRC1", "CSF1R", "MSR1", "MS4A7", "MS4A4A", "MS4A6A", "VSIG4", "FOLR2", "LYVE1",
         "SIGLEC1", "CD14", "FCGR1A", "MARCO", "F13A1", "STAB1", "MAF", "TREM2", "APOC1", "MERTK", "SLC40A1", "LILRB5", "CD86", "ITGAM"]


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE); proc = ROOT / cfg["paths"]["processed"]
    N = ROOT / "results/15_niches"; meta = json.load(open(N / "meta.json")); spots = pd.read_csv(N / "spot_niches.tsv", sep="\t")
    inputs = set(open(proc / cfg["atlas"]["frozen_dir"] / "genes.txt").read().split())
    man = pd.read_csv(ROOT / "results/01_manifest/sample_manifest.tsv", sep="\t", dtype=str)
    elig = man[(man["role"] == "discovery_lung") & (man["boundary_eligible"] == "True")]["section_id"].tolist()
    detected = None
    for s in elig:
        a = ad.read_h5ad(proc / "qc" / f"{s}.h5ad"); X = a.layers["lognorm"] if "lognorm" in a.layers else a.X
        nz = pd.Series((X > 0).sum(0).A1 if hasattr(X, "toarray") else (X > 0).sum(0), index=a.var_names)
        det = set(nz[nz > 0].index); detected = det if detected is None else detected & det
    panel = pd.DataFrame({"gene": PANEL, "network_input": [g in inputs for g in PANEL], "detected_in_all_lung_sections": [g in detected for g in PANEL]})
    genes = panel.query("~network_input and detected_in_all_lung_sections")["gene"].tolist(); log(f"macrophage genes outside the network input: {genes}")
    c = json.loads(json.dumps(cfg)); c["genes"]["t_cell_holdout"] = genes
    sec, seg, _ = run_interface_cohort(spots, proc / "qc", ROOT / "results/05_deconvolution/weights", c, meta["interface_niche"], elig, cfg["v2"]["patient_of"])
    pm = pooled_models(seg, cluster="patient", n_perm=cfg["interface"]["n_permutations"], seed=cfg["seed"], outcomes=["E_seg", "E_seg_noniche", "E_seg_distal"])
    pm.insert(0, "score", "macrophage genes outside the network input"); log(f"\n{pm.round(3).to_string()}")
    out = [sd / "panel.tsv", sd / "interface_pooled_models.tsv", sd / "interface_sections.tsv"]
    write_table(panel, out[0]); write_table(pm, out[1]); write_table(sec, out[2])
    write_provenance(STAGE, [N / "spot_niches.tsv", N / "meta.json"], out, cfg, extra={"genes": genes}, t_start=t0)


if __name__ == "__main__":
    main()
