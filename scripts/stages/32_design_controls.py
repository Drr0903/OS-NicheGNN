"""Stage 32 — what each component of the encoder contributes, on the eight development patients (leave-one-patient-out, three
seeds, parameter counts matched at 2.1–2.2 M): the full network and the network without its graph attention branch, without its
ring-mean branch, with a single fusion expert or with the two branch encodings concatenated in place of the experts, without the
neighbourhood targets, without the neighbourhood skip and without the morphology input; the full network is also scored with
the held-out patient's morphology block at its training mean.
Reads results/23_arch_search/arch_metrics_{E9,E10e05,A9*,E5fus,E7cat}_P*.tsv; writes results/32_design_controls/{patient_level,
summary, paired}.tsv
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.arch_eval import paired  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "32_design_controls"
FULL = "full network"
VARIANTS = {"E9": FULL, "E10e05": FULL, "A9noattn": "without the graph attention branch", "A9noring": "without the ring-mean branch",
            "E5fus": "single fusion expert", "E7cat": "branches concatenated", "A9notgt": "without neighbourhood targets",
            "A9noskip": "without neighbourhood skip", "A9nomorph": "without morphology input"}
METRICS = ["tcell_nbhd_rho", "comp_nbhd_rho", "macro_f1", "tcell_rho", "comp_rho", "boundary_f1", "chaos"]


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE)
    files = sorted((ROOT / "results/23_arch_search").glob("arch_metrics_*_P*.tsv")); dev = cfg["arch"]["dev_patients"]
    tag = lambda f: f.name[len("arch_metrics_"):].rsplit("_P", 1)[0]  # noqa: E731
    files = [f for f in files if tag(f) in VARIANTS]
    d = pd.concat([pd.read_csv(f, sep="\t").assign(tag=tag(f)) for f in files], ignore_index=True); d = d[d["heldout_patient"].isin(dev)]
    d = d[(d["method"].str.endswith("_z") & (d["tag"] != "E10e05")) | (d["method"].str.endswith("_z_testnomorph") & (d["tag"] == "E10e05"))]
    d["image"] = d["method"].str.endswith("_testnomorph").map({True: "absent", False: "present"}); d["variant"] = d["tag"].map(VARIANTS)
    pat = d.groupby(["variant", "image", "heldout_patient"])[METRICS + ["train_ari_vs_patient"]].mean().reset_index()
    assert pat.groupby(["variant", "image"])["heldout_patient"].nunique().eq(len(dev)).all(), "a development fold is missing"
    summ = pat.groupby(["variant", "image"])[METRICS + ["train_ari_vs_patient"]].mean().reset_index()
    ix = lambda v, im: pat[(pat.variant == v) & (pat.image == im)].set_index("heldout_patient")  # noqa: E731
    comps = [(FULL, "absent", FULL, "present")] + [(v, "present", FULL, "present") for v in dict.fromkeys(VARIANTS.values()) if v != FULL]
    rows = []
    for a, ia, b, ib in comps:
        A, B = ix(a, ia), ix(b, ib)
        for m in METRICS + ["train_ari_vs_patient"]:
            if m == "train_ari_vs_patient" and ia != ib:
                continue
            rows.append({"comparison": f"{a} (image {ia}) − {b} (image {ib})" if ia != ib else f"{a} − {b}", "metric": m, **paired(A[m], B[m], cfg["seed"])})
    # the full network on all 19 held-out patients with their morphology block present and at its training mean
    fa = sorted((ROOT / "results/23_arch_search").glob("arch_metrics_E9nm_P*.tsv")) + sorted((ROOT / "results/23_arch_search").glob("arch_metrics_E10e05_P*.tsv"))
    da = pd.concat([pd.read_csv(f, sep="\t") for f in fa], ignore_index=True)
    zp = da[da["method"].str.endswith("_z")].groupby("heldout_patient")[METRICS].mean(); za = da[da["method"].str.endswith("_z_testnomorph")].groupby("heldout_patient")[METRICS].mean()
    nm = pd.DataFrame([{"comparison": "image absent − image present (all patients)", "metric": m, **paired(za[m], zp[m], cfg["seed"])} for m in METRICS])
    write_table(nm, sd / "image_absent_all_patients.tsv"); log(nm.round(3).to_string()); files = files + fa
    out = [sd / "patient_level.tsv", sd / "summary.tsv", sd / "paired.tsv", sd / "image_absent_all_patients.tsv"]
    write_table(pat, out[0]); write_table(summ, out[1]); write_table(pd.DataFrame(rows), out[2]); log(summ.round(3).to_string())
    write_provenance(STAGE, files, out, cfg, t_start=t0)


if __name__ == "__main__":
    main()
