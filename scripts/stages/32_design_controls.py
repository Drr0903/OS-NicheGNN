"""Stage 32 — two controls of the encoder design on the eight development patients (leave-one-patient-out, three seeds).

  missing image   the held-out patient embedded with its morphology block at the training mean (0) by the trained network,
                  and networks trained with the image of a training section withheld (block at its mean) with probability
                  0.3 or 0.5, scored with and without the held-out image
  ring-mean MLP   attention replaced by a residual MLP on [own profile, ring-1 mean, ring-2–3 mean] of matched parameter count
                  (hidden width 416: 2.10 M against 2.15 M parameters), same targets and linear skip
Reads results/23_arch_search/arch_metrics_E{2base,2md3,2md5,4rmlp}_P*.tsv (stage 23 with --eval-nomorph, --morph-drop, --anchor
ringmlp); writes results/32_design_controls/{patient_level, summary, paired}.tsv
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
VARIANTS = {"2base": "trained network", "2md3": "image withheld in training, p = 0.3", "2md5": "image withheld in training, p = 0.5",
            "4rmlp": "ring-mean MLP (parameter-matched)"}
METRICS = ["tcell_nbhd_rho", "comp_nbhd_rho", "macro_f1", "tcell_rho", "comp_rho"]


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE)
    files = sorted((ROOT / "results/23_arch_search").glob("arch_metrics_E*_P*.tsv"))
    d = pd.concat([pd.read_csv(f, sep="\t").assign(variant=re.search(r"_E(\w+?)_P", f.name).group(1)) for f in files], ignore_index=True)
    d = d[d["variant"].isin(VARIANTS)]; d = d[d["method"].str.endswith("_z") | d["method"].str.endswith("_z_testnomorph")]
    d["image"] = d["method"].str.endswith("_testnomorph").map({True: "absent", False: "present"}); d["variant"] = d["variant"].map(VARIANTS)
    pat = d.groupby(["variant", "image", "heldout_patient"])[METRICS + ["train_ari_vs_patient"]].mean().reset_index()
    pat["train_ari_vs_patient"] = pat.groupby(["variant", "heldout_patient"])["train_ari_vs_patient"].transform("max")
    assert pat.groupby(["variant", "image"])["heldout_patient"].nunique().eq(len(cfg["arch"]["dev_patients"])).all(), "a development fold is missing"
    summ = pat.groupby(["variant", "image"])[METRICS + ["train_ari_vs_patient"]].mean().reset_index()
    ix = lambda v, im: pat[(pat.variant == v) & (pat.image == im)].set_index("heldout_patient")  # noqa: E731
    comps = [("trained network", "absent", "trained network", "present")] \
        + [(VARIANTS[k], im, "trained network", im) for k in VARIANTS if k != "2base" for im in ("present", "absent")]
    rows = []
    for a, ia, b, ib in comps:
        A, B = ix(a, ia), ix(b, ib)
        for m in METRICS + ["train_ari_vs_patient"]:
            if m == "train_ari_vs_patient" and ia != ib:
                continue
            rows.append({"comparison": f"{a} (image {ia}) − {b} (image {ib})", "metric": m, **paired(A[m], B[m], cfg["seed"])})
    out = [sd / "patient_level.tsv", sd / "summary.tsv", sd / "paired.tsv"]
    write_table(pat, out[0]); write_table(summ, out[1]); write_table(pd.DataFrame(rows), out[2]); log(summ.round(3).to_string())
    write_provenance(STAGE, files, out, cfg, t_start=t0)


if __name__ == "__main__":
    main()
