"""Stage 23d — BANKSY with the neighbourhood size of OS-NicheGNN (36 spatial neighbours, rings 1–3) on the stage-23 folds.

Same leave-one-patient-out folds, gene selection, held-out evaluation and niche metrics as stage 23; BANKSY is fitted on the
training patients with k_geom = 36 (λ = 0.8, 20 components) and scored with and without the H&E morphology components.
Usage: python scripts/stages/23d_banksy_k36.py --patients P01 --tag _P01 ;  --merge _P01,...
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.arch_eval import image_graphs, paired, pca_graphs  # noqa: E402
from lib.arch_metrics import niche_metrics, targets  # noqa: E402
from lib.banksy import BanksyEmbedder  # noqa: E402
from lib.context_probe import load_weights  # noqa: E402
from lib.evalprobe import evaluate_heldout  # noqa: E402
from lib.interface import t_score  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.v2 import load_cohort, make_graphs  # noqa: E402

STAGE = "23d_banksy_k36"
K_GEOM = 36
METRICS = ["macro_f1", "boundary_f1", "tcell_rho", "tcell_nbhd_rho", "comp_rho", "comp_nbhd_rho", "chaos", "pas", "mix"]


def run_fold(cfg, adatas, labels, patient_of, site_of, P):
    secs = list(adatas); K = cfg["v2"]["k_fixed"]; seed = cfg["seed"]
    test = [s for s in secs if patient_of[s] == P]; train = [s for s in secs if patient_of[s] != P]
    G, _, _, _ = make_graphs(cfg, adatas, labels, patient_of, train, train + test); Gp = pca_graphs(G, train, cfg["remedies"]["pca_dim"], seed)
    Gi, _ = image_graphs(G, Gp, train, cfg["remedies"]["pca_dim"], seed, ROOT / cfg["paths"]["processed"] / "image_features")
    T = targets(G, {s: t_score(adatas[s], cfg["genes"]["t_cell_holdout"]) for s in G}, load_weights(G))
    b = cfg["baselines"]["banksy"]; be = BanksyEmbedder(b["lambda"], K_GEOM, b["n_pcs"], seed).fit([G[s] for s in train])
    bk = {s: be.transform(G[s]) for s in train + test}; rows = []
    for name, emb in (("banksy_k36", bk), ("banksy_k36+img", {s: np.hstack([bk[s], Gi[s].x]) for s in train + test})):
        m = evaluate_heldout(emb, [G[s] for s in train], [G[s] for s in test], cfg, seed); m.pop("per_section_boundary", None)
        nm = niche_metrics(emb, G, T, train, test, K, seed)
        rows.append({"method": name, "seed": seed, "heldout_patient": P, "heldout_site": "+".join(sorted({site_of[s] for s in test})), **m, **nm})
        log(f"fold {P} {name}: F1 {m['macro_f1']:.3f} T {nm['tcell_rho']:.3f}/{nm['tcell_nbhd_rho']:.3f} comp {nm['comp_rho']:.3f}/{nm['comp_nbhd_rho']:.3f}")
    return rows


def summarise(df: pd.DataFrame, cfg: dict) -> list[Path]:
    """Paired patient-level bootstrap of the atlas encoder against BANKSY-36 and BANKSY-20, over all patients, the confirmation
    patients and the patients not used to select the network."""
    sd = stage_dir(STAGE); O = cfg["atlas"]["encoder"]
    ref = pd.read_csv(ROOT / "results/23_arch_search/arch_patient_level.tsv", sep="\t"); ref = ref[ref.method.isin([O, "banksy"])]
    pat = pd.concat([df.groupby(["method", "heldout_patient"])[METRICS].mean().reset_index(), ref[["method", "heldout_patient"] + METRICS]], ignore_index=True)
    S = pat[pat.method == O].set_index("heldout_patient"); rows = []
    for base in ("banksy_k36", "banksy_k36+img", "banksy"):
        R = pat[pat.method == base].set_index("heldout_patient")
        nondev = [p for p in R.index if p not in cfg["arch"]["dev_patients"]]
        for split, pts in (("all", list(R.index)), ("confirm", cfg["arch"]["confirm_patients"]), ("nondev", nondev)):
            ix = S.index.intersection(R.index).intersection(pts)
            for metric in METRICS:
                rows.append({"comparison": f"{O} − {base}", "patients": split, "metric": metric, **paired(S.loc[ix, metric], R.loc[ix, metric], cfg["seed"])})
    out = [sd / "banksy_k36_patient_level.tsv", sd / "banksy_k36_paired.tsv"]
    write_table(pat, out[0]); write_table(pd.DataFrame(rows), out[1]); return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--patients", default=""); ap.add_argument("--tag", default=""); ap.add_argument("--merge", default="")
    a = ap.parse_args(); t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE)
    if a.merge:
        files = [sd / f"banksy_k36_metrics{t}.tsv" for t in a.merge.split(",")]; df = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
        write_table(df, sd / "banksy_k36_metrics.tsv"); out = summarise(df, cfg)
        write_provenance(STAGE, files, out + [sd / "banksy_k36_metrics.tsv"], cfg, t_start=t0); return
    adatas, labels, patient_of, site_of = load_cohort(cfg); pats = [p for p in sorted(set(patient_of.values())) if not a.patients or p in a.patients.split(",")]
    rows = []
    for P in pats:
        rows += run_fold(cfg, adatas, labels, patient_of, site_of, P); write_table(pd.DataFrame(rows), sd / f"banksy_k36_metrics{a.tag}.tsv")
    write_provenance(STAGE + f"_part{a.tag}", [], [sd / f"banksy_k36_metrics{a.tag}.tsv"], cfg, extra={"patients": pats}, t_start=t0)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
