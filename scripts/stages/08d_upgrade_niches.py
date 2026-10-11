"""Stage 08d —  (post hoc): are the upgraded model's niches interpretable tissue states?

For each seed of the upgraded final model (embeddings written by stage 08c) the frozen-model niche steps are replayed
(Leiden with fixed K on a patient-balanced sample → centroid prototypes → nearest-prototype assignment) and every niche is characterised by its region-label
composition (purity = largest region fraction), its patient spread and its odds ratio of lying within one ring of the tumour
boundary; the same statistics are computed for the frozen model for comparison. A representation whose clusters are shared
by all patients only because they cut across tissue compartments would show low purity everywhere.
Outputs results/08_niches/upgrade_niche_purity.tsv (per niche) and upgrade_niche_purity_summary.tsv (per representation)
"""
from __future__ import annotations

import json
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.data import build_graph, load_labels, load_qc_sections  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "08_niches"
s09c = import_module("09c_representation_controls")


def characterise(name: str, emb: dict, G: dict, secs: list[str], patient_of: dict, K: int, cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["seed"])
    spots, inter, _ = s09c.niche_pipeline(emb, G, secs, patient_of, K, cfg, rng)
    rows = []
    for k in range(K):
        m = spots["niche"] == k
        comp = spots.loc[m, "histology_label"].value_counts(normalize=True)
        a_, b_ = (m & spots["near_boundary"]).sum(), (~m & spots["near_boundary"]).sum(); c_, d_ = (m & ~spots["near_boundary"]).sum(), (~m & ~spots["near_boundary"]).sum()
        share = pd.crosstab(spots["patient"], spots["niche"], normalize="index").get(k, pd.Series(dtype=float))
        rows.append({"representation": name, "niche": k, "n_spots": int(m.sum()), "purity": float(comp.max()) if len(comp) else np.nan, "dominant_label": comp.idxmax() if len(comp) else "",
                     "tumor_frac": float(comp.get("tumor_cellular", 0.0)), "lung_frac": float(comp.get("lung_parenchyma", 0.0)), "pauci_frac": float(comp.get("paucicellular", 0.0)),
                     "odds_ratio_near_boundary": float((a_ * d_) / max(b_ * c_, 1)), "n_patients_present": int((share >= 0.02).sum()), "max_patient_share": float(pd.crosstab(spots["niche"], spots["patient"], normalize="index").loc[k].max()),
                     "is_interface_niche": k == inter})
    return pd.DataFrame(rows)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="upgrade", choices=["upgrade", "invariant"]); V = ap.parse_args().variant
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"])
    disc = cfg["cohorts"]["discovery"]; patient_of = disc["patient_groups"]; proc = ROOT / cfg["paths"]["processed"]
    secs = [s for s in disc["lung_sections"] if (proc / "qc" / f"{s}.h5ad").exists()]
    adatas = load_qc_sections(cfg, secs); labels = {s: load_labels(cfg, s) for s in secs}
    genes = open(proc / "frozen" / "genes.txt").read().split(); sc_ = np.load(proc / "frozen" / "scaler.npz"); K = json.load(open(proc / "frozen" / "meta.json"))["K"]
    G = {s: build_graph(adatas[s], genes, sc_["mu"], sc_["sd"], cfg, patient_of[s], labels[s]) for s in secs}
    parts = [characterise("OS-NicheGNN (frozen)", {s: np.load(proc / "embeddings" / "frozen" / f"{s}.npy") for s in secs}, G, secs, patient_of, K, cfg)]
    for i in range(cfg["n_seeds"]):
        seed = cfg["seed"] + i; d = proc / "embeddings" / V / str(seed)
        if d.exists():
            parts.append(characterise(f"{V} seed {i + 1}", {s: np.load(d / f"{s}.npy") for s in secs}, G, secs, patient_of, K, cfg))
    out = pd.concat(parts, ignore_index=True); sd = stage_dir(STAGE); write_table(out, sd / f"{V}_niche_purity.tsv")
    summ = out.groupby("representation").agg(purity_spot_weighted=("purity", lambda x: float(np.average(x, weights=out.loc[x.index, "n_spots"]))), purity_min=("purity", "min"), purity_max=("purity", "max"),
                                             n_niches_purity_ge_0_8=("purity", lambda x: int((x >= 0.8).sum())), n_niches_present_in_half_patients=("n_patients_present", lambda x: int((x >= 5).sum())),
                                             max_or_near_boundary=("odds_ratio_near_boundary", "max"), interface_niche_purity=("purity", lambda x: float(x[out.loc[x.index, "is_interface_niche"]].iloc[0]))).reset_index()
    write_table(summ, sd / f"{V}_niche_purity_summary.tsv"); log(f"\n{summ.round(3).to_string()}")
    write_provenance(STAGE + f"_{V}_niches", [sd / f"{V}_identity_diagnostics.tsv"], [sd / f"{V}_niche_purity.tsv"], cfg, t_start=t0)


if __name__ == "__main__":
    main()
