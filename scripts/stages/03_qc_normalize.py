"""Stage 03 — per-section QC, common gene universe, log-normalisation.

Inputs   data/processed/gse293065/<sec>.h5ad, data/processed/external/<OS>.h5ad
Outputs  data/processed/qc/<sec>.h5ad          layers: counts, lognorm ; all genes kept, var["in_universe"] flags the model pool
         results/03_qc/section_qc_summary.tsv   spots before/after, thresholds used
         results/03_qc/gene_universe.tsv        genes measured on both platforms (discovery ∩ every external lung object)
         results/03_qc/t_cell_holdout_status.tsv

Gene universe = genes present in the discovery probe set AND in every external lung object.
Only gene *identities* of the external cohort are used here (a platform property), never their values.
QC thresholds are adaptive per section (quantile of total counts) so the two platforms are not forced
through one absolute cutoff.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "03_qc"


def qc_section(a: ad.AnnData, cfg: dict) -> tuple[ad.AnnData, dict]:
    q = cfg["qc"]
    n0 = a.n_obs
    thr_counts = float(np.quantile(a.obs["n_counts"], q["min_counts_quantile"]))
    keep = (a.obs["n_counts"] >= thr_counts) & (a.obs["n_genes"] >= q["min_genes_abs"])
    a.obs["qc_pass"] = keep.to_numpy()
    a.obs["flag_high_mito"] = (a.obs["pct_mito"] > q["max_mito_pct"]).to_numpy()
    a = a[a.obs["qc_pass"]].copy()
    return a, {"n_spots_before": n0, "n_spots_after": int(a.n_obs), "thr_counts": thr_counts, "n_flag_high_mito": int(a.obs["flag_high_mito"].sum())}


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    set_global_seed(cfg["seed"])
    proc = ROOT / cfg["paths"]["processed"]
    out_dir = proc / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    disc_files = sorted((proc / "gse293065").glob("*.h5ad"))
    ext_files = sorted((proc / "external").glob("*.h5ad"))
    ext_lung = set(cfg["cohorts"]["external"]["lung_sections"])

    # ---- gene universe ------------------------------------------------------------------
    disc_genes = set(ad.read_h5ad(disc_files[0], backed="r").var_names)
    universe = set(disc_genes)
    ext_gene_sets = {}
    for f in ext_files:
        s = f.stem
        g = set(ad.read_h5ad(f, backed="r").var_names)
        ext_gene_sets[s] = g
        if s in ext_lung:
            universe &= g
    universe = sorted(universe)
    log(f"gene universe: {len(universe)} (discovery {len(disc_genes)}; external per-object {[len(v) for v in ext_gene_sets.values()]})")
    sd = stage_dir(STAGE)
    write_table(pd.DataFrame({"gene": universe}), sd / "gene_universe.tsv")
    tset = cfg["genes"]["t_cell_holdout"]
    write_table(pd.DataFrame({"gene": tset, "in_discovery": [g in disc_genes for g in tset], "in_universe": [g in universe for g in tset]}), sd / "t_cell_holdout_status.tsv")

    # ---- per-section QC + normalisation --------------------------------------------------
    rows, outputs = [], []
    for f in disc_files + ext_files:
        a = ad.read_h5ad(f)
        cohort = "discovery" if f.parent.name == "gse293065" else "external"
        a, info = qc_section(a, cfg)
        # keep EVERY gene of the object (T-cell readout and marker checks use the full set);
        # the cross-platform universe is flagged in var and is the only pool the model may draw from
        a.var["in_universe"] = a.var_names.isin(universe)
        info["n_universe_present"] = int(a.var["in_universe"].sum())
        info["n_t_holdout_present"] = int(sum(g in a.var_names for g in tset))
        a.layers["counts"] = a.X.copy()
        sc.pp.normalize_total(a, target_sum=cfg["genes"]["normalize_target_sum"])
        sc.pp.log1p(a)
        a.layers["lognorm"] = a.X.copy()
        a.obs["cohort"] = cohort
        a.write_h5ad(out_dir / f"{f.stem}.h5ad")
        outputs.append(out_dir / f"{f.stem}.h5ad")
        rows.append({"section": f.stem, "cohort": cohort, **info, "n_genes_universe": a.n_vars,
                     "median_counts_after": float(np.median(np.asarray(a.layers["counts"].sum(1)).ravel()))})
        log(f"{f.stem} ({cohort}): {info['n_spots_before']}→{info['n_spots_after']} spots, thr {info['thr_counts']:.0f}")
    write_table(pd.DataFrame(rows), sd / "section_qc_summary.tsv")
    write_provenance(STAGE, disc_files + ext_files, outputs + [sd / "gene_universe.tsv", sd / "section_qc_summary.tsv"], cfg,
                     extra={"n_genes_universe": len(universe)}, t_start=t0)


if __name__ == "__main__":
    main()
