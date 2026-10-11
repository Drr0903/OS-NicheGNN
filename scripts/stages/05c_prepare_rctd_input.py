"""Stage 05c — export QC-passed section counts for RCTD (R cannot read h5ad here).

Writes data/processed/rctd_input/<sec>/{counts.mtx (genes × spots), genes.tsv, barcodes.tsv, coords.tsv}
using the ORIGINAL full gene set of each object (not the cross-platform universe): RCTD benefits from all
genes shared with the reference, and deconvolution is interpretation, not model input.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import anndata as ad
import pandas as pd
import scipy.io
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, log, set_global_seed, write_provenance  # noqa: E402


def main() -> None:
    t0 = time.time()
    cfg = load_config(); set_global_seed(cfg["seed"])
    proc = ROOT / cfg["paths"]["processed"]
    out_root = proc / "rctd_input"; out_root.mkdir(exist_ok=True)
    qc_files = sorted((proc / "qc").glob("*.h5ad"))
    outputs = []
    for f in qc_files:
        q = ad.read_h5ad(f, backed="r")
        keep = q.obs_names
        src = proc / ("gse293065" if q.obs["cohort"].iloc[0] == "discovery" else "external") / f"{f.stem}.h5ad"
        a = ad.read_h5ad(src)[keep].copy()
        d = out_root / f.stem; d.mkdir(exist_ok=True)
        scipy.io.mmwrite(d / "counts.mtx", sp.csr_matrix(a.X).T.tocoo())
        pd.Series(a.var_names).to_csv(d / "genes.tsv", index=False, header=False)
        pd.Series(a.obs_names).to_csv(d / "barcodes.tsv", index=False, header=False)
        pd.DataFrame({"x": a.obsm["spatial"][:, 0], "y": a.obsm["spatial"][:, 1]}, index=a.obs_names).to_csv(d / "coords.tsv", sep="\t")
        outputs.append(d / "counts.mtx")
        log(f"{f.stem}: {a.n_obs} spots × {a.n_vars} genes exported")
    write_provenance("05_deconvolution_input", qc_files, outputs, cfg, t_start=t0)


if __name__ == "__main__":
    main()
