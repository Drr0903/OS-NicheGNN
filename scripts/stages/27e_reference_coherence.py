"""Stage 27e () — spatial coherence of each reference annotation.

For every cohort of the stage-27 benchmark: the fraction of a scored spot's ring-1 neighbours that carry its class, against
the same fraction expected by chance from the class frequencies of the section (sum of squared class shares); section means.
Outputs results/27_benchmark/reference_coherence.tsv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.generality import spot_pitch  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.spatial import ring_neighbors  # noqa: E402

STAGE = "27_benchmark"; B = (ROOT / "data/benchmark")


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE); rows = []
    for cohort in ("os", "crc", "breast"):
        for f in sorted((B / cohort).glob("*.h5ad")):
            a = ad.read_h5ad(f); t = a.obs["truth"].astype(str).to_numpy(); xy = a.obsm["spatial"].astype(float); ok = t != "unscored"
            nb = ring_neighbors(xy, spot_pitch(xy), rings=(1,))
            same = [np.mean(t[n][ok[n]] == t[i]) for i, n in enumerate(nb) if ok[i] and ok[n].sum() > 0]
            p = np.unique(t[ok], return_counts=True)[1] / ok.sum()
            rows.append({"cohort": cohort, "section": f.stem, "neighbour_same_class": float(np.mean(same)), "expected_by_chance": float((p ** 2).sum())})
    d = pd.DataFrame(rows); s = d.groupby("cohort")[["neighbour_same_class", "expected_by_chance"]].mean().reset_index()
    write_table(s, sd / "reference_coherence.tsv"); print(s.round(3).to_string())
    write_provenance(STAGE + "_coherence", [], [sd / "reference_coherence.tsv"], cfg, t_start=t0)


if __name__ == "__main__":
    main()
