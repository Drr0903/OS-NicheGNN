"""Stage 27c ( addendum) — the osteosarcoma cohort in the stage-27 benchmark format.

Reference labels come from the source atlas authors: their ICA spot embeddings (GSE293065) and their curated component
annotation (Supplementary Data of the atlas paper, Use = True). Component weights are z-scored over all spots; a spot takes
the class of its highest-scoring component (tumour / lung epithelium / stroma-vascular / myeloid / lymphoid), unscored if
that z is below 1. Expression and H&E features are the study's own QC objects and stage-04c features.
Outputs <benchmark_dir>/os/<section>.h5ad (+ _img.npy); results/27_benchmark/os_reference_{mapping,counts}.tsv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "27_benchmark"; OUT = (ROOT / "data/benchmark/os")
LUNG = {"Pneumocytes", "Ciliated", "Clara cell", "Airway goblet", "Serous cells", "Epithelial", "Parenchyme"}


def ic_class(t: str, broad: str) -> str:
    if t == "Cancer":
        return "tumour"
    if t == "Myeloid":
        return "myeloid"
    if t == "Lymphoid":
        return "lymphoid"
    return "lung epithelium" if broad in LUNG else "stroma / vascular"


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE); OUT.mkdir(parents=True, exist_ok=True)
    ann_f = ROOT / "data/refs/PMC13470005/suppl/41467_2026_74603_MOESM3_ESM.xlsx"; emb_f = ROOT / cfg["paths"]["raw_gse293065"] / ".." / "GSE293065_ica.cell.embeddings.all.csv.gz"
    ann = pd.read_excel(ann_f, header=1); ann = ann[ann["Use"].astype(str) == "True"]; cls = {r.IC: ic_class(r.Type, str(r.broad)) for r in ann.itertuples()}
    E = pd.read_csv(emb_f, index_col=0)[list(cls)]; Z = (E - E.mean()) / E.std()
    top = Z.to_numpy().argmax(1); zmax = Z.to_numpy().max(1)
    lab = pd.Series(np.where(zmax >= 1, [cls[Z.columns[i]] for i in top], "unscored"), index=[i.split("_", 1)[0] + "|" + i.split("_", 1)[1] for i in Z.index])
    write_table(pd.DataFrame({"IC": list(cls), "type": [ann.set_index("IC").loc[c, "Type"] for c in cls], "broad": [ann.set_index("IC").loc[c, "broad"] for c in cls], "class": list(cls.values())}), sd / "os_reference_mapping.tsv")
    proc = ROOT / cfg["paths"]["processed"]; rows, outs = [], []
    for s in cfg["v2"]["sections"]:
        a = ad.read_h5ad(proc / "qc" / f"{s}.h5ad"); F = np.load(proc / "image_features" / f"{s}.npy"); assert len(F) == a.n_obs
        a.obs["patient"] = cfg["v2"]["patient_of"][s]; a.obs["truth"] = lab.reindex([f"{s}|{b}" for b in a.obs_names]).fillna("unscored").to_numpy()
        a.obs["raw_label"] = a.obs["truth"]
        if "lognorm" not in a.layers:
            raise SystemExit(f"{s}: QC object has no lognorm layer")
        a.write_h5ad(OUT / f"{s}.h5ad"); np.save(OUT / f"{s}_img.npy", F.astype(np.float32)); outs += [OUT / f"{s}.h5ad"]
        rows.append({"section": s, "patient": a.obs["patient"].iloc[0], "n_spots": a.n_obs, **a.obs["truth"].value_counts().to_dict()})
        log(f"{s}: {a.obs['truth'].value_counts().to_dict()}")
    write_table(pd.DataFrame(rows).fillna(0), sd / "os_reference_counts.tsv")
    write_provenance(STAGE + "_os", [ann_f, emb_f], outs + [sd / "os_reference_counts.tsv"], cfg, extra={"amendment": "A21"}, t_start=t0)


if __name__ == "__main__":
    main()
