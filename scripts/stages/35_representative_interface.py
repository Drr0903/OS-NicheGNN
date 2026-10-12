"""Stage 35 — plotting inputs for the representative tumour–lung interfaces of Figure 5a.

For two lung sections with an evaluable interface and a high coverage of it by the peritumoural band, this stage writes the
cropped H&E image and one row per spot in cropped-image coordinates: atlas niche, band membership, interface spots, the T-cell
score (44 held-out genes), the macrophage score (CD68, CD163, C1QA, C1QB, MRC1, CSF1R; mean log-normalized expression, the
genes of stage 30) and the router weight of the host-tissue expert (mean over the five networks, stage 34).
Inputs : data/processed/{qc/<sec>.h5ad, qc_images/<sec>_hires.png, expert_routing/frozen_s*.parquet}, results/15_niches/
Outputs: results/35_representative_interface/{<sec>_hires_crop.png, <sec>_spots.tsv, PROVENANCE.json}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.interface import t_score  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "35_representative_interface"
SECTIONS = ["MR2", "MR18"]  # lung sections with an evaluable interface and band coverage of 0.19 and 0.23 (results/15_niches/interface_section_metrics.tsv)
MACROPHAGE = ["CD68", "CD163", "C1QA", "C1QB", "MRC1", "CSF1R"]


def host_weight(cfg: dict) -> pd.DataFrame:
    """Router weight of the host-tissue expert per spot, averaged over the five networks (expert oriented as in stage 34)."""
    spots = pd.read_csv(ROOT / "results/15_niches/spot_niches.tsv", sep="\t", usecols=["section", "barcode", "histology_label"]); per = []
    for f in sorted((ROOT / cfg["paths"]["processed"] / "expert_routing").glob("frozen_s*.parquet")):
        r = pd.read_parquet(f, columns=["section", "barcode", "gate_0", "gate_1"]).merge(spots, on=["section", "barcode"])
        lp = r[r["histology_label"] == "lung_parenchyma"]; g = "gate_0" if lp["gate_0"].mean() > lp["gate_1"].mean() else "gate_1"
        per.append(r.set_index(["section", "barcode"])[g])
    return pd.concat(per, axis=1).mean(1).rename("host_expert_weight").reset_index()


def export(cfg: dict, sec: str, sd: Path, niches: pd.DataFrame, hw: pd.DataFrame, band: int) -> Path:
    proc = ROOT / cfg["paths"]["processed"]
    img = np.asarray(Image.open(proc / "qc_images" / f"{sec}_hires.png").convert("RGB"))
    a = ad.read_h5ad(proc / "qc" / f"{sec}.h5ad"); xy = a.obsm["spatial"] * a.uns["scalefactors"]["tissue_hires_scalef"]
    ys, xs = np.where(img.mean(-1) < 225); pad = 30  # crop to the tissue, the rule of stage 04b
    r0, r1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad, img.shape[0]); c0, c1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad, img.shape[1])
    Image.fromarray(img[r0:r1, c0:c1]).save(sd / f"{sec}_hires_crop.png")
    d = pd.DataFrame({"barcode": list(a.obs_names), "x": xy[:, 0] - c0, "y": xy[:, 1] - r0, "t_score": t_score(a, cfg["genes"]["t_cell_holdout"]),
                      "macrophage_score": t_score(a, MACROPHAGE)})
    n = niches[niches["section"] == sec][["barcode", "niche", "unmatched", "tumor_lung_interface", "histology_label"]]
    d = d.merge(n, on="barcode", how="left").merge(hw[hw["section"] == sec][["barcode", "host_expert_weight"]], on="barcode", how="left")
    d["band"] = (d["niche"] == band) & ~d["unmatched"].fillna(True).astype(bool)
    f = sd / f"{sec}_spots.tsv"; write_table(d, f); log(f"{sec}: {len(d)} spots, {int(d['band'].sum())} band spots, {int(d['tumor_lung_interface'].fillna(False).sum())} interface spots")
    return f


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE)
    niches = pd.read_csv(ROOT / "results/15_niches/spot_niches.tsv", sep="\t"); band = json.load(open(ROOT / "results/15_niches/meta.json"))["interface_niche"]
    hw = host_weight(cfg); outs = [export(cfg, s, sd, niches, hw, band) for s in SECTIONS] + [sd / f"{s}_hires_crop.png" for s in SECTIONS]
    write_provenance(STAGE, [ROOT / "results/15_niches/spot_niches.tsv"], outs, cfg, extra={"sections": SECTIONS}, t_start=t0)


if __name__ == "__main__":
    main()
