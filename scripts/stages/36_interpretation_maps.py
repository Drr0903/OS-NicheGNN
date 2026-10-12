"""Stage 36 — plotting inputs for the interpretability figure (Figure 3a, c).

Router-weight maps: for one section per site (PT2 bone, MR2 lung, MR14 lymph node, LR17 soft tissue) the cropped H&E image
and one row per spot in cropped-image coordinates with the atlas niche, the region label and the router weight of the
host-tissue expert (mean over the five networks, stage 34 spot files; expert oriented as in stage 34).
Attention example: two spots of MR2, a peritumoural-band spot on the tumour–lung interface and a tumour spot far from the
interface, both with all 36 neighbours in tissue; for each neighbour the last-layer attention weight (heads averaged),
mean and s.d. over the five networks of the delivered encoder.
Inputs : data/processed/{qc, qc_images, image_features, expert_routing/frozen_s*.parquet, <frozen_dir>}, results/15_niches/
Outputs: results/36_interpretation_maps/{<sec>_hires_crop.png, <sec>_spots.tsv, attention_example.tsv, PROVENANCE.json}
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.arch_train import prep  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.v2 import load_cohort, make_graphs  # noqa: E402

STAGE = "36_interpretation_maps"
SECTIONS = ["PT2", "MR2", "MR14", "LR17"]  # one section per site: bone primary, lung, lymph node, soft tissue
EXAMPLE = "MR2"


def _load(name, file):
    sp = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file)); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def maps(cfg, sd: Path, s35) -> list[Path]:
    """Cropped H&E and per-spot table (niche, label, band, host-tissue weight) for each section, by the stage-35 export."""
    niches = pd.read_csv(ROOT / "results/15_niches/spot_niches.tsv", sep="\t"); band = json.load(open(ROOT / "results/15_niches/meta.json"))["interface_niche"]
    hw = s35.host_weight(cfg); outs = []
    for sec in SECTIONS:
        f = s35.export(cfg, sec, sd, niches, hw, band); outs += [f, sd / f"{sec}_hires_crop.png"]
    return outs


def pick(d: pd.DataFrame, nvalid: np.ndarray) -> dict[str, int]:
    """Row indices of the two example spots: the interface band spot nearest the centroid of the interface band spots, and the
    tumour spot farthest from every interface spot; both with all 36 neighbours in tissue."""
    full = nvalid == 36; xy = d[["x", "y"]].to_numpy()
    ib = np.where(full & d["band"].to_numpy(bool) & d["tumor_lung_interface"].fillna(False).to_numpy(bool))[0]
    ifc = np.where(d["tumor_lung_interface"].fillna(False).to_numpy(bool))[0]
    tu = np.where(full & (d["histology_label"] == "tumor_cellular").to_numpy())[0]
    c = xy[ib].mean(0); b = ib[np.argmin(((xy[ib] - c) ** 2).sum(1))]
    dist = np.sqrt(((xy[tu][:, None] - xy[ifc][None]) ** 2).sum(-1)).min(1); t = tu[np.argmax(dist)]
    return {"interface band spot": int(b), "tumour spot": int(t)}


@torch.no_grad()
def attention_example(cfg, sd: Path) -> Path:
    s28 = _load("s28", "28_interpretation.py"); fro = ROOT / cfg["paths"]["processed"] / cfg["atlas"]["frozen_dir"]; pcs = joblib.load(fro / "pca.joblib")
    adatas, labels, patient_of, _ = load_cohort(cfg); secs = list(adatas); G, genes, _, _ = make_graphs(cfg, adatas, labels, patient_of, secs, secs)
    assert genes == open(fro / "genes.txt").read().split(), "gene list differs from the delivered encoder"
    g = G[EXAMPLE]; F = np.load(ROOT / cfg["paths"]["processed"] / "image_features" / f"{EXAMPLE}.npy")
    g = dataclasses.replace(g, x=np.hstack([pcs["expr_pca"].transform(g.x) / pcs["expr_sd"], pcs["img_pca"].transform(F) / pcs["img_sd"]]).astype(np.float32))
    d = pd.read_csv(sd / f"{EXAMPLE}_spots.tsv", sep="\t").set_index("barcode").loc[list(g.barcodes)].reset_index()
    s = prep(g, cfg["device"]); nvalid = s.nb.valid.sum(1).cpu().numpy(); ex = pick(d, nvalid)
    rows = torch.tensor(list(ex.values()), device=cfg["device"]); W = []
    for f in sorted(fro.glob("model_seed*.pt")):
        m = s28.load_encoder(cfg, fro, int(f.stem.replace("model_seed", "")), g.x.shape[1]); W.append(m.attention_rows(s.x, s.nb, rows)[:, -1].cpu().numpy())
    W = np.stack(W); idx = s.nb.idx[rows].cpu().numpy(); ring = s.nb.pos[rows][..., 2:5].argmax(-1).cpu().numpy() + 1; out = []
    for k, (name, r) in enumerate(ex.items()):
        for t in range(idx.shape[1]):
            if idx[k, t] >= 0:
                out.append({"example": name, "centre_barcode": d.loc[r, "barcode"], "neighbour_barcode": d.loc[idx[k, t], "barcode"], "ring": int(ring[k, t]),
                            "alpha_mean": float(W[:, k, t].mean()), "alpha_sd": float(W[:, k, t].std())})
    f = sd / "attention_example.tsv"; write_table(pd.DataFrame(out), f); log(f"attention example: {ex}"); return f


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE)
    outs = maps(cfg, sd, _load("s35", "35_representative_interface.py")); outs.append(attention_example(cfg, sd))
    write_provenance(STAGE, [ROOT / "results/15_niches/spot_niches.tsv"], outs, cfg, extra={"sections": SECTIONS, "example_section": EXAMPLE}, t_start=t0)


if __name__ == "__main__":
    main()
