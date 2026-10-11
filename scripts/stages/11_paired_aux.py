"""Stage 11 — auxiliary, descriptive comparison of the four primary–lung-metastasis pairs (PT1–4 vs MR1–4).

Pre-defined tumour programmes are scored on tumour-cellular spots (image label) of each section and summarised
per section (mean z-score across spots); MR − PT differences are reported per pair with no inferential claim
(n = 4 pairs, different organs, treatment between samples). The frozen niche model is also applied to the
primary sections and niche fractions are reported descriptively — primaries have no lung interface, so no
interface metric is computed for them.
Outputs results/11_paired/{programme_scores.tsv, paired_differences.tsv, primary_niche_fractions.tsv}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data import build_graph, load_labels  # noqa: E402
from lib.repro import frozen_name, ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.train import embed_sections, make_model  # noqa: E402

STAGE = "11_paired"
PROGRAMMES = {
    "proliferation": ["MKI67", "TOP2A", "CDK1", "CCNB1", "BIRC5", "UBE2C", "PCNA", "TYMS"],
    "osteoblastic": ["RUNX2", "SP7", "ALPL", "IBSP", "SPP1", "BGLAP", "COL1A1", "SPARC"],
    "chondroblastic": ["SOX9", "COL2A1", "ACAN", "COL9A1", "COL11A1", "COMP"],
    "ecm_fibroblastic": ["COL3A1", "FN1", "POSTN", "COL5A1", "COL6A3", "TNC", "LUM"],
    "hypoxia": ["VEGFA", "CA9", "SLC2A1", "LDHA", "PGK1", "NDRG1", "BNIP3", "ADM"],
    "interferon": ["ISG15", "IFI6", "IFIT1", "IFIT3", "MX1", "STAT1", "IFI27", "OAS1"],
    "mhc_i": ["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "PSMB9"],
    "tgfb_response": ["TGFB1", "TGFBI", "SERPINE1", "CTGF", "CCN2", "THBS1", "COL4A1", "LTBP2"],
}


def programme_scores(a: ad.AnnData, mask: np.ndarray) -> dict:
    X = a.layers["lognorm"]
    out = {}
    for p, genes in PROGRAMMES.items():
        g = [x for x in genes if x in a.var_names]
        if not g:
            out[p] = np.nan; continue
        v = np.asarray(X[:, [a.var_names.get_loc(x) for x in g]].todense())
        z = (v - v.mean(0)) / (v.std(0) + 1e-6)
        out[p] = float(z[mask].mean()) if mask.any() else np.nan
        out[f"n_genes_{p}"] = len(g)
    return out


def main() -> None:
    t0 = time.time()
    cfg = load_config(); set_global_seed(cfg["seed"])
    proc = ROOT / cfg["paths"]["processed"]; qc = proc / "qc"
    pairs = cfg["cohorts"]["discovery"]["paired_primary"]
    secs = list(pairs) + list(pairs.values())
    rows = []
    for s in secs:
        a = ad.read_h5ad(qc / f"{s}.h5ad"); lab = load_labels(cfg, s)
        lab = lab.reindex(a.obs_names) if lab is not None else None
        tumour = (lab["label"] == "tumor_cellular").to_numpy() if lab is not None else np.ones(a.n_obs, bool)
        rows.append({"section": s, "site": "primary_bone" if s.startswith("PT") else "lung_metastasis", "n_spots": a.n_obs, "n_tumor_spots": int(tumour.sum()), **programme_scores(a, tumour)})
    ps = pd.DataFrame(rows)
    sd = stage_dir(STAGE); write_table(ps, sd / "programme_scores.tsv")
    diff = []
    for pt, mr in pairs.items():
        r1, r2 = ps.set_index("section").loc[pt], ps.set_index("section").loc[mr]
        diff.append({"pair": f"{pt}-{mr}", **{p: float(r2[p] - r1[p]) for p in PROGRAMMES}})
    d = pd.DataFrame(diff); write_table(d, sd / "paired_differences.tsv")
    log(f"MR − PT programme differences (n=4 pairs):\n{d.round(2).to_string()}")
    # frozen niche model on primaries (descriptive)
    fro = proc / frozen_name(); meta = json.load(open(fro / "meta.json"))
    genes = open(fro / "genes.txt").read().split(); sc_ = np.load(fro / "scaler.npz")
    model = make_model("nichegnn", len(genes), cfg).to(cfg["device"]); model.load_state_dict(torch.load(fro / f"model_seed{meta['frozen_seed']}.pt", map_location=cfg["device"]))
    protos = np.load(fro / "prototypes.npy")
    frac_rows = []
    for s in list(pairs):
        a = ad.read_h5ad(qc / f"{s}.h5ad"); g = build_graph(a, genes, sc_["mu"], sc_["sd"], cfg, f"P{s}", load_labels(cfg, s))
        z = embed_sections(model, [g], "nichegnn", cfg, cfg["device"])[s]
        dmat = np.linalg.norm(z[:, None, :] - protos[None], axis=2); lab = dmat.argmin(1); un = dmat.min(1) > meta["unmatched_threshold"]
        fr = pd.Series(lab).value_counts(normalize=True).reindex(range(meta["K"]), fill_value=0)
        frac_rows.append({"section": s, "n_spots": a.n_obs, "frac_unmatched": float(un.mean()), **{f"niche_{k}": float(fr[k]) for k in range(meta["K"])}})
    pf = pd.DataFrame(frac_rows); write_table(pf, sd / "primary_niche_fractions.tsv")
    log(f"primary niche fractions:\n{pf.round(3).to_string()}")
    write_provenance(STAGE, [qc], [sd / "programme_scores.tsv", sd / "paired_differences.tsv", sd / "primary_niche_fractions.tsv"], cfg, t_start=t0)


if __name__ == "__main__":
    main()
