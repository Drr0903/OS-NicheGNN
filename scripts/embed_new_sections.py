"""Embed new Visium sections with the trained OS-NicheGNN atlas encoder.

Input per section: a quality-controlled AnnData (layers["lognorm"], obsm["spatial"], uns["spot_pitch_fullres_px"],
obs["section"], obs["cohort"]) and its Phikon spot features (stage 04c, one row per spot). The frozen directory holds the
gene list, the scaler, the expression and morphology PCA models and the encoder weights (data/processed/<atlas.frozen_dir>, frozen_v5 for the published networks).
Output: one latent per spot (<out>/<section>.npy, 256 dimensions), the input of the niche assignment (Methods).
Usage: python scripts/embed_new_sections.py --h5ad new.h5ad --img new_img.npy --out embeddings/ [--seed 20260914]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.arch import NicheEncoder  # noqa: E402
from lib.arch_ext import attach  # noqa: E402
from lib.arch_train import embed_arch  # noqa: E402
from lib.data import build_graph  # noqa: E402
from lib.repro import ROOT, load_config  # noqa: E402


def load_encoder(cfg: dict, fro: Path, seed: int, in_dim: int, device: str) -> NicheEncoder:
    meta = json.load(open(fro / "meta.json")); ac = {**cfg["arch"], **(meta.get("arch_overrides") or {})}
    nt = tuple(t for t, w in (ac.get("niche_targets") or {}).items() if w > 0)
    model = NicheEncoder(in_dim, cfg["atlas"]["context"], ac["d"], ac["z_dim"], ac["heads"], ac["n_layers"], ac["dropout"], nt,
                         ac.get("linear_heads", False), ac.get("prop_skip", False), ac.get("grad_skip", False))
    model = attach(model, ac, in_dim); model.load_state_dict(torch.load(fro / f"model_seed{seed}.pt", map_location="cpu"))
    model.max_ring = int(ac.get("max_ring", 3)); return model.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--h5ad", nargs="+", required=True); ap.add_argument("--img", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=20260914); ap.add_argument("--frozen", default="")
    a = ap.parse_args(); cfg = load_config(); fro = Path(a.frozen) if a.frozen else ROOT / cfg["paths"]["processed"] / cfg["atlas"]["frozen_dir"]
    device = cfg["device"] if torch.cuda.is_available() else "cpu"; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    genes = open(fro / "genes.txt").read().split(); sc = np.load(fro / "scaler.npz"); P = joblib.load(fro / "pca.joblib"); model = None
    for h, f in zip(a.h5ad, a.img):
        adata = ad.read_h5ad(h); F = np.load(f); assert len(F) == adata.n_obs, f"{h}: image rows {len(F)} != spots {adata.n_obs}"
        g = build_graph(adata, genes, sc["mu"], sc["sd"], cfg, str(adata.obs["section"].iloc[0]), None)
        x = np.hstack([P["expr_pca"].transform(g.x) / P["expr_sd"], P["img_pca"].transform(F) / P["img_sd"]]).astype(np.float32)
        g = dataclasses.replace(g, x=x); model = model or load_encoder(cfg, fro, a.seed, x.shape[1], device)
        z = embed_arch(model, [g], device)[g.section]["z"]; np.save(out / f"{g.section}.npy", z.astype(np.float32))
        print(f"{g.section}: {z.shape[0]} spots embedded ({z.shape[1]} dimensions)")


if __name__ == "__main__":
    main()
