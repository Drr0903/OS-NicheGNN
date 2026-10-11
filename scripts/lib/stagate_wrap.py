"""STAGATE (Dong & Zhang, Nat Commun 2022) via the authors' STAGATE_pyG package, run TRANSDUCTIVELY.

STAGATE is a graph-attention autoencoder trained on one section at a time; it has no frozen-model transfer
mode, so for a held-out patient it is *refitted on that patient's own sections* (label-free) and the
resulting embedding is evaluated with the same linear probe (probe still fitted on training patients'
STAGATE embeddings — each obtained by fitting STAGATE on those sections). This is the "transductive /
within-section refit" column in the results tables and must not be read as inductive external validation.
Default hyper-parameters of the package: hidden_dims=[512, 30], 1000 epochs in the paper; we use the value
in config (500) for run time, identical for every section, and rad_cutoff = ring-1 distance.
"""
from __future__ import annotations

import os
import warnings

os.environ.setdefault("TQDM_DISABLE", "1")

import anndata as ad
import numpy as np
import pandas as pd
import torch


def stagate_embed(g, cfg: dict, seed: int, device: str = "cuda") -> np.ndarray:
    import STAGATE_pyG as ST

    st = cfg["baselines"]["stagate"]
    a = ad.AnnData(X=g.x.astype(np.float32), obs=pd.DataFrame(index=[str(b) for b in g.barcodes]))
    a.obsm["spatial"] = g.xy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ST.Cal_Spatial_Net(a, rad_cutoff=g.pitch * (st["rad_cutoff_rings"] + 0.4), verbose=False)
        torch.manual_seed(seed); np.random.seed(seed)
        a.var["highly_variable"] = True
        a = ST.train_STAGATE(a, hidden_dims=st["hidden"], n_epochs=st["epochs"], lr=st["lr"], random_seed=seed, device=torch.device(device), verbose=False)
    return np.asarray(a.obsm["STAGATE"], dtype=np.float32)
