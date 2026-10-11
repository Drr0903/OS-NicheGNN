"""Neighbourhood-composition probes ().

own_r2              held-out R² of the spot's own eleven-type RCTD composition predicted from the embedding
context_r2          held-out R² of the ring-2–3 mean composition (the spot's micro-environment) from the embedding
context_r2_partial  the same for the part of the ring-2–3 composition not linearly explained by the spot's own composition
                    (residuals of a regression fitted on the training patients), i.e. context information beyond composition
Ridge probes on standardised embeddings, trained on (a sample of) the training patients' spots, R² variance-weighted over
the eleven types and computed on all of the held-out patient's spots pooled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from .repro import ROOT
from .spatial import neighbor_mean, ring_neighbors


def load_weights(G: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per section: (own composition, ring-2–3 mean composition), rows aligned to the graph's spots (NaN where RCTD is missing)."""
    out = {}
    for s, g in G.items():
        w = pd.read_csv(ROOT / "results/05_deconvolution/weights" / f"{s}_rctd_weights.tsv", sep="\t").set_index("barcode").reindex(g.barcodes)
        Wm = w.to_numpy(float); ok = ~np.isnan(Wm).any(1); Wf = np.where(ok[:, None], Wm, 0.0)
        nb = ring_neighbors(g.xy, g.pitch, rings=(2, 3)); nb_ok = [v[ok[v]] for v in nb]
        C = np.asarray(neighbor_mean(Wf, nb_ok), float); valid = ok & np.array([len(v) > 0 for v in nb_ok])
        C[~valid] = np.nan; Wm[~ok] = np.nan; out[s] = (Wm, C)
    return out


def propagate(g) -> np.ndarray:
    """Non-learned graph propagation: [x, ring-1 mean, ring-2–3 mean] of the section's (PCA) features."""
    n1 = ring_neighbors(g.xy, g.pitch, rings=(1,)); n23 = ring_neighbors(g.xy, g.pitch, rings=(2, 3))
    return np.hstack([g.x, np.asarray(neighbor_mean(g.x, n1)), np.asarray(neighbor_mean(g.x, n23))]).astype(np.float32)


def _stack(emb, W, secs, which):
    Z = np.vstack([emb[s] for s in secs]); Y = np.vstack([W[s][which] for s in secs]); O = np.vstack([W[s][0] for s in secs])
    ok = ~np.isnan(Y).any(1) & ~np.isnan(O).any(1); return Z[ok], Y[ok], O[ok]


def context_r2(emb: dict, G: dict, W: dict, train: list[str], test: list[str], rc: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed); Ztr, Ctr, Otr = _stack(emb, W, train, 1); Zte, Cte, Ote = _stack(emb, W, test, 1)
    if len(Zte) < 50:
        return {"own_r2": np.nan, "context_r2": np.nan, "context_r2_partial": np.nan}
    if len(Ztr) > rc["context_max_train_spots"]:
        i = rng.choice(len(Ztr), rc["context_max_train_spots"], replace=False); Ztr, Ctr, Otr = Ztr[i], Ctr[i], Otr[i]
    sc = StandardScaler().fit(Ztr); A, B = sc.transform(Ztr), sc.transform(Zte)

    def r2(Ytr, Yte):
        pred = Ridge(alpha=rc["context_ridge_alpha"]).fit(A, Ytr).predict(B)
        keep = Yte.std(0) > 1e-6  # types absent from the held-out patient carry no variance to explain
        return float(r2_score(Yte[:, keep], pred[:, keep], multioutput="variance_weighted"))

    lr = LinearRegression().fit(Otr, Ctr)
    return {"own_r2": r2(Otr, Ote), "context_r2": r2(Ctr, Cte), "context_r2_partial": r2(Ctr - lr.predict(Otr), Cte - lr.predict(Ote))}
