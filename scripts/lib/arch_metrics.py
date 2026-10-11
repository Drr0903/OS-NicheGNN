"""third set — held-out-patient metrics of what a niche representation is for.

  tcell_rho / tcell_nbhd_rho      Spearman ρ between ridge-predicted and observed T-cell score of the spot / of its ring-1–3 mean
  comp_rho / comp_nbhd_rho        mean over the eleven RCTD types of ρ for the spot's own / its ring-2–3 composition
  chaos / pas                     spatial coherence of niches transferred to the held-out sections (k-means K on training patients)
  mix                             fraction of held-out spots' 30 nearest neighbours from other patients ÷ its value under perfect mixing
Ridge probes are fitted on (a sample of) the training patients' spots on standardised representations; ρ is computed per held-out
section and averaged over the patient's sections. Nothing here is fitted on the held-out patient.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .spatial import neighbor_mean, ring_neighbors


def targets(G: dict, tscore: dict, W: dict) -> dict:
    """Per section: T score, ring-1–3 mean T score, own composition, ring-2–3 composition (NaN rows where RCTD is missing)."""
    out = {}
    for s, g in G.items():
        nb = ring_neighbors(g.xy, g.pitch, rings=(1, 2, 3)); t = tscore[s]
        out[s] = {"t": t, "t_nb": np.asarray(neighbor_mean(t[:, None], nb)).ravel(), "c": W[s][0], "c_nb": W[s][1]}
    return out


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    ok = ~np.isnan(a) & ~np.isnan(b)
    return float(spearmanr(a[ok], b[ok])[0]) if ok.sum() > 20 and np.std(b[ok]) > 0 and np.std(a[ok]) > 0 else np.nan


def probe_rhos(emb: dict, T: dict, train: list[str], test: list[str], seed: int, n_max: int = 30000) -> dict:
    rng = np.random.default_rng(seed); Z = np.vstack([emb[s] for s in train]); sc = StandardScaler().fit(Z); idx = rng.choice(len(Z), min(n_max, len(Z)), replace=False)
    out = {}
    for key, name, multi in [("t", "tcell_rho", False), ("t_nb", "tcell_nbhd_rho", False), ("c", "comp_rho", True), ("c_nb", "comp_nbhd_rho", True)]:
        Y = np.vstack([T[s][key].reshape(len(emb[s]), -1) for s in train])[idx]; X = sc.transform(Z[idx]); ok = ~np.isnan(Y).any(1)
        r = Ridge(alpha=1.0).fit(X[ok], Y[ok]); vals = []
        for s in test:
            P = r.predict(sc.transform(emb[s])); Yt = T[s][key].reshape(len(emb[s]), -1)
            v = [_rho(P[:, j], Yt[:, j]) for j in range(Yt.shape[1])]; vals.append(np.nanmean(v) if not np.all(np.isnan(v)) else np.nan)
        out[name] = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else np.nan
    return out


def chaos_pas(lab: np.ndarray, xy: np.ndarray, pitch: float) -> tuple[float, float]:
    d_tot, n = 0.0, 0
    for k in np.unique(lab):
        p = xy[lab == k]
        if len(p) < 2:
            continue
        d, _ = cKDTree(p).query(p, k=2); d_tot += d[:, 1].sum() / pitch; n += len(p)
    _, nn = cKDTree(xy).query(xy, k=min(11, len(xy)))
    pas = float(np.mean((lab[nn[:, 1:]] != lab[:, None]).sum(1) > 6))
    return d_tot / max(n, 1), pas


def coherence(emb: dict, G: dict, train: list[str], test: list[str], K: int, seed: int, n_max: int = 30000) -> dict:
    rng = np.random.default_rng(seed); Z = np.vstack([emb[s] for s in train]); sc = StandardScaler().fit(Z)
    km = KMeans(K, n_init=5, random_state=seed).fit(sc.transform(Z[rng.choice(len(Z), min(n_max, len(Z)), replace=False)]))
    c, p = zip(*[chaos_pas(km.predict(sc.transform(emb[s])), G[s].xy, G[s].pitch) for s in test])
    return {"chaos": float(np.mean(c)), "pas": float(np.mean(p))}


def mixing(emb: dict, G: dict, train: list[str], test: list[str], seed: int, per_patient: int = 800, k: int = 30) -> dict:
    rng = np.random.default_rng(seed); pats = sorted({G[s].patient for s in train + test}); held = G[test[0]].patient; Z, P = [], []
    for pt in pats:
        zs = np.vstack([emb[s] for s in train + test if G[s].patient == pt]); i = rng.choice(len(zs), min(per_patient, len(zs)), replace=False); Z.append(zs[i]); P += [pt] * len(i)
    Z = StandardScaler().fit_transform(np.vstack(Z)); P = np.array(P); h = np.where(P == held)[0]
    _, nn = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z[h]); other = (P[nn[:, 1:]] != held).mean()
    expected = (len(P) - len(h)) / (len(P) - 1)
    return {"mix": float(other / expected)}


def niche_metrics(emb: dict, G: dict, T: dict, train: list[str], test: list[str], K: int, seed: int) -> dict:
    return {**probe_rhos(emb, T, train, test, seed), **coherence(emb, G, train, test, K, seed), **mixing(emb, G, train, test, seed)}
