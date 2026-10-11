"""generality of the patient-identity finding on other multi-patient Visium datasets.

Dataset-agnostic helpers: build SectionGraphs from per-section AnnData (lognorm layer, spatial coordinates, coarse region
label), fit the representations used in the main study on one dataset (standardised expression, PCA, BANKSY, OS-NicheGNN,
view-contrastive variant), cluster them three ways with a fixed K (k-means, plain-kNN Leiden, patient-balanced Leiden) and
report ARI vs patient, ARI vs region, cluster presence across patients and the patient-identity index
PII = ARI_patient / (ARI_patient + ARI_region).
"""
from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from .banksy import BanksyEmbedder
from .cluster import leiden_fixed_k
from .data import SectionGraph, fit_scaler, select_hvgs
from .spatial import ring_edge_index, ring_neighbors
from .train import embed_sections, train_model


def qc_and_lognorm(a: ad.AnnData, min_genes: int = 100, pct_low: float = 1.0) -> ad.AnnData:
    """Same rule as stage 03: drop spots below the 1st percentile of total counts or with < min_genes genes; lognorm to 1e4."""
    sc.pp.filter_genes(a, min_cells=3)
    tot = np.asarray(a.X.sum(1)).ravel(); ng = np.asarray((a.X > 0).sum(1)).ravel()
    keep = (tot >= np.percentile(tot, pct_low)) & (ng >= min_genes); a = a[keep].copy()
    a.layers["counts"] = a.X.copy(); sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a); a.layers["lognorm"] = sp.csr_matrix(a.X)
    return a


def spot_pitch(xy: np.ndarray) -> float:
    from sklearn.neighbors import NearestNeighbors
    d, _ = NearestNeighbors(n_neighbors=2).fit(xy).kneighbors(xy); return float(np.median(d[:, 1]))


def make_graphs(adatas: dict[str, ad.AnnData], patient_of: dict[str, str], cfg: dict, n_hvg: int = 2000) -> tuple[dict[str, SectionGraph], list[str]]:
    genes = select_hvgs(list(adatas.values()), n_hvg); mu, sd = fit_scaler(list(adatas.values()), genes)
    G = {}
    for s, a in adatas.items():
        X = np.asarray(a[:, genes].layers["lognorm"].todense(), dtype=np.float32); X = np.clip((X - mu) / sd, -10, 10)
        xy = a.obsm["spatial"].astype(float); pitch = spot_pitch(xy); g = cfg["graph"]
        sg = SectionGraph(section=s, patient=patient_of[s], cohort="generality", x=X, edge_near=ring_edge_index(xy, pitch, rings=(g["near_rings"],)),
                          edge_outer=ring_edge_index(xy, pitch, rings=tuple(g["outer_rings"])), xy=xy, pitch=pitch, barcodes=a.obs_names.to_numpy())
        lab = a.obs["region"].to_numpy(dtype=object); sg.labels = lab
        nb = ring_neighbors(xy, pitch, rings=(1,)); tum = lab == "tumour"
        sg.boundary = np.array([tum[i] and (len(nb[i]) > 0 and (~tum[nb[i]]).any()) for i in range(len(lab))]); sg.interface = np.zeros(len(lab), bool)
        G[s] = sg
    return G, genes


def invariant_representations(G: dict[str, SectionGraph], cfg: dict, device: str, n_seeds: int, log=print) -> dict[str, dict[str, np.ndarray]]:
    """ variant (spatial-neighbour + cross-patient anchor positives) trained on one dataset."""
    from .upgrade import mnn_anchors
    secs = list(G); inv = cfg["invariant"]; anchors = mnn_anchors([G[s] for s in secs], n_pcs=inv["anchor_pcs"], k=inv["anchor_k"], seed=cfg["seed"]); reps = {}
    for i in range(n_seeds):
        seed = cfg["seed"] + i
        model, info = train_model("nichegnn", [G[s] for s in secs], [], json.loads(json.dumps(cfg)), seed, device=device, upgrade={"lambda_anchor": inv["lambda_anchor_grid"][-1], "lambda_nbr": inv["lambda_nbr"], "anchors": anchors})
        reps[f"OS-NicheGNN + invariant seed {i + 1}"] = embed_sections(model, [G[s] for s in secs], "nichegnn", cfg, device); log(f"    invariant seed {i + 1}: {info['epochs_run']} epochs")
    return reps


def representations(G: dict[str, SectionGraph], cfg: dict, device: str, n_seeds: int, contrastive_lambda: float, log=print) -> dict[str, dict[str, np.ndarray]]:
    secs = list(G); reps = {"standardised expression": {s: G[s].x for s in secs}}
    with threadpool_limits(limits=8):
        pca = PCA(cfg["baselines"]["pca_dim"], random_state=cfg["seed"]).fit(np.vstack([G[s].x for s in secs])); reps["PCA-50"] = {s: pca.transform(G[s].x) for s in secs}
        b = cfg["baselines"]["banksy"]; be = BanksyEmbedder(b["lambda"], b["k_geom"], b["n_pcs"], cfg["seed"]).fit([G[s] for s in secs]); reps["BANKSY-20"] = {s: be.transform(G[s]) for s in secs}
    cfg_t = json.loads(json.dumps(cfg))
    for i in range(n_seeds):
        seed = cfg["seed"] + i
        model, info = train_model("nichegnn", [G[s] for s in secs], [], cfg_t, seed, device=device); reps[f"OS-NicheGNN seed {i + 1}"] = embed_sections(model, [G[s] for s in secs], "nichegnn", cfg, device)
        log(f"    GNN seed {i + 1}: {info['epochs_run']} epochs")
        model, info = train_model("nichegnn", [G[s] for s in secs], [], cfg_t, seed, device=device, upgrade={"lambda_con": contrastive_lambda}); reps[f"OS-NicheGNN + contrastive seed {i + 1}"] = embed_sections(model, [G[s] for s in secs], "nichegnn", cfg, device)
    return reps


def cluster_diagnostics(name: str, emb: dict[str, np.ndarray], G: dict[str, SectionGraph], K: int, seed: int, classes: list[str], per_patient: int = 1500) -> list[dict]:
    """Three clusterings at fixed K on a patient-balanced sample; labels propagated to all spots by nearest centroid."""
    rng = np.random.default_rng(seed); secs = list(G); pat_of = {s: G[s].patient for s in secs}
    Z, P = [], []
    for pat in sorted(set(pat_of.values())):
        zs = np.vstack([emb[s] for s in secs if pat_of[s] == pat]); idx = rng.choice(len(zs), min(per_patient, len(zs)), replace=False); Z.append(zs[idx]); P += [pat] * len(idx)
    Z = np.vstack(Z); P = np.array(P)
    Zall = np.vstack([emb[s] for s in secs]); Pall = np.concatenate([[pat_of[s]] * G[s].n for s in secs]); Lall = np.concatenate([G[s].labels for s in secs]); ev = np.isin(Lall, classes)
    out = []
    for method in ["k-means", "Leiden (plain kNN)", "Leiden (patient-balanced kNN)"]:
        with threadpool_limits(limits=8):
            if method == "k-means":
                lab = KMeans(K, n_init=10, random_state=seed).fit_predict(Z)
            elif method == "Leiden (plain kNN)":
                lab, _ = leiden_fixed_k(Z, K, seed)
            else:
                lab, _ = leiden_fixed_k(Z, K, seed, batch=P, k_within=3)
        cents = np.vstack([Z[lab == c].mean(0) for c in range(lab.max() + 1)]); la = np.linalg.norm(Zall[:, None, :] - cents[None], axis=2).argmin(1)
        ari_p = adjusted_rand_score(Pall, la); ari_r = adjusted_rand_score(Lall[ev], la[ev])
        frac = pd.crosstab(Pall, la, normalize="index"); present = int(((frac >= 0.02).sum(0) >= frac.shape[0] / 2).sum())
        out.append({"representation": name, "clustering": method, "K": K, "ari_vs_patient": float(ari_p), "ari_vs_region": float(ari_r), "pii": float(ari_p / max(ari_p + ari_r, 1e-9)),
                    "n_clusters_present_in_half_patients": present, "mean_max_patient_share": float(pd.crosstab(la, Pall, normalize="index").max(1).mean())})
    return out
