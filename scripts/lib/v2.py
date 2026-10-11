"""Cohort helpers: all 26 discovery sections (19 patients, five sites), region labels harmonised across sites,
graphs on the discovery gene set, and the patient-invariant OS-NicheGNN configuration."""
from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .cluster import leiden_fixed_k
from .data import SectionGraph, build_graph, fit_scaler, holdout_genes, load_labels, load_qc_sections, select_hvgs
from .repro import ROOT
from .train import embed_sections, train_model
from .upgrade import mnn_anchors

CLASSES = ["tumor_cellular", "lung_parenchyma", "paucicellular"]


def load_cohort(cfg: dict) -> tuple[dict[str, ad.AnnData], dict[str, pd.DataFrame], dict[str, str], dict[str, str]]:
    v2 = cfg["v2"]; secs = [s for s in v2["sections"] if (ROOT / cfg["paths"]["processed"] / "qc" / f"{s}.h5ad").exists()]
    adatas = load_qc_sections(cfg, secs); labels = {}
    for s in secs:
        lab = load_labels(cfg, s).copy()
        if v2["site_of"][s] != "lung":  # the lung-marker class is not meaningful outside the lung: merge into non-tumour stroma/other
            lab.loc[lab["label"] == "lung_parenchyma", "label"] = "paucicellular"; lab["tumor_lung_interface"] = False
        labels[s] = lab
    return adatas, labels, {s: v2["patient_of"][s] for s in secs}, {s: v2["site_of"][s] for s in secs}


def make_graphs(cfg: dict, adatas: dict, labels: dict, patient_of: dict, train_secs: list[str], all_secs: list[str]) -> tuple[dict[str, SectionGraph], list[str], np.ndarray, np.ndarray]:
    """HVGs and scaler fitted on train_secs only (discovery gene set, T-cell and label genes excluded); graphs for all_secs."""
    genes = select_hvgs([adatas[s] for s in train_secs], cfg["genes"]["n_hvg"], holdout_genes(cfg), universe=False)
    mu, sd = fit_scaler([adatas[s] for s in train_secs], genes)
    return {s: build_graph(adatas[s], genes, mu, sd, cfg, patient_of[s], labels[s]) for s in all_secs}, genes, mu, sd


def compute_anchors(cfg: dict, train_graphs: list[SectionGraph], gate: bool | None = None) -> dict:
    inv = cfg["invariant"]; g = cfg["v2"]["model"].get("anchor_gate", False) if gate is None else gate
    return mnn_anchors(train_graphs, n_pcs=inv["anchor_pcs"], k=inv["anchor_k"], seed=cfg["seed"], gate=g)


def variant_weights(cfg: dict, kind_variant: str) -> dict:
    """Weights of the two positive terms encoded in the variant name (A10): _w0.1 / _w0.25 (both terms), _anchoronly, _nbronly."""
    m = cfg["v2"]["model"]; la, ln = m["lambda_anchor"], m["lambda_nbr"]
    if "_w" in kind_variant:
        w = float(kind_variant.split("_w")[1].split("_")[0]); la, ln = w, w
    if "anchoronly" in kind_variant:
        ln = 0.0
    if "nbronly" in kind_variant:
        la = 0.0
    return {"lambda_anchor": la, "lambda_nbr": ln, "lambda_con": m["lambda_con"]}


def invariant_upgrade(cfg: dict, train_graphs: list[SectionGraph], anchors: dict | None = None, kind_variant: str = "invariant") -> dict:
    w = variant_weights(cfg, kind_variant)
    return {**w, "anchors": (anchors if anchors is not None else compute_anchors(cfg, train_graphs)) if w["lambda_anchor"] > 0 else {}}


def harmony_correct(emb: dict[str, np.ndarray], patient_of: dict[str, str], seed: int) -> dict[str, np.ndarray]:
    """Harmony (harmonypy) on a stacked embedding with patient as the batch variable (A11 comparator); transductive by design."""
    import harmonypy as hm
    secs = list(emb); Z = np.vstack([emb[s] for s in secs]); meta = pd.DataFrame({"patient": np.concatenate([[patient_of[s]] * len(emb[s]) for s in secs])})
    ho = hm.run_harmony(Z, meta, "patient", random_state=seed, verbose=False); Zc = np.asarray(ho.Z_corr); Zc = Zc.T if Zc.shape[0] != Z.shape[0] else Zc
    out, off = {}, 0
    for s in secs:
        out[s] = Zc[off:off + len(emb[s])].astype(np.float32); off += len(emb[s])
    return out


def train_and_embed(cfg: dict, kind_variant: str, tr_g: list[SectionGraph], va_g: list[SectionGraph], all_g: list[SectionGraph], seed: int, device: str, cfg_train: dict | None = None, anchors: dict | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """kind_variant ∈ {"gnn", "invariant[...]", "hybrid_invariant[...]", "mlp", "mlp_invariant[...]"}; returns embeddings for all_g and the
    training info. Anchors (fold-specific) may be passed in. "mlp_*" (A11) uses the non-spatial MLP encoder with the same losses."""
    mlp = kind_variant.startswith("mlp"); kv = kind_variant[4:] if kind_variant.startswith("mlp_") else ("gnn" if mlp else kind_variant)
    base = kv.split("_")[0] if not kv.startswith("hybrid") else "hybrid_invariant"
    if anchors is None and "_gated" in kv:
        anchors = compute_anchors(cfg, tr_g, gate=True)
    upg = invariant_upgrade(cfg, tr_g, anchors, kv) if base in ("invariant", "hybrid_invariant") else None
    kind = "mlp_ae" if mlp else ("nichegnn_hybrid" if base == "hybrid_invariant" else "nichegnn")
    model, info = train_model(kind, tr_g, va_g, cfg_train or cfg, seed, device=device, upgrade=upg)
    return embed_sections(model, all_g, kind, cfg, device), info | {"model": model, "kind": kind}


def identity_index(emb: dict[str, np.ndarray], G: dict[str, SectionGraph], K: int, seed: int, per_patient: int = 800) -> dict:
    """Balanced-Leiden communities (fixed K) on a patient-balanced sample → ARI vs patient, ARI vs region, PII, presence."""
    rng = np.random.default_rng(seed); pats = sorted({g.patient for g in G.values()}); Z, P, L = [], [], []
    for pat in pats:
        secs = [s for s in G if G[s].patient == pat]; zs = np.vstack([emb[s] for s in secs]); ls = np.concatenate([G[s].labels for s in secs])
        idx = rng.choice(len(zs), min(per_patient, len(zs)), replace=False); Z.append(zs[idx]); P += [pat] * len(idx); L.append(ls[idx])
    Z = np.vstack(Z); P = np.array(P); L = np.concatenate(L); lab, _ = leiden_fixed_k(Z, K, seed, batch=P, k_within=3); ev = np.isin(L, CLASSES)
    ari_p = adjusted_rand_score(P, lab); ari_r = adjusted_rand_score(L[ev], lab[ev]); frac = pd.crosstab(P, lab, normalize="index")
    return {"ari_vs_patient": float(ari_p), "ari_vs_region": float(ari_r), "pii": float(ari_p / max(ari_p + ari_r, 1e-9)), "n_present_half_patients": int(((frac >= 0.02).sum(0) >= len(pats) / 2).sum()), "K": K}
