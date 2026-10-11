"""Frozen-representation evaluation shared by every method.

Region task   : identical L2-regularised multinomial logistic regression ("linear probe") trained on the
                embeddings + image-derived labels of the TRAINING patients, applied to the held-out patient.
                Classes: tumor_cellular / lung_parenchyma / paucicellular (unevaluable spots excluded).
Boundary task : tumour-boundary spots predicted from the probe's labels are compared with the image-derived
                boundary with a tolerance of `boundary_tolerance_rings` hex rings (default 1):
                a predicted boundary spot counts as a hit if a true boundary spot lies within tolerance and
                vice-versa → boundary precision / recall / F1.
All metrics are reported per held-out patient (sections of one patient pooled), never as pooled spots. Patients whose
held-out sections contain a single region class (e.g. an all-tumour section) get accuracy only; macro-F1 and boundary
F1 are NaN for them and they are excluded from those means.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from .data import SectionGraph
from .spatial import ring_neighbors

CLASSES = ["tumor_cellular", "lung_parenchyma", "paucicellular"]


def fit_probe(Z: np.ndarray, y: np.ndarray, seed: int):
    sc = StandardScaler().fit(Z)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=seed).fit(sc.transform(Z), y)
    return sc, clf


def boundary_from_labels(lab: np.ndarray, xy: np.ndarray, pitch: float) -> np.ndarray:
    nb = ring_neighbors(xy, pitch, rings=(1,))
    is_t = lab == "tumor_cellular"; ev = np.isin(lab, CLASSES)
    return np.array([bool(is_t[i] and any(ev[j] and not is_t[j] for j in nb[i])) for i in range(len(lab))])


def boundary_prf(pred_b: np.ndarray, true_b: np.ndarray, xy: np.ndarray, pitch: float, tol_rings: int) -> dict:
    tol = (tol_rings + 0.45) * pitch
    if true_b.sum() < 30 or pred_b.sum() == 0:  # < 30 true boundary spots: boundary metrics not evaluable
        return {"boundary_precision": np.nan, "boundary_recall": np.nan, "boundary_f1": np.nan, "n_true_boundary": int(true_b.sum()), "n_pred_boundary": int(pred_b.sum())}
    tt, tp = cKDTree(xy[true_b]), cKDTree(xy[pred_b])
    prec = np.mean(tt.query(xy[pred_b])[0] <= tol)
    rec = np.mean(tp.query(xy[true_b])[0] <= tol)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"boundary_precision": float(prec), "boundary_recall": float(rec), "boundary_f1": float(f1), "n_true_boundary": int(true_b.sum()), "n_pred_boundary": int(pred_b.sum())}


def evaluate_heldout(emb: dict[str, np.ndarray], train_graphs: list[SectionGraph], test_graphs: list[SectionGraph], cfg: dict, seed: int) -> dict:
    """Probe on training patients' embeddings → metrics on the held-out patient (all its sections pooled)."""
    Ztr = np.vstack([emb[g.section] for g in train_graphs]); ytr = np.concatenate([g.labels for g in train_graphs])
    m = np.isin(ytr, CLASSES)
    sc, clf = fit_probe(Ztr[m], ytr[m], seed)
    ys, ps, brows = [], [], []
    for g in test_graphs:
        pred = clf.predict(sc.transform(emb[g.section]))
        ev = np.isin(g.labels, CLASSES)
        ys.append(g.labels[ev]); ps.append(pred[ev])
        pb = boundary_from_labels(pred, g.xy, g.pitch)
        tb = g.boundary if g.boundary is not None else boundary_from_labels(g.labels, g.xy, g.pitch)
        brows.append({"section": g.section, **boundary_prf(pb, tb, g.xy, g.pitch, cfg["histology"]["boundary_tolerance_rings"])})
    y, p = np.concatenate(ys), np.concatenate(ps)
    # classes scored = classes actually present in the held-out patient (≥ 20 spots); a single-class patient
    # yields accuracy only (macro-F1 / balanced accuracy undefined → NaN) and is flagged
    present = [c for c in CLASSES if (y == c).sum() >= 20]
    out = {"n_eval_spots": int(len(y)), "n_classes_present": len(present), "accuracy": float((y == p).mean()),
           "macro_f1": float(f1_score(y, p, average="macro", labels=present)) if len(present) >= 2 else np.nan,
           "balanced_accuracy": float(balanced_accuracy_score(y[np.isin(y, present)], p[np.isin(y, present)])) if len(present) >= 2 else np.nan}
    b = pd.DataFrame(brows)
    # patient-level boundary F1: pool boundary counts across the patient's sections (weighted by true boundary spots)
    w = b["n_true_boundary"].fillna(0).to_numpy(float)
    for k in ["boundary_precision", "boundary_recall", "boundary_f1"]:
        v = b[k].to_numpy(float); ok = ~np.isnan(v) & (w > 0)
        out[k] = float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan
    out["per_section_boundary"] = brows
    return out


def cluster_agreement(emb_a: np.ndarray, emb_b: np.ndarray, k: int, seed: int) -> float:
    from sklearn.cluster import KMeans

    la = KMeans(k, n_init=5, random_state=seed).fit_predict(emb_a)
    lb = KMeans(k, n_init=5, random_state=seed).fit_predict(emb_b)
    return float(adjusted_rand_score(la, lb))
