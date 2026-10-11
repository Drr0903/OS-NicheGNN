"""Stage 25 () — are the niche definitions reusable on a patient the model has never seen?

Per leave-one-patient-out fold (the stage-14/23 split and inner-validation patient), the atlas encoder (config `atlas`,
including `arch_overrides`) is trained on the training patients only, with the fold-fitted profile of stage 23. Its latent z
of a patient-balanced sample of the training patients is clustered exactly as the atlas is (K = v2.k_fixed, config
`niches.graph`), prototypes are the community centroids, and the held-out patient's spots are assigned to the nearest
prototype; spots beyond the 97.5 % quantile of the training-spot distances are unmatched. Fold niches are matched one-to-one
to the full-cohort atlas niches on the training spots (Hungarian assignment on the contingency table), and on the held-out
patient we report: agreement of matched labels with the full atlas, ARI, unmatched fraction and per-niche recall.
The same quantities on the training spots give the ceiling set by re-clustering alone.

Usage: python scripts/stages/25_niche_reuse.py --patients P01,P02 --tag _P01   ;   --merge _P01,_P02,...
Outputs results/25_niche_reuse/{reuse_patient.tsv, reuse_niche.tsv, reuse_summary.tsv}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.arch_eval import image_graphs, pca_graphs  # noqa: E402
from lib.arch_train import embed_arch, train_arch  # noqa: E402
from lib.cluster import assign, atlas_batch, cluster_prototypes, integrate  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.v2 import compute_anchors, load_cohort, make_graphs  # noqa: E402

STAGE = "25_niche_reuse"
NT = int(os.environ.get("OMP_NUM_THREADS", 8)); torch.set_num_threads(NT)  # threadpool limits above the BLAS start-up count crash LU with torch loaded


def atlas_cfg(cfg: dict) -> dict:
    """cfg with the atlas encoder's architecture overrides (width, niche targets) applied to `arch`."""
    c = json.loads(json.dumps(cfg)); c["arch"].update(cfg["atlas"].get("arch_overrides") or {}); return c


def match(fold_lab: np.ndarray, atlas_lab: np.ndarray, K: int) -> dict[int, int]:
    """One-to-one map fold niche → atlas niche maximising the number of shared training spots."""
    ct = pd.crosstab(fold_lab, atlas_lab).reindex(index=range(K), columns=range(K), fill_value=0).to_numpy()
    r, c = linear_sum_assignment(-ct); return dict(zip(r.tolist(), c.tolist()))


def scores(fold: np.ndarray, atlas: np.ndarray, unm_f: np.ndarray, unm_a: np.ndarray, m: dict, K: int) -> tuple[dict, list]:
    both = ~unm_f & ~unm_a; mapped = np.array([m[int(v)] for v in fold])
    agree = float((mapped[both] == atlas[both]).mean()) if both.any() else np.nan
    rec = [{"atlas_niche": k, "n_atlas_spots": int(((atlas == k) & both).sum()),
            "recall": float((mapped[(atlas == k) & both] == k).mean()) if ((atlas == k) & both).any() else np.nan} for k in range(K)]
    return {"agreement": agree, "ari": float(adjusted_rand_score(atlas[both], mapped[both])) if both.sum() > 1 else np.nan,
            "unmatched_fold": float(unm_f.mean()), "unmatched_atlas": float(unm_a.mean()), "n_spots": int(len(fold))}, rec


def run_fold(cfg, adatas, labels, patient_of, atlas, P):
    secs = list(adatas); pats = sorted(set(patient_of.values())); K = cfg["v2"]["k_fixed"]; seed = cfg["seed"]
    test = [s for s in secs if patient_of[s] == P]; train = [s for s in secs if patient_of[s] != P]
    rng = np.random.default_rng(seed + pats.index(P)); tr_p = sorted({patient_of[s] for s in train}); val_pat = tr_p[int(rng.integers(len(tr_p)))]  # = stage 14/23
    val = [s for s in train if patient_of[s] == val_pat]; fit = [s for s in train if s not in val]
    G, _, _, _ = make_graphs(cfg, adatas, labels, patient_of, train, train + test); Gp = pca_graphs(G, train, cfg["remedies"]["pca_dim"], seed)
    if cfg["atlas"]["multimodal"]:
        _, Gp = image_graphs(G, Gp, train, cfg["remedies"]["pca_dim"], seed, ROOT / cfg["paths"]["processed"] / "image_features")
    anchors = compute_anchors(cfg, [Gp[s] for s in fit])
    model, info = train_arch(cfg["atlas"]["context"], [Gp[s] for s in fit], [Gp[s] for s in val], cfg, seed, cfg["device"], anchors, cfg["atlas"].get("lambda_anchor", cfg["v2"]["model"]["lambda_anchor"]))
    E = embed_arch(model, [Gp[s] for s in train + test], cfg["device"])
    Z = integrate({s: E[s]["z"] for s in E}, patient_of, seed, cfg)  # the atlas space (label-free; includes the held-out spots)
    r2 = np.random.default_rng(seed); Zs, Ps, idx = [], [], {}
    for p in sorted({patient_of[s] for s in train}):
        zz = np.vstack([Z[s] for s in train if patient_of[s] == p]); keep = r2.choice(len(zz), min(1500, len(zz)), replace=False); Zs.append(zz[keep]); Ps += [p] * len(keep)
    with threadpool_limits(limits=NT):
        _, protos, _ = cluster_prototypes(np.vstack(Zs), K, seed, batch=atlas_batch(cfg, np.array(Ps)), k_within=cfg["niches"]["k_within"])
    lab, dist = {}, {}
    for s in train + test:
        lab[s], dist[s] = assign(Z[s], protos)
    thr = float(np.quantile(np.concatenate([dist[s] for s in train]), cfg["niches"]["unmatched_quantile"]))
    A = {s: atlas[atlas["section"] == s] for s in train + test}
    for s in A:
        assert (A[s]["barcode"].to_numpy() == np.asarray(G[s].barcodes)).all(), f"{s}: atlas spots differ from the fold graph"
    cat = lambda ss, f: np.concatenate([f(s) for s in ss])  # noqa: E731
    m = match(cat(train, lambda s: lab[s]), cat(train, lambda s: A[s]["niche"].to_numpy()), K)
    rows, nrows = [], []
    for scope, ss in [("held_out", test), ("training", train)]:
        sc, rec = scores(cat(ss, lambda s: lab[s]), cat(ss, lambda s: A[s]["niche"].to_numpy()), cat(ss, lambda s: dist[s] > thr),
                         cat(ss, lambda s: A[s]["unmatched"].to_numpy(bool)), m, K)
        rows.append({"heldout_patient": P, "scope": scope, **sc, "epochs_run": info["epochs_run"]})
        nrows += [{"heldout_patient": P, "scope": scope, **r} for r in rec]
    log(f"fold {P}: held-out agreement {rows[0]['agreement']:.3f} ARI {rows[0]['ari']:.3f} unmatched {rows[0]['unmatched_fold']:.3f}; training agreement {rows[1]['agreement']:.3f}")
    return rows, nrows


def merge(cfg, tags: list[str], t0: float) -> None:
    sd = stage_dir(STAGE)
    pt = pd.concat([pd.read_csv(sd / f"reuse_patient{t}.tsv", sep="\t") for t in tags], ignore_index=True)
    nt = pd.concat([pd.read_csv(sd / f"reuse_niche{t}.tsv", sep="\t") for t in tags], ignore_index=True)
    summ = pt.groupby("scope")[["agreement", "ari", "unmatched_fold", "unmatched_atlas"]].agg(["mean", "median", "min"]); summ.columns = ["_".join(c) for c in summ.columns]
    out = [sd / "reuse_patient.tsv", sd / "reuse_niche.tsv", sd / "reuse_summary.tsv"]
    write_table(pt, out[0]); write_table(nt.groupby(["scope", "atlas_niche"])[["recall"]].agg(["mean", "count"]).reset_index().pipe(lambda d: d.set_axis(["scope", "atlas_niche", "recall_mean", "n_patients"], axis=1)), out[1])
    write_table(summ.reset_index(), out[2]); log(summ.round(3).to_string())
    write_provenance(STAGE, [ROOT / "results/15_niches/spot_niches.tsv"], out, cfg, t_start=t0)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--patients", default=""); ap.add_argument("--tag", default=""); ap.add_argument("--merge", default="")
    a = ap.parse_args(); t0 = time.time(); cfg = atlas_cfg(load_config()); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE)
    if a.merge:
        merge(cfg, a.merge.split(","), t0); return
    adatas, labels, patient_of, _ = load_cohort(cfg)
    atlas = pd.read_csv(ROOT / "results/15_niches/spot_niches.tsv", sep="\t", usecols=["section", "barcode", "niche", "unmatched"])
    rows, nrows = [], []
    for P in [p for p in sorted(set(patient_of.values())) if not a.patients or p in a.patients.split(",")]:
        r, n = run_fold(cfg, adatas, labels, patient_of, atlas, P); rows += r; nrows += n
        write_table(pd.DataFrame(rows), sd / f"reuse_patient{a.tag}.tsv"); write_table(pd.DataFrame(nrows), sd / f"reuse_niche{a.tag}.tsv")


if __name__ == "__main__":
    with threadpool_limits(limits=NT):
        main()
