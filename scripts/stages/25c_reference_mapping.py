"""Stage 25c — niche transfer to a new patient through a fixed reference, and agreement as a function of coverage.

Per leave-one-patient-out fold (split, inner-validation patient, seed and encoder settings of stages 25/25b) the encoder is
trained on the training patients and four ways of placing the held-out patient in the atlas space are compared:
  transductive  scaler, PCA-30 and Harmony fitted on all spots including the held-out patient (stage 25)
  inductive     scaler, PCA and Harmony fitted on the training patients; held-out spots projected, left uncorrected (stage 25b)
  mapped        as inductive, then the held-out patient is mapped onto the fixed Harmony reference in the manner of Symphony
                (Kang et al. 2021): soft assignment to the reference clusters, and per cluster a ridge estimate of the held-out
                patient's offset from the reference terms, which stay unchanged; nothing of the reference is refitted
  raw           scaler and PCA fitted on the training patients, no Harmony anywhere
Prototypes and the unmatched threshold always come from the training spots only (as stage 25b). Representations: OS-NicheGNN
and the three baselines of stage 25b. References: every representation is compared with a full-cohort partition built with
one protocol (single seed, 1,500 spots per patient, fixed-K Leiden, nearest prototype; stage 25b `--atlas`, and here the same
for OS-NicheGNN from the trained atlas encoder); OS-NicheGNN is also compared with the consensus atlas of stage 15.
Coverage: held-out spots are ordered by their distance to the nearest prototype; agreement is computed among the closest
fraction c of spots (c = 0.05 … 1), giving a risk–coverage curve per patient, its area and the agreement at 90 % coverage.

Usage: python scripts/stages/25c_reference_mapping.py --atlas
       python scripts/stages/25c_reference_mapping.py --patients P01 --tag _P01
       python scripts/stages/25c_reference_mapping.py --merge _P01,_P02,...
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.arch_eval import paired  # noqa: E402
from lib.arch_train import embed_arch, train_arch  # noqa: E402
from lib.cluster import assign, atlas_batch, cluster_prototypes  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.v2 import CLASSES, compute_anchors, load_cohort  # noqa: E402


def _load(name, file):
    sp = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file)); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


s25 = _load("s25", "25_niche_reuse.py"); s25b = _load("s25b", "25b_transfer_controls.py")
STAGE = "25c_reference_mapping"
NT = int(os.environ.get("OMP_NUM_THREADS", 8)); torch.set_num_threads(NT)
GNN = s25b.GNN; MLP = "ring-mean MLP"; REPS = [GNN, MLP] + s25b.BASELINES; MODES = ["transductive", "inductive", "mapped", "raw"]
COV = np.round(np.arange(0.05, 1.0001, 0.05), 2); GNN_ATLAS = "gnn_atlas_spots.tsv"; MLP_ATLAS = "mlp_atlas_spots.tsv"


VARIANT = {"noattn": ("without the attention branch", "ringmlp", 384, {}), "noring": ("without the ring-mean branch", "xattn", 240, {}),
           "single": ("single fusion expert", "ringxattn", 256, {"moe_experts": 0, "moe_sharp": 0.0}),
           "concat": ("branches concatenated", "ringxattn", 264, {"moe_experts": 0, "moe_sharp": 0.0, "concat_fuse": True}),
           "mlp": ("ring-mean MLP", "ringmlp", 416, {}), "fus": ("MLP + attention", "ringxattn", 256, {}),
           "fusg": ("MLP + gated attention", "ringxattn", 240, {"ring_gate": True}),
           "cat": ("branches concatenated", "ringxattn", 264, {"concat_fuse": True}),
           "moe2": ("mixture of two MLP experts", "ringxattn", 240, {"moe_experts": 2}),
           "moe4": ("mixture of four MLP experts", "ringxattn", 224, {"moe_experts": 4}),
           "moe2s": ("mixture of two MLP experts, committed routing", "ringxattn", 240, {"moe_experts": 2, "moe_sharp": 0.1, "moe_balance": 0.1}),
           "moe2e05": ("mixture of two MLP experts, routing entropy 0.05", "ringxattn", 240, {"moe_experts": 2, "moe_sharp": 0.05, "moe_balance": 0.1}),
           "moe2e02": ("mixture of two MLP experts, routing entropy 0.02", "ringxattn", 240, {"moe_experts": 2, "moe_sharp": 0.02, "moe_balance": 0.1}),
           "moe4s": ("mixture of four MLP experts, committed routing", "ringxattn", 224, {"moe_experts": 4, "moe_sharp": 0.1, "moe_balance": 0.1})}
VAR = "mlp"  # set by --variant


def mlp_cfg(cfg: dict) -> dict:
    """A parameter-matched alternative encoder (VARIANT[VAR]): the ring-mean MLP of stage 32, or that MLP as backbone with
    cross-attention added, plainly or as a gated residual."""
    import json
    _, ctx, d, extra = VARIANT[VAR]; c = json.loads(json.dumps(cfg)); c["atlas"]["context"] = ctx; c["arch"]["d"] = d; c["arch"].update(extra); return c


def build_mlp_atlas(cfg, adatas, labels, patient_of) -> None:
    """The control network trained on all 19 patients as the atlas encoder is (fixed number of epochs, main seed), and its
    full-cohort reference partition by the same single-seed protocol."""
    import json
    secs = list(adatas); G, Gp = s25b.profile_graphs(cfg, adatas, labels, patient_of, secs, secs); c = mlp_cfg(cfg)
    c["arch"]["max_epochs"] = cfg["atlas"]["frozen_epochs"]; c["arch"]["patience"] = 10**9
    model, info = train_arch(c["atlas"]["context"], [Gp[s] for s in secs], [], c, cfg["seed"], cfg["device"], None, 0.0)
    emb = {s: v["z"] for s, v in embed_arch(model, [Gp[s] for s in secs], cfg["device"]).items()}
    Z = s25b.space(emb, patient_of, cfg["seed"], cfg, secs); lab, unm = s25b.transfer(Z, secs, patient_of, cfg)
    rows = [pd.DataFrame({"representation": VARIANT[VAR][0], "section": s, "barcode": np.asarray(G[s].barcodes), "niche": lab[s], "unmatched": unm[s]}) for s in secs]
    write_table(pd.concat(rows, ignore_index=True), stage_dir(STAGE) / f"{VAR}_atlas_spots.tsv")
    log(f"atlas {VARIANT[VAR][0]}: {info['n_params']} parameters, {info['epochs_run']} epochs; niche sizes {np.bincount(np.concatenate([lab[s] for s in secs])).tolist()}")


def spot_dir(cfg) -> Path:
    d = ROOT / cfg["paths"]["processed"] / "transfer_folds"; d.mkdir(parents=True, exist_ok=True); return d


def harmony(Z: np.ndarray, batch: np.ndarray, seed: int):
    import harmonypy as hm
    ho = hm.run_harmony(Z, pd.DataFrame({"p": batch}), "p", random_state=seed, verbose=False, ncores=int(__import__("os").environ.get("OMP_NUM_THREADS", 8)))
    Zc = np.asarray(ho.Z_corr); Zc = Zc.T if Zc.shape[0] != Z.shape[0] else Zc
    R = np.asarray(ho.R); R = R if R.shape[0] == Z.shape[0] else R.T
    Y = np.asarray(ho.Y); Y = Y if Y.shape[0] == Z.shape[1] else Y.T
    return Zc, R, Y


def symphony_map(Zq: np.ndarray, Zref: np.ndarray, R: np.ndarray, Y: np.ndarray, sigma: float = 0.1, lam: float = 1.0) -> np.ndarray:
    """Map one query batch onto a fixed Harmony reference (Symphony): soft clusters of the query from the reference centroids,
    then per cluster k a ridge regression with the reference terms N_k = Σ R_ik and C_k = Σ R_ik z_i (corrected reference),
    whose query-batch coefficient is removed from the query spots in proportion to their membership."""
    Yn = Y / np.linalg.norm(Y, axis=0, keepdims=True); Zn = Zq / np.linalg.norm(Zq, axis=1, keepdims=True)
    D = 2 * (1 - Zn @ Yn); Rq = np.exp(-(D - D.min(1, keepdims=True)) / sigma); Rq /= Rq.sum(1, keepdims=True)
    Nr = R.sum(0); C = R.T @ Zref; out = Zq.copy()
    for k in range(R.shape[1]):
        r = Rq[:, k]; nq = r.sum(); sq = r @ Zq
        A = np.array([[Nr[k] + nq, nq], [nq, nq + lam]]); B = np.vstack([C[k] + sq, sq])
        out -= r[:, None] * np.linalg.solve(A, B)[1][None]
    return out.astype(np.float32)


def place(emb: dict, patient_of: dict, train: list[str], test: list[str], mode: str, cfg: dict) -> dict:
    seed = cfg["seed"]; secs = train + test
    if mode == "transductive":
        return s25b.space(emb, patient_of, seed, cfg, secs)
    if mode == "inductive":
        return s25b.space(emb, patient_of, seed, cfg, train)
    d = emb[secs[0]].shape[1]; Xf = np.vstack([emb[s] for s in train]); scl = StandardScaler().fit(Xf)
    pca = PCA(min(30, d), random_state=seed).fit(scl.transform(Xf)); P = {s: pca.transform(scl.transform(emb[s])).astype(np.float32) for s in secs}
    if mode == "raw":
        return P
    Zf = np.vstack([P[s] for s in train]); Zc, R, Y = harmony(Zf, np.concatenate([[patient_of[s]] * len(P[s]) for s in train]), seed); out, o = {}, 0
    for s in train:
        out[s] = Zc[o:o + len(P[s])].astype(np.float32); o += len(P[s])
    Zq = symphony_map(np.vstack([P[s] for s in test]), Zc, R, Y); o = 0
    for s in test:
        out[s] = Zq[o:o + len(P[s])]; o += len(P[s])
    return out


def prototypes(Z: dict, proto_secs: list[str], patient_of: dict, cfg: dict) -> tuple[dict, dict, float]:
    seed, K = cfg["seed"], cfg["v2"]["k_fixed"]; r2 = np.random.default_rng(seed); Zs, Ps = [], []
    for p in sorted({patient_of[s] for s in proto_secs}):
        zz = np.vstack([Z[s] for s in proto_secs if patient_of[s] == p]); keep = r2.choice(len(zz), min(1500, len(zz)), replace=False); Zs.append(zz[keep]); Ps += [p] * len(keep)
    with threadpool_limits(limits=NT):
        _, protos, _ = cluster_prototypes(np.vstack(Zs), K, seed, batch=atlas_batch(cfg, np.array(Ps)), k_within=cfg["niches"]["k_within"])
    lab, dist = {}, {}
    for s in Z:
        lab[s], dist[s] = assign(Z[s], protos)
    return lab, dist, float(np.quantile(np.concatenate([dist[s] for s in proto_secs]), cfg["niches"]["unmatched_quantile"]))


def build_gnn_atlas(cfg, adatas, labels, patient_of) -> None:
    """The stage-25b baseline protocol applied to the trained atlas encoder (seed of the main run)."""
    secs = list(adatas); G, _ = s25b.profile_graphs(cfg, adatas, labels, patient_of, secs, secs)
    ed = ROOT / cfg["paths"]["processed"] / "embeddings" / cfg["atlas"]["emb_dir"] / str(cfg["seed"])
    emb = {s: np.load(ed / f"{s}.npy") for s in secs}
    for s in secs:
        assert len(emb[s]) == G[s].n, f"{s}: latent rows {len(emb[s])} != spots {G[s].n}"
    Z = s25b.space(emb, patient_of, cfg["seed"], cfg, secs); lab, unm = s25b.transfer(Z, secs, patient_of, cfg)
    rows = [pd.DataFrame({"representation": GNN, "section": s, "barcode": np.asarray(G[s].barcodes), "niche": lab[s], "unmatched": unm[s]}) for s in secs]
    write_table(pd.concat(rows, ignore_index=True), stage_dir(STAGE) / GNN_ATLAS)
    log(f"atlas {GNN} (single seed): niche sizes {np.bincount(np.concatenate([lab[s] for s in secs])).tolist()}")


def run_fold(cfg, adatas, labels, patient_of, refs, P, mlp: bool = False) -> tuple[list[dict], pd.DataFrame]:
    secs = list(adatas); pats = sorted(set(patient_of.values())); K = cfg["v2"]["k_fixed"]; seed = cfg["seed"]
    test = [s for s in secs if patient_of[s] == P]; train = [s for s in secs if patient_of[s] != P]
    rng = np.random.default_rng(seed + pats.index(P)); tr_p = sorted({patient_of[s] for s in train}); val_pat = tr_p[int(rng.integers(len(tr_p)))]
    val = [s for s in train if patient_of[s] == val_pat]; fit = [s for s in train if s not in val]
    G, Gp = s25b.profile_graphs(cfg, adatas, labels, patient_of, train, train + test)
    fz = spot_dir(cfg) / f"{VAR if mlp else 'gnn'}_{P}.npz"; c = mlp_cfg(cfg) if mlp else cfg
    if fz.exists():
        E = dict(np.load(fz)); log(f"fold {P}: encoder latent loaded from {fz.name}")
    else:
        anchors = compute_anchors(cfg, [Gp[s] for s in fit]); t0 = time.time()
        model, info = train_arch(c["atlas"]["context"], [Gp[s] for s in fit], [Gp[s] for s in val], c, seed, cfg["device"], anchors, cfg["atlas"].get("lambda_anchor", cfg["v2"]["model"]["lambda_anchor"]))
        E = {s: v["z"] for s, v in embed_arch(model, [Gp[s] for s in train + test], cfg["device"]).items()}; np.savez(fz, **E)
        log(f"fold {P}: encoder trained in {time.time() - t0:.0f}s ({info['epochs_run']} epochs)")
    reps = {VARIANT[VAR][0]: E} if mlp else {GNN: E, **s25b.baseline_reps(G, Gp, train, cfg)}; rows, spots = [], []
    cat = lambda ss, f: np.concatenate([f(s) for s in ss])  # noqa: E731
    reg = cat(test, lambda s: np.asarray(G[s].labels).astype(str))
    for name, emb in reps.items():
        for mode in MODES:
            Z = place(emb, patient_of, train, test, mode, cfg); lab, dist, thr = prototypes(Z, train, patient_of, cfg)
            for rname, ref in refs[name].items():
                m = s25.match(cat(train, lambda s: lab[s]), cat(train, lambda s: ref[s]["niche"].to_numpy()), K)
                f = np.array([m[int(v)] for v in cat(test, lambda s: lab[s])]); dd = cat(test, lambda s: dist[s])
                a = cat(test, lambda s: ref[s]["niche"].to_numpy()); au = cat(test, lambda s: ref[s]["unmatched"].to_numpy(bool))
                spots.append(pd.DataFrame({"representation": name, "mode": mode, "reference": rname, "heldout_patient": P, "dist": dd, "thr": thr,
                                           "niche": f, "ref_niche": a, "ref_unmatched": au, "region": reg}))
                ok = (dd <= thr) & ~au; ev = np.isin(reg, CLASSES) & (dd <= thr)
                rows.append({"representation": name, "mode": mode, "reference": rname, "heldout_patient": P, "n_spots": len(f),
                             "unmatched": float((dd > thr).mean()), "agreement": float((f[ok] == a[ok]).mean()) if ok.any() else np.nan,
                             "ari": float(adjusted_rand_score(a[ok], f[ok])) if ok.sum() > 1 else np.nan,
                             "ari_region": float(adjusted_rand_score(reg[ev], f[ev])) if ev.sum() > 1 and len(set(reg[ev])) > 1 else np.nan})
                r = rows[-1]; log(f"fold {P} {name} {mode} vs {rname}: agreement {r['agreement']:.3f} unmatched {r['unmatched']:.3f} region ARI {r['ari_region']:.3f}")
    return rows, pd.concat(spots, ignore_index=True)


def coverage_curves(sp: pd.DataFrame) -> pd.DataFrame:
    """Per patient × representation × mode × reference: agreement among the closest fraction c of held-out spots."""
    out = []
    for key, d in sp.groupby(["representation", "mode", "reference", "heldout_patient"]):
        d = d.sort_values("dist", kind="stable"); n = len(d); hit = (d["niche"].to_numpy() == d["ref_niche"].to_numpy()); valid = ~d["ref_unmatched"].to_numpy(bool)
        reg = d["region"].to_numpy(); f = d["niche"].to_numpy()
        for c in COV:
            k = max(1, int(round(c * n))); v = valid[:k]; ev = np.isin(reg[:k], CLASSES)
            out.append(dict(zip(["representation", "mode", "reference", "heldout_patient"], key), coverage=c,
                            agreement=float(hit[:k][v].mean()) if v.any() else np.nan,
                            ari_region=float(adjusted_rand_score(reg[:k][ev], f[:k][ev])) if ev.sum() > 1 and len(set(reg[:k][ev])) > 1 else np.nan))
    return pd.DataFrame(out)


def merge(cfg, tags: list[str], t0: float) -> None:
    sd = stage_dir(STAGE)
    tags = tags + [f"_{v}{t}" for v in VARIANT for t in tags if (sd / f"mapping_long_{v}{t}.tsv").exists()]; long = pd.concat([pd.read_csv(sd / f"mapping_long{t}.tsv", sep="\t") for t in tags], ignore_index=True)
    sp = pd.concat([pd.read_parquet(spot_dir(cfg) / f"spots{t}.parquet") for t in tags], ignore_index=True); cv = coverage_curves(sp)
    key = ["representation", "mode", "reference", "heldout_patient"]
    aurc = cv.groupby(key)["agreement"].mean().rename("agreement_area"); a90 = cv[cv.coverage == 0.9].set_index(key)[["agreement", "ari_region"]].add_suffix("_cov90")
    pt = long.set_index(key).join(aurc).join(a90).reset_index(); metrics = ["agreement", "unmatched", "ari", "ari_region", "agreement_area", "agreement_cov90", "ari_region_cov90"]
    srows = []
    for (rep, mode, ref), d in pt.groupby(["representation", "mode", "reference"]):
        for mt in metrics:
            srows.append({"type": "mean", "representation": rep, "mode": mode, "reference": ref, "comparator": "", "metric": mt, "n_patients": int(d[mt].notna().sum()),
                          "mean": float(d[mt].mean()), "median": float(d[mt].median()), "min": float(d[mt].min())})
    own = pt[pt.reference == "same-protocol atlas"]; idx = lambda r, m: own[(own.representation == r) & (own["mode"] == m)].set_index("heldout_patient")  # noqa: E731
    comps = [(GNN, m, b, m) for m in MODES for b in s25b.BASELINES] + [(GNN, m, GNN, "transductive") for m in ["inductive", "mapped", "raw"]] \
        + [(b, "mapped", b, "inductive") for b in s25b.BASELINES] + [(GNN, m, v[0], m) for v in VARIANT.values() for m in MODES if v[0] in set(own.representation)] \
        + [(v[0], m, MLP, m) for v in VARIANT.values() if v[0] != MLP for m in MODES if v[0] in set(own.representation)]
    comps = list(dict.fromkeys(comps))
    for rep, mode, cr, cm in comps:
        a, b = idx(rep, mode), idx(cr, cm)
        for mt in metrics:
            srows.append({"type": "paired_difference", "representation": rep, "mode": mode, "reference": "same-protocol atlas", "comparator": f"{cr} ({cm})", "metric": mt, **paired(a[mt], b[mt], cfg["seed"])})
    curve = cv.groupby(["representation", "mode", "reference", "coverage"])[["agreement", "ari_region"]].agg(["mean", "median"]); curve.columns = ["_".join(c) for c in curve.columns]
    out = [sd / "mapping_patient.tsv", sd / "mapping_summary.tsv", sd / "coverage_curve.tsv", sd / "coverage_patient.tsv"]
    write_table(pt, out[0]); write_table(pd.DataFrame(srows), out[1]); write_table(curve.reset_index(), out[2]); write_table(cv, out[3])
    log(pd.DataFrame(srows).query("type == 'mean' and reference != 'consensus atlas'").pivot_table(index=["representation", "mode"], columns="metric", values="mean").round(3).to_string())
    write_provenance(STAGE, [ROOT / "results/15_niches/spot_niches.tsv", ROOT / "results/25b_transfer_controls" / s25b.ATLAS_TSV, sd / GNN_ATLAS], out, cfg, t_start=t0,
                     extra={"folds": tags, "mapped": "Symphony-style mapping onto the training-patient Harmony reference (sigma 0.1, ridge 1)", "coverage_grid": COV.tolist()})


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--patients", default=""); ap.add_argument("--tag", default=""); ap.add_argument("--merge", default="")
    ap.add_argument("--atlas", action="store_true"); ap.add_argument("--device", default="")
    ap.add_argument("--mlp", action="store_true", help="a parameter-matched alternative encoder instead (tags get _<variant>)")
    ap.add_argument("--variant", default="mlp", choices=list(VARIANT))
    a = ap.parse_args(); global VAR; VAR = a.variant; t0 = time.time(); cfg = s25.atlas_cfg(load_config()); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE)
    if a.device:
        cfg["device"] = a.device
    if a.merge:
        merge(cfg, a.merge.split(","), t0); return
    adatas, labels, patient_of, _ = load_cohort(cfg)
    if a.atlas:
        (build_mlp_atlas if a.mlp else build_gnn_atlas)(cfg, adatas, labels, patient_of); return
    by = lambda d: {s: dd.reset_index(drop=True) for s, dd in d.groupby("section")}  # noqa: E731
    if a.mlp:  # an alternative encoder against its own reference only
        refs = {VARIANT[VAR][0]: {"same-protocol atlas": by(pd.read_csv(sd / f"{VAR}_atlas_spots.tsv", sep="\t"))}}; a.tag = f"_{VAR}" + a.tag
    else:
        gcons = pd.read_csv(ROOT / "results/15_niches/spot_niches.tsv", sep="\t", usecols=["section", "barcode", "niche", "unmatched"])
        bref = pd.read_csv(ROOT / "results/25b_transfer_controls" / s25b.ATLAS_TSV, sep="\t"); gref = pd.read_csv(sd / GNN_ATLAS, sep="\t")
        refs = {GNN: {"same-protocol atlas": by(gref), "consensus atlas": by(gcons)}, **{n: {"same-protocol atlas": by(d)} for n, d in bref.groupby("representation")}}
    rows, spots = [], []
    for P in [p for p in sorted(set(patient_of.values())) if not a.patients or p in a.patients.split(",")]:
        r, s = run_fold(cfg, adatas, labels, patient_of, refs, P, a.mlp); rows += r; spots.append(s)
        write_table(pd.DataFrame(rows), sd / f"mapping_long{a.tag}.tsv"); pd.concat(spots, ignore_index=True).to_parquet(spot_dir(cfg) / f"spots{a.tag}.parquet")
    write_provenance(STAGE + f"_part{a.tag}", [], [sd / f"mapping_long{a.tag}.tsv"], cfg, t_start=t0)


if __name__ == "__main__":
    with threadpool_limits(limits=NT):
        main()
