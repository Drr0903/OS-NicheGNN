"""Stage 25b — controls for the niche transfer of stage 25: a strictly inductive variant and non-learned baselines.

Per leave-one-patient-out fold (identical split, inner-validation patient, seed, epochs and encoder settings to stage 25):
  transductive  the stage-25 procedure: standardise → PCA-30 → Harmony by patient fitted on all spots incl. the held-out
                patient, prototypes from a patient-balanced sample of training spots, nearest-prototype assignment,
                unmatched beyond the 97.5 % quantile of training-spot distances.
  inductive     nothing is fitted on the held-out patient: scaler, PCA and Harmony are fitted on the training patients' spots
                only; held-out spots are standardised and projected with the training scaler/PCA and are not Harmony-corrected.
Representations: OS-NicheGNN (atlas encoder, retrained per fold) and three baselines without training — the spot profile
(expression PCA-50 + H&E PCA-50, fold-fitted as stage 23), the same profile with ring-1 and ring-2–3 means, and BANKSY-20.
References: OS-NicheGNN is matched to the full-cohort atlas (results/15_niches); each baseline is matched to its own
full-cohort partition built here on all 19 patients with the same atlas procedure (single seed, balanced sample of 1,500 spots
per patient, fixed-K Leiden, nearest-prototype assignment, 97.5 % unmatched rule; no multi-seed consensus).
Region check: ARI of the transferred niches with the image-derived region labels (three classes), unmatched spots excluded.

Usage: python scripts/stages/25b_transfer_controls.py --atlas                       (baseline full-cohort partitions)
       python scripts/stages/25b_transfer_controls.py --patients P01 --tag _P01 [--device cpu]
       python scripts/stages/25b_transfer_controls.py --merge _P01,_P02,...
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
from lib.arch_eval import image_graphs, paired, pca_graphs  # noqa: E402
from lib.arch_train import embed_arch, train_arch  # noqa: E402
from lib.banksy import BanksyEmbedder  # noqa: E402
from lib.cluster import assign, atlas_batch, cluster_prototypes, integrate  # noqa: E402
from lib.context_probe import propagate  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.v2 import CLASSES, compute_anchors, load_cohort, make_graphs  # noqa: E402

_spec = importlib.util.spec_from_file_location("s25", Path(__file__).with_name("25_niche_reuse.py")); s25 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s25)
STAGE = "25b_transfer_controls"
NT = int(os.environ.get("OMP_NUM_THREADS", 8)); torch.set_num_threads(NT)
GNN = "OS-NicheGNN"; BASELINES = ["spot profile", "profile + ring means", "BANKSY-20"]
ATLAS_TSV = "baseline_atlas_spots.tsv"


def space(emb: dict, patient_of: dict, seed: int, cfg: dict, fit: list[str]) -> dict:
    """Atlas space. fit = all sections → the stage-25 (transductive) integration; fit = training sections → scaler, PCA and
    Harmony fitted on those sections only, other sections projected with the training scaler/PCA and left uncorrected."""
    secs = list(emb); d = emb[secs[0]].shape[1]
    if set(fit) == set(secs) and d >= 30:
        return integrate(emb, patient_of, seed, cfg)  # exactly lib.cluster.integrate (stage 25)
    import harmonypy as hm
    Xf = np.vstack([emb[s] for s in fit]); scl = StandardScaler().fit(Xf); pca = PCA(min(30, d), random_state=seed).fit(scl.transform(Xf))
    out = {s: pca.transform(scl.transform(emb[s])).astype(np.float32) for s in secs}
    Zf = np.vstack([out[s] for s in fit])
    ho = hm.run_harmony(Zf, pd.DataFrame({"p": np.concatenate([[patient_of[s]] * len(out[s]) for s in fit])}), "p", random_state=seed, verbose=False, ncores=int(__import__("os").environ.get("OMP_NUM_THREADS", 8)))
    Zc = np.asarray(ho.Z_corr); Zc = Zc.T if Zc.shape[0] != Zf.shape[0] else Zc; o = 0
    for s in fit:
        out[s] = Zc[o:o + len(out[s])].astype(np.float32); o += len(out[s])
    return out


def transfer(Z: dict, proto_secs: list[str], patient_of: dict, cfg: dict) -> tuple[dict, dict]:
    """Prototypes from a patient-balanced sample (1,500 spots per patient) of proto_secs, as stage 25; labels and unmatched flags
    for every section (threshold = unmatched quantile of the proto_secs distances)."""
    seed, K = cfg["seed"], cfg["v2"]["k_fixed"]; r2 = np.random.default_rng(seed); Zs, Ps = [], []
    for p in sorted({patient_of[s] for s in proto_secs}):
        zz = np.vstack([Z[s] for s in proto_secs if patient_of[s] == p]); keep = r2.choice(len(zz), min(1500, len(zz)), replace=False); Zs.append(zz[keep]); Ps += [p] * len(keep)
    with threadpool_limits(limits=NT):
        _, protos, _ = cluster_prototypes(np.vstack(Zs), K, seed, batch=atlas_batch(cfg, np.array(Ps)), k_within=cfg["niches"]["k_within"])
    lab, dist = {}, {}
    for s in Z:
        lab[s], dist[s] = assign(Z[s], protos)
    thr = float(np.quantile(np.concatenate([dist[s] for s in proto_secs]), cfg["niches"]["unmatched_quantile"]))
    return lab, {s: dist[s] > thr for s in Z}


def baseline_reps(G: dict, Gp: dict, fit: list[str], cfg: dict) -> dict[str, dict]:
    b = cfg["baselines"]["banksy"]; be = BanksyEmbedder(b["lambda"], b["k_geom"], b["n_pcs"], cfg["seed"]).fit([G[s] for s in fit])
    return {"spot profile": {s: Gp[s].x for s in Gp}, "profile + ring means": {s: propagate(Gp[s]) for s in Gp},
            "BANKSY-20": {s: be.transform(G[s]) for s in G}}


def profile_graphs(cfg, adatas, labels, patient_of, fit, secs):
    G, _, _, _ = make_graphs(cfg, adatas, labels, patient_of, fit, secs); Gp = pca_graphs(G, fit, cfg["remedies"]["pca_dim"], cfg["seed"])
    if cfg["atlas"]["multimodal"]:
        _, Gp = image_graphs(G, Gp, fit, cfg["remedies"]["pca_dim"], cfg["seed"], ROOT / cfg["paths"]["processed"] / "image_features")
    return G, Gp


def build_atlas(cfg, adatas, labels, patient_of) -> None:
    secs = list(adatas); G, Gp = profile_graphs(cfg, adatas, labels, patient_of, secs, secs); rows = []
    for name, emb in baseline_reps(G, Gp, secs, cfg).items():
        Z = space(emb, patient_of, cfg["seed"], cfg, secs); lab, unm = transfer(Z, secs, patient_of, cfg)
        for s in secs:
            rows.append(pd.DataFrame({"representation": name, "section": s, "barcode": np.asarray(G[s].barcodes), "niche": lab[s], "unmatched": unm[s]}))
        log(f"atlas {name}: niche sizes {np.bincount(np.concatenate([lab[s] for s in secs])).tolist()}")
    write_table(pd.concat(rows, ignore_index=True), stage_dir(STAGE) / ATLAS_TSV)


def evaluate(lab, unm, ref, G, train, test, K) -> list[dict]:
    cat = lambda ss, f: np.concatenate([f(s) for s in ss])  # noqa: E731
    for s in train + test:
        assert (ref[s]["barcode"].to_numpy() == np.asarray(G[s].barcodes)).all(), f"{s}: reference spots differ from the fold graph"
    m = s25.match(cat(train, lambda s: lab[s]), cat(train, lambda s: ref[s]["niche"].to_numpy()), K); out = []
    for scope, ss in [("held_out", test), ("training", train)]:
        f, u = cat(ss, lambda s: lab[s]), cat(ss, lambda s: unm[s]); sc, _ = s25.scores(f, cat(ss, lambda s: ref[s]["niche"].to_numpy()), u, cat(ss, lambda s: ref[s]["unmatched"].to_numpy(bool)), m, K)
        reg = cat(ss, lambda s: np.asarray(G[s].labels).astype(str)); ev = np.isin(reg, CLASSES) & ~u
        out.append({"scope": scope, **sc, "ari_region": float(adjusted_rand_score(reg[ev], f[ev])) if ev.sum() > 1 and len(set(reg[ev])) > 1 else np.nan, "n_region_spots": int(ev.sum())})
    return out


def run_fold(cfg, adatas, labels, patient_of, refs, P) -> list[dict]:
    secs = list(adatas); pats = sorted(set(patient_of.values())); K = cfg["v2"]["k_fixed"]; seed = cfg["seed"]
    test = [s for s in secs if patient_of[s] == P]; train = [s for s in secs if patient_of[s] != P]
    rng = np.random.default_rng(seed + pats.index(P)); tr_p = sorted({patient_of[s] for s in train}); val_pat = tr_p[int(rng.integers(len(tr_p)))]  # = stage 14/23/25
    val = [s for s in train if patient_of[s] == val_pat]; fit = [s for s in train if s not in val]
    G, Gp = profile_graphs(cfg, adatas, labels, patient_of, train, train + test)
    anchors = compute_anchors(cfg, [Gp[s] for s in fit]); t0 = time.time()
    model, info = train_arch(cfg["atlas"]["context"], [Gp[s] for s in fit], [Gp[s] for s in val], cfg, seed, cfg["device"], anchors, cfg["atlas"].get("lambda_anchor", cfg["v2"]["model"]["lambda_anchor"]))
    E = embed_arch(model, [Gp[s] for s in train + test], cfg["device"]); log(f"fold {P}: encoder trained in {time.time() - t0:.0f}s ({info['epochs_run']} epochs)")
    reps = {GNN: {s: E[s]["z"] for s in E}, **baseline_reps(G, Gp, train, cfg)}; rows = []
    for name, emb in reps.items():
        for mode, fitset in [("transductive", train + test), ("inductive", train)]:
            Z = space(emb, patient_of, seed, cfg, fitset); lab, unm = transfer(Z, train, patient_of, cfg)
            for r in evaluate(lab, unm, refs[name], G, train, test, K):
                rows.append({"representation": name, "mode": mode, "heldout_patient": P, **r, "epochs_run": info["epochs_run"]})
            h = rows[-2]; log(f"fold {P} {name} {mode}: held-out agreement {h['agreement']:.3f} ARI {h['ari']:.3f} unmatched {h['unmatched_fold']:.3f} region ARI {h['ari_region']:.3f}; training {rows[-1]['agreement']:.3f}")
    return rows


def merge(cfg, tags: list[str], t0: float) -> None:
    sd = stage_dir(STAGE); long = pd.concat([pd.read_csv(sd / f"transfer_long{t}.tsv", sep="\t") for t in tags], ignore_index=True)
    key = ["representation", "mode", "heldout_patient"]; val = ["agreement", "ari", "unmatched_fold", "unmatched_atlas", "ari_region", "n_spots", "n_region_spots"]
    w = long.pivot_table(index=key, columns="scope", values=val, aggfunc="first"); w.columns = [f"{v}_{'heldout' if s == 'held_out' else 'training'}" for v, s in w.columns]
    w = w.reset_index().merge(long.groupby(key)["epochs_run"].first().reset_index(), on=key)
    w["agreement_ratio"] = w["agreement_heldout"] / w["agreement_training"]
    w = w.rename(columns={"unmatched_fold_heldout": "unmatched_heldout", "unmatched_fold_training": "unmatched_training"})
    metrics = ["agreement_heldout", "agreement_training", "agreement_ratio", "ari_heldout", "ari_training", "unmatched_heldout", "unmatched_training", "ari_region_heldout", "ari_region_training"]
    srows = []
    for (rep, mode), d in w.groupby(["representation", "mode"]):
        for mt in metrics:
            srows.append({"type": "mean", "representation": rep, "mode": mode, "comparator": "", "metric": mt, "n_patients": int(d[mt].notna().sum()),
                          "mean": float(d[mt].mean()), "median": float(d[mt].median()), "min": float(d[mt].min())})
    idx = lambda rep, mode: w[(w.representation == rep) & (w["mode"] == mode)].set_index("heldout_patient")  # noqa: E731
    comps = [(GNN, m, b, m) for m in ["transductive", "inductive"] for b in BASELINES] + [(GNN, "inductive", GNN, "transductive")]
    for rep, mode, cmp_rep, cmp_mode in comps:
        a, b = idx(rep, mode), idx(cmp_rep, cmp_mode)
        for mt in ["agreement_heldout", "agreement_ratio", "ari_heldout", "unmatched_heldout", "ari_region_heldout"]:
            srows.append({"type": "paired_difference", "representation": rep, "mode": mode, "comparator": f"{cmp_rep} ({cmp_mode})", "metric": mt, **paired(a[mt], b[mt], cfg["seed"])})
    out = [sd / "transfer_patient.tsv", sd / "transfer_summary.tsv"]
    write_table(w[key + ["epochs_run", "n_spots_heldout", "n_spots_training"] + metrics + ["unmatched_atlas_heldout", "n_region_spots_heldout"]], out[0]); summ = pd.DataFrame(srows); write_table(summ, out[1])
    log(summ[summ.type == "mean"].pivot_table(index=["representation", "mode"], columns="metric", values="mean").round(3).to_string())
    write_provenance(STAGE, [ROOT / "results/15_niches/spot_niches.tsv", sd / ATLAS_TSV], out + [sd / ATLAS_TSV], cfg, t_start=t0,
                     extra={"folds": tags, "baseline_atlas": "single seed, balanced 1,500 spots per patient, fixed-K Leiden, no consensus",
                            "inductive": "scaler/PCA/Harmony fitted on training patients only; held-out projected, not Harmony-corrected"})


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--patients", default=""); ap.add_argument("--tag", default=""); ap.add_argument("--merge", default="")
    ap.add_argument("--atlas", action="store_true"); ap.add_argument("--device", default="")
    a = ap.parse_args(); t0 = time.time(); cfg = s25.atlas_cfg(load_config()); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE)
    if a.device:
        cfg["device"] = a.device
    if a.merge:
        merge(cfg, a.merge.split(","), t0); return
    adatas, labels, patient_of, _ = load_cohort(cfg)
    if a.atlas:
        build_atlas(cfg, adatas, labels, patient_of); return
    gref = pd.read_csv(ROOT / "results/15_niches/spot_niches.tsv", sep="\t", usecols=["section", "barcode", "niche", "unmatched"])
    bref = pd.read_csv(sd / ATLAS_TSV, sep="\t")
    refs = {GNN: {s: d.reset_index(drop=True) for s, d in gref.groupby("section")}}
    for name, d in bref.groupby("representation"):
        refs[name] = {s: dd.reset_index(drop=True) for s, dd in d.groupby("section")}
    rows = []
    for P in [p for p in sorted(set(patient_of.values())) if not a.patients or p in a.patients.split(",")]:
        rows += run_fold(cfg, adatas, labels, patient_of, refs, P); write_table(pd.DataFrame(rows), sd / f"transfer_long{a.tag}.tsv")


if __name__ == "__main__":
    with threadpool_limits(limits=NT):
        main()
