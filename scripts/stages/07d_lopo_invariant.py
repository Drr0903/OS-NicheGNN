"""Stage 07d —  (post hoc): patient-invariant OS-NicheGNN variants (spatial-neighbour + cross-patient anchor
positives, optional view term) under leave-one-patient-out with nested selection.

Per fold: anchors = mutual nearest neighbours (PCA-50 of standardised expression, k = 10) between spots of different *training*
patients (inner-validation patient excluded from training but its anchors are not needed). Grid λ_anchor × λ_con with λ_nbr fixed.
Selection on training patients only: J2 = inner-validation probe macro-F1 + w · (ARI_region − ARI_patient) of balanced-graph
Leiden communities (K fixed) of the training-patient embeddings. Outputs mirror stage 07c with the prefix `invariant`.
Usage: python 07d_lopo_invariant.py [--patients P01,P02] [--n_seeds N] [--tag T] [--merge tags]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.cluster import leiden_fixed_k  # noqa: E402
from lib.data import holdout_genes, load_labels, load_qc_sections  # noqa: E402
from lib.evalprobe import evaluate_heldout  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.train import embed_sections, train_model  # noqa: E402
from lib.upgrade import mnn_anchors  # noqa: E402

STAGE = "07_lopo"; PREFIX = "invariant"
s07 = import_module("07_lopo_cv")


def train_aris(emb: dict, G: dict, train_secs: list[str], patient_of: dict, K: int, seed: int, per_patient: int = 500) -> tuple[float, float]:
    rng = np.random.default_rng(seed); Z, P, L = [], [], []
    for pat in sorted({patient_of[s] for s in train_secs}):
        secs = [s for s in train_secs if patient_of[s] == pat]; zs = np.vstack([emb[s] for s in secs]); ls = np.concatenate([G[s].labels for s in secs])
        idx = rng.choice(len(zs), min(per_patient, len(zs)), replace=False); Z.append(zs[idx]); P += [pat] * len(idx); L.append(ls[idx])
    Z = np.vstack(Z); P = np.array(P); L = np.concatenate(L); lab, _ = leiden_fixed_k(Z, K, seed, batch=P, k_within=3)
    ev = np.isin(L, ["tumor_cellular", "lung_parenchyma", "paucicellular"])
    return float(adjusted_rand_score(P, lab)), float(adjusted_rand_score(L[ev], lab[ev]))


def run_fold(cfg, G, train_secs, val_secs, test_secs, seeds, device, emb_root, P, patient_of):
    up = cfg["upgrade"]; inv = cfg["invariant"]; rows, sel = [], []
    tr_g = [G[s] for s in train_secs if s not in val_secs]; va_g = [G[s] for s in val_secs]; te_g = [G[s] for s in test_secs]; all_tr = tr_g + va_g
    t0 = time.time(); anchors = mnn_anchors(tr_g, n_pcs=inv["anchor_pcs"], k=inv["anchor_k"], seed=cfg["seed"]); n_anc = sum(len(v[0]) for v in anchors.values()) // 2
    log(f"  anchors: {n_anc} mutual pairs over {len(anchors) // 2} section pairs ({time.time() - t0:.0f}s)")
    for la, lc in itertools.product(inv["lambda_anchor_grid"], inv["lambda_con_grid"]):
        Js = []
        for seed in seeds:
            t0 = time.time(); upg = {"lambda_anchor": la, "lambda_con": lc, "lambda_nbr": inv["lambda_nbr"], "anchors": anchors}
            model, info = train_model("nichegnn", tr_g, va_g, cfg, seed, device=device, upgrade=upg)
            emb = embed_sections(model, [G[s] for s in train_secs + test_secs], "nichegnn", cfg, device)
            inner = evaluate_heldout(emb, tr_g, va_g, cfg, seed); inner.pop("per_section_boundary", None)
            f1_in = inner["macro_f1"] if inner["macro_f1"] == inner["macro_f1"] else inner["accuracy"]
            ari_p, ari_r = train_aris(emb, G, train_secs, patient_of, up["cluster_k"], seed)
            J = f1_in + inv["selection_weight"] * (ari_r - ari_p); Js.append(J)
            held = evaluate_heldout(emb, all_tr, te_g, cfg, seed); per = held.pop("per_section_boundary")
            rows.append({"method": f"nichegnn_{PREFIX}", "lambda_anchor": la, "lambda_con": lc, "lambda_nbr": inv["lambda_nbr"], "ablation": "full", "seed": seed, "heldout_patient": P, "fold": f"LOPO_{P}",
                         "n_train_sections": len(all_tr), "n_test_sections": len(te_g), "run_time_s": round(time.time() - t0, 1), "failed": False, "error": "", **held,
                         "inner_val_macro_f1": f1_in, "train_patient_ari": ari_p, "train_region_ari": ari_r, "J": J, "epochs_run": info["epochs_run"], "n_params": info["n_params"],
                         "best_val_masked_mse": info["best_val_masked_mse"], "per_section_boundary": json.dumps(per)})
            d = emb_root / f"{PREFIX}_anc{la}_con{lc}_{seed}"; d.mkdir(parents=True, exist_ok=True)
            for s in test_secs:
                np.save(d / f"{s}.npy", emb[s])
            log(f"    anchor {la} con {lc} seed {seed}: inner F1 {f1_in:.3f} ARI_pat {ari_p:.3f} ARI_reg {ari_r:.3f} J {J:.3f} | held-out F1 {held['macro_f1']:.3f} bF1 {held['boundary_f1']:.3f} ({info['epochs_run']} ep, {time.time() - t0:.0f}s)")
        sel.append({"heldout_patient": P, "lambda_anchor": la, "lambda_con": lc, "lambda_nbr": inv["lambda_nbr"], "J_mean": float(np.mean(Js)), "J_sd": float(np.std(Js))})
    sel = pd.DataFrame(sel); best = sel.sort_values("J_mean", ascending=False).iloc[0]
    for r in rows:
        r["selected"] = bool(r["lambda_anchor"] == best["lambda_anchor"] and r["lambda_con"] == best["lambda_con"])
    sel["selected"] = (sel["lambda_anchor"] == best["lambda_anchor"]) & (sel["lambda_con"] == best["lambda_con"])
    log(f"  fold {P}: selected anchor {best['lambda_anchor']} con {best['lambda_con']} (J2 {best['J_mean']:.3f})")
    return rows, sel


def merge_parts(cfg: dict, tags: list[str]) -> None:
    sd = stage_dir(STAGE)
    df = pd.concat([pd.read_csv(sd / f"lopo_metrics_gene_holdout_{PREFIX}{t}.tsv", sep="\t") for t in tags], ignore_index=True)
    sel = pd.concat([pd.read_csv(sd / f"{PREFIX}_selection{t}.tsv", sep="\t") for t in tags], ignore_index=True)
    write_table(df, sd / f"lopo_metrics_gene_holdout_{PREFIX}.tsv"); write_table(sel, sd / f"{PREFIX}_selection.tsv")
    s07.summarise(df[df["selected"]].copy(), sd, f"gene_holdout_{PREFIX}")
    write_provenance(STAGE + f"_gene_holdout_{PREFIX}", [sd / f"lopo_metrics_gene_holdout_{PREFIX}{t}.tsv" for t in tags], [sd / f"lopo_metrics_gene_holdout_{PREFIX}.tsv"], cfg, extra={"amendment": "A7", "merged_parts": tags})


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--patients", default=""); ap.add_argument("--n_seeds", type=int, default=0); ap.add_argument("--tag", default=""); ap.add_argument("--merge", default="")
    args = ap.parse_args(); t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"])
    if args.merge:
        merge_parts(cfg, args.merge.split(",")); return
    disc = cfg["cohorts"]["discovery"]; patient_of = disc["patient_groups"]
    secs = [s for s in disc["lung_sections"] if (ROOT / cfg["paths"]["processed"] / "qc" / f"{s}.h5ad").exists()]
    adatas = load_qc_sections(cfg, secs); labels = {s: load_labels(cfg, s) for s in secs}; exclude = holdout_genes(cfg)
    all_pats = sorted({patient_of[s] for s in secs}); patients = [p for p in all_pats if not args.patients or p in args.patients.split(",")]
    seeds = [cfg["seed"] + i for i in range(args.n_seeds or cfg["n_seeds"])]
    emb_root = ROOT / cfg["paths"]["processed"] / "embeddings" / f"lopo_gene_holdout_{PREFIX}{args.tag}"; sd = stage_dir(STAGE); rows, sels = [], []
    for i, P in enumerate(patients):
        test_secs = [s for s in secs if patient_of[s] == P]; train_secs = [s for s in secs if patient_of[s] != P]
        tr_pats = sorted({patient_of[s] for s in train_secs}); rng = np.random.default_rng(cfg["seed"] + all_pats.index(P))
        val_pat = tr_pats[int(rng.integers(len(tr_pats)))]; val_secs = [s for s in train_secs if patient_of[s] == val_pat]
        log(f"[fold {i + 1}/{len(patients)}] held-out {P} {test_secs}; inner val {val_pat}")
        G, _ = s07.fold_graphs(cfg, adatas, labels, train_secs, test_secs, patient_of, exclude)
        r, sel = run_fold(cfg, G, train_secs, val_secs, test_secs, seeds, cfg["device"], emb_root, P, patient_of); rows += r; sels.append(sel)
        pd.DataFrame(rows).to_csv(sd / f"lopo_metrics_gene_holdout_{PREFIX}{args.tag}.partial.tsv", sep="\t", index=False)
    df = pd.DataFrame(rows); write_table(df, sd / f"lopo_metrics_gene_holdout_{PREFIX}{args.tag}.tsv"); write_table(pd.concat(sels, ignore_index=True), sd / f"{PREFIX}_selection{args.tag}.tsv")
    s07.summarise(df[df["selected"]].copy(), sd, f"gene_holdout_{PREFIX}{args.tag}")
    write_provenance(STAGE + f"_gene_holdout_{PREFIX}{args.tag}", [ROOT / cfg["paths"]["processed"] / "qc"], [sd / f"lopo_metrics_gene_holdout_{PREFIX}{args.tag}.tsv"], cfg, extra={"amendment": "A7", "patients": patients, "seeds": seeds, "grid": cfg["invariant"]}, t_start=t0)


if __name__ == "__main__":
    main()
