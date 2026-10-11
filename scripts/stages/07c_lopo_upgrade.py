"""Stage 07c —  (post hoc): leave-one-patient-out evaluation of the upgraded OS-NicheGNN variants
(patient-adversarial `adv`, view-contrastive `con`, both) with nested configuration selection.

Per fold (held-out patient P; inner-validation patient V as in stage 07):
  for every (λ_adv, λ_con) in the grid and every seed → train on the training patients minus V, early-stop on V's masked loss,
  embed everything; J = probe macro-F1 on V (probe fitted on the other training patients; accuracy if V has one class)
  − w · ARI(Leiden clusters, fixed K, on training-patient embeddings, patient identity). The configuration with the best mean J over seeds is
  the fold's model; its held-out metrics on P are the fold's result (P is never used for selection).
Outputs (results/07_lopo/): lopo_metrics_gene_holdout_upgrade.tsv (all configs, all seeds; `selected` flag),
  lopo_summary_gene_holdout_upgrade.tsv / lopo_patient_level_gene_holdout_upgrade.tsv (selected configuration only,
  method = "nichegnn_upgrade"), upgrade_selection.tsv (per fold: J per configuration).
Usage: python 07c_lopo_upgrade.py [--patients P01,P02] [--n_seeds N] [--tag T]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.data import holdout_genes, load_labels, load_qc_sections  # noqa: E402
from lib.evalprobe import evaluate_heldout  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.train import embed_sections, train_model  # noqa: E402
from lib.upgrade import patient_ari  # noqa: E402

STAGE = "07_lopo"
s07 = import_module("07_lopo_cv")


def variant_name(la: float, lc: float) -> str:
    return ("adv" if la > 0 else "") + ("+" if la > 0 and lc > 0 else "") + ("con" if lc > 0 else "") or "none"


def run_fold(cfg, G, train_secs, val_secs, test_secs, seeds, device, emb_root, P, patient_of):
    up = cfg["upgrade"]; rows, sel = [], []
    tr_g = [G[s] for s in train_secs if s not in val_secs]; va_g = [G[s] for s in val_secs]; te_g = [G[s] for s in test_secs]; all_tr = tr_g + va_g
    grid = [(la, lc) for la, lc in itertools.product(up["lambda_adv_grid"], up["lambda_con_grid"])]
    for la, lc in grid:
        Js = []
        for seed in seeds:
            t0 = time.time()
            model, info = train_model("nichegnn", tr_g, va_g, cfg, seed, device=device, upgrade={"lambda_adv": la, "lambda_con": lc})
            emb = embed_sections(model, [G[s] for s in train_secs + test_secs], "nichegnn", cfg, device)
            inner = evaluate_heldout(emb, tr_g, va_g, cfg, seed); inner.pop("per_section_boundary", None)
            f1_in = inner["macro_f1"] if inner["macro_f1"] == inner["macro_f1"] else inner["accuracy"]
            ari = patient_ari({s: emb[s] for s in train_secs}, patient_of, up["cluster_k"], seed)
            J = f1_in - up["selection_ari_weight"] * ari; Js.append(J)
            held = evaluate_heldout(emb, all_tr, te_g, cfg, seed); per = held.pop("per_section_boundary")
            rows.append({"method": "nichegnn_upgrade", "variant": variant_name(la, lc), "lambda_adv": la, "lambda_con": lc, "ablation": "full", "seed": seed, "heldout_patient": P, "fold": f"LOPO_{P}",
                         "n_train_sections": len(all_tr), "n_test_sections": len(te_g), "run_time_s": round(time.time() - t0, 1), "failed": False, "error": "",
                         **held, "inner_val_macro_f1": f1_in, "train_patient_ari": ari, "J": J, "epochs_run": info["epochs_run"], "n_params": info["n_params"],
                         "best_val_masked_mse": info["best_val_masked_mse"], "per_section_boundary": json.dumps(per)})
            d = emb_root / f"upgrade_adv{la}_con{lc}_{seed}"; d.mkdir(parents=True, exist_ok=True)
            for s in test_secs:
                np.save(d / f"{s}.npy", emb[s])
            log(f"    adv {la} con {lc} seed {seed}: inner F1 {f1_in:.3f} ARI_pat {ari:.3f} J {J:.3f} | held-out F1 {held['macro_f1']:.3f} bF1 {held['boundary_f1']:.3f} ({info['epochs_run']} ep, {time.time() - t0:.0f}s)")
        sel.append({"heldout_patient": P, "lambda_adv": la, "lambda_con": lc, "variant": variant_name(la, lc), "J_mean": float(np.mean(Js)), "J_sd": float(np.std(Js))})
    sel = pd.DataFrame(sel); best = sel.sort_values("J_mean", ascending=False).iloc[0]
    for r in rows:
        r["selected"] = bool(r["lambda_adv"] == best["lambda_adv"] and r["lambda_con"] == best["lambda_con"])
    sel["selected"] = (sel["lambda_adv"] == best["lambda_adv"]) & (sel["lambda_con"] == best["lambda_con"])
    log(f"  fold {P}: selected adv {best['lambda_adv']} con {best['lambda_con']} (J {best['J_mean']:.3f})")
    return rows, sel


def merge_parts(cfg: dict, tags: list[str]) -> None:
    sd = stage_dir(STAGE)
    df = pd.concat([pd.read_csv(sd / f"lopo_metrics_gene_holdout_upgrade{t}.tsv", sep="\t") for t in tags], ignore_index=True)
    sel = pd.concat([pd.read_csv(sd / f"upgrade_selection{t}.tsv", sep="\t") for t in tags], ignore_index=True)
    write_table(df, sd / "lopo_metrics_gene_holdout_upgrade.tsv"); write_table(sel, sd / "upgrade_selection.tsv")
    s07.summarise(df[df["selected"]].copy(), sd, "gene_holdout_upgrade")
    write_provenance(STAGE + "_gene_holdout_upgrade", [sd / f"lopo_metrics_gene_holdout_upgrade{t}.tsv" for t in tags], [sd / "lopo_metrics_gene_holdout_upgrade.tsv", sd / "upgrade_selection.tsv"], cfg,
                     extra={"amendment": "A4", "merged_parts": tags, "n_patients": int(df["heldout_patient"].nunique()), "grid": cfg["upgrade"]})


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--patients", default=""); ap.add_argument("--n_seeds", type=int, default=0); ap.add_argument("--tag", default="")
    ap.add_argument("--merge", default="", help="comma-separated tags of partial runs to merge")
    args = ap.parse_args(); t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"])
    if args.merge:
        merge_parts(cfg, args.merge.split(",")); return
    disc = cfg["cohorts"]["discovery"]; patient_of = disc["patient_groups"]
    secs = [s for s in disc["lung_sections"] if (ROOT / cfg["paths"]["processed"] / "qc" / f"{s}.h5ad").exists()]
    adatas = load_qc_sections(cfg, secs); labels = {s: load_labels(cfg, s) for s in secs}; exclude = holdout_genes(cfg)
    all_pats = sorted({patient_of[s] for s in secs}); patients = [p for p in all_pats if not args.patients or p in args.patients.split(",")]
    seeds = [cfg["seed"] + i for i in range(args.n_seeds or cfg["n_seeds"])]
    emb_root = ROOT / cfg["paths"]["processed"] / "embeddings" / f"lopo_gene_holdout_upgrade{args.tag}"; sd = stage_dir(STAGE)
    rows, sels = [], []
    for i, P in enumerate(patients):
        test_secs = [s for s in secs if patient_of[s] == P]; train_secs = [s for s in secs if patient_of[s] != P]
        tr_pats = sorted({patient_of[s] for s in train_secs}); rng = np.random.default_rng(cfg["seed"] + all_pats.index(P))
        val_pat = tr_pats[int(rng.integers(len(tr_pats)))]; val_secs = [s for s in train_secs if patient_of[s] == val_pat]
        log(f"[fold {i + 1}/{len(patients)}] held-out {P} {test_secs}; inner val {val_pat}")
        G, _ = s07.fold_graphs(cfg, adatas, labels, train_secs, test_secs, patient_of, exclude)
        r, sel = run_fold(cfg, G, train_secs, val_secs, test_secs, seeds, cfg["device"], emb_root, P, patient_of); rows += r; sels.append(sel)
        pd.DataFrame(rows).to_csv(sd / f"lopo_metrics_gene_holdout_upgrade{args.tag}.partial.tsv", sep="\t", index=False)
    df = pd.DataFrame(rows); write_table(df, sd / f"lopo_metrics_gene_holdout_upgrade{args.tag}.tsv")
    write_table(pd.concat(sels, ignore_index=True), sd / f"upgrade_selection{args.tag}.tsv")
    s07.summarise(df[df["selected"]].copy(), sd, f"gene_holdout_upgrade{args.tag}")
    write_provenance(STAGE + f"_gene_holdout_upgrade{args.tag}", [ROOT / cfg["paths"]["processed"] / "qc"], [sd / f"lopo_metrics_gene_holdout_upgrade{args.tag}.tsv"], cfg,
                     extra={"amendment": "A4", "patients": patients, "seeds": seeds, "grid": cfg["upgrade"]}, t_start=t0)


if __name__ == "__main__":
    main()
