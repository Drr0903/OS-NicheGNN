"""Stage 24 () — the frozen encoder of the atlas: the A18 selected architecture trained on all 19 patients.

Spot profile = [expression PCA-50, H&E morphology PCA-50] (both PCA fitted on all sections); a cross-attention context lets each
spot weight its ring-1–3 neighbours by its own state; masked-spot reconstruction plus the cross-patient anchor term (A10).
The representation used downstream is [spot profile, learned context] (the `_skip` form scored in stage 23).
Training length is fixed (atlas.frozen_epochs, the median of the leave-one-patient-out runs), no early stopping.

  --seeds 20260914,20260915   train these seeds and write encoder weights + embeddings (can run in parallel per GPU)
  --retrain                   write to the re-training directories instead (run-to-run sensitivity, stage 19)
  --finalise                  merge the per-seed identity rows, write meta.json and PROVENANCE
Outputs data/processed/<frozen_dir>/{genes.txt, scaler.npz, pca.joblib, model_seed*.pt, meta.json},
        data/processed/embeddings/<emb_dir>/<seed>/<section>.npy, results/24_frozen_encoder/identity_seed*.tsv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.arch_train import embed_arch, train_arch  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.v2 import compute_anchors, identity_index, load_cohort, make_graphs  # noqa: E402

STAGE = "24_frozen_encoder"


def profiles(cfg: dict, G: dict, secs: list[str], fro: Path) -> dict:
    """Expression PCA-50 and morphology PCA-50 fitted on all sections, each divided by one global s.d. (= lib.arch_eval)."""
    import dataclasses
    dim = cfg["remedies"]["pca_dim"]; seed = cfg["seed"]; feat = ROOT / cfg["paths"]["processed"] / "image_features"
    X = np.vstack([G[s].x for s in secs]); pe = PCA(dim, random_state=seed).fit(X); se = float(pe.transform(X).std())
    F = {s: np.load(feat / f"{s}.npy") for s in secs}
    for s in secs:
        assert len(F[s]) == G[s].n, f"{s}: image rows {len(F[s])} != spots {G[s].n}"
    Fa = np.vstack([F[s] for s in secs]); pi = PCA(dim, random_state=seed).fit(Fa); si = float(pi.transform(Fa).std())
    joblib.dump({"expr_pca": pe, "expr_sd": se, "img_pca": pi, "img_sd": si}, fro / "pca.joblib")
    return {s: dataclasses.replace(G[s], x=np.hstack([pe.transform(G[s].x) / se, pi.transform(F[s]) / si]).astype(np.float32)) for s in secs}


VARIANTS = {  # A19 candidates scored in stage 23: (context, multimodal, representation)
    "mm_xattn_anchor_skip": ("xattn", True, "skip"), "xattn_anchor_z": ("xattn", False, "z"), "mm_xattn_anchor_z": ("xattn", True, "z"),
    "xattn_anchor_skip": ("xattn", False, "skip")}


def dirs(cfg: dict, retrain: bool, variant: str = "") -> tuple[Path, Path]:
    a = cfg["atlas"]; proc = ROOT / cfg["paths"]["processed"]
    if variant:
        return proc / f"frozen_{variant}{'_retrain' if retrain else ''}", proc / "embeddings" / f"{variant}{'_retrain' if retrain else ''}"
    return proc / a["retrain_frozen_dir" if retrain else "frozen_dir"], proc / "embeddings" / a["retrain_emb_dir" if retrain else "emb_dir"]


def train_seeds(cfg: dict, seeds: list[int], retrain: bool, variant: str = "", epochs: int = 0) -> None:
    a = cfg["atlas"]; K = cfg["v2"]["k_fixed"]; sd = stage_dir(STAGE); fro, embd = dirs(cfg, retrain, variant); fro.mkdir(parents=True, exist_ok=True)
    context, mm, rep = VARIANTS.get(variant or a["encoder"], (a["context"], a["multimodal"], a["representation"]))
    adatas, labels, patient_of, _ = load_cohort(cfg); secs = list(adatas)
    G, genes, mu, sd_ = make_graphs(cfg, adatas, labels, patient_of, secs, secs)
    open(fro / "genes.txt", "w").write("\n".join(genes)); np.savez(fro / "scaler.npz", mu=mu, sd=sd_)
    Gm = profiles(cfg, G, secs, fro)
    if not mm:  # expression PCA only
        import dataclasses
        Gm = {s: dataclasses.replace(Gm[s], x=Gm[s].x[:, :cfg["remedies"]["pca_dim"]].copy()) for s in secs}
    anchors = compute_anchors(cfg, [Gm[s] for s in secs])
    log(f"{len(secs)} sections, profile dim {Gm[secs[0]].x.shape[1]}, {sum(len(v[0]) for v in anchors.values()) // 2} anchor pairs")
    cfg_f = json.loads(json.dumps(cfg)); cfg_f["arch"].update(a.get("arch_overrides") or {})  # width / niche targets of the atlas encoder
    cfg_f["arch"]["max_epochs"] = epochs or a["frozen_epochs"]; cfg_f["arch"]["patience"] = 10**9
    for seed in seeds:
        model, info = train_arch(context, [Gm[s] for s in secs], [], cfg_f, seed, cfg["device"], anchors, cfg["atlas"].get("lambda_anchor", cfg["v2"]["model"]["lambda_anchor"]))
        torch.save(model.state_dict(), fro / f"model_seed{seed}.pt"); E = embed_arch(model, [Gm[s] for s in secs], cfg["device"])
        d = embd / str(seed); d.mkdir(parents=True, exist_ok=True); emb = {}
        for s in secs:
            emb[s] = (np.hstack([Gm[s].x, E[s]["c"]]) if rep == "skip" else E[s]["z"]).astype(np.float32); np.save(d / f"{s}.npy", emb[s])
        idx = identity_index(emb, G, K, seed); log(f"seed {seed}: {info['epochs_run']} epochs, {info['train_time_s']} s; identity {idx}")
        tag = ("_retrain" if retrain else "") + (f"_{variant}" if variant and variant != a["encoder"] else "")
        write_table(pd.DataFrame([{"seed": seed, **{k: v for k, v in info.items() if k != "context"}, **idx}]), sd / f"identity_seed{seed}{tag}.tsv")


def finalise(cfg: dict, retrain: bool) -> None:
    t0 = time.time(); sd = stage_dir(STAGE); fro, embd = dirs(cfg, retrain); tag = "_retrain" if retrain else ""
    files = sorted(sd.glob(f"identity_seed*{tag}.tsv")) if retrain else sorted(f for f in sd.glob("identity_seed*.tsv") if "_retrain" not in f.name)
    idt = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True); write_table(idt, sd / f"identity{tag}.tsv")
    adatas, _, patient_of, site_of = load_cohort(cfg); secs = list(adatas); a = cfg["atlas"]
    meta = {"plan": "v2", "amendment": a.get("amendment", "A19"), "variant": a["encoder"], "arch_overrides": a.get("arch_overrides"), "kind": "arch_" + a["context"], "multimodal": a["multimodal"], "K": cfg["v2"]["k_fixed"],
            "frozen_seed": cfg["seed"], "n_epochs": a["frozen_epochs"], "n_genes": len(open(fro / "genes.txt").read().split()), "sections": secs,
            "patients": sorted(set(patient_of.values())), "sites": site_of, "seeds": idt["seed"].tolist(),
            "identity_mean": idt[["ari_vs_patient", "ari_vs_region", "pii", "n_present_half_patients"]].mean().to_dict()}
    json.dump(meta, open(fro / "meta.json", "w"), indent=2); log(json.dumps(meta["identity_mean"]))
    write_provenance(STAGE + tag, files, [fro / "meta.json", sd / f"identity{tag}.tsv"], cfg, extra={k: v for k, v in meta.items() if k not in ("sections", "sites")}, t_start=t0)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default=""); ap.add_argument("--variant", default=""); ap.add_argument("--epochs", type=int, default=0); ap.add_argument("--retrain", action="store_true"); ap.add_argument("--finalise", action="store_true")
    a = ap.parse_args(); cfg = load_config(); set_global_seed(cfg["seed"])
    if a.finalise:
        finalise(cfg, a.retrain); return
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else [cfg["seed"] + i for i in range(cfg["n_seeds"])]
    train_seeds(cfg, seeds, a.retrain, a.variant, a.epochs)


if __name__ == "__main__":
    main()
