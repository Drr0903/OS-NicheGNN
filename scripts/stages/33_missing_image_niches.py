"""Stage 33 — which niches depend on the image: the atlas sections re-assigned with their morphology input absent.

The trained atlas encoder (seed of the main run) embeds every discovery section twice, with its H&E morphology block and with
the block at its training mean (0), exactly as the independent sections OS1–OS9 without usable images were embedded (stage 29,
--image none). Spots are assigned to the nearest atlas niche centroid of the latent (stage 29 rule: centroids of the atlas
niches in the seed latent, unmatched beyond the 97.5 % quantile of the atlas spots' distances to their own centroid).
Per atlas niche: the fraction of its spots that keep their niche (recall), the niche they move to, and the unmatched fraction,
pooled and per patient. The assignment with the image gives the agreement of the centroid rule itself.
Outputs results/33_missing_image_niches/{niche_recall, confusion, patient_recall}.tsv
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from importlib import import_module
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import interpret as I  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.v2 import load_cohort, make_graphs  # noqa: E402

STAGE = "33_missing_image_niches"; torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
s28 = import_module("28_interpretation")


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE); dev = cfg["device"]; seed = cfg["seed"]
    proc = ROOT / cfg["paths"]["processed"]; fro = proc / cfg["atlas"]["frozen_dir"]; pcs = joblib.load(fro / "pca.joblib"); dim = cfg["remedies"]["pca_dim"]
    adatas, labels, patient_of, _ = load_cohort(cfg); secs = list(adatas); G, _, _, _ = make_graphs(cfg, adatas, labels, patient_of, secs, secs)
    F = {s: np.load(proc / "image_features" / f"{s}.npy") for s in secs}
    prof = lambda g, f: np.hstack([pcs["expr_pca"].transform(g.x) / pcs["expr_sd"], pcs["img_pca"].transform(f) / pcs["img_sd"]]).astype(np.float32)  # noqa: E731
    model = s28.load_encoder(cfg, fro, seed, 2 * dim)
    meta = json.load(open(ROOT / "results/15_niches/meta.json")); K = meta["K"]; names = {int(k): v for k, v in meta["names"].items()}
    spots = pd.read_csv(ROOT / "results/15_niches/spot_niches.tsv", sep="\t")
    nd = {s: spots[spots["section"] == s].set_index("barcode").reindex(G[s].barcodes) for s in secs}
    atlas = np.concatenate([np.where(nd[s]["unmatched"].fillna(True).astype(bool), -1, nd[s]["niche"].fillna(-1)).astype(int) for s in secs])
    pat = np.concatenate([[patient_of[s]] * G[s].n for s in secs]); out = {}
    for cond in ("present", "absent"):
        Z = []
        for s in secs:
            x = prof(G[s], F[s])
            if cond == "absent":
                x[:, dim:] = 0.0
            Z.append(I.latent(model, dataclasses.replace(G[s], x=x), dev))
        out[cond] = np.vstack(Z)
    C = np.vstack([out["present"][atlas == k].mean(0) for k in range(K)])
    dd = np.linalg.norm(out["present"][atlas >= 0] - C[atlas[atlas >= 0]], axis=1); thr = float(np.quantile(dd, cfg["niches"]["unmatched_quantile"]))
    rec, conf, prec = [], [], []
    for cond, Z in out.items():
        d = np.linalg.norm(Z[:, None] - C[None], axis=2); lab = np.where(d.min(1) > thr, -1, d.argmin(1)); ok = atlas >= 0
        for k in range(K):
            m = ok & (atlas == k); moved = pd.Series(lab[m]).value_counts(normalize=True)
            top = moved.drop(k, errors="ignore"); top = top[top.index >= 0]
            rec.append({"image": cond, "niche": k, "name": names[k], "n_spots": int(m.sum()), "recall": float((lab[m] == k).mean()), "unmatched": float((lab[m] == -1).mean()),
                        "main_destination": names.get(int(top.index[0]), "") if len(top) else "", "main_destination_fraction": float(top.iloc[0]) if len(top) else 0.0})
            for p in np.unique(pat[m]):
                mp = m & (pat == p)
                if mp.sum() >= 20:
                    prec.append({"image": cond, "niche": k, "patient": p, "n_spots": int(mp.sum()), "recall": float((lab[mp] == k).mean())})
        ct = pd.crosstab(pd.Series(atlas[ok], name="atlas_niche"), pd.Series(lab[ok], name="assigned"), normalize="index").reset_index(); ct.insert(0, "image", cond); conf.append(ct)
        log(f"image {cond}: overall agreement {(lab[ok] == atlas[ok]).mean():.3f}, unmatched {(lab[ok] == -1).mean():.3f}")
    o = [sd / "niche_recall.tsv", sd / "confusion.tsv", sd / "patient_recall.tsv"]
    rr = pd.DataFrame(rec); write_table(rr, o[0]); write_table(pd.concat(conf, ignore_index=True), o[1]); write_table(pd.DataFrame(prec), o[2])
    log("\n" + rr.pivot(index="name", columns="image", values="recall").round(3).to_string())
    write_provenance(STAGE, [ROOT / "results/15_niches/spot_niches.tsv", fro / f"model_seed{seed}.pt"], o, cfg, t_start=t0, extra={"unmatched_threshold": thr})


if __name__ == "__main__":
    main()
