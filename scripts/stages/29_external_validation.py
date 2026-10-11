"""Stage 29 () — the delivered OS-NicheGNN applied, without any refit, to an independent osteosarcoma cohort.

External cohort OS1–OS9 (seven lung and two lymph-node metastases; stage 02b/03/04/05). The delivered encoder (seed 1), its
gene list, scaler and principal-component models embed every external section; H&E features are Phikon on the distributed
image (low resolution). Spots are assigned to the nearest atlas niche centroid of the latent (centroids of the delivered
niches in the seed-1 discovery latent; unmatched beyond the 97.5 % quantile of discovery spot-to-centroid distances).
Outcomes (A22): (i) region recovery of the discovery-fitted linear probe against the external algorithmic labels;
(ii) niche recurrence and composition correlation with the discovery atlas; (iii) Spearman of the niche T-cell scores;
(iv) boundary odds ratio of the band; (v) interface coverage–T association in the lung sections (pooled model).
Outputs results/29_external/{region_metrics, niche_by_section, niche_comparison, interface_section_metrics,
        interface_pooled_models}.tsv, summary.json
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
from PIL import Image
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import interpret as I  # noqa: E402
from lib.data import build_graph, load_labels, load_qc_sections  # noqa: E402
from lib.evalprobe import CLASSES, boundary_from_labels, boundary_prf, fit_probe  # noqa: E402
from lib.inference import pooled_models  # noqa: E402
from lib.interface import run_interface_cohort, t_score  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.v2 import load_cohort, make_graphs  # noqa: E402

STAGE = "29_external"; torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
s04c = import_module("04c_image_features"); s28 = import_module("28_interpretation")


def ext_image_features(cfg, a, sec) -> np.ndarray:
    from transformers import ViTModel
    ic = cfg["image"]; model = ViTModel.from_pretrained(ic["model"], revision=ic["revision"], add_pooling_layer=False).to(cfg["device"]).eval()
    img = np.asarray(Image.open(ROOT / cfg["paths"]["processed"] / "external_export" / sec / "image.png").convert("RGB")); pitch = float(a.uns["spot_pitch_img_px"])
    b = a.copy(); b.uns["scalefactors"] = {"tissue_hires_scalef": 1.0}; side = max(8, int(round(ic["patch_spots"] * 0.55 * pitch)))  # coordinates are image pixels
    return s04c.section_features(model, b, img, side, cfg["device"]).astype(np.float32)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--image", choices=["lowres", "none"], default="lowres", help="none: morphology block set to its mean (0)"); args = ap.parse_args()
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE) / args.image; sd.mkdir(exist_ok=True); dev = cfg["device"]; seed = cfg["seed"]
    proc = ROOT / cfg["paths"]["processed"]; fro = proc / cfg["atlas"]["frozen_dir"]; pcs = joblib.load(fro / "pca.joblib"); dim = cfg["remedies"]["pca_dim"]
    genes = open(fro / "genes.txt").read().split(); sc_ = np.load(fro / "scaler.npz")
    ext = cfg["cohorts"]["external"]; lung, ln = ext["lung_sections"], ext["lymph_node_sections"]; secs = lung + ln
    # discovery side: latent, niches, centroids, probe
    adatas, labels, patient_of, _ = load_cohort(cfg); dsecs = list(adatas)
    G, g2, _, _ = make_graphs(cfg, adatas, labels, patient_of, dsecs, dsecs); assert g2 == genes
    F = {s: np.load(proc / "image_features" / f"{s}.npy") for s in dsecs}
    prof = lambda g, f: np.hstack([pcs["expr_pca"].transform(g.x) / pcs["expr_sd"], pcs["img_pca"].transform(f) / pcs["img_sd"]]).astype(np.float32)  # noqa: E731
    model = s28.load_encoder(cfg, fro, seed, 2 * dim)
    Zd = {s: I.latent(model, dataclasses.replace(G[s], x=prof(G[s], F[s])), dev) for s in dsecs}
    spots = pd.read_csv(ROOT / "results/15_niches/spot_niches.tsv", sep="\t"); meta = json.load(open(ROOT / "results/15_niches/meta.json")); K = meta["K"]; kb = meta["interface_niche"]
    names = {int(k): v for k, v in meta["names"].items()}
    nd = {s: spots[spots["section"] == s].set_index("barcode").reindex(G[s].barcodes) for s in dsecs}
    Zall = np.vstack([Zd[s] for s in dsecs]); nall = np.concatenate([np.where(nd[s]["unmatched"].fillna(True).astype(bool), -1, nd[s]["niche"].fillna(-1)).astype(int) for s in dsecs])
    C = np.vstack([Zall[nall == k].mean(0) for k in range(K)]); dd = np.linalg.norm(Zall[nall >= 0] - C[nall[nall >= 0]], axis=1); thr = float(np.quantile(dd, cfg["niches"]["unmatched_quantile"]))
    lab_d = np.concatenate([np.asarray(G[s].labels, dtype=object) for s in dsecs]); m = np.isin(lab_d, CLASSES); scaler, clf = fit_probe(Zall[m], lab_d[m], seed)
    # external side
    ea = load_qc_sections(cfg, secs); el = {s: load_labels(cfg, s) for s in secs}
    EG = {s: build_graph(ea[s], genes, sc_["mu"], sc_["sd"], cfg, f"EXT_{s}", el[s]) for s in secs}
    reg, rows = [], []
    from sklearn.metrics import f1_score
    for s in secs:
        x = prof(EG[s], ext_image_features(cfg, ea[s], s) if args.image == "lowres" else np.zeros((EG[s].n, 768), np.float32) + pcs["img_pca"].mean_)
        if args.image == "none":
            x[:, dim:] = 0.0
        g = dataclasses.replace(EG[s], x=x); z = I.latent(model, g, dev)
        d = np.linalg.norm(z[:, None] - C[None], axis=2); niche = d.argmin(1); um = d.min(1) > thr
        pred = clf.predict(scaler.transform(z)); ev = np.isin(EG[s].labels, CLASSES); r = {"section": s, "organ": "lung" if s in lung else "lymph_node", "n_eval_spots": int(ev.sum())}
        if ev.sum() > 20:
            r["macro_f1"] = float(f1_score(EG[s].labels[ev], pred[ev], average="macro"))
            r.update(boundary_prf(boundary_from_labels(pred, EG[s].xy, EG[s].pitch), EG[s].boundary, EG[s].xy, EG[s].pitch, cfg["histology"]["boundary_tolerance_rings"]))
        reg.append(r)
        rows.append(pd.DataFrame({"section": s, "patient": f"EXT_{s}", "organ": r["organ"], "barcode": EG[s].barcodes, "niche": niche, "dist_to_prototype": d.min(1), "unmatched": um,
                                  "histology_label": EG[s].labels, "tumor_boundary": EG[s].boundary, "tumor_lung_interface": EG[s].interface, "x": EG[s].xy[:, 0], "y": EG[s].xy[:, 1],
                                  "t_score": t_score(ea[s], cfg["genes"]["t_cell_holdout"])}))
        log(f"{s}: {r.get('macro_f1', float('nan')):.3f} macro-F1, unmatched {um.mean():.3f}")
    sp = pd.concat(rows, ignore_index=True); write_table(pd.DataFrame(reg), sd / "region_metrics.tsv")
    mt = sp[~sp["unmatched"]]; write_table(pd.crosstab(mt["section"], mt["niche"], normalize="index").reindex(columns=range(K), fill_value=0).reset_index(), sd / "niche_by_section.tsv")
    # niche comparison: presence, composition, T score, boundary odds
    w_dir = ROOT / "results/05_deconvolution/weights"; W = []
    for s in secs:
        w = pd.read_csv(w_dir / f"{s}_rctd_weights.tsv", sep="\t").set_index("barcode"); sub = sp[sp["section"] == s].set_index("barcode"); W.append(w.reindex(sub.index).assign(niche=sub["niche"].values, unmatched=sub["unmatched"].values, section=s))
    W = pd.concat(W); types = [c for c in W.columns if c not in ("niche", "unmatched", "section")]; ec = W[~W["unmatched"]].groupby("niche")[types].mean()
    dc = pd.read_csv(ROOT / "results/15_niches/niche_composition_rctd.tsv", sep="\t").set_index("niche")
    tsd = pd.read_csv(ROOT / "results/15_niches/niche_tcell_patient_level.tsv", sep="\t").set_index("niche")["mean"]
    te = mt.groupby(["section", "niche"])["t_score"].agg(["mean", "size"]).reset_index(); te = te[te["size"] >= cfg["v2"]["min_spots_per_patient_niche"]].groupby("niche")["mean"].mean()
    frac = pd.crosstab(mt["section"], mt["niche"], normalize="index").reindex(columns=range(K), fill_value=0)
    cmp = []
    for k in range(K):
        cmp.append({"niche": k, "name": names[k], "n_spots": int((mt["niche"] == k).sum()), "n_sections_present_2pct": int((frac[k] >= 0.02).sum()),
                    "composition_r": float(stats.pearsonr(ec.loc[k, types], dc.loc[k, types])[0]) if k in ec.index and k in dc.index else np.nan,
                    "t_score_external": float(te.get(k, np.nan)), "t_score_discovery": float(tsd.get(k, np.nan))})
    cmp = pd.DataFrame(cmp)
    # boundary odds of the band (spots within one ring of a tumour-boundary spot), as stage 15
    from lib.spatial import ring_neighbors
    near_b = []
    for s in secs:
        sub = mt[mt["section"] == s]; g = EG[s]; nbh = ring_neighbors(g.xy, g.pitch, rings=(1,)); b = np.asarray(g.boundary, bool)
        nearb = np.array([b[i] or b[nbh[i]].any() for i in range(len(b))]); idx = pd.Index(g.barcodes).get_indexer(sub["barcode"]); near_b.append(pd.DataFrame({"niche": sub["niche"].to_numpy(), "near": nearb[idx]}))
    nbd = pd.concat(near_b)
    def odds(k):
        a_ = nbd[nbd["niche"] == k]["near"]; o_ = nbd[nbd["niche"] != k]["near"]
        return float((a_.mean() / max(1 - a_.mean(), 1e-9)) / (o_.mean() / max(1 - o_.mean(), 1e-9))) if len(a_) else np.nan
    cmp["odds_ratio_near_boundary"] = [odds(k) for k in range(K)]; write_table(cmp, sd / "niche_comparison.tsv")
    ok = cmp.dropna(subset=["t_score_external", "t_score_discovery"])
    summ = {"macro_f1_lung_mean": float(pd.DataFrame(reg).query("organ == 'lung'")["macro_f1"].mean()), "unmatched_fraction": float(sp["unmatched"].mean()),
            "n_niches_present": int((cmp["n_sections_present_2pct"] > 0).sum()), "composition_r_mean": float(cmp["composition_r"].mean()),
            "t_order_spearman": float(stats.spearmanr(ok["t_score_external"], ok["t_score_discovery"]).correlation) if len(ok) >= 4 else None, "n_niches_t": int(len(ok)),
            "band_odds_ratio": float(cmp.loc[cmp["niche"] == kb, "odds_ratio_near_boundary"].iloc[0])}
    # interface: band coverage vs T (lung sections), same rules as discovery
    man = pd.read_csv(ROOT / "results/01_manifest/sample_manifest.tsv", sep="\t", dtype=str).set_index("section_id")
    elig = [s for s in lung if man.loc[s, "boundary_eligible"] == "True"]
    sp = sp.merge(W.reset_index()[["barcode", "section"] + types], on=["barcode", "section"], how="left")
    sec, seg, con = run_interface_cohort(sp, proc / "qc", w_dir, cfg, kb, elig, {s: f"EXT_{s}" for s in elig})
    write_table(sec, sd / "interface_section_metrics.tsv")
    if len(seg):
        pm = pooled_models(seg, cluster="section", n_perm=cfg["interface"]["n_permutations"], seed=seed); write_table(pm, sd / "interface_pooled_models.tsv")
        k_ = pm.set_index(["outcome", "covariate_set"]).loc[("E_seg", "none")]
        summ.update(interface_coef=float(k_.get("coef", np.nan)), interface_p=float(k_.get("p_t_gminus1", np.nan)), interface_n_sections=int(k_.get("n_sections", 0)))
    write_table(sp.drop(columns=["x", "y"]), sd / "spot_niches_external.tsv")
    json.dump(summ, open(sd / "summary.json", "w"), indent=2); log(json.dumps(summ, indent=1))
    write_provenance(f"{STAGE}_{args.image}", [fro / f"model_seed{seed}.pt"], [sd / "niche_comparison.tsv", sd / "region_metrics.tsv", sd / "summary.json"], cfg, extra=summ, t_start=t0)


if __name__ == "__main__":
    main()
