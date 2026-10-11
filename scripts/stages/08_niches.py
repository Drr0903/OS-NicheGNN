"""Stage 08 — frozen model, niche prototypes and their characterisation (discovery cohort only).

1. Final OS-NicheGNN: HVGs (+T-cell hold-out) and scaler fitted on ALL discovery lung sections; trained on all
   ten patients for a fixed number of epochs = median epochs_run of the LOPO runs (no early stopping, no
   test data). n_seeds models are trained; the set in advance frozen model is seed index 0, the others
   quantify seed stability (ARI of niche labels).
2. Prototypes : Leiden communities on a kNN graph of the patient-balanced training embeddings with the
   resolution searched for exactly K = niches.k_fixed communities; prototypes = community centroids. Stability = mean ARI
   over `stability_bootstrap` refits that each drop two random patients (at the fixed K). Presence of every niche in ≥ half
   of the patients is reported for K = 4–8 descriptively; K is no longer chosen from the data.
3. Unmatched rule: distance to the nearest prototype above the training `unmatched_quantile`.
4. Characterisation tables: niche × RCTD composition, niche × image label, niche × patient fractions,
   boundary enrichment (odds of a niche within ring-1 of the tumour boundary). The "interface niche" used
   downstream is the non-tumour-dominant niche with the highest boundary odds ratio.
Outputs
  data/processed/frozen/{model_seed*.pt, genes.txt, scaler.npz, prototypes.npy, meta.json}
  data/processed/embeddings/frozen/<section>.npy
  results/08_niches/{k_selection.tsv, spot_niches.tsv, niche_composition_rctd.tsv, niche_histology.tsv,
                     niche_by_patient.tsv, niche_boundary_enrichment.tsv, seed_stability.tsv, niche_names.tsv}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.cluster import assign, cluster_prototypes, leiden_fixed_k  # noqa: E402
from lib.data import build_graph, fit_scaler, holdout_genes, load_labels, load_qc_sections, select_hvgs  # noqa: E402
from lib.repro import ROOT, TRACK, frozen_name, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.spatial import ring_neighbors  # noqa: E402
from lib.train import embed_sections, train_model  # noqa: E402

STAGE = "08_niches"


def balanced_subsample(emb: dict, patient_of: dict, per_patient: int, rng) -> tuple[np.ndarray, np.ndarray]:
    Z, P = [], []
    for pat in sorted(set(patient_of.values())):
        zs = np.vstack([emb[s] for s in emb if patient_of[s] == pat])
        idx = rng.choice(len(zs), min(per_patient, len(zs)), replace=False)
        Z.append(zs[idx]); P += [pat] * len(idx)
    return np.vstack(Z), np.array(P)


def k_diagnostics(Z, P, cfg, rng) -> tuple[pd.DataFrame, int]:
    """Descriptive: presence of every niche in ≥ half of the patients for each candidate K (single Leiden fit) and
    patient-drop stability at the fixed K. K itself is fixed by configuration (A5)."""
    rows = []; pats = sorted(set(P)); K = int(cfg["niches"]["k_fixed"]); knn = cfg["niches"]["knn"]
    for k in cfg["niches"]["k_candidates"]:
        lab, res = leiden_fixed_k(Z, k, cfg["seed"], batch=P, k_within=cfg["niches"]["k_within"])
        aris = []
        if k == K:
            for b in range(cfg["niches"]["stability_bootstrap"]):
                drop = rng.choice(pats, 2, replace=False); keep = ~np.isin(P, drop)
                lb, _ = leiden_fixed_k(Z[keep], k, cfg["seed"] + b, batch=P[keep], k_within=cfg["niches"]["k_within"]); aris.append(adjusted_rand_score(lab[keep], lb))
        frac = pd.crosstab(P, lab, normalize="index"); present = (frac >= 0.02).sum(0) >= len(pats) / 2
        rows.append({"k": k, "resolution": res, "stability_ari_mean": float(np.mean(aris)) if aris else np.nan, "stability_ari_sd": float(np.std(aris)) if aris else np.nan,
                     "n_niches_present_in_half_patients": int(present.sum()), "all_niches_present": bool(present.all()), "selected": k == K})
        log(f"  K = {k}: resolution {res:.3f}, niches present in ≥ half of patients {int(present.sum())}/{k}" + (f", stability ARI {np.mean(aris):.2f}" if aris else ""))
    return pd.DataFrame(rows), K


def main() -> None:
    t0 = time.time()
    cfg = load_config(); set_global_seed(cfg["seed"])
    disc = cfg["cohorts"]["discovery"]; patient_of = disc["patient_groups"]
    secs = [s for s in disc["lung_sections"] if (ROOT / cfg["paths"]["processed"] / "qc" / f"{s}.h5ad").exists()]
    adatas = load_qc_sections(cfg, secs); labels = {s: load_labels(cfg, s) for s in secs}
    exclude = holdout_genes(cfg)
    genes = select_hvgs([adatas[s] for s in secs], cfg["genes"]["n_hvg"], exclude)
    mu, sd = fit_scaler([adatas[s] for s in secs], genes)
    G = {s: build_graph(adatas[s], genes, mu, sd, cfg, patient_of[s], labels[s]) for s in secs}
    lopo = pd.read_csv(ROOT / "results/07_lopo/lopo_metrics_gene_holdout.tsv", sep="\t")
    ep = lopo[(lopo["method"] == "nichegnn") & (lopo["ablation"] == "full") & (~lopo["failed"])]["epochs_run"]
    n_epochs = int(np.median(ep)) if len(ep) else cfg["train"]["max_epochs"]
    log(f"final model: {len(secs)} sections, {len(genes)} genes, fixed {n_epochs} epochs (median of LOPO)")
    cfg_final = json.loads(json.dumps(cfg)); cfg_final["train"]["max_epochs"] = n_epochs; cfg_final["train"]["patience"] = 10**9
    fro = ROOT / cfg["paths"]["processed"] / frozen_name(); fro.mkdir(exist_ok=True)
    emb_dir = ROOT / cfg["paths"]["processed"] / "embeddings" / frozen_name(); emb_dir.mkdir(parents=True, exist_ok=True)
    upgrade = None
    if TRACK == "invariant":  #  second track: the adopted patient-invariant configuration
        from lib.upgrade import mnn_anchors
        inv = json.load(open(ROOT / "results/08_niches/invariant_summary.json")); assert inv["upgrade_adopted"], "A7 criteria not met — no second track"
        upgrade = {"lambda_anchor": inv["lambda_anchor"], "lambda_con": inv["lambda_con"], "lambda_nbr": cfg["invariant"]["lambda_nbr"], "anchors": mnn_anchors([G[s] for s in secs], n_pcs=cfg["invariant"]["anchor_pcs"], k=cfg["invariant"]["anchor_k"], seed=cfg["seed"])}
        log(f"track invariant: lambda_anchor {inv['lambda_anchor']}, lambda_con {inv['lambda_con']}, lambda_nbr {cfg['invariant']['lambda_nbr']}")
    seeds = [cfg["seed"] + i for i in range(cfg["n_seeds"])]
    embs, infos = {}, []
    for seed in seeds:
        model, info = train_model("nichegnn", [G[s] for s in secs], [], cfg_final, seed, device=cfg["device"], upgrade=upgrade)
        torch.save(model.state_dict(), fro / f"model_seed{seed}.pt")
        embs[seed] = embed_sections(model, [G[s] for s in secs], "nichegnn", cfg, cfg["device"])
        infos.append({k: v for k, v in info.items() if k != "history"})
        log(f"  seed {seed}: final train masked-MSE {info['history'][-1]['train_loss']:.4f}")
    open(fro / "genes.txt", "w").write("\n".join(genes)); np.savez(fro / "scaler.npz", mu=mu, sd=sd)
    frozen_seed = seeds[0]
    for s in secs:
        np.save(emb_dir / f"{s}.npy", embs[frozen_seed][s])
    rng = np.random.default_rng(cfg["seed"])
    Z, P = balanced_subsample(embs[frozen_seed], patient_of, 1500, rng)
    # sklearn (OpenBLAS/OpenMP) after torch in one process segfaults on this 256-core host unless BLAS threads are capped
    tp = threadpool_limits(limits=8); tp.__enter__()
    ksel, K = k_diagnostics(Z, P, cfg, rng)
    log(f"K diagnostics (K fixed at {K} by A5):\n{ksel.round(3).to_string()}")
    _, protos, cinfo = cluster_prototypes(Z, K, cfg["seed"], batch=P, k_within=cfg["niches"]["k_within"]); np.save(fro / "prototypes.npy", protos)
    log(f"Leiden (balanced kNN, K = {K}): resolution {cinfo['resolution']:.3f}; agreement of nearest-centroid assignment with communities ARI {cinfo['centroid_agreement_ari']:.2f}")
    # assign all discovery spots; unmatched threshold from training distances
    rows = []
    for s in secs:
        lab, dist = assign(embs[frozen_seed][s], protos)
        g = G[s]
        rows.append(pd.DataFrame({"section": s, "patient": patient_of[s], "barcode": g.barcodes, "niche": lab, "dist_to_prototype": dist,
                                  "histology_label": g.labels, "tumor_boundary": g.boundary, "tumor_lung_interface": g.interface, "x": g.xy[:, 0], "y": g.xy[:, 1]}))
    spots = pd.concat(rows, ignore_index=True)
    thr = float(np.quantile(spots["dist_to_prototype"], cfg["niches"]["unmatched_quantile"]))
    spots["unmatched"] = spots["dist_to_prototype"] > thr
    # seed stability of niche labels (Leiden with the same fixed K on other seeds' embeddings, prototypes → all spots, ARI vs frozen)
    stab = []; lf = spots["niche"].to_numpy()
    for seed in seeds[1:]:
        Zs, Ps = balanced_subsample(embs[seed], patient_of, 1500, np.random.default_rng(cfg["seed"]))
        _, pr_s, _ = cluster_prototypes(Zs, K, cfg["seed"], batch=Ps, k_within=cfg["niches"]["k_within"]); ls = np.concatenate([assign(embs[seed][s], pr_s)[0] for s in secs])
        stab.append({"seed": seed, "ari_vs_frozen_seed": adjusted_rand_score(lf, ls)})
    # characterisation
    sd_ = stage_dir(STAGE)
    w_dir = ROOT / "results/05_deconvolution/weights"
    comp_rows = []
    for s in secs:
        p = w_dir / f"{s}_rctd_weights.tsv"
        if p.exists():
            w = pd.read_csv(p, sep="\t").set_index("barcode"); sub = spots[spots["section"] == s].set_index("barcode")
            w = w.reindex(sub.index); w["niche"] = sub["niche"].values; w["section"] = s; comp_rows.append(w)
    if comp_rows:
        W = pd.concat(comp_rows); types = [c for c in W.columns if c not in ("niche", "section")]
        write_table(W.groupby("niche")[types].mean().reset_index(), sd_ / "niche_composition_rctd.tsv")
        spots = spots.merge(W.reset_index()[["barcode", "section"] + types], on=["barcode", "section"], how="left")
    write_table(pd.crosstab(spots["niche"], spots["histology_label"], normalize="index").reset_index(), sd_ / "niche_histology.tsv")
    write_table(pd.crosstab(spots["patient"], spots["niche"], normalize="index").reset_index(), sd_ / "niche_by_patient.tsv")
    # boundary enrichment: odds ratio of niche among spots within ring-1 of a tumour-boundary spot vs elsewhere
    near_b = np.zeros(len(spots), bool); off = 0
    for s in secs:
        g = G[s]; nb = ring_neighbors(g.xy, g.pitch, rings=(1,)); b = g.boundary
        nb_flag = np.array([b[i] or b[nb[i]].any() if len(nb[i]) else b[i] for i in range(g.n)])
        near_b[off:off + g.n] = nb_flag; off += g.n
    spots["near_boundary"] = near_b
    enr = []
    for k in range(K):
        a = ((spots["niche"] == k) & spots["near_boundary"]).sum(); b = ((spots["niche"] != k) & spots["near_boundary"]).sum()
        c = ((spots["niche"] == k) & ~spots["near_boundary"]).sum(); d = ((spots["niche"] != k) & ~spots["near_boundary"]).sum()
        enr.append({"niche": k, "n_spots": int((spots["niche"] == k).sum()), "frac_near_boundary": a / max(a + c, 1), "odds_ratio_near_boundary": (a * d) / max(b * c, 1),
                    "tumor_frac_histology": float(((spots["niche"] == k) & (spots["histology_label"] == "tumor_cellular")).sum() / max((spots["niche"] == k).sum(), 1))})
    enr = pd.DataFrame(enr)
    tumor_dom = enr[enr["tumor_frac_histology"] >= 0.6]["niche"].tolist()
    cand = enr[~enr["niche"].isin(tumor_dom)]
    interface_niche = int(cand.sort_values("odds_ratio_near_boundary", ascending=False)["niche"].iloc[0])
    enr["is_interface_niche"] = enr["niche"] == interface_niche
    write_table(enr, sd_ / "niche_boundary_enrichment.tsv")
    # neutral, rule-based niche names (functional labels are NOT assigned here)
    hist_tab = pd.crosstab(spots["niche"], spots["histology_label"], normalize="index")
    names, used = {}, {}
    for k in range(K):
        tf = float(hist_tab.loc[k].get("tumor_cellular", 0.0)); lf = float(hist_tab.loc[k].get("lung_parenchyma", 0.0))
        base = "fibroblast-rich boundary" if k == interface_niche else "tumour-enriched" if tf >= 0.6 else "lung-associated" if lf >= 0.5 else "mixed"
        used[base] = used.get(base, 0) + 1
        names[k] = base if base == "fibroblast-rich boundary" else f"{base} {used[base]}"
    json.dump({str(k): v for k, v in names.items()}, open(sd_ / "niche_names.json", "w"), indent=2, ensure_ascii=False)
    write_table(pd.DataFrame({"niche": list(names), "name": list(names.values())}), sd_ / "niche_names.tsv")
    write_table(ksel, sd_ / "k_selection.tsv"); write_table(pd.DataFrame(stab), sd_ / "seed_stability.tsv")
    write_table(spots, sd_ / "spot_niches.tsv")
    meta = {"K": K, "track": TRACK or "primary", "upgrade": ({k: v for k, v in upgrade.items() if k != "anchors"} if upgrade else None), "clustering": "leiden_fixed_k", "leiden_resolution": cinfo["resolution"], "centroid_agreement_ari": cinfo["centroid_agreement_ari"], "k_within": cfg["niches"]["k_within"], "frozen_seed": frozen_seed, "n_epochs": n_epochs, "unmatched_threshold": thr, "interface_niche": interface_niche, "tumor_dominant_niches": tumor_dom,
            "n_genes": len(genes), "sections": secs, "train_infos": infos}
    json.dump(meta, open(fro / "meta.json", "w"), indent=2)
    write_provenance(STAGE, [ROOT / cfg["paths"]["processed"] / "qc", ROOT / "results/07_lopo/lopo_metrics_gene_holdout.tsv"],
                     [fro / "prototypes.npy", sd_ / "spot_niches.tsv", sd_ / "k_selection.tsv"], cfg, extra=meta, t_start=t0)


if __name__ == "__main__":
    main()
