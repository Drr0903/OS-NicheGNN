"""Stage 27b () — clustering against pathologist annotations in two independent cancer cohorts.

Parts (run separately, merged with --merge):
  joint    every method fitted on all sections of the cohort (no label used); fixed-K clustering of all sections together
           (Leiden, k-means; K = number of ground-truth classes in the cohort) and of each section alone (K = classes present
           in the section, ≥ 10 spots); ARI vs patient of the joint Leiden partition
  stagate  STAGATE (package defaults, 1,000 epochs) per section, per-section clustering only
  lopo     leave-one-patient-out: representation and prototypes from the training patients, held-out spots transferred
Methods: PCA-50, PCA-50 + H&E, propagated PCA + H&E, BANKSY-20 (paper defaults), OS-NicheGNN latent z with the atlas
configuration (config atlas.arch_overrides), three seeds for the learned models.
Usage: python 27b_benchmark.py --cohort crc --part joint ; --part lopo --patients A121573 --tag _crc_A121573 ; --merge
Outputs results/27_benchmark/{section_scores, summary, paired}.tsv
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import benchmark as B  # noqa: E402
from lib.arch_eval import paired  # noqa: E402
from lib.arch_train import embed_arch, train_arch  # noqa: E402
from lib.banksy import BanksyEmbedder  # noqa: E402
from lib.context_probe import propagate  # noqa: E402
from lib.generality import make_graphs, qc_and_lognorm  # noqa: E402,F401
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.v2 import compute_anchors  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

STAGE = "27_benchmark"; ROOT_B = (ROOT / "data/benchmark")
NT = int(os.environ.get("OMP_NUM_THREADS", 8)); torch.set_num_threads(NT)


def atlas_cfg(cfg: dict) -> dict:
    c = json.loads(json.dumps(cfg)); c["arch"].update(cfg["atlas"].get("arch_overrides") or {}); return c


def reps(cfg, G, F, fit_secs, all_secs, P, seeds):
    """{method: [(seed, {section: embedding})]} with every fitted component estimated on `fit_secs` only."""
    dim = cfg["remedies"]["pca_dim"]; Gm = B.profiles(G, F, fit_secs, dim, cfg["seed"]); out = {}
    if cfg.get("model_only"):
        return {**model_reps(cfg, Gm, fit_secs, all_secs, P, seeds)}
    out["PCA-50"] = [(cfg["seed"], {s: Gm[s].x[:, :dim] for s in all_secs})]
    out["PCA-50 + H&E"] = [(cfg["seed"], {s: Gm[s].x for s in all_secs})]
    out["propagated PCA + H&E"] = [(cfg["seed"], {s: propagate(Gm[s]) for s in all_secs})]
    b = cfg["baselines"]["banksy"]; be = BanksyEmbedder(b["lambda"], b["k_geom"], b["n_pcs"], cfg["seed"]).fit([G[s] for s in fit_secs])
    out["BANKSY-20"] = [(cfg["seed"], {s: be.transform(G[s]) for s in all_secs})]
    return out if cfg.get("baselines_only") else {**out, **model_reps(cfg, Gm, fit_secs, all_secs, P, seeds)}


def model_reps(cfg, Gm, fit_secs, all_secs, P, seeds):
    out = {}; name = "OS-NicheGNN" + cfg.get("model_tag", "")
    pats = sorted({P[s] for s in fit_secs}); rng = np.random.default_rng(cfg["seed"]); vp = pats[int(rng.integers(len(pats)))]
    val = [s for s in fit_secs if P[s] == vp]; fit = [s for s in fit_secs if P[s] != vp]
    anchors = compute_anchors(cfg, [Gm[s] for s in fit]); out[name] = []
    for seed in seeds:
        model, info = train_arch(cfg["atlas"]["context"], [Gm[s] for s in fit], [Gm[s] for s in val], cfg, seed, cfg["device"], anchors, cfg["v2"]["model"]["lambda_anchor"])
        E = embed_arch(model, [Gm[s] for s in all_secs], cfg["device"]); out[name].append((seed, {s: E[s]["z"] for s in all_secs}))
        log(f"  {name} seed {seed}: {info['epochs_run']} epochs")
    return out


def with_smoothing(out: dict, G: dict) -> dict:
    """A21 addendum 7: every representation also in a smoothed form — the mean over the spot and its ring-1 neighbours —
    as niche-level clustering input (the same operation for every method)."""
    from lib.spatial import neighbor_mean, ring_neighbors
    nb = {s: [np.append(v, i) for i, v in enumerate(ring_neighbors(G[s].xy, G[s].pitch, rings=(1,)))] for s in G}
    sm = {f"{m} +smooth": [(seed, {s: np.asarray(neighbor_mean(e, nb[s]), dtype=np.float32) for s, e in emb.items()}) for seed, emb in runs] for m, runs in out.items()}
    return {**out, **sm}


def truth_of(G, s):
    return np.asarray(G[s].labels).astype(str)


def k_cohort(G) -> int:
    t = np.concatenate([truth_of(G, s) for s in G]); return int(len(set(t) - {B.UNSCORED}))


def k_section(G, s) -> int:
    v = pd.Series(truth_of(G, s)); v = v[v != B.UNSCORED].value_counts(); return int((v >= 10).sum())


def part_joint(cfg, cohort, G, F, P, seeds):
    secs = sorted(G); K = k_cohort(G); rows = []
    for meth, runs in with_smoothing(reps(cfg, G, F, secs, secs, P, seeds), G).items():
        for seed, emb in runs:
            Z = np.vstack([emb[s] for s in secs]); sec_of = np.concatenate([[s] * len(emb[s]) for s in secs])
            pats = np.array([P[s] for s in sec_of])
            for how in ("leiden", "kmeans"):
                for prep in ("raw", "pc30", "pc30_harmony"):
                    with threadpool_limits(limits=NT):
                        lab = B.cluster(Z, K, seed, how, prep, pats)
                    ari_p = float(adjusted_rand_score(pats, lab)); cl = how if prep == "raw" else f"{how}_{prep}"
                    for s in secs:
                        rows.append({"cohort": cohort, "part": "joint", "clustering": cl, "method": meth, "seed": seed, "section": s, "patient": P[s], "K": K,
                                     **B.scores(lab[sec_of == s], truth_of(G, s)), "ari_vs_patient": ari_p})
                for s in secs:
                    rows += per_section_rows(cohort, meth, seed, s, emb[s], G, P, how)
            log(f"  {cohort} {meth} s{seed}: joint Leiden ARI {np.mean([r['ari'] for r in rows if r['method'] == meth and r['seed'] == seed and r['part'] == 'joint' and r['clustering'] == 'leiden']):.3f}")
    return rows


def per_section_rows(cohort, meth, seed, s, e, G, P, how) -> list[dict]:
    """Per-section clustering with K = classes present: raw and 30-PC input; for k-means also the Gaussian mixture on 30 PCs
    (the spatial-domain standard) with and without ring-1 majority refinement."""
    ks = k_section(G, s); out = []; base = {"cohort": cohort, "part": "per_section", "method": meth, "seed": seed, "section": s, "patient": P[s], "K": ks}
    with threadpool_limits(limits=NT):
        for prep in ("raw", "pc30"):
            out.append({**base, "clustering": how if prep == "raw" else f"{how}_{prep}", **B.scores(B.cluster(e, ks, seed, how, prep), truth_of(G, s))})
        if how == "kmeans":
            lab = B.cluster(e, ks, seed, "gmm", "pc30"); out.append({**base, "clustering": "gmm_pc30", **B.scores(lab, truth_of(G, s))})
            out.append({**base, "clustering": "gmm_pc30_refined", **B.scores(B.refine(lab, G[s].xy, G[s].pitch), truth_of(G, s))})
    return out


def hvg_matrix(a, n_top: int = 3000) -> np.ndarray:
    import scanpy as sc
    b = a.copy(); b.X = b.layers["counts"].copy()
    sc.pp.highly_variable_genes(b, flavor="seurat_v3", n_top_genes=n_top); b = b[:, b.var["highly_variable"]].copy()
    sc.pp.normalize_total(b, target_sum=1e4); sc.pp.log1p(b); sc.pp.scale(b, zero_center=False, max_value=10)
    X = b.X.toarray() if hasattr(b.X, "toarray") else np.asarray(b.X)
    return X.astype(np.float32)


@torch.no_grad()
def readouts(model, z: np.ndarray, g, device: str) -> dict[str, np.ndarray]:
    """Decoded profile of every spot and its mean over the spot and its ring-1 neighbours."""
    from lib.spatial import neighbor_mean, ring_neighbors
    model.eval(); xr = model.dec(torch.tensor(z, device=device)).cpu().numpy().astype(np.float32)
    nb = [np.append(v, i) for i, v in enumerate(ring_neighbors(g.xy, g.pitch, rings=(1,)))]
    return {" recon": xr, " recon ring1": np.asarray(neighbor_mean(xr, nb), dtype=np.float32)}


def part_section_model(cfg, cohort, A, G, F, P, seeds, patients):
    """A23 single-section mode of OS-NicheGNN: the encoder is trained on one section at a time (profiles fitted on that
    section, full-section batches, a fixed number of epochs, no early stopping), as GraphST and STAGATE are, and that
    section is clustered."""
    c = json.loads(json.dumps(cfg)); c["arch"].update(max_epochs=c["arch"].get("section_epochs", 600), patience=10 ** 9, centre_batch=8192)
    rows, name = [], "OS-NicheGNN (single section)" + cfg.get("model_tag", "")
    for s in sorted(G):
        if patients and P[s] not in patients:
            continue
        if cfg["arch"].get("section_input") == "hvg3000":  # GraphST-style input: the section's own 3,000 HVGs, scaled, no PCA
            Gm = {s: dataclasses.replace(G[s], x=hvg_matrix(A[s]))}
        elif cfg["arch"].get("section_input") == "hvg3000_img":  # + 50 H&E principal components, block scaled to the expression block's total variance
            X = hvg_matrix(A[s]); M = PCA(50, random_state=cfg["seed"]).fit_transform(F[s]); M = M * np.sqrt(X.var(0).sum() / M.var(0).sum())
            Gm = {s: dataclasses.replace(G[s], x=np.hstack([X, M]).astype(np.float32))}
        else:
            Gm = B.profiles({s: G[s]}, {s: F[s]}, [s], cfg["remedies"]["pca_dim"], cfg["seed"])
        Z = []
        for seed in seeds:
            model, info = train_arch(cfg["atlas"]["context"], [Gm[s]], [], c, seed, cfg["device"])
            e = embed_arch(model, [Gm[s]], cfg["device"])[s]["z"]; Z.append((e - e.mean(0)) / (e.std(0) + 1e-6))
            for how in ("leiden", "kmeans"):
                rows += per_section_rows(cohort, name, seed, s, e, G, P, how)
            if cfg["arch"].get("section_readouts"):  # the decoded (denoised) profile x̂ = Dec(z), alone and averaged over ring 1
                for meth, xr in readouts(model, e, G[s], cfg["device"]).items():
                    for how in ("leiden", "kmeans"):
                        rows += per_section_rows(cohort, name + meth, seed, s, xr, G, P, how)
        if cfg["arch"].get("section_ensemble"):  # seed ensemble: the standardized latents of all seeds, concatenated
            for how in ("leiden", "kmeans"):
                rows += per_section_rows(cohort, name + " ensemble", seeds[0], s, np.hstack(Z).astype(np.float32), G, P, how)
        log(f"  {name} {s}: refined GMM ARI {np.mean([r['ari'] for r in rows if r['section'] == s and r['clustering'] == 'gmm_pc30_refined']):.3f}")
    return rows


def part_graphst(cfg, cohort, A, G, P, seeds):
    """GraphST (Long et al. 2023; package defaults: 3,000 HVGs of its own preprocessing, 600 epochs), one section at a time."""
    import anndata as ad
    from GraphST import GraphST
    rows = []
    for s in sorted(G):
        a0 = A[s]; a = ad.AnnData(X=a0.layers["counts"].copy(), obs=a0.obs[[]].copy(), var=a0.var[[]].copy()); a.obsm["spatial"] = a0.obsm["spatial"]
        for seed in seeds:
            m = GraphST.GraphST(a.copy(), device=torch.device(cfg["device"]), random_seed=seed); out = m.train(); e = np.asarray(out.obsm["emb"], dtype=np.float32)
            for meth, runs in with_smoothing({"GraphST": [(seed, {s: e})]}, {s: G[s]}).items():
                for how in ("leiden", "kmeans"):
                    rows += per_section_rows(cohort, meth, seed, s, runs[0][1][s], G, P, how)
        log(f"  GraphST {s}: ARI {np.mean([r['ari'] for r in rows if r['section'] == s and r['clustering'] == 'gmm_pc30_refined']):.3f}")
    return rows


def part_stagate(cfg, cohort, G, P, seeds):
    from lib.stagate_wrap import stagate_embed
    c = json.loads(json.dumps(cfg)); c["baselines"]["stagate"]["epochs"] = 1000; rows = []  # package / paper default
    for s in sorted(G):
        ks = k_section(G, s)
        for seed in seeds:
            e0 = stagate_embed(G[s], c, seed, cfg["device"])
            for meth, runs in with_smoothing({"STAGATE": [(seed, {s: e0})]}, {s: G[s]}).items():
                for how in ("leiden", "kmeans"):
                    rows += per_section_rows(cohort, meth, seed, s, runs[0][1][s], G, P, how)
        log(f"  STAGATE {s}: ARI {np.mean([r['ari'] for r in rows if r['section'] == s]):.3f}")
    return rows


def part_lopo(cfg, cohort, G, F, P, seeds, patients):
    secs = sorted(G); K = k_cohort(G); rows = []
    for hp in patients:
        test = [s for s in secs if P[s] == hp]; train = [s for s in secs if P[s] != hp]
        for meth, runs in with_smoothing(reps(cfg, G, F, train, secs, P, seeds), G).items():
            for seed, emb in runs:
                for how in ("leiden", "kmeans"):
                    for prep in ("raw", "pc30"):
                        with threadpool_limits(limits=NT):
                            lab = B.transfer(emb, train, test, K, seed, how, prep=prep)
                        rows += [{"cohort": cohort, "part": "lopo", "clustering": how if prep == "raw" else f"{how}_{prep}", "method": meth, "seed": seed, "section": s,
                                  "patient": hp, "K": K, **B.scores(lab[s], truth_of(G, s))} for s in test]
        log(f"  lopo {cohort} {hp}: " + "; ".join(f"{m} {np.mean([r['ari'] for r in rows if r['patient'] == hp and r['method'] == m and r['clustering'] == 'leiden']):.3f}" for m in sorted({r['method'] for r in rows})))
    return rows


def merge(cfg, t0):
    sd = stage_dir(STAGE); df = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(sd.glob("section_scores_*.tsv"))], ignore_index=True)
    ss = cfg.get("single_section", {}); R = f"OS-NicheGNN (single section){ss.get('model_tag', '')}R"  # the same mode, run with the decoded readouts
    df = df[~df["method"].str.startswith(R) | ((df["cohort"] == "os_seurat") & (df["method"] == R))]  # readout selection rows (DLPFC donor 1) and decoded readouts stay in their files
    df["method"] = df["method"].replace({R: R[:-1]})
    pat = df.groupby(["cohort", "part", "clustering", "method", "patient"])[["ari", "nmi"]].mean().reset_index()  # seeds and sections averaged per patient
    summ = pat.groupby(["cohort", "part", "clustering", "method"])[["ari", "nmi"]].agg(["mean", "std", "count"]); summ.columns = ["_".join(c) for c in summ.columns]
    ap = df[df["part"] == "joint"].groupby(["cohort", "clustering", "method"])["ari_vs_patient"].mean()
    summ = summ.reset_index().merge(ap.reset_index(), on=["cohort", "clustering", "method"], how="left"); rows = []
    ss = cfg.get("single_section", {}); sel = [f"OS-NicheGNN (single section){ss[k]}" for k in ("model_tag", "image_tag") if k in ss]
    pp = pat.assign(part=pat["part"].replace({"section_model": "per_section"}))  # the single-section mode is scored with the per-section baselines
    pp = pp[~pp["method"].str.startswith("OS-NicheGNN (single section)") | pp["method"].isin(sel)]
    for (c, p, cl), g in pp.groupby(["cohort", "part", "clustering"]):
        for name in ["OS-NicheGNN"] + sel:
            ours = g[g["method"] == name].set_index("patient")
            if ours.empty:
                continue
            for m in sorted(set(g["method"]) - {"OS-NicheGNN"} - set(sel)):
                ref = g[g["method"] == m].set_index("patient")
                for metric in ("ari", "nmi"):
                    ix = ours.index.intersection(ref.index)
                    rows.append({"cohort": c, "part": p, "clustering": cl, "comparison": f"{name} − {m}", "metric": metric, **paired(ours.loc[ix, metric], ref.loc[ix, metric], cfg["seed"])})
    out = [sd / "section_scores.tsv", sd / "summary.tsv", sd / "paired.tsv"]
    write_table(df, out[0]); write_table(summ, out[1]); write_table(pd.DataFrame(rows), out[2]); log(f"\n{summ.round(3).to_string()}")
    write_provenance(STAGE, [], out, cfg, extra={"amendment": "A21"}, t_start=t0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cohort", default="crc"); ap.add_argument("--part", default="joint"); ap.add_argument("--patients", default="")
    ap.add_argument("--tag", default=""); ap.add_argument("--merge", action="store_true")
    ap.add_argument("--model-only", action="store_true", help="only the learned model (candidate configurations)")
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--arch", default="", help="JSON overrides of cfg.arch, e.g. '{\"linear_heads\": true}'")
    ap.add_argument("--lambda-anchor", type=float, default=None); ap.add_argument("--model-tag", default="")
    ap.add_argument("--context", default="", help="encoder context instead of the atlas one (ringmlp, ringxattn)"); a = ap.parse_args()
    t0 = time.time(); cfg = atlas_cfg(load_config()); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE)
    if a.context:
        cfg["atlas"]["context"] = a.context
    cfg["model_only"] = a.model_only; cfg["baselines_only"] = a.baselines_only; cfg["model_tag"] = a.model_tag; cfg["arch"].update(json.loads(a.arch) if a.arch else {})
    cfg["v2"]["model"]["lambda_anchor"] = a.lambda_anchor if a.lambda_anchor is not None else cfg["atlas"].get("lambda_anchor", cfg["v2"]["model"]["lambda_anchor"])
    if a.merge:
        merge(cfg, t0); return
    A, F, P = B.load(ROOT_B / a.cohort)
    with threadpool_limits(limits=NT):
        G, genes = make_graphs(A, P, cfg)
    seeds = [cfg["seed"] + i for i in range(3)]
    log(f"{a.cohort}: {len(G)} sections, {len(set(P.values()))} patients, K = {k_cohort(G)}, {len(genes)} genes")
    if a.part == "joint":
        rows = part_joint(cfg, a.cohort, G, F, P, seeds)
    elif a.part == "stagate":
        rows = part_stagate(cfg, a.cohort, G, P, seeds)
    elif a.part == "section_model":
        rows = part_section_model(cfg, a.cohort, A, G, F, P, seeds, a.patients.split(",") if a.patients else [])
    elif a.part == "graphst":
        rows = part_graphst(cfg, a.cohort, A, G, P, seeds)
    else:
        rows = part_lopo(cfg, a.cohort, G, F, P, seeds, a.patients.split(",") if a.patients else sorted(set(P.values())))
    f = sd / f"section_scores_{a.cohort}_{a.part}{a.tag}.tsv"; write_table(pd.DataFrame(rows), f)
    write_provenance(STAGE + f"_{a.cohort}_{a.part}{a.tag}", [], [f], cfg, extra={"amendment": "A21", "arch": cfg["arch"]}, t_start=t0)


if __name__ == "__main__":
    main()
