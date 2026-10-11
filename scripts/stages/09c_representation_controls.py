"""Stage 09c — the identical niche → interface → model pipeline on other representations.

For each representation R in {PCA-50, BANKSY-20, OS-NicheGNN seeds 2–5} the frozen-model steps of stage 08 are
replayed on the discovery lung sections: embeddings → Leiden with the fixed K (A5) on a patient-balanced sample →
prototype assignment + 97.5 % unmatched rule → interface niche by the set in advance rule (non-tumour-dominant niche
with the largest odds ratio of lying within one ring of the tumour boundary) → stage 09 coverage / E readouts →
pooled within-section models (lib.inference). If the association is representation-agnostic, it is a property of
the tissue, not of the GNN; if only the GNN shows it, the GNN matters; if it vanishes with a different seed, it is
fragile. Outputs results/09_interface/representation_controls.tsv (+ per-representation pooled model tables).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.banksy import BanksyEmbedder  # noqa: E402
from lib.cluster import cluster_prototypes  # noqa: E402
from lib.data import build_graph, load_labels, load_qc_sections  # noqa: E402
from lib.inference import pooled_models  # noqa: E402
from lib.interface import run_interface_cohort  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.spatial import ring_neighbors  # noqa: E402
from lib.train import embed_sections, make_model  # noqa: E402

STAGE = "09_interface"
K_SENSITIVITY = (5, 7)  # the fixed K is 6 (A5); neighbours of the fixed K are the sensitivity


def niche_pipeline(emb: dict, G: dict, secs: list[str], patient_of: dict, K: int, cfg: dict, rng) -> tuple[pd.DataFrame, int, dict]:
    """k-means prototypes on a patient-balanced sample → spot table with niche / unmatched → interface niche by rule."""
    Z, P = [], []
    for pat in sorted(set(patient_of[s] for s in secs)):
        zs = np.vstack([emb[s] for s in secs if patient_of[s] == pat]); idx = rng.choice(len(zs), min(1500, len(zs)), replace=False); Z.append(zs[idx]); P += [pat] * len(idx)
    Z = np.vstack(Z)
    _, protos, _ = cluster_prototypes(Z, K, cfg["seed"], batch=np.array(P), k_within=cfg["niches"]["k_within"])
    rows = []
    for s in secs:
        z = emb[s]; d = np.linalg.norm(z[:, None, :] - protos[None], axis=2); g = G[s]
        rows.append(pd.DataFrame({"section": s, "patient": patient_of[s], "barcode": g.barcodes, "niche": d.argmin(1), "dist_to_prototype": d.min(1),
                                  "histology_label": g.labels, "tumor_boundary": g.boundary, "tumor_lung_interface": g.interface}))
    spots = pd.concat(rows, ignore_index=True)
    spots["unmatched"] = spots["dist_to_prototype"] > np.quantile(spots["dist_to_prototype"], cfg["niches"]["unmatched_quantile"])
    near_b = np.zeros(len(spots), bool); off = 0
    for s in secs:
        g = G[s]; nb = ring_neighbors(g.xy, g.pitch, rings=(1,)); b = g.boundary
        near_b[off:off + g.n] = np.array([b[i] or (b[nb[i]].any() if len(nb[i]) else False) for i in range(g.n)]); off += g.n
    spots["near_boundary"] = near_b
    enr = []
    for k in range(K):
        m = spots["niche"] == k
        a_, b_ = (m & spots["near_boundary"]).sum(), (~m & spots["near_boundary"]).sum(); c_, d_ = (m & ~spots["near_boundary"]).sum(), (~m & ~spots["near_boundary"]).sum()
        enr.append({"niche": k, "n_spots": int(m.sum()), "odds_ratio": (a_ * d_) / max(b_ * c_, 1), "tumor_frac": float((m & (spots["histology_label"] == "tumor_cellular")).sum() / max(m.sum(), 1)),
                    "n_patients_present": int((pd.crosstab(spots["patient"], spots["niche"], normalize="index").get(k, pd.Series(dtype=float)) >= 0.02).sum())})
    enr = pd.DataFrame(enr); cand = enr[enr["tumor_frac"] < 0.6]
    inter = int(cand.sort_values("odds_ratio", ascending=False)["niche"].iloc[0]) if len(cand) else int(enr.sort_values("odds_ratio", ascending=False)["niche"].iloc[0])
    info = enr.set_index("niche").loc[inter].to_dict(); info["interface_niche"] = inter
    return spots, inter, info


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"])
    disc = cfg["cohorts"]["discovery"]; patient_of = disc["patient_groups"]; proc = ROOT / cfg["paths"]["processed"]
    secs = [s for s in disc["lung_sections"] if (proc / "qc" / f"{s}.h5ad").exists()]
    man = pd.read_csv(ROOT / "results/01_manifest/sample_manifest.tsv", sep="\t", dtype=str)
    elig = man[(man["role"] == "discovery_lung") & (man["boundary_eligible"] == "True")]["section_id"].tolist()
    adatas = load_qc_sections(cfg, secs); labels = {s: load_labels(cfg, s) for s in secs}
    genes = open(proc / "frozen" / "genes.txt").read().split(); sc_ = np.load(proc / "frozen" / "scaler.npz"); meta = json.load(open(proc / "frozen" / "meta.json")); K = meta["K"]
    G = {s: build_graph(adatas[s], genes, sc_["mu"], sc_["sd"], cfg, patient_of[s], labels[s]) for s in secs}
    reps = {}
    with threadpool_limits(limits=8):
        pca = PCA(cfg["baselines"]["pca_dim"], random_state=cfg["seed"]).fit(np.vstack([G[s].x for s in secs])); reps["PCA-50"] = {s: pca.transform(G[s].x) for s in secs}
        b = cfg["baselines"]["banksy"]; be = BanksyEmbedder(b["lambda"], b["k_geom"], b["n_pcs"], cfg["seed"]).fit([G[s] for s in secs]); reps["BANKSY-20"] = {s: be.transform(G[s]) for s in secs}
    for seed in [cfg["seed"] + i for i in range(0, cfg["n_seeds"])]:
        model = make_model("nichegnn", len(genes), cfg).to(cfg["device"]); model.load_state_dict(torch.load(proc / "frozen" / f"model_seed{seed}.pt", map_location=cfg["device"]))
        emb = embed_sections(model, [G[s] for s in secs], "nichegnn", cfg, cfg["device"])
        if seed == cfg["seed"]:  # frozen seed: K sensitivity — the selected K was at the edge of the 4–8 grid
            for k_alt in K_SENSITIVITY:
                reps[f"OS-NicheGNN seed 1, K = {k_alt}"] = (emb, k_alt)
        else:
            reps[f"OS-NicheGNN seed {seed - cfg['seed'] + 1}"] = emb
    sd = stage_dir(STAGE); rows = []
    ref = pd.read_csv(sd / "pooled_models.tsv", sep="\t"); ref["representation"] = "OS-NicheGNN seed 1 (frozen)"; rows.append(ref)
    ref_seg = pd.read_csv(sd / "segments.tsv", sep="\t")
    infos = []
    for name, emb in reps.items():
        rng = np.random.default_rng(cfg["seed"]); k_use = K
        if isinstance(emb, tuple):
            emb, k_use = emb
        spots, inter, info = niche_pipeline(emb, G, secs, patient_of, k_use, cfg, rng)
        info["K"] = k_use
        sec, seg, con = run_interface_cohort(spots, proc / "qc", ROOT / "results/05_deconvolution/weights", cfg, inter, elig, patient_of)
        pm = pooled_models(seg, cluster="patient", n_perm=cfg["interface"]["n_permutations"], seed=cfg["seed"]) if len(seg) else pd.DataFrame()
        pm["representation"] = name.replace(", K = ", " K"); rows.append(pm)
        mg = seg.merge(ref_seg[["section", "segment", "coverage"]], on=["section", "segment"], suffixes=("", "_frozen"))
        info.update({"representation": name.replace(", K = ", " K"), "coverage_mean": float(sec["coverage"].mean()), "n_sections_with_variable_coverage": int(((sec["coverage"] > 0) & (sec["coverage"] < 1)).sum()),
                     "coverage_pearson_r_vs_frozen": float(np.corrcoef(mg["coverage"], mg["coverage_frozen"])[0, 1]) if len(mg) > 2 else np.nan,
                     "frac_segments_same_coverage_class": float(((mg["coverage"] >= 0.75) == (mg["coverage_frozen"] >= 0.75)).mean()) if len(mg) else np.nan}); infos.append(info)
        tag = name.replace(" ", "_").replace("(", "").replace(")", "")
        write_table(sec, sd / f"repctl_{tag}_section_metrics.tsv"); write_table(seg, sd / f"repctl_{tag}_segments.tsv")
        log(f"{name}: interface niche {inter} (OR {info['odds_ratio']:.2f}, tumour frac {info['tumor_frac']:.2f}, present in {info['n_patients_present']} patients), coverage mean {info['coverage_mean']:.2f}")
    out = pd.concat(rows, ignore_index=True); write_table(out, sd / "representation_controls.tsv"); write_table(pd.DataFrame(infos), sd / "representation_controls_niches.tsv")
    log(f"\n{out[out['covariate_set'] == 'full'][['representation', 'outcome', 'n_segments', 'coef', 'ci95_lo', 'ci95_hi', 'p_t_gminus1', 'p_perm_restricted']].round(3).to_string()}")
    write_provenance(STAGE + "_representation_controls", [proc / "frozen" / "meta.json"], [sd / "representation_controls.tsv"], cfg, t_start=t0)


if __name__ == "__main__":
    main()
