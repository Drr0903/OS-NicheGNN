"""Stage 08b — how much of each representation is patient identity? (diagnostic, all discovery lung sections)

For PCA, BANKSY, the frozen OS-NicheGNN and, as an exploratory variant, OS-NicheGNN trained on per-section
standardised input, embeddings of all discovery lung spots are clustered (Leiden, fixed K; ) and the
labels are compared with (i) patient identity and (ii) the region reference label by adjusted Rand index (ARI)
and normalised mutual information (NMI). A representation whose clusters track patients rather than tissue
compartments cannot yield niches shared across patients — which is what the set in advance K rule detected.
Also reported: the silhouette of patient identity in each embedding (patient-balanced subsample).
Outputs results/08_niches/patient_identity_diagnostics.tsv
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.banksy import BanksyEmbedder  # noqa: E402
from lib.cluster import leiden_fixed_k  # noqa: E402
from lib.data import build_graph, fit_scaler, holdout_genes, load_labels, load_qc_sections, select_hvgs  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.train import embed_sections, make_model, train_model  # noqa: E402

STAGE = "08_niches"


def section_standardise(G: dict) -> dict:
    """Exploratory: z-score every gene within each section (removes section-level mean/scale)."""
    import copy
    out = {}
    for s, g in G.items():
        h = copy.copy(g); x = g.x
        h.x = np.clip((x - x.mean(0)) / (x.std(0) + 1e-6), -10, 10).astype(np.float32); out[s] = h
    return out


def diagnostics(name: str, emb: dict, G: dict, K: int, seed: int, rng) -> dict:
    secs = list(G)
    Z = np.vstack([emb[s] for s in secs]); pat = np.concatenate([[G[s].patient] * G[s].n for s in secs]); lab = np.concatenate([G[s].labels for s in secs])
    km, _ = leiden_fixed_k(Z, K, seed, batch=pat, k_within=3)
    with threadpool_limits(limits=8):
        idx = np.concatenate([rng.choice(np.where(pat == p)[0], min(800, (pat == p).sum()), replace=False) for p in np.unique(pat)])
        sil = float(silhouette_score(Z[idx], pat[idx]))
    ev = np.isin(lab, ["tumor_cellular", "lung_parenchyma", "paucicellular"])
    frac = pd.crosstab(pat, km, normalize="index")
    return {"representation": name, "K": K, "ari_vs_patient": float(adjusted_rand_score(pat, km)), "nmi_vs_patient": float(normalized_mutual_info_score(pat, km)),
            "ari_vs_region": float(adjusted_rand_score(lab[ev], km[ev])), "nmi_vs_region": float(normalized_mutual_info_score(lab[ev], km[ev])),
            "silhouette_patient": sil, "clusters_present_in_half_patients": int(((frac >= 0.02).sum(0) >= len(frac) / 2).sum()),
            "mean_max_patient_share_per_cluster": float(pd.crosstab(km, pat, normalize="index").max(1).mean())}


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"])
    disc = cfg["cohorts"]["discovery"]; patient_of = disc["patient_groups"]
    secs = [s for s in disc["lung_sections"] if (ROOT / cfg["paths"]["processed"] / "qc" / f"{s}.h5ad").exists()]
    adatas = load_qc_sections(cfg, secs); labels = {s: load_labels(cfg, s) for s in secs}
    genes = open(ROOT / cfg["paths"]["processed"] / "frozen" / "genes.txt").read().split(); sc_ = np.load(ROOT / cfg["paths"]["processed"] / "frozen" / "scaler.npz")
    meta = json.load(open(ROOT / cfg["paths"]["processed"] / "frozen" / "meta.json")); K = meta["K"]
    G = {s: build_graph(adatas[s], genes, sc_["mu"], sc_["sd"], cfg, patient_of[s], labels[s]) for s in secs}
    rng = np.random.default_rng(cfg["seed"]); rows = []
    # frozen OS-NicheGNN
    emb = {s: np.load(ROOT / cfg["paths"]["processed"] / "embeddings" / "frozen" / f"{s}.npy") for s in secs}
    rows.append(diagnostics("OS-NicheGNN (frozen)", emb, G, K, cfg["seed"], rng)); log(rows[-1])
    # PCA and BANKSY fitted on all discovery sections (same genes)
    with threadpool_limits(limits=8):
        pca = PCA(cfg["baselines"]["pca_dim"], random_state=cfg["seed"]).fit(np.vstack([G[s].x for s in secs]))
        rows.append(diagnostics("PCA", {s: pca.transform(G[s].x) for s in secs}, G, K, cfg["seed"], rng)); log(rows[-1])
        b = cfg["baselines"]["banksy"]; be = BanksyEmbedder(b["lambda"], b["k_geom"], b["n_pcs"], cfg["seed"]).fit([G[s] for s in secs])
        rows.append(diagnostics("BANKSY", {s: be.transform(G[s]) for s in secs}, G, K, cfg["seed"], rng)); log(rows[-1])
    # raw standardised expression (no representation learning)
    rows.append(diagnostics("standardised HVG expression", {s: G[s].x for s in secs}, G, K, cfg["seed"], rng)); log(rows[-1])
    # exploratory: per-section standardised input
    Gs = section_standardise(G)
    cfg_final = json.loads(json.dumps(cfg)); cfg_final["train"]["max_epochs"] = meta["n_epochs"]; cfg_final["train"]["patience"] = 10**9
    model, _ = train_model("nichegnn", [Gs[s] for s in secs], [], cfg_final, cfg["seed"], device=cfg["device"])
    emb_s = embed_sections(model, [Gs[s] for s in secs], "nichegnn", cfg, cfg["device"])
    rows.append(diagnostics("OS-NicheGNN, per-section standardised input (exploratory)", emb_s, Gs, K, cfg["seed"], rng)); log(rows[-1])
    with threadpool_limits(limits=8):
        pca_s = PCA(cfg["baselines"]["pca_dim"], random_state=cfg["seed"]).fit(np.vstack([Gs[s].x for s in secs]))
        rows.append(diagnostics("PCA, per-section standardised input (exploratory)", {s: pca_s.transform(Gs[s].x) for s in secs}, Gs, K, cfg["seed"], rng)); log(rows[-1])
    sd = stage_dir(STAGE); out = pd.DataFrame(rows); write_table(out, sd / "patient_identity_diagnostics.tsv")
    log(f"\n{out.round(3).to_string()}")
    write_provenance(STAGE + "_patient_identity", [ROOT / cfg["paths"]["processed"] / "frozen" / "meta.json"], [sd / "patient_identity_diagnostics.tsv"], cfg, t_start=t0)


if __name__ == "__main__":
    main()
