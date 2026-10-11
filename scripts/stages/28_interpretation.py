"""Stage 28 () — interpretation of the atlas encoder and of the niches.

(1) attention: last-layer attention of every spot over its ring-1–3 neighbours, against each ring's / same-label share
(2) parts of the latent: norm of the learned part fuse([own, context]) and of the linear neighbourhood skip; niche
    re-assignment when the attention context is removed
(3) gene attribution: gradient × input of each niche's direction (niche centroid − mean latent) on the genes
(4) niche re-assignment when the linear neighbourhood skip is removed, the neighbours are rewired, or the H&E morphology
    or the expression block of the profile is set to zero
(5) source-atlas programmes: mean z-scored ICA component weight (authors' curated components) per niche and class
Outputs results/28_interpretation/{attention, latent_parts, ablation_reassignment, gene_attribution, author_programmes}.tsv
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import interpret as I  # noqa: E402
from lib.arch import NicheEncoder  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.v2 import load_cohort, make_graphs  # noqa: E402

STAGE = "28_interpretation"; torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))


def load_encoder(cfg: dict, fro: Path, seed: int, in_dim: int) -> NicheEncoder:
    ac = {**cfg["arch"], **(cfg["atlas"].get("arch_overrides") or {})}; nt = tuple(t for t, w in (ac.get("niche_targets") or {}).items() if w > 0)
    m = NicheEncoder(in_dim, cfg["atlas"]["context"], ac["d"], ac["z_dim"], ac["heads"], ac["n_layers"], ac["dropout"], nt, ac.get("linear_heads", False), ac.get("prop_skip", False), ac.get("grad_skip", False))
    from lib.arch_ext import attach
    m = attach(m, ac, in_dim); m.load_state_dict(torch.load(fro / f"model_seed{seed}.pt", map_location="cpu")); return m.to(cfg["device"]).eval()


def author_programmes(cfg: dict, spots: pd.DataFrame) -> pd.DataFrame:
    ann = pd.read_excel(ROOT / "data/refs/PMC13470005/suppl/41467_2026_74603_MOESM3_ESM.xlsx", header=1); ann = ann[ann["Use"].astype(str) == "True"].set_index("IC")
    E = pd.read_csv(ROOT / cfg["paths"]["raw_gse293065"] / ".." / "GSE293065_ica.cell.embeddings.all.csv.gz", index_col=0)[list(ann.index)]
    E.index = [i.split("_", 1)[0] + "|" + i.split("_", 1)[1] for i in E.index]; Z = (E - E.mean()) / E.std()
    sp = spots[~spots["unmatched"].astype(bool)]; key = sp["section"] + "|" + sp["barcode"]; Zs = Z.reindex(key.to_numpy()); Zs["niche"] = sp["niche"].to_numpy()
    m = Zs.groupby("niche").mean().T; m["type"] = ann.loc[m.index, "Type"].to_numpy(); m["broad"] = ann.loc[m.index, "broad"].to_numpy(); m["annotation"] = ann.loc[m.index, "Annotation"].astype(str).str[:60].to_numpy()
    return m.reset_index(names="IC")


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE); dev = cfg["device"]; seed = cfg["seed"]
    fro = ROOT / cfg["paths"]["processed"] / cfg["atlas"]["frozen_dir"]; pcs = joblib.load(fro / "pca.joblib"); dim = cfg["remedies"]["pca_dim"]
    adatas, labels, patient_of, _ = load_cohort(cfg); secs = list(adatas)
    G, genes, _, _ = make_graphs(cfg, adatas, labels, patient_of, secs, secs)
    assert genes == open(fro / "genes.txt").read().split(), "gene list differs from the delivered encoder"
    F = {s: np.load(ROOT / cfg["paths"]["processed"] / "image_features" / f"{s}.npy") for s in secs}
    Gm = {s: dataclasses.replace(G[s], x=np.hstack([pcs["expr_pca"].transform(G[s].x) / pcs["expr_sd"], pcs["img_pca"].transform(F[s]) / pcs["img_sd"]]).astype(np.float32)) for s in secs}
    model = load_encoder(cfg, fro, seed, Gm[secs[0]].x.shape[1])
    spots = pd.read_csv(ROOT / "results/15_niches/spot_niches.tsv", sep="\t"); names = json.load(open(ROOT / "results/15_niches/niche_names.json"))
    sp = {s: spots[spots["section"] == s].set_index("barcode").reindex(G[s].barcodes) for s in secs}
    niche = {s: np.where(sp[s]["unmatched"].fillna(True).astype(bool), -1, sp[s]["niche"].fillna(-1)).astype(int) for s in secs}
    Z = {s: I.latent(model, Gm[s], dev) for s in secs}
    allZ = np.vstack([Z[s] for s in secs]); alln = np.concatenate([niche[s] for s in secs]); K = int(alln.max()) + 1
    C = np.vstack([allZ[alln == k].mean(0) for k in range(K)]); mu = allZ.mean(0)
    # (1) attention
    att = []
    for s in secs:
        W, idx, ring = I.attention(model, Gm[s], dev)
        att += [{"section": s, **r} for r in I.attention_summary(W, idx, ring, np.asarray(G[s].labels).astype(str), np.asarray(G[s].boundary, bool), niche[s])]
    A = pd.DataFrame(att); A["n_w"] = A["n_spots"]
    att_s = A.groupby(["what", "ring", "scope"]).apply(lambda d: pd.Series({"n_spots": d["n_spots"].sum(), "attention": np.average(d["attention"], weights=d["n_w"]),
                                                                           "share_of_neighbours": np.average(d["share_of_neighbours"], weights=d["n_w"])}), include_groups=False).reset_index()
    att_s["enrichment"] = att_s["attention"] / att_s["share_of_neighbours"]; write_table(att_s, sd / "attention.tsv"); log(f"\n{att_s[att_s['scope'].isin(['all', 'tumour boundary'])].round(3).to_string()}")
    # (2) parts of the latent and (4) morphology
    parts, abl = [], []
    for s in secs:
        P2 = I.latent(model, Gm[s], dev, parts=True); parts.append(pd.DataFrame({"niche": niche[s], "learned_norm": np.linalg.norm(P2[0], axis=1), "skip_norm": np.linalg.norm(P2[1], axis=1)}))
        x_noimg = Gm[s].x.copy(); x_noimg[:, dim:] = 0.0; x_noexpr = Gm[s].x.copy(); x_noexpr[:, :dim] = 0.0
        for what, Z1 in [("attention context removed", I.latent_without_context(model, Gm[s], dev)), ("neighbourhood skip removed", I.latent_without_skip(model, Gm[s], dev)),
                         ("neighbours rewired", I.latent_rewired(model, Gm[s], dev, seed + 1000)), ("H&E morphology removed", I.latent(model, Gm[s], dev, x=x_noimg)),
                         ("expression removed", I.latent(model, Gm[s], dev, x=x_noexpr))]:
            ch = I.reassign_changed(Z[s], Z1, C, niche[s]); abl += [{"section": s, "perturbation": what, "niche": int(k), "fraction_changed": float(v), "n": int((niche[s] == k).sum())} for k, v in ch.items() if k >= 0]
    P = pd.concat(parts); P = P[P["niche"] >= 0]; P["skip_share"] = P["skip_norm"] / (P["skip_norm"] + P["learned_norm"])
    write_table(P.groupby("niche")[["learned_norm", "skip_norm", "skip_share"]].mean().reset_index().assign(name=lambda d: d["niche"].astype(str).map(names)), sd / "latent_parts.tsv")
    B = pd.DataFrame(abl); B = B.groupby(["perturbation", "niche"]).apply(lambda d: np.average(d["fraction_changed"], weights=d["n"]), include_groups=False).rename("fraction_changed").reset_index()
    B["name"] = B["niche"].astype(str).map(names); write_table(B, sd / "ablation_reassignment.tsv"); log(f"\n{B.pivot(index='niche', columns='perturbation', values='fraction_changed').round(3).to_string()}")
    # (3) gene attribution (expression PCs → genes; standardised expression as input)
    back = pcs["expr_pca"].components_ / pcs["expr_sd"]; rows = []
    for k in range(K):
        d = torch.tensor(C[k] - mu, device=dev, dtype=torch.float32); acc = np.zeros(len(genes)); n = 0
        for s in secs:
            r = np.where(niche[s] == k)[0]
            if len(r) < 20:
                continue
            r = r[np.random.default_rng(seed).choice(len(r), min(len(r), 600), replace=False)]
            acc += I.gene_attribution(model, Gm[s], dev, d, r, back, G[s].x) * len(r); n += len(r)
        if n:
            a = acc / n; top = np.argsort(-a)[:30]
            rows += [{"niche": k, "name": names[str(k)], "rank": i + 1, "gene": genes[j], "attribution": float(a[j])} for i, j in enumerate(top)]
    write_table(pd.DataFrame(rows), sd / "gene_attribution.tsv"); log(pd.DataFrame(rows).groupby("niche")["gene"].apply(lambda g: ", ".join(g[:10])).to_string())
    # (5) source-atlas programmes
    ap = author_programmes(cfg, spots); write_table(ap, sd / "author_programmes.tsv")
    cls = ap.groupby("type")[[k for k in range(K) if k in ap.columns]].mean().round(2); log(f"\nauthor programme class × niche (mean z)\n{cls.to_string()}")
    out = [sd / f for f in ("attention.tsv", "latent_parts.tsv", "ablation_reassignment.tsv", "gene_attribution.tsv", "author_programmes.tsv")]
    write_provenance(STAGE, [fro / f"model_seed{seed}.pt", ROOT / "results/15_niches/spot_niches.tsv"], out, cfg, t_start=t0)


if __name__ == "__main__":
    main()
