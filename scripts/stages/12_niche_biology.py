"""Stage 12 — patient-level pseudobulk characterisation of the niches (discovery cohort).

For each (patient, niche) the raw counts of all matched spots are summed (pseudobulk; sections of one patient
pooled), converted to log2-CPM. Interface niche vs tumour-dominant niche and vs the lung-associated niche are
compared with a paired Wilcoxon test across patients (unit = patient) plus mean log2 fold change; genes with
FDR < 0.05 and |log2FC| > 1 are reported. Hallmark gene-set enrichment = hypergeometric test on the
up-regulated gene list against the gene universe. Ligand–receptor candidates: for curated pairs, the
co-occurrence of ligand expression in interface-niche spots and receptor expression in ring-1-adjacent tumour
spots is compared with a within-section spot-label permutation null (spatial-adjacency-specific, exploratory).
Outputs results/12_biology/{pseudobulk_log2cpm.tsv, de_interface_vs_tumor.tsv, de_interface_vs_lung.tsv,
        hallmark_enrichment.tsv, lr_adjacency.tsv}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import frozen_name, results_path, ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.spatial import ring_neighbors  # noqa: E402

STAGE = "12_biology"
LR_PAIRS = [("CXCL12", "CXCR4"), ("CCL2", "CCR2"), ("CSF1", "CSF1R"), ("SPP1", "CD44"), ("TGFB1", "TGFBR2"), ("VEGFA", "KDR"), ("PDGFB", "PDGFRB"),
            ("IL6", "IL6R"), ("CXCL8", "CXCR2"), ("CCL5", "CCR5"), ("TNFSF11", "TNFRSF11A"), ("MIF", "CD74"), ("APOE", "TREM2"), ("LGALS9", "HAVCR2"),
            ("CD274", "PDCD1"), ("HLA-E", "KLRC1"), ("FN1", "ITGA5"), ("COL1A1", "ITGB1"), ("THBS1", "CD47"), ("SEMA3A", "NRP1"), ("GAS6", "AXL"), ("ANGPT2", "TEK")]


def pseudobulk(cfg, spots, secs, tumor_niches, lung_niches):
    qc = ROOT / cfg["paths"]["processed"] / "qc"
    genes = None
    for s in secs:  # genes common to every section (sections differ slightly in probe-set annotation)
        v = ad.read_h5ad(qc / f"{s}.h5ad", backed="r").var_names
        genes = v if genes is None else genes.intersection(v)
    genes = pd.Index(sorted(genes))
    pb = {}
    for s in secs:
        a = ad.read_h5ad(qc / f"{s}.h5ad")[:, genes]
        sub = spots[(spots["section"] == s) & (~spots["unmatched"])].set_index("barcode").reindex(a.obs_names)
        C = a.layers["counts"]
        for k in sorted(sub["niche"].dropna().unique()):
            m = (sub["niche"] == k).to_numpy()
            if m.sum() < 20:
                continue
            key = (sub["patient"].iloc[0], int(k)); v = np.asarray(C[m].sum(0)).ravel()
            pb[key] = pb.get(key, 0) + v
    # pooled pseudo-niches per patient: -1 = all tumour-dominant niches, -2 = all lung-associated niches
    # (tumour niches are largely patient-specific, so pooling is the only way to obtain a paired contrast)
    for pat in {k[0] for k in pb}:
        for code, members in ((-1, tumor_niches), (-2, lung_niches)):
            vs = [pb[(pat, k)] for k in members if (pat, k) in pb]
            if vs:
                pb[(pat, code)] = np.sum(vs, axis=0)
    df = pd.DataFrame(pb, index=genes); df.columns = pd.MultiIndex.from_tuples(df.columns, names=["patient", "niche"])
    cpm = df / df.sum(0) * 1e6
    return np.log2(cpm + 1)


def paired_de(l2: pd.DataFrame, a: int, b: int) -> pd.DataFrame:
    pats = [p for p in l2.columns.get_level_values(0).unique() if (p, a) in l2.columns and (p, b) in l2.columns]
    if len(pats) < 4:
        return pd.DataFrame()
    A = l2[[(p, a) for p in pats]].to_numpy(); B = l2[[(p, b) for p in pats]].to_numpy()
    expressed = (A.mean(1) > 1) | (B.mean(1) > 1)
    diff = A - B
    pv = np.array([stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0 for d in diff])
    out = pd.DataFrame({"gene": l2.index, "log2fc_mean": diff.mean(1), "n_patients": len(pats), "n_positive": (diff > 0).sum(1), "p_wilcoxon": pv})
    out = out[expressed].copy(); out["fdr"] = multipletests(out["p_wilcoxon"], method="fdr_bh")[1]
    return out.sort_values("p_wilcoxon")


def hallmark(genes_up: list[str], universe: list[str], gmt: Path) -> pd.DataFrame:
    rows = []; U = set(universe); up = set(genes_up) & U
    for line in open(gmt):
        parts = line.strip().split("\t"); name, gs = parts[0], set(parts[2:]) & U
        if len(gs) < 5:
            continue
        k = len(up & gs)
        p = stats.hypergeom.sf(k - 1, len(U), len(gs), len(up)) if k else 1.0
        rows.append({"gene_set": name, "set_size_in_universe": len(gs), "overlap": k, "p": p, "genes": ",".join(sorted(up & gs))})
    df = pd.DataFrame(rows)
    if len(df):
        df["fdr"] = multipletests(df["p"], method="fdr_bh")[1]
    return df.sort_values("p")


def lr_adjacency(cfg, spots, secs, interface_niche: int, tumor_niches: list[int], n_perm: int = 500) -> pd.DataFrame:
    qc = ROOT / cfg["paths"]["processed"] / "qc"; rng = np.random.default_rng(cfg["seed"]); rows = []
    for L, R in LR_PAIRS:
        obs_all, null_all = [], []
        for s in secs:
            a = ad.read_h5ad(qc / f"{s}.h5ad")
            if L not in a.var_names or R not in a.var_names:
                continue
            sub = spots[spots["section"] == s].set_index("barcode").reindex(a.obs_names)
            niche = sub["niche"].fillna(-1).to_numpy(int)
            xy = a.obsm["spatial"]; pitch = float(a.uns["spot_pitch_fullres_px"] if "spot_pitch_fullres_px" in a.uns else a.uns["spot_pitch_img_px"])
            nb = ring_neighbors(xy, pitch, rings=(1,))
            lig = np.asarray(a[:, L].layers["lognorm"].todense()).ravel(); rec = np.asarray(a[:, R].layers["lognorm"].todense()).ravel()
            def score(nv):
                iface = np.where(nv == interface_niche)[0]
                vals = [lig[i] * rec[nb[i]][np.isin(nv[nb[i]], tumor_niches)].mean() for i in iface if len(nb[i]) and np.isin(nv[nb[i]], tumor_niches).any()]
                return float(np.mean(vals)) if vals else np.nan
            o = score(niche)
            if np.isnan(o):
                continue
            nulls = [score(rng.permutation(niche)) for _ in range(n_perm)]
            obs_all.append(o); null_all.append(np.nanmean(nulls))
        if len(obs_all) >= 4:
            d = np.array(obs_all) - np.array(null_all)
            rows.append({"ligand": L, "receptor": R, "n_sections": len(obs_all), "mean_obs": float(np.mean(obs_all)), "mean_null": float(np.mean(null_all)),
                         "log2_ratio": float(np.log2((np.mean(obs_all) + 1e-3) / (np.mean(null_all) + 1e-3))), "n_sections_positive": int((d > 0).sum()), "p_wilcoxon_sections": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0})
    return pd.DataFrame(rows).sort_values("p_wilcoxon_sections") if rows else pd.DataFrame()


def main() -> None:
    t0 = time.time()
    cfg = load_config(); set_global_seed(cfg["seed"])
    meta = json.load(open(ROOT / cfg["paths"]["processed"] / frozen_name() / "meta.json"))
    spots = pd.read_csv(results_path("08_niches/spot_niches.tsv"), sep="\t")
    secs = meta["sections"]; inter = meta["interface_niche"]; tum = meta["tumor_dominant_niches"]
    hist = pd.read_csv(results_path("08_niches/niche_histology.tsv"), sep="\t").set_index("niche")
    lung_niches = [int(k) for k in hist.index if hist.loc[k, "lung_parenchyma"] >= 0.5 and int(k) != inter]
    sd = stage_dir(STAGE)
    l2 = pseudobulk(cfg, spots, secs, tum, lung_niches); l2.to_csv(sd / "pseudobulk_log2cpm.tsv", sep="\t")
    de_t = paired_de(l2, inter, -1); write_table(de_t, sd / "de_interface_vs_tumor.tsv")
    de_l = paired_de(l2, inter, -2); write_table(de_l, sd / "de_interface_vs_lung.tsv")
    log(f"paired pseudobulk: interface vs pooled tumour niches {tum} and vs pooled lung niches {lung_niches}")
    gmt = ROOT / "data/raw/msigdb/h.all.v2023.2.Hs.symbols.gmt"
    if len(de_t) and gmt.exists():
        up = de_t[(de_t["fdr"] < 0.05) & (de_t["log2fc_mean"] > 1)]["gene"].tolist()
        hm = hallmark(up, l2.index.tolist(), gmt); write_table(hm, sd / "hallmark_enrichment_interface_vs_tumor.tsv")
        log(f"interface vs tumour: {len(up)} up genes; top hallmark:\n{hm.head(8)[['gene_set', 'overlap', 'p', 'fdr']].to_string()}")
    lr = lr_adjacency(cfg, spots, secs, inter, tum); write_table(lr, sd / "lr_adjacency.tsv")
    log(f"LR adjacency candidates:\n{lr.head(10).round(3).to_string() if len(lr) else 'none'}")
    write_provenance(STAGE, [results_path("08_niches/spot_niches.tsv")], [sd / "de_interface_vs_tumor.tsv", sd / "lr_adjacency.tsv"], cfg,
                     extra={"interface_niche": inter, "tumor_niches": tum, "lung_niches": lung_niches}, t_start=t0)


if __name__ == "__main__":
    main()
