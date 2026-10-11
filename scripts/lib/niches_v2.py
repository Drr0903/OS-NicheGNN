"""Plan v2 niche helpers shared by stages 15 and 16: pseudobulk per (group, niche), patient-level marker tests,
niche–niche adjacency enrichment against a within-section permutation null, and niche naming by rule."""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .repro import ROOT
from .spatial import ring_neighbors

IMMUNE = ["T_NK", "B_plasma", "Myeloid", "Mast", "Alveolar_macrophage"]; VASC = ["Endothelial", "Pericyte_SMC"]


def common_genes(cfg: dict, secs: list[str]) -> pd.Index:
    qc = ROOT / cfg["paths"]["processed"] / "qc"; genes = None
    for s in secs:
        v = ad.read_h5ad(qc / f"{s}.h5ad", backed="r").var_names; genes = v if genes is None else genes.intersection(v)
    return pd.Index(sorted(genes))


def pseudobulk(cfg: dict, spots: pd.DataFrame, secs: list[str], genes: pd.Index, group_col: str = "patient", min_spots: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw counts summed per (group, niche) over matched spots; returns log2 CPM table (genes × (group, niche)) and spot counts."""
    qc = ROOT / cfg["paths"]["processed"] / "qc"; pb, n = {}, {}
    for s in secs:
        a = ad.read_h5ad(qc / f"{s}.h5ad")[:, genes]; sub = spots[(spots["section"] == s) & (~spots["unmatched"].astype(bool))].set_index("barcode").reindex(a.obs_names)
        C = a.layers["counts"]
        if sub["niche"].notna().sum() == 0:  # no matched spot of this section in the requested scope
            continue
        for k in sorted(sub["niche"].dropna().unique()):
            m = (sub["niche"] == k).to_numpy(); key = (sub[group_col].dropna().iloc[0], int(k)); v = np.asarray(C[m].sum(0)).ravel()
            pb[key] = pb.get(key, 0) + v; n[key] = n.get(key, 0) + int(m.sum())
        m_all = sub["niche"].notna().to_numpy(); key = (sub[group_col].dropna().iloc[0], -1); pb[key] = pb.get(key, 0) + np.asarray(C[m_all].sum(0)).ravel(); n[key] = n.get(key, 0) + int(m_all.sum())
    keep = [k for k in pb if n[k] >= min_spots]; df = pd.DataFrame({k: pb[k] for k in keep}, index=genes); df.columns = pd.MultiIndex.from_tuples(df.columns, names=[group_col, "niche"])
    cpm = df / df.sum(0) * 1e6; return np.log2(cpm + 1), pd.Series({k: n[k] for k in keep})


def niche_markers(l2: pd.DataFrame, K: int, min_groups: int = 4) -> pd.DataFrame:
    """Per niche: genes higher in the niche than in the rest of the same patient's spots (paired Wilcoxon across patients)."""
    rows = []
    for k in range(K):
        pats = [p for p in l2.columns.get_level_values(0).unique() if (p, k) in l2.columns and (p, -1) in l2.columns]
        if len(pats) < min_groups:
            continue
        A = l2.loc[:, [(p, k) for p in pats]].to_numpy(); B = l2.loc[:, [(p, -1) for p in pats]].to_numpy(); d = A - B
        expressed = (A.mean(1) > 1) | (B.mean(1) > 1); pv = np.ones(len(l2))
        for i in np.where(expressed)[0]:
            try:
                pv[i] = stats.wilcoxon(d[i]).pvalue if np.any(d[i] != 0) else 1.0
            except ValueError:
                pv[i] = 1.0
        fdr = multipletests(pv, method="fdr_bh")[1]
        out = pd.DataFrame({"niche": k, "gene": l2.index, "log2fc_mean": d.mean(1), "n_patients": len(pats), "n_patients_up": (d > 0).sum(1), "p_wilcoxon": pv, "fdr": fdr})
        rows.append(out[expressed].sort_values(["fdr", "log2fc_mean"], ascending=[True, False]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def adjacency_enrichment(spots: pd.DataFrame, G: dict, K: int, n_perm: int, seed: int) -> pd.DataFrame:
    """Ring-1 contacts between niches per section vs a within-section label permutation null → log2(obs/expected) and z."""
    rng = np.random.default_rng(seed); rows = []
    for s, g in G.items():
        sub = spots[spots["section"] == s].set_index("barcode").reindex(g.barcodes); lab = sub["niche"].fillna(-1).to_numpy(int); lab[sub["unmatched"].fillna(True).to_numpy(bool)] = -1
        nb = ring_neighbors(g.xy, g.pitch, rings=(1,)); pairs = np.array([(i, j) for i, v in enumerate(nb) for j in v if j > i]); ok = (lab[pairs[:, 0]] >= 0) & (lab[pairs[:, 1]] >= 0); pairs = pairs[ok]
        if len(pairs) < 50:
            continue
        def counts(l):
            M = np.zeros((K, K)); a, b = l[pairs[:, 0]], l[pairs[:, 1]]; np.add.at(M, (a, b), 1); np.add.at(M, (b, a), 1); return M
        obs = counts(lab); valid = lab >= 0; null = np.zeros((n_perm, K, K))
        for r in range(n_perm):
            lp = lab.copy(); lp[valid] = rng.permutation(lab[valid]); null[r] = counts(lp)
        mu, sd = null.mean(0), null.std(0) + 1e-9
        for a in range(K):
            for b in range(a, K):
                if obs[a, b] + mu[a, b] > 0:
                    rows.append({"section": s, "patient": g.patient, "niche_a": a, "niche_b": b, "observed": obs[a, b], "expected": mu[a, b], "log2_ratio": np.log2((obs[a, b] + 1) / (mu[a, b] + 1)), "z": (obs[a, b] - mu[a, b]) / sd[a, b]})
    return pd.DataFrame(rows)


def name_niches(comp: pd.DataFrame, hist: pd.DataFrame, enr: pd.DataFrame, interface_niche: int) -> dict[int, str]:
    """Descriptive names by relative enrichment (A11, revised A13).

    Partners are the biological axes of the eleven-type reference, with endothelium + pericyte pooled as "vascular" and
    T/NK + B/plasma as "lymphoid" (splitting them dilutes one axis across two types). Niches are grouped into tumour-dominant
    (>= 60 % tumour-labelled), lung parenchyma (>= 50 % lung-labelled) and the rest; within a group of three or more niches each
    partner weight is z-scored across that group, otherwise across all niches. A partner names the niche only if its weight
    reaches MIN_W and its z reaches 1; partners within 0.25 z of the leader are resolved in favour of the larger absolute weight.
    Names describe composition, not mechanism."""
    MIN_W = 0.05  # a naming partner must reach 5 % mean RCTD weight; without this floor a near-zero type can win the z-score
    AXES = {"fibroblast": ["Fibroblast"], "osteoclast": ["Osteoclast"], "vascular": ["Endothelial", "Pericyte_SMC"], "myeloid": ["Myeloid"],
            "alveolar-macrophage": ["Alveolar_macrophage"], "lymphoid": ["T_NK", "B_plasma"], "mast": ["Mast"], "epithelial": ["Lung_epithelium"]}
    W = pd.DataFrame({a: comp[[c for c in cs if c in comp.columns]].sum(1) for a, cs in AXES.items()})
    tf = hist["tumor_cellular"] if "tumor_cellular" in hist.columns else pd.Series(0.0, index=hist.index)
    lf = hist["lung_parenchyma"] if "lung_parenchyma" in hist.columns else pd.Series(0.0, index=hist.index)
    tum = [k for k in comp.index if k != interface_niche and tf.get(k, 0) >= 0.6]
    lung = [k for k in comp.index if k != interface_niche and tf.get(k, 0) < 0.6 and lf.get(k, 0) >= 0.5]
    names, used = {}, {}
    for k in comp.index:
        if k == interface_niche:
            names[int(k)] = "peritumoural stroma (boundary)"; continue
        grp, members = ("tumour", tum) if k in tum else (("lung parenchyma", lung) if k in lung else ("stroma", [x for x in comp.index if x != interface_niche and x not in tum and x not in lung]))
        ref = members if len(members) >= 3 else list(comp.index)
        z = (W.loc[k] - W.loc[ref].mean()) / W.loc[ref].std(ddof=0).replace(0, np.nan)
        cand = z[[a for a in z.index if W.loc[k, a] >= MIN_W]].dropna()
        cand = cand[cand >= 1]
        if len(cand):
            top = cand.max(); near = [a for a in cand.index if top - cand[a] <= 0.25]
            best = max(near, key=lambda a: float(W.loc[k, a]))
            base = f"{best}-enriched {grp}"
        else:
            base = f"{grp}, mixed stroma" if grp == "tumour" else grp
        used[base] = used.get(base, 0) + 1; names[int(k)] = base
    counts = pd.Series(list(names.values())).value_counts(); run = {}
    for k, v in names.items():
        if counts[v] > 1:
            run[v] = run.get(v, 0) + 1; names[k] = f"{v} {run[v]}"
    return names
