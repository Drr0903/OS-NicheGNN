"""Interface geometry and T-cell readout — shared by stages 09 (discovery), 09b/09c (comparators) and 10 (external).

Definitions (config/run.yaml → interface; the readout variants are the sensitivity analyses reported with them):
  boundary spots      tumour spots (reference label) with ≥1 non-tumour evaluable ring-1 neighbour and lung parenchyma
                      within 3 rings ("tumour–lung interface")
  covered             a boundary spot whose ring-1/2 non-tumour neighbours contain ≥1 matched interface-niche spot
  coverage C_s        fraction of boundary spots covered
  runs / continuity   boundary spots chained along connected components (ring-1/2 adjacency); covered runs are maximal
                      consecutive covered stretches; continuity = mean run length, fragmentation = n_runs / n_covered
  inner band          tumour spots within `band_rings` of a boundary spot
  outer band          non-tumour evaluable spots within `band_rings` of a boundary spot            → E (original)
  outer band, no-niche outer-band spots that are NOT interface-niche spots                          → E_noniche
  distal band         non-tumour evaluable spots between band_rings and band_rings+3 (rings 3–5)  → E_distal
  T score             mean log-normalised expression of the locked T-cell gene set (genes present in the object)
  E                   log((mean T_outer + eps) / (mean T_inner + eps)); unevaluable if both means < min_t_signal
  segments            consecutive chunks of `segment_length_spots` boundary spots along each chain; each segment gets
                      coverage, the three E variants and the mean of EVERY RCTD weight over its band spots (w_<type>),
                      plus grouped fractions (w_tumor_frac, w_fibro_frac, w_myeloid_frac) and w_T_NK
The no-niche and distal variants exist because coverage and the original outer band share spots (a circularity flagged in
review): they measure the T signal beyond the niche itself.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .spatial import ring_neighbors

CLASSES = ("tumor_cellular", "lung_parenchyma", "paucicellular")
GROUPS = {"tumor_frac": ["Osteosarcoma"], "fibro_frac": ["Fibroblast", "Pericyte_SMC"], "myeloid_frac": ["Myeloid", "Osteoclast", "Alveolar_macrophage"]}


def t_score(a, genes: list[str]) -> np.ndarray:
    g = [x for x in genes if x in a.var_names]
    X = a.layers["lognorm"]
    return np.asarray(X[:, [a.var_names.get_loc(x) for x in g]].mean(1)).ravel() if g else np.zeros(a.n_obs)


def order_chain(idx: np.ndarray, xy: np.ndarray, pitch: float) -> list[np.ndarray]:
    """Split boundary spots into connected chains (ring-1/2 adjacency) and order each by a greedy nearest-neighbour walk."""
    if len(idx) == 0:
        return []
    sub = xy[idx]
    nb = ring_neighbors(sub, pitch, rings=(1, 2))
    seen = np.zeros(len(idx), bool); chains = []
    for s in range(len(idx)):
        if seen[s]:
            continue
        comp = []; stack = [s]; seen[s] = True
        while stack:
            u = stack.pop(); comp.append(u)
            for v in nb[u]:
                if not seen[v]:
                    seen[v] = True; stack.append(v)
        comp = np.array(comp)
        deg = np.array([len(nb[u]) for u in comp]); start = comp[deg.argmin()]
        order = [start]; left = set(comp.tolist()); left.discard(start)
        while left:
            cur = order[-1]; cand = np.array(sorted(left)); d = np.linalg.norm(sub[cand] - sub[cur], axis=1)
            nxt = cand[d.argmin()]; order.append(nxt); left.discard(nxt)
        chains.append(idx[np.array(order)])
    return chains


def _E(t: np.ndarray, inner: np.ndarray, outer: np.ndarray, eps: float) -> tuple[float, float, float]:
    ti = float(t[inner].mean()) if inner.any() else np.nan
    to = float(t[outer].mean()) if outer.any() else np.nan
    e = float(np.log((to + eps) / (ti + eps))) if inner.any() and outer.any() else np.nan
    return ti, to, e


def interface_metrics(xy: np.ndarray, pitch: float, labels: np.ndarray, boundary: np.ndarray, niche: np.ndarray, interface_niche: int,
                      unmatched: np.ndarray, tscore: np.ndarray, weights: pd.DataFrame | None, cfg: dict) -> tuple[dict, pd.DataFrame]:
    ic = cfg["interface"]; eps = ic["eps"]; br = ic["band_rings"]
    is_t = labels == "tumor_cellular"; ev = np.isin(labels, CLASSES); non_t = ev & ~is_t
    nb12 = ring_neighbors(xy, pitch, rings=(1, 2))
    bidx = np.where(boundary)[0]
    if len(bidx) == 0:
        return {"n_boundary": 0, "evaluable": False}, pd.DataFrame()
    niche_ok = (niche == interface_niche) & ~unmatched
    covered = np.array([bool(niche_ok[nb12[i]][non_t[nb12[i]]].any()) if len(nb12[i]) else False for i in bidx])
    dist_b = cKDTree(xy[bidx]).query(xy)[0]
    band = dist_b <= (br + 0.45) * pitch; distal = (dist_b > (br + 0.45) * pitch) & (dist_b <= (br + 3.45) * pitch)
    inner, outer = band & is_t, band & non_t
    outer_nn, outer_d = outer & ~niche_ok, distal & non_t
    t_in, t_out, E = _E(tscore, inner, outer, eps); _, t_out_nn, E_nn = _E(tscore, inner, outer_nn, eps); _, t_out_d, E_d = _E(tscore, inner, outer_d, eps)
    evaluable_T = bool(inner.any() and outer.any() and max(t_in, t_out) >= ic["min_t_signal"])
    chains = order_chain(bidx, xy, pitch); cov_of = dict(zip(bidx.tolist(), covered.tolist()))
    runs, seg_rows, seg_id = [], [], 0
    L = ic["segment_length_spots"]
    for ch in chains:
        r = 0
        for c in (cov_of[i] for i in ch):
            if c:
                r += 1
            elif r:
                runs.append(r); r = 0
        if r:
            runs.append(r)
        for start in range(0, len(ch), L):
            seg = ch[start:start + L]
            if len(seg) < ic["min_segment_spots"]:
                continue
            ds = cKDTree(xy[seg]).query(xy)[0]
            sband = ds <= (br + 0.45) * pitch; sdistal = (ds > (br + 0.45) * pitch) & (ds <= (br + 3.45) * pitch)
            s_in, s_out = sband & is_t, sband & non_t
            if not (s_in.any() and s_out.any()):
                continue
            ti, to, e = _E(tscore, s_in, s_out, eps); _, to_nn, e_nn = _E(tscore, s_in, s_out & ~niche_ok, eps); _, to_d, e_d = _E(tscore, s_in, sdistal & non_t, eps)
            row = {"segment": seg_id, "n_spots": len(seg), "coverage": float(np.mean([cov_of[i] for i in seg])), "t_inner": ti, "t_outer": to, "E_seg": e,
                   "t_outer_noniche": to_nn, "E_seg_noniche": e_nn, "n_outer_noniche": int((s_out & ~niche_ok).sum()), "t_outer_distal": to_d, "E_seg_distal": e_d, "n_outer_distal": int((sdistal & non_t).sum()),
                   "n_inner": int(s_in.sum()), "n_outer": int(s_out.sum()), "x": float(xy[seg, 0].mean()), "y": float(xy[seg, 1].mean())}
            if weights is not None:
                wb = weights.iloc[np.where(sband & ev)[0]]
                for c in weights.columns:
                    row[f"w_{c}"] = float(wb[c].mean())
            seg_rows.append(row); seg_id += 1
    n_cov = int(covered.sum())
    metrics = {"n_boundary": int(len(bidx)), "n_chains": len(chains), "coverage": float(covered.mean()), "n_covered": n_cov, "n_runs": len(runs),
               "continuity_mean_run": float(np.mean(runs)) if runs else 0.0, "continuity_max_run": int(max(runs)) if runs else 0, "fragmentation": (len(runs) / n_cov) if n_cov else np.nan,
               "n_inner_band": int(inner.sum()), "n_outer_band": int(outer.sum()), "n_outer_noniche": int(outer_nn.sum()), "n_outer_distal": int(outer_d.sum()),
               "t_inner": t_in, "t_outer": t_out, "E_s": E, "t_outer_noniche": t_out_nn, "E_s_noniche": E_nn, "t_outer_distal": t_out_d, "E_s_distal": E_d,
               "frac_outer_band_niche": float(niche_ok[outer].mean()) if outer.any() else np.nan,
               "evaluable": bool(len(bidx) >= 30), "evaluable_T": evaluable_T, "n_segments": len(seg_rows)}
    return metrics, pd.DataFrame(seg_rows)


def matched_segment_contrast(seg: pd.DataFrame, match_cols: list[str], hi: float = 0.75, lo: float = 0.25, caliper: float = 0.5, seed: int = 0, outcome: str = "E_seg") -> dict:
    """Within one section: high- vs low-coverage segments matched 1:1 on standardised composition covariates."""
    cols = [c for c in match_cols if c in seg.columns]
    H = seg[seg["coverage"] >= hi]; Lw = seg[seg["coverage"] <= lo]
    out = {"n_high": len(H), "n_low": len(Lw), "n_matched": 0, "matched_diff_E": np.nan, "unmatched_diff_E": float(H[outcome].mean() - Lw[outcome].mean()) if len(H) and len(Lw) else np.nan}
    if len(H) == 0 or len(Lw) == 0 or not cols:
        return out
    Z = seg[cols]; mu, sd = Z.mean(), Z.std().replace(0, 1)
    zh, zl = ((H[cols] - mu) / sd).to_numpy(), ((Lw[cols] - mu) / sd).to_numpy()
    tree = cKDTree(zl); used = set(); diffs = []
    rng = np.random.default_rng(seed)
    for i in rng.permutation(len(H)):
        d, j = tree.query(zh[i], k=min(len(Lw), 5)); d, j = np.atleast_1d(d), np.atleast_1d(j)
        for dd, jj in zip(d, j):
            if jj not in used and dd <= caliper * np.sqrt(len(cols)):
                used.add(jj); diffs.append(H[outcome].iloc[i] - Lw[outcome].iloc[jj]); break
    out["n_matched"] = len(diffs); out["matched_diff_E"] = float(np.mean(diffs)) if diffs else np.nan
    return out


def restricted_permutation_p(seg: pd.DataFrame, n_perm: int, seed: int, outcome: str = "E_seg") -> dict:
    if len(seg) < 6 or seg["coverage"].nunique() < 2:
        return {"spearman_cov_E": np.nan, "perm_p": np.nan}
    from scipy.stats import spearmanr
    obs = spearmanr(seg["coverage"], seg[outcome]).correlation
    rng = np.random.default_rng(seed); cov = seg["coverage"].to_numpy(); E = seg[outcome].to_numpy()
    null = np.array([spearmanr(rng.permutation(cov), E).correlation for _ in range(n_perm)])
    return {"spearman_cov_E": float(obs), "perm_p": float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))}


def load_weights(w_dir: Path, section: str, obs_names) -> pd.DataFrame | None:
    """All RCTD weights (reindexed to the section's spots) plus grouped fractions."""
    p = w_dir / f"{section}_rctd_weights.tsv"
    if not p.exists():
        return None
    W = pd.read_csv(p, sep="\t").set_index("barcode").reindex(obs_names).fillna(0.0)
    for g, cols in GROUPS.items():
        W[g] = W[[c for c in cols if c in W.columns]].sum(1)
    return W


def run_interface_cohort(spots: pd.DataFrame, qc_dir: Path, w_dir: Path, cfg: dict, interface_niche: int, sections: list[str], patient_of: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import anndata as ad
    from .repro import log
    ic = cfg["interface"]; tgenes = cfg["genes"]["t_cell_holdout"]
    sec_rows, seg_all, con_rows = [], [], []
    for s in sections:
        sub = spots[spots["section"] == s].set_index("barcode")
        a = ad.read_h5ad(qc_dir / f"{s}.h5ad"); sub = sub.reindex(a.obs_names)
        ts = t_score(a, tgenes); W = load_weights(w_dir, s, a.obs_names)
        xy = a.obsm["spatial"].astype(float)
        pitch = float(a.uns["spot_pitch_fullres_px"] if "spot_pitch_fullres_px" in a.uns else a.uns["spot_pitch_img_px"])
        met, seg = interface_metrics(xy, pitch, sub["histology_label"].to_numpy(dtype=object), sub["tumor_lung_interface"].fillna(False).to_numpy(bool),
                                     sub["niche"].fillna(-1).to_numpy(int), interface_niche, sub["unmatched"].fillna(True).to_numpy(bool), ts, W, cfg)
        met.update({"section": s, "patient": patient_of.get(s, s), "n_t_genes_present": int(sum(g in a.var_names for g in tgenes))})
        if met["n_t_genes_present"] < 5:
            met["evaluable_T"] = False
        if len(seg):
            seg["section"] = s; seg["patient"] = patient_of.get(s, s); seg["evaluable_T"] = met["evaluable_T"]; seg_all.append(seg)
            con = matched_segment_contrast(seg, [f"w_{c}" for c in ic["match_vars"]], seed=cfg["seed"])
            con.update(restricted_permutation_p(seg, ic["n_permutations"], cfg["seed"]))
            con.update({"section": s, "patient": patient_of.get(s, s)}); con_rows.append(con)
        sec_rows.append(met)
        log(f"{s}: boundary {met['n_boundary']}, coverage {met.get('coverage', np.nan):.2f}, E_s {met.get('E_s', np.nan):.3f} (no-niche {met.get('E_s_noniche', np.nan):.3f}, distal {met.get('E_s_distal', np.nan):.3f}), segments {met.get('n_segments', 0)}")
    return pd.DataFrame(sec_rows), (pd.concat(seg_all, ignore_index=True) if seg_all else pd.DataFrame()), pd.DataFrame(con_rows)
