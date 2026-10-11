"""Stage 18 (plan v2, ) — a signalling layer for the niches.

(a) PROGENy pathway activity and (b) CollecTRI transcription-factor activity are scored per spot with the univariate linear
model of decoupler on the log-normalised matrix (T-cell readout genes removed), averaged within patient and niche, and
compared across niches at the patient level (Friedman test, Benjamini-Hochberg). (c) Every spot of a lung section with an
evaluable interface is given a signed ring distance to the nearest tumour-lung interface spot (negative inside the tumour),
and the T-cell score and pathway activities are profiled along it. (d) The per-niche pathway means are repeated on the
Harmony-corrected atlas of stage 17, because a statement about a niche has to hold in both atlases .

Outputs (results/18_signalling/): pathway_by_patient_niche.tsv, pathway_niche_tests.tsv, tf_by_patient_niche.tsv,
tf_top_per_niche.tsv, interface_distance_profile.tsv, pathway_by_niche_harmony.tsv, meta.json.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data import ring_neighbors  # noqa: E402
from lib.interface import t_score  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "18_signalling"
warnings.filterwarnings("ignore")


def score_sections(cfg: dict, spots: pd.DataFrame, secs: list[str], net: pd.DataFrame, drop: set[str], min_n: int) -> pd.DataFrame:
    """Univariate linear model activity per spot, returned as one long table of (section, barcode, source, score)."""
    import decoupler as dc
    qc = ROOT / cfg["paths"]["processed"] / "qc"; out = []
    net = net[~net["target"].isin(drop)]
    keep = net.groupby("source")["target"].size(); net = net[net["source"].isin(keep[keep >= min_n].index)]
    for s in secs:
        a = ad.read_h5ad(qc / f"{s}.h5ad")
        a = a[:, [g for g in a.var_names if g not in drop]].copy()
        dc.mt.ulm(data=a, net=net, tmin=min_n, verbose=False)
        act = a.obsm["score_ulm"]
        d = pd.DataFrame(act.to_numpy() if hasattr(act, "to_numpy") else act, index=a.obs_names, columns=list(act.columns))
        d = d.reset_index(names="barcode").melt(id_vars="barcode", var_name="source", value_name="score"); d["section"] = s
        out.append(d)
    return pd.concat(out, ignore_index=True)


def by_patient_niche(scores: pd.DataFrame, spots: pd.DataFrame, min_spots: int) -> pd.DataFrame:
    m = spots[~spots["unmatched"].astype(bool)][["section", "barcode", "patient", "niche"]]
    d = scores.merge(m, on=["section", "barcode"], how="inner")
    n = d.groupby(["patient", "niche", "source"]).size().rename("n_spots")
    out = d.groupby(["patient", "niche", "source"])["score"].mean().to_frame("activity").join(n).reset_index()
    return out[out["n_spots"] >= min_spots]


def complete_block(w: pd.DataFrame, min_patients: int = 6, min_niches: int = 3) -> pd.DataFrame:
    """Only four patients carry all eight niches, so the Friedman test runs on the largest complete patient-by-niche block:
    the niche with the most missing patients is dropped until the complete-case block is large enough. The niches tested are recorded."""
    while w.shape[1] >= min_niches:
        cc = w.dropna()
        if len(cc) >= min_patients:
            return cc
        w = w.drop(columns=[w.isna().sum().idxmax()])
    return pd.DataFrame()


def niche_tests(bpn: pd.DataFrame, names: dict) -> pd.DataFrame:
    rows = []
    for src, d in bpn.groupby("source"):
        cc = complete_block(d.pivot_table(index="patient", columns="niche", values="activity"))
        if cc.empty:
            continue
        stat, p = stats.friedmanchisquare(*[cc[c].to_numpy() for c in cc.columns])
        rows.append({"source": src, "n_patients": len(cc), "n_niches": cc.shape[1], "niches_tested": ",".join(str(int(c)) for c in cc.columns),
                     "friedman_stat": float(stat), "p": float(p),
                     "highest_niche": int(cc.mean().idxmax()), "highest_name": names[int(cc.mean().idxmax())],
                     "lowest_niche": int(cc.mean().idxmin()), "lowest_name": names[int(cc.mean().idxmin())],
                     "range_patient_mean": float(cc.mean().max() - cc.mean().min())})
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    df["fdr"] = multipletests(df["p"], method="fdr_bh")[1]
    return df.sort_values("p")


def signed_distance(cfg: dict, spots: pd.DataFrame, secs: list[str], max_ring: int) -> pd.DataFrame:
    """Ring distance to the nearest tumour–lung interface spot, negative inside the tumour, positive on the lung side."""
    qc = ROOT / cfg["paths"]["processed"] / "qc"; rows = []
    for s in secs:
        a = ad.read_h5ad(qc / f"{s}.h5ad", backed="r"); sub = spots[spots["section"] == s].set_index("barcode").reindex(a.obs_names)
        xy = a.obsm["spatial"]; pitch = float(np.median(np.sort(np.linalg.norm(xy[:200, None] - xy[None, :200], axis=2) + np.eye(min(200, len(xy))) * 1e9, axis=1)[:, 0]))
        nb = ring_neighbors(xy, pitch, rings=(1,))
        seed = sub["tumor_lung_interface"].fillna(False).to_numpy(bool)
        if seed.sum() == 0:
            continue
        dist = np.full(len(sub), np.nan); dist[seed] = 0.0; frontier = np.where(seed)[0]
        for r in range(1, max_ring + 1):
            nxt = []
            for i in frontier:
                for j in nb[i]:
                    if np.isnan(dist[j]):
                        dist[j] = r; nxt.append(j)
            frontier = np.array(nxt, dtype=int)
            if len(frontier) == 0:
                break
        lab = sub["histology_label"].to_numpy()
        sign = np.where(lab == "tumor_cellular", -1.0, 1.0)
        rows.append(pd.DataFrame({"section": s, "barcode": sub.index, "ring": dist, "signed_ring": dist * sign}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); v2 = cfg["v2"]; sd = stage_dir(STAGE)
    N = ROOT / "results/15_niches"; spots = pd.read_csv(N / "spot_niches.tsv", sep="\t"); meta = json.load(open(N / "meta.json"))
    names = {int(k): v for k, v in meta["names"].items()}; secs = v2["sections"]; drop = set(cfg["genes"]["t_cell_holdout"])
    pr = pd.read_csv(ROOT / "data/raw/progeny_human_top500.tsv", sep="\t"); ct = pd.read_csv(ROOT / "data/raw/collectri_human.tsv", sep="\t")
    with threadpool_limits(limits=8):
        log("scoring PROGENy pathways")
        ps = score_sections(cfg, spots, secs, pr[["source", "target", "weight"]], drop, 15)
        bp = by_patient_niche(ps, spots, v2["min_spots_per_patient_niche"]); bp["name"] = bp["niche"].map(names)
        write_table(bp, sd / "pathway_by_patient_niche.tsv")
        m_ = spots[~spots["unmatched"].astype(bool)][["section", "barcode", "niche"]]  # section level, for the paired internal-state comparison
        bs = ps.merge(m_, on=["section", "barcode"]).groupby(["section", "niche", "source"])["score"].agg(activity="mean", n_spots="size").reset_index()
        write_table(bs[bs["n_spots"] >= v2["min_spots_per_patient_niche"]], sd / "pathway_by_section_niche.tsv")
        pt = niche_tests(bp, names); write_table(pt, sd / "pathway_niche_tests.tsv"); log(f"\n{pt.round(4).to_string()}")
        log("scoring CollecTRI regulons")
        ts = score_sections(cfg, spots, secs, ct[["source", "target", "weight"]], drop, 15)
        bt = by_patient_niche(ts, spots, v2["min_spots_per_patient_niche"]); bt["name"] = bt["niche"].map(names)
        write_table(bt, sd / "tf_by_patient_niche.tsv")
        tt = niche_tests(bt, names); write_table(tt, sd / "tf_niche_tests.tsv")
        top = (bt.groupby(["niche", "source"])["activity"].mean().reset_index().sort_values(["niche", "activity"], ascending=[True, False]).groupby("niche").head(5))
        top["name"] = top["niche"].map(names); write_table(top, sd / "tf_top_per_niche.tsv"); log(f"\n{top.round(2).to_string()}")
    # (c) signed distance profile on the lung sections with an evaluable interface
    man = pd.read_csv(ROOT / "results/01_manifest/sample_manifest.tsv", sep="\t", dtype=str)
    elig = man[(man["role"] == "discovery_lung") & (man["boundary_eligible"] == "True")]["section_id"].tolist()
    dist = signed_distance(cfg, spots, elig, 5)
    if len(dist):
        qc = ROOT / cfg["paths"]["processed"] / "qc"; tsc = []
        for s in elig:
            a = ad.read_h5ad(qc / f"{s}.h5ad"); tsc.append(pd.DataFrame({"section": s, "barcode": a.obs_names, "t_score": t_score(a, cfg["genes"]["t_cell_holdout"])}))
        tsc = pd.concat(tsc, ignore_index=True)
        d = dist.merge(tsc, on=["section", "barcode"]).merge(spots[["section", "barcode", "patient"]], on=["section", "barcode"])
        d = d.merge(ps[ps["source"].isin(["TGFb", "TNFa", "Hypoxia", "JAK-STAT", "VEGF"])].pivot_table(index=["section", "barcode"], columns="source", values="score").reset_index(), on=["section", "barcode"], how="left")
        d = d.dropna(subset=["signed_ring"])
        prof = d.groupby(["patient", "signed_ring"])[["t_score"] + [c for c in d.columns if c in ("TGFb", "TNFa", "Hypoxia", "JAK-STAT", "VEGF")]].mean().reset_index()
        write_table(prof, sd / "interface_distance_profile.tsv")
        log(f"\n{prof.groupby('signed_ring')[['t_score', 'TGFb']].mean().round(3).to_string()}")
    # (e) composition adjustment: a niche difference in pathway activity must not be explained by its cell-type mixture
    ALL_TYPES = ["Alveolar_macrophage", "B_plasma", "Endothelial", "Fibroblast", "Lung_epithelium", "Mast", "Myeloid", "Osteoclast", "Osteosarcoma", "Pericyte_SMC", "T_NK"]
    REF_TYPE = "Fibroblast"  # the eleven fractions sum to one, so one is omitted as the reference or the design is rank deficient with an intercept
    types = [c for c in spots.columns if c in ALL_TYPES and c != REF_TYPE]
    comp = spots[~spots["unmatched"].astype(bool)].groupby(["patient", "niche"])[types].mean().reset_index()  # ten free fractions; the eleventh is the reference
    bpc = bp.merge(comp, on=["patient", "niche"], how="left"); rows = []
    import statsmodels.formula.api as smf
    for src, d in bpc.groupby("source"):
        d = d.dropna(subset=types + ["activity"]).copy()
        if d["niche"].nunique() < 3 or d["patient"].nunique() < 5:
            continue
        d["niche"] = d["niche"].astype(int).astype(str)
        for adj, cov in [("unadjusted", ""), ("composition", " + " + " + ".join(types))]:
            try:
                m = smf.ols(f"activity ~ C(niche) + C(patient){cov}", d).fit(cov_type="cluster", cov_kwds={"groups": d["patient"]})
            except Exception:
                continue
            for k in sorted(d["niche"].unique()):
                key = f"C(niche)[T.{k}]"
                if key in m.params.index:
                    rows.append({"source": src, "niche": int(k), "name": names[int(k)], "adjustment": adj, "coef_vs_reference": float(m.params[key]),
                                 "ci95_lo": float(m.conf_int().loc[key, 0]), "ci95_hi": float(m.conf_int().loc[key, 1]), "p": float(m.pvalues[key]),
                                 "reference_niche": int(sorted(d["niche"].unique())[0]), "n_rows": len(d), "n_patients": int(d["patient"].nunique())})
    adjdf = pd.DataFrame(rows)
    if len(adjdf):
        adjdf["fdr"] = np.nan
        for adj in adjdf["adjustment"].unique():
            m_ = adjdf["adjustment"] == adj; adjdf.loc[m_, "fdr"] = multipletests(adjdf.loc[m_, "p"], method="fdr_bh")[1]
    write_table(adjdf, sd / "pathway_composition_adjusted.tsv")
    log(f"composition-adjusted contrasts: {int((adjdf['adjustment'] == 'composition').sum())} rows, {int(((adjdf['adjustment'] == 'composition') & (adjdf['fdr'] < 0.1)).sum())} at FDR < 0.1")
    # (d) the same pathway means on the Harmony atlas ( rule)
    hz = ROOT / "results/17_robustness/spot_niches_harmony.tsv"
    if hz.exists():
        sph = pd.read_csv(hz, sep="\t"); bph = by_patient_niche(ps, sph, v2["min_spots_per_patient_niche"])
        write_table(bph.groupby(["niche", "source"])["activity"].mean().reset_index(), sd / "pathway_by_niche_harmony.tsv")
    json.dump({"plan": "v2", "amendment": "A15", "n_pathways": int(ps["source"].nunique()), "n_regulons": int(ts["source"].nunique()),
               "t_cell_genes_removed": len(drop)}, open(sd / "meta.json", "w"), indent=2)
    write_provenance(STAGE, [N / "spot_niches.tsv"], [sd / "pathway_niche_tests.tsv", sd / "tf_top_per_niche.tsv"], cfg, extra={"plan": "v2", "amendment": "A15"}, t_start=t0)


if __name__ == "__main__":
    main()
