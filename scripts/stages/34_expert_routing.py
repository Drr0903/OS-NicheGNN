"""Stage 34 — what the mixture-of-experts fusion layer routes: per-spot expert weights of the fused encoder and their tissue context.

The fused encoder (ring-mean branch h over [profile, ring-1 mean, ring-2–3 mean]; attention branch c over the ring-1–3
neighbours; fusion layer = E MLP experts mixed per spot by a softmax router reading [h; c]) is trained on all atlas sections
with the atlas protocol of stage 25c (`--variant moe2|moe4`, one run per seed). Per spot it records
  gate_k        router weight of expert k
  attn_rel_k    reliance of expert k on the attention branch: ‖e_k([h; c]) − e_k([h; c̄])‖ / ‖e_k([h; c])‖, c̄ = section mean of c
  gate_shift    total-variation change of the router weights when c is replaced by c̄
  hetero        neighbourhood heterogeneity: mean per-component dispersion over the ring-1–3 neighbours (input space)
`--summarise` joins these with the atlas annotation (niche, image-derived region label, boundary flags, RCTD fractions) and the
signed ring distance to the tumour–lung interface, matches experts across seeds by their gate correlation, and writes
patient-level summaries: expert weight per niche and region, Spearman correlation of each expert's weight with heterogeneity
and cell-type fractions per section (summarised over patients), the weight profile across the interface, and routing by
metastatic site and between primary and relapsed lesions (matched pairs, overall and within each niche; MR10 in time order).

Usage: python scripts/stages/34_expert_routing.py --variant frozen          (the delivered atlas encoder, all seeds)
       python scripts/stages/34_expert_routing.py --variant moe2 --seed-index 0
       python scripts/stages/34_expert_routing.py --summarise --variant moe2
Outputs data/processed/expert_routing/<variant>_s<seed>.parquet; results/34_expert_routing/<variant>_*.tsv
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.arch import niche_target_values  # noqa: E402
from lib.arch_train import prep, train_arch  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.v2 import load_cohort  # noqa: E402

STAGE = "34_expert_routing"; torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
CELLS = ["Alveolar_macrophage", "B_plasma", "Endothelial", "Fibroblast", "Lung_epithelium", "Mast", "Myeloid", "Osteoclast", "Osteosarcoma", "Pericyte_SMC", "T_NK"]


def _load(name, file):
    sp = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file)); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def spot_dir(cfg) -> Path:
    d = ROOT / cfg["paths"]["processed"] / "expert_routing"; d.mkdir(parents=True, exist_ok=True); return d


@torch.no_grad()
def route(model, g, device: str, chunk: int = 4096) -> pd.DataFrame:
    """Router weights, attention reliance of each expert and router shift without the attention context, per spot."""
    model.eval(); s = prep(g, device, max_ring=getattr(model, "max_ring", 3)); fz = model.fuse; H, C = [], []
    for ch in torch.arange(s.x.shape[0], device=device).split(chunk):
        h, c = model.parts_rows(s.x, s.nb, ch); H.append(h); C.append(c)
    H, C = torch.cat(H), torch.cat(C); cbar = C.mean(0, keepdim=True).expand_as(C)
    u, ub = torch.cat([H, C], 1), torch.cat([H, cbar], 1)
    g1, g0 = torch.softmax(fz.router(u), 1), torch.softmax(fz.router(ub), 1)
    out = {f"gate_{k}": g1[:, k].cpu().numpy() for k in range(g1.shape[1])}
    for k, e in enumerate(fz.experts):
        a, b = e(u), e(ub); out[f"attn_rel_{k}"] = ((a - b).norm(dim=1) / a.norm(dim=1).clamp_min(1e-8)).cpu().numpy()
    out["gate_shift"] = (0.5 * (g1 - g0).abs().sum(1)).cpu().numpy()
    out["hetero"] = niche_target_values(s.x, s.nb)["disp"].mean(1).cpu().numpy()
    return pd.DataFrame({"section": g.section, "barcode": np.asarray(g.barcodes), **out})


def run_seed(cfg, variant: str, k: int) -> None:
    s25c = _load("s25c", "25c_reference_mapping.py"); s25b = s25c.s25b; s25c.VAR = variant; c = s25c.mlp_cfg(cfg)
    c["arch"]["max_epochs"] = cfg["atlas"]["frozen_epochs"]; c["arch"]["patience"] = 10**9; seed = cfg["seed"] + k
    adatas, labels, patient_of, _ = load_cohort(cfg); secs = list(adatas)
    _, Gp = s25b.profile_graphs(cfg, adatas, labels, patient_of, secs, secs)
    model, info = train_arch(c["atlas"]["context"], [Gp[s] for s in secs], [], c, seed, cfg["device"], None, 0.0)
    df = pd.concat([route(model, Gp[s], cfg["device"]) for s in secs], ignore_index=True); df["seed"] = seed
    f = spot_dir(cfg) / f"{variant}_s{seed}.parquet"; df.to_parquet(f)
    g = df.filter(regex=r"^gate_\d+$")
    log(f"{variant} seed {seed}: {info['n_params']} parameters, {info['epochs_run']} epochs; mean gates {g.mean().round(3).tolist()}; "
        f"spots with max gate > 0.9: {(g.max(1) > 0.9).mean():.3f}")


def run_frozen(cfg) -> None:
    """Routing of the delivered atlas encoder: every seed of data/processed/<frozen_dir>, profiles as in stage 28."""
    s28 = _load("s28", "28_interpretation.py"); import dataclasses, joblib
    from lib.v2 import make_graphs
    V = os.environ.get("PROJECT018_ATLAS_VARIANT", ""); fro = ROOT / cfg["paths"]["processed"] / (f"frozen_{V}" if V else cfg["atlas"]["frozen_dir"])
    pcs = joblib.load(fro / "pca.joblib")
    adatas, labels, patient_of, _ = load_cohort(cfg); secs = list(adatas); G, genes, _, _ = make_graphs(cfg, adatas, labels, patient_of, secs, secs)
    assert genes == open(fro / "genes.txt").read().split(), "gene list differs from the delivered encoder"
    F = {s: np.load(ROOT / cfg["paths"]["processed"] / "image_features" / f"{s}.npy") for s in secs}
    Gm = {s: dataclasses.replace(G[s], x=np.hstack([pcs["expr_pca"].transform(G[s].x) / pcs["expr_sd"], pcs["img_pca"].transform(F[s]) / pcs["img_sd"]]).astype(np.float32)) for s in secs}
    for f in sorted(fro.glob("model_seed*.pt")):
        seed = int(f.stem.replace("model_seed", "")); model = s28.load_encoder(cfg, fro, seed, Gm[secs[0]].x.shape[1])
        df = pd.concat([route(model, Gm[s], cfg["device"]) for s in secs], ignore_index=True); df["seed"] = seed
        df.to_parquet(spot_dir(cfg) / f"frozen{V}_s{seed}.parquet"); g = df.filter(regex=r"^gate_\d+$")
        log(f"delivered encoder seed {seed}: mean gates {g.mean().round(3).tolist()}; spots with max gate > 0.9: {(g.max(1) > 0.9).mean():.3f}")


def align(runs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Relabel the experts of every seed to those of the first by the Hungarian match of their gate correlations."""
    ref = runs[0]; K = ref.filter(regex=r"^gate_\d+$").shape[1]; out = [ref]
    for r in runs[1:]:
        M = np.array([[np.corrcoef(ref[f"gate_{i}"], r[f"gate_{j}"])[0, 1] for j in range(K)] for i in range(K)])
        _, col = linear_sum_assignment(-M); ren = {}
        for i, j in enumerate(col):
            ren[f"gate_{j}"] = f"gate_{i}"; ren[f"attn_rel_{j}"] = f"attn_rel_{i}"
        out.append(r.rename(columns=ren)); log(f"  seed {int(r.seed.iloc[0])}: expert match r = {np.round(M[np.arange(K), col], 3).tolist()}")
    return out


def remodelling(cfg, D: pd.DataFrame, gates: list[str], rel: list[str], out: dict) -> None:
    """Routing by metastatic site and between primary and relapsed lesions: section-level means (all spots and within each
    niche); site and lesion type compared over patients (Mann–Whitney), the matched primary–relapse pairs as paired differences
    (and the fifth, soft-tissue pair flagged as sensitivity), and the two relapses of MR10 in time order."""
    from scipy.stats import mannwhitneyu
    v2 = cfg["v2"]; pairs = v2["paired"]; sens = v2.get("paired_sensitivity", {}) or {}
    D = D.copy(); D["lesion"] = np.where(D.section.str.startswith("PT"), "primary", "relapse")
    D["entropy"] = -(D[gates].clip(1e-9) * np.log(D[gates].clip(1e-9))).sum(1); val = gates + rel + ["gate_shift", "entropy"]
    sec = D.groupby(["section", "patient", "site", "lesion"])[val].mean().reset_index(); out["by_section"] = sec
    secn = D.groupby(["section", "patient", "site", "lesion", "niche_name"])[val].mean().reset_index(); out["by_section_niche"] = secn
    pt = sec.groupby(["patient", "site", "lesion"])[val].mean().reset_index(); rows = []
    for v in val:  # primary against relapse, and lung against non-lung relapse, one value per patient and lesion type
        for lab, a, b in [("relapse vs primary", pt[pt.lesion == "relapse"][v], pt[pt.lesion == "primary"][v]),
                          ("lung vs non-lung relapse", pt[(pt.lesion == "relapse") & (pt.site == "lung")][v], pt[(pt.lesion == "relapse") & (pt.site != "lung")][v])]:
            rows.append({"contrast": lab, "weight": v, "mean_a": a.mean(), "mean_b": b.mean(), "n_a": len(a), "n_b": len(b),
                         "p_mannwhitney": mannwhitneyu(a, b).pvalue if len(a) > 1 and len(b) > 1 else np.nan})
    out["site_lesion_tests"] = pd.DataFrame(rows); out["by_site"] = pt.groupby(["site", "lesion"])[val].mean().reset_index()
    rows = []
    for p_, m in {**pairs, **sens}.items():  # matched pairs: overall and within each niche present in both lesions
        a, b = sec.set_index("section").loc[p_], sec.set_index("section").loc[m]
        rows.append({"pair": f"{p_}→{m}", "patient": a.patient, "niche_name": "all spots", "sensitivity_pair": p_ in sens, **{f"delta_{v}": b[v] - a[v] for v in val}})
        na, nb = secn[secn.section == p_].set_index("niche_name"), secn[secn.section == m].set_index("niche_name")
        for n in na.index.intersection(nb.index):
            rows.append({"pair": f"{p_}→{m}", "patient": a.patient, "niche_name": n, "sensitivity_pair": p_ in sens, **{f"delta_{v}": nb.loc[n, v] - na.loc[n, v] for v in val}})
    out["paired_delta"] = pd.DataFrame(rows)
    out["longitudinal_MR10"] = sec[sec.section.isin(v2["longitudinal"])].sort_values("section")
    out["lesion_models"] = lesion_models(cfg, D, ["gate_0", "attn_rel_1"])


def lesion_models(cfg, D: pd.DataFrame, weights: list[str]) -> pd.DataFrame:
    """Primary against relapse with patients as the unit of independence: (i) the matched pairs, (ii) the patients who contribute
    only one kind of lesion (independent groups), (iii) one row per section, OLS of the weight on relapse with cluster-robust
    standard errors by patient and P from t with G − 1 degrees of freedom, unadjusted and adjusted for lung site, the
    tumour-labelled and lung-labelled fractions and the main deconvolved fractions."""
    import statsmodels.formula.api as smf
    from scipy.stats import mannwhitneyu, t as tdist
    v2 = cfg["v2"]; pairs = dict(v2["paired"]); rows = []
    D = D.copy(); D["relapse"] = (~D.section.str.startswith("PT")).astype(int); D["lung_site"] = (D["site"] == "lung").astype(int)
    D["tumour_frac"] = (D["histology_label"] == "tumor_cellular").astype(float); D["lung_frac"] = (D["histology_label"] == "lung_parenchyma").astype(float)
    cov = ["tumour_frac", "lung_frac", "Osteosarcoma", "Fibroblast", "Endothelial", "Myeloid"]
    S = D.groupby(["section", "patient"])[weights + ["relapse", "lung_site"] + cov].mean().reset_index()
    kinds = S.groupby("patient")["relapse"].agg(lambda x: set(x)); only_p = [p for p, k in kinds.items() if k == {0}]; only_r = [p for p, k in kinds.items() if k == {1}]
    for w in weights:
        dl = [S.set_index("section").loc[m, w] - S.set_index("section").loc[p_, w] for p_, m in pairs.items()]
        rows.append({"weight": w, "analysis": "matched pairs", "n": len(dl), "estimate": float(np.mean(dl)), "n_up": int(sum(x > 0 for x in dl)), "p": float("nan")})
        pt = S.groupby(["patient", "relapse"])[w].mean().reset_index(); a = pt[pt.patient.isin(only_r)][w]; b = pt[pt.patient.isin(only_p)][w]
        rows.append({"weight": w, "analysis": "patients with one kind of lesion", "n": f"{len(b)} primary, {len(a)} relapse", "estimate": float(a.mean() - b.mean()),
                     "mean_primary": float(b.mean()), "mean_relapse": float(a.mean()), "p": float(mannwhitneyu(a, b).pvalue)})
        G = S["patient"].nunique()
        for label, covs in [("all sections, unadjusted", []), ("all sections, adjusted for site and composition", ["lung_site"] + cov)]:
            m = smf.ols(f"{w} ~ relapse" + "".join(f" + {c}" for c in covs), S).fit(cov_type="cluster", cov_kwds={"groups": S["patient"]})
            b_, se = m.params["relapse"], m.bse["relapse"]; q = tdist.ppf(0.975, G - 1)
            rows.append({"weight": w, "analysis": label, "n": f"{len(S)} sections, {G} patients", "estimate": float(b_), "ci95_lo": float(b_ - q * se), "ci95_hi": float(b_ + q * se),
                         "p": float(2 * tdist.sf(abs(b_ / se), G - 1))})
    return pd.DataFrame(rows)


def summarise(cfg, variant: str, t0: float) -> None:
    sd = stage_dir(STAGE); files = sorted(spot_dir(cfg).glob(f"{variant}_s*.parquet"))
    runs = [pd.read_parquet(f).set_index(["section", "barcode"]).sort_index().reset_index() for f in files]; runs = align(runs)
    K = runs[0].filter(regex=r"^gate_\d+$").shape[1]; val = [f"gate_{k}" for k in range(K)] + [f"attn_rel_{k}" for k in range(K)] + ["gate_shift", "hetero"]
    R = pd.concat(runs).groupby(["section", "barcode"])[val].mean().reset_index()  # mean over seeds
    V = os.environ.get("PROJECT018_ATLAS_VARIANT", ""); N = ROOT / cfg["paths"]["processed"] / f"atlas_{V}" / "15_niches" if V else ROOT / "results/15_niches"
    sp = pd.read_csv(N / "spot_niches.tsv", sep="\t"); D = R.merge(sp, on=["section", "barcode"], how="inner")
    s18 = _load("s18", "18_niche_signalling.py"); lung = sorted(D.loc[D.tumor_lung_interface.astype(bool), "section"].unique())
    dist = s18.signed_distance(cfg, sp, lung, 5); D = D.merge(dist, on=["section", "barcode"], how="left")
    gates = [f"gate_{k}" for k in range(K)]; rel = [f"attn_rel_{k}" for k in range(K)]
    names = {str(k): v for k, v in pd.read_json(N / "niche_names.json", typ="series").items()}
    D["niche_name"] = D["niche"].astype(str).map(names)
    if K == 2:  # orient the experts by the region labels: expert 0 = the one weighted more on lung-parenchyma spots (host tissue)
        lp = D[D["histology_label"] == "lung_parenchyma"]
        if len(lp) and lp["gate_1"].mean() > lp["gate_0"].mean():
            D = D.rename(columns={"gate_0": "gate_1", "gate_1": "gate_0", "attn_rel_0": "attn_rel_1", "attn_rel_1": "attn_rel_0"})
        log("expert 0 = host-tissue expert (higher weight on lung-parenchyma spots), expert 1 = tumour expert")
    out = {}
    # (a) expert weight per niche and per image-derived region: patient-level means, then the mean over patients
    for key in ["niche_name", "histology_label", "near_boundary"]:
        pt = D.groupby(["patient", key])[gates + rel + ["gate_shift", "hetero"]].mean().reset_index()
        out[f"by_{key}"] = pt.groupby(key).agg(**{f"{c}_mean": (c, "mean") for c in gates + rel + ["gate_shift", "hetero"]}, n_patients=("patient", "nunique")).reset_index()
        out[f"by_{key}_patient"] = pt
    # (b) per section Spearman of each expert's weight with heterogeneity and cell-type fractions; per patient the mean over its
    #     sections; over patients the median, the number positive and a Wilcoxon signed-rank P against zero
    rows = []
    for s, d in D.groupby("section"):
        for gk in gates + ["gate_shift"]:
            for v in ["hetero"] + CELLS:
                if d[v].std() > 0:
                    rows.append({"section": s, "patient": d.patient.iloc[0], "weight": gk, "covariate": v, "rho": spearmanr(d[gk], d[v])[0]})
    cs = pd.DataFrame(rows).groupby(["patient", "weight", "covariate"]).rho.mean().reset_index(); summ = []
    for (gk, v), d in cs.groupby(["weight", "covariate"]):
        r = d.rho.dropna(); p = wilcoxon(r).pvalue if len(r) >= 5 and (r != 0).any() else np.nan
        summ.append({"weight": gk, "covariate": v, "n_patients": len(r), "median_rho": r.median(), "n_positive": int((r > 0).sum()), "wilcoxon_p": p})
    out["covariates"] = pd.DataFrame(summ); out["covariates_patient"] = cs
    # (c) expert weight across the tumour–lung interface (signed ring distance, negative inside the tumour), patient level
    dd = D.dropna(subset=["signed_ring"]); pt = dd.groupby(["patient", "signed_ring"])[gates + ["gate_shift", "hetero"]].mean().reset_index()
    out["interface_profile_patient"] = pt; out["interface_profile"] = pt.groupby("signed_ring")[gates + ["gate_shift", "hetero"]].mean().reset_index()
    remodelling(cfg, D, gates, rel, out)
    outs = []
    for k, v in out.items():
        f = sd / f"{variant}_{k}.tsv"; write_table(v, f); outs.append(f)
    log(f"\n{out['by_niche_name'][['niche_name'] + [g + '_mean' for g in gates] + ['n_patients']].round(3).to_string()}")
    log(f"\n{out['covariates'][out['covariates'].covariate.isin(['hetero', 'T_NK', 'Fibroblast', 'Osteosarcoma', 'Myeloid'])].round(3).to_string()}")
    write_provenance(STAGE + f"_{variant}", files, outs, cfg, t_start=t0)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="moe2"); ap.add_argument("--seed-index", type=int, default=0)
    ap.add_argument("--summarise", action="store_true"); ap.add_argument("--device", default="")
    a = ap.parse_args(); t0 = time.time(); s25 = _load("s25", "25_niche_reuse.py"); cfg = s25.atlas_cfg(load_config()); set_global_seed(cfg["seed"])
    if a.device:
        cfg["device"] = a.device
    if a.summarise:
        summarise(cfg, a.variant, t0)
    elif a.variant == "frozen":  # the delivered atlas encoder
        run_frozen(cfg)
    else:
        run_seed(cfg, a.variant, a.seed_index)


if __name__ == "__main__":
    main()
