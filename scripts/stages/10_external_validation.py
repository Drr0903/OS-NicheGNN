"""Stage 10 — one-shot external evaluation of the FROZEN pipeline on the Visium CytAssist cohort (OS1–OS7).

Everything below is fixed before this script is first run : gene list, scaler, encoder
weights (frozen seed), prototypes, unmatched threshold, linear probe (fitted on discovery embeddings and
discovery image labels), interface niche id, band/segment rules, T-cell gene set. External sections only pass
forward through the network; nothing is refitted. Lymph-node sections OS8/OS9 are embedded and reported
descriptively but excluded from every lung-interface statistic.

Checks reported
  A. input compatibility     genes present / missing, spots, pitch
  B. region + boundary       probe macro-F1 / balanced accuracy / boundary F1 vs external image labels, per specimen
  C. niche matching          fraction of spots unmatched; niche RCTD-composition correlation discovery↔external
  D. interface ↔ T readout   coverage, continuity, E_s, matched contrast direction per specimen; specimen-level summary
Outputs results/10_external/{compatibility.tsv, region_metrics.tsv, niche_matching.tsv, spot_niches_external.tsv,
        section_metrics.tsv, segments.tsv, matched_contrast.tsv, summary.json}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data import build_graph, load_labels, load_qc_sections  # noqa: E402
from lib.evalprobe import CLASSES, boundary_from_labels, boundary_prf, fit_probe  # noqa: E402
from lib.inference import pooled_models  # noqa: E402
from lib.interface import run_interface_cohort  # noqa: E402
from lib.repro import ROOT, frozen_name, load_config, log, results_path, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.train import embed_sections, make_model  # noqa: E402

STAGE = "10_external"


def main() -> None:
    t0 = time.time()
    cfg = load_config(); set_global_seed(cfg["seed"])
    proc = ROOT / cfg["paths"]["processed"]; fro = proc / frozen_name()
    meta = json.load(open(fro / "meta.json"))
    genes = open(fro / "genes.txt").read().split(); sc_ = np.load(fro / "scaler.npz"); mu, sd = sc_["mu"], sc_["sd"]
    protos = np.load(fro / "prototypes.npy"); K = meta["K"]; thr = meta["unmatched_threshold"]; inter = meta["interface_niche"]
    model = make_model("nichegnn", len(genes), cfg).to(cfg["device"])
    model.load_state_dict(torch.load(fro / f"model_seed{meta['frozen_seed']}.pt", map_location=cfg["device"])); model.eval()
    ext = cfg["cohorts"]["external"]; lung = ext["lung_sections"]; ln = ext["lymph_node_sections"]
    secs = [s for s in lung + ln if (proc / "qc" / f"{s}.h5ad").exists()]
    adatas = load_qc_sections(cfg, secs); labels = {s: load_labels(cfg, s) for s in secs}
    sd_ = stage_dir(STAGE)
    # A. compatibility
    comp = []
    for s in secs:
        a = adatas[s]; present = [g for g in genes if g in a.var_names]
        comp.append({"section": s, "organ": "lung" if s in lung else "lymph_node", "n_spots": a.n_obs, "n_model_genes": len(genes), "n_present": len(present), "n_missing": len(genes) - len(present), "pitch_img_px": float(a.uns["spot_pitch_img_px"])})
    comp = pd.DataFrame(comp); write_table(comp, sd_ / "compatibility.tsv")
    assert comp.loc[comp["organ"] == "lung", "n_missing"].max() == 0, "gene universe guarantees presence in lung objects; check stage 03"
    log(f"missing model genes per section (zero-filled; lymph nodes only): {dict(zip(comp['section'], comp['n_missing']))}")
    G = {s: build_graph(adatas[s], genes, mu, sd, cfg, f"EXT_{s}", labels[s]) for s in secs}
    emb = embed_sections(model, [G[s] for s in secs], "nichegnn", cfg, cfg["device"])
    emb_dir = proc / "embeddings" / "external"; emb_dir.mkdir(parents=True, exist_ok=True)
    for s in secs:
        np.save(emb_dir / f"{s}.npy", emb[s])
    # B. probe fitted on discovery frozen embeddings + discovery image labels
    disc_secs = meta["sections"]; disc_emb = {s: np.load(proc / "embeddings" / frozen_name() / f"{s}.npy") for s in disc_secs}
    disc_lab = {s: load_labels(cfg, s) for s in disc_secs}
    Ztr = np.vstack([disc_emb[s] for s in disc_secs]); ytr = np.concatenate([disc_lab[s]["label"].to_numpy(dtype=object) for s in disc_secs])
    # align label rows to embedding rows (stage 04 wrote labels in qc h5ad order = graph order)
    m = np.isin(ytr, CLASSES); scaler, clf = fit_probe(Ztr[m], ytr[m], cfg["seed"])
    from sklearn.metrics import balanced_accuracy_score, f1_score
    reg = []
    for s in secs:
        g = G[s]; pred = clf.predict(scaler.transform(emb[s])); ev = np.isin(g.labels, CLASSES)
        r = {"section": s, "organ": "lung" if s in lung else "lymph_node", "n_eval_spots": int(ev.sum())}
        if ev.sum() > 20:
            r["macro_f1"] = float(f1_score(g.labels[ev], pred[ev], average="macro")); r["balanced_accuracy"] = float(balanced_accuracy_score(g.labels[ev], pred[ev]))
            r.update(boundary_prf(boundary_from_labels(pred, g.xy, g.pitch), g.boundary, g.xy, g.pitch, cfg["histology"]["boundary_tolerance_rings"]))
        reg.append(r)
    reg = pd.DataFrame(reg); write_table(reg, sd_ / "region_metrics.tsv")
    # C. niche assignment + matching
    rows = []
    for s in secs:
        z = emb[s]; d = np.linalg.norm(z[:, None, :] - protos[None], axis=2); g = G[s]
        rows.append(pd.DataFrame({"section": s, "patient": f"EXT_{s}", "organ": "lung" if s in lung else "lymph_node", "barcode": g.barcodes, "niche": d.argmin(1), "dist_to_prototype": d.min(1),
                                  "unmatched": d.min(1) > thr, "histology_label": g.labels, "tumor_boundary": g.boundary, "tumor_lung_interface": g.interface, "x": g.xy[:, 0], "y": g.xy[:, 1]}))
    spots = pd.concat(rows, ignore_index=True)
    w_dir = ROOT / "results/05_deconvolution/weights"
    comp_rows = []
    for s in secs:
        p = w_dir / f"{s}_rctd_weights.tsv"
        if p.exists():
            w = pd.read_csv(p, sep="\t").set_index("barcode"); sub = spots[spots["section"] == s].set_index("barcode"); w = w.reindex(sub.index)
            w["niche"] = sub["niche"].values; w["section"] = s; comp_rows.append(w)
    match = {"frac_unmatched_lung": float(spots.loc[spots["organ"] == "lung", "unmatched"].mean()), "frac_unmatched_ln": float(spots.loc[spots["organ"] == "lymph_node", "unmatched"].mean()) if (spots["organ"] == "lymph_node").any() else None}
    if comp_rows:
        W = pd.concat(comp_rows); types = [c for c in W.columns if c not in ("niche", "section")]
        ext_comp = W[W["section"].isin(lung)].groupby("niche")[types].mean()
        disc_comp = pd.read_csv(results_path("08_niches/niche_composition_rctd.tsv"), sep="\t").set_index("niche")
        common = [t for t in types if t in disc_comp.columns]
        corr = {int(k): float(stats.pearsonr(ext_comp.loc[k, common], disc_comp.loc[k, common])[0]) if k in disc_comp.index else np.nan for k in ext_comp.index}
        match["niche_composition_pearson_r"] = corr; match["niche_composition_pearson_r_mean"] = float(np.nanmean(list(corr.values())))
        write_table(ext_comp.reset_index(), sd_ / "niche_composition_rctd_external.tsv")
        spots = spots.merge(W.reset_index()[["barcode", "section"] + types], on=["barcode", "section"], how="left")
    write_table(pd.crosstab(spots["section"], spots["niche"], normalize="index").reset_index(), sd_ / "niche_by_section.tsv")
    write_table(spots, sd_ / "spot_niches_external.tsv")
    json.dump(match, open(sd_ / "niche_matching.json", "w"), indent=2)
    # D. interface ↔ T readout with identical rules (lung specimens only, boundary-eligible)
    man = pd.read_csv(ROOT / "results/01_manifest/sample_manifest.tsv", sep="\t", dtype=str)
    elig = [s for s in lung if s in secs and man.set_index("section_id").loc[s, "boundary_eligible"] == "True"]
    sec, seg, con = run_interface_cohort(spots, proc / "qc", w_dir, cfg, inter, elig, {s: f"EXT_{s}" for s in elig})
    write_table(sec, sd_ / "section_metrics.tsv"); write_table(seg, sd_ / "segments.tsv"); write_table(con, sd_ / "matched_contrast.tsv")
    summ = {"n_lung_sections": len([s for s in secs if s in lung]), "n_boundary_eligible": len(elig), "region": {"macro_f1_mean_lung": float(reg[reg["organ"] == "lung"]["macro_f1"].mean()), "boundary_f1_mean_lung": float(reg[reg["organ"] == "lung"]["boundary_f1"].mean())},
            "niche_matching": match}
    ev = sec[sec["evaluable_T"]] if len(sec) else sec
    if len(ev):
        m2 = ev.merge(con[["section", "matched_diff_E", "unmatched_diff_E", "n_matched", "spearman_cov_E", "perm_p"]], on="section", how="left")
        summ["interface"] = {"n_evaluable_T": int(len(ev)), "coverage": m2["coverage"].round(3).tolist(), "E_s": m2["E_s"].round(3).tolist(),
                             "matched_diff_E": m2["matched_diff_E"].round(3).tolist(), "n_matched_diff_positive": int((m2["matched_diff_E"] > 0).sum()), "n_with_matched": int(m2["matched_diff_E"].notna().sum()),
                             "spearman_coverage_vs_E_specimen": float(stats.spearmanr(ev["coverage"], ev["E_s"]).correlation) if len(ev) >= 4 else None}
        pm = pooled_models(seg, cluster="section", n_perm=cfg["interface"]["n_permutations"], seed=cfg["seed"]); write_table(pm, sd_ / "pooled_models.tsv")
        rich = sec.loc[sec["n_t_genes_present"] >= 10, "section"].tolist()  # sensitivity: specimens with ≥ 10 held-out T genes present
        if len(rich) >= 3:
            pm10 = pooled_models(seg[seg["section"].isin(rich)], cluster="section", n_perm=cfg["interface"]["n_permutations"], seed=cfg["seed"]); pm10["specimens"] = ",".join(rich)
            write_table(pm10, sd_ / "pooled_models_min10genes.tsv")
        key = pm.set_index(["outcome", "covariate_set"])
        summ["segment_model"] = {f"{o}__{c}": {k: (None if (not isinstance(v, str) and pd.isna(v)) else (v if isinstance(v, str) else float(v))) for k, v in key.loc[(o, c)].drop(["covariates"]).items()} for o in key.index.get_level_values(0).unique() for c in ["none", "3type", "full"] if (o, c) in key.index}
        summ["E_variants_specimen_level"] = {v: float(stats.spearmanr(ev["coverage"], ev[v]).correlation) for v in ["E_s", "E_s_noniche", "E_s_distal"] if len(ev) >= 4}
        disc_pl = pd.read_csv(results_path("09_interface/patient_level.tsv"), sep="\t")
        dm = disc_pl["matched_diff_E"].dropna()
        summ["direction_replicated_matched"] = bool(len(dm) and np.sign(dm.mean()) == np.sign(np.nanmean(m2["matched_diff_E"]))) if m2["matched_diff_E"].notna().any() else None
    json.dump(summ, open(sd_ / "summary.json", "w"), indent=2); log(json.dumps(summ, indent=1))
    write_provenance(STAGE, [fro / "prototypes.npy", fro / f"model_seed{meta['frozen_seed']}.pt"], [sd_ / "region_metrics.tsv", sd_ / "section_metrics.tsv", sd_ / "spot_niches_external.tsv"], cfg, extra=summ, t_start=t0)


if __name__ == "__main__":
    main()
