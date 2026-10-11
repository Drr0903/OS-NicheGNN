"""Stage 08c —  (post hoc): final upgraded model on all ten patients → patient-identity diagnostics and K rule.

The configuration selected most often across LOPO folds (results/07_lopo/upgrade_selection.tsv) is trained on all discovery
lung sections for the frozen model's epoch count with the same five seeds; for each seed the embeddings are clustered
(Leiden, fixed K; A5) and compared with patient identity and region label exactly as in stage 08b, and the set in advance K rule
(every niche ≥ 2 % of spots in at least half of the patients) is evaluated for K = 4–8. Success criteria of A4 are checked.
Outputs results/08_niches/upgrade_identity_diagnostics.tsv, upgrade_k_rule.tsv, upgrade_summary.json;
        data/processed/embeddings/upgrade/<seed>/<section>.npy
"""
from __future__ import annotations

import json
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.cluster import assign, cluster_prototypes  # noqa: E402
from lib.data import build_graph, load_labels, load_qc_sections  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.train import embed_sections, train_model  # noqa: E402

STAGE = "08_niches"
s08b = import_module("08b_patient_identity")


def k_rule(emb: dict, G: dict, patient_of: dict, ks: range, seed: int, cfg_knn: int = 15) -> pd.DataFrame:
    rng = np.random.default_rng(seed); Z, P = [], []
    for pat in sorted(set(patient_of.values())):
        zs = np.vstack([emb[s] for s in emb if patient_of[s] == pat]); idx = rng.choice(len(zs), min(1500, len(zs)), replace=False); Z.append(zs[idx]); P += [pat] * len(idx)
    Z = np.vstack(Z); P = np.array(P); rows = []
    for K in ks:
        _, protos, _ = cluster_prototypes(Z, K, seed, batch=P, k_within=3)
        lab = np.concatenate([assign(emb[s], protos)[0] for s in emb]); pat = np.concatenate([[patient_of[s]] * len(emb[s]) for s in emb])
        frac = pd.crosstab(pat, lab, normalize="index"); present = (frac >= 0.02).sum(0)
        rows.append({"k": K, "n_niches_present_in_half_patients": int((present >= len(frac) / 2).sum()), "all_niches_present": bool((present >= len(frac) / 2).all())})
    return pd.DataFrame(rows)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="upgrade", choices=["upgrade", "invariant"]); args = ap.parse_args(); V = args.variant
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"])
    disc = cfg["cohorts"]["discovery"]; patient_of = disc["patient_groups"]; proc = ROOT / cfg["paths"]["processed"]
    secs = [s for s in disc["lung_sections"] if (proc / "qc" / f"{s}.h5ad").exists()]
    adatas = load_qc_sections(cfg, secs); labels = {s: load_labels(cfg, s) for s in secs}
    genes = open(proc / "frozen" / "genes.txt").read().split(); sc_ = np.load(proc / "frozen" / "scaler.npz"); meta = json.load(open(proc / "frozen" / "meta.json"))
    G = {s: build_graph(adatas[s], genes, sc_["mu"], sc_["sd"], cfg, patient_of[s], labels[s]) for s in secs}
    sel = pd.read_csv(ROOT / f"results/07_lopo/{V}_selection.tsv", sep="\t"); sel = sel[sel["selected"]]
    key = "lambda_adv" if V == "upgrade" else "lambda_anchor"
    counts = sel.groupby([key, "lambda_con"]).size().reset_index(name="n").sort_values(["n", key, "lambda_con"], ascending=[False, True, True])  # ties → simpler (smaller λ) configuration
    la, lc = float(counts.iloc[0][key]), float(counts.iloc[0]["lambda_con"])
    upg = {"lambda_adv": la, "lambda_con": lc} if V == "upgrade" else {"lambda_anchor": la, "lambda_con": lc, "lambda_nbr": cfg["invariant"]["lambda_nbr"]}
    log(f"most frequently selected configuration ({V}): {key} {la} lambda_con {lc} ({sel.groupby([key, 'lambda_con']).size().to_dict()})")
    cfg_final = json.loads(json.dumps(cfg)); cfg_final["train"]["max_epochs"] = meta["n_epochs"]; cfg_final["train"]["patience"] = 10**9
    base = pd.read_csv(ROOT / "results/08_niches/patient_identity_diagnostics.tsv", sep="\t").set_index("representation").loc["OS-NicheGNN (frozen)"]
    rows, krows = [], []
    for i in range(cfg["n_seeds"]):
        seed = cfg["seed"] + i; rng = np.random.default_rng(seed)
        if V == "invariant":
            from lib.upgrade import mnn_anchors
            upg["anchors"] = mnn_anchors([G[s] for s in secs], n_pcs=cfg["invariant"]["anchor_pcs"], k=cfg["invariant"]["anchor_k"], seed=cfg["seed"])
        model, _ = train_model("nichegnn", [G[s] for s in secs], [], cfg_final, seed, device=cfg["device"], upgrade=upg)
        emb = embed_sections(model, [G[s] for s in secs], "nichegnn", cfg, cfg["device"])
        d = proc / "embeddings" / V / str(seed); d.mkdir(parents=True, exist_ok=True)
        for s in secs:
            np.save(d / f"{s}.npy", emb[s])
        r = s08b.diagnostics(f"OS-NicheGNN {V} ({key.replace('lambda_', '')} {la}, con {lc}) seed {i + 1}", emb, G, meta["K"], seed, rng); r.update({"seed": seed, key: la, "lambda_con": lc}); rows.append(r); log(r)
        kr = k_rule(emb, G, patient_of, range(4, 9), seed, cfg["niches"]["knn"]); kr["seed"] = seed; krows.append(kr)
    out = pd.DataFrame(rows); kr = pd.concat(krows, ignore_index=True); sd = stage_dir(STAGE)
    write_table(out, sd / f"{V}_identity_diagnostics.tsv"); write_table(kr, sd / f"{V}_k_rule.tsv")
    lopo_up = pd.read_csv(ROOT / f"results/07_lopo/lopo_summary_gene_holdout_{V}.tsv", sep="\t").iloc[0]
    lopo_base = pd.read_csv(ROOT / "results/07_lopo/lopo_summary_gene_holdout.tsv", sep="\t").set_index(["method", "ablation"]).loc[("nichegnn", "full")]
    summ = {"variant": V, key: la, "lambda_con": lc, "ari_patient_mean": float(out["ari_vs_patient"].mean()), "ari_region_mean": float(out["ari_vs_region"].mean()),
            "ari_patient_frozen": float(base["ari_vs_patient"]), "ari_region_frozen": float(base["ari_vs_region"]),
            "clusters_in_half_patients_mean": float(out["clusters_present_in_half_patients"].mean()), "clusters_in_half_patients_frozen": int(base["clusters_present_in_half_patients"]),
            "k_rule_any_all_present": bool(kr["all_niches_present"].any()), "k_rule_n_present_k8_mean": float(kr[kr["k"] == 8]["n_niches_present_in_half_patients"].mean()),
            "lopo_macro_f1_upgrade": float(lopo_up["macro_f1_patient_mean"]), "lopo_macro_f1_frozen": float(lopo_base["macro_f1_patient_mean"]),
            "lopo_boundary_f1_upgrade": float(lopo_up["boundary_f1_patient_mean"]), "lopo_boundary_f1_frozen": float(lopo_base["boundary_f1_patient_mean"])}
    summ["criterion_1_macro_f1"] = summ["lopo_macro_f1_upgrade"] >= summ["lopo_macro_f1_frozen"] - 0.02
    summ["criterion_2_ari"] = (summ["ari_patient_frozen"] - summ["ari_patient_mean"] >= 0.10) and (summ["ari_region_frozen"] - summ["ari_region_mean"] <= 0.03)
    summ["criterion_3_niches"] = summ["clusters_in_half_patients_mean"] > summ["clusters_in_half_patients_frozen"]
    summ["upgrade_adopted"] = bool(summ["criterion_1_macro_f1"] and summ["criterion_2_ari"] and summ["criterion_3_niches"])
    json.dump(summ, open(sd / f"{V}_summary.json", "w"), indent=2); log(json.dumps(summ, indent=1))
    write_provenance(STAGE + f"_{V}", [ROOT / f"results/07_lopo/{V}_selection.tsv"], [sd / f"{V}_identity_diagnostics.tsv", sd / f"{V}_summary.json"], cfg, extra=summ, t_start=t0)


if __name__ == "__main__":
    main()
