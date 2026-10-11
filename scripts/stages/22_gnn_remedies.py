"""Stage 22 () — why the OS-NicheGNN encoder underuses its graph, and objectives that make it use it.

Per leave-one-patient-out fold (19 patients, the stage-14 fold and inner-validation assignment):
  references   PCA-50, BANKSY-20, standardised expression, non-learned propagation of PCA-50 ([P, ring-1 mean, ring-2–3 mean]),
               plain OS-NicheGNN and the plain MLP autoencoder (stage-14 objective)
  diagnostic   the plain OS-NicheGNN embedded on a degree-preserving rewired graph and with no edges
  remedies     gnn_pca (PCA input), mae_gene / mae_pca (GraphMAE whole-spot masking), dgi_pca (local infomax + reconstruction)
  --anchor wave3  the A17 third set (skip_gnnpca, skip_mae)
  --anchor wave2  the A17 second set (gnn_noaux, gnn_nearonly, gnn_invmask, gnn_self)
  --anchor V   the remedy V trained with the anchor-only objective of A10 (second step of A17)
Metrics per method × seed × held-out patient: the stage-14 region probe (macro-F1, boundary F1), held-out R² of the spot's own
and of its ring-2–3 neighbourhood RCTD composition (ridge probe on training patients), training-embedding identity indices.
Outputs results/22_gnn_remedies/{remedy_metrics<tag>.tsv, remedy_summary.tsv, remedy_paired.tsv}
Usage: python scripts/stages/22_gnn_remedies.py --patients P01 --tag _P01 ;  --merge _P01,_P02,... ; --anchor mae_pca
"""
from __future__ import annotations

import argparse
import dataclasses
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
from lib.context_probe import context_r2, load_weights, propagate  # noqa: E402
from lib.evalprobe import evaluate_heldout  # noqa: E402
from lib.remedies import embed_remedy, embed_skip, train_remedy  # noqa: E402
from lib.sattn import SelfContextAttention, gate_profile  # noqa: E402
from lib.repro import load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.train import embed_sections, train_model  # noqa: E402
from lib.v2 import compute_anchors, identity_index, load_cohort, make_graphs, train_and_embed  # noqa: E402

STAGE = "22_gnn_remedies"
WAVE2 = [("gnn_noaux", {"use_aux_context": False}, {}), ("gnn_nearonly", {"use_outer": False, "use_diff": False}, {}),
         ("gnn_invmask", {}, {"inverted_mask": True}),
         ("gnn_self", {"use_outer": False, "use_diff": False, "use_aux_context": False}, {"inverted_mask": True})]
torch.set_num_threads(8)


def pca_graphs(G: dict, train: list[str], dim: int, seed: int) -> dict:
    """PCA fitted on the training patients' standardised expression; scores divided by one global training s.d."""
    pca = PCA(dim, random_state=seed).fit(np.vstack([G[s].x for s in train])); sc = float(np.vstack([pca.transform(G[s].x) for s in train]).std())
    return {s: dataclasses.replace(g, x=(pca.transform(g.x) / sc).astype(np.float32)) for s, g in G.items()}


def run_fold(cfg, adatas, labels, patient_of, site_of, P, anchor_variant: str | None):
    secs = list(adatas); all_pats = sorted(set(patient_of.values())); K = cfg["v2"]["k_fixed"]; rc = cfg["remedies"]; dev = cfg["device"]
    test = [s for s in secs if patient_of[s] == P]; train = [s for s in secs if patient_of[s] != P]
    rng = np.random.default_rng(cfg["seed"] + all_pats.index(P)); tr_pats = sorted({patient_of[s] for s in train}); val_pat = tr_pats[int(rng.integers(len(tr_pats)))]  # = stage 14
    val = [s for s in train if patient_of[s] == val_pat]; fit = [s for s in train if s not in val]
    G, _, _, _ = make_graphs(cfg, adatas, labels, patient_of, train, train + test); Gp = pca_graphs(G, train, rc["pca_dim"], cfg["seed"])
    W = load_weights(G); seeds = [cfg["seed"] + i for i in range(rc["seeds"])]; rows = []
    all_tr = [G[s] for s in train]; te = [G[s] for s in test]
    log(f"fold {P}: test {test}, inner val {val_pat}")

    def rec(method, seed, emb, info=None, identity=True, extra=None):
        m = evaluate_heldout(emb, all_tr, te, cfg, seed); m.pop("per_section_boundary", None)
        cx = context_r2(emb, G, W, train, test, rc, seed)
        idx = identity_index({s: emb[s] for s in train}, {s: G[s] for s in train}, K, seed) if identity else {}
        rows.append({"method": method, "seed": seed, "heldout_patient": P, "heldout_site": "+".join(sorted({site_of[s] for s in test})), **m, **cx,
                     **{f"train_{k}": v for k, v in idx.items()}, "epochs_run": (info or {}).get("epochs_run"), **(extra or {})})
        log(f"  {method} s{seed}: F1 {m['macro_f1']:.3f} bF1 {m['boundary_f1']:.3f} ctxR2 {cx['context_r2']:.3f} ownR2 {cx['own_r2']:.3f}")

    if anchor_variant == "sattn":  # self-versus-context attention, without and with the anchor-only term
        sl = [Gp[s] for s in train + test]; anchors = compute_anchors(cfg, [Gp[s] for s in fit])
        for seed in seeds:
            for name, lam in [("sattn", 0.0), ("sattn_anchor0.5", 0.5), ("sattn_anchor1.0", 1.0)]:
                model, info = train_remedy("mae", [Gp[s] for s in fit], [Gp[s] for s in val], cfg, seed, dev, anchors if lam else None, lam, factory=SelfContextAttention)
                rec(name, seed, embed_remedy(model, sl, dev), info, extra=gate_profile(model, [Gp[s] for s in test], dev))
                if lam == 0.0:
                    rec(f"{name}_rewired", seed, embed_remedy(model, sl, dev, rewire_seed=seed + rc["rewire_seed_offset"]), identity=False)
        return rows
    if anchor_variant == "skip_gnnpca":  # A17 step 2: the selected remedy trained with the anchor-only objective of A10
        sl = [Gp[s] for s in train + test]; anchors = compute_anchors(cfg, [Gp[s] for s in fit])
        for seed in seeds:
            emb, info = train_and_embed(cfg, "invariant_anchoronly", [Gp[s] for s in fit], [Gp[s] for s in val], sl, seed, dev, anchors=anchors)
            rec("gnnpca_anchoronly", seed, emb, info); rec("skip_gnnpca_anchoronly", seed, embed_skip(info["model"].to(dev), sl, dev), info)
        return rows
    if anchor_variant == "wave3":  # A17 third set: own PCA kept, learned neighbourhood summaries appended
        sl = [Gp[s] for s in train + test]
        for seed in seeds:
            _, info = train_and_embed(cfg, "gnn", [Gp[s] for s in fit], [Gp[s] for s in val], sl, seed, dev); rec("skip_gnnpca", seed, embed_skip(info["model"].to(dev), sl, dev), info)
            model, info = train_remedy("mae", [Gp[s] for s in fit], [Gp[s] for s in val], cfg, seed, dev); rec("skip_mae", seed, embed_skip(model.gnn, sl, dev), info)
        return rows
    if anchor_variant == "wave2":  # A17 second set: attribution of the over-smoothing
        for seed in seeds:
            for name, model_kw, train_kw in WAVE2:
                c = json.loads(json.dumps(cfg)); c["model"].update(model_kw); c["train"].update(train_kw); gl = [G[s] for s in train + test]
                model, info = train_model("nichegnn", [G[s] for s in fit], [G[s] for s in val], c, seed, device=dev); rec(name, seed, embed_sections(model, gl, "nichegnn", c, dev), info)
                if name == "gnn_self":
                    rec(f"{name}_rewired", seed, embed_sections(model, gl, "nichegnn", c, dev, ablation="rewired", seed=seed + rc["rewire_seed_offset"]), identity=False)
        return rows
    if anchor_variant:
        anchors = compute_anchors(cfg, [G[s] for s in fit]); src = Gp if anchor_variant.endswith("_pca") else G
        for seed in seeds:
            model, info = train_remedy(anchor_variant.split("_")[0], [src[s] for s in fit], [src[s] for s in val], cfg, seed, dev, anchors, cfg["v2"]["model"]["lambda_anchor"])
            rec(f"{anchor_variant}_anchoronly", seed, embed_remedy(model, [src[s] for s in train + test], dev), info)
        return rows
    with threadpool_limits(limits=8):
        rec("pca", seeds[0], {s: Gp[s].x for s in train + test})
        rec("raw", seeds[0], {s: G[s].x for s in train + test}, identity=False)
        rec("propagated_pca", seeds[0], {s: propagate(Gp[s]) for s in train + test})
        b = cfg["baselines"]["banksy"]; be = BanksyEmbedder(b["lambda"], b["k_geom"], b["n_pcs"], seeds[0]).fit(all_tr); rec("banksy", seeds[0], {s: be.transform(G[s]) for s in train + test})
    for seed in seeds:
        emb, info = train_and_embed(cfg, "gnn", [G[s] for s in fit], [G[s] for s in val], [G[s] for s in train + test], seed, dev); rec("gnn_ref", seed, emb, info)
        model = info["model"]; gl = [G[s] for s in train + test]
        rec("gnn_ref_rewired", seed, embed_sections(model, gl, "nichegnn", cfg, dev, ablation="rewired", seed=seed + rc["rewire_seed_offset"]), identity=False)
        rec("gnn_ref_no_edges", seed, embed_sections(model, gl, "nichegnn", cfg, dev, ablation="no_edges"), identity=False)
        emb, info = train_and_embed(cfg, "mlp", [G[s] for s in fit], [G[s] for s in val], gl, seed, dev); rec("mlp_ref", seed, emb, info)
        emb, info = train_and_embed(cfg, "gnn", [Gp[s] for s in fit], [Gp[s] for s in val], [Gp[s] for s in train + test], seed, dev); rec("gnn_pca", seed, emb, info)
        for name, obj, src in [("mae_gene", "mae", G), ("mae_pca", "mae", Gp), ("dgi_pca", "dgi", Gp)]:
            model, info = train_remedy(obj, [src[s] for s in fit], [src[s] for s in val], cfg, seed, dev); sl = [src[s] for s in train + test]
            rec(name, seed, embed_remedy(model, sl, dev), info)
            if obj == "mae":
                rec(f"{name}_rewired", seed, embed_remedy(model, sl, dev, rewire_seed=seed + rc["rewire_seed_offset"]), identity=False)
    return rows


def paired(a: pd.Series, b: pd.Series, seed: int) -> dict:
    from scipy import stats
    d = (a - b).dropna(); rng = np.random.default_rng(seed); boots = [rng.choice(d.to_numpy(), len(d)).mean() for _ in range(5000)]
    return {"n_patients": len(d), "mean_diff": float(d.mean()), "ci95_lo": float(np.percentile(boots, 2.5)), "ci95_hi": float(np.percentile(boots, 97.5)),
            "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum()), "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if (d != 0).any() else 1.0}


def summarise(df: pd.DataFrame, cfg: dict) -> list[Path]:
    sd = stage_dir(STAGE); mets = ["macro_f1", "boundary_f1", "context_r2", "own_r2", "context_r2_partial"]
    pat = df.groupby(["method", "heldout_patient"])[mets].mean().reset_index()
    summ = pat.groupby("method")[mets].agg(["mean", "std"]); summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    tr = df.groupby("method")[[c for c in df.columns if c.startswith("train_") and c != "train_K"]].mean()
    summ = summ.join(tr).reset_index().sort_values("macro_f1_mean", ascending=False); rows = []
    for ref in ["gnn_ref", "banksy"]:
        R = pat[pat.method == ref].set_index("heldout_patient")
        for m in sorted(set(pat.method) - {ref}):
            S = pat[pat.method == m].set_index("heldout_patient")
            for metric in ["macro_f1", "boundary_f1", "context_r2"]:
                rows.append({"comparison": f"{m} − {ref}", "metric": metric, **paired(S[metric], R.reindex(S.index)[metric], cfg["seed"])})
    out = [sd / "remedy_summary.tsv", sd / "remedy_paired.tsv", sd / "remedy_patient_level.tsv"]
    write_table(summ, out[0]); write_table(pd.DataFrame(rows), out[1]); write_table(pat, out[2]); log(summ.round(3).to_string()); return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--patients", default=""); ap.add_argument("--tag", default=""); ap.add_argument("--merge", default=""); ap.add_argument("--anchor", default="")
    a = ap.parse_args(); t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE)
    if a.merge:
        files = [sd / f"remedy_metrics{t}.tsv" for t in a.merge.split(",")]; df = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
        write_table(df, sd / "remedy_metrics.tsv"); out = summarise(df, cfg); write_provenance(STAGE, files, out + [sd / "remedy_metrics.tsv"], cfg, t_start=t0); return
    adatas, labels, patient_of, site_of = load_cohort(cfg); pats = [p for p in sorted(set(patient_of.values())) if not a.patients or p in a.patients.split(",")]
    rows = []
    for P in pats:
        rows += run_fold(cfg, adatas, labels, patient_of, site_of, P, a.anchor or None)
        write_table(pd.DataFrame(rows), sd / f"remedy_metrics{a.tag}.tsv")
    write_provenance(STAGE + f"_part{a.tag}", [], [sd / f"remedy_metrics{a.tag}.tsv"], cfg, extra={"patients": pats, "anchor": a.anchor}, t_start=t0)


if __name__ == "__main__":
    with threadpool_limits(limits=8):
        main()
