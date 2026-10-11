"""Stage 23 () — neighbour-selecting spot encoders, scored by the stage-14/22 leave-one-patient-out protocol.

Per fold (stage-14 fold and inner-validation patient): PCA-50 fitted on the training patients; references PCA-50, BANKSY-20,
propagated PCA; learned encoders with context ∈ cfg.arch.contexts (none = own residual MLP only, xattn, mamba, xmamba), three
seeds, each scored as `_z` (fused latent) and `_skip` ([own PCA, learned context]); `+mnnproj` on PCA, BANKSY and every `_skip`.
`--anchor C` trains context C with the anchor-only objective of A10 (step 2). Metrics as stage 22 (no composition probe).
Usage: python scripts/stages/23_arch_search.py --patients P01 --tag _P01 ;  --anchor xattn --tag _a_P01 ;  --merge _P01,...
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.arch_eval import image_graphs, mnn_projection, paired, pca_graphs  # noqa: E402
from lib.arch_train import embed_arch, train_arch  # noqa: E402
from lib.banksy import BanksyEmbedder  # noqa: E402
from lib.arch_metrics import niche_metrics, targets  # noqa: E402
from lib.context_probe import load_weights, propagate  # noqa: E402
from lib.interface import t_score  # noqa: E402
from lib.evalprobe import evaluate_heldout  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.v2 import compute_anchors, identity_index, load_cohort, make_graphs  # noqa: E402

STAGE = "23_arch_search"
NT = int(os.environ.get("OMP_NUM_THREADS", 8))
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))


def run_fold(cfg, adatas, labels, patient_of, site_of, P, anchor_ctx: str | None, contexts: list[str], image: bool = False):
    secs = list(adatas); all_pats = sorted(set(patient_of.values())); K = cfg["v2"]["k_fixed"]; ac = cfg["arch"]; dev = cfg["device"]
    test = [s for s in secs if patient_of[s] == P]; train = [s for s in secs if patient_of[s] != P]
    rng = np.random.default_rng(cfg["seed"] + all_pats.index(P)); tr_pats = sorted({patient_of[s] for s in train}); val_pat = tr_pats[int(rng.integers(len(tr_pats)))]  # = stage 14
    val = [s for s in train if patient_of[s] == val_pat]; fit = [s for s in train if s not in val]
    G, _, _, _ = make_graphs(cfg, adatas, labels, patient_of, train, train + test); Gp = pca_graphs(G, train, cfg["remedies"]["pca_dim"], cfg["seed"])
    seeds = [cfg["seed"] + i for i in range(cfg["remedies"]["seeds"])]; rows = []; all_tr = [G[s] for s in train]; te = [G[s] for s in test]
    T = targets(G, {s: t_score(adatas[s], cfg["genes"]["t_cell_holdout"]) for s in G}, load_weights(G))  # A18 third set
    log(f"fold {P}: test {test}, inner val {val_pat}")

    def rec(method, seed, emb, info=None, identity=True):
        m = evaluate_heldout(emb, all_tr, te, cfg, seed); m.pop("per_section_boundary", None)
        idx = identity_index({s: emb[s] for s in train}, {s: G[s] for s in train}, K, seed) if identity else {}
        with threadpool_limits(limits=NT):
            nm = niche_metrics(emb, G, T, train, test, K, seed)
        rows.append({"method": method, "seed": seed, "heldout_patient": P, "heldout_site": "+".join(sorted({site_of[s] for s in test})), **m, **nm,
                     **{f"train_{k}": v for k, v in idx.items()}, "epochs_run": (info or {}).get("epochs_run"), "train_time_s": (info or {}).get("train_time_s")})
        log(f"  {method} s{seed}: F1 {m['macro_f1']:.3f} bF1 {m['boundary_f1']:.3f} ARIp {idx.get('ari_vs_patient', np.nan):.3f} "
            f"T {nm['tcell_rho']:.3f}/{nm['tcell_nbhd_rho']:.3f} comp {nm['comp_rho']:.3f}/{nm['comp_nbhd_rho']:.3f} chaos {nm['chaos']:.3f} pas {nm['pas']:.3f} mix {nm['mix']:.3f}")

    def rec_proj(method, seed, emb):
        if ac.get("no_proj"):  # the anchor-direction projection is not part of the selection
            return
        with threadpool_limits(limits=NT):
            rec(f"{method}+mnnproj", seed, mnn_projection(emb, G, train, ac["mnnproj_rank"], seed))

    pre = ""; sfx = ac.get("name_suffix", "")
    if image:  # second set: [expression PCA, H&E morphology PCA] as the spot profile; references with and without the image
        Gi, Gm = image_graphs(G, Gp, train, cfg["remedies"]["pca_dim"], cfg["seed"], ROOT / cfg["paths"]["processed"] / "image_features")
        if not contexts or "refs" in contexts:
            with threadpool_limits(limits=NT):
                rec("img", seeds[0], {s: Gi[s].x for s in train + test}); mm = {s: Gm[s].x for s in train + test}; rec("pca+img", seeds[0], mm); rec_proj("pca+img", seeds[0], mm)
                rec("propagated_pca+img", seeds[0], {s: propagate(Gm[s]) for s in train + test})
                b = cfg["baselines"]["banksy"]; be = BanksyEmbedder(b["lambda"], b["k_geom"], b["n_pcs"], seeds[0]).fit(all_tr)
                rec("banksy+img", seeds[0], {s: np.hstack([be.transform(G[s]), Gi[s].x]) for s in train + test})
            contexts = [c for c in contexts if c != "refs"] or cfg["arch"]["contexts"]
        Gp, pre = Gm, "mm_"
    sl = [Gp[s] for s in train + test]
    if anchor_ctx:
        anchors = compute_anchors(cfg, [Gp[s] for s in fit])
        for seed in seeds:
            model, info = train_arch(anchor_ctx, [Gp[s] for s in fit], [Gp[s] for s in val], cfg, seed, dev, anchors, cfg["v2"]["model"]["lambda_anchor"])
            E = embed_arch(model, sl, dev); rec(f"{pre}{anchor_ctx}_anchor{sfx}_z", seed, {s: E[s]["z"] for s in E}, info)
            if sfx:  # does the latent itself depend on the true arrangement of the neighbours?
                Er = embed_arch(model, sl, dev, rewire_seed=seed + cfg["remedies"]["rewire_seed_offset"])
                rec(f"{pre}{anchor_ctx}_anchor{sfx}_z_rewired", seed, {s: Er[s]["z"] for s in Er}, identity=False)
            if E[sl[0].section]["c"] is not None:
                sk = {s: np.hstack([Gp[s].x, E[s]["c"]]) for s in E}; rec(f"{pre}{anchor_ctx}_anchor{sfx}_skip", seed, sk, info); rec_proj(f"{pre}{anchor_ctx}_anchor{sfx}_skip", seed, sk)
        return rows
    if not image and (not contexts or "refs" in contexts):
        with threadpool_limits(limits=NT):
            pc = {s: Gp[s].x for s in train + test}; rec("pca", seeds[0], pc); rec_proj("pca", seeds[0], pc)
            rec("propagated_pca", seeds[0], {s: propagate(Gp[s]) for s in train + test})
            b = cfg["baselines"]["banksy"]; be = BanksyEmbedder(b["lambda"], b["k_geom"], b["n_pcs"], seeds[0]).fit(all_tr)
            bk = {s: be.transform(G[s]) for s in train + test}; rec("banksy", seeds[0], bk); rec_proj("banksy", seeds[0], bk)
    for ctx in [c for c in (contexts or ac["contexts"]) if c != "refs"]:
        for seed in seeds:
            model, info = train_arch(ctx, [Gp[s] for s in fit], [Gp[s] for s in val], cfg, seed, dev)
            E = embed_arch(model, sl, dev); rec(f"{pre}{ctx}_z", seed, {s: E[s]["z"] for s in E}, info)
            if E[sl[0].section]["c"] is not None:
                sk = {s: np.hstack([Gp[s].x, E[s]["c"]]) for s in E}; rec(f"{pre}{ctx}_skip", seed, sk, info); rec_proj(f"{pre}{ctx}_skip", seed, sk)
                Er = embed_arch(model, sl, dev, rewire_seed=seed + cfg["remedies"]["rewire_seed_offset"])
                rec(f"{pre}{ctx}_skip_rewired", seed, {s: np.hstack([Gp[s].x, Er[s]["c"]]) for s in Er}, identity=False)
    return rows


METRICS = ["macro_f1", "boundary_f1", "tcell_rho", "tcell_nbhd_rho", "comp_rho", "comp_nbhd_rho", "chaos", "pas", "mix"]
REFS = ["banksy", "pca", "propagated_pca", "banksy+img", "pca+img", "propagated_pca+img"]


def summarise(df: pd.DataFrame, cfg: dict) -> list[Path]:
    """Patient-level means (seeds averaged) of every metric; development / confirmation / all splits for the region probe; paired
    patient-level bootstrap of every metric against each reference (CHAOS / PAS: lower is better, sign kept as raw difference)."""
    sd = stage_dir(STAGE); ac = cfg["arch"]
    pat = df.groupby(["method", "heldout_patient"])[METRICS].mean().reset_index()
    summ = pat.groupby("method")[METRICS].mean()
    for split, pts in [("dev", ac["dev_patients"]), ("confirm", ac["confirm_patients"])]:
        summ[f"macro_f1_{split}"] = pat[pat.heldout_patient.isin(pts)].groupby("method").macro_f1.mean()
    summ["n_patients_f1"] = pat.dropna(subset=["macro_f1"]).groupby("method").heldout_patient.nunique()
    summ["n_patients"] = pat.groupby("method").heldout_patient.nunique()
    tr = df.groupby("method")[[c for c in df.columns if c.startswith("train_") and c not in ("train_K", "train_time_s")]].mean()
    summ = summ.join(tr).reset_index().sort_values("macro_f1", ascending=False); rows = []
    for ref in [r for r in REFS if r in set(pat.method)]:
        R = pat[pat.method == ref].set_index("heldout_patient")
        for m in sorted(set(pat.method) - {ref}):
            S = pat[pat.method == m].set_index("heldout_patient")
            for metric in METRICS:
                ix = S.index.intersection(R.index)
                rows.append({"comparison": f"{m} − {ref}", "metric": metric, **paired(S.loc[ix, metric], R.loc[ix, metric], cfg["seed"])})
    out = [sd / "arch_summary.tsv", sd / "arch_paired.tsv", sd / "arch_patient_level.tsv"]
    write_table(summ, out[0]); write_table(pd.DataFrame(rows), out[1]); write_table(pat, out[2]); log(summ.round(3).to_string()); return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--patients", default=""); ap.add_argument("--tag", default=""); ap.add_argument("--merge", default="")
    ap.add_argument("--anchor", default=""); ap.add_argument("--contexts", default=""); ap.add_argument("--image", action="store_true")
    ap.add_argument("--targets", default="", help="niche targets of the latent, e.g. ring1=0.5,outer=0.5,disp=0.25 (A20); suffix _nt<codes>")
    ap.add_argument("--linear-heads", action="store_true"); ap.add_argument("--prop-skip", action="store_true"); ap.add_argument("--grad-skip", action="store_true")
    ap.add_argument("--moe", action="store_true"); ap.add_argument("--clip", type=float, default=0.0); ap.add_argument("--dgi", type=float, default=0.0)
    ap.add_argument("--img-scale", type=float, default=1.0)
    ap.add_argument("--lambda-anchor", type=float, default=None, help="override v2.model.lambda_anchor (suffix _la<value>)")
    ap.add_argument("--no-proj", action="store_true", help="skip the +mnnproj rows")
    ap.add_argument("--dims", default="", help="hidden width,latent width (e.g. 256,128); methods are suffixed _d<h>z<l>")
    a = ap.parse_args(); t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); sd = stage_dir(STAGE)
    cfg["arch"]["no_proj"] = a.no_proj
    if a.dims:
        d, z = map(int, a.dims.split(",")); cfg["arch"].update(d=d, z_dim=z, name_suffix=f"_d{d}z{z}")
    sfx_extra = ("_lin" if a.linear_heads else "") + ("_ps" if a.prop_skip else "") + ("_gr" if a.grad_skip else "") + ("_moe" if a.moe else "") + ("_clip" if a.clip else "") + ("_dgi" if a.dgi else "") + (f"_img{a.img_scale:g}" if a.img_scale != 1.0 else "")
    cfg["arch"].update(img_scale=a.img_scale, linear_heads=a.linear_heads, prop_skip=a.prop_skip, grad_skip=a.grad_skip, moe=a.moe, clip=a.clip, dgi=a.dgi)
    if a.lambda_anchor is not None:
        cfg["v2"]["model"]["lambda_anchor"] = a.lambda_anchor; sfx_extra += f"_la{a.lambda_anchor:g}"
    if a.targets:
        nt = {k: float(v) for k, v in (kv.split("=") for kv in a.targets.split(","))}
        cfg["arch"]["niche_targets"] = nt; cfg["arch"]["name_suffix"] = cfg["arch"].get("name_suffix", "") + "_nt" + "".join(k[0] for k in nt if nt[k] > 0)
    if sfx_extra:
        cfg["arch"]["name_suffix"] = cfg["arch"].get("name_suffix", "") + sfx_extra
    if a.merge:
        files = [sd / f"arch_metrics{t}.tsv" for t in a.merge.split(",")]; df = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
        write_table(df, sd / "arch_metrics.tsv"); out = summarise(df, cfg); write_provenance(STAGE, files, out + [sd / "arch_metrics.tsv"], cfg, t_start=t0); return
    adatas, labels, patient_of, site_of = load_cohort(cfg); pats = [p for p in sorted(set(patient_of.values())) if not a.patients or p in a.patients.split(",")]
    rows = []; ctxs = [c for c in a.contexts.split(",") if c]
    for P in pats:
        rows += run_fold(cfg, adatas, labels, patient_of, site_of, P, a.anchor or None, ctxs, a.image)
        write_table(pd.DataFrame(rows), sd / f"arch_metrics{a.tag}.tsv")
    write_provenance(STAGE + f"_part{a.tag}", [], [sd / f"arch_metrics{a.tag}.tsv"], cfg, extra={"patients": pats, "anchor": a.anchor, "contexts": ctxs, "image": a.image, "dims": a.dims, "targets": a.targets}, t_start=t0)


if __name__ == "__main__":
    with threadpool_limits(limits=NT):
        main()
