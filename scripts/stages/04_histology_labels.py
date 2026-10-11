"""Stage 04 — tissue-region reference labels (tumour / lung parenchyma / paucicellular) with orthogonal checks.

Labelling rule (fixed, identical for both cohorts; see notes/DEVIATIONS.md 2026-09-15 for what was tried first):
  1. lung_parenchyma  — locked lung-epithelium marker score (config genes.region_label_panel.lung), cohort-wide
                        two-component Gaussian mixture, upper component. Spatially crisp in every section tested.
  2. tumor_cellular   — among non-lung spots, an inferCNV-style copy-number signal (lib/cnv.py; reference = the
                        cohort's lung spots; genes ordered by GRCh38 position, 100-gene windows) summarised as
                        signal magnitude and correlation with the section's dominant CNV profile; cohort-wide
                        two-component mixture on (log signal, correlation), upper component = malignant.
                        Osteosarcoma is highly aneuploid, so this does not depend on differentiation markers,
                        which fail in poorly differentiated lesions.
  3. paucicellular    — everything else (reactive stroma, immune infiltrate, vessels, necrosis).
  Ring-1 majority smoothing. Boundary flags / eligibility as set in advance (≥100 tumour, ≥50 lung, ≥30 interface).

Orthogonal checks written to results/04_histology/image_label_agreement.tsv: (i) AUC of the H&E air-space
fraction for lung vs rest (image only), (ii) AUC of haematoxylin OD for tumour vs paucicellular, (iii) discovery
only: AUC of the source atlas' own cancer-component weight (sum of ICA components annotated "Cancer" by the
original authors) for tumour vs paucicellular — an author-curated, independently derived tumour signal.
Model inputs exclude the marker panel and T-cell genes (config genes.*); the CNV signal uses all genes but is
never an input to the representation.

Outputs
  data/processed/histology/<sec>_labels.tsv   per-spot scores (lung marker, cnv_signal, cnv_corr, image), label, flags
  data/processed/qc_images/labels_<sec>.png   overlay for visual verification
  results/04_histology/{section_label_summary.tsv, image_label_agreement.tsv, mixture_components.tsv}
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2hed
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.cnv import cnv_profiles, cnv_scores, ordered_genes  # noqa: E402
from lib.spatial import ring_neighbors  # noqa: E402

STAGE = "04_histology"
Image.MAX_IMAGE_PIXELS = None
GROUPS = {"tumor": ["Osteosarcoma"], "lung": ["Lung_epithelium", "Alveolar_macrophage"],
          "other": ["Fibroblast", "Pericyte_SMC", "Endothelial", "Myeloid", "Osteoclast", "T_NK", "B_plasma", "Mast"]}


def ica_cancer_weight(cfg: dict) -> pd.Series | None:
    """Sum of positive weights of the source atlas' ICA components annotated Type == Cancer (Use == True)."""
    emb = ROOT / cfg["paths"]["raw_gse293065"] / ".." / "GSE293065_ica.cell.embeddings.all.csv.gz"
    ann = ROOT / "data/refs/PMC13470005/suppl/41467_2026_74603_MOESM3_ESM.xlsx"
    if not emb.exists() or not ann.exists():
        return None
    a = pd.read_excel(ann, header=1)
    ics = a.loc[(a["Type"] == "Cancer") & (a["Use"].astype(str) == "True"), "IC"].tolist()
    e = pd.read_csv(emb, index_col=0)
    w = e[[c for c in ics if c in e.columns]].clip(lower=0).sum(1)
    w.index = [i.split("_", 1)[0] + "|" + i.split("_", 1)[1] for i in w.index]
    return w


def two_component_upper(x: np.ndarray, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    gmm = GaussianMixture(2, n_init=5, random_state=seed).fit(x)
    hi = int(np.argmax(gmm.means_[:, 0]))
    return gmm.predict(x) == hi, pd.DataFrame(gmm.means_, columns=[f"f{i}" for i in range(x.shape[1])]).assign(weight=gmm.weights_, upper=[i == hi for i in range(2)])


def image_and_coords(a: ad.AnnData, cohort: str, cfg: dict):
    proc = ROOT / cfg["paths"]["processed"]
    if cohort == "discovery":
        img = np.asarray(Image.open(proc / "qc_images" / f"{a.obs['section'].iloc[0]}_hires.png").convert("RGB"))
        s = a.uns["scalefactors"]["tissue_hires_scalef"]
        return img, a.obsm["spatial"] * s, float(a.uns["spot_pitch_fullres_px"] * s)
    img = np.asarray(Image.open(proc / "external_export" / a.obs["section"].iloc[0] / "image.png").convert("RGB"))
    return img, a.obsm["spatial"], float(a.uns["spot_pitch_img_px"])


def patch_features(img: np.ndarray, xy: np.ndarray, pitch: float) -> pd.DataFrame:
    """H&E patch features (orthogonal check only): air-space fraction relative to the image background, H/E OD."""
    half = max(3, int(round(0.65 * pitch)))
    H, W, _ = img.shape
    imgf = img.astype(np.float32) / 255.0
    lum = 0.299 * imgf[..., 0] + 0.587 * imgf[..., 1] + 0.114 * imgf[..., 2]
    sat = imgf.max(-1) - imgf.min(-1)
    c = 50
    bg = float(np.median(np.concatenate([lum[:c, :c].ravel(), lum[-c:, -c:].ravel(), lum[:c, -c:].ravel(), lum[-c:, :c].ravel()])))
    tissue = (lum < bg - 0.04) | (sat > 0.12)
    hed = rgb2hed(imgf)
    rows = []
    for x, y in xy:
        cc, rr = int(round(x)), int(round(y))
        r0, r1, c0, c1 = max(0, rr - half), min(H, rr + half + 1), max(0, cc - half), min(W, cc + half + 1)
        T = tissue[r0:r1, c0:c1] if (r1 > r0 and c1 > c0) else np.zeros((1, 1), bool)
        if T.sum() < 4:
            rows.append(dict(img_tissue_frac=float(T.mean()) if T.size else 0.0, img_hem_od=np.nan, img_eos_od=np.nan)); continue
        rows.append(dict(img_tissue_frac=float(T.mean()), img_hem_od=float(hed[r0:r1, c0:c1, 0][T].mean()), img_eos_od=float(hed[r0:r1, c0:c1, 1][T].mean())))
    return pd.DataFrame(rows)


def marker_scores(a: ad.AnnData, panel: dict[str, list[str]]) -> pd.DataFrame:
    X = a.layers["lognorm"]
    out = {}
    for k, genes in panel.items():
        g = [x for x in genes if x in a.var_names]
        out[f"score_{k}"] = np.asarray(X[:, [a.var_names.get_loc(x) for x in g]].mean(1)).ravel() if g else np.zeros(a.n_obs)
    return pd.DataFrame(out, index=a.obs_names)


def smooth_labels(lab: np.ndarray, nbrs: list[np.ndarray]) -> np.ndarray:
    out = lab.copy()
    for i, nb in enumerate(nbrs):
        if len(nb) < 3:
            continue
        vals, cnt = np.unique(np.concatenate([[lab[i]], lab[nb]]), return_counts=True)
        if cnt.max() > (len(nb) + 1) / 2:
            out[i] = vals[cnt.argmax()]
    return out


def overlay(img, xy, lab, pitch, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"tumor_cellular": "#7b2cbf", "lung_parenchyma": "#2a9d8f", "paucicellular": "#e9c46a"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img); ax[0].axis("off")
    ax[1].imshow(img, alpha=0.35)
    for k, c in col.items():
        m = lab == k
        if m.any():
            ax[1].scatter(xy[m, 0], xy[m, 1], s=max(1.0, (pitch * 0.5) ** 2 * 0.6), c=c, label=f"{k} ({m.sum()})", linewidths=0)
    ax[1].axis("off"); ax[1].legend(loc="lower left", fontsize=7, markerscale=2, frameon=True)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def safe_auc(y: np.ndarray, x: np.ndarray) -> float:
    ok = ~np.isnan(x)
    y, x = y[ok], x[ok]
    return float(roc_auc_score(y, x)) if len(y) > 20 and 0 < y.sum() < len(y) else np.nan


def main() -> None:
    t0 = time.time()
    cfg = load_config(); set_global_seed(cfg["seed"])
    panel = cfg["genes"]["region_label_panel"]
    proc = ROOT / cfg["paths"]["processed"]
    out_dir = proc / "histology"; out_dir.mkdir(exist_ok=True); (proc / "qc_images").mkdir(exist_ok=True)
    w_dir = ROOT / "results/05_deconvolution/weights"
    positions = pd.read_csv(ROOT / "metadata/gene_positions_grch38_ens113.tsv", sep="\t")
    ica = ica_cancer_weight(cfg)
    files = sorted((proc / "qc").glob("*.h5ad"))
    per = {}
    for f in files:
        a = ad.read_h5ad(f); cohort = a.obs["cohort"].iloc[0]
        img, xy, pitch = image_and_coords(a, cohort, cfg)
        df = pd.concat([marker_scores(a, panel), patch_features(img, xy, pitch).set_index(a.obs_names)], axis=1)
        df["section"] = f.stem; df["cohort"] = cohort
        per[f.stem] = (a, df, xy, pitch, img)
        log(f"{f.stem}: {len(df)} spots loaded")
    summaries, agree, outputs, comps = [], [], [], []
    for cohort in ["discovery", "external"]:
        keys = [k for k, v in per.items() if v[1]["cohort"].iloc[0] == cohort]
        if not keys:
            continue
        allf = pd.concat([per[k][1] for k in keys])
        # 1. provisional lung spots (cohort-wide 2-component mixture on the lung marker score) — used ONLY as the
        #    CNV reference; the final lung call is made among non-tumour spots (step 3)
        prov_lung, _ = two_component_upper(allf[["score_lung"]].to_numpy(), cfg["seed"])
        genes = None
        for k in keys:
            g = ordered_genes(per[k][0].var_names, positions)
            genes = g if genes is None else genes.loc[genes.index.intersection(g.index)]
        for it in range(2):
            ref_rows = []
            for k in keys:
                a = per[k][0]; m = prov_lung[(allf["section"] == k).to_numpy()]
                if m.sum() >= 20:
                    idx = [a.var_names.get_loc(x) for x in genes.index]
                    E = a.layers["lognorm"][:, idx]; E = np.asarray(E.todense() if hasattr(E, "todense") else E)
                    ref_rows.append(E[m])
            ref_mean = np.vstack(ref_rows).mean(0).astype(np.float32) if ref_rows else None
            log(f"{cohort} iteration {it}: CNV reference = {sum(len(r) for r in ref_rows)} lung spots from {len(ref_rows)} sections; {len(genes)} positioned genes")
            sig_all, cor_all = [], []
            for k in keys:
                a = per[k][0]; m = prov_lung[(allf["section"] == k).to_numpy()]
                P, _, _ = cnv_profiles(a.layers["lognorm"], a.var_names, genes, m, ref_mean=ref_mean)
                nb1 = ring_neighbors(per[k][2], per[k][3], rings=(1,))
                P = np.vstack([P[[i] + list(nb1[i])].mean(0) for i in range(P.shape[0])])   # ring-1 spatial smoothing
                sig, cor = cnv_scores(P)
                per[k][1]["cnv_signal"] = sig; per[k][1]["cnv_corr"] = cor
                sig_all.append(sig); cor_all.append(cor)
            allf["cnv_signal"] = np.concatenate(sig_all); allf["cnv_corr"] = np.concatenate(cor_all)
            # 2. tumour: cohort-wide 2-component mixture on (log CNV signal, CNV correlation), all spots
            feats = np.column_stack([np.log(allf["cnv_signal"].to_numpy() + 1e-6), allf["cnv_corr"].to_numpy()])
            is_tum, comp_t = two_component_upper(feats, cfg["seed"])
            # 3. lung: 2-component mixture on the lung score fitted on NON-tumour spots only
            nt = ~is_tum
            lung_nt, comp_l = two_component_upper(allf.loc[nt, ["score_lung"]].to_numpy(), cfg["seed"])
            is_lung = np.zeros(len(allf), bool); is_lung[np.where(nt)[0]] = lung_nt
            prov_lung = is_lung
        comps.append(comp_t.assign(cohort=cohort, model="cnv_log_signal_corr")); comps.append(comp_l.assign(cohort=cohort, model="lung_score_nontumour"))
        allf["is_tumor"] = is_tum; allf["is_lung"] = is_lung
        log(f"{cohort}: lung {allf['is_lung'].sum()}, tumour {is_tum.sum()}, other {len(allf) - allf['is_lung'].sum() - is_tum.sum()}")
        for k in keys:
            a, df, xy, pitch, img = per[k]
            sub = allf[allf["section"] == k]
            lab = np.where(sub["is_tumor"].to_numpy(), "tumor_cellular", np.where(sub["is_lung"].to_numpy(), "lung_parenchyma", "paucicellular")).astype(object)
            nb = ring_neighbors(xy, pitch, rings=(1,)); nb3 = ring_neighbors(xy, pitch, rings=(1, 2, 3))
            lab = smooth_labels(lab, nb)
            df["label"] = lab
            wp = w_dir / f"{k}_rctd_weights.tsv"
            if wp.exists():
                W = pd.read_csv(wp, sep="\t").set_index("barcode").reindex(a.obs_names).fillna(0.0)
                for g_, cols in GROUPS.items():
                    df[f"w_{g_}"] = W[[c for c in cols if c in W.columns]].sum(1).to_numpy()
            if ica is not None and cohort == "discovery":
                df["ica_cancer_weight"] = ica.reindex([f"{k}|{b}" for b in a.obs_names]).to_numpy()
            is_t = lab == "tumor_cellular"; is_l = lab == "lung_parenchyma"
            df["tumor_boundary"] = [bool(is_t[i] and (~is_t[nb[i]]).any()) if len(nb[i]) else False for i in range(len(df))]
            tb = df["tumor_boundary"].to_numpy()
            df["tumor_lung_interface"] = [bool(tb[i] and is_l[nb3[i]].any()) for i in range(len(df))]
            p = out_dir / f"{k}_labels.tsv"; write_table(df.reset_index().rename(columns={"index": "barcode"}), p); outputs.append(p)
            overlay(img, xy, lab, pitch, proc / "qc_images" / f"labels_{k}.png")
            counts = pd.Series(lab).value_counts(); n_if = int(df["tumor_lung_interface"].sum())
            eligible = bool(counts.get("tumor_cellular", 0) >= 100 and counts.get("lung_parenchyma", 0) >= 50 and n_if >= 30)
            summaries.append({"section": k, "cohort": cohort, **{c: int(counts.get(c, 0)) for c in ["tumor_cellular", "lung_parenchyma", "paucicellular"]},
                              "n_tumor_boundary": int(tb.sum()), "n_tumor_lung_interface": n_if, "boundary_eligible": eligible})
            nl_ = ~is_l
            agree.append({"section": k, "cohort": cohort, "img_px_per_pitch": round(pitch, 1),
                          "auc_airspace_lung_vs_rest": safe_auc(is_l, -df["img_tissue_frac"].to_numpy(float)),
                          "auc_hemOD_tumor_vs_pauci": safe_auc(is_t[nl_], df.loc[nl_, "img_hem_od"].to_numpy(float)) if nl_.sum() > 20 else np.nan,
                          "auc_ica_cancer_tumor_vs_pauci": safe_auc(is_t[nl_], df.loc[nl_, "ica_cancer_weight"].to_numpy(float)) if ("ica_cancer_weight" in df and nl_.sum() > 20) else np.nan,
                          "auc_tumorpanel_tumor_vs_pauci": safe_auc(is_t[nl_], df.loc[nl_, "score_tumor"].to_numpy(float)) if nl_.sum() > 20 else np.nan,
                          "auc_rctd_tumor_vs_pauci": safe_auc(is_t[nl_], df.loc[nl_, "w_tumor"].to_numpy(float)) if ("w_tumor" in df and nl_.sum() > 20) else np.nan})
            log(f"{k}: T {counts.get('tumor_cellular', 0)} L {counts.get('lung_parenchyma', 0)} O {counts.get('paucicellular', 0)} | interface {n_if} | eligible {eligible}")
    sd_ = stage_dir(STAGE)
    summ = pd.DataFrame(summaries); write_table(summ, sd_ / "section_label_summary.tsv")
    ag = pd.DataFrame(agree); write_table(ag, sd_ / "image_label_agreement.tsv")
    write_table(pd.concat(comps), sd_ / "mixture_components.tsv")
    log(f"label agreement (AUC):\n{ag.round(3).to_string()}")
    man_path = ROOT / "results/01_manifest/sample_manifest.tsv"
    man = pd.read_csv(man_path, sep="\t", dtype=str)
    for _, r in summ.iterrows():
        m = man["section_id"] == r["section"]
        man.loc[m, "boundary_eligible"] = str(r["boundary_eligible"])
        if not r["boundary_eligible"]:
            man.loc[m & (man["exclusion_reason"].fillna("") == ""), "exclusion_reason"] = "no evaluable tumour–lung interface"
    write_table(man, man_path)
    write_provenance(STAGE, files, outputs + [sd_ / "section_label_summary.tsv", man_path], cfg,
                     extra={"eligible_sections": summ.loc[summ["boundary_eligible"], "section"].tolist()}, t_start=t0)


if __name__ == "__main__":
    main()
