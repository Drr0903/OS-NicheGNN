"""Stage 13b — is patient identity a general property of unsupervised niches in multi-patient Visium data?

Two public datasets with pathologist spot annotations are analysed independently of osteosarcoma (own genes, own model):
  crc    Zenodo 7760264 — 14 Space Ranger sections, 7 patients (colorectal cancer; tumour / stroma / normal mucosa …)
  breast Zenodo 4739739 — 6 sections, 6 patients (breast cancer; spot "Classification" in the metadata)
Per dataset: QC → 2,000 HVGs → standardise → SectionGraphs → representations (standardised expression, PCA-50, BANKSY-20,
OS-NicheGNN ×5 seeds, view-contrastive variant ×5 seeds) → three clusterings at K = 6 (k-means, plain-kNN Leiden,
patient-balanced Leiden) → ARI vs patient / region, presence across patients, patient-identity index; plus the leave-one-patient
-out linear probe (macro-F1) for PCA, BANKSY and OS-NicheGNN. Labels are collapsed to tumour / stroma-immune / normal.
Outputs results/13_generality/{<dataset>_manifest.tsv, cluster_diagnostics.tsv, probe_lopo.tsv, label_mapping.tsv}
Usage: python 13b_generality.py [--datasets crc,breast] [--n_seeds N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lib.evalprobe as evalprobe  # noqa: E402
from lib.evalprobe import evaluate_heldout  # noqa: E402
from lib.generality import cluster_diagnostics, invariant_representations, make_graphs, qc_and_lognorm, representations  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402
from lib.train import embed_sections, train_model  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from lib.banksy import BanksyEmbedder  # noqa: E402

STAGE = "13_generality"; D = (ROOT / "data/generality"); CLASSES = ["tumour", "stroma_immune", "normal"]
CRC_ORDER = [("tumour", ["tumor", "tumour"]), ("stroma_immune", ["stroma", "connective", "ic aggre", "desmoplastic", "edema", "hemosiderin"]),
             ("normal", ["epithelium", "epitehlium", "submucosa", "lamina", "lam propria", "muscularis", "glandular", "squamous", "mucosa"])]


def coarse_label(raw: str, dataset: str) -> str:
    r = str(raw).strip().lower()
    if dataset == "crc":  # "tumor&stroma…" counts as tumour (tumour present); pure stroma / immune aggregates → stroma_immune; mucosal wall → normal
        for v, keys in CRC_ORDER:
            if any(k in r for k in keys):
                return v
        return "other"
    # breast (Wu et al. 2021 "Classification"): invasive cancer / DCIS → tumour; stroma / lymphocytes / adipose → stroma_immune; normal (+ stroma) → normal
    if any(k in r for k in ["artefact", "uncertain", "nan", "necrosis"]):
        return "other"
    if "normal" in r:
        return "normal"
    if "cancer" in r or "dcis" in r or "tumour" in r or "tumor" in r:
        return "tumour"
    if any(k in r for k in ["stroma", "lymphocyte", "adipose", "tls"]):
        return "stroma_immune"
    return "other"


def load_crc() -> tuple[dict[str, ad.AnnData], dict[str, str], pd.DataFrame]:
    ann_zip = D / "crc" / "Pathology_SpotAnnotations.zip"; ann = {}
    with zipfile.ZipFile(ann_zip) as z:
        for n in z.namelist():
            if n.lower().endswith(".csv") and not n.endswith("/"):
                df = pd.read_csv(z.open(n), sep=None, engine="python"); ann[Path(n).stem] = df
    adatas, patient_of, rows = {}, {}, []
    for zp in sorted((D / "crc").glob("SN*.zip")):
        sid = zp.stem; pat = sid.split("_")[1]
        with zipfile.ZipFile(zp) as z:
            base = sid + "/"; z.extractall(D / "crc" / "unz", members=[m for m in z.namelist() if m.startswith(base) and ("filtered_feature_bc_matrix.h5" in m or "spatial/tissue_positions" in m or "scalefactors" in m)])
        d = D / "crc" / "unz" / sid; a = sc.read_10x_h5(d / "filtered_feature_bc_matrix.h5"); a.var_names_make_unique()
        pos = pd.read_csv(d / "spatial" / "tissue_positions_list.csv", header=None, index_col=0); pos.columns = ["in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
        pos = pos.reindex(a.obs_names); a.obsm["spatial"] = pos[["pxl_col", "pxl_row"]].to_numpy(float)
        key = [k for k in ann if sid.replace("_X", "") in k or k in sid][0] if any(sid.replace("_X", "") in k or k in sid for k in ann) else None
        if key is None:
            log(f"  {sid}: no annotation file found among {list(ann)[:5]} … skipped"); continue
        df = ann[key]; bc = [c for c in df.columns if "barcode" in c.lower()][0]; lc = [c for c in df.columns if c != bc][0]
        lab = df.set_index(bc)[lc].reindex(a.obs_names); a.obs["raw_label"] = lab.astype(str).values; a.obs["region"] = [coarse_label(x, "crc") for x in a.obs["raw_label"]]
        a.obs["section"] = sid; a.obs["cohort"] = "generality"; adatas[sid] = a; patient_of[sid] = pat
        rows.append({"dataset": "crc", "section": sid, "patient": pat, "n_spots": a.n_obs, "annotation_column": lc})
    return adatas, patient_of, pd.DataFrame(rows)


def _open_maybe_gz(path: Path):
    import gzip
    with open(path, "rb") as fh:
        magic = fh.read(2)
    return gzip.open(path, "rt") if magic == b"\x1f\x8b" else open(path, "rt")


def read_mtx_dir(d: Path) -> ad.AnnData:
    """10x-style triplet (matrix / features|genes / barcodes) whose files may or may not really be gzipped."""
    from scipy.io import mmread
    mtx = next(iter(sorted(d.glob("matrix.mtx*")))); feat = next(iter(sorted(list(d.glob("features.tsv*")) + list(d.glob("genes.tsv*"))))); bcs = next(iter(sorted(d.glob("barcodes.tsv*"))))
    with _open_maybe_gz(mtx) as fh:
        X = sp.csr_matrix(mmread(fh).T)
    f = pd.read_csv(_open_maybe_gz(feat), sep="\t", header=None); b = pd.read_csv(_open_maybe_gz(bcs), sep="\t", header=None)
    sym = f.iloc[:, 1] if f.shape[1] > 1 else f.iloc[:, 0]
    a = ad.AnnData(X=X, obs=pd.DataFrame(index=b.iloc[:, 0].astype(str).values), var=pd.DataFrame(index=sym.astype(str).values)); a.var_names_make_unique(); return a


def load_breast() -> tuple[dict[str, ad.AnnData], dict[str, str], pd.DataFrame]:
    unz = D / "breast" / "unz"; unz.mkdir(exist_ok=True)
    for t in ["filtered_count_matrices", "spatial", "metadata"]:
        if not any(unz.glob(f"{t}*")):
            with tarfile.open(D / "breast" / f"{t}.tar.gz") as tf:
                tf.extractall(unz)
    adatas, patient_of, rows = {}, {}, []
    for mdir in sorted(glob.glob(str(unz / "filtered_count_matrices" / "*"))):
        sid = Path(mdir).name.replace("_filtered_count_matrix", "")
        try:
            a = read_mtx_dir(Path(mdir))
        except Exception as e:  # noqa: BLE001
            log(f"  {sid}: matrix folder not understood ({e!r}): {list(Path(mdir).glob('*'))[:5]}"); continue
        spdir = [p for p in (unz / "spatial").glob(f"*{sid}*")]
        pos_file = next(iter([p for d_ in spdir for p in Path(d_).rglob("tissue_positions_list.csv")]), None)
        if pos_file is None:
            log(f"  {sid}: positions not found"); continue
        pos = pd.read_csv(pos_file, header=None, index_col=0); pos.columns = ["in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"][: pos.shape[1]]
        pos = pos.reindex(a.obs_names); a = a[~pos["pxl_row"].isna().to_numpy()].copy(); pos = pos.reindex(a.obs_names); a.obsm["spatial"] = pos[["pxl_col", "pxl_row"]].to_numpy(float)
        meta_file = next(iter([p for p in (unz / "metadata").rglob(f"*{sid}*")]), None)
        if meta_file is None:
            log(f"  {sid}: metadata not found"); continue
        md = pd.read_csv(meta_file, index_col=0); lc = [c for c in md.columns if "classif" in c.lower() or "patholog" in c.lower()][0]
        a.obs["raw_label"] = md[lc].reindex(a.obs_names).astype(str).values; a.obs["region"] = [coarse_label(x, "breast") for x in a.obs["raw_label"]]
        a.obs["section"] = sid; a.obs["cohort"] = "generality"; adatas[sid] = a; patient_of[sid] = sid
        rows.append({"dataset": "breast", "section": sid, "patient": sid, "n_spots": a.n_obs, "annotation_column": lc})
    return adatas, patient_of, pd.DataFrame(rows)


def probe_lopo(G: dict, patient_of: dict, cfg: dict, device: str, n_seeds: int, dataset: str) -> list[dict]:
    """Leave-one-patient-out linear probe for PCA-50, BANKSY-20 and OS-NicheGNN (as in stage 07, one seed for the linear ones)."""
    evalprobe.CLASSES[:] = CLASSES
    secs = list(G); pats = sorted(set(patient_of.values())); rows = []
    for P in pats:
        te = [G[s] for s in secs if patient_of[s] == P]; tr = [G[s] for s in secs if patient_of[s] != P]
        with __import__("threadpoolctl").threadpool_limits(limits=8):
            pca = PCA(cfg["baselines"]["pca_dim"], random_state=cfg["seed"]).fit(np.vstack([g.x for g in tr])); emb = {g.section: pca.transform(g.x) for g in tr + te}
            m = evaluate_heldout(emb, tr, te, cfg, cfg["seed"]); m.pop("per_section_boundary", None); rows.append({"dataset": dataset, "method": "PCA-50", "seed": cfg["seed"], "heldout_patient": P, **m})
            b = cfg["baselines"]["banksy"]; be = BanksyEmbedder(b["lambda"], b["k_geom"], b["n_pcs"], cfg["seed"]).fit(tr); emb = {g.section: be.transform(g) for g in tr + te}
            m = evaluate_heldout(emb, tr, te, cfg, cfg["seed"]); m.pop("per_section_boundary", None); rows.append({"dataset": dataset, "method": "BANKSY-20", "seed": cfg["seed"], "heldout_patient": P, **m})
        for i in range(n_seeds):
            seed = cfg["seed"] + i; va = [tr[0]]; trn = tr[1:] if len(tr) > 1 else tr
            model, _ = train_model("nichegnn", trn, va, cfg, seed, device=device); emb = embed_sections(model, tr + te, "nichegnn", cfg, device)
            m = evaluate_heldout(emb, tr, te, cfg, seed); m.pop("per_section_boundary", None); rows.append({"dataset": dataset, "method": "OS-NicheGNN", "seed": seed, "heldout_patient": P, **m})
        log(f"  probe {dataset} held-out {P}: " + "; ".join(f"{r['method']} F1 {r['macro_f1']:.3f}" for r in rows[-(2 + n_seeds):]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", default="crc,breast"); ap.add_argument("--n_seeds", type=int, default=0)
    ap.add_argument("--append_invariant", action="store_true", help="only add the A7 variant's cluster diagnostics to an existing cluster_diagnostics.tsv"); args = ap.parse_args()
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); n_seeds = args.n_seeds or cfg["n_seeds"]; sd = stage_dir(STAGE)
    def flush(name: str, rows: list, ds: str) -> None:
        f = sd / f"{name}.tsv"; new = pd.DataFrame(rows)
        if f.exists():
            old = pd.read_csv(f, sep="\t"); new = pd.concat([old[old["dataset"] != ds], new], ignore_index=True)
        write_table(new, f)
    for ds in args.datasets.split(","):
        diag, probe, mani, maps = [], [], [], []
        log(f"=== {ds}"); adatas, patient_of, man = load_crc() if ds == "crc" else load_breast()
        for s in list(adatas):
            adatas[s] = qc_and_lognorm(adatas[s])
        maps += [{"dataset": ds, "raw_label": k, "coarse": v, "n_spots": int(n)} for (k, v), n in pd.concat([a.obs[["raw_label", "region"]] for a in adatas.values()]).value_counts().items()]
        G, genes = make_graphs(adatas, patient_of, cfg); mani.append(man.assign(n_spots_qc=[G[s].n for s in man["section"]], n_hvg=len(genes)))
        if args.append_invariant:
            reps = invariant_representations(G, cfg, cfg["device"], n_seeds, log); rows = []
            for name, emb in reps.items():
                for r in cluster_diagnostics(name, emb, G, cfg["niches"]["k_fixed"], cfg["seed"], CLASSES):
                    r["dataset"] = ds; rows.append(r)
                log(f"  {name}: " + "; ".join(f"{r['clustering']} PII {r['pii']:.2f} (pat {r['ari_vs_patient']:.2f} / reg {r['ari_vs_region']:.2f}, present {r['n_clusters_present_in_half_patients']})" for r in rows[-3:]))
            f = sd / "cluster_diagnostics.tsv"; old = pd.read_csv(f, sep="\t"); old = old[~((old["dataset"] == ds) & old["representation"].str.contains("invariant"))]
            write_table(pd.concat([old, pd.DataFrame(rows)], ignore_index=True), f); continue
        log(f"  {len(G)} sections, {len(set(patient_of.values()))} patients, {sum(g.n for g in G.values())} spots after QC; label counts {pd.Series(np.concatenate([g.labels for g in G.values()])).value_counts().to_dict()}")
        reps = representations(G, cfg, cfg["device"], n_seeds, cfg["upgrade"]["lambda_con_grid"][-1], log)
        for name, emb in reps.items():
            for r in cluster_diagnostics(name, emb, G, cfg["niches"]["k_fixed"], cfg["seed"], CLASSES):
                r["dataset"] = ds; diag.append(r)
            log(f"  {name}: " + "; ".join(f"{r['clustering']} PII {r['pii']:.2f} (pat {r['ari_vs_patient']:.2f} / reg {r['ari_vs_region']:.2f}, present {r['n_clusters_present_in_half_patients']})" for r in diag[-3:]))
        flush("cluster_diagnostics", diag, ds); flush("label_mapping", maps, ds); flush("manifest", pd.concat(mani, ignore_index=True).to_dict("records"), ds)
        probe += probe_lopo(G, patient_of, cfg, cfg["device"], n_seeds, ds); flush("probe_lopo", probe, ds)
    write_provenance(STAGE, [D / "CHECKSUMS.md5"], [sd / "cluster_diagnostics.tsv", sd / "probe_lopo.tsv"], cfg, extra={"amendment": "A6", "datasets": args.datasets.split(",")}, t_start=t0)


if __name__ == "__main__":
    main()
