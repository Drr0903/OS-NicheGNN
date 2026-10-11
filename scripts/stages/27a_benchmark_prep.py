"""Stage 27a () — pathologist-annotated benchmark cohorts: ground-truth classes and H&E features.

Cohorts (downloaded by stage 13a): colorectal cancer (Zenodo 7760264, 14 sections / 7 patients) and breast cancer
(Zenodo 4739739, 6 sections / 6 patients). Per section: the stage-13b reader, the stage-03 QC rule, the ground-truth class
of A21 from the pathologist's spot label (unscored labels → "unscored"), and Phikon features of the hires H&E image by the
stage-04c procedure (patch of `image.patch_spots` spot diameters, CLS token).
Outputs <benchmark_dir>/<cohort>/<section>.h5ad (QC'd counts + lognorm, obs: patient, raw_label, truth) and
        <benchmark_dir>/<cohort>/<section>_img.npy (rows aligned); results/27_benchmark/{prep_manifest, truth_mapping}.tsv
"""
from __future__ import annotations

import json
import sys
import time
import zipfile
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.generality import qc_and_lognorm  # noqa: E402
from lib.repro import ROOT, load_config, log, set_global_seed, stage_dir, write_provenance, write_table  # noqa: E402

STAGE = "27_benchmark"; s13 = import_module("13b_generality"); s04c = import_module("04c_image_features")
D = s13.D; OUT = (ROOT / "data/benchmark")
Image.MAX_IMAGE_PIXELS = None


def truth_crc(raw: str) -> str:
    r = str(raw).strip().lower()
    if r in ("", "nan") or "exclude" in r:
        return "unscored"
    if r.startswith("tumor&stroma") or r.startswith("tumour&stroma"):
        return "tumour & stroma"
    if "tumor" in r or "tumour" in r:
        return "tumour"
    if r.startswith("stroma"):
        return "stroma"
    if any(k in r for k in ["epithel", "epitehl", "lamina", "lam propria", "glandular"]):
        return "normal mucosa"
    if any(k in r for k in ["submucosa", "connective", "muscularis", "edema", "hemosiderin"]):
        return "wall / connective tissue"
    return "unscored"


def truth_breast(raw: str) -> str:
    r = str(raw).strip().lower()
    if r in ("", "nan") or any(k in r for k in ["artefact", "uncertain"]):
        return "unscored"
    if "invasive cancer" in r or "cancer trapped" in r:
        return "invasive cancer"
    if "dcis" in r:
        return "DCIS"
    if "necrosis" in r:
        return "necrosis"
    if "normal" in r:
        return "normal glands / ducts"
    if "lymphocyte" in r or "tls" in r:
        return "lymphocytes / TLS"
    if "stroma" in r or "adipose" in r:
        return "stroma / adipose"
    return "unscored"


def hires(cohort: str, sid: str) -> tuple[Path, dict]:
    """Path of the hires H&E image and the Space Ranger scale factors of one section."""
    if cohort == "crc":
        d = D / "crc" / "unz" / sid / "spatial"
        if not (d / "tissue_hires_image.png").exists():
            with zipfile.ZipFile(D / "crc" / f"{sid}.zip") as z:
                z.extractall(D / "crc" / "unz", members=[m for m in z.namelist() if m.endswith(("tissue_hires_image.png", "scalefactors_json.json"))])
    else:
        d = next(p for p in (D / "breast" / "unz" / "spatial").glob(f"*{sid}*"))
        d = next(iter([p.parent for p in d.rglob("tissue_hires_image.png")]))
    return d / "tissue_hires_image.png", json.load(open(d / "scalefactors_json.json"))


def main() -> None:
    from transformers import ViTModel
    t0 = time.time(); cfg = load_config(); set_global_seed(cfg["seed"]); ic = cfg["image"]; sd = stage_dir(STAGE)
    model = ViTModel.from_pretrained(ic["model"], revision=ic["revision"], add_pooling_layer=False).to(cfg["device"]).eval()
    man, maps, outs = [], [], []
    for cohort, loader, truth in [("crc", s13.load_crc, truth_crc), ("breast", s13.load_breast, truth_breast)]:
        adatas, patient_of, _ = loader(); od = OUT / cohort; od.mkdir(parents=True, exist_ok=True)
        for sid, a in adatas.items():
            a = qc_and_lognorm(a); a.obs["patient"] = patient_of[sid]; a.obs["truth"] = [truth(x) for x in a.obs["raw_label"]]
            ip, sf = hires(cohort, sid); a.uns["scalefactors"] = sf
            img = np.asarray(Image.open(ip).convert("RGB")); side = int(round(ic["patch_spots"] * sf["spot_diameter_fullres"] * sf["tissue_hires_scalef"]))
            F = s04c.section_features(model, a, img, side, cfg["device"]).astype(np.float32)
            for c in ("region",):
                a.obs.pop(c) if c in a.obs else None
            a.write_h5ad(od / f"{sid}.h5ad"); np.save(od / f"{sid}_img.npy", F); outs += [od / f"{sid}.h5ad", od / f"{sid}_img.npy"]
            vc = a.obs["truth"].value_counts()
            man.append({"cohort": cohort, "section": sid, "patient": patient_of[sid], "n_spots_qc": a.n_obs, "patch_px": side, **{f"n_{k}": int(v) for k, v in vc.items()}})
            maps += [{"cohort": cohort, "raw_label": k, "truth": truth(k), "n_spots": int(v)} for k, v in a.obs["raw_label"].value_counts().items()]
            log(f"{cohort} {sid}: {a.n_obs} spots, patch {side}px, truth {vc.to_dict()}")
    mp = pd.DataFrame(maps).groupby(["cohort", "raw_label", "truth"], as_index=False)["n_spots"].sum().sort_values(["cohort", "truth", "n_spots"], ascending=[True, True, False])
    write_table(pd.DataFrame(man), sd / "prep_manifest.tsv"); write_table(mp, sd / "truth_mapping.tsv")
    write_provenance(STAGE + "_prep", [D / "CHECKSUMS.md5"], outs + [sd / "prep_manifest.tsv", sd / "truth_mapping.tsv"], cfg, extra={"amendment": "A21"}, t_start=t0)


if __name__ == "__main__":
    main()
