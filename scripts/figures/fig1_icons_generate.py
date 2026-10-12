"""Illustrated icons for Figure 1, generated once with an image model and stored as transparent PNGs.

Only pictograms are generated (no text, no numbers); layout, arrows, the network diagram, labels and equations are drawn
in v2_Fig1_overview.py, so all text in the figure stays vector and editable. The prompts and the files are kept in
figures/source_data/Fig1_icons/, so the figure is redrawn from them without calling the model again.

    GPTIMAGE_KEY=... python scripts/figures/fig1_icons_generate.py [--only femur lungs ...]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "figures" / "source_data" / "Fig1_icons"
BASE = "https://cpa.onelyi.com/v1"
STYLE = ("Single scientific icon in BioRender style: clean illustration with soft flat shading and gentle highlights, "
         "thin dark outline, muted professional palette, centred with a small margin, no text, no letters, no numbers, "
         "no labels, no shadow on the ground, isolated on a fully transparent background.")
ICONS = {
    "femur": "A human femur bone, vertical, with an osteosarcoma tumour mass growing at the distal metaphysis near the knee, "
             "bone in ivory, tumour in muted purple-red.",
    "lungs": "A pair of human lungs with trachea, front view, soft pink, with three small purple-red metastatic tumour nodules.",
    "lymph_node": "A single bean-shaped lymph node with afferent vessels, soft green, with one small purple-red tumour nodule.",
    "heart": "An anatomical human heart, front view, muted red, with one small purple-red tumour nodule on its wall.",
    "soft_tissue": "A block of soft tissue cut in cross-section showing skin, fat and muscle layers, with one purple-red tumour nodule.",
    "patients": "Three simple human bust silhouettes side by side, neutral grey-blue, representing patients.",
    "chemo": "An intravenous chemotherapy infusion bag with tubing, pale yellow liquid, representing treatment.",
    "visium": "A glass microscope slide with a square capture area containing a regular grid of small round spots over a "
              "pink and purple haematoxylin and eosin stained tumour tissue section.",
    "he_patch": "A square microscopy image tile of haematoxylin and eosin stained osteosarcoma: pleomorphic purple tumour cells "
                "with pink osteoid matrix, rounded corners.",
    "slides_stack": "A small stack of four glass microscope slides with stained tissue sections, slightly fanned, representing "
                    "independent tissue datasets.",
    "macrophage": "A single macrophage immune cell with irregular extended pseudopodia, soft orange-pink cytoplasm and a "
                  "kidney-shaped nucleus.",
    "t_cell": "A single round T lymphocyte with small surface receptors, soft teal-blue, large round nucleus.",
    "tumour_cell": "A single large pleomorphic osteosarcoma tumour cell, muted purple, irregular nucleus.",
    "fibroblast": "A single spindle-shaped fibroblast cell, elongated, soft yellow-beige with an oval nucleus.",
    "osteoclast": "A single large multinucleated osteoclast cell with a ruffled border, soft lilac with several nuclei.",
}


def generate(name: str, key: str) -> str:
    prompt = f"{STYLE} Subject: {ICONS[name]}"
    body = json.dumps({"model": "gpt-image-2", "prompt": prompt, "size": "1024x1024", "background": "transparent",
                       "quality": "high", "n": 1}).encode()
    req = urllib.request.Request(f"{BASE}/images/generations", data=body,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    (OUT / f"{name}.png").write_bytes(base64.b64decode(d["data"][0]["b64_json"]))
    return prompt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*")
    a = ap.parse_args(); key = os.environ["GPTIMAGE_KEY"]; OUT.mkdir(parents=True, exist_ok=True)
    names = a.only or list(ICONS)
    with ThreadPoolExecutor(6) as ex:
        prompts = dict(zip(names, ex.map(lambda n: generate(n, key), names)))
    rec = OUT / "prompts.json"
    old = json.loads(rec.read_text()) if rec.exists() else {}
    rec.write_text(json.dumps({**old, **prompts}, indent=1))
    print("generated:", ", ".join(names))


if __name__ == "__main__":
    main()
