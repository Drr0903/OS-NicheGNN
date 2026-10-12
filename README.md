# OS-NicheGNN

Code for *OS-NicheGNN: an interpretable mixture-of-experts framework for cross-patient spatial niche mapping in osteosarcoma*.

OS-NicheGNN is a multimodal spatial graph neural network for 10x Visium data. Every spot is described by its expression
and by the morphology of its H&E patch (Phikon embedding) and reads its surroundings in two ways: a ring-mean branch
encodes the spot with the mean profiles of its hexagonal rings, and a graph attention branch lets that encoding query the
individual neighbours of rings 1–3 through cross-attention with relative-position tokens. A mixture of two MLP experts,
weighted for every spot by a softmax router, fuses the two branches, and a linear skip of the ring means is added to the
256-dimensional latent. The network is trained to reconstruct masked spots and, through linear heads, the mean and
spread of each neighbourhood, with balance and entropy terms on the router. Niches are defined on this latent across
patients (cohort mode) and new patients are mapped onto them; a single-section mode with a local graph-infomax objective
delineates the spatial domains of one section.

## Installation

```bash
conda env create -f environment.yml        # Python: torch 2.5.1 (CUDA 12.1), PyG 2.6.1, scanpy 1.10.4
conda env create -f environment_r.yml      # R: Seurat 5, spacexr (RCTD deconvolution)
conda activate project018
```

## Data

All data are public. `scripts/stages/00_download.sh` fetches the osteosarcoma Visium atlas (GEO GSE293065) and the
single-cell references (GSE162454, GSE152048, HLCA core); `scripts/stages/13a_download_generality.sh` and
`scripts/stages/00_download_dlpfc.sh` fetch the colorectal cancer, breast cancer and DLPFC cohorts used for the
clustering comparison. Data are written to `data/`.

## Running

```bash
bash run_all.sh --resume      # every stage in order; a stage is skipped when its outputs exist
python scripts/stages/15_niche_atlas.py    # any stage can be run on its own
```

All parameters are in `config/run.yaml`. Each stage writes tables to `results/<stage>/` together with a
`PROVENANCE.json` (input and output checksums, seed, configuration hash, environment). Figure scripts in
`scripts/figures/` read only these tables and write their plotting inputs to `figures/source_data/`.

| stages | content |
|---|---|
| 00–05 | download, quality control, region reference labels, H&E features, RCTD deconvolution |
| 14, 23, 24 | leave-one-patient-out validation, architecture comparison, the trained encoder |
| 15, 17, 19, 20 | niche atlas, robustness, retraining, recurrence by patient |
| 16, 18, 21, 26 | remodelling between primaries and metastases, signalling, gradients |
| 25, 25b, 25c | niche transfer to held-out patients, simple representations, mapping onto a fixed atlas and component variants |
| 27a–f | clustering against pathologist annotation (colorectal, breast), DLPFC layers and osteosarcoma references |
| 28 | interpretation: attention, latent parts, perturbations, gene attribution |
| 29 | independent osteosarcoma cohort |
| 30, 31 | osteoclast and myeloid programmes, components of the interface |
| 32, 33 | component analysis (each part removed, single expert, concatenation) and niches without the image |
| 34 | expert routing: router weights by tissue, across the interface and from diagnosis to relapse |
| 35 | representative tumour–lung interfaces (Figure 5a) |

Model code: `scripts/lib/arch.py` (encoder: ring-mean and graph attention branches), `scripts/lib/arch_ext.py` (mixture-of-experts
fusion, concatenation variant, graph infomax), `scripts/lib/arch_train.py` (objectives and training).

## Trained network

`model/frozen_v5/` holds the five atlas networks trained on all 19 patients, its 2,000-gene list, scaler and the
expression and morphology PCA models. To embed new sections without retraining, copy it to `data/processed/frozen_v5/`
and run

```bash
python scripts/embed_new_sections.py --h5ad new_section.h5ad --img new_section_phikon.npy --out embeddings/
```

The AnnData needs `layers["lognorm"]`, `obsm["spatial"]` and the spot pitch (stage 03 output); the Phikon features come from
`scripts/stages/04c_image_features.py`. The 256-dimensional latents are then reduced, integrated and assigned to the atlas
niches as described in the Methods.

## Licence

MIT
