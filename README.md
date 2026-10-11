# OS-NicheGNN

Code for *Spatial graph neural network identification of osteosarcoma metastatic niches and microenvironment remodelling*.

OS-NicheGNN is a multimodal spatial graph attention network for 10x Visium data. Every spot is described by its
expression and by the morphology of its H&E patch (Phikon embedding), queries its neighbours in hexagonal rings 1–3
through cross-attention with relative-position tokens, and is encoded into a latent that is trained to reconstruct
masked spots and, through linear heads, the mean and spread of its neighbourhood. Niches are defined on this latent
across patients (cohort mode) and new patients are assigned to them; a single-section mode with a local graph-infomax
objective delineates the spatial domains of one section.

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
| 25 | niche transfer to held-out patients |
| 27a–f | clustering against pathologist annotation (colorectal, breast), DLPFC layers and osteosarcoma references |
| 28 | interpretation: attention, latent parts, perturbations, gene attribution |
| 29 | independent osteosarcoma cohort |

Model code: `scripts/lib/arch.py` (encoder), `scripts/lib/arch_ext.py` (graph infomax, cross-modal alignment, mixture
of experts), `scripts/lib/arch_train.py` (objectives and training).

## Licence

MIT
