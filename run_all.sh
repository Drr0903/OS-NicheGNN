#!/usr/bin/env bash
# project018 master orchestrator.  Usage: bash run_all.sh [--resume|--force] [--skip=05b,10]
# Each stage is skipped under --resume when its declared output exists. Logs → logs/<stage>.log
set -uo pipefail
cd "$(dirname "$0")"
MODE=resume; SKIP=""
for a in "$@"; do case $a in --force) MODE=force;; --resume) MODE=resume;; --skip=*) SKIP=${a#--skip=};; esac; done
CONDA=${CONDA_BASE:-$(conda info --base)}
export OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 TQDM_DISABLE=1
PY=$CONDA/envs/project018/bin/python
RS=$CONDA/envs/project018_r/bin/Rscript
mkdir -p logs
run() {  # run <stage-name> <output-file-to-check> <command...>
  local name=$1 out=$2; shift 2
  if [[ ",$SKIP," == *",$name,"* ]]; then echo "[skip:flag] $name"; return; fi
  if [[ $MODE == resume && -e $out ]]; then echo "[skip:done] $name ($out)"; return; fi
  echo "[$(date +%T)] ▶ $name"
  if "$@" > "logs/$name.log" 2>&1; then echo "[$(date +%T)] ✔ $name"; else echo "[$(date +%T)] ✘ $name — see logs/$name.log"; exit 1; fi
}
run 00_download            data/raw/GSE293065/extracted                       bash scripts/stages/00_download.sh
run 01_manifest            results/01_manifest/PROVENANCE.json                $PY scripts/stages/01_manifest_and_load.py
run 03_qc                  results/03_qc/PROVENANCE.json                      $PY scripts/stages/03_qc_normalize.py
run 05a_reference          results/05_deconvolution_reference/PROVENANCE.json $PY scripts/stages/05a_build_reference.py
run 05c_rctd_input         results/05_deconvolution_input/PROVENANCE.json     $PY scripts/stages/05c_prepare_rctd_input.py
run 05b_rctd               results/05_deconvolution/weights/LR19_rctd_weights.tsv $RS scripts/stages/05b_rctd.R
run 04_histology           results/04_histology/PROVENANCE.json               $PY scripts/stages/04_histology_labels.py
run 04b_panel_exports      results/04b_panel_exports/PROVENANCE.json          $PY scripts/stages/04b_panel_exports.py
# ---- plan v2 (all 26 sections, 19 patients; stages 07–13 of plan v1 are archived and not part of the v2 build) ----
run 14_lopo                results/14_model/lopo_metrics.tsv                  $PY scripts/stages/14_model_all_sites.py --lopo --methods raw,pca,banksy,pca_harmony,banksy_harmony,nichegnn,nichegnn_invariant,nichegnn_hybrid_invariant,nichegnn_invariant_anchoronly,mlp_invariant_anchoronly
run 14b_variant_select     results/14_model/variant_choice.json               $PY scripts/stages/14b_variant_select.py --variants gnn,invariant,invariant_gated,hybrid_invariant,hybrid_invariant_gated,invariant_anchoronly,invariant_nbronly,invariant_w0.1,invariant_w0.25,raw,pca,banksy,pca_harmony,banksy_harmony,gnn_harmonised,invariant_anchoronly_harmonised,mlp_invariant_anchoronly
run 14_final               data/processed/frozen_v2/meta.json                 $PY scripts/stages/14_model_all_sites.py --final
run 15_niches              results/15_niches/PROVENANCE.json                  $PY scripts/stages/15_niche_atlas.py
run 16_remodelling         results/16_remodelling/PROVENANCE.json             $PY scripts/stages/16_remodelling.py
run 17_robustness          results/17_robustness/PROVENANCE.json              $PY scripts/stages/17_representation_robustness.py
run 18_signalling          results/18_signalling/PROVENANCE.json              $PY scripts/stages/18_niche_signalling.py
run 19_retrain_sens        results/19_retrain_sensitivity/PROVENANCE.json     $PY scripts/stages/19_retrain_sensitivity.py
run 20_recurrence_unit     results/20_recurrence_unit/PROVENANCE.json         $PY scripts/stages/20_recurrence_unit.py
run 21_smad_band          results/21_smad_band/PROVENANCE.json               $PY scripts/stages/21_smad_band_check.py
for f in scripts/figures/v2_Fig*.py; do n=$(basename "$f" .py); n=${n#v2_}; run "fig_$n" "figures/$n.pdf" $PY "$f"; done
$PY scripts/build_supp_tables.py && $PY scripts/check_numbers.py && $PY scripts/build_manuscript.py && $PY scripts/build_submission.py
echo "[$(date +%T)] all stages finished"
