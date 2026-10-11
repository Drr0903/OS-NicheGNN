#!/usr/bin/env bash
# human DLPFC Visium (spatialLIBD; Maynard et al. 2021) — counts, hires image, positions, scale factors, layer map.
set -uo pipefail
D="$(cd "$(dirname "$0")/../.." && pwd)"/data/dlpfc; mkdir -p "$D"
S3=https://spatial-dlpfc.s3.us-east-2.amazonaws.com; GH=https://raw.githubusercontent.com/LieberInstitute/HumanPilot/master/10X
dl() { [ -s "$1" ] || { curl -sSL --retry 3 -o "$1.part" "$2" && mv "$1.part" "$1"; }; }
dl "$D/barcode_level_layer_map.tsv" "$GH/barcode_level_layer_map.tsv"
for s in 151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676; do
  mkdir -p "$D/$s"
  dl "$D/$s/filtered_feature_bc_matrix.h5" "$S3/h5/${s}_filtered_feature_bc_matrix.h5" &
  dl "$D/$s/tissue_hires_image.png" "$GH/$s/tissue_hires_image.png" &
  dl "$D/$s/tissue_positions_list.txt" "$GH/$s/tissue_positions_list.txt" &
  dl "$D/$s/scalefactors_json.json" "$GH/$s/scalefactors_json.json" &
  wait
done
cd "$D" && md5sum barcode_level_layer_map.tsv */* > CHECKSUMS.md5
echo "DLPFC_DONE $(ls -d 15* | wc -l) sections"
