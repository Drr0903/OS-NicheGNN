#!/usr/bin/env bash
# public multi-patient Visium datasets with pathologist spot annotations (parallel download + checksum list).
set -uo pipefail
D="$(cd "$(dirname "$0")/../.." && pwd)"/data/generality; mkdir -p "$D/crc" "$D/breast"
dl() { local out=$1 url=$2; if [ ! -s "$out" ]; then curl -sSL --retry 3 -o "$out.part" "$url" && mv "$out.part" "$out"; fi; }
export -f dl
{ for f in SN048_A121573_Rep1 SN048_A121573_Rep2 SN048_A416371_Rep1 SN048_A416371_Rep2 SN123_A551763_Rep1 SN123_A595688_Rep1 SN123_A798015_Rep1 SN123_A938797_Rep1_X SN124_A551763_Rep2 SN124_A595688_Rep2 SN124_A798015_Rep2 SN124_A938797_Rep2 SN84_A120838_Rep1 SN84_A120838_Rep2 Pathology_SpotAnnotations; do
    echo "$D/crc/$f.zip https://zenodo.org/api/records/7760264/files/$f.zip/content"; done
  for f in filtered_count_matrices spatial metadata; do echo "$D/breast/$f.tar.gz https://zenodo.org/api/records/4739739/files/$f.tar.gz/content"; done; } | xargs -P 6 -n 2 bash -c 'dl "$0" "$1"'
cd "$D" && md5sum crc/*.zip breast/*.tar.gz > CHECKSUMS.md5
echo "DOWNLOAD_DONE $(ls crc/*.zip | wc -l) crc files, $(ls breast/*.tar.gz | wc -l) breast files"
