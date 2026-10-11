#!/usr/bin/env bash
# Stage 00 — fetch every public input of project018 and record checksums.
# Idempotent: existing complete files are skipped (curl -C - resumes partial ones).
# Layout: data/raw/<source>/...   ; checksums: metadata/checksums_raw.sha256
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)
RAW="$HERE/data/raw"
META="$HERE/metadata"
mkdir -p "$RAW" "$META"
LOG="$HERE/logs/00_download.log"; mkdir -p "$HERE/logs"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Is)] 00_download start"

fetch() {  # fetch <url> <dest>
  local url=$1 dest=$2
  if [ -s "$dest" ] && [ ! -f "$dest.part" ]; then echo "skip  $dest"; return; fi
  echo "fetch $url"
  touch "$dest.part"
  curl -L --fail --retry 8 --retry-delay 15 -C - -o "$dest" "$url"
  rm -f "$dest.part"
}

# ---- 1. GSE293065 (Visium V1 FFPE, 26 sections; discovery cohort) --------------
G=$RAW/GSE293065; mkdir -p "$G"
fetch https://ftp.ncbi.nlm.nih.gov/geo/series/GSE293nnn/GSE293065/suppl/GSE293065_RAW.tar "$G/GSE293065_RAW.tar"
fetch https://ftp.ncbi.nlm.nih.gov/geo/series/GSE293nnn/GSE293065/suppl/GSE293065_ica.cell.embeddings.all.csv.gz "$G/GSE293065_ica.cell.embeddings.all.csv.gz"
fetch https://ftp.ncbi.nlm.nih.gov/geo/series/GSE293nnn/GSE293065/suppl/GSE293065_ica.feature.loadings.all.csv.gz "$G/GSE293065_ica.feature.loadings.all.csv.gz"
fetch https://ftp.ncbi.nlm.nih.gov/geo/series/GSE293nnn/GSE293065/suppl/filelist.txt "$G/filelist.txt"
fetch "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE293065&targ=gsm&form=text&view=full" "$G/GSE293065_gsm_full.soft"
if [ ! -d "$G/extracted" ]; then mkdir -p "$G/extracted"; tar -xf "$G/GSE293065_RAW.tar" -C "$G/extracted"; fi

# ---- 2. Zenodo 14594702 (Visium CytAssist FFPE, OS1-OS9; external cohort) --------
Z=$RAW/zenodo_14594702; mkdir -p "$Z"
curl -sL https://zenodo.org/api/records/14594702 -o "$Z/record.json"
for s in OS1 OS2 OS3 OS4 OS5 OS6 OS7 OS8 OS9; do
  fetch "https://zenodo.org/api/records/14594702/files/$s.rds/content" "$Z/$s.rds"
done

# ---- 3. GSE162454 (6 primary osteosarcoma scRNA-seq; reference) ----------------
S=$RAW/GSE162454; mkdir -p "$S"
fetch https://ftp.ncbi.nlm.nih.gov/geo/series/GSE162nnn/GSE162454/suppl/GSE162454_RAW.tar "$S/GSE162454_RAW.tar"
if [ ! -d "$S/extracted" ]; then mkdir -p "$S/extracted"; tar -xf "$S/GSE162454_RAW.tar" -C "$S/extracted"; fi

# ---- 4. GSE152048 (11 osteosarcoma scRNA-seq lesions; reference) ---------------
T=$RAW/GSE152048; mkdir -p "$T"
for bc in BC2 BC3 BC5 BC6 BC10 BC11 BC16 BC17 BC20 BC21 BC22; do
  fetch "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE152nnn/GSE152048/suppl/GSE152048_${bc}.matrix.tar.gz" "$T/GSE152048_${bc}.matrix.tar.gz"
done
fetch "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE152048&targ=gsm&form=text&view=full" "$T/GSE152048_gsm_full.soft"

# ---- 4b. Human Lung Cell Atlas core (CELLxGENE, CC-BY-4.0; normal lung reference) --
H=$RAW/HLCA; mkdir -p "$H"
curl -sL https://api.cellxgene.cziscience.com/curation/v1/collections/6f6d381a-7701-4781-935c-db10d30de293 -o "$H/collection.json"
fetch https://datasets.cellxgene.cziscience.com/688185ad-11c2-4172-a53a-f4f1f4076860.h5ad "$H/hlca_core.h5ad"

# ---- 5. checksums ---------------------------------------------------------------
( cd "$RAW" && find . -type f \( -name '*.tar' -o -name '*.rds' -o -name '*.tar.gz' -o -name '*.h5ad' \) -print0 \
  | sort -z | xargs -0 sha256sum ) > "$META/checksums_raw.sha256"
echo "[$(date -Is)] 00_download done"
