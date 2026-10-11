#!/usr/bin/env Rscript
# Stage 05b — RCTD (spacexr) full-mode deconvolution of every QC-passed section against the coarse reference.
# Inputs : data/processed/reference/{reference_counts.mtx,reference_genes.tsv,reference_cell_types.tsv}
#          data/processed/rctd_input/<sec>/{counts.mtx,genes.tsv,barcodes.tsv,coords.tsv}   (written by 05c_prepare_rctd_input.py)
# Outputs: results/05_deconvolution/weights/<sec>_rctd_weights.tsv   normalised weights per spot (rows sum to 1)
# Non-integer SpotClean counts of the external cohort are rounded, as RCTD models Poisson counts.
suppressPackageStartupMessages({library(spacexr); library(Matrix); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "..", ".."))
ref_dir <- file.path(root, "data/processed/reference"); in_root <- file.path(root, "data/processed/rctd_input")
out_dir <- file.path(root, "results/05_deconvolution/weights"); dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
n_cores <- as.integer(Sys.getenv("RCTD_CORES", "16"))
t_start <- Sys.time()
set.seed(20260914)

cnt <- as(readMM(file.path(ref_dir, "reference_counts.mtx")), "CsparseMatrix")
genes <- readLines(file.path(ref_dir, "reference_genes.tsv"))
ct <- read.delim(file.path(ref_dir, "reference_cell_types.tsv"), row.names = 1, check.names = FALSE)
rownames(cnt) <- genes; colnames(cnt) <- rownames(ct)
cell_types <- factor(ct$coarse_type); names(cell_types) <- rownames(ct)
nUMI <- colSums(cnt)
reference <- Reference(cnt, cell_types, nUMI, min_UMI = 100)
cat("reference:", ncol(cnt), "cells,", nrow(cnt), "genes,", nlevels(cell_types), "types\n")

sections <- if (length(args)) args else list.dirs(in_root, full.names = FALSE, recursive = FALSE)
for (s in sections) {
  outf <- file.path(out_dir, paste0(s, "_rctd_weights.tsv"))
  if (file.exists(outf)) { cat("skip", s, "\n"); next }
  d <- file.path(in_root, s)
  m <- as(readMM(file.path(d, "counts.mtx")), "CsparseMatrix")   # genes × spots
  m@x <- round(m@x)
  rownames(m) <- readLines(file.path(d, "genes.tsv")); colnames(m) <- readLines(file.path(d, "barcodes.tsv"))
  co <- read.delim(file.path(d, "coords.tsv"), row.names = 1)
  puck <- SpatialRNA(co[colnames(m), c("x", "y")], m, colSums(m))
  cat("[", format(Sys.time(), "%H:%M:%S"), "]", s, ":", ncol(m), "spots\n")
  rctd <- create.RCTD(puck, reference, max_cores = n_cores, UMI_min = 50, UMI_min_sigma = 100, CELL_MIN_INSTANCE = 20)
  rctd <- run.RCTD(rctd, doublet_mode = "full")
  w <- as.matrix(normalize_weights(rctd@results$weights))
  out <- data.frame(barcode = rownames(w), w, check.names = FALSE)
  write.table(out, outf, sep = "\t", quote = FALSE, row.names = FALSE)
}
# ---- provenance (same schema as scripts/lib/repro.py: a stage without PROVENANCE.json counts as not run) ----
hash_many <- function(paths) {
  paths <- paths[file.exists(paths)]
  if (!length(paths)) return(structure(list(), names = character(0)))
  setNames(lapply(paths, function(f) list(md5 = unname(tools::md5sum(f)), bytes = as.numeric(file.info(f)$size))),
           sub(paste0("^", root, "/"), "", paths))
}
git <- function(a) tryCatch(system2("git", c("-C", root, a), stdout = TRUE, stderr = FALSE)[1], error = function(e) NA_character_)
cfg_path <- file.path(root, "config/run.yaml")
si <- sessionInfo()
prov <- list(
  stage = "05_deconvolution",
  finished_utc = format(as.POSIXct(Sys.time(), tz = "UTC"), "%Y-%m-%dT%H:%M:%SZ"),
  elapsed_s = round(as.numeric(difftime(Sys.time(), t_start, units = "secs")), 1),
  seed = 20260914L,
  config_path = cfg_path,
  config_sha256 = unname(tools::md5sum(cfg_path)),  # md5 of the config; the Python stages record sha256
  git_commit = git("rev-parse HEAD"),
  git_dirty = length(system2("git", c("-C", root, "status", "--porcelain"), stdout = TRUE)) > 0,
  inputs = hash_many(c(file.path(ref_dir, c("reference_counts.mtx", "reference_genes.tsv", "reference_cell_types.tsv")),
                       list.files(in_root, pattern = "counts.mtx$", full.names = TRUE, recursive = TRUE))),
  outputs = hash_many(list.files(out_dir, pattern = "_rctd_weights.tsv$", full.names = TRUE)),
  environment = list(r_version = si$R.version$version.string, platform = si$platform,
                     packages = sapply(si$otherPkgs, function(p) as.character(p$Version))),
  extra = list(doublet_mode = "full", UMI_min = 50, UMI_min_sigma = 100, CELL_MIN_INSTANCE = 20, max_cores = n_cores)
)
writeLines(jsonlite::toJSON(prov, auto_unbox = TRUE, pretty = TRUE, null = "null"),
           file.path(root, "results/05_deconvolution/PROVENANCE.json"))
cat("done\n")
