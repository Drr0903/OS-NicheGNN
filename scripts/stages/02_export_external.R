#!/usr/bin/env Rscript
# Stage 02a — export the external Visium CytAssist Seurat objects (Zenodo 14594702, OS1–OS9)
# to plain files that Python can read without R:
#   data/processed/external_export/<OS>/counts.mtx        Spatial assay "counts" (SpotClean-decontaminated, non-integer)
#   data/processed/external_export/<OS>/genes.tsv, barcodes.tsv
#   data/processed/external_export/<OS>/meta.tsv          all meta.data columns (nCount, contamination_rate, seurat_clusters)
#   data/processed/external_export/<OS>/coords.tsv        imagerow/imagecol (image pixel coordinates at stored-image scale)
#   data/processed/external_export/<OS>/scalefactors.json
#   data/processed/external_export/<OS>/image.png         the stored H&E image (whatever resolution the object carries)
#   results/02_external_export/external_object_summary.tsv
suppressPackageStartupMessages({library(Seurat); library(Matrix); library(jsonlite); library(png)})
args <- commandArgs(trailingOnly = TRUE)
root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "..", ".."))
raw <- file.path(root, "data/raw/zenodo_14594702")
out_root <- file.path(root, "data/processed/external_export"); dir.create(out_root, showWarnings = FALSE, recursive = TRUE)
res_dir <- file.path(root, "results/02_external_export"); dir.create(res_dir, showWarnings = FALSE, recursive = TRUE)
sections <- if (length(args)) args else paste0("OS", 1:9)
rows <- list()
for (s in sections) {
  f <- file.path(raw, paste0(s, ".rds")); if (!file.exists(f)) { message("missing ", f); next }
  message("[", format(Sys.time(), "%H:%M:%S"), "] ", s)
  o <- readRDS(f)
  od <- file.path(out_root, s); dir.create(od, showWarnings = FALSE, recursive = TRUE)
  cnt <- GetAssayData(o, assay = "Spatial", layer = "counts")
  writeMM(cnt, file.path(od, "counts.mtx"))
  writeLines(rownames(cnt), file.path(od, "genes.tsv"))
  writeLines(colnames(cnt), file.path(od, "barcodes.tsv"))
  md <- o@meta.data; md$barcode <- rownames(md)
  write.table(md, file.path(od, "meta.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
  img_name <- names(o@images)[1]
  co <- GetTissueCoordinates(o, image = img_name); co$barcode <- rownames(co)
  write.table(co, file.path(od, "coords.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
  sf <- ScaleFactors(o@images[[img_name]])
  img <- GetImage(o@images[[img_name]], mode = "raw")
  writePNG(img, file.path(od, "image.png"))
  write_json(list(spot = sf$spot, fiducial = sf$fiducial, hires = sf$hires, lowres = sf$lowres,
                  image_dim = dim(img)[1:2], seurat_version = as.character(o@version),
                  assays = names(o@assays), counts_integer = all(cnt@x == round(cnt@x))),
             file.path(od, "scalefactors.json"), auto_unbox = TRUE, pretty = TRUE)
  rows[[s]] <- data.frame(section = s, n_spots = ncol(cnt), n_genes = nrow(cnt), counts_integer = all(cnt@x == round(cnt@x)),
                          counts_max = max(cnt@x), median_nCount_Spatial = median(md$nCount_Spatial),
                          median_contamination_rate = if ("contamination_rate" %in% names(md)) median(md$contamination_rate) else NA,
                          n_seurat_clusters = if ("seurat_clusters" %in% names(md)) length(unique(md$seurat_clusters)) else NA,
                          image_rows = dim(img)[1], image_cols = dim(img)[2], scale_hires = sf$hires, scale_lowres = sf$lowres,
                          fiducial_fullres = sf$fiducial, meta_columns = paste(setdiff(names(md), "barcode"), collapse = "|"),
                          seurat_version = as.character(o@version))
  rm(o, cnt); gc(verbose = FALSE)
}
summ <- do.call(rbind, rows)
write.table(summ, file.path(res_dir, "external_object_summary.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
print(summ[, c("section", "n_spots", "n_genes", "counts_integer", "image_rows", "image_cols", "scale_hires")])
