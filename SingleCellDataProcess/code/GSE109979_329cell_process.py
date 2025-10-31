# -*- coding: utf-8 -*-
"""
Process GSE109979 (329 cells) into {data.csv, labels.csv}

Usage:
  python SingleCellDataProcess/code/GSE109979_329cell_process.py ./SingleCellDataProcess/output/ ./SingleCellDataProcess/

Args:
  sys.argv[1] -> out root (e.g., ./SingleCellDataProcess/output/)
  sys.argv[2] -> data_process_path root (e.g., ./SingleCellDataProcess/)

Input expected (already downloaded as you showed in 'temporary/'):
  ./SingleCellDataProcess/temporary/GSE109979_329Cell_RPKM.txt.gz

Output:
  ./SingleCellDataProcess/output/GSE109979_329cell/GSE109979_329cell_data.csv
  ./SingleCellDataProcess/output/GSE109979_329cell/GSE109979_329cell_labels.csv

Notes:
- No meta CSV is required. We derive a coarse “cell type” by taking the prefix
  before '_' (fallback to '-' then full title) from each sample title in the header.
- The raw matrix is assumed to be a tab/space-separated text with first column
  = gene (or feature) name, subsequent columns = samples.
"""

import sys, os, gzip, shutil
import numpy as np
import pandas as pd

# Import helper utilities from the same folder as your other processors
from auxilary import makeFolder, writeCellGeneCSV, writeCellLabelsCSV

# ----------------------- Config -----------------------
data = 'GSE109979_329cell'  # output folder & file prefix
RAW_GZ_NAME = 'GSE109979_329Cell_RPKM.txt.gz'  # the file you already have
# ------------------------------------------------------

# Parse CLI
if len(sys.argv) < 3:
    print("Usage: python GSE109979_329cell_process.py <out_root> <data_process_path>")
    sys.exit(1)

out_root = sys.argv[1]
data_process_path = sys.argv[2]

outpath = os.path.join(out_root, f"{data}/")
temp_folder = os.path.join(data_process_path, "temporary/")
makeFolder(temp_folder)
makeFolder(outpath)

# A temp work folder (deleted at the end)
temp_temp_folder = os.path.join(temp_folder, f"{data}_temporary/")
makeFolder(temp_temp_folder)

raw_gz_path = os.path.join(temp_folder, RAW_GZ_NAME)
if not os.path.exists(raw_gz_path):
    print(f"[ERROR] Expected raw file not found:\n  {raw_gz_path}\n"
          f"Please place the compressed matrix here (as shown in your screenshot).")
    sys.exit(1)
else:
    print("RAW data found:", raw_gz_path)

# Decompress to a temporary plain-text file
plain_txt = os.path.join(temp_temp_folder, f"{data}.txt")
with gzip.open(raw_gz_path, 'rb') as fin, open(plain_txt, 'wb') as fout:
    shutil.copyfileobj(fin, fout)

# Read all lines (assume moderate size as 329 cells)
with open(plain_txt, 'r', encoding='utf-8', errors='ignore') as f:
    raw_lines = f.readlines()

if not raw_lines:
    print("[ERROR] Empty file after decompression.")
    shutil.rmtree(temp_temp_folder, ignore_errors=True)
    sys.exit(1)

# Header: first line with tokens = [gene_col_header, sample1, sample2, ...]
header = raw_lines[0].strip().split()
if len(header) < 2:
    print("[ERROR] Header line has <2 columns. Unexpected format.")
    shutil.rmtree(temp_temp_folder, ignore_errors=True)
    sys.exit(1)

sample_titles = header[1:]  # all columns except the first (gene column)
print(f"Detected {len(sample_titles)} samples from header.")

# Infer a coarse 'cell type' from title prefix (before '_' or '-')
def infer_cell_type(title: str) -> str:
    if '_' in title:
        return title.split('_')[0]
    if '-' in title:
        return title.split('-')[0]
    return title  # fallback = full title

Cell_type_list = [infer_cell_type(t) for t in sample_titles]
Sample_Name_list = sample_titles[:]  # keep original titles as sample IDs

# Build gene list from first column of each subsequent row
gene_list = []
for i, line in enumerate(raw_lines[1:], start=1):
    parts = line.strip().split()
    if not parts:
        continue
    gene = parts[0].upper()
    gene_list.append(gene)

M = len(gene_list)
N = len(Sample_Name_list)
print('Number of genes:', M)
print('Number of cells:', N)

# Allocate and fill expression matrix (M x N)
MATRIX = np.zeros((M, N), dtype=float)

bad_rows = 0
for i, line in enumerate(raw_lines[1:], start=1):
    parts = line.strip().split()
    if len(parts) < (1 + N):
        # row has fewer values than samples; try to pad/skip
        bad_rows += 1
        continue
    try:
        row_vals = np.array(parts[1:1+N], dtype=float)
    except ValueError:
        bad_rows += 1
        continue
    MATRIX[i-1, :] = row_vals

if bad_rows > 0:
    print(f"WARNING: {bad_rows} data rows could not be parsed cleanly.")

# Build label dict
unique_cell_types = sorted(list(set(Cell_type_list)))
cell_type_to_idx = {ct: i for i, ct in enumerate(unique_cell_types)}
Cell_label_list = [cell_type_to_idx[ct] for ct in Cell_type_list]

print("Cell type counts:")
counts = {ct: 0 for ct in unique_cell_types}
for ct in Cell_type_list:
    counts[ct] += 1
for k, v in counts.items():
    print(f"  {k}: {v}")

# Write outputs
data_csv = os.path.join(outpath, f"{data}_data.csv")
labels_csv = os.path.join(outpath, f"{data}_labels.csv")

writeCellGeneCSV(data_csv, Sample_Name_list, gene_list, MATRIX)
writeCellLabelsCSV(labels_csv, Sample_Name_list, Sample_Name_list, Cell_type_list, Cell_label_list)

# Clean up temp
shutil.rmtree(temp_temp_folder, ignore_errors=True)
print("\nDone.")
print("Wrote:")
print(" ", data_csv)
print(" ", labels_csv)