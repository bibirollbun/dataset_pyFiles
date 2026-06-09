# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import subprocess
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
subprocess.run(["pip", "install", "goatools", "-q"], check=True)
import goatools
from goatools.obo_parser import GODag
from functools import lru_cache
import pickle, os, time

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or preÃ‡ssing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import os, glob

for f in glob.glob("/kaggle/working/*.tsv"):
    os.remove(f)

print("ğŸ§¹ Archivos .tsv anteriores eliminados")

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# detect_random_experimental_addition.py
# --------------------------------------
# Performs ONE random experiment comparing two GAF releases.

# Each run:
#   - Randomly selects two distinct versions (v_start < v_end)
#   - Detects new experimental annotations appearing in v_end
#     that were not present in v_start
#   - Saves submission_<v_start>_to_<v_end>.tsv in /kaggle/working

# âœ… Designed for Kaggle:
# Input  -> /kaggle/input/fantasia-output/GAF_EXPLORE_CAFA5/
# Output -> /kaggle/working/
# """

# import polars as pl
# from pathlib import Path
# import random

# # =====================================
# # CONFIGURATION
# # =====================================
# FILES = {
#     214: '/kaggle/input/fantasia-output/GAF_EXPLORE_CAFA5/goa_uniprot_all.gaf.214/goa_uniprot_all_subset.214.tsv',
#     215: '/kaggle/input/fantasia-output/GAF_EXPLORE_CAFA5/goa_uniprot_all.gaf.215/goa_uniprot_all_subset.215.tsv',
#     216: '/kaggle/input/fantasia-output/GAF_EXPLORE_CAFA5/goa_uniprot_all.gaf.216/goa_uniprot_all_subset.216.tsv',
#     217: '/kaggle/input/fantasia-output/GAF_EXPLORE_CAFA5/goa_uniprot_all.gaf.217/goa_uniprot_all_subset.217.tsv',
#     218: '/kaggle/input/fantasia-output/GAF_EXPLORE_CAFA5/goa_uniprot_all.gaf.218/goa_uniprot_all_subset.218.tsv',
#     219: '/kaggle/input/fantasia-output/GAF_EXPLORE_CAFA5/goa_uniprot_all.gaf.219/goa_uniprot_all_subset.219.tsv',
#     220: '/kaggle/input/fantasia-output/GAF_EXPLORE_CAFA5/goa_uniprot_all.gaf.220/goa_uniprot_all_subset.220.tsv',
#     221: '/kaggle/input/fantasia-output/GAF_EXPLORE_CAFA5/goa_uniprot_all.gaf.221/goa_uniprot_all_subset.221.tsv',
#     222: '/kaggle/input/fantasia-output/GAF_EXPLORE_CAFA5/goa_uniprot_all.gaf.222/goa_uniprot_all_subset.222.tsv',
#     223: '/kaggle/input/fantasia-output/GAF_EXPLORE_CAFA5/goa_uniprot_all.gaf.223/goa_uniprot_all_subset.223.tsv',
#     224: '/kaggle/input/fantasia-output/GAF_EXPLORE_CAFA5/goa_uniprot_all.gaf.224/goa_uniprot_all_subset.224.tsv',
#     225: '/kaggle/input/fantasia-output/goa_uniprot_all.gaf.225/goa_uniprot_all_subset.225.tsv',
#     226: '/kaggle/input/fantasia-output/goa_uniprot_all.gaf.226/goa_uniprot_all_subset.226.tsv',
# }

# EXPERIMENTAL = {
#     "EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP", "IC", "TAS","IBA","IEA"
# }

# # =====================================
# # LOAD GAF FUNCTION
# # =====================================
# def load_gaf(version, path):
#     print(f"ğŸ“‚ Loading version {version} ...")
#     df = (
#         pl.read_csv(path, separator="\t", has_header=True)
#           .select(["protein_id", "go_term", "evidence_code", "qualifier"])
#           .with_columns([
#               pl.col("protein_id").cast(pl.Utf8).str.strip_chars(),
#               pl.col("go_term").cast(pl.Utf8).str.to_uppercase().str.strip_chars(),
#               pl.col("evidence_code").cast(pl.Utf8).str.to_uppercase().str.strip_chars(),
#               pl.col("qualifier").cast(pl.Utf8).str.to_uppercase().str.strip_chars(),
#           ])
#           .filter(pl.col("evidence_code").is_in(EXPERIMENTAL))
#           .filter(~pl.col("qualifier").is_in(["NOT", "!NOT"]))
#           .select(["protein_id", "go_term"])
#           .unique()
#     )
#     return df

# # =====================================
# # RANDOM SINGLE EXPERIMENT
# # =====================================
# versions = sorted(FILES.keys())
# v_start, v_end = sorted(random.sample(versions, 2))

# print(f"ğŸ�² Random experiment selected:")
# print(f"   ğŸ”¹ Start version: {v_start}")
# print(f"   ğŸ”¹ End version:   {v_end}")
# print(f"   ğŸ”¹ Path A: {FILES[v_start]}")
# print(f"   ğŸ”¹ Path B: {FILES[v_end]}")

# # Load datasets
# gaf_start = load_gaf(v_start, FILES[v_start])
# gaf_end = load_gaf(v_end, FILES[v_end])

# # Detect additions (new experimental annotations)
# added = gaf_end.join(gaf_start, on=["protein_id", "go_term"], how="anti")
# added = added.with_columns(pl.lit(1.0).alias("score"))

# # Save submission
# out_file = Path(f"/kaggle/working/submission.tsv")
# added.write_csv(out_file, separator="\t", include_header=False)

# print(f"\nâœ… Submission created: {out_file}")
# print(f"â�• {added.height:,} new experimental annotations detected.")
# print(f"ğŸ�� Experiment {v_start} â†’ {v_end} completed successfully.\n")



import pandas as pd

INPUT_FILE = "/kaggle/input/fantasia-output/submission.tsv"
OUTPUT_FILE = "submission.tsv"

# Leer SIN asumir cabecera, pero asignando nombres
df = pd.read_csv(
    INPUT_FILE,
    sep="\t",
    header=None,
    names=["protein", "GO_term", "confidence"],
    usecols=[0, 1, 2],
)

print("Antes de limpiar:")
print(df.head())

# Quitar filas basura tipo cabecera repetida
df = df[df["protein"] != "protein"]

# Asegurar que 'confidence' es numÃ©rico
df["confidence"] = 1

# Filtrar por umbral
# df = df[df["confidence"] >= 0.6]

print("\nDespuÃ©s de filtrar:")
print(df.head())

# Guardar SIN cabecera (requisito de la competiciÃ³n)
df.to_csv(OUTPUT_FILE, sep="\t", header=False, index=False)

print(f"\nâœ… Archivo '{OUTPUT_FILE}' generado correctamente.")
print(f"Total de filas: {len(df)}")



# EstadÃ­sticos descriptivos del score
print("\nğŸ“Š EstadÃ­sticos del score:")
print(df['confidence'].describe())












percentiles = df['confidence'].quantile([0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
print("\nğŸ“Š Percentiles del score:")
print(percentiles)




















