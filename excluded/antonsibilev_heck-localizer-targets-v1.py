#!/usr/bin/env python3
"""
quick_check_localizers.py
────────────────────────────────────────────────────────
Flag every localizer row that refers to a series folder
containing just **one DICOM file** (Enhanced multi-frame).

Output:
  logs/localizer_single.csv  – rows that need special handling
"""

import csv
from pathlib import Path
from tqdm import tqdm

SERIES_ROOT = Path("/kaggle/input/rsna-intracranial-aneurysm-detection/series")
CSV_LOC     = Path("/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv")
LOG_DIR     = Path("/kaggle/working/logs"); LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_SINGLE  = LOG_DIR / "localizer_single.csv"

# Cache: series_uid → number of .dcm files
slice_count = {}

def get_slice_count(series_uid: str) -> int:
    """Return number of .dcm files in the series folder (cached)."""
    if series_uid not in slice_count:
        sdir = SERIES_ROOT / series_uid
        slice_count[series_uid] = len(list(sdir.glob("*.dcm"))) if sdir.exists() else 0
    return slice_count[series_uid]

n_single = 0
with CSV_LOC.open() as f_in, LOG_SINGLE.open("w", newline="") as f_out:
    reader = csv.DictReader(f_in)
    writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
    writer.writeheader()

    for row in tqdm(reader, desc="checking"):
        uid = row["SeriesInstanceUID"]
        if get_slice_count(uid) == 1:          # only one DICOM → multi-frame
            writer.writerow(row)
            n_single += 1

print(f"✔ scanned {len(slice_count)} unique series")
print(f"   → rows pointing to single-file series : {n_single}")
print(f"Log saved to: {LOG_SINGLE}")


