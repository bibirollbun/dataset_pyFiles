import os
import pandas as pd
import subprocess
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import math
from tqdm import tqdm

# =========================
# PATHS
# =========================
INPUT_DIR = "/kaggle/input/malware-classification"
ARCHIVE_PATH = os.path.join(INPUT_DIR, "train.7z")
LABEL_CSV = os.path.join(INPUT_DIR, "trainLabels.csv")
OUTPUT_DIR = "/kaggle/working/asm_labeled"

# =========================
# LABEL MAP
# =========================
label_to_family = {
    8: "Obfuscator.ACY"
}

# =========================
# PREPARE OUTPUT FOLDERS
# =========================
for fam in label_to_family.values():
    os.makedirs(os.path.join(OUTPUT_DIR, fam), exist_ok=True)

# =========================
# READ & FILTER CSV
# =========================
df = pd.read_csv(LABEL_CSV)
df = df[df["Class"].isin(label_to_family.keys())].reset_index(drop=True)
print(f"ğŸ“Š CSV selected samples: {len(df)}")

# =========================
# CHUNKING
# =========================
num_chunks = min(3, cpu_count())   # Safe for Kaggle
chunk_size = math.ceil(len(df) / num_chunks)
chunks = [df[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]

print(f"âš™ï¸� Using {num_chunks} parallel workers")
for i, c in enumerate(chunks):
    print(f"  â€¢ Chunk {i+1}: {len(c)} files")

# =========================
# EXTRACTION FUNCTION
# =========================
def extract_files(df_chunk):
    extracted_ids = set()
    family_extracted = defaultdict(int)
    failed_ids = []

    for _, row in tqdm(
        df_chunk.iterrows(),
        total=len(df_chunk),
        desc=f"Worker PID {os.getpid()}",
        leave=False
    ):
        file_id = row["Id"]
        family = label_to_family[row["Class"]]
        out_dir = os.path.join(OUTPUT_DIR, family)
        asm_in_7z = f"train/{file_id}.asm"

        cmd = [
            "7z", "x",
            ARCHIVE_PATH,
            asm_in_7z,
            f"-o{out_dir}",
            "-y"
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if result.returncode == 0:
            extracted_ids.add(file_id)
            family_extracted[family] += 1
        else:
            failed_ids.append(file_id)

    return extracted_ids, dict(family_extracted), failed_ids

# =========================
# PARALLEL EXTRACTION
# =========================
if __name__ == "__main__":
    with Pool(num_chunks) as pool:
        results = pool.map(extract_files, chunks)

    # =========================
    # MERGE RESULTS
    # =========================
    all_extracted_ids = set()
    all_family_extracted = defaultdict(int)
    all_failed_ids = []

    for extracted_ids, family_extracted, failed_ids in results:
        all_extracted_ids.update(extracted_ids)
        for fam, count in family_extracted.items():
            all_family_extracted[fam] += count
        all_failed_ids.extend(failed_ids)

    # =========================
    # FINAL REPORT
    # =========================
    print("\nâœ… Extraction completed")
    print("ğŸ“‚ Extracted files per family:")
    for fam, cnt in all_family_extracted.items():
        print(f"   {fam}: {cnt}")

    print(f"\nâ�Œ Failed extractions: {len(all_failed_ids)}")

    if all_failed_ids:
        failed_path = os.path.join(OUTPUT_DIR, "failed_ids.txt")
        with open(failed_path, "w") as f:
            for fid in all_failed_ids:
                f.write(f"{fid}\n")
        print(f"ğŸ“� Failed IDs saved to: {failed_path}")



!7z a -tzip /kaggle/working/asm_labeled.zip /kaggle/working/asm_labeled/*

