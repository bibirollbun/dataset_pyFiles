import os
import pandas as pd
import subprocess
from tqdm import tqdm

# =========================
# CONFIG
# =========================
CLASS_ID = 2                 # Lollipop
FAMILY_NAME = "Lollipop"
CHUNK_SIZE = 500
CHUNK_ID = 1               # ğŸ”´ CHANGE THIS PER NOTEBOOK

# =========================
# PATHS (Kaggle)
# =========================
INPUT_DIR = "/kaggle/input/malware-classification"
ARCHIVE_PATH = os.path.join(INPUT_DIR, "train.7z")
LABEL_CSV = os.path.join(INPUT_DIR, "trainLabels.csv")

WORK_DIR = "/kaggle/working"
OUTPUT_DIR = os.path.join(WORK_DIR, "asm_labeled", FAMILY_NAME)

# 7z output per chunk
ARCHIVE_OUT = os.path.join(
    WORK_DIR,
    f"{FAMILY_NAME}_ASM_chunk_{CHUNK_ID}.7z"
)

# =========================
# PREPARE OUTPUT FOLDER
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"âœ… Output folder ready: {OUTPUT_DIR}")

# =========================
# LOAD & FILTER CSV
# =========================
df = pd.read_csv(LABEL_CSV)
df = df[df["Class"] == CLASS_ID].reset_index(drop=True)

TOTAL = len(df)
START = CHUNK_ID * CHUNK_SIZE
END = min(START + CHUNK_SIZE, TOTAL)

if START >= TOTAL:
    raise ValueError("â�Œ Chunk ID exceeds available samples")

chunk_df = df.iloc[START:END]

print(f"\nğŸ“Š Family        : {FAMILY_NAME}")
print(f"ğŸ“¦ Total samples : {TOTAL}")
print(f"ğŸ”� Chunk ID      : {CHUNK_ID}")
print(f"ğŸ“‚ Processing    : {START} â†’ {END - 1}")
print(f"ğŸ“„ Files in chunk: {len(chunk_df)}")

# =========================
# EXTRACT ASM FILES
# =========================
extracted = 0
failed = 0

for _, row in tqdm(
    chunk_df.iterrows(),
    total=len(chunk_df),
    desc=f"Extracting {FAMILY_NAME} ASM",
    unit="file"
):
    file_id = row["Id"]
    asm_path_in_7z = f"train/{file_id}.asm"

    cmd = [
        "7z", "x",
        ARCHIVE_PATH,
        asm_path_in_7z,
        f"-o{OUTPUT_DIR}",
        "-y"
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    if result.returncode == 0:
        extracted += 1
    else:
        failed += 1

# =========================
# CREATE 7z ARCHIVE
# =========================
print("\nğŸ“¦ Compressing chunk to 7z...")

subprocess.run(
    [
        "7z", "a", "-t7z", "-y",
        ARCHIVE_OUT,
        OUTPUT_DIR
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

# =========================
# CLEANUP (IMPORTANT)
# =========================
print("ğŸ§¹ Cleaning temporary extracted files...")
subprocess.run(["rm", "-rf", OUTPUT_DIR])

# =========================
# SUMMARY
# =========================
print("\nâœ… CHUNK COMPLETE")
print(f"ğŸ“� Extracted ASM  : {extracted}")
print(f"â�Œ Failed         : {failed}")
print(f"ğŸ“¦ 7z archive     : {ARCHIVE_OUT}")
print("\nğŸ�‰ READY FOR DOWNLOAD")





