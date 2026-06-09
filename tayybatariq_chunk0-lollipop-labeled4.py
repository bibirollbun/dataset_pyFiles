import os
import pandas as pd
import subprocess
from tqdm import tqdm

# =========================
# CONFIG
# =========================
CLASS_ID = 2
FAMILY_NAME = "Lollipop"

CHUNK_SIZE = 500
CHUNK_ID = 3

SUB_CHUNK_SIZE = 250
SUB_CHUNK_ID = 0     # âœ… FIRST HALF (1500â€“1749)

# =========================
# PATHS
# =========================
INPUT_DIR = "/kaggle/input/malware-classification"
ARCHIVE_PATH = os.path.join(INPUT_DIR, "train.7z")
LABEL_CSV = os.path.join(INPUT_DIR, "trainLabels.csv")

WORK_DIR = "/kaggle/working"
OUTPUT_DIR = os.path.join(WORK_DIR, "asm_labeled", FAMILY_NAME)

ARCHIVE_OUT = os.path.join(
    WORK_DIR,
    "Lollipop_ASM_chunk_3_sub_0.7z"
)

# =========================
# PREPARE OUTPUT
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("âœ… Output folder ready (Chunk 3 / Sub 0)")

# =========================
# LOAD CSV
# =========================
df = pd.read_csv(LABEL_CSV)
df = df[df["Class"] == CLASS_ID].reset_index(drop=True)

TOTAL = len(df)

CHUNK_START = CHUNK_ID * CHUNK_SIZE        # 1500
CHUNK_END = min(CHUNK_START + CHUNK_SIZE, TOTAL)

SUB_START = CHUNK_START + (SUB_CHUNK_ID * SUB_CHUNK_SIZE)  # 1500
SUB_END = min(SUB_START + SUB_CHUNK_SIZE, CHUNK_END)       # 1750

chunk_df = df.iloc[SUB_START:SUB_END]

print(f"ðŸ“‚ Processing CSV index {SUB_START} â†’ {SUB_END - 1}")

# =========================
# EXTRACTION
# =========================
for _, row in tqdm(
    chunk_df.iterrows(),
    total=len(chunk_df),
    desc="Extracting ASM",
    unit="file"
):
    file_id = row["Id"]
    asm_path = f"train/{file_id}.asm"

    subprocess.run(
        ["7z", "x", ARCHIVE_PATH, asm_path, f"-o{OUTPUT_DIR}", "-y"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

# =========================
# COMPRESS
# =========================
print("ðŸ“¦ Creating 7z archive...")
subprocess.run(["7z", "a", "-t7z", "-y", ARCHIVE_OUT, OUTPUT_DIR])

# =========================
# CLEANUP
# =========================
subprocess.run(["rm", "-rf", OUTPUT_DIR])

print("âœ… DONE â†’ Lollipop_ASM_chunk_3_sub_0.7z")





