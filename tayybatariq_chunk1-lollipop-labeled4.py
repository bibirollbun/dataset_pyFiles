import os
import pandas as pd
import subprocess
from tqdm import tqdm

# =========================
# CONFIG (CHANGE ONLY THESE)
# =========================
CLASS_ID = 2
FAMILY_NAME = "Lollipop"

CHUNK_SIZE = 500
CHUNK_ID = 3              # 1500â€“1999

SUB_CHUNK_SIZE = 250
SUB_CHUNK_ID = 1          # 1750â€“1999

# =========================
# PATHS (Kaggle)
# =========================
INPUT_DIR = "/kaggle/input/malware-classification"
ARCHIVE_PATH = os.path.join(INPUT_DIR, "train.7z")
LABEL_CSV = os.path.join(INPUT_DIR, "trainLabels.csv")

WORK_DIR = "/kaggle/working"
OUTPUT_DIR = os.path.join(WORK_DIR, "asm_labeled", FAMILY_NAME)

ARCHIVE_OUT = os.path.join(
    WORK_DIR,
    f"{FAMILY_NAME}_ASM_chunk_{CHUNK_ID}_sub_{SUB_CHUNK_ID}.7z"
)

# =========================
# PREPARE OUTPUT
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"âœ… Output folder ready (Chunk {CHUNK_ID} / Sub {SUB_CHUNK_ID})")

# =========================
# LOAD CSV
# =========================
df = pd.read_csv(LABEL_CSV)
df = df[df["Class"] == CLASS_ID].reset_index(drop=True)

TOTAL = len(df)

CHUNK_START = CHUNK_ID * CHUNK_SIZE
CHUNK_END = min(CHUNK_START + CHUNK_SIZE, TOTAL)

SUB_START = CHUNK_START + (SUB_CHUNK_ID * SUB_CHUNK_SIZE)
SUB_END = min(SUB_START + SUB_CHUNK_SIZE, CHUNK_END)

if SUB_START >= TOTAL:
    raise ValueError("â�Œ Invalid sub-chunk range")

chunk_df = df.iloc[SUB_START:SUB_END]

print(f"ğŸ“‚ Processing CSV index {SUB_START} â†’ {SUB_END - 1}")
print(f"ğŸ“„ Files count: {len(chunk_df)}")

# =========================
# EXTRACTION
# =========================
extracted = 0
failed = 0

for _, row in tqdm(
    chunk_df.iterrows(),
    total=len(chunk_df),
    desc="Extracting ASM",
    unit="file"
):
    file_id = row["Id"]
    asm_path = f"train/{file_id}.asm"

    result = subprocess.run(
        ["7z", "x", ARCHIVE_PATH, asm_path, f"-o{OUTPUT_DIR}", "-y"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    if result.returncode == 0:
        extracted += 1
    else:
        failed += 1

print(f"âœ” Extracted: {extracted}")
print(f"â�Œ Failed   : {failed}")

# =========================
# VERIFY FILES BEFORE ZIP
# =========================
asm_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".asm")]

if len(asm_files) == 0:
    raise RuntimeError("â�Œ No ASM files found â€” aborting compression")

# =========================
# SAFE COMPRESSION
# =========================
print("ğŸ“¦ Creating 7z archive...")

zip_result = subprocess.run(
    ["7z", "a", "-t7z", "-y", ARCHIVE_OUT, OUTPUT_DIR],
)

# =========================
# CLEANUP ONLY IF SUCCESS
# =========================
if zip_result.returncode == 0:
    print("ğŸ§¹ Compression successful â€” cleaning extracted files")
    subprocess.run(["rm", "-rf", OUTPUT_DIR])
else:
    print("â�Œ Compression failed â€” extracted files kept")

# =========================
# FINAL SUMMARY
# =========================
print("\nâœ… DONE")
print(f"ğŸ“� Extracted ASM : {extracted}")
print(f"â�Œ Failed        : {failed}")
print(f"ğŸ“¦ Archive       : {ARCHIVE_OUT}")
print("ğŸ�‰ READY FOR DOWNLOAD")



import subprocess
import os

SOURCE_DIR = "/kaggle/working/asm_labeled/Lollipop"
OUT_7Z = "/kaggle/working/Lollipop_ASM_FINAL.7z"

# Safety check
if not os.path.exists(SOURCE_DIR):
    raise RuntimeError("â�Œ Source folder does not exist")

print("ğŸ“¦ Compressing existing ASM folder...")

subprocess.run(
    ["7z", "a", "-t7z", "-y", OUT_7Z, SOURCE_DIR],
    check=True
)

print("âœ… Compression complete")
print(f"ğŸ“¦ Archive created: {OUT_7Z}")



import subprocess
import os

SOURCE_DIR = "/kaggle/working/asm_labeled/Lollipop"
OUT_7Z = "/kaggle/working/Lollipop_ASM_FINAL1.7z"

# Safety check
if not os.path.exists(SOURCE_DIR):
    raise RuntimeError("â�Œ Source folder does not exist")

print("ğŸ“¦ Compressing ASM folder (percentage progress)...\n")

process = subprocess.Popen(
    [
        "7z", "a",
        "-t7z",
        "-y",
        "-bsp1",        # âœ… percentage progress
        OUT_7Z,
        SOURCE_DIR
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

# Print progress live
for line in process.stdout:
    print(line, end="")

process.wait()

if process.returncode == 0:
    print("\nâœ… Compression finished successfully!")
    print(f"ğŸ“¦ Archive created: {OUT_7Z}")
else:
    print("\nâ�Œ Compression failed!")



!7z a -t7z -mx=1 -y /kaggle/working/Lollipop_ASM_FINAL.7z /kaggle/working/asm_labeled/Lollipop





