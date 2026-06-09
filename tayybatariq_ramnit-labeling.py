import os
import pandas as pd
import subprocess
from tqdm import tqdm
from zipfile import ZipFile

# =========================
# PATHS (Kaggle competition)
# =========================
INPUT_DIR = "/kaggle/input/malware-classification"
ARCHIVE_PATH = os.path.join(INPUT_DIR, "train.7z")
LABEL_CSV = os.path.join(INPUT_DIR, "trainLabels.csv")

WORK_DIR = "/kaggle/working"
OUTPUT_DIR = os.path.join(WORK_DIR, "asm_labeled")
RAMNIT_DIR = os.path.join(OUTPUT_DIR, "Ramnit")
ZIP_PATH = os.path.join(WORK_DIR, "Ramnit_ASM.zip")

# =========================
# CREATE OUTPUT FOLDER
# =========================
os.makedirs(RAMNIT_DIR, exist_ok=True)
print("âœ… Output folder ready: Ramnit")

# =========================
# READ & FILTER CSV
# =========================
df = pd.read_csv(LABEL_CSV)
df = df[df["Class"] == 1]   # Ramnit only

print(f"ğŸ“Š Total Ramnit samples: {len(df)}")

# =========================
# EXTRACTION FROM train.7z
# =========================
extracted = 0
failed = 0

for _, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Extracting Ramnit ASM files",
    unit="file"
):
    file_id = row["Id"]
    asm_in_7z = f"train/{file_id}.asm"

    cmd = [
        "7z", "x",
        ARCHIVE_PATH,
        asm_in_7z,
        f"-o{RAMNIT_DIR}",
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
# ZIP THE OUTPUT FOLDER
# =========================
print("\nğŸ“¦ Creating ZIP archive...")

with ZipFile(ZIP_PATH, 'w') as zipf:
    for root, _, files in os.walk(RAMNIT_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, OUTPUT_DIR)
            zipf.write(file_path, arcname)

# =========================
# FINAL SUMMARY
# =========================
print("\nâœ… PROCESS COMPLETE")
print(f"ğŸ“� Ramnit ASM extracted : {extracted}")
print(f"â�Œ Failed extractions  : {failed}")
print(f"ğŸ“¦ ZIP file created    : {ZIP_PATH}")
print("\nğŸ�‰ RAMNIT DATASET READY FOR DOWNLOAD!")



!7z a -t7z /kaggle/working/Ramnit_ASM.7z /kaggle/working/asm_labeled/Ramnit





