import os
import pandas as pd

# =========================
# PATHS
# =========================
INPUT_DIR = "/kaggle/input/malware-classification"
ARCHIVE_PATH = os.path.join(INPUT_DIR, "train.7z")
LABEL_CSV = "/kaggle/input/malware-classification/trainLabels.csv"
OUTPUT_DIR = "/kaggle/working/asm_labeled"

# =========================
# LABEL â†’ FAMILY MAP
# =========================
label_to_family = {
    1: "Ramnit",
    2: "Lollipop",
    3: "Kelihos_ver3",
    8: "Obfuscator.ACY",
    9: "Gatak"
}

# Create folders
os.makedirs(OUTPUT_DIR, exist_ok=True)
for family in label_to_family.values():
    os.makedirs(os.path.join(OUTPUT_DIR, family), exist_ok=True)

print(LABEL_CSV)
print("âœ… Output folders ready")



# Read CSV
df = pd.read_csv(LABEL_CSV)

# Keep only selected classes
df = df[df["Class"].isin(label_to_family.keys())]

print(f"ğŸ“Š Total ASM files to extract: {len(df)}")
df.head()



import subprocess
from tqdm import tqdm

missing = 0
extracted = 0

for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting ASM files"):
    file_id = row["Id"]
    family = label_to_family[row["Class"]]

    asm_name = f"{file_id}.asm"
    output_path = os.path.join(OUTPUT_DIR, family)

    cmd = [
        "7z", "x",
        ARCHIVE_PATH,
        asm_name,
        f"-o{output_path}",
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
        missing += 1

print("\nâœ… Extraction complete")
print(f"âœ” Extracted files : {extracted}")
print(f"âš  Missing files  : {missing}")



!ls /kaggle/input/malware-classification


!find /kaggle/working/asm_labeled -name "*.asm" | wc -l


import os
import pandas as pd
import subprocess
from tqdm import tqdm
from collections import defaultdict

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
    9: "Gatak"
}

# =========================
# PREPARE FOLDERS
# =========================
for fam in label_to_family.values():
    os.makedirs(os.path.join(OUTPUT_DIR, fam), exist_ok=True)

# =========================
# READ CSV
# =========================
df = pd.read_csv(LABEL_CSV)
df = df[df["Class"].isin(label_to_family.keys())]

print(f"\nğŸ“Š CSV selected samples: {len(df)}")

# =========================
# TRACE STRUCTURES
# =========================
expected_ids = set(df["Id"])
extracted_ids = set()
family_expected = defaultdict(int)
family_extracted = defaultdict(int)
failed_ids = []

# =========================
# EXTRACTION
# =========================
for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting ASM with trace"):
    file_id = row["Id"]
    family = label_to_family[row["Class"]]
    family_expected[family] += 1

    asm_in_7z = f"train/{file_id}.asm"
    out_dir = os.path.join(OUTPUT_DIR, family)

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

print("\nâœ… Extraction phase finished")
print(f"Extracted files per family: {dict(family_extracted)}")
print(f"Failed extractions: {len(failed_ids)}")



import os

disk_ids = set()

for root, _, files in os.walk(OUTPUT_DIR):
    for f in files:
        if f.endswith(".asm"):
            disk_ids.add(f.replace(".asm", ""))

print(f"\nğŸ“� ASM files found on disk: {len(disk_ids)}")



missing_on_disk = expected_ids - disk_ids
extra_on_disk = disk_ids - expected_ids

print("\nğŸ”� CONSISTENCY REPORT")
print(f"âœ” Expected from CSV     : {len(expected_ids)}")
print(f"âœ” Extracted (7z success): {len(extracted_ids)}")
print(f"âœ” Found on disk         : {len(disk_ids)}")
print(f"â�Œ Missing on disk      : {len(missing_on_disk)}")
print(f"âš  Extra on disk         : {len(extra_on_disk)}")



print("\nğŸ“‚ FAMILY-WISE TRACE")
print("-" * 45)

for fam in label_to_family.values():
    print(
        f"{fam:15s} | "
        f"Expected: {family_expected[fam]:5d} | "
        f"Extracted: {family_extracted[fam]:5d}"
    )



!ls /kaggle/working/


import os
import pandas as pd
import subprocess
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import math

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
    9: "Gatak"
}

# =========================
# PREPARE FOLDERS
# =========================
for fam in label_to_family.values():
    os.makedirs(os.path.join(OUTPUT_DIR, fam), exist_ok=True)

# =========================
# READ CSV
# =========================
df = pd.read_csv(LABEL_CSV)
df = df[df["Class"].isin(label_to_family.keys())]
print(f"ğŸ“Š CSV selected samples: {len(df)}")

# =========================
# CHUNKING
# =========================
num_chunks = 3
chunk_size = math.ceil(len(df) / num_chunks)
chunks = [df[i*chunk_size : (i+1)*chunk_size] for i in range(num_chunks)]

# =========================
# EXTRACTION FUNCTION
# =========================
def extract_files(df_chunk):
    extracted_ids = set()
    family_extracted = defaultdict(int)
    failed_ids = []

    for _, row in df_chunk.iterrows():
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

        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
    with Pool(min(num_chunks, cpu_count())) as pool:
        results = pool.map(extract_files, chunks)

    all_extracted_ids = set()
    all_family_extracted = defaultdict(int)
    all_failed_ids = []

    for extracted_ids, family_extracted, failed_ids in results:
        all_extracted_ids.update(extracted_ids)
        for fam, count in family_extracted.items():
            all_family_extracted[fam] += count
        all_failed_ids.extend(failed_ids)

    print("\nâœ… All chunks finished")
    print(f"Total extracted files per family: {dict(all_family_extracted)}")
    print(f"Total failed extractions: {len(all_failed_ids)}")



!7z a -tzip /kaggle/working/asm_labeled.zip /kaggle/working/asm_labeled/*


from IPython.display import HTML
import os

zip_file = '/kaggle/working/asm_labeled.zip'
filename = os.path.basename(zip_file)
size_mb = os.path.getsize(zip_file) / (1024 * 1024)

HTML(f'''
<div style="padding: 20px; background: #f5f5f5; border-radius: 10px;">
    <h3>ğŸ“� File Ready for Download</h3>
    <p><strong>File:</strong> {filename}</p>
    <p><strong>Size:</strong> {size_mb:.2f} MB</p>
    <p><a href="{zip_file}" download style="
        display: inline-block;
        padding: 10px 20px;
        background: #0066cc;
        color: white;
        text-decoration: none;
        border-radius: 5px;
        font-weight: bold;
    ">â¬‡ï¸� Download {filename}</a></p>
</div>
''')


!ls -lh /kaggle/working



!7z a -tzip -v1g /kaggle/working/asm_labeled_parts.zip /kaggle/working/asm_labeled/*


!ls -lh /kaggle/working



from IPython.display import FileLink
FileLink('/kaggle/working/asm_labeled_parts.zip.001')


