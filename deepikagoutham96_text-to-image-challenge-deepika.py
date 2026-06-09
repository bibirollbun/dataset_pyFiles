# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ==========================================================
# SECTION 1: Setup & Configuration
# ==========================================================
# Purpose:
#   - Import libraries
#   - Define configuration parameters
#   - Prepare output directories
#   - (Optional) Install required packages
# Notes:
#   - Assign parameters printed to be saved in config-dreamlayer.json
#   - Includes model, steps, guidance, size, seed, and prompt file path
# ==========================================================



# ============================================================
# DreamLayer Text-to-Image Challenge – Verbose Generator
# ============================================================
import os, json, time, math
import torch
import pandas as pd
import matplotlib.pyplot as plt
from diffusers import StableDiffusionXLPipeline


# ----------------- CONFIG -----------------
config = {
    "model": "stabilityai/stable-diffusion-xl-base-1.0",
    "num_inference_steps": 35,
    "guidance_scale": 8.5,
    "height": 768,
    "width": 768,
    "seed": 1234,
    "batch_size": 1,
    "prompt_file": "/kaggle/input/text-to-image-challenge/DreamLayer-Prompt-Kaggle.txt",
    "output_dir": "/kaggle/working/output"
}
# ------------------------------------------


os.makedirs(config["output_dir"], exist_ok=True)

# Log all fixed parameters
print("========== CONFIGURATION ==========")
for k, v in config.items():
    print(f"{k:20s}: {v}")
print("===================================\n")


# ==========================================================
# SECTION 2: Image Generation
# ==========================================================
# Purpose:
#   - Load your chosen diffusion model (e.g., SDXL)
#   - Generate one image per text prompt
#   - Save all outputs to /kaggle/working/output/
#   - Create results.csv mapping prompt_id → file_name
#   - Save config-dreamlayer.json with all fixed params
# Guidelines followed:
#   - Do NOT rename images (use zero-padded names: 0001.png, 0002.png, ...)
#   - Log progress for each prompt
#   - Show periodic image previews in rows for clarity and runtime validation
# ==========================================================



# Load model
print("Loading model... this may take 1–2 minutes ⏳")
pipe = StableDiffusionXLPipeline.from_pretrained(
    config["model"], torch_dtype=torch.float16
).to("cuda")
print(" Model loaded.\n")


# Fix seed for reproducibility
generator = torch.Generator("cuda").manual_seed(config["seed"])


# Read prompts
with open(config["prompt_file"], "r", encoding="utf-8") as f:
    prompts = [
        p.strip() for p in f
        if p.strip() and not p.strip().startswith("#")
    ]

print(f"Total prompts found: {len(prompts)}\n")


# Generation loop
rows = []
t0 = time.time()
for i, prompt in enumerate(prompts, start=1):
    print(f"\n  [{i}/{len(prompts)}] Generating for prompt:")
    print(f"     {prompt}")
    start = time.time()

    # Generate image
    result = pipe(
        prompt,
        num_inference_steps=config["num_inference_steps"],
        guidance_scale=config["guidance_scale"],
        height=config["height"],
        width=config["width"],
        generator=generator,
    )
    image = result.images[0]
    name = f"{i:04}.png"
    save_path = os.path.join(config["output_dir"], name)
    image.save(save_path)
    duration = time.time() - start
    print(f" Saved {name} ({duration:.1f}s)")

    rows.append({"prompt_id": i, "file_name": name, "prompt": prompt})

    # Display every few images as preview
    if i % 4 == 0 or i == len(prompts):
        fig, axes = plt.subplots(1, min(4, len(rows[-4:])), figsize=(16, 4))
        for ax, rec in zip(axes, rows[-4:]):
            ax.imshow(plt.imread(os.path.join(config["output_dir"], rec["file_name"])))
            ax.axis("off")
            ax.set_title(f"{rec['prompt_id']:04d}")
        plt.show()

total_time = time.time() - t0
print(f"\n Completed all {len(prompts)} prompts in {total_time/60:.2f} minutes.")


# Save results and config
df = pd.DataFrame(rows)
csv_path = os.path.join(config["output_dir"], "results.csv")
df.to_csv(csv_path, index=False)
json_path = os.path.join(config["output_dir"], "config-dreamlayer.json")
json.dump(config, open(json_path, "w"), indent=2)


print("\n Output summary:")
print(f"Images folder : {config['output_dir']}")
print(f"Results CSV   : {csv_path}")
print(f"Config JSON   : {json_path}")
print("Use these for DreamLayer report and Kaggle submission.\n")
!ls -lh $config_output_dir | head


import json, pandas as pd

print("=== Config file preview ===")
with open("/kaggle/working/output/config-dreamlayer.json") as f:
    cfg = json.load(f)
print(json.dumps(cfg, indent=2))

print("\n=== Results CSV preview (first 5 rows) ===")
df = pd.read_csv("/kaggle/working/output/results.csv")
print(df.head())
print("\nTotal rows:", len(df))



import os, zipfile, textwrap

# === ZIP your output folder ===
zip_path = "/kaggle/working/output_bundle.zip"
folder_to_zip = "/kaggle/working/output"

print("Zipping all generated files...")
!zip -r -q $zip_path $folder_to_zip
print(f"Created zip: {zip_path}  (size: {os.path.getsize(zip_path)/1e6:.2f} MB)\n")


# ==========================================================
# SECTION 3: Download or Access Public Folder 
# ==========================================================
# Purpose:
#   - Download the previously generated images/results from Google Drive
#     OR verify your existing output directory on Kaggle
# Use when:
#   - You already uploaded the output contents to Drive
#   - You want to test loading them back for evaluation
# ==========================================================



!pip install ultralytics gdown


import os
import pandas as pd
import nltk
nltk.download(['punkt', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'wordnet', 'omw-1.4'])
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from ultralytics import YOLO
import gdown
import numpy as np

#upload zipped img and result files to the below link to avoid gdown limit error
drive_link = "https://drive.google.com/file/d/1z2V_dCB-NlD5xcU7vQS4jt56SNjcWf8u/view?usp=sharing"
prompt_dataset = "https://docs.google.com/document/d/1VJHTKDMPapyLbACAIOWvzRdrUhV70y03Mbyr_y--AN0/edit?usp=sharing"


# ==========================================================
#  Drive Link Clarification — Explaining folder vs. zip approach
# ==========================================================

expected_drive_link = "https://drive.google.com/drive/folders/1GiJrhamvanfSBHIMTnMiHOkg4S9Wu5N1?usp=sharing"
zipped_drive_link = "https://drive.google.com/file/d/1z2V_dCB-NlD5xcU7vQS4jt56SNjcWf8u/view?usp=sharing"

print(" DreamLayer Drive Link Verification")
print("------------------------------------------------------------")
print(f"Expected Drive folder structure link:\n{expected_drive_link}\n")
print(
    "This Drive folder contains the expected DreamLayer output files — "
    "all generated images (0001.png, 0002.png, ...), 'results.csv', "
    "and 'config-dreamlayer.json'.\n\n"
    "However, when using gdown.download_folder(), the process throws a "
    "FolderContentsMaximumLimitError. This happens even though the folder "
    "has the correct structure, because gdown imposes a strict limit on "
    "listing Drive folder contents (≈50 items).\n"
)
print(
    "To overcome this limitation and maintain full reproducibility, "
    "the folder contents have been compressed into a single ZIP archive "
    "and uploaded instead. This ensures efficient loading and avoids the "
    "Drive API limits encountered earlier.\n"
)
print(f"Alternative zipped file link:\n{zipped_drive_link}\n")
print(" The notebook will now use this ZIP version for extraction and evaluation.")
print("------------------------------------------------------------")



def download_yolo_file():
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    return 'yolov8n.pt'


yolo_model = download_yolo_file()


# ==========================================================
#  Smart ZIP Auto-Downloader for DreamLayer Evaluation
# ==========================================================

import os, zipfile, gdown, pandas as pd

def parse_and_verify_zip_from_drive(drive_link, output_dir="kaggle_resources"):
    """
    Downloads a ZIP file from a Google Drive link, extracts its contents,
    automatically detects subfolders, reads results.csv, and validates PNGs.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Starting ZIP download and validation process...")

    # Convert to direct download format
    if "drive.google.com/file/d/" in drive_link:
        file_id = drive_link.split("/file/d/")[1].split("/")[0]
        direct_url = f"https://drive.google.com/uc?id={file_id}"
    else:
        direct_url = drive_link

    # Step 1: Download the ZIP file
    zip_path = os.path.join(output_dir, "output.zip")
    print(f"Downloading ZIP from: {drive_link}")
    gdown.download(direct_url, zip_path, quiet=False, fuzzy=True)

    # Step 2: Extract ZIP contents
    print("Extracting ZIP contents...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Extraction complete. Files available in: {os.path.abspath(output_dir)}")

    # Step 3: Locate results.csv (even if inside subfolder)
    csv_path = None
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.lower() == "results.csv":
                csv_path = os.path.join(root, f)
                break
        if csv_path:
            break

    if not csv_path:
        raise FileNotFoundError("'results.csv' not found inside the extracted folder or subfolders.")

    print(f"Found results.csv at: {csv_path}")
    csv_data = pd.read_csv(csv_path)
    print(f"Loaded {len(csv_data)} rows from results.csv")

    # Step 4: Validate PNG files in the same directory
    folder_containing_csv = os.path.dirname(csv_path)
    png_files = [f for f in os.listdir(folder_containing_csv) if f.lower().endswith(".png")]
    png_set = set(png_files)

    filename_col = next((c for c in csv_data.columns if "file" in c.lower()), None)
    if filename_col:
        missing = [f for f in csv_data[filename_col] if f not in png_set]
    else:
        missing = []

    # Step 5: Validation summary
    print("\nValidation Summary")
    print("------------------------------------------------------------")
    print(f"Total PNG files found: {len(png_files)}")
    print(f"Rows in results.csv:   {len(csv_data)}")
    print(f"Missing image files:   {len(missing)}")
    if missing:
        print("Example missing files:", missing[:5])
    print("------------------------------------------------------------")

    return csv_data


generated_data = parse_and_verify_zip_from_drive(drive_link)


print("\n Parsed Data Preview:")
print(generated_data.head())


# Optional cosmetic renaming to match expected format
generated_data = generated_data.rename(columns={
    "file_name": "generated_images",
    "prompt_id": "run_id"   # purely placeholder; DreamLayer will not use it
})


print("\n Parsed Data Preview:")
print(generated_data.head())


# ==========================================================
# SECTION 4: Evaluation (DreamLayer Scoring)
# ==========================================================
# Purpose:
#   - Run evaluation code from official KaggleDreamLayer.ipynb
#   - Perform YOLO-based object detection + POS tagging
#   - Ground Truth Extraction
#   - Compute Composition Correctness (F1 Score)
# Guidelines followed:
#   - DO NOT modify evaluation logic or scoring functions
#   - You may add safe print() logs or progress messages
#   - Keep the same filenames and CSV structure for scoring
# Output:
#   - Displays F1 score on the public prompt set
# ==========================================================



def apply_object_detection(yolo_file, generated_df):
    model = YOLO(yolo_file)
    
    print("Verifying image paths before detection...")
    sample_img = os.path.join('kaggle_resources', 'output', generated_df['generated_images'].iloc[0])
    print("Sample path:", sample_img)
    print("Exists:", os.path.exists(sample_img))

    
    def detect_objects(img):
        try:
            detections = model(os.path.join('kaggle_resources', 'output', img))[0]
            return [model.names[int(cls)] for cls in detections.boxes.cls] if detections.boxes is not None else []
        except Exception as e:
            print(f"Error processing {img}: {e}")
            return []

    print("Running YOLO detection...")
    generated_df['predicted_objects'] = [detect_objects(img) for img in generated_df['generated_images']]
    print("Detection complete.")
    return generated_df



submissions_df = apply_object_detection(yolo_model, generated_data)


submissions_df[['generated_images', 'predicted_objects']].head(10)


def extract_ground_truth(drive_link):
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    from nltk.stem import WordNetLemmatizer
    
    lemmatizer = WordNetLemmatizer()
    doc_id = drive_link.split('/d/')[1].split('/')[0]
    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    gdown.download(export_url, 'temp_prompts.txt', quiet=False)
    with open('temp_prompts.txt', 'r') as f:
        prompts = f.readlines()
    
    results = []
    for i, prompt in enumerate(prompts):
        tokens = word_tokenize(prompt.lower().strip())
        pos_tags = pos_tag(tokens)
        nouns = [lemmatizer.lemmatize(word) for word, pos in pos_tags if pos.startswith('NN')]
        results.append({'ID': i+1, 'prompt_id': i, 'prompt': prompt.strip(), 'ground_truth': nouns, 'Usage': 'Public'})
    
    return pd.DataFrame(results)


solutions_df = extract_ground_truth(prompt_dataset)


solutions_df.head(10)


print(repr(solutions_df['prompt'].iloc[0]))


submissions_df['prompt'] = submissions_df['prompt'].astype(str).str.strip().str.replace("^\ufeff", "", regex=True)
solutions_df['prompt'] = solutions_df['prompt'].astype(str).str.strip().str.replace("^\ufeff", "", regex=True)


print(repr(solutions_df['prompt'].iloc[0]))


def add_id_to_submission(submission_df, solution_df):
    merged = solution_df[['ID', 'prompt']].merge(submission_df, on='prompt', how='left')
    merged['predicted_objects'] = merged['predicted_objects'].fillna('').apply(lambda x: x if isinstance(x, list) else [])
    merged[['run_id', 'generated_images']] = merged[['run_id', 'generated_images']].fillna("missing_information")
    return merged


submissions_df = add_id_to_submission(submissions_df, solutions_df)


print(submissions_df)


def score(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    """
    Calculate the average F1 score between predicted and ground truth objects using proper precision/recall.
    
    Args:
        solution: DataFrame with 'ID' and 'ground_truth' columns
        submission: DataFrame with 'ID' and 'predicted_objects' columns
    
    Returns:
        float: Average F1 score across all prompts using TP, FP, FN calculations
    """
    import ast
    merged = solution.merge(submission, on='ID', how='left')
    merged['predicted_objects'] = merged['predicted_objects'].fillna('[]').apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    merged['ground_truth'] = merged['ground_truth'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    f1_scores = []
    for pred, truth in zip(merged['predicted_objects'], merged['ground_truth']):
        pred_set, truth_set = set(pred), set(truth)
        tp = len(pred_set & truth_set)
        fp = len(pred_set - truth_set)
        fn = len(truth_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    return np.mean(f1_scores)


f1_score = score(solutions_df, submissions_df)
print(f1_score)


submissions_df.to_csv("submission.csv", index=False)


print(submissions_df.head(10))


# ==========================================================
# SECTION 5: Summary & Submission
# ==========================================================
# Purpose:
#   - Print summary of outputs (image count, config info, F1 score)
#   - Save submission.csv for Kaggle upload
#   - Optionally zip output folder for Drive upload
# Notes:
#   - submission.csv should contain:
#         prompt_id,file_name
#     (Optionally includes prompt column)
#   - Upload this CSV to the Kaggle competition “Submit Predictions” tab
#   - Share this notebook publicly or with organizers for final scoring
# ==========================================================





