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


# ==============================
#  Installs & Imports
# ==============================
!pip -q install --no-input diffusers==0.30.0 transformers==4.44.2 accelerate==0.34.2 safetensors==0.4.5
!pip -q install --no-input compel==2.0.3 ultralytics==8.3.23 nltk==3.9.1

import os, json, math, time, re, random, zipfile
import torch
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from IPython.display import display
import nltk
nltk.download(['punkt','averaged_perceptron_tagger','averaged_perceptron_tagger_eng','wordnet','omw-1.4'])



# ==============================
# High-F1 Realistic Image Generation
# ==============================
!pip install -q diffusers==0.31.0 transformers accelerate safetensors

import torch, os, math, time, psutil
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from IPython.display import display
from PIL import Image


# ----------------------------
# Configuration
# ----------------------------
BASE_MODEL = "stabilityai/stable-diffusion-2-1-base"
NEGATIVE_PROMPT = "cartoon, painting, sketch, text, watermark, deformed, blurry, unrealistic, drawing"
POSITIVE_PROMPT_TAG = "realistic, photo, detailed, natural lighting, 8k resolution"

INFERENCE_STEPS = 30
GUIDANCE_SCALE = 5.5
IMAGE_RESOLUTION = (512, 512)
BASE_SEED = 1234
device = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = "high_f1_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ==============================
#  Configuration
# ==============================
PROMPT_FILE = "/kaggle/input/text-to-image-challenge/DreamLayer-Prompt-Kaggle.txt"
OUT_BASE    = "/kaggle/working/improved_outputs3"
OUT_IMGDIR  = os.path.join(OUT_BASE, "refined")
os.makedirs(OUT_IMGDIR, exist_ok=True)



# ----------------------------
# Load Model
# ----------------------------
print("Loading Stable Diffusion 2.1 realistic model...")
scheduler = EulerDiscreteScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)
pipe.enable_attention_slicing()
torch.set_grad_enabled(False)
print(" Model loaded successfully!\n")


# ----------------------------
# Emphasize nouns (generic)
# ----------------------------
lemmatizer = WordNetLemmatizer()
def emphasize_nouns(prompt):
    tokens = word_tokenize(prompt)
    tagged = pos_tag(tokens)
    new_prompt = []
    for word, tag in tagged:
        if tag.startswith("NN"):
            new_prompt.append(f"({word}:1.15)")
        elif tag.startswith("VB"):
            new_prompt.append(f"{word}")
        elif tag.startswith("IN") or tag in ["CD"]:
            new_prompt.append(f"{word}")
        else:
            new_prompt.append(word)
    return " ".join(new_prompt)


# ----------------------------
# Prompt File (dataset)
# ----------------------------
prompt_file = "/kaggle/input/text-to-image-challenge/DreamLayer-Prompt-Kaggle.txt"
with open(prompt_file, "r", encoding="utf-8") as f:
    PROMPTS = [p.strip() for p in f.readlines() if p.strip() and not p.startswith("#")]
print(f"ğŸ“„ Loaded {len(PROMPTS)} prompts.")


# ==============================
# Model Configuration Summary (for Log)
# ==============================
print("\nğŸ”§ Model Configuration Summary")
print("-" * 70)
print(f"Model Name           : {BASE_MODEL}")
print(f"Scheduler            : {pipe.scheduler.__class__.__name__}")
print(f"Inference Steps      : {INFERENCE_STEPS}")
print(f"Guidance Scale       : {GUIDANCE_SCALE}")
print(f"Resolution (HÃ—W)     : {IMAGE_RESOLUTION}")
print(f"Device               : {device}")
print(f"Seed Base            : {BASE_SEED}")
print(f"Negative Prompt      : {NEGATIVE_PROMPT}")
print(f"Positive Context     : {POSITIVE_PROMPT_TAG}")
print(f"Prompt File Path     : {prompt_file}")
print(f"Output Directory     : {OUT_DIR}")
print("-" * 70)
print(f"Total Prompts Loaded : {len(PROMPTS)}")
print("Next, Starts image generation...\n")



# ----------------------------
# Generation Loop
# ----------------------------
results = []
start_time = time.time()

for i, raw_prompt in enumerate(PROMPTS, start=1):
    rewritten = f"{POSITIVE_PROMPT_TAG}, {emphasize_nouns(raw_prompt)}"
    seed = BASE_SEED + i
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print(f"\n--- [{i}/{len(PROMPTS)}] Generating: {raw_prompt}")
    
    image = pipe(
        prompt=rewritten,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=IMAGE_RESOLUTION[0],
        width=IMAGE_RESOLUTION[1],
        generator=generator
    ).images[0]

    filename = f"{i:04}.png"
    image.save(os.path.join(OUT_DIR, filename))
    results.append({"prompt_id": i, "prompt": raw_prompt, "file_name": filename})

    # Show preview every 5 images
    if i % 5 == 0 or i == len(PROMPTS):
        fig, axs = plt.subplots(1, 5, figsize=(20, 4))
        for j, recent in enumerate(results[-5:]):
            img_path = os.path.join(OUT_DIR, recent["file_name"])
            img = Image.open(img_path)
            axs[j].imshow(img)
            axs[j].set_title(f"#{recent['prompt_id']} {recent['prompt'][:30]}...", fontsize=8)
            axs[j].axis("off")
        plt.tight_layout()
        display(fig)
        plt.close(fig)

    torch.cuda.empty_cache()

elapsed = (time.time() - start_time) / 60
print(f"\n Generation complete in {elapsed:.2f} minutes!")
print(f"Images saved in: {OUT_DIR}")


# ==============================
# Save Results, Config, and Zip for Submission
# ==============================
import json, zipfile

# --- 1. Save results.csv ---
df = pd.DataFrame(results)
csv_path = os.path.join(OUT_DIR, "results.csv")
df.to_csv(csv_path, index=False)
print(f" Saved results file: {csv_path}")

# --- 2. Save config.json ---
config_data = {
    "model_name": BASE_MODEL,
    "scheduler": pipe.scheduler.__class__.__name__,
    "inference_steps": INFERENCE_STEPS,
    "guidance_scale": GUIDANCE_SCALE,
    "resolution": IMAGE_RESOLUTION,
    "negative_prompt": NEGATIVE_PROMPT,
    "positive_prompt_context": POSITIVE_PROMPT_TAG,
    "seed_base": BASE_SEED,
    "num_prompts": len(PROMPTS),
    "output_directory": OUT_DIR,
}
config_path = os.path.join(OUT_DIR, "config.json")
with open(config_path, "w") as f:
    json.dump(config_data, f, indent=4)
print(f" Saved configuration file: {config_path}")

# --- 3. Verify folder contents ---
print("\n Folder Summary:")
for root, _, files in os.walk(OUT_DIR):
    for file in files:
        path = os.path.join(root, file)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {file:<30} {size_mb:6.2f} MB")

# --- 4. Zip everything for download/upload ---
zip_name = f"{OUT_DIR}.zip"
zip_path = os.path.join("/kaggle/working", zip_name)
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(OUT_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, OUT_DIR)
            zipf.write(file_path, arcname)
print(f"\n Created ZIP: {zip_path}")

# --- 5. Summary ---
print("\n All files ready for evaluation & submission!")
print(f"Results CSV : {csv_path}")
print(f"Config JSON : {config_path}")
print(f"ZIP File    : {zip_path}")



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
drive_link = "https://drive.google.com/file/d/10NkcwjDJxf0H4eliUfhCWHltfq-FYKK9/view?usp=sharing"
prompt_dataset = "https://docs.google.com/document/d/1VJHTKDMPapyLbACAIOWvzRdrUhV70y03Mbyr_y--AN0/edit?usp=sharing"


# ==========================================================
#  Drive Link Clarification â€” Explaining folder vs. zip approach
# ==========================================================

expected_drive_link = "https://drive.google.com/drive/folders/1xlvP3D9Db6nKpwZLlNmXhnGsw6C3qFx0?usp=sharing"
zipped_drive_link = "https://drive.google.com/file/d/10NkcwjDJxf0H4eliUfhCWHltfq-FYKK9/view?usp=sharing"

print(" DreamLayer Drive Link Verification")
print("------------------------------------------------------------")
print(f"Expected Drive folder structure link:\n{expected_drive_link}\n")
print(
    "This Drive folder contains the expected DreamLayer output files â€” "
    "all generated images (0001.png, 0002.png, ...), 'results.csv', "
    "and 'config-dreamlayer.json'.\n\n"
    "However, when using gdown.download_folder(), the process throws a "
    "FolderContentsMaximumLimitError. This happens even though the folder "
    "has the correct structure, because gdown imposes a strict limit on "
    "listing Drive folder contents (â‰ˆ50 items).\n"
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
#  Evaluation (DreamLayer Scoring)
# ==========================================================


def apply_object_detection(yolo_file, generated_df):
    model = YOLO(yolo_file)
    
    print("Verifying image paths before detection...")
    # /kaggle/working/high_f1_outputs/0001.png
    sample_img = os.path.join('/kaggle', 'working', 'high_f1_outputs', generated_df['generated_images'].iloc[0])
    print("Sample path:", sample_img)
    print("Exists:", os.path.exists(sample_img))

    
    def detect_objects(img):
        try:
            detections = model(os.path.join('/kaggle', 'working', 'high_f1_outputs', img))[0]
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


## To try later:
def emphasize_nouns(prompt):
    tokens = word_tokenize(prompt)
    tagged = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()

    nouns = [lemmatizer.lemmatize(w) for w, t in tagged if t.startswith("NN")]
    enhanced_prompt = prompt.strip()

    # Amplify core nouns
    for noun in nouns:
        enhanced_prompt += f", full view of {noun}, clearly visible {noun}"

    # Add realism & scene anchors
    enhanced_prompt = (
        f"realistic detailed photograph, natural lighting, {enhanced_prompt}, "
        "accurate colors, high clarity, camera focus on main objects"
    )
    return enhanced_prompt





