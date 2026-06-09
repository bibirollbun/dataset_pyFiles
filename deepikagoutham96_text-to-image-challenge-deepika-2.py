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


# Minimal installation for Kaggle runtime (faster and compatible)
!pip install diffusers==0.30.0 transformers==4.45.0 compel==2.0.2 safetensors==0.4.5 accelerate==0.34.2 --quiet


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
drive_link = "https://drive.google.com/drive/folders/1ZiiOqfN7WDKvUFOVGp99j8Y8TOQrlhwo?usp=sharing"
prompt_dataset = "https://docs.google.com/document/d/1VJHTKDMPapyLbACAIOWvzRdrUhV70y03Mbyr_y--AN0/edit?usp=sharing"


# ==============================
# Improved F1 Image Generation Setup
# ==============================

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler
from compel import Compel
from PIL import Image
import pandas as pd
import os

# --- Configurable parameters ---
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
REFINER_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"

INFERENCE_STEPS_BASE = 40
INFERENCE_STEPS_REFINER = 20
GUIDANCE_SCALE = 8.0
IMAGE_RESOLUTION = (768, 768)

NEGATIVE_PROMPT = "low quality, blurry, text, watermark, cropped, deformed, out of frame, duplicate, lowres"

# Output directories
os.makedirs("improved_outputs/base", exist_ok=True)
os.makedirs("improved_outputs/refined", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")



# ==============================
# Weighted SDXL Base + Refiner Generator
# ==============================

import nltk, random, time
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from PIL import Image
import torch
from compel import Compel
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler

nltk.download(["punkt", "averaged_perceptron_tagger", "wordnet", "omw-1.4"], quiet=True)


import torch
from diffusers import DiffusionPipeline

torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)

#  Enable memory-efficient features
from accelerate import infer_auto_device_map, init_empty_weights
import gc, os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"

# Optional: automatic CPU offload for attention layers
enable_offload = True



# ---------- Load base + refiner (VRAM-Safe) ----------
print("Loading SDXL Base and Refiner models (VRAM-optimized)...")

import os, torch, gc
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler
from compel import Compel
from nltk.stem import WordNetLemmatizer

# --- CUDA & memory configuration ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)

# --- Use CPU offload if VRAM is tight ---
enable_offload = True

# --- Load Base model ---
pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

if enable_offload:
    pipe.enable_model_cpu_offload()
    print(" Enabled CPU offload for Base model")
else:
    pipe = pipe.to(device)

# --- Load Refiner model ---
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    REFINER_MODEL,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
if enable_offload:
    refiner.enable_model_cpu_offload()
    print(" Enabled CPU offload for Refiner model")
else:
    refiner = refiner.to(device)

# --- Setup Compel & Lemmatizer ---
compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
lemmatizer = WordNetLemmatizer()

print(" SDXL Base + Refiner loaded successfully with memory optimization.")



# ---------- Prompt-weighted generation ----------
import re
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

def emphasize_nouns(
    prompt: str,
    noun_w=1.40, adj_w=1.15, verb_w=1.10,
    count_w=1.60, rel_w=1.20, part_w=1.25,
    pair_w=1.35, sep_w=1.25, state_w=1.20
):
    """
    Generic F1-oriented prompt enhancer for SDXL to improve YOLO detection.

     Emphasizes:
      â€¢ Nouns â†’ ensures object presence
      â€¢ Counts â†’ ensures correct number of instances
      â€¢ Relations/Prepositions â†’ clarifies spatial layout
      â€¢ Parts (slice/piece/half/etc.) â†’ shows visible sections
      â€¢ Fullness/Overflow states â†’ depicts realistic object density
      â€¢ Verbs â†’ adds light relational meaning when relevant

     Avoids:
      â€¢ Object-specific hacks
      â€¢ Dataset-specific biases
    """

    if not prompt or not prompt.strip():
        return prompt

    text = prompt.strip().lower()
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)

    # ----- Core mappings -----
    num_words = {
        "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,
        "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
        "pair":2,"couple":2,"few":3,"several":4,"many":6,
        "multiple":5
    }

    quantity_words = {"pair","couple","few","several","many","multiple"}

    spatial_phrases = [
        "next to","beside","by his side","by her side","by their side",
        "on top of","in front of","in the middle of","in the center of",
        "left of","right of","above","below","under","between","near"
    ]

    part_terms = {
        "slice","piece","half","section","part","tip","edge","corner",
        "middle","center","top","bottom","side"
    }

    fullness_terms = {
        "full","filled","overflow","overflowing","crowded","packed","stuffed","loaded"
    }

    relation_verbs = {
        "hold","holding","carry","carrying","put","placing","place",
        "pour","wear","wearing","grasp","grasping","contain","fill"
    }

    # ----- POS-based emphasis -----
    emphasized = []
    for word, pos in tags:
        if re.fullmatch(r"\d+", word):
            emphasized.append(f"({word}:{count_w})")
        elif pos.startswith("NN"):
            emphasized.append(f"({word}:{noun_w})")
        elif pos.startswith("JJ"):
            emphasized.append(f"({word}:{adj_w})")
        elif pos.startswith("VB") and word in relation_verbs:
            emphasized.append(f"({word}:{verb_w})")
        elif word in num_words:
            emphasized.append(f"({word}:{count_w})")
        else:
            emphasized.append(word)
    rewritten = " ".join(emphasized)

    # ----- Quantity emphasis -----
    plural_intent = False
    if any(re.search(rf"\b{k}\b", text) for k in num_words.keys()):
        if not re.search(r"\bone\b", text):
            plural_intent = True
    if re.search(r"\b\d+\b", text) and not re.search(r"\b1\b", text):
        plural_intent = True

    if plural_intent:
        rewritten += f" (side by side:{pair_w}) (separate objects:{sep_w}) (not overlapping:{rel_w})"

    # ----- Spatial relations -----
    for phrase in spatial_phrases:
        if phrase in text:
            if phrase in {"next to","beside","by his side","by her side","by their side","between"}:
                rewritten += f" (side by side:{pair_w}) (separate objects:{sep_w}) (not overlapping:{rel_w})"
            elif phrase in {"above","below","under","left of","right of","in front of"}:
                rewritten += f" (clear spatial arrangement:{rel_w}) (not overlapping:{rel_w})"
            elif phrase == "on top of":
                rewritten += f" (stacked correctly:{rel_w})"

    if "next to" in text or "beside" in text:
        rewritten += f" (not on top:{rel_w})"

    # ----- Part / section emphasis -----
    if any(t in text for t in part_terms):
        rewritten += f" (part clearly visible:{part_w}) (section detail:{part_w})"
    if re.search(r"\b(missing|removed|cut|cut out|section missing|slice missing)\b", text):
        rewritten += f" (section cut out:{part_w}) (visible cut area:{part_w})"

    # ----- Fullness / overflow states -----
    if any(t in text for t in fullness_terms):
        rewritten += f" (show fullness:{state_w}) (overflow visible:{state_w}) (filled to capacity:{state_w})"

    # ----- Container relations -----
    if re.search(r"\b(holding|carry|carrying|with|into|inside|in)\b", text):
        rewritten += f" (both objects visible:{sep_w}) (separate objects:{sep_w})"

    # ----- Clean formatting -----
    rewritten = re.sub(r"\s+", " ", rewritten).strip()
    return rewritten


# Preview utility
def preview_rewrites(prompts, n=5):
    print("Previewing prompt rewrites (F1-optimized generic mode):")
    for i, p in enumerate(prompts[:n], start=1):
        print(f"\n[{i}] Original : {p}")
        print(f"    Rewritten: {emphasize_nouns(p)}")




def generate_with_refiner(prompts, output_dir="improved_outputs"):
    results = []
    for i, raw_prompt in enumerate(tqdm(prompts, desc="Generating images")):
        seed = 1234 + i
        generator = torch.Generator(device=device).manual_seed(seed)
        prompt = emphasize_nouns(raw_prompt.strip())

        try:
            # --- Stage 1: Base SDXL ---
            base_image = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=INFERENCE_STEPS_BASE,
                guidance_scale=GUIDANCE_SCALE,
                height=IMAGE_RESOLUTION[0],
                width=IMAGE_RESOLUTION[1],
                generator=generator
            ).images[0]

            # --- Stage 2: Refiner ---
            refined = refiner(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                image=base_image,
                strength=0.3,
                num_inference_steps=INFERENCE_STEPS_REFINER,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator
            ).images[0]

            fname = f"{i+1:04}.png"
            refined.save(os.path.join(output_dir, "refined", fname))

            results.append({"prompt_id": i+1, "prompt": raw_prompt.strip(), "file_name": fname})
            if (i+1) % 5 == 0:
                print(f" Saved {i+1} images so far...")

        except Exception as e:
            print(f" Error generating image {i+1}: {e}")
            continue

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "results.csv"), index=False)
    print("\nGeneration complete! All results saved to:", os.path.join(output_dir, "results.csv"))
    return df



# ============================================
#  Set Prompt File Path (Kaggle Dataset)
# ============================================
prompt_file = "/kaggle/input/text-to-image-challenge/DreamLayer-Prompt-Kaggle.txt"

# Quick check
if os.path.exists(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    print(f" Found prompt file with {len(lines)} usable prompts.")
else:
    raise FileNotFoundError(f" Prompt file not found at: {prompt_file}")



# ==============================
# SDXL Generation (Full Logs + Stacked Image Previews)
# ==============================
import matplotlib.pyplot as plt
from IPython.display import display
import time, math, torch, os, psutil
from PIL import Image

# â”€â”€â”€ Print Configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(" Model Configuration Summary")
print("-" * 70)
print(f"Base Model           : {BASE_MODEL}")
print(f"Refiner Model        : {REFINER_MODEL}")
print(f"Inference Steps Base : {INFERENCE_STEPS_BASE}")
print(f"Inference Steps Ref. : {INFERENCE_STEPS_REFINER}")
print(f"Scheduler            : DPMSolverMultistep (Karras)")
print(f"Guidance Scale       : {GUIDANCE_SCALE}")
print(f"Resolution (HÃ—W)     : {IMAGE_RESOLUTION}")
print(f"Negative Prompt      : {NEGATIVE_PROMPT}")
print(f"Device               : {device}")
print(f"Prompt File          : {prompt_file}")
print(f"Output Directory     : improved_outputs/refined")
print("-" * 70)
time.sleep(1)

# â”€â”€â”€ Load Prompts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with open(prompt_file, "r", encoding="utf-8") as f:
    all_prompts = [p.strip() for p in f.readlines()
                   if p.strip() and not p.strip().startswith("#")]

print(f" Total prompts detected: {len(all_prompts)}")
print("Starting generation with persistent logs...\n")

os.makedirs("improved_outputs/refined", exist_ok=True)
results = []
start_time = time.time()

# â”€â”€â”€ VRAM-Safe Setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)

# â”€â”€â”€ Generation Loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for i, raw_prompt in enumerate(all_prompts, start=1):
    seed = 1234 + i
    generator = torch.Generator(device=device).manual_seed(seed)
    weighted_prompt = emphasize_nouns(raw_prompt)

    print("\n" + "=" * 80)
    print(f"â–¶ Generating Prompt #{i}/{len(all_prompts)}")
    print(f"Prompt: {raw_prompt}")
    print("-" * 80)

    try:
        # --- Base Generation ---
        base_img = pipe(
            prompt=weighted_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=INFERENCE_STEPS_BASE,
            guidance_scale=GUIDANCE_SCALE,
            height=IMAGE_RESOLUTION[0],
            width=IMAGE_RESOLUTION[1],
            generator=generator
        ).images[0]

        # --- Refiner Pass (light mode) ---
        refined = refiner(
            prompt=weighted_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=base_img,
            strength=0.3,
            num_inference_steps=min(10, INFERENCE_STEPS_REFINER),
            guidance_scale=GUIDANCE_SCALE,
            generator=generator
        ).images[0]

        # --- Save Image ---
        filename = f"{i:04}.png"
        save_path = os.path.join("improved_outputs", "refined", filename)
        refined.save(save_path)
        results.append({"prompt_id": i, "prompt": raw_prompt, "file_name": filename})

        # --- Display Preview (Persistent) ---
        img = Image.open(save_path)
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.title(f"#{i} â€“ {raw_prompt[:60]}...", fontsize=8)
        plt.axis("off")
        display(plt.gcf())
        plt.close()

        # --- Status Info ---
        elapsed = time.time() - start_time
        print(f"âœ” Done | Time elapsed: {elapsed/60:.1f} min")
        print(f"   GPU VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"   System RAM: {psutil.virtual_memory().percent}%")
        print(f"   Saved: {save_path}")

        # --- Cleanup to Free VRAM ---
        del base_img, refined, img
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error on prompt #{i}: {e}")
        torch.cuda.empty_cache()
        continue

# â”€â”€â”€ Save Results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
df = pd.DataFrame(results)
csv_path = os.path.join("improved_outputs", "results.csv")
df.to_csv(csv_path, index=False)

print("\nGeneration completed successfully!")
print(f"Images saved to: improved_outputs/refined/")
print(f"Results CSV    : {csv_path}")
print(f"Total images   : {len(df)}")



# ==============================
# Save Results, Config, Verify & Zip for Submission
# ==============================
import json
import zipfile

# --- 1ï¸� Paths and Basic Setup ---
base_dir = "/kaggle/working/improved_outputs"
refined_dir = os.path.join(base_dir, "refined")
csv_path = os.path.join(base_dir, "results.csv")
config_path = os.path.join(base_dir, "config-dreamlayer.json")

# --- 2ï¸� Save Configuration File ---
config_data = {
    "base_model": BASE_MODEL,
    "refiner_model": REFINER_MODEL,
    "scheduler": "DPMSolverMultistepScheduler (Karras)",
    "guidance_scale": GUIDANCE_SCALE,
    "resolution": f"{IMAGE_RESOLUTION[0]}x{IMAGE_RESOLUTION[1]}",
    "inference_steps_base": INFERENCE_STEPS_BASE,
    "inference_steps_refiner": INFERENCE_STEPS_REFINER,
    "negative_prompt": NEGATIVE_PROMPT,
    "prompt_file": prompt_file,
    "num_prompts": len(all_prompts),
    "output_dir": refined_dir,
    "seed_strategy": "1234 + i",
    "prompt_rewriter": "emphasize_nouns (F1-optimized generic version)"
}

with open(config_path, "w") as f:
    json.dump(config_data, f, indent=4)
print(f" Config file saved at: {config_path}")

# --- 3ï¸� Verify Files ---
print("\n Preview: results.csv (first 10 rows)")
results_df = pd.read_csv(csv_path)
display(results_df.head(10))

print("\n Preview: config-dreamlayer.json")
with open(config_path, "r") as f:
    print(f.read())

# --- 4ï¸� Create ZIP Archive for Easy Download / Submission ---
zip_filename = "DreamLayer_Improved_Submission.zip"
zip_path = os.path.join("/kaggle/working", zip_filename)

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(refined_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, base_dir)
            zipf.write(file_path, os.path.join("DreamLayer_Submission", arcname))
    # Include metadata files
    zipf.write(csv_path, "DreamLayer_Submission/results.csv")
    zipf.write(config_path, "DreamLayer_Submission/config-dreamlayer.json")

# --- 5ï¸� Summary ---
print("\n ZIP Archive Created Successfully!")
print(f" ZIP Path : {zip_path}")
print(f"  Images  : {len(results_df)}")
print(f" CSV File : {csv_path}")
print(f"  Config  : {config_path}")
print("-" * 70)
print(" All files follow DreamLayer competition folder format.")



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
drive_link = "https://drive.google.com/file/d/1j8u-cCYWnQyZ59P1P9xZ3li_rf7-yrnX/view?usp=sharing"
prompt_dataset = "https://docs.google.com/document/d/1VJHTKDMPapyLbACAIOWvzRdrUhV70y03Mbyr_y--AN0/edit?usp=sharing"


# ==========================================================
#  Drive Link Clarification â€” Explaining folder vs. zip approach
# ==========================================================

expected_drive_link = "https://drive.google.com/drive/folders/1ZiiOqfN7WDKvUFOVGp99j8Y8TOQrlhwo?usp=sharing"
zipped_drive_link = "https://drive.google.com/file/d/1j8u-cCYWnQyZ59P1P9xZ3li_rf7-yrnX/view?usp=sharing"

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
# Evaluation (DreamLayer Scoring)
# ==========================================================


def apply_object_detection(yolo_file, generated_df):
    model = YOLO(yolo_file)
    
    print("Verifying image paths before detection...")
    # /kaggle/working/kaggle_resources/output2
    sample_img = os.path.join('/kaggle', 'working', 'kaggle_resources', 'output2', generated_df['generated_images'].iloc[0])
    print("Sample path:", sample_img)
    print("Exists:", os.path.exists(sample_img))

    
    def detect_objects(img):
        try:
            detections = model(os.path.join('/kaggle', 'working', 'kaggle_resources', 'output2', img))[0]
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




