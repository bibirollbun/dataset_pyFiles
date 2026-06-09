# ============================================================
# TEXT-TO-IMAGE CHALLENGE - DREAMLAYER
# Kaggle Notebook - Lucas Alves Martins
# ============================================================
drive_link = "https://drive.google.com/drive/folders/1r3rudNpli51mfMoH_h4qv7XA-SnCMEJf?usp=share_link"
print("Drive link configurado:", drive_link)


!pip install diffusers transformers accelerate safetensors pillow
!pip install ultralytics opencv-python pandas


import torch
from pathlib import Path
from PIL import Image
import pandas as pd
from diffusers import StableDiffusionXLPipeline
from diffusers import DiffusionPipeline
from ultralytics import YOLO
import random, json
# from huggingface_hub import login
# login("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")
import os


# --- Setup ---
prompts_file = "/kaggle/input/text-to-image-challenge/DreamLayer-Prompt-Kaggle.txt"
out_dir = Path("output_images")
out_dir.mkdir(exist_ok=True)

# Choose model
model_id = "SG161222/RealVisXL_V4.0_Lightning"    
device = "cuda" if torch.cuda.is_available() else "cpu"

# Seed
seed = 42
generator = torch.Generator(device=device).manual_seed(seed)

# --- Load pipeline ---
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    use_safetensors=True
).to(device)

# Enable safe memory management (Kaggle-friendly)
pipe.enable_attention_slicing()
pipe.enable_sequential_cpu_offload()

print("âœ… Modelo carregado:", model_id)


with open(prompts_file, 'r', encoding='utf-8') as f:
    prompts = [l.strip() for l in f if l.strip() and not l.startswith("#")]

print("Total de prompts carregados:", len(prompts))
print("Exemplo:", prompts[:5])


for i, prompt in enumerate(prompts, start=1):
    filename = f"{i:04d}.png"
    out_path = out_dir / filename
    post_prefix = " ,detailed, 8k"
    
    image = pipe(
        prompt + post_prefix,
        guidance_scale=9.5,
        num_inference_steps=45,
        generator=generator
    ).images[0]
    
    image.save(out_path)
    
    if i <= 5:
        display(image)
        print(prompt)
    print("Generated:", out_path)
    torch.cuda.empty_cache()


model = YOLO("yolov8x.pt")
results = []

for img_path in sorted(out_dir.glob("*.png")):
    r = model(img_path, verbose=False)
    labels = set()
    
    for det in r:
        for box in det.boxes:
            cls = int(box.cls.item())
            name = det.names[cls]
            labels.add(name)
    
    prompt_id = img_path.stem  # ex: "0001"
    results.append({
        "ID": prompt_id,
        "predictions": ";".join(sorted(labels))
    })

df = pd.DataFrame(results)

expected_ids = [f"{i:04d}" for i in range(1, 50)]
df = df.set_index("ID").reindex(expected_ids).reset_index()
df["predictions"] = df["predictions"].fillna("")  # se faltar prediÃ§Ã£o, deixa vazio
df["ID"] = df["ID"].astype(str)
df.to_csv("results.csv", index=False, encoding='utf-8')
df.to_csv("submission.csv", index=False, encoding='utf-8')

df.head(50)


config = {
    "model": model_id,
    "seed": seed,
    "guidance_scale": 7.5,
    "num_inference_steps": 30,
    "num_prompts": len(prompts)
}
with open("config-dreamlayer.json", "w") as f:
    json.dump(config, f, indent=4)

print("File config-dreamlayer.json created")


print("âœ… Images and .CSVs on Google Drive folder.")
print("Link to access the Image and CSVs results:", drive_link)


# List what you produced
import os, glob, textwrap, subprocess, pandas as pd

print("Working dir:", os.getcwd())
print("\nTop-level files:")
!ls -lh

print("\nA few images:")
!ls -lh output_images | head -n 10

# Quick sanity checks
assert os.path.exists("submission.csv"), "submission.csv not found."
assert os.path.exists("results.csv"), "results.csv not found."
assert os.path.isdir("output_images"), "output_images/ not found."


import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')  # <--- add this
nltk.download('stopwords')


from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
import re

stop = set(stopwords.words('english'))

def extract_expected_objects(prompt):
    tokens = [w.lower() for w in word_tokenize(prompt) if re.match(r"[A-Za-z]+", w)]
    tagged = pos_tag(tokens)
    nouns = [word for word, pos in tagged if pos.startswith("NN") and word not in stop]
    return set(nouns)

prompt = "A zebra chews a flower in a fenced in field."
print(extract_expected_objects(prompt))


import pandas as pd

df = pd.read_csv("results.csv")
df.head()


import pandas as pd
import nltk, re
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords

# --- Setup NLTK ---
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')

stop = set(stopwords.words('english'))

# --- Helper functions ---
def extract_expected_objects(prompt):
    tokens = [w.lower() for w in word_tokenize(prompt) if re.match(r"[A-Za-z]+", w)]
    tagged = pos_tag(tokens)
    nouns = [word for word, pos in tagged if pos.startswith("NN") and word not in stop]
    return set(nouns)

def to_set(s):
    if isinstance(s, str):
        return set([x.strip().lower() for x in re.split('[;,]', s) if x.strip()])
    return set()

def f1_score_per_prompt(expected, detected):
    if not expected and not detected:
        return 1.0
    if not expected or not detected:
        return 0.0
    tp = len(expected & detected)
    fp = len(detected - expected)
    fn = len(expected - detected)
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    return 0 if (p + r) == 0 else 2 * p * r / (p + r)

# --- Load data ---
prompts_path = "/kaggle/input/text-to-image-challenge/DreamLayer-Prompt-Kaggle.txt"
submission_path = "submission.csv"

# Read prompts
prompts = []
with open(prompts_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)

# Read submission (ID,predictions)
df = pd.read_csv(submission_path)
df['prompt'] = prompts[:len(df)]

# Extract objects
df['expected'] = df['prompt'].apply(extract_expected_objects)
df['detected'] = df['predictions'].apply(to_set)

# Compute F1 per prompt
df['F1'] = df.apply(lambda r: f1_score_per_prompt(r['expected'], r['detected']), axis=1)

# Save per-image F1 for reference
df.to_csv("per_image_f1_results.csv", index=False)

# Display summary
overall_f1 = df['F1'].mean()
print(f"ðŸ“Š Average F1 for guidance=9.5 & prefix='a detailed realistic photo of': {overall_f1:.4f}")

# Show first few detailed results
for i in range(min(5, len(df))):
    print(f"\nImage {df.loc[i,'ID']} â€” Prompt: {df.loc[i,'prompt']}")
    print("Expected:", df.loc[i,'expected'])
    print("Detected:", df.loc[i,'detected'])
    print("F1:", round(df.loc[i,'F1'], 3))

