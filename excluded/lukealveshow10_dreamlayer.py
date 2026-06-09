# ============================================================
# TEXT-TO-IMAGE CHALLENGE - DREAMLAYER
# Kaggle Notebook - Lucas Alves Martins
# ============================================================
drive_link = "https://drive.google.com/drive/folders/17ceDU2wQ8RdFxX2D_FJYRK6r5WtUejiv?usp=drive_link"
print("Drive link configurado:", drive_link)


!pip install diffusers transformers accelerate safetensors pillow
!pip install ultralytics opencv-python pandas


import torch
from pathlib import Path
from PIL import Image
import pandas as pd
from diffusers import StableDiffusionPipeline
from ultralytics import YOLO
import random, json


# Configuração
prompts_file = "/kaggle/input/text-to-image-challenge/DreamLayer-Prompt-Kaggle.txt"
out_dir = Path("output_images")
out_dir.mkdir(exist_ok=True)

model_id = "runwayml/stable-diffusion-v1-5"   # Modelo de difusão
device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 42
generator = torch.Generator(device=device).manual_seed(seed)

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
)
pipe = pipe.to(device)

print("Modelo carregado:", model_id)


with open(prompts_file, 'r', encoding='utf-8') as f:
    prompts = [l.strip() for l in f if l.strip() and not l.startswith("#")]

print("Total de prompts carregados:", len(prompts))
print("Exemplo:", prompts[:5])


for i, prompt in enumerate(prompts, start=1):
    filename = f"{i:04d}.png"
    out_path = out_dir / filename
    
    image = pipe(
        prompt,
        guidance_scale=7.5,
        num_inference_steps=30,
        generator=generator
    ).images[0]
    
    image.save(out_path)
    
    if i <= 3:
        display(image)
    print("Generate:", out_path)


model = YOLO("yolov8n.pt")
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
df["predictions"] = df["predictions"].fillna("")  # se faltar predição, deixa vazio
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


print("✅ Images and .CSVs on Google Drive folder.")
print("Link to access the Image and CSVs results:", drive_link)

