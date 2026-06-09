
DRIVE_FOLDER_LINK = "https://drive.google.com/drive/folders/1Y4-DnH2W27_jvpXTrkwFIPYAvzj_vk-m?dmr=1&ec=wgc-drive-globalnav-goto"  
TEAM_NAME = "your_team_name"    
AUTHOR = "Your Name"           
SEED = 42                     
NOTE = "Generated with DreamLayer / SD pipeline"  
# ===========================================================
print("Drive link (must point to folder that will contain images + results):", DRIVE_FOLDER_LINK)




%pip install --upgrade pip
%pip install diffusers transformers accelerate safetensors torch torchvision
%pip install ultralytics        
%pip install spacy
!python -m spacy download en_core_web_sm








with open("/kaggle/input/text-to-image-challenge/DreamLayer-Prompt-Kaggle.txt", "r") as f:
    lines = f.readlines()

# Keep only non-empty lines that do NOT start with "#"
prompts = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

# Convert into DataFrame

print(prompts)


# !pip install --upgrade --quiet numpy==1.26.4 scipy==1.14.1 "transformers>=4.44.0,<4.46.0" "diffusers>=0.29.0,<0.31.0"

!pip install diffusers transformers accelerate safetensors pillow
!pip install ultralytics opencv-python pandas



# import torch
# from pathlib import Path
# from PIL import Image
# import pandas as pd
# from diffusers import StableDiffusionPipeline
# from ultralytics import YOLO
# import random, json


# import torch
# from diffusers import StableDiffusionPipeline
# from transformers import CLIPTokenizer

# print("Torch:", torch.__version__)
# print("Diffusers import OK ✅")

!pip install --upgrade --quiet transformers==4.45.2 diffusers==0.30.0 accelerate==0.34.2
!pip install --upgrade --quiet numpy==1.26.4


import torch
from pathlib import Path
from PIL import Image
import pandas as pd
from diffusers import StableDiffusionPipeline
from ultralytics import YOLO
import random, json


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


# !pip install --upgrade --quiet numpy==1.26.4



out_dir = Path("output_images")
out_dir.mkdir(exist_ok=True)

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


from pathlib import Path
from ultralytics import YOLO
import pandas as pd

# 1. Collect all generated PNG files in /content
image_dir = Path("/content")
image_files = sorted(image_dir.glob("*.png"))   # gives a list of Path objects

# 2. Run YOLO
model = YOLO("yolov8n.pt")
results = []

for img_path in image_files:
    r = model(str(img_path), verbose=False)  # YOLO accepts str or Path
    labels = set()

    for det in r:
        for box in det.boxes:
            cls = int(box.cls.item())
            name = det.names[cls]
            labels.add(name)

    prompt_id = img_path.stem  # works since img_path is a Path
    results.append({
        "ID": prompt_id,
        "predictions": ";".join(sorted(labels))
    })

# 3. Build dataframe
df_exp = pd.DataFrame(results)

expected_ids = [f"{i:04d}" for i in range(1, 50)]
df_exp = df_exp.set_index("ID").reindex(expected_ids).reset_index()
df_exp["predictions"] = df_exp["predictions"].fillna("")
df_exp["ID"] = df_exp["ID"].astype(str)

# 4. Save results
df_exp.to_csv("results.csv", index=False, encoding="utf-8")
df_exp.to_csv("submission.csv", index=False, encoding="utf-8")

df_exp.head(50)






