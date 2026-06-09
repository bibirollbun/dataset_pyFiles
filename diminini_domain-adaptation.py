!nvidia-smi


import shutil, os
import time
from pathlib import Path
import pandas as pd
import random


INPUT_DIR = "/kaggle/input"
WORKING_DIR = "/kaggle/working"


DATASET_INPUT_DIR = f"{INPUT_DIR}/test-tools/dataset"
DATASET_WORKING_DIR = f"{WORKING_DIR}/dataset"

# enables caching for faster training
if os.path.isdir(DATASET_WORKING_DIR):
    shutil.rmtree(DATASET_WORKING_DIR)

shutil.copytree(DATASET_INPUT_DIR, DATASET_WORKING_DIR)


REPO_DIR = f"{WORKING_DIR}/repos"
if not os.path.isdir(REPO_DIR):
    os.mkdir(REPO_DIR)

CUT_DIR = f"{REPO_DIR}/CUT"
WCT2_DIR = f"{REPO_DIR}/WCT2"

os.chdir(REPO_DIR)

if not os.path.isdir(CUT_DIR):
    !git clone https://github.com/taesungp/contrastive-unpaired-translation.git CUT

os.chdir(CUT_DIR)
!pip install -r requirements.txt

os.chdir(REPO_DIR)

if not os.path.isdir(WCT2_DIR):
    !git clone https://github.com/clovaai/WCT2.git

!pip install dominate


SCENARIO = 1
EXPERIMENT_CONFIG = '1distractors_T0'  # 0baseline, 1distractors_T0, 1distractors_T1, 2background, 3materials, 4lighting, 6wct
seeds = [1, 10, 42]


os.chdir(WORKING_DIR)

start_time = time.time()

for seed in seeds:
    print(f"Start training for scenario {SCENARIO+1} {EXPERIMENT_CONFIG} with seed {seed}")
    intermediate_time = time.time()

    model = YOLO("yolov8m.pt")
    
    results = model.train(data=f"{DATASET_WORKING_DIR}/scenario{SCENARIO+1}/{EXPERIMENT_CONFIG}.yaml",
                          epochs=100,
                          imgsz=640,
                          project=f"{SCENARIO}",
                          name=f"{EXPERIMENT_CONFIG}_{seed}",
                          seed=seed,
                          freeze=9,
                          exist_ok=True,
                          cache=True)

    
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - intermediate_time))
    print(f'Training took {elapsed_time}')

elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
print(f'Total took {elapsed_time}')


import os
import pandas as pd
from ultralytics import YOLO
import numpy as np

os.chdir(WORKING_DIR)

all_results = []

# Schleife über alle Seeds
for seed in seeds:
    eval_model = YOLO(f"{SCENARIO}/{EXPERIMENT_CONFIG}_{seed}/weights/last.pt")
    metrics = eval_model.val(
        data=f"{DATASET_WORKING_DIR}/scenario{SCENARIO+1}/{EXPERIMENT_CONFIG}.yaml",
        split="test",
        conf=0.001,
        plots=True,
        exist_ok=True,
        project=f"{SCENARIO}",
        name=f"{EXPERIMENT_CONFIG}_{seed}_test"
    )

    names = eval_model.names

    df = pd.DataFrame({
        "class": [names[i] for i in range(len(metrics.box.maps))],
        "AP50": metrics.box.ap50,
        "AP": metrics.box.ap,
        "mAP50": [metrics.box.map50]*len(metrics.box.maps),
        "mAP": [metrics.box.map]*len(metrics.box.maps)
    })

    all_results.append(df)

# Alle Ergebnisse kombinieren
combined = pd.concat(all_results)

# Mittelwert und Std pro Klasse
summary = combined.groupby("class").agg(
    AP50_mean=("AP50", "mean"),
    AP50_std=("AP50", "std"),
    AP_mean=("AP", "mean"),
    AP_std=("AP", "std")
).reset_index()

# Formatieren als "mean ± std"
summary["AP50"] = summary.apply(lambda row: f"{row['AP50_mean']:.4f} ± {row['AP50_std']:.4f}", axis=1)
summary["AP"] = summary.apply(lambda row: f"{row['AP_mean']:.4f} ± {row['AP_std']:.4f}", axis=1)

# Nur Klasse, AP50, AP
summary_latex = summary[["class", "AP50", "AP"]]

# Gesamt-mAP berechnen über alle Seeds
map50_mean = combined["mAP50"].mean()
map50_std = combined["mAP50"].std()
map_mean = combined["mAP"].mean()
map_std = combined["mAP"].std()

# mAP-Zeile erstellen
map_row = pd.DataFrame({
    "class": ["mAP"],
    "AP50": [f"{map50_mean:.4f} ± {map50_std:.4f}"],
    "AP": [f"{map_mean:.4f} ± {map_std:.4f}"]
})

# Tabelle erweitern
summary_latex = pd.concat([summary_latex, map_row], ignore_index=True)

# LaTeX-Tabelle schreiben (UTF-8 für ±)
with open(f"{SCENARIO}_{EXPERIMENT_CONFIG}_results_summary.tex", "w", encoding="utf-8") as f:
    f.write(summary_latex.to_latex(index=False, escape=False))



#model = YOLO("/kaggle/working/runs/detect/train36/weights/best.pt")

image_dir = Path(f"{WORKING_DIR}/images/test")

image_files = list(image_dir.glob("*.jpg"))

index = 8
image_path = image_files[index]

results = model(str(image_path))

for result in results:
    boxes = result.boxes
    result.show()


os.chdir(WCT2_DIR)
if not os.path.exists("results"):
    os.mkdir("results")
else:
    shutil.rmtree("results")
    os.mkdir("results")
    
if not os.path.exists("data"):
    os.mkdir("data")
else:
    shutil.rmtree("data")
    os.mkdir("data")


os.chdir("data")
if not os.path.exists("style"):
    os.mkdir("style")
if not os.path.exists("content"):
    os.mkdir("content")

CONTENT_PATH = f"{DATASET_WORKING_DIR}/scenario{SCENARIO}/images/0baseline/train"
STYLE_PATH = f"{DATASET_WORKING_DIR}/da/style/{SCENARIO}"

rng = random.Random(42)
style_files = [p for p in Path(STYLE_PATH).iterdir() if p.is_file()]

for path in os.listdir(CONTENT_PATH):
    if os.path.isdir(path):
        continue

    content_file_name = Path(path).name

    style_file_name = rng.choice(style_files).name

    shutil.copyfile(f"{CONTENT_PATH}/{content_file_name}", f"{WCT2_DIR}/data/content/{content_file_name}")
    shutil.copyfile(f"{STYLE_PATH}/{style_file_name}", f"{WCT2_DIR}/data/style/{content_file_name}")


os.chdir(WCT2_DIR)
shutil.rmtree("results")
!python transfer.py --option_unpool cat5 -s --content ./data/content --style ./data/style --output ./results/ --image_size 640 


os.chdir(WCT2_DIR)
shutil.make_archive("/kaggle/working/wct_images","zip","./results")


os.chdir(WCT2_DIR)
shutil.make_archive("/kaggle/working/test","zip","./data/content")


# DATA PREPARATION
# training and test data must be in one folder divived by domains, subfolders for division must be names trainA & trainB (trainA -> trainB) and testA & testB resspectetvily

os.chdir(CUT_DIR)
os.chdir("datasets")

if not os.path.isdir("images"):
    os.mkdir("images")
else:
    shutil.rmtree("images")
    os.mkdir("images")

os.chdir("images")
os.mkdir("trainB")

DA_DATASET_DIR = f"{DATASET_WORKING_DIR}/da"

os.symlink(f"{DA_DATASET_DIR}/content", f"{CUT_DIR}/datasets/images/trainA", target_is_directory=True)
shutil.copytree(f"{DA_DATASET_DIR}/style/0", f"{CUT_DIR}/datasets/images/trainB", dirs_exist_ok=True)
shutil.copytree(f"{DA_DATASET_DIR}/style/1", f"{CUT_DIR}/datasets/images/trainB", dirs_exist_ok=True)


os.chdir(f"{CUT_DIR}/datasets/images")

os.symlink(f"{DATASET_WORKING_DIR}/scenario{SCENARIO}/images/0baseline", f"{CUT_DIR}/datasets/images/testA", target_is_directory=True)
os.symlink(f"{CUT_DIR}/datasets/images/trainB", f"{CUT_DIR}/datasets/images/testB", target_is_directory=True)



# TRAINING
# to learn how to tranlaste synthetic images to real

os.chdir(CUT_DIR)
!python train.py \
  --dataroot /kaggle/working/repos/CUT/datasets/images \
  --name syn2real_fastcut \
  --CUT_mode FastCUT \
  --n_epochs 200 \
  --n_epochs_decay 200 \
  --batch_size 16 \
  --nce_idt \
  --lambda_identity 1.0 \
  --lambda_NCE 2.0 \
  --nce_layers 0,2,4,6 \
  --preprocess scale_shortside_and_crop --load_size 640 --crop_size 160


# TRAINING
# to translate synthetic images to real

os.chdir(CUT_DIR)
!python test.py --dataroot ./datasets/images \
    --CUT_mode FastCUT \
    --phase test \
    --name syn2real_fastcut \
    --preprocess none
# --num_test 


os.chdir(CUT_DIR)
shutil.make_archive("/kaggle/working/images","zip","./results/syn2real_fastcut/test_latest/images/fake_B/")


!pip install -U "transformers>=4.51" "diffusers>=0.35"


from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe.to("cuda")
prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
#image and mask_image should be PIL images.
#The mask structure is white for inpainting and black for keeping as is
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
image.save("./yellow_cat_on_park_bench.png")


from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch

pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")

img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
mask_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png"

image = load_image(img_url).resize((1024, 1024))
mask_image = load_image(mask_url).resize((1024, 1024))

prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
generator = torch.Generator(device="cuda").manual_seed(0)

image = pipe(
  prompt=prompt,
  image=image,
  mask_image=mask_image,
  guidance_scale=8.0,
  num_inference_steps=20,  # steps between 15 and 30 work well for us
  strength=0.99,  # make sure to use `strength` below 1.0
  generator=generator,
).images[0]
image.save("./yellow_cat_on_park_bench.png")


from diffusers import StableDiffusionInpaintPipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-inpainting",
    torch_dtype=torch.float16, variant="fp16"
).to('cuda')
prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
#image and mask_image should be PIL images.
#The mask structure is white for inpainting and black for keeping as is
image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
image.save("./yellow_cat_on_park_bench.png")


for name in os.listdir(WORKING_DIR):
    path = os.path.join(WORKING_DIR, name)
    if os.path.isfile(path) or os.path.islink(path):
        os.unlink(path)
    else:
        shutil.rmtree(path)


import torch, gc

gc.collect()            
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
print(torch.cuda.mem_get_info())

