from IPython.display import clear_output
!pip install git+https://github.com/3lc-ai/3lc-ultralytics@develop
clear_output()


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
tlc_key = user_secrets.get_secret("API Key")



!3lc login {tlc_key}
clear_output()


import tlc
from tlc_ultralytics import Settings, YOLO
from pathlib import Path
import csv
import os
import numpy as np
import pandas as pd
import random
import torch
# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


import yaml
from pathlib import Path

# 1. Mount points
falcon_mount = Path('/kaggle/input/falcon-multiclass-cheerios-soup/falcon-multiclass-cheerios-soup')
v2_mount     = Path('/kaggle/input/falcon-multiclass-cheerios-soupv2/falcon-multiclass-cheerios-soupV2')
chal_mount   = Path('/kaggle/input/multi-class-object-detection-challenge/Starter_Dataset')

# 2. Auto-discover original Falcon root
train_img_dirs = list(falcon_mount.glob('**/train/images'))
if not train_img_dirs:
    raise FileNotFoundError(f"No nested train/images under {falcon_mount}")
falcon_root = train_img_dirs[0].parents[1]  # go up two levels

# 3. Auto-discover V2 root (assumes scenarios at v2_mount/ScenarioX/train/images)
v2_img_dirs = list(v2_mount.glob('**/Scenario1/train/images'))
if not v2_img_dirs:
    raise FileNotFoundError(f"No Scenario1/train/images under {v2_mount}")
v2_root = v2_img_dirs[0].parents[2]

# 4. Build train list
train_dirs = [
    str(falcon_root/'train'/'images'),
    str(falcon_root/'val'  /'images')  # if you also want to oversample the original val
]
for i in range(1, 7):
    train_dirs += [
        str(v2_root/f"Scenario{i}" / 'train' / 'images'),
        str(v2_root/f"Scenario{i}" / 'val'   / 'images')
    ]

# 5. Use **only** the competition val folder for validation
val_dirs = [
    str(chal_mount/'val'/'images')
]

# 6. Test set
test_dir = str(chal_mount/'testImages'/'images')

# 7. Assemble and write YAML
data = {
    'train': train_dirs,
    'val':   val_dirs,
    'test':  test_dir,
    'nc':    2,
    'names': ['cheerios', 'soup']
}

with open('data.yaml', 'w') as f:
    yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

print("✅ data.yaml:")
print(open('data.yaml').read())


PROJECT_NAME = "Duality-3LC-Kaggle"  # Place all 3LC Tables and Runs in the same project

# This for loop allows you to create multiple 3LC Tables (e.g., train and val sets) in one go
for split in ["train", "val"]:
    table = tlc.Table.from_yolo(
        dataset_yaml_file="data.yaml",  # the yolo_params.yaml file in the data folder you generate from Falcon
        split=split,
        table_name="initial",
        dataset_name=split,
        project_name=PROJECT_NAME,
    )

    print(f"Created table with URL: {table.url}")



PROJECT_NAME = "Duality-3LC-Kaggle"  # Place all 3LC Tables and Runs in the same project

RUN_NAME = "run-1"  # Define the run name to organize all your runs in a nice way

# Set 3LC specific settings
settings = Settings(
    project_name=PROJECT_NAME,
    run_name=RUN_NAME,
    run_description="description of the run",
)

# Update the URLs for the train and val tables when you make data revisions in 3LC Dashboard
train_table = tlc.Table.from_url("/root/.local/share/3LC/projects/Duality-3LC-Kaggle/datasets/train/tables/initial")  # Hint: Copy Table URLs from Dashboard
val_table = tlc.Table.from_url("/root/.local/share/3LC/projects/Duality-3LC-Kaggle/datasets/val/tables/initial")

model = YOLO("yolo11x.pt")

# You may add any YOLO arguments here
model.train(
    tables={"train": train_table, "val": val_table},
    settings=settings,
    epochs=75,                
    batch=16,                   
    imgsz=512,
    patience=50,               
    optimizer='SGD',
    momentum=0.937,          
    lr0=0.001,                
    weight_decay=0.0005,       
    cos_lr=True,               
    save_period=5,             
    workers=4,
    # Augmentations
    close_mosaic=15,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.5,
    fliplr=0.5,
    translate=0.1,
    scale=0.5,
    shear=0.01,
    agnostic_nms=True,
    project=PROJECT_NAME,
    name=RUN_NAME,
)

