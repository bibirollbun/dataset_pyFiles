!pip install zarr cryoet_data_portal -q


from typing import Tuple
import os
import glob
import shutil
import scipy
import pandas as pd
import zarr
from cryoet_data_portal import Client, Dataset
import numpy as np
from tqdm import tqdm
import cv2 

client = Client()

base_dir = '/dataset'
tmp_dir = '/tmp'

os.makedirs(base_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)
# Datasets by Author
ds_1 = Dataset.find(client, [Dataset.authors.name == "Yi-Wei Chang"])
ds_2 = Dataset.find(client, [Dataset.authors.name == "Ariane Briegel"])
ds_3 = Dataset.find(client, [Dataset.authors.name == "Morgan Beeby"])

ds_all = [*ds_1,*ds_2,*ds_3]

print("="*25)
print("N_DATASETS:", len(ds_all))
print("="*25)

example = pd.read_csv('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv')
brendanartley_labels = pd.read_csv('/kaggle/input/cryoet-flagellar-motors-dataset/labels.csv')


row_id = 0
new_labels = example.iloc[0:0].copy()

D,H,W = 128,512,512

os.makedirs(tmp_dir, exist_ok=True)
# ========= Process Single Dataset ==========
for i, ds in enumerate(ds_all):
    # Process runs
    print(f"Processing dataset {i}/{len(ds_all)}: {ds.title}")
    for run in tqdm(ds.runs, desc=f"Current: {ds.title}"):
        try:
            labels= brendanartley_labels[brendanartley_labels.tomo_id == run.name]
            if not len(labels):
                continue

            tomo= run.tomograms[0]

            shape = tomo.size_z, tomo.size_y, tomo.size_x

            zarr_path = f"{base_dir}/tmp/{run.name}.zarr"
            if not os.path.exists(zarr_path):
                zarr_path = f'{tmp_dir}/{run.name}.zarr'
                tomo.download_omezarr(dest_path=tmp_dir)



            out_dir = f"{base_dir}/train/{run.name}"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            if len(os.listdir(out_dir))<tomo.size_z:
                arr = zarr.open(zarr_path, mode='r')
                arr = arr[0]

                shape = arr.shape

                for i, img in enumerate(arr):
                    slicename = f"{out_dir}/slice_{str(i).zfill(4)}.jpg"
                    if not os.path.exists(slicename):
                        cv2.imwrite(slicename, (img*255).astype(np.uint8))



            

            for i,row in labels.iterrows():
                new_labels.loc[len(new_labels)] = {
                    "row_id": row_id,
                    "tomo_id": run.name,
                    "Motor axis 0": row.z * (shape[0]/D),
                    "Motor axis 1": row.y * (shape[1]/H),
                    "Motor axis 2": row.x * (shape[2]/W),
                    "Array shape (axis 0)": shape[0],
                    "Array shape (axis 1)": shape[1],
                    "Array shape (axis 2)": shape[2],
                    "Voxel spacing": tomo.voxel_spacing,
                    "Number of motors":len(labels)
                }
                row_id +=1

            
        except Exception as e:
            print(e)
        shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
    #     break # <-------------- COMMENT OUT HERE FOR FULL COLLECTION
    # break  # <-------------- COMMENT OUT HERE FOR FULL COLLECTION
        
new_labels.to_csv(f"{base_dir}/labels.csv")


!ls /dataset



import os

import json
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
key = user_secrets.get_secret("kagglekey")
user = user_secrets.get_secret("kaggleuser")

os.environ['KAGGLE_USERNAME']= user
os.environ['KAGGLE_KEY']= key

import kaggle

# Step 1: Set up dataset directory
DATASET_DIR = "/dataset"

# Step 3: Create metadata file (required by Kaggle API)
metadata_path = os.path.join(DATASET_DIR, "dataset-metadata.json")
metadata_content = {
    "title": "BYU extra data",
    "id": f"{user}/byu-extra-data",
    "licenses": [{"name": "CC0-1.0"}]
}

# Save metadata as JSON
with open(metadata_path, "w") as f:
    json.dump(metadata_content, f, indent=4)

# Step 4: Upload dataset to Kaggle
#kaggle.api.dataset_create_new(DATASET_DIR, "Initial dataset upload")
kaggle.api.dataset_create_version(DATASET_DIR, "Updated dataset with new files", delete_old_versions=True, dir_mode="zip")

print("Dataset uploaded successfully!")

