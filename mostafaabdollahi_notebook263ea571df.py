# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing,CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os

base_path = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification"
test_path = os.path.join(base_path, "test_images")

import pandas as pd
df = pd.read_csv(base_path+"/train_series_descriptions.csv")
stirs=df.query("series_description=='Sagittal T2/STIR'")



stirs


import os
import shutil
import tqdm
import time
import pandas as pd


def downloader(x, y, z="8.dcm"):
    src_path = f"/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/{x}/{y}/{z}"
    dst_folder = f"/kaggle/working/GGHADID/{x}"
    dst_path = os.path.join(dst_folder, z)

    if not os.path.exists(src_path):
        print(f"❌ File not found: {src_path}")
        return

    os.makedirs(dst_folder, exist_ok=True)
    shutil.copy(src_path, dst_path)
    print(f"✅ Copied {z} to {dst_folder}")

for i in tqdm.tqdm(range(len(stirs))):
    study_id = str(stirs.iloc[i, 0])
    series_id = str(stirs.iloc[i, 1])
    downloader(study_id, series_id)


import shutil

shutil.make_archive('/kaggle/working/GGHADID_zip', 'zip', '/kaggle/working/GGHADID')



import os
import shutil
from pathlib import Path

# Source folder
src_dir = Path("/kaggle/working/GGHADID")
output_base = Path("/kaggle/working/GGHADID_parts")

# Create output base folder
output_base.mkdir(parents=True, exist_ok=True)

# Get all study folders
all_study_folders = sorted([f for f in src_dir.iterdir() if f.is_dir()])
total = len(all_study_folders)
chunk_size = total // 10 + (total % 10 > 0)  # Ensure all files included

# Split into 10 parts
for i in range(10):
    part_dir = output_base / f"part_{i+1}"
    part_dir.mkdir(exist_ok=True)
    
    for folder in all_study_folders[i * chunk_size: (i + 1) * chunk_size]:
        shutil.copytree(folder, part_dir / folder.name)

print("✅ Folder split complete.")



for i in range(1, 11):
    folder_to_zip = output_base / f"part_{i}"
    zip_path = f"/kaggle/working/GGHADID_part{i}.zip"
    shutil.make_archive(zip_path.replace(".zip", ""), 'zip', folder_to_zip)



from google.colab import auth  # not available, so do this:



!pip install PyDrive




import os
import json
import PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Save uploaded secret
with open("/kaggle/input/kaggledrive/client_secret_609625822155-4ssv6gjc4c4gg789jac0vpg2ju8adg8c.apps.googleusercontent.com.json", "r") as f:
    secret = json.load(f)

with open("/kaggle/input/kaggledrive/client_secret_609625822155-4ssv6gjc4c4gg789jac0vpg2ju8adg8c.apps.googleusercontent.com.json", "w") as f:
    json.dump(secret, f)

# Authenticate
gauth = GoogleAuth()
gauth.LoadClientConfigFile("client_secrets.json")
gauth.LocalWebserverAuth()  # Triggers browser login, will NOT work directly in Kaggle


