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


import pandas as pd
import os

# --- LOAD CSV FILES ---

# Messidor
messidor_csv = pd.read_csv("/kaggle/input/messidor2preprocess/messidor_data.csv")
print("Messidor CSV columns:", messidor_csv.columns.tolist())
print("Messidor shape:", messidor_csv.shape)

# APTOS
aptos_csv = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")
print("APTOS CSV columns:", aptos_csv.columns.tolist())
print("APTOS shape:", aptos_csv.shape)


# --- CREATE IMAGE PATHS ---

# Messidor image folder
messidor_images = "/kaggle/input/messidor2preprocess/messidor-2"

messidor_csv["image_path"] = messidor_csv["id_code"].apply(
    lambda x: f"{messidor_images}/{x}.png"
)

# APTOS image folder
aptos_images = "/kaggle/input/aptos2019-blindness-detection/train_images"

aptos_csv["image_path"] = aptos_csv["id_code"].apply(
    lambda x: f"{aptos_images}/{x}.png"
)


# --- MATCH LABEL COLUMN NAMES ---

# Both have 'diagnosis' already
messidor_csv = messidor_csv.rename(columns={"diagnosis": "label"})
aptos_csv = aptos_csv.rename(columns={"diagnosis": "label"})


# --- KEEP ONLY FINAL COLUMNS ---

messidor_final = messidor_csv[["image_path", "label"]]
aptos_final   = aptos_csv[["image_path", "label"]]

print("Messidor final shape:", messidor_final.shape)
print("APTOS final shape:", aptos_final.shape)


# --- COMBINE DATASETS ---

combined_csv = pd.concat([messidor_final, aptos_final], ignore_index=True)
print("Combined shape:", combined_csv.shape)


# --- SAVE COMBINED CSV ---

combined_csv.to_csv("/kaggle/working/retinal_combined.csv", index=False)
print(" Saved: /kaggle/working/retinal_combined.csv")


combined_csv.to_csv("/kaggle/working/retinal_combined.csv", index=False)

