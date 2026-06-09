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
from pathlib import Path
import kagglehub

# === Configuration ===

#dataset_path variable is created to get Dataset ID i.e. its CSV file name. 
dataset_path= Path('/kaggle/input/brazil-landsat-noaa-images-with-id-for-analysis/noaa_cdr_6830d3c9e730715a.csv')

#OpenAI model used for this Notebook is OpenAI o4-mini. 
model_family = "GPT-4"
model_version = "OpenAI o4-mini"
developer_name = "Vijaya Laxmi Kumbaji"
dataset_id = dataset_path.name
task_description = (
    "Describe surface features in plain English from NOAA NDVI metadata including "
    "satellite, sensor, projection, and spatial coverage."
)

# === Load Dataset ===

# Try reading with Latin-1 encoding
try:
    df = pd.read_csv(dataset_path, encoding='latin1')
except UnicodeDecodeError:
    # Fallback to cp1252 if latin1 fails
    df = pd.read_csv(dataset_path, encoding='cp1252')

print("Dataset loaded successfully!")


# === Print Output ===
print(f"OpenAI model information:")
print(f"Model Family: {model_family}")
print(f"Model Version: {model_version}")
print(f"Dataset ID: {dataset_id}")
print(f"Developer / Participant name: {developer_name}")
print(f"Task: {task_description}")

# Display first few rows of the dataset
print("\nSample Data: This is a small sample data of 5 rows and 32 columns showing beginning date and end date of data collected on Brazil.") 
print("\nThe following information is gathered in the Dataset:") 
print("\nsatellite: NOAA-14")
print("\nsensor: AVHRR")
print("\nBegin Date (of data collection): 01/01/2000") 
print("\nEnd Date )of data collection) : 01/31/2018.") 
print("\nProjection: ''GEOGRAPHIC''' information.")
print(df.head())




