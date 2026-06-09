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


!pip list | grep -E 'tensorflow|syft|torch'


import os

# Check all datasets available in Kaggle input directory
print("Available datasets in Kaggle environment:")
print(os.listdir("/kaggle/input"))


base_dir = "/kaggle/input/rsna-intracranial-hemorrhage-detection"
print("Dataset structure:")
for item in os.listdir(base_dir):
    print(item)


import os

# Define the dataset path
base_dir = "/kaggle/input/rsna-intracranial-hemorrhage-detection"

# List dataset files and folders
print("Dataset structure:")
for item in os.listdir(base_dir):
    item_path = os.path.join(base_dir, item)
    if os.path.isdir(item_path):
        print(f"[DIR] {item}")
    else:
        print(f"[FILE] {item}")


import os

# Define dataset path
base_dir = "/kaggle/input/rsna-intracranial-hemorrhage-detection"

# List files and folders
print("Dataset structure:")
for item in os.listdir(base_dir):
    item_path = os.path.join(base_dir, item)
    if os.path.isdir(item_path):
        print(f"[DIR] {item}")
        print(os.listdir(item_path)[:5])  # Show first 5 files in the directory
    else:
        print(f"[FILE] {item}")


import pandas as pd

# Define path to labels CSV
train_csv_path = "/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train.csv"

# Load the CSV file
df = pd.read_csv(train_csv_path)

# Display basic info and first few rows
print("Dataset Info:")
print(df.info())

print("\nFirst few rows:")
print(df.head())


# Split 'ID' column into 'image_id' and 'hemorrhage_type'
df[['image_id', 'hemorrhage_type']] = df['ID'].str.rsplit('_', n=1, expand=True)

# Drop the original 'ID' column
df.drop(columns=['ID'], inplace=True)

# Display updated dataset
print("Updated Dataset Structure:")
print(df.head())

