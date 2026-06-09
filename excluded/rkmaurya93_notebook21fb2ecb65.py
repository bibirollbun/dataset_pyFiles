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

# Define dataset path
data_path = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/ot/ot/"
train_csv_path = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/train.csv"

# Load training data
train_df = pd.read_csv(train_csv_path)

# Create full paths for each npy file
train_df["file_path"] = train_df["id"].apply(lambda x: os.path.join(data_path, x))

# Check if paths are correctly assigned
print(train_df.head())  # Ensure file paths are linked properly


import pandas as pd

# Load CSV file
train_df = pd.read_csv("/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/train.csv")

# Print the column names



print(train_df.head())


import numpy as np

for _, row in train_df.iterrows():
    file_path = row["file_path"]
    
    try:
        img_data = np.load(file_path)
        print(f"File: {file_path}, Shape: {img_data.shape}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")


if img_data.shape != (128, 128, 125):
    print(f"Reshaping {file_path} from {img_data.shape} to (128, 128, 125)")
    img_data = np.resize(img_data, (128, 128, 125))  # Use cautiously!


if img_data.shape == (128, 128, 125):
    X.append(img_data)
    y.append(row["label"])
else:
    print(f"Skipping {file_path} due to shape mismatch: {img_data.shape}")


X = np.array(X)
y = np.array(y)
print(f"Final dataset shape: {X.shape}, Labels shape: {y.shape}")


import numpy as np

X, y = [], []

# Loop through dataset to load images
for _, row in train_df.iterrows():
    file_path = row["file_path"]
    label = row["label"]

    # Load the image file
    img_data = np.load(file_path)

    X.append(img_data)  # Store image data
    y.append(label)  # Store corresponding label

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} samples with shape {X[0].shape}")




