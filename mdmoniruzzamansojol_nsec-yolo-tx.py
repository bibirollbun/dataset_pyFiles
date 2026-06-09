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
df = pd.read_csv('/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv')
df.head()


import pydicom
import matplotlib.pyplot as plt
dicom_file = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train/000434271f63a053c4128a0ba6352c7f.dicom"
ds = pydicom.dcmread(dicom_file)
plt.imshow(ds.pixel_array, cmap="gray")
plt.axis("off")
plt.title("Sample Chest X-ray")
plt.show()


import os
data_path = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection"
csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
print("CSV Tables:", csv_files)


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data_path = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection"

print("Files in dataset folder:")
print(os.listdir(data_path))

train_df = pd.read_csv(f"{data_path}/train.csv")

# Unique class IDs and names
class_mapping = train_df[['class_id', 'class_name']].drop_duplicates().sort_values('class_id')
print(class_mapping)
# Load only the real CSVs
train_df = pd.read_csv(f"{data_path}/train.csv")
sample_submission_df = pd.read_csv(f"{data_path}/sample_submission.csv")
# test


# Filter only rows that have bounding boxes (i.e., not "No finding")
bbox_df = train_df[train_df["class_name"] != "No finding"]

# Count bounding boxes per class
bbox_counts = bbox_df["class_name"].value_counts().sort_values()
print(bbox_counts)

# Horizontal bar plot
plt.figure(figsize=(10, 8))
bbox_counts.plot(kind="barh", color="skyblue")
plt.title("Number of Bounding Boxes per Class")
plt.xlabel("Box Count")
plt.ylabel("Class Name")
plt.grid(axis="x")
plt.tight_layout()
plt.show()

# Filter out 'No finding'
bbox_df = train_df[train_df["class_name"] != "No finding"].copy()
print(bbox_df)
# Calculate width, height, and area
bbox_df["width"] = bbox_df["x_max"] - bbox_df["x_min"]
bbox_df["height"] = bbox_df["y_max"] - bbox_df["y_min"]
bbox_df["area"] = bbox_df["width"] * bbox_df["height"]
# Average area per class
avg_area = bbox_df.groupby("class_name")["area"].mean().sort_values()
print(avg_area)
plt.figure(figsize=(10, 6))
avg_area.plot(kind='barh', color='lightcoral')
plt.title("Average Bounding Box Area per Class")
plt.xlabel("Area (pixels)")
plt.ylabel("Class")
plt.grid(axis='x')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='class_name', y='width', data=bbox_df)
plt.xticks(rotation=90)
plt.title('Bounding Box Width Distribution by Class')
plt.grid(True)
plt.show()






