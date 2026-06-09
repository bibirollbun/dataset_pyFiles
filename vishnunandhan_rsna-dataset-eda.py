# Imports and Data Loading
import os
import time
import glob
import json
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snsa
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import matplotlib.patches as patches
from matplotlib import animation, rc

# For CV and metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             roc_curve, auc, accuracy_score, roc_auc_score)
from sklearn.preprocessing import label_binarize

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the base path and read CSV files
train_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/'

train  = pd.read_csv(os.path.join(train_path, 'train.csv'))
label = pd.read_csv(os.path.join(train_path, 'train_label_coordinates.csv'))
train_desc  = pd.read_csv(os.path.join(train_path, 'train_series_descriptions.csv'))
test_desc   = pd.read_csv(os.path.join(train_path, 'test_series_descriptions.csv'))
sub         = pd.read_csv(os.path.join(train_path, 'sample_submission.csv'))

# Quick check of the dataframes
print("Train.csv head:")
print(train.head(5))




def generate_image_paths(df, data_dir):
    image_paths = []
    for study_id, series_id in zip(df['study_id'], df['series_id']):
        study_dir = os.path.join(data_dir, str(study_id))
        series_dir = os.path.join(study_dir, str(series_id))
        if os.path.exists(series_dir):
            images = os.listdir(series_dir)
            image_paths.extend([os.path.join(series_dir, img) for img in images])
    return image_paths

# Generate image paths for train and test data
train_image_paths = generate_image_paths(train_desc, os.path.join(train_path, 'train_images'))
test_image_paths = generate_image_paths(test_desc, os.path.join(train_path, 'test_images'))
print("Example train image path:", train_image_paths[2])
print("Total train series:", len(train_desc))
print("Total train images found:", len(train_image_paths))



# Function to reshape a single row of the train CSV
def reshape_row(row):
    data = {'study_id': [], 'condition': [], 'level': [], 'severity': []}
    for column, value in row.items():
        if column not in ['study_id', 'series_id', 'instance_number', 'x', 'y', 'series_description']:
            parts = column.split('_')
            condition = ' '.join([word.capitalize() for word in parts[:-2]])
            level = parts[-2].capitalize() + '/' + parts[-1].capitalize()
            data['study_id'].append(row['study_id'])
            data['condition'].append(condition)
            data['level'].append(level)
            data['severity'].append(value)
    return pd.DataFrame(data)


# Reshape train CSV from wide to long format
new_train_df = pd.concat([reshape_row(row) for _, row in train.iterrows()], ignore_index=True)
print("Reshaped train data:")
print(new_train_df.head(5))


# Print column names for clarity
print("\nColumns in new_train_df:", ",".join(new_train_df.columns))
print("Columns in label:", ",".join(label.columns))
print("Columns in test_desc:", ",".join(test_desc.columns))
print("Columns in sub:", ",".join(sub.columns))


# Merge dataframes: new_train_df, label and train_desc
merged_df = pd.merge(new_train_df, label, on=['study_id', 'condition', 'level'], how='inner')
final_merged_df = pd.merge(merged_df, train_desc, on=['series_id', 'study_id'], how='inner')
print("Merged data sample:")
print(final_merged_df.head(5))


# Create new columns: row_id and image_path
final_merged_df['row_id'] = (final_merged_df['study_id'].astype(str) + '_' +
                             final_merged_df['condition'].str.lower().str.replace(' ', '_') + '_' +
                             final_merged_df['level'].str.lower().str.replace('/', '_'))
final_merged_df['image_path'] = (os.path.join(train_path, 'train_images') + '/' +
                                 final_merged_df['study_id'].astype(str) + '/' +
                                 final_merged_df['series_id'].astype(str) + '/' +
                                 final_merged_df['instance_number'].astype(str) + '.dcm')
print("Data with new columns:")
print(final_merged_df.head(5))


# Map severity labels to lower-case format
final_merged_df['severity'] = final_merged_df['severity'].map({'Normal/Mild': 'normal_mild',
                                                               'Moderate': 'moderate',
                                                               'Severe': 'severe'})

# Set train_data as final_merged_df copy
train_data = final_merged_df.copy()
print("Train data shape before filtering:", train_data.shape)


# 1) copy & strip out any stray whitespace
df = final_merged_df.copy()
df['series_description'] = df['series_description'].astype(str).str.strip()

# 2) look at the unique views you actually have
print("ALL views in the data →", df['series_description'].unique())

# 3) build the counts pivot
counts = (
    df
    .groupby(['series_description','condition'])
    .size()                     # count rows
    .unstack(fill_value=0)      # make a DataFrame: rows=view, cols=condition
)

# 4) reorder to the three you care about (fill missing with 0)
desired = ['Sagittal T2/STIR','Sagittal T1','Axial T2']
counts = counts.reindex(desired, fill_value=0)

# 5) see it
print(counts)



df = final_merged_df.copy()

# clean up the keys (optional)
df['series_description'] = df['series_description'].str.strip()
df['severity']           = df['severity'].str.lower().str.replace('/','_')  # e.g. "Normal/Mild"→"normal_mild"

# now group by view → condition → severity, and count
cond_sev_counts = (
    df
    .groupby(['series_description','condition','severity'])
    .size()  # count rows
    .unstack(fill_value=0)  # pivot severity → columns
)

print(cond_sev_counts)



import seaborn as sns
import matplotlib.pyplot as plt

# 1) prep a “long” df
df = final_merged_df.copy()
df['series_description'] = df['series_description'].str.strip()
df['severity'] = (df['severity']
                   .str.lower()
                   .str.replace('/','_'))    # normal_mild, moderate, severe

# 2) draw a separate count‐bar chart for each series_description
g = sns.catplot(
    data=df,
    x='condition',
    hue='severity',
    col='series_description',
    kind='count',
    palette='muted',
    height=4,
    aspect=1.2,
    sharey=False            # let each panel scale independently
)

# 3) polish
g.set_axis_labels("", "Count")
g.set_titles("{col_name}")  # just show the view name
for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.show()


# ——————————————————————————————
# 1) Total label‐rows per condition
label_counts = (
    final_merged_df
    .groupby('condition')
    .size()
    .sort_values(ascending=False)
    .rename("num_labels")
)

# 2) Total unique images per condition
image_counts = (
    final_merged_df
    .groupby('condition')['image_path']
    .nunique()
    .sort_values(ascending=False)
    .rename("num_images")
)

# Combine into one DataFrame
cond_summary = pd.concat([label_counts, image_counts], axis=1)
print(cond_summary)



import matplotlib.pyplot as plt

# plot num_images per condition
cond_summary['num_images'].plot.barh(figsize=(8, 5))
plt.title("Number of Unique Images per Condition")
plt.xlabel("Unique DICOM Files")
plt.ylabel("Condition")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()




# train_desc      : the DataFrame of all series in train
# final_merged_df : the DataFrame where each row is one (slice,condition,level) annotation

from pathlib import Path

# 1) total files on disk under train_images/
train_images_dir = Path('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images')
all_paths = list(train_images_dir.rglob('*.dcm'))
print("Total DICOM files in train:", len(all_paths))

# 2) how many of those were actually annotated at least once?
# final_merged_df['image_path'] should hold the full path to each labelled slice
n_labelled = final_merged_df['image_path'].nunique()
print("Unique images with ≥1 label:", n_labelled)





