import pydicom
import glob, os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import re
import shutil
import os


rd='/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification'


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


dfc = pd.read_csv(f"{rd}/train_label_coordinates.csv")


df = pd.read_csv(f'{rd}/train_series_descriptions.csv')
df.head()


df['series_description'].value_counts()


df[df['study_id']==4003253]


def imread_and_imwirte(src_path, dst_path):
    dicom_data = pydicom.dcmread(src_path)
    image = dicom_data.pixel_array
    image = (image - image.min()) / (image.max() - image.min() +1e-6) * 255
    img = cv2.resize(image, (512, 512),interpolation=cv2.INTER_CUBIC)
    assert img.shape==(512,512)
    cv2.imwrite(dst_path, img)


st_ids = df['study_id'].unique()
st_ids[:3], len(st_ids)


desc = list(df['series_description'].unique())
desc


st_ids = st_ids[:500]
for idx, si in enumerate(tqdm(st_ids, total=len(st_ids))):
    pdf = df[df['study_id']==si]
    for ds in desc:
        ds_ = ds.replace('/', '_')
        pdf_ = pdf[pdf['series_description']==ds]
        os.makedirs(f'mri_png/{si}/{ds_}', exist_ok=True)
        allimgs = []
        for i, row in pdf_.iterrows():
            pimgs = glob.glob(f'{rd}/train_images/{row["study_id"]}/{row["series_id"]}/*.dcm')
            pimgs = sorted(pimgs, key=natural_keys)
            allimgs.extend(pimgs)

        if len(allimgs)==0:
            print(si, ds, 'has no images')
            continue

        if ds == 'Axial T2':
            for j, impath in enumerate(allimgs):
                dst = f'mri_png/{si}/{ds}/{j:03d}.png'
                imread_and_imwirte(impath, dst)

        elif ds == 'Sagittal T2/STIR':

            step = len(allimgs) / 10.0
            st = len(allimgs)/2.0 - 4.0*step
            end = len(allimgs)+0.0001
            for j, i in enumerate(np.arange(st, end, step)):
                dst = f'mri_png/{si}/{ds_}/{j:03d}.png'
                ind2 = max(0, int((i-0.5001).round()))
                imread_and_imwirte(allimgs[ind2], dst)

            assert len(glob.glob(f'mri_png/{si}/{ds_}/*.png'))==10

        elif ds == 'Sagittal T1':
            step = len(allimgs) / 10.0
            st = len(allimgs)/2.0 - 4.0*step
            end = len(allimgs)+0.0001
            for j, i in enumerate(np.arange(st, end, step)):
                dst = f'mri_png/{si}/{ds}/{j:03d}.png'
                ind2 = max(0, int((i-0.5001).round()))
                imread_and_imwirte(allimgs[ind2], dst)

            assert len(glob.glob(f'mri_png/{si}/{ds}/*.png'))==10


rd = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification'
des = '/kaggle/working/'

shutil.copy(f"{rd}/train_label_coordinates.csv", des)
shutil.copy(f"{rd}/train_series_descriptions.csv", des)
shutil.copy(f"{rd}/train.csv", des)
print("File moved to output directory.")


df = pd.read_csv(f'{des}/train.csv')
df = df.iloc[:500]
df.to_csv(f'{des}/train_500.csv', index=False)


import pandas as pd
import os

# Load the primary dataset
primary_df = pd.read_csv(f'{des}/train_500.csv')

# Get the list of valid study_id values
valid_study_ids = primary_df['study_id'].unique()

# Define the target directory
target_directory = '/kaggle/working/'

# Process each CSV file in the directory
for filename in os.listdir(target_directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(target_directory, filename)
        
        df = pd.read_csv(file_path)
        
        if 'study_id' in df.columns:
            # Keep only rows with study_id values present in train_500.csv
            df_filtered = df[df['study_id'].isin(valid_study_ids)]
            
            df_filtered.to_csv(file_path, index=False)
            print(f"Updated {filename}: Kept only rows with study_id present in train_500.csv.")
        else:
            print(f"'study_id' column not found in {filename}. Skipping this file.")



target_directory = '/kaggle/working/mri_png'
directory_count = 0

with os.scandir(target_directory) as entries:
    for entry in entries:
        if entry.is_dir():
            directory_count += 1

print(f'Total number of directories in "{target_directory}": {directory_count}')


primary_df['study_id'].nunique()




