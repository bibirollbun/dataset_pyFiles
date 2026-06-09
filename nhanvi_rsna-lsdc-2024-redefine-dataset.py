import pydicom
import glob, os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import re
from multiprocessing import Pool


path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification'


dfd = pd.read_csv(f'{path}/train_series_descriptions.csv')
dfd.head()


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def imread_and_imwrite(src_dst):
    src_path, dst_path = src_dst
    dicom_data = pydicom.dcmread(src_path)
    image = dicom_data.pixel_array
    image = ((image - image.min()) / (image.max() - image.min() + 1e-6) * 255).astype(np.uint8)
    img = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(dst_path, img)

def process_study(si, desc, path):
    pdf = dfd[dfd['study_id'] == si]
    for ds in desc:
        ds_ = ds.replace('/', '_')
        pdf_ = pdf[pdf['series_description'] == ds]
        os.makedirs(f'cvt_png/{si}/{ds_}', exist_ok=True)
        allimgs = []

        for _, row in pdf_.iterrows():
            pimgs = sorted(
                glob.glob(f'{path}/train_images/{row["study_id"]}/{row["series_id"]}/*.dcm'),
                key=natural_keys,
            )
            allimgs.extend(pimgs)

        if not allimgs:
            print(si, ds, 'has no images')
            continue

        tasks = []
        if ds == 'Axial T2':
            for j, impath in enumerate(allimgs):
                dst = f'cvt_png/{si}/{ds}/{j:03d}.png'
                tasks.append((impath, dst))

        elif ds in {'Sagittal T2/STIR', 'Sagittal T1'}:
            step = len(allimgs) / 10.0
            st = len(allimgs) / 2.0 - 4.0 * step
            end = len(allimgs) + 0.0001
            for j, i in enumerate(np.arange(st, end, step)):
                dst = f'cvt_png/{si}/{ds_}/{j:03d}.png'
                ind2 = max(0, int(round(i - 0.5001)))
                tasks.append((allimgs[ind2], dst))
            assert len(tasks) == 10

        with Pool(processes=8) as pool:  # Adjust the number of processes based on CPU cores
            pool.map(imread_and_imwrite, tasks)


st_ids = dfd['study_id'].unique()
desc = list(dfd['series_description'].unique()) #['Sagittal T2/STIR', 'Sagittal T1', 'Axial T2']
for si in tqdm(st_ids, total=len(st_ids)):
    process_study(si, desc, path)

