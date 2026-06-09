# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from fastai.vision.all import *
from fastai.medical.imaging import *

print("fastai is ready!")


def parse_data(df):
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]

    parsed = {}

    for _, row in df.iterrows():
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom':  f'/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/{pid}.dcm',
                'label': row['Target'],
                'boxes': [],
                "class_info": row['class']
            }

        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed
    


df = pd.read_csv('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')
df_info = pd.read_csv('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv')
df_merged = df_info.merge(df, on='patientId')

parsed = parse_data( df_merged.groupby('class', group_keys=False)
    .apply(lambda x: x.sample(n=min(500, len(x)), random_state=42)))

print(f"Parsed {len(parsed)} patients")

list(parsed.items())[:2]  # Show first 2 entries


def get_x(o):
    return o['dicom']

def get_y(o):
    return str(o['label'])

items = list(parsed.values())

dblock = DataBlock(
    blocks=(ImageBlock(cls=PILDicom), CategoryBlock),
    get_x=get_x,
    get_y=get_y,
    splitter=RandomSplitter(seed=42),
    item_tfms=Resize(224),
    batch_tfms=aug_transforms()
)

dls = dblock.dataloaders(items, bs=32)

dls.show_batch(max_n=9, figsize=(8,8))


learn = vision_learner(dls, resnet34, metrics=accuracy)

# learn.lr_find()

learn.fine_tune(2, base_lr=1e-3)


learn.export()


from fastai.vision.all import ClassificationInterpretation

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
interp.plot_top_losses(9, figsize=(8,8))

