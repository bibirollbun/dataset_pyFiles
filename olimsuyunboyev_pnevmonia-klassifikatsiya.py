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


import torch 
import fastai
from fastai.vision.all import *
from ipywidgets import widgets

torch.__version__, fastai.__version__


torch.cuda.is_available()


from pathlib import Path

path = Path('/kaggle/input/pnevmoniya/train')
path


block = DataBlock(
    blocks    = (ImageBlock, CategoryBlock),
    get_items = get_image_files,
    splitter = RandomSplitter(seed=42, valid_pct=0.2),
    get_y = parent_label,
    item_tfms = Resize(244)
)


dls = block.dataloaders(path)


dls.show_batch()


learn = vision_learner(dls, resnet34, metrics=accuracy)
learn.fine_tune(3)


testing_path = ('/kaggle/input/pnevmoniya/test')


solution_file = pd.read_csv('/kaggle/input/pnevmoniya/sample_solution.csv')


solution_file.head(10)


for i in range(0,624):
    label = solution_file.iloc[i,0]
    img = PILImage.create(Path(f'../input/pnevmoniya/test/{label}'))
    pred, _, prob = learn.predict(img)
    solution_file.iloc[i,1] = pred


solution_file.head()


solution_file.to_csv('Pnevmania_tashxisi.csv', index=False)




