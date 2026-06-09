# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
import fastai
from fastai.vision.all import *
from ipywidgets import widgets

from pathlib import Path

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


torch.__version__, fastai.__version__


torch.cuda.is_available()


path = Path('/kaggle/input/pnevmoniya/train')
path


d_block = DataBlock(
    blocks = (ImageBlock, CategoryBlock),
    get_items = get_image_files,
    splitter = RandomSplitter(valid_pct=0.2, seed=42),
    get_y = parent_label,
    item_tfms = Resize(224)
)


dls = d_block.dataloaders(path)


dls.show_batch()


learn = cnn_learner(dls, resnet34, metrics=accuracy)


learn.fine_tune(3)


accur = ClassificationInterpretation.from_learner(learn)
accur.plot_confusion_matrix()


accur.print_classification_report()


test = ('/kaggle/input/pnevmoniya/test')


submission = pd.read_csv('/kaggle/input/pnevmoniya/sample_solution.csv')


submission.head()


length = len(submission)
for i in range(length):
    label = submission.loc[i, 'id']
    img = PILImage.create(Path(f"/kaggle/input/pnevmoniya/test/{label}"))
    pred, pred_id, probs = learn.predict(img)
    submission.loc[i, 'labels'] = np.array(probs[1])


submission.head()


submission.to_csv('solution.csv', index=False)

