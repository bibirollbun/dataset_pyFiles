import numpy as np 
import pandas as pd 

from fastai.vision.all import *
from ipywidgets import widgets

import requests 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# kerakli kutubxonalar


path = Path("/kaggle/input/pnevmoniya/train")

# train uchun dataset


data_block = DataBlock(
    blocks = (ImageBlock, CategoryBlock),
    get_items = get_image_files,
    splitter = RandomSplitter(valid_pct = 0.2, seed = 42),
    get_y = parent_label,
    item_tfms = Resize(224)  
)

# datablock yaratildi

# ImageBlock - Rasmli model uchun, CategoryBlock - kasal yoki kasal emasligini aniqlash uchun categorical hisoblanadi
# get_image_files - rasmli files qabul qilib oladi
# valid_pct = 0.2 - validation uchun 20%. train uchun - 80% data olinadi
# seed = 42 - random chiqarganda saqlanishini ta'minlaydi
# parent_label - label uchun
# Resize(224) - image saqlanishi va transformation qilinishi


data_loaders = data_block.dataloaders(path)

# dataloaders


learn = cnn_learner(data_loaders, resnet34, metrics = accuracy)

# Model


learn.fine_tune(3)

# epoch = 3


inter_prepetaion = ClassificationInterpretation.from_learner(learn)
inter_prepetaion.plot_confusion_matrix()

# Confusion Matrix orqali tekshirish


inter_prepetaion.print_classification_report()

# Classification Report orqali natija aniqlandi


submit = pd.read_csv('/kaggle/input/pnevmoniya/sample_solution.csv')
submit.head()

# Competition uchun sample solution ko'rish


for i in range(len(submit)):
    label = submit.loc[i, 'id']
    img = PILImage.create(Path(f"/kaggle/input/pnevmoniya/test/{label}"))
    pred, pred_id, probs = learn.predict(img)
    submit.loc[i, 'labels'] = np.array(probs[1])


submit.head()


submit.to_csv("submission_DL_pnevmaniya_aniqlash.csv",index=False)

