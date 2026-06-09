# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# https://www.kaggle.com/competitions/bengaliai-cv19/submissions
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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
import albumentations as A
import albumentations.pytorch

from torch.utils.data import Dataset
from sklearn.metrics import recall_score
from torchvision import models


model = torch.load('/kaggle/input/bengali-trial/model.pth', weights_only=False)



class GraphemeDataset(Dataset):
    def __init__(self,df, transform, _type='train'):
        self.df = df
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        image = self.df.iloc[idx][1:].values.reshape(137, 236).astype(np.uint8)
        image = 255 - image
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, 2)
        if self.transform is not None:
            image = self.transform(image=image)['image']
        return image, self.df.iloc[idx][0]

valid_augmentation = A.Compose([
    A.Normalize(normalization="min_max"),
    A.pytorch.transforms.ToTensorV2()
])


n_grapheme=168
n_vowel=11
n_consonant=7
model.eval()
test_data = ['test_image_data_0.parquet','test_image_data_1.parquet','test_image_data_2.parquet','test_image_data_3.parquet']
predictions = []
row_ids = []
batch_size=256
for fname in test_data:
    data = pd.read_parquet(f'/kaggle/input/bengaliai-cv19/{fname}')
    test_image = GraphemeDataset(data, valid_augmentation)
    test_loader = torch.utils.data.DataLoader(test_image,batch_size=256,shuffle=False, pin_memory = True, num_workers = 4)
    with torch.no_grad():
        for inputs, names in tqdm(test_loader,total=len(test_loader)):
            for name in names:
                row_ids += [f"{name}_grapheme_root", f"{name}_vowel_diacritic", f"{name}_consonant_diacritic"]
            inputs = inputs.cuda()
            logits = model(inputs)
            logits_split = torch.split(logits, [n_grapheme, n_vowel, n_consonant], dim=1)

            grapheme = logits_split[0].cpu().argmax(dim=1).data.numpy()
            vowel = logits_split[1].cpu().argmax(dim=1).data.numpy()
            cons = logits_split[2].cpu().argmax(dim=1).data.numpy()
            # print(grapheme)
            # print(vowel)
            # print(cons)
            # print("-----")
            # for i in range(len(cons)):
            #     if cons[i] == 7:
            #         cons[i] = 2

            predictions.append(np.stack([grapheme, vowel, cons], axis=1))
            


predictions = np.concatenate(predictions, axis=0).flatten()


submission = pd.DataFrame({'row_id':row_ids,'target':predictions},columns=['row_id','target'])
submission.to_csv('submission.csv',index=False)
submission




