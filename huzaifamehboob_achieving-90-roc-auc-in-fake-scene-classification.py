pip install "scikit-learn>=1.4.0,<1.5.3"



!pip install autogluon


import numpy as np
import pandas as pd
import os 

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import cv2
from scipy.ndimage import convolve
from PIL import Image,ImageFilter
import argparse
import os
import random,numpy
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms


import warnings
warnings.filterwarnings('ignore')


import torch
torch.cuda.empty_cache()


train_info = pd.read_csv('/kaggle/input/cidaut-ai-fake-scene-classification-2024/train.csv')

if not os.path.exists(f"/kaggle/working/train/"):
    os.mkdir(f"/kaggle/working/train/")
    os.mkdir(f"/kaggle/working/train/real/")
    os.mkdir(f"/kaggle/working/train/editada/")

for i in zip(train_info['image'], train_info['label'], range(len(train_info['image']))):
    image = Image.open(f"/kaggle/input/cidaut-ai-fake-scene-classification-2024/Train/{i[0]}")

    if i[1] == "real":
        
        if not os.path.exists(f"/kaggle/working/train/real/"):
            os.mkdir(f"/kaggle/working/train/real/")
        image.filter(ImageFilter.EDGE_ENHANCE_MORE).save(
            f"/kaggle/working/train/real/{i[0]}"
        )
    elif i[1] == "editada":

        if not os.path.exists(f"/kaggle/working/train/editada/"):
            os.mkdir(f"/kaggle/working/train/editada/")
        image.filter(ImageFilter.EDGE_ENHANCE_MORE).save(
            f"/kaggle/working/train/editada/{i[0]}"
        )



train_info=os.listdir("/kaggle/input/cidaut-ai-fake-scene-classification-2024/Test")

if os.path.exists(f"/kaggle/working/test/")==False:
    os.mkdir(f"/kaggle/working/test/")

for i in train_info:
    
    image = Image.open(f"/kaggle/input/cidaut-ai-fake-scene-classification-2024/Test/{i}")
    image.filter(filter=ImageFilter.EDGE_ENHANCE_MORE).save(f"/kaggle/working/test/{i}")


import pandas as pd
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from autogluon.multimodal import MultiModalPredictor

train_transform = transforms.Compose([
    # transforms.Resize(450),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train = ImageFolder(root='/kaggle/working/train/', transform=train_transform)

image_paths = [sample[0] for sample in train.samples]
labels = [sample[1] for sample in train.samples]
train_df = pd.DataFrame({"image": image_paths, "label": labels})

train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)



predictor = MultiModalPredictor(label="label", problem_type="classification", eval_metric="auc")

predictor.fit(
    train_data=train_df,
    tuning_data=val_df,
    hyperparameters={
        "optimization.patience": 3,
    }
)




test =pd.read_csv("/kaggle/input/cidaut-ai-fake-scene-classification-2024/sample_submission.csv")

test_path = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/Test/'

test['image'] = test_path + test['image']


preds = predictor.predict(test, as_pandas=False)


sub = pd.read_csv('/kaggle/input/cidaut-ai-fake-scene-classification-2024/sample_submission.csv')

sub['label'] = preds

sub.to_csv('submission1.csv', index=False)




