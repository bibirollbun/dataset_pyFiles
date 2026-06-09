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


!pip install -U albumentations


!pip install efficientnet-pytorch



import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import cast

import pytorch_lightning as pl
from torchmetrics.classification import F1Score
import os
from sklearn.utils.class_weight import compute_class_weight
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from transformers import get_cosine_schedule_with_warmup
from torchvision.models import efficientnet_v2_l, resnet50, efficientnet_b7

import cv2

from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose,
    Resize,
    OneOf,
    RandomBrightnessContrast,
    MotionBlur,
    MedianBlur,
    GaussianBlur,
    VerticalFlip,
    HorizontalFlip,
    ShiftScaleRotate,
    Normalize,
    Emboss,
    Sharpen,
    Blur,
    PiecewiseAffine
)

from efficientnet_pytorch import EfficientNet


class PlantDataset(Dataset):

    def __init__(self, df, transform, train = True, src_path='/kaggle/input/plant-pathology-2020-fgvc7/'):
        self.df = df
        self.transform = transform
        self.src_path = src_path
        self.train = train
        
        labels = self.df.iloc[:, 1:5].to_numpy()
        self.labels = np.argmax(labels, axis = -1) if train else None
        self.images = self.df.iloc[:, 0].to_numpy()

        if train:
            class_weights = compute_class_weight(class_weight = 'balanced', classes=np.unique(self.labels), y=self.labels)
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        else:
            self.class_weights = None

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image_number = self.images[idx]
        full_path = os.path.join(self.src_path, 'images', image_number + '.jpg')
        image = cv2.imread(full_path) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.labels[idx] if self.labels is not None else None
        if self.train:
            return self.transform(image=image)['image'], label
        else:
            return self.transform(image=image)['image']


def worker_init_fn(worker_id):
    RANDOM_SEED = 50
    os.sched_setaffinity(0, range(cast(int, os.cpu_count())))
    torch.manual_seed(RANDOM_SEED + worker_id)


class PlantDatamodule(pl.LightningDataModule):

    def __init__(self, src_path='/kaggle/input/plant-pathology-2020-fgvc7/', batch_size: int = 4, num_workers = 4):
        super().__init__()
        self.src_path = src_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        WIDTH = 650
        HEIGHT = 450
        
        self.train_transform = Compose([
        Resize(HEIGHT, WIDTH),
        RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1),
        OneOf([MotionBlur(blur_limit=3), MedianBlur(blur_limit=3), GaussianBlur(blur_limit=3)], p=0.5),
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=20,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=1,
        ),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ToTensorV2()
        ]
        )
        self.test_transform = Compose(
                [
                    Resize(HEIGHT, WIDTH),
                    Normalize(),
                    ToTensorV2()
                ]
        )
                
    def setup(self, stage: str = None):
        train_label_df = pd.read_csv(os.path.join(self.src_path, 'train.csv'))
        self.test_label_df = pd.read_csv(os.path.join(self.src_path, 'test.csv'))

        self.train_label_df, self.val_label_df = train_test_split(train_label_df, 
                            test_size=0.002,
                            stratify=train_label_df[['healthy', 'multiple_diseases', 'rust', 'scab']],
                            random_state=50)

        self.traindata = PlantDataset(self.train_label_df, self.train_transform)
        self.valdata   = PlantDataset(self.val_label_df, self.test_transform)
        self.testdata  = PlantDataset(self.test_label_df, self.train_transform, train = False)

        self.class_weights = self.traindata.class_weights

    def train_dataloader(self):
        return DataLoader(self.traindata,
                         batch_size = self.batch_size,
                         num_workers = self.num_workers,
                         worker_init_fn = worker_init_fn,
                         shuffle = True)
        
    def val_dataloader(self):
        return DataLoader(self.valdata,
                         batch_size = self.batch_size,
                         num_workers = self.num_workers,
                         worker_init_fn = worker_init_fn,
                         shuffle = False)
    
    def test_dataloader(self):
        return DataLoader(self.testdata,
                         batch_size = self.batch_size,
                         num_workers = self.num_workers,
                         worker_init_fn = worker_init_fn,
                         shuffle = False)


class PlantEfficientModule(pl.LightningModule):
    def __init__(self, class_weights: torch.Tensor, submission_size, train_len, dropout_rate: float = 0.1, batch_size: int = 4,
                epochs = 39, tta_model_num = 7):
        super().__init__()
        
        self.model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=4) 

        self.criterion = nn.CrossEntropyLoss(weight = class_weights)
        self.val_f1 = F1Score(task='multiclass', num_classes = 4)
        self.submission_np = np.zeros((submission_size, 4))
        self.batch_size = batch_size
        self.train_len = train_len
        self.epochs = epochs
        self.tta_model_num = tta_model_num

        self.tta_runs = 0

    def forward(self, x):
        logits = self.model(x)
        return logits


    def training_step(self, batch, batch_idx):
        images, labels = batch
        
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        self.log('train/loss', loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        
        outputs = self(images)
        preds = outputs.argmax(dim = -1)
            
        loss = self.criterion(outputs, labels)
        self.val_f1.update(preds, labels)
        
        self.log('val/loss', loss)
        
        return loss
    
    def on_validation_epoch_end(self):
        self.log('val/f1', self.val_f1.compute())
        
    def test_step(self, batch, batch_idx):
        images = batch

        outputs = self(images)
        probas = torch.softmax(outputs.cpu(), dim=1).squeeze().numpy()
        self.submission_np[batch_idx * self.batch_size:(batch_idx+1)*self.batch_size] += probas
            
    def on_test_epoch_end(self):
        self.tta_runs += 1
        if self.tta_runs == self.tta_model_num:
            self.submission_np /= self.tta_runs
            
            submission = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')
            submission_test = submission.copy()
            
            submission_test[['healthy', 'multiple_diseases', 'rust', 'scab']] = self.submission_np
            submission_test.to_csv('submission_test.csv', index=False)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.00006, weight_decay=0.0001)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=self.train_len * 3, 
                                            num_training_steps=self.train_len * self.epochs)
        
        return [optimizer], [scheduler]


from pytorch_lightning.loggers import TensorBoardLogger
import datetime

data_module = PlantDatamodule()
data_module.setup("fit")

efficient_module = PlantEfficientModule(data_module.class_weights, len(data_module.testdata), len(data_module.traindata))

datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
art_dir_name = f"{datetime_str}"

logger = TensorBoardLogger(
    save_dir = os.getcwd(),
    version=art_dir_name,
    name="plant_logs")

trainer = pl.Trainer(max_epochs=39, logger=logger)
trainer.fit(efficient_module, data_module)

#trainer.test(efficient_module, data_module)


for _ in range(7):
    trainer.test(efficient_module, data_module)


def apply_label_smoothing(df, target, alpha, threshold):
    # 타깃값 복사
    df_target = df[target].copy()
    k = len(target) # 타깃값 개수
    
    for idx, row in df_target.iterrows():
        if (row > threshold).any():     
            row = (1 - alpha)*row + alpha/k
            df_target.iloc[idx] = row     
    return df_target 

alpha = 0.001 
threshold = 0.999 

submission_test_ls =pd.read_csv('submission_test.csv')

target = ['healthy', 'multiple_diseases', 'rust', 'scab']
submission_test_ls[target] = apply_label_smoothing(submission_test_ls, target, 
                                                   alpha, threshold)

submission_test_ls.to_csv('submission_test_ls.csv', index=False)


submission_test_ls


submission_test = pd.read_csv('submission_test.csv')
submission_test.to_csv('submission_test_ls_new.csv', index=False)


submission_test = pd.read_csv('submission_test_ls_new.csv')
submission_test.to_csv('plz.csv', index=False)

