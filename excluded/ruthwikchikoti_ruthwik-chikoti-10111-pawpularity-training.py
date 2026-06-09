# import timm
# from pprint import pprint
# model_names = timm.list_models(pretrained=True)
# pprint(model_names)


import sys
sys.path.append("../input/tez-lib/")
from accelerate import Accelerator
accelerator = Accelerator()

import random
import numpy as np
import torch
import os

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(42)


import torch.nn as nn

from tez import Tez, TezConfig
import tez
import albumentations
import pandas as pd
import cv2
import numpy as np
import timm
import torch.nn as nn
from sklearn import metrics
import torch
from tez.callbacks import EarlyStopping
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
class args:
    batch_size = 8
    image_size = 384
    epochs = 15  
    fold = 0


class PawpularDataset:
    def __init__(self, image_paths, dense_features, targets, augmentations):
        self.image_paths = image_paths
        self.dense_features = dense_features
        self.targets = targets
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, item):
        image = cv2.imread(self.image_paths[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.augmentations is not None:
            image = self.augmentations(image=image)["image"]
            
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        features = self.dense_features[item, :]
        targets = self.targets[item] / 100.0  # normalize to 0–1
        
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "features": torch.tensor(features, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.float),
        }



class PawpularModel(tez.Model):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("resnet50", pretrained=True, in_chans=3, num_classes=0, global_pool='avg')
        
        in_features = self.model.num_features
        
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(in_features + 12, 1)
        
        self.step_scheduler_after = "epoch"

    def monitor_metrics(self, outputs, targets):
        # We multiply by 100 here to get the RMSE on the original scale (1-100)
        outputs = torch.sigmoid(outputs).detach().cpu().numpy() * 100
        targets = targets.detach().cpu().numpy() * 100
        rmse = np.sqrt(metrics.mean_squared_error(targets, outputs))
        return {"rmse": rmse}

    def fetch_optimizer(self):
        opt = torch.optim.AdamW(self.parameters(), lr=2.5e-05, weight_decay=0.01)
        return opt

    def fetch_scheduler(self):
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return sch

    def forward(self, image, features, targets=None):
        x = self.model(image)
        x = torch.cat([x, features], dim=1)
        x = self.dropout(x)
        x = self.out(x)

        if targets is not None:
            # FIX: Using MSELoss for regression, which is correct.
            loss = nn.MSELoss()(torch.sigmoid(x), targets.view(-1, 1))
            metrics = self.monitor_metrics(x, targets)
            return x, loss, metrics
        return x, 0, {}


train_aug = albumentations.Compose(
    [
        albumentations.LongestMaxSize(args.image_size, p=1),
        albumentations.PadIfNeeded(args.image_size, args.image_size, p=1, border_mode=cv2.BORDER_CONSTANT),
        
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.1),
        albumentations.Rotate(limit=180, p=0.5),
        albumentations.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5
        ),
        
        albumentations.HueSaturationValue(
            hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
        ),
        albumentations.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
        ),
        albumentations.CoarseDropout(p=0.5),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

valid_aug = albumentations.Compose(
    [
        albumentations.LongestMaxSize(args.image_size, p=1),
        albumentations.PadIfNeeded(args.image_size,args.image_size, p=1, border_mode=cv2.BORDER_CONSTANT),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)


df = pd.read_csv("/kaggle/input/ruthwik-chikoti-10111-petfinder-kfolds/train_5folds.csv")
print(df.head())

for i in range(5):
    print(f'training fold: {i} start')
    args.fold = i
    
    df_train = df[df.kfold != args.fold].reset_index(drop=True)
    df_valid = df[df.kfold == args.fold].reset_index(drop=True)
    
    dense_features = [
        'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
        'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
    ]
    
    train_img_paths = [f"../input/petfinder-pawpularity-score/train/{x}.jpg" for x in df_train["Id"].values]
    valid_img_paths = [f"../input/petfinder-pawpularity-score/train/{x}.jpg" for x in df_valid["Id"].values]
    
    train_dataset = PawpularDataset(
        image_paths=train_img_paths,
        dense_features=df_train[dense_features].values,
        targets=df_train.Pawpularity.values / 100.0,
        augmentations=train_aug,
    )
    
    valid_dataset = PawpularDataset(
        image_paths=valid_img_paths,
        dense_features=df_valid[dense_features].values,
        targets=df_valid.Pawpularity.values / 100.0,
        augmentations=valid_aug,
    )
    
    model = PawpularModel()
    
    # FIX 1: Re-create the TezConfig object. The callback needs it.
    config = TezConfig(
        val_strategy="batch", # This is the value the callback is looking for
    )

    # FIX 2: Attach the config object to the model instance.
    model.config = config

    es = EarlyStopping(
        monitor="valid_rmse",
        model_path=f"model_resnet50_f{args.fold}.bin",
        patience=4,
        mode="min",
        save_weights_only=True,
    )
    
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        callbacks=[es],
        epochs=args.epochs,
        train_bs=args.batch_size,
        valid_bs=2 * args.batch_size,
        fp16=True
    )
    print(f'training fold: {i} complete')




