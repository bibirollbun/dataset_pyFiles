import sys
sys.path.append("../input/tez-lib/")


# import timm
# from pprint import pprint
# model_names = timm.list_models(pretrained=True)
# pprint(model_names)


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
from sklearn.metrics import mean_squared_error
np.Inf = np.inf


class args:
    batch_size = 8
    image_size = 384
    epochs = 10
    fold = 0


MODELS_TO_TRAIN = ["inception_v3"]


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
            augmented = self.augmentations(image=image)
            image = augmented["image"]
            
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        features = self.dense_features[item, :]
        targets = self.targets[item]
        
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "features": torch.tensor(features, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.float),
        }


class PawpularModel(nn.Module):
    def __init__(self, model_name="resnet50", dense_dim=12, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=3)
        n_features = self.model.get_classifier().in_features
        self.model.reset_classifier(0)
        self.out = nn.Linear(n_features + dense_dim, 1)

    
    def monitor_metrics(self, outputs, targets):
        outputs = torch.sigmoid(outputs).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        rmse = np.sqrt(mean_squared_error(targets, outputs))
        return {"rmse": torch.tensor(rmse)}
        
    def optimizer_scheduler(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-6)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return opt, sch

    def forward(self, image, features, targets=None):
        image_features = self.model(image)
        x = torch.cat([image_features, features], dim=1)
        output = self.out(x)
        
        if targets is not None:
            pos_weight = torch.tensor([55.7]).to(image.device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fn(output, targets.view(-1, 1))
            
            # This call now gets the RMSE metric
            metrics = self.monitor_metrics(output, targets)
            
            return output, loss, metrics
        return output, 0, {}


train_aug = albumentations.Compose(
    [
        albumentations.LongestMaxSize(args.image_size, p=1),
        albumentations.PadIfNeeded(args.image_size,args.image_size, p=1,border_mode=0),
        
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
        albumentations.PadIfNeeded(args.image_size,args.image_size, p=1,border_mode=0),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)


df = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/train.csv")
df_folds = pd.read_csv("/kaggle/input/10186-paawanjotkaur-folds/train_5folds.csv")
df['kfold'] = df_folds['kfold']


for model_name in MODELS_TO_TRAIN:
    print("\n" + "#"*50)
    print(f"### Training Model: {model_name.upper()} | Fold: {args.fold} ###")
    print("#"*50)

    
    df_train = df[df.kfold != args.fold].reset_index(drop=True)
    df_valid = df[df.kfold == args.fold].reset_index(drop=True)

    dense_features_list = [
    'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
    'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
    ]
    
    train_img_paths = [f"/kaggle/input/petfinder-pawpularity-score/train/{x}.jpg" for x in df_train['Id'].values]
    valid_img_paths = [f"/kaggle/input/petfinder-pawpularity-score/train/{x}.jpg" for x in df_valid['Id'].values]
    
    
    train_dataset = PawpularDataset(
        image_paths=train_img_paths,
        dense_features=df_train[dense_features_list].values,
        targets=df_train.Pawpularity.values,
        augmentations=train_aug,
    )
    valid_dataset = PawpularDataset(
        image_paths=valid_img_paths,
        dense_features=df_valid[dense_features_list].values,
        targets=df_valid.Pawpularity.values,
        augmentations=valid_aug,
    )

    
    model = PawpularModel(model_name=model_name, dense_dim=len(dense_features_list))
    model = Tez(model)

    
    config = TezConfig(
        training_batch_size=args.batch_size,
        validation_batch_size=2 * args.batch_size,
        epochs=args.epochs,
        step_scheduler_after="epoch",
        step_scheduler_metric="valid_rmse",
        fp16=True,
        val_strategy="batch",
    )

    
    
    es = EarlyStopping(
        monitor="valid_rmse", 
        model_path=f"model_{model_name}_rmse_f{args.fold}.bin",
        patience=3,
        mode="min", 
        save_weights_only=True,
    )

    
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        callbacks=[es],
        config=config,
    )
    print(f"--- Training for {model_name} complete ---")

print("\nAll models have been trained and saved.")

