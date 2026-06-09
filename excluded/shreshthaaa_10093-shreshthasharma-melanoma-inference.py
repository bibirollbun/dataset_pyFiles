import sys
sys.path.append("../input/tez-lib/")


import tez
from tez import Tez, TezConfig
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
import math
from sklearn.preprocessing import LabelEncoder


class args:
    batch_size = 64
    image_size = 384


class MelanomaDataset:
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

        if self.targets is not None:
            target = self.targets[item]
        else:
            target = 0.0
        
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "features": torch.tensor(features, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.float),
        }


class MelanomaModel(nn.Module):
    def __init__(self):
        super().__init__()        
        self.model = timm.create_model("resnet50", pretrained=True, in_chans=3)

        
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(1000+4, 1)
        
        self.step_scheduler_after = "epoch"


    def monitor_metrics(self, outputs, targets, loss):
        valid_binaryloss = loss
        if str(valid_binaryloss) == 'nan':
            valid_binaryloss = float('inf')
        return {"binaryloss": valid_binaryloss}

    def optimizer_scheduler(self):
        opt = torch.optim.AdamW(self.parameters(), lr=2.5e-05, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return opt,sch

    def forward(self, image, features, targets=None):

        x = self.model(image)
        x = self.dropout(x)
        x = torch.cat([x, features], dim=1)
        x = self.dropout(x)
        x = self.out(x)

        if targets is not None:
            loss = nn.BCEWithLogitsLoss()(x, targets.view(-1, 1).type_as(x))
            metrics = self.monitor_metrics(x, targets, loss)
            return x, loss, metrics
        return x, 0, {}



test_aug = albumentations.Compose(
    [
        albumentations.Resize(args.image_size, args.image_size, p=1),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)


super_final_predictions = []
for i in range(3):
    model = MelanomaModel()
    model = Tez(model)
    model.load(f"/kaggle/input/10093-shreshthasharma-melanoma-training/model_f{i}.bin", weights_only=True)
    
    df_test = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv")

    for col in ['diagnosis', 'benign_malignant']:
        if col not in df_test.columns:
            df_test[col] = 0  # Add dummy columns

    possible_categorical_cols = ['sex', 'anatom_site_general_challenge', 'diagnosis', 'benign_malignant']
    categorical_cols = [col for col in possible_categorical_cols if col in df_test.columns]

    le = LabelEncoder()
    for col in categorical_cols:
        df_test[col] = le.fit_transform(df_test[col].astype(str))

    df_test[categorical_cols] = df_test[categorical_cols].astype(float)

    test_img_paths = [f"/kaggle/input/siim-isic-melanoma-classification/jpeg/test/{x}.jpg" for x in df_test["image_name"].values]
    
    dense_features = categorical_cols
    
    test_dataset = MelanomaDataset(
        image_paths=test_img_paths,
        dense_features=df_test[dense_features].values,
        targets=np.ones(len(test_img_paths)),
        augmentations=test_aug,
    )
    test_predictions = model.predict(test_dataset, batch_size=2*args.batch_size, n_jobs=-1)
    
    final_test_predictions = []
    for preds in tqdm(test_predictions):
        final_test_predictions.extend(preds.ravel().tolist())

    super_final_predictions.append(final_test_predictions)
    
super_final_predictions = np.mean(np.column_stack(super_final_predictions), axis=1)
df_test["target"] = super_final_predictions
df_test = df_test[["image_name", "target"]]
df_test.to_csv("submission.csv", index=False)


print(df_test.head())




