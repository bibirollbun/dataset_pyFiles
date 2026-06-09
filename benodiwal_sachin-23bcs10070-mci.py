import sys
sys.path.append("/kaggle/input/tez-main/tez-main")


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


class args:
    batch_size = 64#16
    image_size = 384 #64


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


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
        
        image = image.astype(np.float32) / 255.0   
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        features = self.dense_features[item, :].astype(np.float32)
        targets = self.targets[item]
        
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "features": torch.tensor(features, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.float),
        }


class MelanomaModel(nn.Module):
    def __init__(self):
        super().__init__()        
        self.model = timm.create_model("resnet50", pretrained=False, num_classes=0)#resnet50#resnet101#eca_nfnet_l1#resnest101e

        
        self.dropout = nn.Dropout(0.5)# increase dropout

        self.out = nn.Linear(2066, 1)

        
        self.step_scheduler_after = "epoch"


    def monitor_metrics(self, outputs, targets, loss):
        return {"bce_loss": loss}

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
            targets = targets.view(-1, 1).float()
            loss = nn.BCEWithLogitsLoss()(x, targets)
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



test_df = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv")

# --- handle missing categorical columns ---
# If 'diagnosis' is missing, create it with default 'unknown'
if "diagnosis" not in test_df.columns:
    test_df["diagnosis"] = "unknown"

# Fill missing site info
test_df["anatom_site_general_challenge"] = test_df["anatom_site_general_challenge"].fillna("unknown")

# --- one-hot encode ---
test_df = pd.get_dummies(
    test_df,
    columns=["diagnosis", "anatom_site_general_challenge"],
    dtype=np.uint8
)

# --- encode sex as 1/0 ---
test_df["sex"] = test_df["sex"].map({"male": 1, "female": 0}).fillna(0)

# --- handle numeric missing values ---
if "age_approx" in test_df.columns:
    test_df["age_approx"] = test_df["age_approx"].fillna(test_df["age_approx"].median())

# --- ensure same feature columns as training ---
expected_features= [
    'image_name','sex', 'age_approx',
    'diagnosis_atypical melanocytic proliferation', 'diagnosis_cafe-au-lait macule',
    'diagnosis_lentigo NOS', 'diagnosis_lichenoid keratosis',
    'diagnosis_melanoma', 'diagnosis_nevus', 'diagnosis_seborrheic keratosis',
    'diagnosis_solar lentigo', 'diagnosis_unknown',
    'anatom_site_general_challenge_head/neck', 'anatom_site_general_challenge_lower extremity',
    'anatom_site_general_challenge_oral/genital', 'anatom_site_general_challenge_palms/soles',
    'anatom_site_general_challenge_torso', 'anatom_site_general_challenge_unknown',
    'anatom_site_general_challenge_upper extremity'
]


for col in expected_features:
    if col not in test_df.columns:
        test_df[col] = 0

# Ensure column order matches training
test_df = test_df[expected_features]





test_df.head()


super_final_predictions = []
df_test = test_df
test_img_paths = [f"/kaggle/input/siim-isic-melanoma-classification/jpeg/test/{x}.jpg" for x in df_test["image_name"].values]
for i in range(5):
    i=0
    model = MelanomaModel()
    model = Tez(model)
    model.load(f"/kaggle/input/mc-moldels/model_f{i}.bin", weights_only=True)
    
    
    dense_features = [
        'sex', 'age_approx', 'diagnosis_atypical melanocytic proliferation', 'diagnosis_cafe-au-lait macule', 'diagnosis_lentigo NOS', 'diagnosis_lichenoid keratosis', 'diagnosis_melanoma', 'diagnosis_nevus', 'diagnosis_seborrheic keratosis', 'diagnosis_solar lentigo', 'diagnosis_unknown', 'anatom_site_general_challenge_head/neck', 'anatom_site_general_challenge_lower extremity', 'anatom_site_general_challenge_oral/genital', 'anatom_site_general_challenge_palms/soles', 'anatom_site_general_challenge_torso', 'anatom_site_general_challenge_unknown', 'anatom_site_general_challenge_upper extremity'
    ]
    
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
    
    # final_test_predictions = [sigmoid(x) * 100 for x in final_test_predictions]
    super_final_predictions.append(final_test_predictions)

super_final_predictions = np.mean(np.column_stack(super_final_predictions), axis=1)
df_test["target"] = super_final_predictions
df_test = df_test[["image_name", "target"]]
df_test.to_csv("submission.csv", index=False)


df=pd.read_csv("/kaggle/working/submission.csv")
df

