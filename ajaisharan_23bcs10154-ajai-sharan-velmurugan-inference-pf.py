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


class args:
    batch_size = 64
    image_size = 384


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


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
    def __init__(self, model_name="inception_v3", dense_dim=12):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False, in_chans=3)
        n_features = self.model.get_classifier().in_features
        self.model.reset_classifier(0)
        self.out = nn.Linear(n_features + dense_dim, 1)

    def forward(self, image, features, targets=None):
        image_features = self.model(image)
        x = torch.cat([image_features, features], dim=1)
        output = self.out(x)

        if targets is not None:
            loss = nn.MSELoss()(output, targets.view(-1, 1))
            metrics = {"rmse": torch.sqrt(loss)}
            return output, loss, metrics
        return output, 0, {}


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

model = PawpularModel()
model = Tez(model)
model.load(f"/kaggle/input/ajai-sharan-velmurugan-23bcs10154-training-pf/model_inception_v3_rmse_f0.bin", weights_only=True)

df_test = pd.read_csv("../input/petfinder-pawpularity-score/test.csv")
test_img_paths = [f"../input/petfinder-pawpularity-score/test/{x}.jpg" for x in df_test["Id"].values]

dense_features = [
    'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
    'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
]

test_dataset = PawpularDataset(
    image_paths=test_img_paths,
    dense_features=df_test[dense_features].values,
    targets=np.ones(len(test_img_paths)),
    augmentations=test_aug,
)
test_predictions = model.predict(test_dataset, batch_size=2*args.batch_size, n_jobs=-1)

final_test_predictions = []
for preds in tqdm(test_predictions):
    final_test_predictions.extend(preds.ravel().tolist())

final_test_predictions = [sigmoid(x) * 100 for x in final_test_predictions]
super_final_predictions.append(final_test_predictions)

super_final_predictions = np.mean(np.column_stack(super_final_predictions), axis=1)
df_test["Pawpularity"] = super_final_predictions
df_test = df_test[["Id", "Pawpularity"]]
df_test.to_csv("submission.csv", index=False)


df_test.head()

