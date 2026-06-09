import sys
sys.path.append("../input/tez-lib/")
sys.path.append("../input/timmmaster/")


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


class args:
    batch_size = 64
    image_size = 512


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


class PawpularModel(tez.Model):
    def __init__(self, model_name):
        super().__init__()

        self.model = timm.create_model(model_name, pretrained=False, in_chans=3)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 128)
        self.dropout = nn.Dropout(0.1)
        self.dense1 = nn.Linear(140, 64)
        self.dense2 = nn.Linear(64, 1)

    def forward(self, image, features, targets=None):

        x = self.model(image)
        x = self.dropout(x)
        x = torch.cat([x, features], dim=1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
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

for fold_ in range(10):
    model = PawpularModel(model_name="tf_efficientnet_b0_ns")
    model.load(f"../input/pawpular-models/model_f{fold_}.bin", device="cuda", weights_only=True)

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
    
    super_final_predictions.append(final_test_predictions)

super_final_predictions = np.mean(np.column_stack(super_final_predictions), axis=1)
df_test["Pawpularity"] = super_final_predictions
df_test = df_test[["Id", "Pawpularity"]]
df_test.to_csv("submission.csv", index=False)


df_test.head()




