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
    batch_size = 64#16
    image_size = 384 #64


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class CustomDataset:
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


# class PawpularModel(tez.Model):
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()        
        self.backbone  = timm.create_model("resnet50", pretrained=False, in_chans=3)#resnet50#resnet101#eca_nfnet_l1#resnest101e
        self.dropout = nn.Dropout(0.5)# increase dropout
        # self.out = nn.Linear(1280+12, 1)
        self.out = nn.Linear(2048, 1)
        # self.out_final = nn.Linear(512, 1)
        
        self.step_scheduler_after = "epoch"


    def monitor_metrics(self, outputs, targets, loss):
        # rmse = torch.sqrt(loss).cpu().detach().numpy()
        rmse = loss
        if str(rmse) == 'nan':
            rmse = float('inf')
        # return {"rmse": rmse}
        return {"rmse": rmse}

    def optimizer_scheduler(self):
        opt = torch.optim.AdamW(self.parameters(), lr=2.5e-05, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return opt,sch

    def forward(self, image, features, targets=None):

        x = self.backbone.forward_features(image)
        x = self.backbone.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.out(x)


        if targets is not None:
            loss = nn.MSELoss()(x, targets.view(-1, 1))
            metrics = self.monitor_metrics(x, targets, loss)
            return x, loss, metrics
        return x, 0, {}

# PawpularModel()


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
i = 0
model = CustomModel()

# Load the state dict manually with strict=False to ignore mismatches
state_dict = torch.load("/kaggle/input/vinay-training-petfinder-my-pawpularity-contest/model_f0.bin", map_location="cpu")

# Filter out keys that don't match
model_state_dict = model.state_dict()
filtered_state_dict = {}

for key, value in state_dict.items():
    # Skip pos_weight
    if key == "pos_weight":
        continue
    
    # Check if key exists in model and shapes match
    if key in model_state_dict:
        if value.shape == model_state_dict[key].shape:
            filtered_state_dict[key] = value
        else:
            print(f"Shape mismatch for {key}: {value.shape} vs {model_state_dict[key].shape}")

# Load the filtered state dict
model.load_state_dict(filtered_state_dict, strict=False)
print("Model loaded successfully!")

# Create Tez config and wrap with Tez
config = TezConfig(
    training_batch_size=args.batch_size,
    validation_batch_size=2*args.batch_size,
    test_batch_size=2*args.batch_size,
    epochs=1,
    fp16=False,
    step_scheduler_after="epoch",
    val_strategy="epoch",
)

model = Tez(model)
model.config = config

df_test = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv")
test_img_paths = [f"/kaggle/input/siim-isic-melanoma-classification/jpeg/test/{x}.jpg" for x in df_test["image_name"].values]

dense_features = []

test_dataset = CustomDataset(
    image_paths=test_img_paths,
    dense_features=np.zeros((len(test_img_paths), 0)),
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


df_test.head()

















