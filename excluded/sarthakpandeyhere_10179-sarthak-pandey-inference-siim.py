import sys
sys.path.append("../input/tez-lib/")


import math
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
from tqdm import tqdm
import albumentations
import timm
import tez
from tez import Tez, TezConfig
from tez.callbacks import EarlyStopping



class args:
    batch_size = 64#16
    image_size = 384 #64


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


class CustomDataset:
    def __init__(self, image_paths, dense_features, targets, augmentations):
        # Initialize dataset with image paths, dense features, targets, and augmentations
        self.image_paths = image_paths
        self.dense_features = dense_features
        self.targets = targets
        self.augmentations = augmentations
        
    def __len__(self):
        # Return total number of samples
        return len(self.image_paths)
    
    def __getitem__(self, item):
        # Read image from file
        image = cv2.imread(self.image_paths[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentations if provided
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
            
        # Convert image to channel-first format and float32
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        # Get dense features and target for this item
        features = self.dense_features[item, :]
        targets = self.targets[item]
        
        # Return dictionary of tensors
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "features": torch.tensor(features, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.float),
        }



class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Base model (can switch architectures by changing model name)
        self.model = timm.create_model("resnet50", pretrained=False, in_chans=3)  # resnet50#resnet101#eca_nfnet_l1#resnest101e
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Final output layer (binary classification)
        self.out = nn.Linear(1000, 1)
        # self.out = nn.Linear(1280+12, 1)
        # self.out_final = nn.Linear(512, 1)
        
        # Scheduler step frequency
        self.step_scheduler_after = "epoch"

    def monitor_metrics(self, outputs, targets, loss):
        # Monitor binary cross-entropy loss
        bce = loss
        if str(bce) == 'nan':
            rmse = float('inf')
        return {"bce": bce}

    def optimizer_scheduler(self):
        # Define optimizer and learning rate scheduler
        opt = torch.optim.AdamW(self.parameters(), lr=2.5e-05, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return opt, sch

    def forward(self, image, features, targets=None):
        # Forward pass through base model
        x = self.model(image)
        x = self.dropout(x)

        # Optionally concatenate dense features
        # x = torch.cat([x, features], dim=1)
        # x = self.dropout(x)

        # Output prediction
        x = self.out(x)
        # x = self.dropout(x)
        # x = self.out_final(x)

        # Compute loss and metrics if targets provided
        if targets is not None:
            loss = nn.BCEWithLogitsLoss()(x, targets.view(-1, 1).float())
            metrics = self.monitor_metrics(x, targets, loss)
            return x, loss, metrics

        # During inference, only return predictions
        return x, 0, {}



# Test-time image augmentation pipeline
test_aug = albumentations.Compose(
    [
        # Resize image to the desired input size
        albumentations.Resize(args.image_size, args.image_size, p=1),
        
        # Normalize image using ImageNet mean and std values
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)



# Initialize list to store final predictions
super_final_predictions = []
i = 0

# Load trained model
model = CustomModel()
model = Tez(model)
model.load("/kaggle/input/10179-sarthakpandey-training-siim/model_f0.bin", weights_only=True)

# Read test data and prepare image paths
df_test = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv")
test_img_paths = [f"/kaggle/input/siim-isic-melanoma-classification/jpeg/test/{x}.jpg" for x in df_test["image_name"].values]

# Define dense features (if any)
dense_features = []

# Create test dataset
test_dataset = CustomDataset(
    image_paths=test_img_paths,
    dense_features=df_test[dense_features].values,
    targets=np.ones(len(test_img_paths)),  # dummy targets
    augmentations=test_aug,
)

# Generate predictions
test_predictions = model.predict(test_dataset, batch_size=2 * args.batch_size, n_jobs=-1)

# Flatten prediction outputs
final_test_predictions = []
for preds in tqdm(test_predictions):
    final_test_predictions.extend(preds.ravel().tolist())

# Apply sigmoid to convert logits to probabilities
super_final_predictions = torch.sigmoid(torch.tensor(final_test_predictions)).numpy()

# Prepare submission file
df_test["target"] = super_final_predictions
df_test = df_test[["image_name", "target"]]
df_test.to_csv("submission.csv", index=False)



df_test.head()

















