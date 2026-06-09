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
import torch
from tqdm import tqdm


class args:
    batch_size = 32
    image_size = 384


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



class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()        
        self.model = timm.create_model("resnet50", pretrained=False, in_chans=3)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(1000, 1)
        self.step_scheduler_after = "epoch"

    def monitor_metrics(self, outputs, targets, loss):
        # Keep metrics as tensor to avoid Tez float error
        return {"rmse": loss.detach()}

    def optimizer_scheduler(self):
        opt = torch.optim.AdamW(self.parameters(), lr=2.5e-5, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=1, eta_min=1e-6
        )
        return opt, sch

    def forward(self, image, features, targets=None):
        x = self.model(image)
        x = self.dropout(x)
        x = self.out(x)

        if targets is not None:
            loss = nn.MSELoss()(x, targets.view(-1, 1))
            metrics = self.monitor_metrics(x, targets, loss)
            return x, loss, metrics
        return x, 0, {}


test_aug = albumentations.Compose([
    albumentations.LongestMaxSize(args.image_size, p=1),
    albumentations.PadIfNeeded(args.image_size, args.image_size, p=1, border_mode=0),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0,
        p=1.0
    ),
], p=1.0)


i = 0  # fold number
model = CustomModel()
model = Tez(model)
model.load(f"/kaggle/input/10121-yash-agarwal-training-siim/model_f{i}.bin", weights_only=True)


df_test = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv")
test_img_paths = [f"/kaggle/input/siim-isic-melanoma-classification/jpeg/test/{x}.jpg" for x in df_test["image_name"].values]
dense_features = []


test_dataset = CustomDataset(
    image_paths=test_img_paths,
    dense_features=df_test[dense_features].values,
    targets=np.ones(len(test_img_paths)),
    augmentations=test_aug,
)


test_predictions = model.predict(test_dataset, batch_size=args.batch_size, n_jobs=-1)


final_test_predictions = []
for preds in tqdm(test_predictions):
    final_test_predictions.extend(torch.sigmoid(torch.tensor(preds)).numpy().ravel().tolist())


df_test["target"] = final_test_predictions
df_test = df_test[["image_name", "target"]]
df_test.to_csv("submission.csv", index=False)
df_test.head()

