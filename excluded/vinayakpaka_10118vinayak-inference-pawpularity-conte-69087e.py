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
import os


class args:
    batch_size = 64 # Increased for inference efficiency
    image_size = 448 # Must match the training image size (448)


def sigmoid(x):
    # This function is not used in the final prediction logic of the corrected code,
    # as the model directly predicts the score without a sigmoid activation, 
    # but it's kept for completeness if needed later.
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
    def __init__(self):
        super().__init__()        
        # Changed to match training: seresnet50, pretrained=True
        self.model = timm.create_model("seresnet50", pretrained=False, in_chans=3) 
        self.dropout = nn.Dropout(0.2) # Changed to match training: 0.2
        # Changed to match training: 1000 (seresnet50 output) + 12 (dense features)
        self.out = nn.Linear(1000+12, 1) 
        self.step_scheduler_after = "epoch"


    def monitor_metrics(self, outputs, targets, loss):
        # FIX: Return the RMSE as a PyTorch tensor, NOT a NumPy array
        rmse = torch.sqrt(loss) 
        return {"rmse": rmse}
    
    def optimizer_scheduler(self):
        # Only needed if you were to fine-tune; included for completeness
        opt = torch.optim.AdamW(self.parameters(), lr=2.5e-05, weight_decay=0.001) 
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=5, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return opt,sch

    def forward(self, image, features, targets=None):

        x = self.model(image)
        x = self.dropout(x)
        # FIX: Concatenate image and dense features as in training
        x = torch.cat([x, features], dim=1) 
        x = self.dropout(x) # Apply dropout again
        x = self.out(x)

        if targets is not None:
            # Using MSELoss for regression
            loss = nn.MSELoss()(x, targets.view(-1, 1))
            metrics = self.monitor_metrics(x, targets, loss)
            return x, loss, metrics
        # Return only the prediction for inference
        return x, 0, {}


test_aug = albumentations.Compose(
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




df_test = pd.read_csv("../input/petfinder-pawpularity-score/test.csv")
test_img_paths = [f"../input/petfinder-pawpularity-score/test/{x}.jpg" for x in df_test["Id"].values]

dense_features = [
    'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
    'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
]

test_dataset = PawpularDataset(
    image_paths=test_img_paths,
    dense_features=df_test[dense_features].values,
    # targets are dummy values for inference
    targets=np.ones(len(test_img_paths)), 
    augmentations=test_aug,
)

# Initialize and load ONLY the fold 3 model
model = PawpularModel()
model = Tez(model)
# Load the specified model path for fold 3.
model_path = "/kaggle/input/10118-vinayak-paka-training-v3-pawpularity-contest/model_f3.bin"
if not os.path.exists(model_path):
    print(f"ERROR: Model file not found at {model_path}")
else:
    model.load(model_path, weights_only=True)
    print(f"Loaded model from {model_path}")


# Perform prediction
test_predictions = model.predict(test_dataset, batch_size=2*args.batch_size, n_jobs=-1)

# Extract predictions from the list of numpy arrays
final_test_predictions = []
for preds in tqdm(test_predictions):
    final_test_predictions.extend(preds.ravel().tolist())

# Post-processing: No sigmoid is needed as the training code does not use it.
# The Pawpularity score ranges from 1 to 100.
super_final_predictions = np.array(final_test_predictions).clip(1, 100)

df_test["Pawpularity"] = super_final_predictions
df_test = df_test[["Id", "Pawpularity"]]
df_test.to_csv("submission.csv", index=False)
print("Submission file created successfully.")



df_test.head()

















