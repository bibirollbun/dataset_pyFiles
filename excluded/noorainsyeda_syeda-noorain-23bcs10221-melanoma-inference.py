import sys
from sklearn.preprocessing import LabelEncoder,StandardScaler
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
        
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "features": torch.tensor(features, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.float),
        }


class MelanomaModel(tez.Model):
    def __init__(self):
        super().__init__()        
        self.model = timm.create_model("resnet50", pretrained=False, in_chans=3)#resnet50#resnet101#eca_nfnet_l1#resnest101e

        
        self.dropout = nn.Dropout(0.5)# increase dropout
        self.out = nn.Linear(1000+3, 512)
        # self.out = nn.Linear(1000, 1)
        self.out_final = nn.Linear(512, 1)
        
        self.step_scheduler_after = "epoch"


    def monitor_metrics(self, outputs, targets):
        outputs = torch.sigmoid(outputs).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        try:
            auc = metrics.roc_auc_score(targets, outputs)
        except ValueError:
            auc = 0.5
        return {"auc": torch.tensor(auc, dtype=torch.float32)}

    def optimizer_scheduler(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=0.01)
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
        x = self.dropout(x)
        x = self.out_final(x)
        
        if targets is not None:
            loss = nn.BCEWithLogitsLoss()(x, targets.view(-1, 1).type_as(x))
            metrics = self.monitor_metrics(x, targets)
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


df_test = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")
df_train = pd.read_csv("/kaggle/input/syeda-noorain-23bcs10221-melanoma-fold-creation/train_5folds.csv") 


df_train['age_approx'] = df_train['age_approx'].fillna(df_train['age_approx'].mean())
df_test['age_approx'] = df_test['age_approx'].fillna(df_train['age_approx'].mean()) 

scaler = StandardScaler()
df_train['age_approx'] = scaler.fit_transform(df_train[['age_approx']])
df_test['age_approx'] = scaler.transform(df_test[['age_approx']]) 


super_final_predictions = []

model = MelanomaModel()
model = Tez(model)
# # model.load(f"/kaggle/input/training-petfinder-my-pawpularity-contest/model_f{i}.bin", weights_only=True)
# model.load(f"/kaggle/input/melanoma_models/pytorch/default/1/model_f{i}.bin", weights_only=True)
# # model.load(f"/kaggle/input/resnet50v1/pytorch/default/1/model_f{i} (1).bin", weights_only=True)


test_img_paths = [f"../input/siim-isic-melanoma-classification/jpeg/test/{x}.jpg" for x in df_test["image_name"].values]

dense_features = [
    'sex','age_approx','anatom_site_general_challenge'
]


for col in ['sex','anatom_site_general_challenge']:
    le = LabelEncoder()
    df_test[col] = le.fit_transform(df_test[col].astype(str))
test_dataset = MelanomaDataset(
    image_paths=test_img_paths,
    dense_features=df_test[dense_features].values.astype(np.float32),
    targets=np.ones(len(test_img_paths)),
    augmentations=test_aug,
)
# test_predictions = model.predict(test_dataset, batch_size=2*args.batch_size, n_jobs=-1)

for i in range(2):
    print(f"--- Predicting with Fold {i} ---")
    
    # Load the weights for the CURRENT fold
    model_path = f"../input/melanoma_models/pytorch/default/1/model_f{i}.bin"
    model.load(model_path, weights_only=True)
    
    # Run prediction for the CURRENT model
    # The output is a generator, so we convert it to a list
    preds_generator = model.predict(test_dataset, batch_size=2*args.batch_size, n_jobs=-1)
    
    current_fold_preds = []
    for batch_preds in tqdm(preds_generator):
        current_fold_preds.extend(batch_preds.ravel().tolist())
    
    super_final_predictions.append(current_fold_preds)

# 3. Average Predictions and Create Submission File (AFTER the loop)
print("--- Averaging predictions and creating submission file ---")

# Use np.column_stack to create a (num_samples, num_folds) array, then average across folds
avg_preds = np.mean(np.column_stack(super_final_predictions), axis=1)

# Apply sigmoid to the averaged logits
final_submission_preds = [sigmoid(x) for x in avg_preds]

df_test["target"] = final_submission_preds
df_test = df_test[["image_name", "target"]]
df_test.to_csv("submission.csv", index=False)

print("Submission file created successfully!")



df_test.head()

















