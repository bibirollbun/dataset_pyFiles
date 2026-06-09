import sys
sys.path.append("../input/tez-lib/")


# import tez
# from tez import Tez, TezConfig
# import albumentations
# import pandas as pd
# import cv2
# import numpy as np
# import timm
# import torch.nn as nn
# from sklearn import metrics
# import torch
# from tez.callbacks import EarlyStopping
# from tqdm import tqdm
# import math


    # class args:
    #     batch_size = 64#16
    #     image_size = 384 #64


# def sigmoid(x):
#     return 1 / (1 + math.exp(-x))


# class CustomDataset:
#     def __init__(self, image_paths, dense_features, targets, augmentations):
#         self.image_paths = image_paths
#         self.dense_features = dense_features
#         self.targets = targets
#         self.augmentations = augmentations
        
#     def __len__(self):
#         return len(self.image_paths)
    
#     def __getitem__(self, item):
#         image = cv2.imread(self.image_paths[item])
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         if self.augmentations is not None:
#             augmented = self.augmentations(image=image)
#             image = augmented["image"]
            
#         image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
#         features = self.dense_features[item, :]
#         targets = self.targets[item]
        
#         return {
#             "image": torch.tensor(image, dtype=torch.float),
#             "features": torch.tensor(features, dtype=torch.float),
#             "targets": torch.tensor(targets, dtype=torch.float),
#         }


# # class CustomModel(tez.Model):
# class CustomModel(nn.Module):
#     def __init__(self):
#         super().__init__()        
#         self.model = timm.create_model("resnet50", pretrained=False, in_chans=3)#resnet50#resnet101#eca_nfnet_l1#resnest101e
#         self.dropout = nn.Dropout(0.5)# increase dropout
#         # self.out = nn.Linear(1280+12, 1)
#         self.out = nn.Linear(self.model.num_features, 1)
#         # self.out_final = nn.Linear(512, 1)
        
#         self.step_scheduler_after = "epoch"


#     def monitor_metrics(self, outputs, targets, loss):
#         # rmse = torch.sqrt(loss).cpu().detach().numpy()
#         rmse = loss
#         if str(rmse) == 'nan':
#             rmse = float('inf')
#         # return {"rmse": rmse}
#         return {"rmse": rmse}

#     def optimizer_scheduler(self):
#         opt = torch.optim.AdamW(self.parameters(), lr=2.5e-05, weight_decay=0.01)
#         sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#             opt, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
#         )
#         return opt,sch

#     def forward(self, image, features, targets=None):

#         x = self.model(image)
#         x = self.dropout(x)
#         # x = torch.cat([x, features], dim=1)
#         # x = self.dropout(x)
#         x = self.out(x)
#         # x = self.dropout(x)
#         # x = self.out_final(x)

#         if targets is not None:
#             loss = nn.MSELoss()(x, targets.view(-1, 1))
#             metrics = self.monitor_metrics(x, targets, loss)
#             return x, loss, metrics
#         return x, 0, {}

# # CustomModel()


# test_aug = albumentations.Compose(
#     [
#         albumentations.Resize(args.image_size, args.image_size, p=1),
#         albumentations.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225],
#             max_pixel_value=255.0,
#             p=1.0,
#         ),
#     ],
#     p=1.0,
# )


# super_final_predictions = []
# i=0
# model = CustomModel()
# model = Tez(model)
# # model.load(f"/kaggle/input/training-petfinder-my-pawpularity-contest/model_f{i}.bin", weights_only=True)
# model.load(f"/kaggle/input/naresh-kumar-t-23bcs1006-training/model_f0.bin", weights_only=True)
# # model.load(f"/kaggle/input/resnet50v1/pytorch/default/1/model_f{i} (1).bin", weights_only=True)

# df_test = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv")
# test_img_paths = [f"/kaggle/input/siim-isic-melanoma-classification/jpeg/test/{x}.jpg" for x in df_test["Id"].values]

# dense_features = [
# ]

# test_dataset = CustomDataset(
#     image_paths=test_img_paths,
#     dense_features=df_test[dense_features].values,
#     targets=np.ones(len(test_img_paths)),
#     augmentations=test_aug,
# )
# test_predictions = model.predict(test_dataset, batch_size=2*args.batch_size, n_jobs=-1)

# final_test_predictions = []
# for preds in tqdm(test_predictions):
#     final_test_predictions.extend(preds.ravel().tolist())

# final_test_predictions = [sigmoid(x) * 100 for x in final_test_predictions]
# super_final_predictions.append(final_test_predictions)

# super_final_predictions = np.mean(np.column_stack(super_final_predictions), axis=1)
# df_test["target"] = super_final_predictions
# df_test = df_test[["image_name", "target"]]
# df_test.to_csv("submission.csv", index=False)


# df_test.head()



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
import math

class args:
    batch_size = 64
    image_size = 384 

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
        
        # Handle dense features gracefully if empty
        features = self.dense_features[item, :] if (self.dense_features is not None and self.dense_features.size) else np.zeros((0,), dtype=np.float32)
        targets = self.targets[item]
        
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "features": torch.tensor(features, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.float),
        }

# --- CORRECTED MODEL DEFINITION (Subclass of tez.Model) ---
class CustomModel(tez.Model): 
    def __init__(self, pos_weight=None):
        super().__init__()        
        # FIX: Renamed to self.backbone to match checkpoint keys (e.g., 'backbone.conv1.weight')
        self.backbone = timm.create_model("resnet50", pretrained=False, in_chans=3, num_classes=0, global_pool="avg")
        
        nf = getattr(self.backbone, "num_features", 2048)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(nf, 1)

        # FIX: loss_fn attribute must be present to match checkpoint key 'loss_fn.pos_weight'
        if pos_weight is not None:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss() 

        self.step_scheduler_after = "epoch"


    def monitor_metrics(self, outputs, targets, loss):
        # Using loss as the metric for simplicity in prediction stub
        rmse = loss
        if str(rmse) == 'nan': rmse = float('inf')
        return {"rmse": rmse}

    def optimizer_scheduler(self):
        # Method required by tez.Model
        opt = torch.optim.AdamW(self.parameters(), lr=2.5e-05, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return opt,sch

    def forward(self, image, features, targets=None):
        x = self.backbone(image) # FIX: Use self.backbone
        x = self.dropout(x)
        x = self.out(x)
        
        if targets is not None:
            loss = self.loss_fn(x.view(-1), targets.view(-1).type_as(x.view(-1))) 
            
            rmse = loss 
            if str(rmse) == 'nan': rmse = float('inf')
            metrics = {"rmse": rmse}
            
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
    p=1.0,)

super_final_predictions = []
i=0

model_nn = CustomModel(pos_weight=None) 

config = TezConfig(
    training_batch_size=args.batch_size, 
    validation_batch_size=2 * args.batch_size, 
    fp16=True, # Assuming FP16 was used in training
)
model = Tez(model_nn, config=config) 
# ----------------------------------------------------------------------


MODEL_PATH = f"/kaggle/input/naresh-kumar-t-23bcs1006-training/model_f0.bin"
print(f"Loading weights from {MODEL_PATH} with strict=False...")

model_dict = torch.load(MODEL_PATH, map_location="cpu")

state_dict_to_load = model_dict.get("state_dict", model_dict)

model_nn.load_state_dict(state_dict_to_load, strict=False)

print("Model weights loaded successfully by ignoring unexpected keys!")
# ----------------------------------------------------------------------


df_test = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv")

# Use 'image_name' if available, otherwise 'Id'
IMAGE_ID_COL = "image_name" if "image_name" in df_test.columns else "Id" 

test_img_paths = [f"/kaggle/input/siim-isic-melanoma-classification/jpeg/test/{x}.jpg" for x in df_test[IMAGE_ID_COL].values]

dense_features = [] 

test_dataset = CustomDataset(
    image_paths=test_img_paths,
    dense_features=df_test[dense_features].values if dense_features else np.zeros((len(df_test), 0)),
    targets=np.ones(len(test_img_paths)),
    augmentations=test_aug,
)

test_predictions = model.predict(test_dataset, batch_size=2*args.batch_size, n_jobs=-1)

final_test_predictions = []
for preds in tqdm(test_predictions):
    # Convert logits (raw output) to probabilities (0-1) using sigmoid
    probs = torch.sigmoid(torch.tensor(preds)).ravel().tolist()
    final_test_predictions.extend(probs)

super_final_predictions.append(final_test_predictions)
super_final_predictions = np.mean(np.column_stack(super_final_predictions), axis=1)

df_test["image_name"] = df_test[IMAGE_ID_COL]
df_test["target"] = super_final_predictions
df_test = df_test[["image_name", "target"]]
df_test.to_csv("submission.csv", index=False)













