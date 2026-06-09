import sys
sys.path.append("../input/tez-lib/")


import tez
from tez import Tez, TezConfig
import albumentations
import pandas as pd
import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from tqdm import tqdm


class args:
    batch_size = 64
    image_size = 384


BEST_MODEL_NAME = "inception_v3"
INFERENCE_FOLD = 0
BASE_IMAGE_PATH = "/kaggle/input/siim-isic-melanoma-classification/jpeg/test/"


class targetDataset:
    def __init__(self, image_paths, dense_features, targets, augmentations):
        self.image_paths = image_paths
        self.dense_features = dense_features
        self.targets = targets 
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, item):
        image_path = self.image_paths[item]
        image = cv2.imread(image_path)
        
       
        if image is None:
            raise FileNotFoundError(f"Could not read image file: {image_path}. Please check the path.")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
            
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        features = self.dense_features[item, :]
        
        
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "features": torch.tensor(features, dtype=torch.float),
        }


class targetModel(nn.Module):
    def __init__(self, model_name="resnet50", dense_dim=12, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=3)
        n_features = self.model.get_classifier().in_features
        self.model.reset_classifier(0)
        self.out = nn.Linear(n_features + dense_dim, 1)

    def forward(self, image, features, targets=None):
        image_features = self.model(image)
        x = torch.cat([image_features, features], dim=1)
        output = self.out(x)
        return output, 0, {}


test_aug = albumentations.Compose([
    albumentations.LongestMaxSize(args.image_size, p=1),
    albumentations.PadIfNeeded(args.image_size, args.image_size, p=1, border_mode=cv2.BORDER_CONSTANT),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
])


df_train = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv")
df_test = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv")


categorical_features = ['sex', 'anatom_site_general_challenge']
df_train = pd.get_dummies(df_train, columns=categorical_features, dummy_na=False)
df_test = pd.get_dummies(df_test, columns=categorical_features, dummy_na=False)


one_hot_cols = [col for col in df_train.columns if any(f"_{cat}" in col for cat in categorical_features)]
numerical_features = ['age_approx']
dense_features_list = numerical_features + one_hot_cols


train_cols = set(df_train.columns)
for col in dense_features_list:
    if col not in df_test.columns:
        df_test[col] = 0
df_test = df_test[df_train.columns.intersection(df_test.columns)]
print(f"Final dense features being used ({len(dense_features_list)}): {dense_features_list}")


print(f"\n--- Getting predictions for the best model: {BEST_MODEL_NAME.upper()} ---")

model = targetModel(model_name=BEST_MODEL_NAME, dense_dim=len(dense_features_list), pretrained=False)
model = Tez(model)


model_path = f"/kaggle/input/ajai-sharan-velmurugan-23bcs10154-training-siim/model_inception_v3_f0.bin"
model.load(model_path, weights_only=True)


test_img_paths = [f"{BASE_IMAGE_PATH}{x}.jpg" for x in df_test["image_name"].values]
test_dataset = targetDataset(
    image_paths=test_img_paths,
    dense_features=df_test[dense_features_list].values,
    targets=np.ones(len(test_img_paths)),
    augmentations=test_aug,
)


predictions_generator = model.predict(test_dataset, batch_size=2*args.batch_size, n_jobs=-1)


final_predictions_logits = []
for preds in tqdm(predictions_generator):
    final_predictions_logits.extend(preds.ravel().tolist())


print("\nCreating submission file from best model...")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

final_predictions_logits = np.array(final_predictions_logits)
final_predictions_prob = sigmoid(final_predictions_logits)

df_test["target"] = final_predictions_prob
submission_df = df_test[["image_name", "target"]]
submission_df.to_csv("submission.csv", index=False)

print("Submission file created successfully!")
submission_df.head()




