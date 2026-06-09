import pandas as pd

# Load your CSVs
# train_df = pd.read_csv('/kaggle/working/train.csv')
# test_df = pd.read_csv('/kaggle/working/test.csv')
# val_df = pd.read_csv('/kaggle/working/val.csv')
train_df = pd.read_csv('/kaggle/input/landmark/train.csv')
test_df = pd.read_csv('/kaggle/input/landmark/test.csv')
val_df = pd.read_csv('/kaggle/input/landmark/val.csv')
# Build mapping
landmark_id_to_idx = {lid: idx for idx, lid in enumerate(sorted(train_df['landmark_id'].unique()))}
NUM_CLASSES = len(landmark_id_to_idx) 

# Map class_idx
train_df['class_idx'] = train_df['landmark_id'].map(landmark_id_to_idx)
test_df['class_idx'] = test_df['landmark_id'].map(landmark_id_to_idx)
val_df['class_idx'] = val_df['landmark_id'].map(landmark_id_to_idx)

NUM_CLASSES = len(landmark_id_to_idx)


DATA_DIR = "/kaggle/input/landmark-recognition-2021/train/"


IMAGE_SIZE = 224
BATCH_SIZE = 32


from torchvision import transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
test_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(),
    ToTensorV2()
])


import torch
import torch.nn as nn
from torchvision import models
from torch.optim import Adam
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, NUM_CLASSES)
)
model = model.to(device)
checkpoint_path = "/kaggle/input/model-effiecient/best_model_resnet_ver_2-2.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)


import os
import numpy as np
from PIL import Image
sample = test_df.sample(90).iloc[0]
img_id = sample["id"]
true_label = sample["landmark_id"]
class_idx = sample["class_idx"]

folder = os.path.join(DATA_DIR, img_id[0], img_id[1], img_id[2])
img_path = os.path.join(folder, f"{img_id}.jpg")
print(img_path)
image = Image.open(img_path).convert("RGB")
image = test_transform(image=np.array(image))["image"].unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    output = model(image)
    pred_idx = output.argmax(dim=1).item()

# Reverse map
idx_to_landmark_id = {v: k for k, v in landmark_id_to_idx.items()}
pred_landmark = idx_to_landmark_id[pred_idx]

print(f"ğŸ–¼ï¸� Image ID: {img_id}")
print(f"âœ… Ground Truth Landmark ID: {true_label}")
print(f"ğŸ�¯ Predicted Landmark ID: {pred_landmark}")


pip install efficientnet_pytorch



import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet

NUM_CLASSES = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- RESNET VERSION 1 --------
resnet_v1 = models.resnet50(pretrained=False)
in_features = resnet_v1.fc.in_features
resnet_v1.fc = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, NUM_CLASSES)
)
resnet_v1.load_state_dict(torch.load("/kaggle/input/model-effiecient/best_model_resnet-5.pth", map_location=device))
resnet_v1 = resnet_v1.to(device)

# -------- RESNET VERSION 2 --------
resnet_v2 = models.resnet50(pretrained=False)
in_features = resnet_v2.fc.in_features
resnet_v2.fc = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, NUM_CLASSES)
)
resnet_v2.load_state_dict(torch.load("/kaggle/input/model-effiecient/best_model_resnet_ver_2-7.pth", map_location=device))
resnet_v2 = resnet_v2.to(device)


# -------- EFFICIENTNET VERSION 1 --------
efficientnet_v1 = EfficientNet.from_name('efficientnet-b2')
in_features = efficientnet_v1._fc.in_features
efficientnet_v1._fc = nn.Linear(in_features, NUM_CLASSES)
efficientnet_v1.load_state_dict(torch.load("/kaggle/input/model-effiecient/best_model_efficientnet-2.pth", map_location=device))
efficientnet_v1 = efficientnet_v1.to(device)

# -------- EFFICIENTNET VERSION 2 --------
efficientnet_v2 = EfficientNet.from_name('efficientnet-b2')
in_features = efficientnet_v2._fc.in_features
efficientnet_v2._fc = nn.Linear(in_features, NUM_CLASSES)
efficientnet_v2.load_state_dict(torch.load("/kaggle/input/model-effiecient/best_model_efficientnet_ver_2-2.pth", map_location=device))
efficientnet_v2 = efficientnet_v2.to(device)

# -------- MODEL DICTIONARY --------
models = {
    "resnet_v1": resnet_v1,
    "resnet_v2": resnet_v2,
    "efficientnet_v1": efficientnet_v1,
    "efficientnet_v2": efficientnet_v2
}



pip install gradio


import gradio as gr
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch

# Albumentations transform cho EfficientNet
transform_efficientnet = A.Compose([
    A.Resize(260, 260),
    A.Normalize(),  # mean/std máº·c Ä‘á»‹nh giá»‘ng ImageNet
    ToTensorV2()
])

# Albumentations transform cho ResNet
transform_resnet = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

def predict(img: Image.Image, model_name: str):
    # Chuyá»ƒn PIL Image -> numpy
    img = np.array(img)

    # Chá»�n transform theo model
    transform = transform_efficientnet if model_name.startswith("efficientnet") else transform_resnet

    # Apply transform
    transformed = transform(image=img)
    img_tensor = transformed["image"].unsqueeze(0).to(device)

    model = models[model_name]
    model.eval()

    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()
    idx_to_landmark_id = {v: k for k, v in landmark_id_to_idx.items()}
    pred_landmark = idx_to_landmark_id[pred]
    return f"Predicted class: {pred_landmark}"

interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil"),
        gr.Dropdown(choices=list(models.keys()), label="Choose Model")
    ],
    outputs="text",
    title="Landmark Classification Demo (4 Models with Albumentations)"
)

interface.launch()

