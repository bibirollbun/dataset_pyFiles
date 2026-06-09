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


df = pd.read_csv('/kaggle/input/landmark/train.csv', header=None, names=["id", "landmark_id"])
print(df)


import pandas as pd

# File paths
train_csv_path = "/kaggle/input/landmark/train.csv"
fixed_csv_path = "/kaggle/input/id-2-names-landmark/ID_to_names/train_with_landmark_names_fixed.csv"

# Load datasets
train_df = pd.read_csv(train_csv_path, header=None, names=["id", "landmark_id"],encoding = "latin1")
fixed_df = pd.read_csv(fixed_csv_path)

# Convert landmark_id to string for consistency
train_df["landmark_id"] = train_df["landmark_id"].astype(str)
fixed_df["landmark_id"] = fixed_df["landmark_id"].astype(str)


import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv('/kaggle/input/landmark/train.csv', header=None, names=["id", "landmark_id"])

# Look up the landmark_id for the given image ID
image_id = '520a0aeb2d338a1c'
row = df[df['id'] == image_id]

if not row.empty:
    landmark_id = row.iloc[0]['landmark_id']
    print(f"Image ID: {image_id} has landmark ID: {landmark_id}")
    
    # Build image path
    path = f"/kaggle/input/landmark-recognition-2021/train/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg"

    # Load and display the image
    if os.path.exists(path):
        img = Image.open(path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Landmark ID: {landmark_id}")
        plt.show()
    else:
        print(f"Image file not found at {path}")
else:
    print(f"Image ID {image_id} not found in CSV.")



import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("/kaggle/input/landmark/train.csv", header=0)  # contains columns: id, landmark_id

# Filter for desired landmark ID
landmark_id = 138982
filtered_df = df[df['landmark_id'] == landmark_id]

print(f"Found {len(filtered_df)} images for landmark ID {landmark_id}")

# Plot first 5 images
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, (idx, row) in enumerate(filtered_df.head(5).iterrows()):
    image_id = row['id']
    # Extract subfolders based on first 3 characters
    subfolder = f"{image_id[0]}/{image_id[1]}/{image_id[2]}"
    image_path = f"/kaggle/input/landmark-recognition-2021/train/{subfolder}/{image_id}.jpg"

    try:
        img = Image.open(image_path)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(image_id)
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        axes[i].axis('off')
        axes[i].set_title("Missing")

plt.tight_layout()
plt.show()


import pandas as pd
import os

# Load the CSV
df = pd.read_csv('/kaggle/input/landmark-recognition-2021/train.csv')

landmark_29_df = df[df['landmark_id'] == 20102]

# Create output directory if it doesn't exist
output_dir = '/kaggle/working/add_images'
os.makedirs(output_dir, exist_ok=True)

# Save to CSV
output_path = os.path.join(output_dir, 'landmark_20102_images.csv')
landmark_29_df.to_csv(output_path, index=False)

print(f"âœ… Saved {len(landmark_29_df)} entries to {output_path}")


df = pd.read_csv('/kaggle/input/landmark-recognition-2021/train.csv')
landmark_counts = df['landmark_id'].value_counts()
top_100_to_200 = landmark_counts.iloc[100:110]
print(top_100_to_200)


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
sample = test_df.sample(90).iloc[0]
img_id = sample["id"]
true_label = sample["landmark_id"]
class_idx = sample["class_idx"]

folder = os.path.join(DATA_DIR, img_id[0], img_id[1], img_id[2])
img_path = os.path.join(folder, f"{img_id}.jpg")

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

    return f"Predicted class: {pred}"

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


import gradio as gr
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained classification models
model_dict = {
    "resnet18": tv_models.resnet18(pretrained=True).to(device),
    "resnet50": tv_models.resnet50(pretrained=True).to(device),
    "efficientnet_b0": tv_models.efficientnet_b0(pretrained=True).to(device),
    "efficientnet_b3": tv_models.efficientnet_b3(pretrained=True).to(device)
}

# Transforms
transform_efficientnet = A.Compose([
    A.Resize(260, 260),
    A.Normalize(),
    ToTensorV2()
])

transform_resnet = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

def predict(img: Image.Image, model_name: str):
    img_np = np.array(img)

    # Step 1: Binary classification
    binary_transformed = transform_mobilenet(image=img_np)
    binary_tensor = binary_transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        binary_output = mobilenet_binary(binary_tensor)
        binary_confidence = torch.sigmoid(binary_output).item()

    if binary_confidence <= 0.20:
        return f"The image does not appear to contain a landmark. (Binary confidence: {binary_confidence:.2f})"

    # Step 2: Main classification
    transform = transform_efficientnet if model_name.startswith("efficientnet") else transform_resnet
    transformed = transform(image=img_np)
    img_tensor = transformed["image"].to(device)

    model = model_dict[model_name]
    model.eval()

    # Step 3: ODIN Score
    temperature = 10  # Use your validated hyperparameter
    epsilon = 0.0014    # Use your validated hyperparameter

    odin_confidence = odin_score(img_tensor, model, temperature, epsilon)
    odin_threshold = 0.022769  # Youden's J

    if odin_confidence < odin_threshold:
        return (
            f"The image might contain a landmark. (Binary confidence: {binary_confidence:.2f})\n"
            f"âš ï¸� Rejected by ODIN (score: {odin_confidence:.4f} < threshold: {odin_threshold})"
        )

    # Step 4: Final classification
    img_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, pred_class = torch.max(probabilities, dim=1)
        confidence = confidence.item()
        pred_class = pred_class.item()

    return (
        f"The image might contain a landmark. (Binary confidence: {binary_confidence:.2f})\n"
        f"ODIN score: {odin_confidence:.4f} (above threshold: {odin_threshold})\n"
        f"Predicted class: {pred_class} (Confidence: {confidence:.2f})"
    )


# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil"),
        gr.Dropdown(choices=list(model_dict.keys()), label="Choose Model")
    ],
    outputs="text",
    title="Landmark Classification Demo (Threshold-based Detection)"
)

interface.launch()



import gradio as gr
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load 4 pretrained classification models
model_dict = {
    "resnet18": tv_models.resnet18(pretrained=True).to(device),
    "resnet50": tv_models.resnet50(pretrained=True).to(device),
    "efficientnet_b0": tv_models.efficientnet_b0(pretrained=True).to(device),
    "efficientnet_b3": tv_models.efficientnet_b3(pretrained=True).to(device)
}

# Load the binary MobileNet model (output=1)
mobilenet_binary = tv_models.mobilenet_v2(pretrained=False)
mobilenet_binary.classifier[1] = nn.Linear(mobilenet_binary.last_channel, 1)
mobilenet_binary.load_state_dict(torch.load("/kaggle/input/mobilenet-landmark-binary-classification/mobilenet_landmark (1).pth", map_location=device))
mobilenet_binary.to(device)
mobilenet_binary.eval()

# Transforms
transform_efficientnet = A.Compose([
    A.Resize(260, 260),
    A.Normalize(),
    ToTensorV2()
])

transform_resnet = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

transform_mobilenet = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

def odin_score(image_tensor, model, temperature, epsilon):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    image_tensor.requires_grad = True  # Needed for gradient computation

    model.eval()  # Ensure in eval mode

    # Forward pass
    logits = model(image_tensor)
    logits = logits / temperature
    pred_class = logits.argmax(dim=1)

    # Compute loss
    loss = F.cross_entropy(logits, pred_class)
    model.zero_grad()
    loss.backward()

    # Perturbation
    gradient = torch.sign(image_tensor.grad.data)
    perturbed = image_tensor - epsilon * gradient
    perturbed = torch.clamp(perturbed, 0, 1)

    # Forward with perturbed input
    with torch.no_grad():
        logits_perturbed = model(perturbed) / temperature
        softmax_scores = F.softmax(logits_perturbed, dim=1)
        score = torch.max(softmax_scores).item()

    return score

def predict(img: Image.Image, model_name: str):
    img_np = np.array(img)

    # Step 1: Binary classification
    binary_transformed = transform_mobilenet(image=img_np)
    binary_tensor = binary_transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        binary_output = mobilenet_binary(binary_tensor)
        binary_confidence = torch.sigmoid(binary_output).item()

    if binary_confidence <= 0.20:
        return f"The image does not appear to contain a landmark. (Binary confidence: {binary_confidence:.2f})"

    # Step 2: Main classification
    transform = transform_efficientnet if model_name.startswith("efficientnet") else transform_resnet
    transformed = transform(image=img_np)
    img_tensor = transformed["image"].to(device)

    model = model_dict[model_name]
    model.eval()

    # Step 3: ODIN Score
    temperature = 10  # Use your validated hyperparameter
    epsilon = 0.0014    # Use your validated hyperparameter

    odin_confidence = odin_score(img_tensor, model, temperature, epsilon)
    odin_threshold = 0.022769  # Youden's J

    if odin_confidence < odin_threshold:
        return (
            f"The image might contain a landmark. (Binary confidence: {binary_confidence:.2f})\n"
            f"âš ï¸� Rejected by ODIN (score: {odin_confidence:.4f} < threshold: {odin_threshold})"
        )

    # Step 4: Final classification
    img_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, pred_class = torch.max(probabilities, dim=1)
        confidence = confidence.item()
        pred_class = pred_class.item()

    return (
        f"The image might contain a landmark. (Binary confidence: {binary_confidence:.2f})\n"
        f"ODIN score: {odin_confidence:.4f} (above threshold: {odin_threshold})\n"
        f"Predicted class: {pred_class} (Confidence: {confidence:.2f})"
    )

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil"),
        gr.Dropdown(choices=list(model_dict.keys()), label="Choose Model")
    ],
    outputs="text",
    title="Landmark Classification Demo (with Binary Landmark Detection)"
)

interface.launch()



import gradio as gr
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load 4 pretrained classification models
model_dict = {
    "resnet18": tv_models.resnet18(pretrained=True).to(device),
    "resnet50": tv_models.resnet50(pretrained=True).to(device),
    "efficientnet_b0": tv_models.efficientnet_b0(pretrained=True).to(device),
    "efficientnet_b3": tv_models.efficientnet_b3(pretrained=True).to(device)
}

# Load the binary MobileNet model (output=1)
mobilenet_binary = tv_models.mobilenet_v2(pretrained=False)
mobilenet_binary.classifier[1] = nn.Linear(mobilenet_binary.last_channel, 1)
mobilenet_binary.load_state_dict(torch.load("/kaggle/input/mobilenet-landmark-binary-classification/mobilenet_landmark (1).pth", map_location=device))
mobilenet_binary.to(device)
mobilenet_binary.eval()

# Transforms
transform_efficientnet = A.Compose([
    A.Resize(260, 260),
    A.Normalize(),
    ToTensorV2()
])

transform_resnet = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

transform_mobilenet = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

def predict(img: Image.Image, model_name: str):
    img_np = np.array(img)

    # Step 1: Run binary classifier
    binary_transformed = transform_mobilenet(image=img_np)
    binary_tensor = binary_transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        binary_output = mobilenet_binary(binary_tensor)
        prob = torch.sigmoid(binary_output).item()
        is_landmark = prob >= 0.5  # You can lower this threshold if needed

    if not is_landmark:
        return f"The image given does not contain a landmark. (Confidence: {prob:.2f})"

    # Step 2: Run selected model
    transform = transform_efficientnet if model_name.startswith("efficientnet") else transform_resnet
    transformed = transform(image=img_np)
    img_tensor = transformed["image"].unsqueeze(0).to(device)

    model = model_dict[model_name]
    model.eval()

    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()

    return f"The image might contain a landmark. (Confidence: {prob:.2f})\nPredicted class: {pred}"

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil"),
        gr.Dropdown(choices=list(model_dict.keys()), label="Choose Model")
    ],
    outputs="text",
    title="Landmark Classification Demo (with Binary Landmark Detection)"
)

interface.launch()



import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, local_feat, global_feat):
        b, c, h, w = local_feat.shape
        global_feat_expanded = global_feat.view(b, c, 1, 1).expand(-1, -1, h, w)
        attn = self.attn(local_feat + global_feat_expanded)
        fused = local_feat * attn + global_feat_expanded * (1 - attn)
        return fused


class DOLG_ArcFace(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        resnet = models.resnet50(pretrained=True)

        # Shared layers
        self.backbone_common = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool, resnet.layer1,
            resnet.layer2, resnet.layer3
        )

        # ResNet layer4 expects input with 1024 channels (not from local_conv!)
        self.backbone_global = resnet.layer4

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_fc = nn.Linear(2048, embedding_dim)

        self.local_conv = nn.Conv2d(1024, embedding_dim, kernel_size=1)

        self.fusion = AttentionFusion(embedding_dim)

        self.head = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        shared_feat = self.backbone_common(x)  # Output: [B, 1024, H, W]

        # Global branch
        global_feat_map = self.backbone_global(shared_feat)  # Output: [B, 2048, H/2, W/2]
        global_feat = self.global_pool(global_feat_map).view(x.size(0), -1)  # [B, 2048]
        global_feat = self.global_fc(global_feat)  # [B, 512]

        # Local branch
        local_feat = self.local_conv(shared_feat)  # [B, 512, H, W]

        # Fuse
        fused_feat = self.fusion(local_feat, global_feat)  # [B, 512, H, W]
        emb = self.head(fused_feat)  # [B, 512]
        return emb


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, input, label):
        # normalize inputs and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # [B, C]

        # compute cos(Î¸ + m)
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        phi = torch.cos(theta + self.m)

        one_hot = F.one_hot(label, num_classes=cosine.size(1)).float().to(input.device)
        logits = cosine * (1 - one_hot) + phi * one_hot
        return logits * self.s



import gradio as gr
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
#from model import DOLG_ArcFace, ArcFace  # Make sure this file exists

# ============ Device ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ Pretrained Classification Models ============
model_dict = {
    "resnet18": tv_models.resnet18(pretrained=True).to(device),
    "resnet50": tv_models.resnet50(pretrained=True).to(device),
    "efficientnet_b0": tv_models.efficientnet_b0(pretrained=True).to(device),
    "efficientnet_b3": tv_models.efficientnet_b3(pretrained=True).to(device)
}

# ============ Add DOLG Model ============
embedding_dim = 512
num_classes = 102  # set this according to your dataset

dolg_model = DOLG_ArcFace(embedding_dim=embedding_dim).to(device)
arcface_head = ArcFace(in_features=embedding_dim, out_features=num_classes).to(device)

# Load weights
checkpoint = torch.load("/kaggle/input/resnet-dolg/kaggle/working/best_model.pth", map_location=device)
dolg_model.load_state_dict(checkpoint["model_state_dict"])
arcface_head.load_state_dict(checkpoint["arcface_state_dict"])
dolg_model.eval()
arcface_head.eval()

model_dict["resnet_dolg"] = (dolg_model, arcface_head)  # special handling for this one

# ============ Load Binary MobileNet ============
mobilenet_binary = tv_models.mobilenet_v2(pretrained=False)
mobilenet_binary.classifier[1] = nn.Linear(mobilenet_binary.last_channel, 1)
mobilenet_binary.load_state_dict(torch.load("/kaggle/input/mobilenet-landmark-binary-classification/mobilenet_landmark (1).pth", map_location=device))
mobilenet_binary.to(device)
mobilenet_binary.eval()

# ============ Transforms ============
transform_efficientnet = A.Compose([
    A.Resize(260, 260),
    A.Normalize(),
    ToTensorV2()
])

transform_resnet = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

transform_mobilenet = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

# ============ Inference Function ============
def predict(img: Image.Image, model_name: str):
    img_np = np.array(img)

    # Step 1: Run binary classifier
    binary_transformed = transform_mobilenet(image=img_np)
    binary_tensor = binary_transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        binary_output = mobilenet_binary(binary_tensor)
        prob = torch.sigmoid(binary_output).item()
        is_landmark = prob >= 0.5

    if not is_landmark:
        return f"The image does not contain a landmark. (Confidence: {prob:.2f})"

    # Step 2: Classify landmark
    transform = transform_efficientnet if model_name.startswith("efficientnet") else transform_resnet
    transformed = transform(image=img_np)
    img_tensor = transformed["image"].unsqueeze(0).to(device)

    # Handle ResNet + DOLG differently
    if model_name == "resnet_dolg":
        dolg_model, arcface_head = model_dict["resnet_dolg"]
        with torch.no_grad():
            embedding = dolg_model(img_tensor)
            logits = arcface_head(embedding, torch.zeros(1, dtype=torch.long).to(device))  # dummy label
            prob_class = F.softmax(logits, dim=1)
            conf, pred = torch.max(prob_class, dim=1)
        return f"Landmark detected (Confidence: {prob:.2f})\nPredicted class (ResNet+DOLG): {pred.item()} (Conf: {conf.item():.2f})"
    
    # Standard models
    model = model_dict[model_name]
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()

    return f"Landmark detected (Confidence: {prob:.2f})\nPredicted class ({model_name}): {pred}"

# ============ Gradio Interface ============
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil"),
        gr.Dropdown(choices=list(model_dict.keys()), label="Choose Model")
    ],
    outputs="text",
    title="Landmark Classification Demo (ResNet + DOLG + ArcFace Integrated)"
)

interface.launch()


import gradio as gr
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load classification models
model_dict = {
    "resnet18": tv_models.resnet18(pretrained=True).to(device),
    "resnet50": tv_models.resnet50(pretrained=True).to(device),
    "efficientnet_b0": tv_models.efficientnet_b0(pretrained=True).to(device),
    "efficientnet_b3": tv_models.efficientnet_b3(pretrained=True).to(device)
}

# Load binary classifier (MobileNet)
mobilenet_binary = tv_models.mobilenet_v2(pretrained=False)
mobilenet_binary.classifier[1] = nn.Linear(mobilenet_binary.last_channel, 1)
mobilenet_binary.load_state_dict(torch.load("/kaggle/input/mobilenet-landmark-binary-classification/mobilenet_landmark (1).pth", map_location=device))
mobilenet_binary.to(device)
mobilenet_binary.eval()

# Transforms
transform_efficientnet = A.Compose([
    A.Resize(260, 260),
    A.Normalize(),
    ToTensorV2()
])

transform_resnet = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

transform_mobilenet = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

def predict(img: Image.Image, model_name: str):
    img_np = np.array(img)

    # Step 1: Binary classification
    binary_transformed = transform_mobilenet(image=img_np)
    binary_tensor = binary_transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        binary_output = mobilenet_binary(binary_tensor)
        binary_confidence = torch.sigmoid(binary_output).item()

    if binary_confidence <= 0.20:
        return f"The image does not appear to contain a landmark. (Binary confidence: {binary_confidence:.2f})"

    # Step 2: Main classification
    transform = transform_efficientnet if model_name.startswith("efficientnet") else transform_resnet
    transformed = transform(image=img_np)
    img_tensor = transformed["image"].unsqueeze(0).to(device)

    model = model_dict[model_name]
    model.eval()

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, pred_class = torch.max(probabilities, dim=1)
        confidence = confidence.item()
        pred_class = pred_class.item()

    return (
        f"The image might contain a landmark. (Binary confidence: {binary_confidence:.2f})\n"
        f"Predicted class: {pred_class} (Confidence: {confidence:.2f})"
    )

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil"),
        gr.Dropdown(choices=list(model_dict.keys()), label="Choose Model")
    ],
    outputs="text",
    title="Landmark Classification with Binary Filter (Threshold 0.20)"
)

interface.launch()



import gradio as gr
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tv_models
import random
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load classification models
model_dict = {
    "resnet18": tv_models.resnet18(pretrained=True).to(device),
    "resnet50": tv_models.resnet50(pretrained=True).to(device),
    "efficientnet_b0": tv_models.efficientnet_b0(pretrained=True).to(device),
    "efficientnet_b3": tv_models.efficientnet_b3(pretrained=True).to(device)
}

# Load binary classifier (MobileNet)
mobilenet_binary = tv_models.mobilenet_v2(pretrained=False)
mobilenet_binary.classifier[1] = nn.Linear(mobilenet_binary.last_channel, 1)
mobilenet_binary.load_state_dict(torch.load("/kaggle/input/mobilenet-landmark-binary-classification/mobilenet_landmark (1).pth", map_location=device))
mobilenet_binary.to(device)
mobilenet_binary.eval()

# Transforms
transform_efficientnet = A.Compose([
    A.Resize(260, 260),
    A.Normalize(),
    ToTensorV2()
])

transform_resnet = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

transform_mobilenet = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

# Load CSV for "I'm feeling lucky"
df = pd.read_csv("/kaggle/input/id-2-names-landmark/ID_to_names/train_with_landmark_names_fixed.csv", encoding="latin1")

# Drop rows with missing ID or parse errors
df = df.dropna(subset=['id'])

# You may want to adjust the path below to point to actual images
IMAGE_BASE_PATH = "/kaggle/input/landmark-recognition-2021/train"  # Change if needed

def predict(img: Image.Image, model_name: str):
    img_np = np.array(img)

    # Step 1: Binary classification
    binary_transformed = transform_mobilenet(image=img_np)
    binary_tensor = binary_transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        binary_output = mobilenet_binary(binary_tensor)
        binary_confidence = torch.sigmoid(binary_output).item()

    if binary_confidence <= 0.20:
        return f"The image does not appear to contain a landmark. (Binary confidence: {binary_confidence:.2f})"

    # Step 2: Main classification
    transform = transform_efficientnet if model_name.startswith("efficientnet") else transform_resnet
    transformed = transform(image=img_np)
    img_tensor = transformed["image"].to(device)

    model = model_dict[model_name]
    model.eval()

    # Step 3: ODIN Score
    temperature = 10  # Use your validated hyperparameter
    epsilon = 0.0014    # Use your validated hyperparameter

    odin_confidence = odin_score(img_tensor, model, temperature, epsilon)
    odin_threshold = 0.022769  # Youden's J

    if odin_confidence < odin_threshold:
        return (
            f"The image might contain a landmark. (Binary confidence: {binary_confidence:.2f})\n"
            f"âš ï¸� Rejected by ODIN (score: {odin_confidence:.4f} < threshold: {odin_threshold})"
        )

    # Step 4: Final classification
    img_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, pred_class = torch.max(probabilities, dim=1)
        confidence = confidence.item()
        pred_class = pred_class.item()

    return (
        f"The image might contain a landmark. (Binary confidence: {binary_confidence:.2f})\n"
        f"ODIN score: {odin_confidence:.4f} (above threshold: {odin_threshold})\n"
        f"Predicted class: {pred_class} (Confidence: {confidence:.2f})"
    )


def build_image_path_from_id(img_id: str):
    # Example: '1420523fe073af12' => /kaggle/input/landmark-recognition-2021/train/1/4/2/1420523fe073af12.jpg
    return f"/kaggle/input/landmark-recognition-2021/train/{img_id[0]}/{img_id[1]}/{img_id[2]}/{img_id}.jpg"

def lucky_guess(model_name):
    random_row = df.sample(1).iloc[0]
    img_id = random_row['id']
    landmark_name = random_row.get("landmark_name", "Unknown Landmark")

    image_path = build_image_path_from_id(img_id)

    if not os.path.exists(image_path):
        return None, f"Image not found: {image_path}"

    img = Image.open(image_path).convert("RGB")
    prediction_text = predict(img, model_name)
    
    # Add landmark name from CSV if available
    prediction_text += f"\nLandmark name (from CSV): {landmark_name}"
    return img, prediction_text



# Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("# Landmark Classification with Binary Filter (Threshold 0.20)")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        model_dropdown = gr.Dropdown(choices=list(model_dict.keys()), label="Choose Model", value="resnet18")

    output_text = gr.Textbox(label="Result")

    with gr.Row():
        predict_btn = gr.Button("Predict")
        lucky_btn = gr.Button("I'm feeling lucky ğŸ�²")

    lucky_image_output = gr.Image(label="Lucky Image")

    predict_btn.click(fn=predict, inputs=[image_input, model_dropdown], outputs=output_text)
    lucky_btn.click(fn=lucky_guess, inputs=[model_dropdown], outputs=[lucky_image_output, output_text])

interface.launch()


from PIL import Image

img_path = "/kaggle/input/landmark-recognition-2021/train/1/6/b/16bbacee81a4230f.jpg"
image = Image.open(img_path)

# To display it in the notebook:
image.show()
# or in Jupyter-compatible notebooks:
display(image)

