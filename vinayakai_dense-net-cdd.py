import timm

model = timm.create_model("hf_hub:timm/densenet169.tv_in1k", pretrained=True)


import pandas as pd
d=pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/train.csv')
print(d.head(25))


# ====================================================
# Cassava Leaf Disease Detection using DenseNet169
# 90:10 Train-Validation Split
# ====================================================

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

# ====================================================
# Step 1: Load Dataset
# ====================================================

# Load CSV
df = pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/train.csv')
df['filepath'] = df['image_id'].apply(
    lambda x: os.path.join('/kaggle/input/cassava-leaf-disease-classification/train_images', x)
)

# Train-Validation Split (90:10)
train_df, val_df = train_test_split(
    df, test_size=0.1, stratify=df['label'], random_state=42
)
print("Train samples:", len(train_df), "Validation samples:", len(val_df))

# ====================================================
# Step 2: Data Transforms & Dataset Class
# ====================================================

IMG_SIZE = 224

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class CassavaDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        path = self.df.loc[idx, 'filepath']
        label = self.df.loc[idx, 'label']
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Dataloaders
train_ds = CassavaDataset(train_df, train_transform)
val_ds = CassavaDataset(val_df, val_transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

# ====================================================
# Step 3: Define Model (DenseNet169)
# ====================================================

NUM_CLASSES = df['label'].nunique()

model = timm.create_model("densenet169", pretrained=True)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, NUM_CLASSES)
)

# ====================================================
# Step 4: Setup Training
# ====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# ====================================================
# Step 5: Training Loop with Checkpointing
# ====================================================

EPOCHS = 10
best_acc = 0.0

for epoch in range(EPOCHS):
    # ---- Training ----
    model.train()
    train_loss, correct, total = 0, 0, 0
    
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    train_acc = correct / total
    train_loss /= total

    # ---- Validation ----
    model.eval()
    val_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    val_acc = correct / total
    val_loss /= total
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    scheduler.step()
    
    # ---- Save Best Model ----
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_densenet169.pth")
        print("âœ… Saved Best Model with Val Acc:", best_acc)

# ====================================================
# Step 6: Load Best Model (for inference later)
# ====================================================

model.load_state_dict(torch.load("best_densenet169.pth"))
model.eval()
print("Loaded best model with validation accuracy:", best_acc)



# ====================================================
# Resume training if notebook froze
# ====================================================

# Load previous best checkpoint
model.load_state_dict(torch.load("best_densenet169.pth"))
model = model.to(device)

# Continue from epoch 7 onward
RESUME_EPOCH = 7
EXTRA_EPOCHS = 10  # train 10 more epochs

for epoch in range(RESUME_EPOCH, RESUME_EPOCH + EXTRA_EPOCHS):
    # ---- Training ----
    model.train()
    train_loss, correct, total = 0, 0, 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{RESUME_EPOCH+EXTRA_EPOCHS} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix({"Loss": f"{train_loss/total:.4f}", "Acc": f"{correct/total:.4f}"})

    train_acc = correct / total
    train_loss /= total

    # ---- Validation ----
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{RESUME_EPOCH+EXTRA_EPOCHS} [Val]"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    val_loss /= total

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    scheduler.step()

    # ---- Save Best Model ----
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_densenet169.pth")
        print("âœ… Saved New Best Model with Val Acc:", best_acc)



import timm
import torch
import torch.nn as nn

def create_densenet169_custom(num_classes=5):
    model = timm.create_model("hf_hub:timm/densenet169.tv_in1k", pretrained=False)
    in_features = model.get_classifier().in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

# Rebuild model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_densenet169_custom(num_classes=5)
model.load_state_dict(torch.load("best_densenet169.pth", map_location=device))
model = model.to(device)
model.eval()





# Evaluation on validation set
from tqdm import tqdm

model.eval()
val_loss, correct, total = 0, 0, 0

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        val_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

val_acc = correct / total
val_loss /= total
print(f"âœ… Final Model | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")



from PIL import Image
from torchvision import transforms

# Same preprocessing as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(path):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(img)
        _, predicted = outputs.max(1)
    return predicted.item()

print("Predicted class:", predict_image("/kaggle/input/cassava-leaf-disease-classification/test_images/2216849948.jpg"))



!pip install grad-cam



# ============================================
# Cassava Dataset Custom Loader
# ============================================
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ----------------------------
# CONFIG
# ----------------------------
csv_file = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
img_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# DATA
# ----------------------------
df = pd.read_csv(csv_file)
label_map = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy"
}

val_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

class CassavaDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = f"{self.img_dir}/{self.df.iloc[idx, 0]}"
        image = Image.open(img_path).convert("RGB")
        label = self.df.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

# Full dataset (you can split into val/test if needed)
val_dataset = CassavaDataset(df, img_dir, transform=val_tfms)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Loaded validation dataset:", len(val_dataset), "images")



import timm
import torch.nn as nn

# Base model
model = timm.create_model("densenet169", pretrained=False)

# Rebuild classifier as it was during training
# (looks like you had a small MLP head instead of single fc)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 5)   # 5 cassava classes
)

# Load weights
model.load_state_dict(torch.load("/kaggle/working/best_densenet169.pth", map_location=device))

model.to(device)
model.eval()

print("Model loaded successfully!")




import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# === 1. Load CSV (image_id -> label mapping) ===
csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
img_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"

df = pd.read_csv(csv_path)
class_names = ['Cassava Bacterial Blight (CBB)',
               'Cassava Brown Streak Disease (CBSD)',
               'Cassava Green Mottle (CGM)',
               'Cassava Mosaic Disease (CMD)',
               'Healthy']

# === 2. Custom Dataset ===
class CassavaDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = f"{self.img_dir}/{self.dataframe.iloc[idx]['image_id']}"
        image = Image.open(img_path).convert("RGB")
        label = self.dataframe.iloc[idx]['label']
        
        if self.transform:
            image = self.transform(image)
        return image, label

# === 3. Validation transforms ===
val_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# === 4. Create dataset & dataloader ===
val_dataset = CassavaDataset(df, img_dir, transform=val_tfms)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)



import torch
import timm
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load base DenseNet169
model = timm.create_model("densenet169", pretrained=False, num_classes=0)  # remove default head

# Add the same custom classifier used during training
model.classifier = nn.Sequential(
    nn.Linear(1664, 512),   # bottleneck layer
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 5)       # final output layer
)

# Load your trained weights
state_dict = torch.load("/kaggle/working/best_densenet169.pth", map_location=device)
model.load_state_dict(state_dict, strict=True)

model.to(device)
model.eval()



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import timm

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CSV and Image Directory ---
csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
img_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
df = pd.read_csv(csv_path)

# --- Custom Dataset ---
class CassavaDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = f"{self.img_dir}/{self.dataframe.iloc[idx]['image_id']}"
        image = Image.open(img_path).convert("RGB")
        label = int(self.dataframe.iloc[idx]['label'])
        if self.transform:
            image = self.transform(image)
        return image, label

# --- Validation Transforms ---
val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Dataset and DataLoader ---
val_dataset = CassavaDataset(df, img_dir, transform=val_tfms)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# --- Model with same classifier as training ---
model = timm.create_model("densenet169", pretrained=False)
# Define the same custom classifier as used in training
model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 5)  # 5 classes for cassava disease
)

# Load the checkpoint
model.load_state_dict(torch.load("/kaggle/working/best_densenet169.pth", map_location=device))
model = model.to(device)
model.eval()

# --- Validation Loop ---
all_labels = []
all_preds = []
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

val_acc = correct / total
print(f"Validation Accuracy: {val_acc:.4f}")

# --- Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=[0,1,2,3,4])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()



val_loss /= total
val_acc = correct / total

print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")



import matplotlib.pyplot as plt

# Example lists, replace these with your actual recorded values
val_losses = [0.5290, 0.5399, 0.4494, 0.4486, 0.4270, 0.4315, 0.4156, 0.4125, 0.4132, 0.4083]
val_accs   = [0.8051, 0.8042, 0.8379, 0.8439, 0.8514, 0.8500, 0.8607, 0.8636, 0.8612, 0.8598]

epochs = range(1, len(val_losses)+1)

plt.figure(figsize=(10,5))

# Validation Loss
plt.subplot(1,2,1)
plt.plot(epochs, val_losses, marker='o', color='red')
plt.title("Validation Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

# Validation Accuracy
plt.subplot(1,2,2)
plt.plot



import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# --- Load your trained model ---
import timm
model = timm.create_model("densenet169", pretrained=False, num_classes=5)
model.load_state_dict(torch.load("/kaggle/working/best_densenet169.pth", map_location="cuda"))
model = model.to("cuda")
model.eval()

# --- Define Grad-CAM ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None):
        output = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(output, 1).item()
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()
        
        weights = torch.mean(self.gradients, dim=(2,3), keepdim=True)
        grad_cam_map = torch.sum(weights * self.activations, dim=1)[0]
        grad_cam_map = F.relu(grad_cam_map)
        grad_cam_map = grad_cam_map - grad_cam_map.min()
        grad_cam_map = grad_cam_map / grad_cam_map.max()
        grad_cam_map = grad_cam_map.cpu().numpy()
        grad_cam_map = cv2.resize(grad_cam_map, (x.size(3), x.size(2)))
        return grad_cam_map

# --- Preprocess an image ---
img_path = "/kaggle/input/cassava-leaf-disease-classification/train_images/1000015157.jpg"
img = Image.open(img_path).convert("RGB")
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
input_tensor = preprocess(img).unsqueeze(0).to("cuda")

# --- Apply Grad-CAM ---
target_layer = model.features[-1]  # Last convolutional layer
grad_cam = GradCAM(model, target_layer)
mask = grad_cam(input_tensor)

# --- Overlay heatmap ---
img_np = np.array(img.resize((224,224)))
heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

plt.figure(figsize=(8,8))
plt.imshow(overlay)
plt.axis('off')
plt.title("Grad-CAM")
plt.show()


