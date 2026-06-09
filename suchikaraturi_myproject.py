!pip install timm
!pip install git+https://github.com/jacobgil/pytorch-grad-cam.git

import pandas as pd
import os
from PIL import Image
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import timm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print("Setup Complete. All libraries are installed and imported.")


KAGGLE_INPUT_DIR = '/kaggle/input/aptos2019-blindness-detection'
CSV_PATH = os.path.join(KAGGLE_INPUT_DIR, 'train.csv')
SOURCE_IMAGES_FOLDER = os.path.join(KAGGLE_INPUT_DIR, 'train_images')

OUTPUT_FOLDER = '/kaggle/working/processed_npdr_data'

print("Step 1: Reading and filtering the CSV file...")
df = pd.read_csv(CSV_PATH)

df_filtered = df[df['diagnosis'].isin([1, 2, 3])].copy()
print(f"Found {len(df_filtered)} images for NPDR stages 1, 2, and 3.")

print("\nStep 2: Creating new directories for each class...")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
for label in ['1', '2', '3']:
    class_dir = os.path.join(OUTPUT_FOLDER, str(label))
    os.makedirs(class_dir, exist_ok=True)
    print(f"- Directory created: {class_dir}")

print("\nStep 3: Resizing and saving images to their new folders...")
for index, row in tqdm(df_filtered.iterrows(), total=df_filtered.shape[0], desc="Processing Images"):
    image_name = f"{row['id_code']}.png"
    label = str(row['diagnosis'])
    source_path = os.path.join(SOURCE_IMAGES_FOLDER, image_name)
    destination_path = os.path.join(OUTPUT_FOLDER, label, image_name)
    
    if os.path.exists(source_path):
        try:
            with Image.open(source_path) as img:
                # [cite_start]Resize the image to 224x224 pixels [cite: 59]
                img_resized = img.resize((224, 224))
                img_resized.save(destination_path)
        except Exception as e:
            print(f"\nCould not process {image_name}. Error: {e}")

print("\nData preparation is complete!")


data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("1. Loading dataset from the processed folder...")
full_dataset = datasets.ImageFolder(OUTPUT_FOLDER, transform=data_transforms)
print(f"Dataset loaded. Found {len(full_dataset)} images in {len(full_dataset.classes)} classes.")

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
print(f"\n2. Data split into {len(train_dataset)} training and {len(val_dataset)} validation images.")

print("\n3. Loading pre-trained Vision Transformer model (vit_base_patch16_224)...")
model = timm.create_model('vit_base_patch16_224', pretrained=True)
num_classes = 3
model.head = nn.Linear(model.head.in_features, num_classes)
print("Model modified for 3 classes.")


import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

import timm
from timm.data import create_transform, Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import CosineLRScheduler
from torch.cuda.amp import autocast, GradScaler

# ==== 1. Custom Dataset ====
class AptosDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.loc[idx, 'id_code'] + '.png'
        label = self.data.loc[idx, 'diagnosis']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# ==== 2. Paths & device ====
train_csv = '/kaggle/input/aptos2019-blindness-detection/train.csv'
train_img_dir = '/kaggle/input/aptos2019-blindness-detection/train_images'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== 3. Augmented Transforms ====
train_transform = create_transform(
    input_size=224,
    is_training=True,
    color_jitter=0.5,
    auto_augment='rand-m9-mstd0.5-inc1',
    interpolation='bicubic',
    re_prob=0.5,
    re_mode='pixel',
    re_count=1,
)

val_transform = create_transform(
    input_size=224,
    is_training=False,
    interpolation='bicubic',
)

# ==== 4. Prepare Dataset and Split ====
full_dataset = AptosDataset(train_csv, train_img_dir, transform=None)
num_train = int(0.8 * len(full_dataset))
num_val = len(full_dataset) - num_train
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [num_train, num_val])

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

# ==== 5. Stratified Weighted Sampler ====
train_labels = [train_dataset.dataset.data.loc[i, 'diagnosis'] for i in train_dataset.indices]
class_sample_counts = np.bincount(train_labels)
weights = 1. / class_sample_counts
sample_weights = [weights[t] for t in train_labels]

sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# ==== 6. Grid Search Params ====
batch_sizes = [16, 32]
learning_rates = [3e-5, 5e-5, 1e-4]  # finer search with smaller LR
NUM_EPOCHS = 12  # more epochs for better convergence
MODEL_SAVE_PATH = 'best_vit_aptos.pth'

# ==== 7. Training Function with MixUp & GradScaler ====
def train_one_run(bs, lr):
    print(f"\n--- Training with batch size: {bs}, learning rate: {lr} ---")
    
    # MixUp
    mixup_fn = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=0.5,
        switch_prob=0.5,
        mode='batch',
        num_classes=5
    )

    train_loader = DataLoader(train_dataset, batch_size=bs, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=5)
    model = model.to(device)

    criterion = SoftTargetCrossEntropy()  # works better with MixUp
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=NUM_EPOCHS,
        lr_min=1e-6,
        warmup_lr_init=1e-6,
        warmup_t=2,
        cycle_limit=1,
        t_in_epochs=True
    )
    scaler = GradScaler()

    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Apply MixUp
            inputs, labels = mixup_fn(inputs, labels)

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

        scheduler.step(epoch + 1)

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)  # standard CE for validation
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct / total

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Saved best model at epoch {epoch+1} with val acc {val_acc:.2f}%")

    return best_acc

# ==== 8. Grid Search ====
best_overall_acc = 0
best_params = {}

for bs in batch_sizes:
    for lr in learning_rates:
        acc = train_one_run(bs, lr)
        if acc > best_overall_acc:
            best_overall_acc = acc
            best_params = {'batch_size': bs, 'learning_rate': lr}

print(f"\n=== Best Accuracy: {best_overall_acc:.2f}% with params: {best_params} ===")



# --- 1. Load the Best Model ---
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.to(device)
model.eval()

# --- 2. Get Predictions ---
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --- 3. Classification Report ---
# Manual mapping of class indices to class names
class_map = {0: 'Mild', 1: 'Moderate', 2: 'Severe', 3: 'Proliferative', 4: 'No DR'}
class_names = [class_map[i] for i in range(5)]  # 5 classes for APTOS 2019

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

# --- 4. Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()



import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. Reshape transform for ViT ---
def reshape_transform(tensor, height=14, width=14):
    # Discard the [CLS] token
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # Bring the channels to the front
    result = result.permute(0, 3, 1, 2)
    return result

# --- 2. Define Grad-CAM ---
target_layers = [model.blocks[-1].norm1]  # last block of ViT
cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

# --- 3. Select first 5 samples from validation subset ---
sample_indices = list(range(5))
sample_tensors = [val_dataset[i][0] for i in sample_indices]  # tensor images
sample_labels = [val_dataset[i][1] for i in sample_indices]   # true labels if needed

# --- 4. Visualization loop ---
for idx, input_tensor in enumerate(sample_tensors):
    input_tensor = input_tensor.unsqueeze(0).to(device)  # add batch dim

    # Model prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, pred_idx = torch.max(output, 1)
        predicted_class_name = class_names[pred_idx.item()]

    # Prepare RGB image for Grad-CAM visualization
    rgb_img = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # C,H,W -> H,W,C
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())  # normalize to 0-1

    # Generate Grad-CAM heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(rgb_img)
    ax1.set_title(f'Original Image\nTrue Label: {sample_labels[idx]}')
    ax1.axis('off')

    ax2.imshow(heatmap)
    ax2.set_title(f'Grad-CAM Heatmap\nPrediction: {predicted_class_name}')
    ax2.axis('off')

    plt.show()



import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. Reshape Transform for Vision Transformers ---
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result

# --- 2. Grad-CAM Setup ---
target_layers = [model.blocks[-1].norm1]
cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

# --- 3. Function to Get Top-K Class Probabilities ---
def get_topk_predictions(img_tensor, model, class_names, k=5):
    input_tensor = img_tensor.unsqueeze(0).to(device)  # add batch dim
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1).squeeze().cpu().numpy()
    
    topk_indices = probs.argsort()[-k:][::-1]
    topk_probs = probs[topk_indices]
    topk_labels = [class_names[i] for i in topk_indices]
    return topk_labels, topk_probs

# --- 4. Visualize Results ---
sample_indices = val_dataset.indices[:5]  # take first 5 samples from validation set

for idx in sample_indices:
    img_tensor, label = val_dataset.dataset[idx]  # returns (tensor, label)
    
    # Prepare image for Grad-CAM visualization
    rgb_img = img_tensor.permute(1, 2, 0).cpu().numpy()  # C,H,W -> H,W,C
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())  # normalize 0-1
    
    # Model prediction
    input_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, prediction_idx = torch.max(output, 1)
        predicted_class_name = class_names[prediction_idx.item()]
    
    # Grad-CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Top-5 probabilities
    topk_labels, topk_probs = get_topk_predictions(img_tensor, model, class_names, k=5)
    
    # --- Plot All Together ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original Image
    ax1.imshow(rgb_img)
    ax1.set_title(f'True Label: {class_names[label]}')
    ax1.axis('off')
    
    # Grad-CAM Heatmap
    ax2.imshow(heatmap)
    ax2.set_title(f'Grad-CAM Heatmap\nPrediction: {predicted_class_name}')
    ax2.axis('off')
    
    # Confidence Bar Plot
    sns.barplot(x=topk_probs, y=topk_labels, palette='mako', ax=ax3)
    ax3.set_title('Top-5 Class Probabilities')
    ax3.set_xlabel('Confidence')
    ax3.set_xlim(0, 1)
    for i, v in enumerate(topk_probs):
        ax3.text(v + 0.01, i, f"{v*100:.1f}%", va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# --- 1. Training & Validation Metrics Visualization ---
def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14,5))

    # Loss
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy', marker='o')
    plt.plot(epochs, history['val_acc'], label='Val Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# --- 2. Model Comparison Radar Chart ---
def plot_model_radar_chart():
    labels = ['Explainability', 'Progression Tracking', 'Accuracy Focus', 'Monitoring Use', 'Modern Architecture']
    num_vars = len(labels)

    vit_gradcam = [5, 5, 4, 5, 5]
    cnn_resnet = [2, 1, 5, 1, 3]
    quellec2017 = [3, 1, 4, 1, 3]
    lam2018 = [2, 1, 4, 1, 4]

    models = {
        "ViT + Grad-CAM": vit_gradcam,
        "CNN (ResNet)": cnn_resnet,
        "Quellec et al.": quellec2017,
        "Lam et al.": lam2018
    }
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for (name, values), color in zip(models.items(), colors):
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=name)
        ax.fill(angles, values, color=color, alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title('Model Comparison Radar Chart', size=16, pad=20)
    ax.set_rlim(0, 5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.show()

# --- 3. Grid Search Results Visualization ---
def plot_grid_search_results(grid_results, param_names):
    """
    grid_results: dict, keys are hyperparameter combinations (as tuples), values are validation accuracies
    param_names: list of strings representing hyperparameter names
    """
    df = pd.DataFrame(list(grid_results.items()), columns=['Params', 'Val_Accuracy'])
    
    # Split tuple params into separate columns
    for i, name in enumerate(param_names):
        df[name] = df['Params'].apply(lambda x: x[i])
    
    # Heatmap if only 2 hyperparameters
    if len(param_names) == 2:
        pivot_table = df.pivot(index=param_names[0], columns=param_names[1], values='Val_Accuracy')
        plt.figure(figsize=(8,6))
        sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")
        plt.title('Grid Search Validation Accuracy Heatmap')
        plt.show()
    else:
        # Bar chart for multi-dimensional hyperparameters
        plt.figure(figsize=(10,6))
        sns.barplot(x='Params', y='Val_Accuracy', data=df)
        plt.xticks(rotation=45, ha='right')
        plt.title('Grid Search Validation Accuracy')
        plt.show()

# --- 4. Example Usage ---
# 4.1 Training history
example_history = {
    'train_loss': [1.2, 0.9, 0.7, 0.5, 0.4],
    'val_loss': [1.3, 1.0, 0.8, 0.6, 0.5],
    'train_acc': [0.55, 0.65, 0.75, 0.85, 0.9],
    'val_acc': [0.50, 0.60, 0.70, 0.78, 0.82]
}
plot_training_history(example_history)

# 4.2 Radar chart
plot_model_radar_chart()

# 4.3 Grid search example
grid_results_example = {
    (0.0001, 16): 0.78,
    (0.0001, 32): 0.80,
    (0.001, 16): 0.82,
    (0.001, 32): 0.85,
    (0.01, 16): 0.81,
    (0.01, 32): 0.83
}
plot_grid_search_results(grid_results_example, param_names=['Learning Rate', 'Batch Size'])


