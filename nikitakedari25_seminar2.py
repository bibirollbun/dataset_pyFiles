import os
import glob
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


dataset_path = "/kaggle/input/alaska2-image-steganalysis"
cover_path = os.path.join(dataset_path, "Cover")
jmipod_path = os.path.join(dataset_path, "JMiPOD")
juniward_path = os.path.join(dataset_path, "JUNIWARD")
uerd_path = os.path.join(dataset_path, "UERD")
test_path = os.path.join(dataset_path, "Test")


# Get list of images
cover_images = glob.glob(cover_path + "/*.jpg")
jmipod_images = glob.glob(jmipod_path + "/*.jpg")
juniward_images = glob.glob(juniward_path + "/*.jpg")
uerd_images = glob.glob(uerd_path + "/*.jpg")
test_images = glob.glob(test_path + "/*.jpg")


print(f"Total Cover Images: {len(cover_images)}")
print(f"Total JMiPOD Images: {len(jmipod_images)}")
print(f"Total JUNIWARD Images: {len(juniward_images)}")
print(f"Total UERD Images: {len(uerd_images)}")
print(f"Total Test Images: {len(test_images)}")


def show_sample_images(folder_paths, titles, n=3):
    fig, axes = plt.subplots(len(folder_paths), n, figsize=(15, 10))
    for i, path in enumerate(folder_paths):
        sample_imgs = random.sample(glob.glob(os.path.join(path, "*.*")), n)
        for j, img_path in enumerate(sample_imgs):
            img = Image.open(img_path)
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_title(titles[i], fontsize=14)
    plt.tight_layout()
    plt.show()

folder_paths = [cover_path, jmipod_path, juniward_path, uerd_path, test_path]
titles = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD', 'Test']

show_sample_images(folder_paths, titles)


import os
import glob
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm


# --------------------------
# 2. PREPROCESSING FUNCTIONS
# --------------------------
# 5x5 high-pass filter kernel (SRM-inspired)
hpf_kernel = torch.tensor([
    [-1,  2, -2,  2, -1],
    [ 2, -6,  8, -6,  2],
    [-2,  8,-12,  8, -2],
    [ 2, -6,  8, -6,  2],
    [-1,  2, -2,  2, -1]
], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 12.0

def get_noise_residual(image_tensor):
    # Convert RGB tensor to grayscale
    grayscale = transforms.functional.rgb_to_grayscale(image_tensor, num_output_channels=1)
    grayscale = grayscale.unsqueeze(0)  # Shape: [1, 1, H, W]
    # Apply high-pass filter (no device transfer here; hpf_kernel is on CPU by default, but you can .to(device) if needed)
    residual = F.conv2d(grayscale, hpf_kernel, padding=2)
    return residual.squeeze(0)  # Shape: [1, H, W]



# --------------------------
# 3. DATASET CLASSES
# --------------------------
class Alaska2RGBDCTDataset(Dataset):
    """
    Standard dataset that loads image, computes DCT coefficients and noise residual.
    Label is provided (0: Cover, 1: JMiPOD, 2: JUNIWARD, 3: UERD).
    """
    def __init__(self, image_folder, label, transform=None, max_samples=20000):
        self.image_paths = glob.glob(os.path.join(image_folder, "*.*"))
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_folder}!")
        random.shuffle(self.image_paths)
        self.image_paths = self.image_paths[:max_samples]
        self.label = label
        self.transform = transform

    def get_dct_coefficients(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        dct = cv2.dct(np.float32(img))
        return torch.tensor(dct).unsqueeze(0)  # shape: [1, 224, 224]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        dct_coeffs = self.get_dct_coefficients(img_path)
        noise_residual = get_noise_residual(image)  # [1, 224, 224]
        return image, dct_coeffs, noise_residual, self.label

# For binary classification (Cover vs. Stego)
class BinaryAlaskaDataset(Alaska2RGBDCTDataset):
    def __init__(self, image_folder, is_stego, transform=None, max_samples=20000):
        # For binary, label 0 for cover and 1 for stego
        super().__init__(image_folder, label=int(is_stego), transform=transform, max_samples=max_samples)

# --------------------------
# 4. TRANSFORM & PATHS
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
base_dir = "/kaggle/input/alaska2-image-steganalysis"
cover_path   = os.path.join(base_dir, "Cover")
jmipod_path  = os.path.join(base_dir, "JMiPOD")
juniward_path= os.path.join(base_dir, "JUNIWARD")
uerd_path    = os.path.join(base_dir, "UERD")
test_path    = os.path.join(base_dir, "Test")



# --------------------------
# 5. CREATE DATASETS & DATALOADERS
# --------------------------
# Multi-class datasets (labels: cover=0, jmipod=1, juniward=2, uerd=3)
cover_dataset    = Alaska2RGBDCTDataset(cover_path, label=0, transform=transform)
jmipod_dataset   = Alaska2RGBDCTDataset(jmipod_path, label=1, transform=transform)
juniward_dataset = Alaska2RGBDCTDataset(juniward_path, label=2, transform=transform)
uerd_dataset     = Alaska2RGBDCTDataset(uerd_path, label=3, transform=transform)

# Concatenate to form a full dataset (for multi-class classification)
from torch.utils.data import ConcatDataset
full_dataset = ConcatDataset([cover_dataset, jmipod_dataset, juniward_dataset, uerd_dataset])
full_loader = DataLoader(full_dataset, batch_size=32, shuffle=True, num_workers=2)

# For binary classification: cover = 0, all stego = 1
binary_dataset = ConcatDataset([
    BinaryAlaskaDataset(cover_path, is_stego=False, transform=transform),
    BinaryAlaskaDataset(jmipod_path, is_stego=True, transform=transform),
    BinaryAlaskaDataset(juniward_path, is_stego=True, transform=transform),
    BinaryAlaskaDataset(uerd_path, is_stego=True, transform=transform)
])
binary_loader = DataLoader(binary_dataset, batch_size=32, shuffle=True, num_workers=2)

# Optionally, create test_loader (here we assume test images are unlabeled, so we set label to 0 as a dummy)
test_dataset = Alaska2RGBDCTDataset(test_path, label=0, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)


# --------------------------
# 6. VISUALIZATION FUNCTIONS (for sanity check)
# --------------------------
def visualize_image_and_dct(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (224, 224))
    dct = cv2.dct(np.float32(img_resized))
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(img_resized, cmap='gray')
    axs[0].set_title("Grayscale Image")
    axs[0].axis('off')
    
    axs[1].imshow(np.log(np.abs(dct) + 1), cmap='inferno')
    axs[1].set_title("DCT Coefficients")
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize a sample image from Cover folder
sample_img_path = random.choice(glob.glob(os.path.join(cover_path, "*.jpg")))
visualize_image_and_dct(sample_img_path)

def visualize_sample(dataset, idx=0):
    image, dct, noise, label = dataset[idx]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
    axs[0].set_title("RGB Image")
    axs[0].axis("off")
    axs[1].imshow(torch.log(torch.abs(dct.squeeze()) + 1).cpu(), cmap='inferno')
    axs[1].set_title("DCT Coefficients")
    axs[1].axis("off")
    axs[2].imshow(noise.squeeze().cpu(), cmap='gray')
    axs[2].set_title("Noise Residual")
    axs[2].axis("off")
    plt.suptitle(f"Label: {label}")
    plt.tight_layout()
    plt.show()

# Test visualizations on a couple of examples
visualize_sample(jmipod_dataset, idx=5)
visualize_sample(juniward_dataset, idx=5)



import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1):
        super(CNNFeatureExtractor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)  # [B, 128]



class ViTFeatureExtractor(nn.Module):
    def __init__(self):
        super(ViTFeatureExtractor, self).__init__()
        self.vit = vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()  # Remove classification head -> output [B, 768]

    def forward(self, x):
        return self.vit(x)


class HybridBinaryClassifier(nn.Module):
    def __init__(self):
        super(HybridBinaryClassifier, self).__init__()
        self.rgb_extractor = ViTFeatureExtractor()           # RGB: [B, 768]
        self.dct_extractor = CNNFeatureExtractor(1)          # DCT: [B, 128]
        self.noise_extractor = CNNFeatureExtractor(1)        # Noise Residual: [B, 128]

        self.classifier = nn.Sequential(
            nn.Linear(768 + 128 + 128, 256),  # Total input: 1024
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Binary classification
        )

    def forward(self, rgb, dct, noise):
        rgb_feat = self.rgb_extractor(rgb)           # [B, 768]
        dct_feat = self.dct_extractor(dct)           # [B, 128]
        noise_feat = self.noise_extractor(noise)     # [B, 128]
        
        combined = torch.cat((rgb_feat, dct_feat, noise_feat), dim=1)  # [B, 1024]
        output = self.classifier(combined)
        return output


from torchvision.models.vision_transformer import vit_b_16


model = HybridBinaryClassifier().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)



def train_binary(model, dataloader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for rgb, dct, noise, labels in tqdm(dataloader):
            rgb, dct, noise, labels = rgb.to(device), dct.to(device), noise.to(device), labels.float().unsqueeze(1).to(device)

            preds = model(rgb, dct, noise)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = (preds > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f} - Accuracy: {acc:.2f}%")


train_binary(model, binary_loader, criterion, optimizer, epochs=3)


# Save the model after training
torch.save(model.state_dict(), "hybrid_binary_classifier.pth")
print("Model saved to 'hybrid_binary_classifier.pth'")


# Load model (if starting fresh)
model = HybridBinaryClassifier()
model.load_state_dict(torch.load("hybrid_binary_classifier.pth"))
model.to(device)


import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for rgb, dct, noise, labels in tqdm(dataloader, desc="Evaluating"):
            rgb, dct, noise, labels = rgb.to(device), dct.to(device), noise.to(device), labels.to(device)

            outputs = model(rgb, dct, noise).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\nEvaluation Metrics:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return acc, prec, rec, f1



evaluate_model(model, binary_loader)


import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm


def evaluate_with_metrics(model, dataloader, device):
    model.eval()
    criterion = nn.BCELoss()

    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0

    with torch.no_grad():
        for rgb, dct, noise, labels in tqdm(dataloader):
            rgb, dct, noise, labels = rgb.to(device), dct.to(device), noise.to(device), labels.float().unsqueeze(1).to(device)

            probs = model(rgb, dct, noise)
            loss = criterion(probs, labels)
            total_loss += loss.item()

            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    auc_score = auc(fpr, tpr)

    print(f"\nLoss: {total_loss / len(dataloader):.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC Score: {auc_score:.4f}")

    # Confusion Matrix Plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # ROC Curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid()
    plt.show()



evaluate_with_metrics(model, test_loader, device)


from sklearn.metrics import classification_report
import pandas as pd

def plot_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    sns.heatmap(df.iloc[:-1, :-1], annot=True, cmap="YlGnBu")
    plt.title("Classification Report")
    plt.show()



def plot_probability_histogram(y_probs, y_true):
    plt.figure(figsize=(6,4))
    plt.hist(y_probs[y_true==0], bins=25, alpha=0.6, label='Class 0')
    plt.hist(y_probs[y_true==1], bins=25, alpha=0.6, label='Class 1')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Probability Distribution per Class')
    plt.legend()
    plt.show()



def plot_threshold_vs_metrics(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    plt.figure(figsize=(7,5))
    plt.plot(thresholds, precision[:-1], label="Precision", color='b')
    plt.plot(thresholds, recall[:-1], label="Recall", color='g')
    plt.plot(thresholds, f1_scores[:-1], label="F1 Score", color='r')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold vs Precision, Recall, F1")
    plt.legend()
    plt.grid()
    plt.show()



from sklearn.metrics import precision_recall_curve

def plot_precision_recall(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, color='purple')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid()
    plt.show()



plot_classification_report(all_labels, all_preds)
plot_precision_recall(all_labels, all_probs)
plot_probability_histogram(all_probs.flatten(), all_labels.flatten())
plot_threshold_vs_metrics(all_labels, all_probs.flatten())



# Rebuild model and load saved weights
model = HybridBinaryClassifier()
model.load_state_dict(torch.load("hybrid_binary_classifier.pth"))
model.to(device)
model.eval()

# Evaluation function
def evaluate_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    total_loss = 0
    criterion = nn.BCELoss()  # Assuming Binary Cross Entropy

    with torch.no_grad():
        for rgb, dct, noise, labels in tqdm(dataloader):
            rgb, dct, noise, labels = rgb.to(device), dct.to(device), noise.to(device), labels.float().unsqueeze(1).to(device)

            preds = model(rgb, dct, noise)
            loss = criterion(preds, labels)

            total_loss += loss.item()
            predicted = (preds > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    print(f"Test Loss: {avg_loss:.4f} - Test Accuracy: {accuracy:.2f}%")



evaluate_model(model, test_loader, device)


import os
import glob
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import vit_b_16
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_curve
from torch.utils.data import ConcatDataset

#---------------------TRANSFORM & PATHS---------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.RandomRotation(degrees=20),  # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

base_dir = "/kaggle/input/alaska2-image-steganalysis"
cover_path   = os.path.join(base_dir, "Cover")
jmipod_path  = os.path.join(base_dir, "JMiPOD")
juniward_path= os.path.join(base_dir, "JUNIWARD")
uerd_path    = os.path.join(base_dir, "UERD")
test_path    = os.path.join(base_dir, "Test")
#---------------------DATASET CLASSES---------------------

class Alaska2RGBDCTDataset(Dataset):
    """
    Standard dataset that loads image, computes DCT coefficients and noise residual.
    Label is provided (0: Cover, 1: JMiPOD, 2: JUNIWARD, 3: UERD).
    """
    def __init__(self, image_folder, label, transform=None, max_samples=20000):
        self.image_paths = glob.glob(os.path.join(image_folder, "*.jpg")) # Corrected the glob pattern
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_folder}!")
        random.shuffle(self.image_paths)
        self.image_paths = self.image_paths[:max_samples]
        self.label = label
        self.transform = transform

    def get_dct_coefficients(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        dct = cv2.dct(np.float32(img))
        return torch.tensor(dct).unsqueeze(0)  # shape: [1, 224, 224]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        dct_coeffs = self.get_dct_coefficients(img_path)
        noise_residual = get_noise_residual(image)  # [1, 224, 224]
        return image, dct_coeffs, noise_residual, self.label


# For binary classification (Cover vs. Stego)
class BinaryAlaskaDataset(Alaska2RGBDCTDataset):
    def __init__(self, image_folder, is_stego, transform=None, max_samples=20000):
        # For binary, label 0 for cover and 1 for stego
        super().__init__(image_folder, label=int(is_stego), transform=transform, max_samples=max_samples)


#---------------------PREPROCESSING FUNCTIONS---------------------

# 5x5 high-pass filter kernel (SRM-inspired)
hpf_kernel = torch.tensor([
    [-1,  2, -2,  2, -1],
    [ 2, -6,  8, -6,  2],
    [-2,  8,-12,  8, -2],
    [ 2, -6,  8, -6,  2],
    [-1,  2, -2,  2, -1]
], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 12.0


def get_noise_residual(image_tensor):
    """
    Convert RGB tensor to grayscale
    """
    grayscale = transforms.functional.rgb_to_grayscale(image_tensor, num_output_channels=1)
    grayscale = grayscale.unsqueeze(0)  # Shape: [1, 1, H, W]

    """
    Apply high-pass filter (no device transfer here; hpf_kernel is on CPU by default, but you can .to(device) if needed)
    """
    residual = F.conv2d(grayscale, hpf_kernel, padding=2)
    return residual.squeeze(0)  # Shape: [1, H, W]

#---------------------CREATE DATASETS & DATALOADERS---------------------
# Multi-class datasets (labels: cover=0, jmipod=1, juniward=2, uerd=3)
cover_dataset    = Alaska2RGBDCTDataset(cover_path, label=0, transform=transform)
jmipod_dataset   = Alaska2RGBDCTDataset(jmipod_path, label=1, transform=transform)
juniward_dataset = Alaska2RGBDCTDataset(juniward_path, label=2, transform=transform)
uerd_dataset     = Alaska2RGBDCTDataset(uerd_path, label=3, transform=transform)

# For binary classification: cover = 0, all stego = 1
binary_dataset = ConcatDataset([
    BinaryAlaskaDataset(cover_path, is_stego=False, transform=transform),
    BinaryAlaskaDataset(jmipod_path, is_stego=True, transform=transform),
    BinaryAlaskaDataset(juniward_path, is_stego=True, transform=transform),
    BinaryAlaskaDataset(uerd_path, is_stego=True, transform=transform)
])

# Calculate class weights for the binary dataset
num_cover = len(glob.glob(os.path.join(cover_path, "*.jpg")))
num_stego = len(binary_dataset) - num_cover  # Estimate, may need adjustment
pos_weight = torch.tensor([num_cover / num_stego]) # Important for imbalanced datasets

binary_loader = DataLoader(binary_dataset, batch_size=32, shuffle=True, num_workers=2)

# Optionally, create test_loader (here we assume test images are unlabeled, so we set label to 0 as a dummy)
test_dataset = Alaska2RGBDCTDataset(test_path, label=0, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

#---------------------MODEL ARCHITECTURE---------------------
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1):
        super(CNNFeatureExtractor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)  # [B, 128]


class ViTFeatureExtractor(nn.Module):
    def __init__(self):
        super(ViTFeatureExtractor, self).__init__()
        self.vit = vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()  # Remove classification head -> output [B, 768]

    def forward(self, x):
        return self.vit(x)


class HybridBinaryClassifier(nn.Module):
    def __init__(self):
        super(HybridBinaryClassifier, self).__init__()
        self.rgb_extractor = ViTFeatureExtractor()           # RGB: [B, 768]
        self.dct_extractor = CNNFeatureExtractor(1)          # DCT: [B, 128]
        self.noise_extractor = CNNFeatureExtractor(1)        # Noise Residual: [B, 128]

        self.classifier = nn.Sequential(
            nn.Linear(768 + 128 + 128, 256),  # Total input: 1024
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Binary classification
        )

    def forward(self, rgb, dct, noise):
        rgb_feat = self.rgb_extractor(rgb)           # [B, 768]
        dct_feat = self.dct_extractor(dct)           # [B, 128]
        noise_feat = self.noise_extractor(noise)     # [B, 128]

        combined = torch.cat((rgb_feat, dct_feat, noise_feat), dim=1)  # [B, 1024]
        output = self.classifier(combined)
        return output

#---------------------TRAINING---------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = HybridBinaryClassifier().to(device)

# Use BCEWithLogitsLoss for numerical stability and to incorporate class weights
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))  # Class imbalance handling
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train_binary(model, dataloader, criterion, optimizer, epochs=5, patience=3): # Added patience for early stopping
    model.train()
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for rgb, dct, noise, labels in tqdm(dataloader):
            rgb, dct, noise, labels = rgb.to(device), dct.to(device), noise.to(device), labels.float().unsqueeze(1).to(device)

            outputs = model(rgb, dct, noise)  # No sigmoid here
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.sigmoid(outputs)  # Apply sigmoid for prediction
            predicted = (preds > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        acc = 100 * correct / total
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} - Accuracy: {acc:.2f}%")

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "hybrid_binary_classifier_best.pth") # Save best model
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping triggered!")
                model.load_state_dict(torch.load("hybrid_binary_classifier_best.pth"))  # Load best model
                return


train_binary(model, binary_loader, criterion, optimizer, epochs=10, patience=3)  # Train for longer, but with early stopping

#---------------------SAVING MODEL---------------------
# Save the model after training
torch.save(model.state_dict(), "hybrid_binary_classifier.pth")
print("Model saved to 'hybrid_binary_classifier.pth'")

#---------------------EVALUATION---------------------
def evaluate_model(model, dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for rgb, dct, noise, labels in tqdm(dataloader, desc="Evaluating"):
            rgb, dct, noise, labels = rgb.to(device), dct.to(device), noise.to(device), labels.to(device)

            outputs = model(rgb, dct, noise).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\nEvaluation Metrics:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return acc, prec, rec, f1

evaluate_model(model, test_loader)








