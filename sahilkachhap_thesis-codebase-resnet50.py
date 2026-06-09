import cv2
import os
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, cohen_kappa_score, confusion_matrix, roc_curve, auc
import time

# Pytorch Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F


dataset = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")
print(f"Total Samples in Dataset: {len(dataset)}")


# Path to APTOS-2019 dataset
aptos_root = "/kaggle/input/aptos2019-blindness-detection"
img_dir = os.path.join(aptos_root, "train_images")
csv_path = os.path.join(aptos_root, "train.csv")

# create full path
dataset['path'] = dataset['id_code'].apply(lambda x: os.path.join(img_dir, f"{x}.png"))

dataset.head()


# Count class distribution
class_counts = dataset['diagnosis'].value_counts().sort_index()

# Bar plot
plt.figure(figsize=(6,4))
sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
plt.title("Class Distribution (APTOS-2019)")
plt.xlabel("DR Class")
plt.ylabel("Number of Images")
plt.show()


# Pie chart
plt.figure(figsize=(6,6))
plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis", len(class_counts)))
plt.title("Class Distribution (Pie Chart)")
plt.show()


plt.figure(figsize=(12, 8))
for i, c in enumerate(sorted(dataset['diagnosis'].unique())):
    sample_paths = dataset[dataset['diagnosis']==c]['path'].sample(3, random_state=42)
    for j, p in enumerate(sample_paths):
        img = cv2.imread(p); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(len(class_counts), 3, i*3 + j + 1)
        plt.imshow(img)
        plt.axis("off")
        if j==1: plt.title(f"Class: {c}")
plt.tight_layout()
plt.show()


# Training transform
train_tf = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                       border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(0.15,0.15,p=0.5),
    A.GaussNoise(std_range=(0.02,0.08), p=0.2),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
], additional_targets={
    'mask_ma':'mask','mask_he':'mask','mask_ex':'mask','mask_se':'mask'
})

# Validation/test transform
val_tf = A.Compose([
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
], additional_targets={
    'mask_ma':'mask','mask_he':'mask','mask_ex':'mask','mask_se':'mask'
})


def retina_crop(img):
    """Crop black borders around retina using green channel + thresholding."""
    g = cv2.GaussianBlur(img[:,:,1], (0,0), 5)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        pad = int(0.02*max(w,h))
        x = max(0,x-pad); y = max(0,y-pad)
        return img[y:y+h+2*pad, x:x+w+2*pad]
    return img

def illumination_correction(img):
    """Correct uneven lighting using Gaussian blur background division."""
    imgf = img.astype(np.float32) / 255.0
    k = int(round(min(img.shape[:2]) * 0.05)) | 1  # kernel size ~5% of min dim
    bg = cv2.GaussianBlur(imgf, (k,k), 0)
    corr = np.clip((imgf/(bg+1e-3))*bg.mean(), 0, 1)
    return (corr*255).astype(np.uint8)

def clahe_green(img):
    """Apply CLAHE on green channel."""
    g = img[:,:,1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g2 = clahe.apply(g)
    img[:,:,1] = g2
    return img



cache_dir = "data/aptos2019_preprocessed"
os.makedirs(cache_dir, exist_ok=True)


def preprocess_and_save(src_paths, dst_dir):
    for path in tqdm(src_paths, desc="Preprocessing & Saving"):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply your preprocessing pipeline
        img = retina_crop(img)
        img = illumination_correction(img)
        img = clahe_green(img)

        # Resize to 224x224 (or whatever you train with)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

        # Save as PNG
        filename = os.path.basename(path)
        save_path = os.path.join(dst_dir, filename)
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # back to BGR for OpenCV save

# Example usage
preprocess_and_save(dataset['path'].values, cache_dir)


dataset['cached_path'] = dataset['id_code'].apply(lambda x: os.path.join(cache_dir, f"{x}.png"))


# Extract labels & paths
img_paths = dataset['cached_path'].values
labels = dataset['diagnosis'].values

dataset.head()


class FundusDataset(Dataset):
    def __init__(self, img_paths, labels=None, masks=None, transform=None, is_train=True):
        """
        img_paths: list of image file paths
        labels: list of int labels (for classification)
        masks: dict of mask paths { 'ma':[], 'he':[], 'ex':[], 'se':[] } (for IDRiD)
        transform: albumentations transform
        is_train: bool, whether training or not
        """
        self.img_paths = img_paths
        self.labels = labels
        self.masks = masks
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Prepare masks if available
        mask_dict = {}
        if self.masks is not None:
            for key in self.masks.keys():
                mask_path = self.masks[key][idx]
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = (mask > 127).astype('uint8')  # binarize {0,1}
                mask_dict[f"mask_{key}"] = mask

        # Apply transforms
        if self.transform:
            if mask_dict:
                aug = self.transform(image=img, **mask_dict)
                img = aug['image']
                masks_out = {k: aug[k] for k in mask_dict.keys()}
                return img, masks_out
            else:
                aug = self.transform(image=img)
                img = aug['image']
                label = self.labels[idx]
                return img, torch.tensor(label, dtype=torch.long)

        # Fallback: return raw
        if mask_dict:
            return img, mask_dict
        else:
            return img, self.labels[idx]


# Stratified split into train/val/test (70/15/15)
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    img_paths, labels, test_size=0.20, stratify=labels, random_state=42
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42
)


# For APTOS classification
train_dataset = FundusDataset(
    img_paths=list(train_paths),
    labels=list(train_labels),
    transform=train_tf
)

# Validation dataset (no augmentation)
val_dataset = FundusDataset(
    img_paths=list(val_paths),
    labels=list(val_labels),
    transform=val_tf
)

# Test dataset (no augmentation)
test_dataset = FundusDataset(
    img_paths=list(test_paths),
    labels=list(test_labels),
    transform=val_tf
)

# For IDRiD lesion validation
# idrid_dataset = FundusDataset(
#     img_paths=idrid_img_paths,
#     masks={
#         'ma': ma_mask_paths,
#         'he': he_mask_paths,
#         'ex': ex_mask_paths,
#         'se': se_mask_paths
#     },
#     transform=val_tf,
#     is_train=False
# )

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


def visualize_samples(dataset, num_samples=4, with_masks=False):
    """
    Visualize original vs transformed fundus images.
    
    Args:
        dataset: instance of FundusDataset (defined earlier).
        num_samples: how many samples to show.
        with_masks: True if dataset returns masks (IDRiD).
    """
    plt.figure(figsize=(12, num_samples * 4))
    
    for i in range(num_samples):
        img_path = dataset.img_paths[i]
        
        # --- Load original (without transforms)
        orig = cv2.imread(img_path)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        
        # --- Get transformed sample from dataset
        sample = dataset[i]
        
        if with_masks:
            img, masks = sample
            img = img.permute(1,2,0).cpu().numpy()  # CHW → HWC
            img = (img - img.min()) / (img.max() - img.min())  # normalize for viewing
        else:
            img, label = sample
            img = img.permute(1,2,0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min())
        
        # --- Plot original and transformed
        plt.subplot(num_samples, 2, 2*i+1)
        plt.imshow(orig)
        plt.title(f"Original {i}")
        plt.axis("off")
        
        plt.subplot(num_samples, 2, 2*i+2)
        plt.imshow(img)
        if with_masks:
            # Overlay one example mask if available
            for k, m in masks.items():
                m = m.squeeze().cpu().numpy()
                plt.contour(m, colors='r', linewidths=0.5)
        plt.title(f"Transformed {i}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()


visualize_samples(train_dataset, num_samples=4, with_masks=False)


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Load pretrained ResNet50
resnet50 = models.resnet50(weights="ResNet50_Weights.DEFAULT")


# Modify final fully connected layer for 5 DR classes
num_features = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_features, 5)


# Move model to GPU
resnet50 = resnet50.to(device)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()   # Later you can add class weights if needed
optimizer = torch.optim.Adam(resnet50.parameters(), lr=1e-4, weight_decay=1e-5)

# Optional: learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

print(resnet50)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20, device="cuda"):
    since = time.time()
    
    best_model_wts = model.state_dict()
    best_acc = 0.0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        # ---- Training phase ----
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # ---- Validation phase ----
        model.eval()
        running_loss, running_corrects, total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

        val_loss = running_loss / total
        val_acc = running_corrects.double() / total
        val_losses.append(val_loss)
        val_accs.append(val_acc.item())

        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # ---- Scheduler step ----
        if scheduler:
            scheduler.step()

        # ---- Deep copy best model ----
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    # Load best weights
    model.load_state_dict(best_model_wts)

    return model, (train_losses, val_losses, train_accs, val_accs)



# Train the ResNet50 model
resnet50, history = train_model(
    model=resnet50,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=30,
    device=device
)


# Unpack history
train_losses, val_losses, train_accs, val_accs = history


# Loss curve
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

# Accuracy curve
plt.figure(figsize=(8,5))
plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()


@torch.no_grad()
def evaluate_model(model, test_loader, device="cuda", save_cm_path="confusion_matrix.png", class_names=None):
    model.eval()
    all_logits, all_probs, all_preds, all_labels = [], [], [], []

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        logits = model(inputs)                            # [B, 5]
        probs = torch.softmax(logits, dim=1)              # [B, 5]
        preds = probs.argmax(dim=1)                       # [B]

        all_logits.append(logits.cpu())
        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_probs  = torch.cat(all_probs).numpy()
    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # --- Metrics ---
    acc = accuracy_score(all_labels, all_preds)
    f1m = f1_score(all_labels, all_preds, average="macro")

    # One-vs-Rest macro AUC (requires prob estimates)
    try:
        auc_macro = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc_macro = float("nan")  # if a class missing in test, AUC may fail

    # Quadratic Weighted Kappa
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")

    # Per-class AUC (OvR)
    per_class_auc = {}
    num_classes = all_probs.shape[1]
    for c in range(num_classes):
        y_true = (all_labels == c).astype(int)
        y_score = all_probs[:, c]
        try:
            per_class_auc[c] = roc_auc_score(y_true, y_score)
        except ValueError:
            per_class_auc[c] = float("nan")

    # Classification report
    print("\n=== Test Metrics ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1m:.4f}")
    print(f"Macro AUC (OvR): {auc_macro:.4f}")
    print(f"Quadratic Weighted Kappa: {qwk:.4f}")

    # Optional: pretty per-class names
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (APTOS Test)")
    plt.tight_layout()
    plt.savefig(save_cm_path, dpi=200)
    plt.show()

    # Return everything you’ll need for uncertainty/calibration later
    results = {
        "acc": acc,
        "f1_macro": f1m,
        "auc_macro": auc_macro,
        "qwk": qwk,
        "per_class_auc": per_class_auc,
        "labels": all_labels,
        "preds": all_preds,
        "probs": all_probs,
        "logits": all_logits,
        "confusion_matrix": cm
    }
    return results


class_names = ["No DR (0)", "Mild (1)", "Moderate (2)", "Severe (3)", "Proliferative (4)"]
test_results = evaluate_model(resnet50, test_loader, device=device, class_names=class_names)


def enable_dropout(model):
    """Enable dropout layers during test-time"""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

@torch.no_grad()
def mc_dropout_predictions_all(model, dataloader, device="cuda", T=30):
    model.eval()
    enable_dropout(model)  # keep dropout active

    all_probs = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        batch_probs = []

        for _ in range(T):
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            batch_probs.append(probs.unsqueeze(0))  # [1,B,C]

        batch_probs = torch.cat(batch_probs, dim=0)  # [T,B,C]
        all_probs.append(batch_probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=1)  # [T,N,C]
    all_labels = np.concatenate(all_labels, axis=0)

    # Predictive mean
    mean_probs = all_probs.mean(axis=0)  # [N,C]

    # --- Uncertainty Measures ---
    # 1. Entropy
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)

    # 2. Variance
    variance = all_probs.var(axis=0).mean(axis=1)

    # 3. Mutual Information (MI)
    # Predictive entropy - expected entropy
    expected_entropy = -np.mean(np.sum(all_probs * np.log(all_probs + 1e-8), axis=2), axis=0)
    MI = entropy - expected_entropy

    # 4. Max Probability (Confidence Score)
    max_prob = mean_probs.max(axis=1)

    preds = mean_probs.argmax(axis=1)
    return {
        "mean_probs": mean_probs,
        "entropy": entropy,
        "variance": variance,
        "mutual_info": MI,
        "max_prob": max_prob,
        "preds": preds,
        "labels": all_labels
    }


def evaluate_uncertainty_all(results):
    labels = results["labels"]
    preds = results["preds"]

    # Binary: 1 = error, 0 = correct
    errors = (preds != labels).astype(int)

    metrics = {}
    for name, scores in {
        "Entropy": results["entropy"],
        "Variance": results["variance"],
        "Mutual Information": results["mutual_info"],
        "Max Probability": -results["max_prob"]  # invert since low prob = high uncertainty
    }.items():
        try:
            auroc = roc_auc_score(errors, scores)
            metrics[name] = auroc
            print(f"{name} AUROC: {auroc:.4f}")
        except ValueError:
            metrics[name] = None
            print(f"{name}: could not compute AUROC (maybe missing error cases).")

    return metrics


# Run MC Dropout with all uncertainty measures
mc_results_all = mc_dropout_predictions_all(resnet50, test_loader, device=device, T=30)

# Evaluate AUROC for all measures
unc_metrics = evaluate_uncertainty_all(mc_results_all)


def plot_uncertainty_hist(results, metric="entropy", bins=30):
    labels = results["labels"]
    preds = results["preds"]
    errors = (preds != labels).astype(int)

    scores = results[metric]

    correct_scores = scores[errors == 0]
    wrong_scores = scores[errors == 1]

    plt.figure(figsize=(7,5))
    sns.histplot(correct_scores, bins=bins, color="green", alpha=0.6, label="Correct")
    sns.histplot(wrong_scores, bins=bins, color="red", alpha=0.6, label="Wrong")
    plt.xlabel(f"{metric.capitalize()} Score")
    plt.ylabel("Count")
    plt.title(f"Histogram of {metric.capitalize()} for Correct vs Incorrect Predictions")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example: plot entropy histogram
plot_uncertainty_hist(mc_results_all, metric="entropy")


def plot_uncertainty_rocs(results):
    labels = results["labels"]
    preds = results["preds"]
    errors = (preds != labels).astype(int)

    plt.figure(figsize=(7,6))

    metrics_to_plot = {
        "Entropy": results["entropy"],
        "Variance": results["variance"],
        "Mutual Information": results["mutual_info"],
        "Max Probability": -results["max_prob"]  # invert: low prob = uncertain
    }

    for name, scores in metrics_to_plot.items():
        fpr, tpr, _ = roc_curve(errors, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUROC = {roc_auc:.2f})")

    plt.plot([0,1],[0,1], "k--", label="Random (0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Error Detection via Uncertainty")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

# Example: plot ROC curves
plot_uncertainty_rocs(mc_results_all)


y_true = mc_results_all["labels"]
y_pred = mc_results_all["preds"]
entropy = mc_results_all["entropy"]
variance = mc_results_all["variance"]


def risk_coverage_curve(y_true, y_pred, uncertainty_scores):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    uncertainty_scores = np.array(uncertainty_scores)

    # Sort by uncertainty (descending = reject first)
    idx = np.argsort(uncertainty_scores)[::-1]

    y_true_sorted = y_true[idx]
    y_pred_sorted = y_pred[idx]

    n = len(y_true)
    coverages = []
    accuracies = []

    for k in range(n):
        y_true_kept = y_true_sorted[k:]
        y_pred_kept = y_pred_sorted[k:]

        if len(y_true_kept) == 0:
            break

        acc = np.mean(y_true_kept == y_pred_kept)
        cov = len(y_true_kept) / n

        accuracies.append(acc)
        coverages.append(cov)

    return np.array(coverages), np.array(accuracies)



cov_e, acc_e = risk_coverage_curve(y_true, y_pred, entropy)

plt.figure(figsize=(7,5))
plt.plot(cov_e, acc_e, linewidth=2)
plt.xlabel("Coverage (Fraction of Samples Kept)")
plt.ylabel("Accuracy on Kept Samples")
plt.title("Risk–Coverage Curve using Predictive Entropy")
plt.grid(True)
plt.tight_layout()
plt.show()


cov_v, acc_v = risk_coverage_curve(y_true, y_pred, variance)

plt.figure(figsize=(7,5))
plt.plot(cov_e, acc_e, label="Entropy", linewidth=2)
plt.plot(cov_v, acc_v, label="Variance", linewidth=2)

plt.xlabel("Coverage")
plt.ylabel("Accuracy")
plt.title("Risk–Coverage Comparison of Uncertainty Measures")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



target_acc = 0.95

valid_idx = np.where(acc_e >= target_acc)[0]

if len(valid_idx) > 0:
    i = valid_idx[0]
    print(f"Target Accuracy: {acc_e[i]:.3f}")
    print(f"Coverage: {cov_e[i]:.3f}")
    print(f"Rejection Rate: {1 - cov_e[i]:.3f}")
else:
    print("Target accuracy not achievable.")



import numpy as np
import matplotlib.pyplot as plt

# Convert coverage to rejection rate
rejection = 1 - cov_e

# Optional: smooth accuracy for readability
def smooth(y, window=5):
    return np.convolve(y, np.ones(window)/window, mode="same")

acc_smooth = smooth(acc_e, window=7)

plt.figure(figsize=(8, 5))

# Main curve
plt.plot(
    rejection,
    acc_smooth,
    linewidth=2.5,
    label="Entropy-based Referral"
)

# Baseline accuracy point
plt.scatter(
    0,
    acc_e[-1],
    color="red",
    zorder=5,
    label="No Referral (Baseline Accuracy)"
)

# Clinical safety thresholds
plt.axhline(0.95, linestyle="--", color="green", alpha=0.7, label="95% Accuracy Target")
plt.axhline(0.98, linestyle="--", color="purple", alpha=0.7, label="98% Accuracy Target")

plt.xlabel("Rejection Rate (Fraction of Samples Referred)")
plt.ylabel("Accuracy on Retained Samples")
plt.title("Risk–Coverage Curve (Accuracy–Rejection) using Predictive Entropy")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



def risk_coverage_curve_with_mask(y_true, y_pred, uncertainty, mask=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    uncertainty = np.array(uncertainty)

    if mask is None:
        mask = np.ones_like(y_true, dtype=bool)

    y_true = y_true[mask]
    y_pred = y_pred[mask]
    uncertainty = uncertainty[mask]

    idx = np.argsort(uncertainty)[::-1]  # high uncertainty rejected first
    y_true = y_true[idx]
    y_pred = y_pred[idx]

    n = len(y_true)
    coverages, accuracies = [], []

    for k in range(n):
        yt = y_true[k:]
        yp = y_pred[k:]

        if len(yt) == 0:
            break

        coverages.append(len(yt) / n)
        accuracies.append(np.mean(yt == yp))

    return np.array(coverages), np.array(accuracies)



SEVERE_CLASS = 3

mask_severe = (mc_results_all["labels"] == SEVERE_CLASS)

cov_s, acc_s = risk_coverage_curve_with_mask(
    mc_results_all["labels"],
    mc_results_all["preds"],
    mc_results_all["entropy"],
    mask=mask_severe
)

plt.figure(figsize=(7,5))
plt.plot(1 - cov_s, acc_s, linewidth=2)
plt.xlabel("Rejection Rate")
plt.ylabel("Accuracy on Severe DR")
plt.title("Class-wise Risk–Coverage (Severe DR)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



entropy_scores = mc_results_all["entropy"]
maxprob_uncertainty = -mc_results_all["max_prob"]  # invert

cov_e, acc_e = risk_coverage_curve_with_mask(
    mc_results_all["labels"],
    mc_results_all["preds"],
    entropy_scores
)

cov_m, acc_m = risk_coverage_curve_with_mask(
    mc_results_all["labels"],
    mc_results_all["preds"],
    maxprob_uncertainty
)


plt.figure(figsize=(8,5))
plt.plot(1 - cov_e, acc_e, label="Entropy", linewidth=2)
plt.plot(1 - cov_m, acc_m, label="Max Probability", linewidth=2)

plt.xlabel("Rejection Rate")
plt.ylabel("Accuracy on Retained Samples")
plt.title("Risk–Coverage Comparison: Entropy vs Max Probability")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


def coverage_vs_sensitivity(y_true, y_pred, uncertainty, positive_class):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    uncertainty = np.array(uncertainty)

    idx = np.argsort(uncertainty)[::-1]
    y_true = y_true[idx]
    y_pred = y_pred[idx]

    n = len(y_true)
    coverages, recalls = [], []

    for k in range(n):
        yt = y_true[k:]
        yp = y_pred[k:]

        if len(yt) == 0:
            break

        mask = (yt == positive_class) | (yp == positive_class)
        if mask.sum() == 0:
            continue

        recall = recall_score(
            yt == positive_class,
            yp == positive_class,
            zero_division=0
        )

        coverages.append(len(yt) / n)
        recalls.append(recall)

    return np.array(coverages), np.array(recalls)



from sklearn.metrics import recall_score

cov_r, rec_r = coverage_vs_sensitivity(
    mc_results_all["labels"],
    mc_results_all["preds"],
    mc_results_all["entropy"],
    positive_class=SEVERE_CLASS
)

plt.figure(figsize=(7,5))
plt.plot(1 - cov_r, rec_r, linewidth=2)
plt.xlabel("Rejection Rate")
plt.ylabel("Sensitivity (Recall) for Severe DR")
plt.title("Coverage vs Sensitivity (Severe DR)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



# import matplotlib.pyplot as plt
# import numpy as np
# import cv2

# def show_case_studies_with_original(results, dataset, original_paths, class_names, metric="entropy"):
#     """
#     Show case studies with both original and preprocessed images.

#     Args:
#         results: dict from mc_dropout_predictions_all
#         dataset: PyTorch dataset (returns preprocessed image, label)
#         original_paths: list of file paths to original fundus images (aligned with dataset order)
#         class_names: list of class names (e.g. ["No DR", "Mild", "Moderate", "Severe", "Proliferative"])
#         metric: uncertainty metric to display ("entropy", "max_prob", etc.)
#     """

#     preds = results["preds"]
#     labels = results["labels"]
#     scores = results[metric]

#     # Pick indices for 3 case types
#     idx_correct_confident = np.argmin(scores + (preds != labels)*10)  # correct + confident
#     idx_wrong_uncertain = np.argmax(scores * (preds != labels))      # wrong + uncertain
#     idx_wrong_confident = np.argmin(scores + (preds == labels)*10)   # wrong + confident
#     chosen_idxs = [idx_correct_confident, idx_wrong_uncertain, idx_wrong_confident]

#     plt.figure(figsize=(12, 8))

#     for i, idx in enumerate(chosen_idxs):
#         # --- Load original image ---
#         orig_img = cv2.imread(original_paths[idx])
#         orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

#         # --- Load preprocessed image (from dataset) ---
#         img_tensor, _ = dataset[idx]  # preprocessed tensor
#         img = img_tensor.permute(1, 2, 0).numpy()
#         img = (img - img.min()) / (img.max() - img.min())  # normalize for display

#         true_label = class_names[labels[idx]]
#         pred_label = class_names[preds[idx]]
#         score = scores[idx]

#         # Plot original (top row)
#         plt.subplot(2, 3, i+1)
#         plt.imshow(orig_img)
#         plt.axis("off")
#         plt.title(f"Original\nTrue: {true_label}\nPred: {pred_label}\n{metric}: {score:.3f}")

#         # Plot preprocessed (bottom row)
#         plt.subplot(2, 3, i+4)
#         plt.imshow(img)
#         plt.axis("off")
#         plt.title("Preprocessed")

#     plt.tight_layout()
#     plt.show()


# class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

# # Example: using entropy
# show_case_studies_with_original(
#     mc_results_all,
#     test_dataset,
#     test_img_paths,      # list of paths to original fundus images
#     class_names,
#     metric="entropy"
# )


# show_case_studies(mc_results_all, test_dataset, metric="max_prob")


class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        # logits: [N, C]
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, valid_loader, device="cuda"):
        self.to(device)
        nll_criterion = nn.CrossEntropyLoss().to(device)

        logits_list, labels_list = [], []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = self.model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        print(f"Optimal temperature: {self.temperature.item():.3f}")
        return self


def expected_calibration_error(probs, labels, n_bins=15):
    # probs: torch tensor [N, C], labels: torch tensor [N]
    confidences, predictions = probs.max(dim=1)  # unpack values & indices
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)

    for i in range(n_bins):
        start, end = bin_boundaries[i], bin_boundaries[i+1]
        mask = (confidences > start) & (confidences <= end)
        if mask.sum() > 0:
            bin_acc = accuracies[mask].float().mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.float().mean()) * torch.abs(bin_acc - bin_conf)

    return ece.item()


def maximum_calibration_error(probs, labels, n_bins=15):
    """
    probs: torch.tensor [N, C]
    labels: torch.tensor [N]
    """
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    mce = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)

    for i in range(n_bins):
        start, end = bin_boundaries[i], bin_boundaries[i+1]
        mask = (confidences > start) & (confidences <= end)
        if mask.sum() > 0:
            bin_acc = accuracies[mask].float().mean()
            bin_conf = confidences[mask].mean()
            mce = torch.max(mce, torch.abs(bin_acc - bin_conf))

    return mce.item()



def classwise_ece(probs, labels, n_bins=15, num_classes=5):
    """
    Computes per-class ECE
    probs: torch.tensor [N, C]
    labels: torch.tensor [N]
    """
    class_ece = {}
    for c in range(num_classes):
        # Consider binary correct vs incorrect for class c
        confidences = probs[:, c]
        predictions = (probs.argmax(dim=1) == c).long()
        accuracies = (labels == c)

        ece = torch.zeros(1, device=probs.device)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)

        for i in range(n_bins):
            start, end = bin_boundaries[i], bin_boundaries[i+1]
            mask = (confidences > start) & (confidences <= end)
            if mask.sum() > 0:
                bin_acc = accuracies[mask].float().mean()
                bin_conf = confidences[mask].mean()
                ece += (mask.float().mean()) * torch.abs(bin_acc - bin_conf)

        class_ece[c] = ece.item()

    return class_ece


def brier_score(probs, labels):
    labels_onehot = np.zeros_like(probs)
    labels_onehot[np.arange(len(labels)), labels] = 1
    return np.mean(np.sum((probs - labels_onehot)**2, axis=1))


def plot_reliability_diagram(probs, labels, n_bins=10):
    """
    probs: torch.tensor [N, C] (logits after softmax)
    labels: torch.tensor [N]
    """
    # Ensure CPU + NumPy for plotting
    probs = probs.detach().cpu()
    labels = labels.detach().cpu()

    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs, bin_confs = [], []

    for i in range(n_bins):
        start, end = bin_boundaries[i], bin_boundaries[i+1]
        mask = (confidences > start) & (confidences <= end)
        if mask.sum() > 0:
            bin_acc = accuracies[mask].float().mean().item()
            bin_conf = confidences[mask].mean().item()
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)

    # Plot
    plt.figure(figsize=(6,6))
    plt.plot([0,1],[0,1],"k--")
    plt.plot(bin_confs, bin_accs, marker="o", label="Model")
    plt.xlabel("Predicted Confidence")
    plt.ylabel("True Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.show()


device = "cuda"

# Wrap model with temperature scaling
scaled_model = ModelWithTemperature(resnet50).set_temperature(val_loader, device=device)

# Collect test logits & probs (before and after scaling)
resnet50.eval()
scaled_model.eval()
logits_list, labels_list = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = resnet50(inputs)
        logits_scaled = scaled_model.temperature_scale(logits)

        logits_list.append(logits)
        labels_list.append(labels)

logits = torch.cat(logits_list)
labels = torch.cat(labels_list)
probs = torch.softmax(logits, dim=1)
probs_scaled = torch.softmax(scaled_model.temperature_scale(logits), dim=1)

# Brier Score
brier_before = brier_score(probs.cpu().numpy(), labels.cpu().numpy())
brier_after = brier_score(probs_scaled.detach().cpu().numpy(), labels.detach().cpu().numpy())
print(f"Brier Score before scaling: {brier_before:.4f}, after: {brier_after:.4f}")

# Compute ECE, MCE, Class-wise ECE
ece_before = expected_calibration_error(probs, labels)
mce_before = maximum_calibration_error(probs, labels)
class_ece_before = classwise_ece(probs, labels, num_classes=5)

ece_after = expected_calibration_error(probs_scaled, labels)
mce_after = maximum_calibration_error(probs_scaled, labels)
class_ece_after = classwise_ece(probs_scaled, labels, num_classes=5)

print("ECE before:", ece_before, "after:", ece_after)
print("MCE before:", mce_before, "after:", mce_after)
print("Class-wise ECE before:", class_ece_before)
print("Class-wise ECE after:", class_ece_after)

# Reliability Diagram
plot_reliability_diagram(probs, labels)
plot_reliability_diagram(probs_scaled, labels)


def plot_classwise_reliability(probs, labels, class_names, n_bins=10, title=""):
    """
    probs: torch.tensor [N, C] (softmax probabilities)
    labels: torch.tensor [N]
    class_names: list of class names, length C
    """

    probs = probs.detach().cpu()
    labels = labels.detach().cpu()
    num_classes = probs.shape[1]

    fig, axes = plt.subplots(1, num_classes, figsize=(4*num_classes, 4), sharey=True)

    for c in range(num_classes):
        ax = axes[c]
        confs = probs[:, c]
        truths = (labels == c).float()

        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        bin_accs, bin_confs = [], []

        for i in range(n_bins):
            start, end = bin_boundaries[i], bin_boundaries[i+1]
            mask = (confs > start) & (confs <= end)
            if mask.sum() > 0:
                bin_acc = truths[mask].mean().item()
                bin_conf = confs[mask].mean().item()
                bin_accs.append(bin_acc)
                bin_confs.append(bin_conf)

        ax.plot([0,1],[0,1],"k--")
        ax.plot(bin_confs, bin_accs, marker="o", label=class_names[c])
        ax.set_title(class_names[c])
        ax.set_xlabel("Confidence")
        if c == 0:
            ax.set_ylabel("Accuracy")
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])

    plt.suptitle(f"Class-wise Reliability Diagrams: {title}")
    plt.tight_layout()
    plt.show()


class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

# Before scaling
plot_classwise_reliability(probs, labels, class_names, title="Before Temperature Scaling")

# After scaling
plot_classwise_reliability(probs_scaled, labels, class_names, title="After Temperature Scaling")


def plot_combined_calibration_all(probs, probs_scaled, labels, class_names, n_bins=10):
    """
    Combined calibration plot:
    - Top: global reliability before & after scaling
    - Bottom: class-wise reliability before & after scaling
    """
    probs = probs.detach().cpu()
    probs_scaled = probs_scaled.detach().cpu()
    labels = labels.detach().cpu()
    num_classes = len(class_names)

    # --- Helper for bin stats ---
    def compute_bin_stats_from_confidences(confs, truths, n_bins):
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        bin_accs, bin_confs = [], []
        for i in range(n_bins):
            start, end = bin_boundaries[i], bin_boundaries[i+1]
            mask = (confs > start) & (confs <= end)
            if mask.sum() > 0:
                bin_accs.append(truths[mask].float().mean().item())
                bin_confs.append(confs[mask].mean().item())
        return bin_confs, bin_accs

    def compute_global_bin_stats(probs, labels, n_bins):
        confidences, predictions = probs.max(dim=1)
        accuracies = predictions.eq(labels)
        return compute_bin_stats_from_confidences(confidences, accuracies, n_bins)

    # --- Global stats ---
    bin_confs_before, bin_accs_before = compute_global_bin_stats(probs, labels, n_bins)
    bin_confs_after, bin_accs_after = compute_global_bin_stats(probs_scaled, labels, n_bins)

    fig = plt.figure(figsize=(16, 10))

    # --- Top: Global reliability ---
    ax1 = plt.subplot2grid((2, num_classes), (0, 0), colspan=num_classes)
    ax1.plot([0,1],[0,1],"k--")
    ax1.plot(bin_confs_before, bin_accs_before, marker="o", label="Before Scaling")
    ax1.plot(bin_confs_after, bin_accs_after, marker="o", label="After Scaling")
    ax1.set_title("Global Reliability Diagram")
    ax1.set_xlabel("Predicted Confidence")
    ax1.set_ylabel("True Accuracy")
    ax1.set_xlim([0,1]); ax1.set_ylim([0,1])
    ax1.legend()

    # --- Bottom: Class-wise reliability before & after ---
    for c in range(num_classes):
        ax = plt.subplot2grid((2, num_classes), (1, c))

        # Before scaling
        confs_before = probs[:, c]
        truths = (labels == c).float()
        bin_confs_b, bin_accs_b = compute_bin_stats_from_confidences(confs_before, truths, n_bins)

        # After scaling
        confs_after = probs_scaled[:, c]
        bin_confs_a, bin_accs_a = compute_bin_stats_from_confidences(confs_after, truths, n_bins)

        # Plot both
        ax.plot([0,1],[0,1],"k--")
        ax.plot(bin_confs_b, bin_accs_b, marker="o", label="Before")
        ax.plot(bin_confs_a, bin_accs_a, marker="o", label="After")
        ax.set_title(class_names[c])
        ax.set_xlim([0,1]); ax.set_ylim([0,1])
        ax.set_xlabel("Confidence")
        if c == 0:
            ax.set_ylabel("True Accuracy")
        if c == num_classes-1:
            ax.legend()

    plt.suptitle("Calibration: Global and Class-wise (Before vs After Scaling)", fontsize=14)
    plt.tight_layout()
    plt.show()


class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

plot_combined_calibration_all(probs, probs_scaled, labels, class_names, n_bins=10)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        # placeholders for gradients and activations
        self.gradients = None
        self.activations = None

        # hook for forward
        def forward_hook(module, input, output):
            self.activations = output.detach()
        # hook for backward
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # register hooks
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        """
        input_tensor: (1, C, H, W) torch tensor
        target_class: int, class index (if None, uses predicted class)
        """
        # forward pass
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # backward pass
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        # compute weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP over H,W
        cam = (weights * self.activations).sum(dim=1).squeeze()

        # relu + normalize
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.cpu().numpy()
        return cam


# Example: setup Grad-CAM
target_layer = resnet50.layer4[-1]
gradcam = GradCAM(resnet50, target_layer)

# pick a test image
img, label = test_dataset[0]   # returns preprocessed tensor + label
input_tensor = img.unsqueeze(0).to(device)

# generate cam
cam = gradcam.generate_cam(input_tensor, target_class=None)

# convert tensor to displayable image
def tensor_to_image(tensor):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).astype(np.uint8)

orig_img = tensor_to_image(img)

# resize CAM to match image
H, W, _ = orig_img.shape
cam_resized = cv2.resize(cam, (W, H))

# heatmap + overlay
heatmap = cv2.applyColorMap(np.uint8(255*cam_resized), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
overlay = cv2.addWeighted(orig_img, 0.5, heatmap, 0.5, 0)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(orig_img); plt.title("Original"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(cam_resized, cmap='jet'); plt.title("Grad-CAM Map"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(overlay); plt.title("Overlay"); plt.axis("off")
plt.show()

