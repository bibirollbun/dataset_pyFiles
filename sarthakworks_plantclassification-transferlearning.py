import os
import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms(model_type, train=True):
    """Return image transforms based on the model type and training mode."""
    input_size = 299 if model_type.lower() == "inception_v3" else 224
    if train:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    return transform


def load_data(data_dir, transform, train_split=0.8, batch_size=32, num_workers=2):
    """Load the dataset, split into train/validation, and create DataLoaders."""
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, dataset


def get_model(model_name, num_classes, device):
    """Instantiate and modify a pre-trained model to match the number of classes."""
    model_name = model_name.lower()
    if model_name == "resnet50":
        model = models.resnet50(weights="DEFAULT")
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(weights="DEFAULT")
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "inception_v3":
        model = models.inception_v3(weights="DEFAULT")
        model.aux_logits = False  # Disable auxiliary output for simplicity
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    return model.to(device)


def train_model(model, optimizer, criterion, train_loader, val_loader, num_epochs, device, scheduler=None):
    """
    Train the model with optional mixed precision and learning rate scheduler.
    Returns a dictionary containing the loss and accuracy metrics.
    """
    metrics = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }
    
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False)
        for inputs, labels in train_progress:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            train_progress.set_postfix(loss=f"{loss.item():.4f}")

        epoch_train_loss = running_loss / total_train
        train_accuracy = correct_train / total_train
        metrics["train_loss"].append(epoch_train_loss)
        metrics["train_accuracy"].append(train_accuracy)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=False)
        with torch.no_grad():
            for inputs, labels in val_progress:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                val_progress.set_postfix(loss=f"{loss.item():.4f}")
        epoch_val_loss = running_val_loss / total_val
        val_accuracy = correct_val / total_val
        metrics["val_loss"].append(epoch_val_loss)
        metrics["val_accuracy"].append(val_accuracy)

        # Update learning rate scheduler if provided
        if scheduler:
            scheduler.step()

    return metrics


def evaluate_model(model, loader, device):
    """Evaluate the model and return ground truth labels and predictions."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, classes, title, save_fig=False, filename=None):
    """Plot and optionally save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    if save_fig and filename:
        plt.savefig(filename)
    plt.show()


def plot_all_models_metrics(metrics_dict, model_names):
    """
    Plot combined performance graphs (loss and accuracy curves) for multiple models.
    metrics_dict is a dictionary mapping model names to their metrics dictionary.
    """
    epochs = range(1, len(metrics_dict[model_names[0]]["train_loss"]) + 1)
    plt.figure(figsize=(14, 6))
    
    # Loss Curves
    plt.subplot(1, 2, 1)
    for model_name in model_names:
        m = metrics_dict[model_name]
        plt.plot(epochs, m["train_loss"], marker='o', label=f"{model_name} Train Loss")
        plt.plot(epochs, m["val_loss"], marker='x', label=f"{model_name} Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves for All Models")
    plt.legend()
    plt.grid(True)
    
    # Accuracy Curves
    plt.subplot(1, 2, 2)
    for model_name in model_names:
        m = metrics_dict[model_name]
        plt.plot(epochs, m["train_accuracy"], marker='o', label=f"{model_name} Train Accuracy")
        plt.plot(epochs, m["val_accuracy"], marker='x', label=f"{model_name} Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves for All Models")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


# Set seed for reproducibility
set_seed(42)


# Configuration parameters
base_path = '/kaggle/input/plant-seedlings-classification'
train_dir = os.path.join(base_path, 'train')
batch_size = 32
num_workers = 2
num_epochs = 10
learning_rate = 1e-4
train_split = 0.8


# Use transforms for models expecting 224x224 inputs
basic_transform = get_transforms(model_type="resnet50", train=True)
train_loader, val_loader, full_dataset = load_data(
    train_dir, basic_transform, train_split, batch_size, num_workers)
    
# Compute class weights
all_labels = [label for _, label in full_dataset.samples]
class_counts = Counter(all_labels)
num_classes = len(full_dataset.classes)
total_samples = len(full_dataset)
weights_per_class = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]
class_weights = torch.FloatTensor(weights_per_class)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
    
# Dictionary to hold performance metrics for each model
all_metrics = {}


# --- ResNet50 ---
print("Training ResNet50...")
model_resnet = get_model("resnet50", num_classes, device)
optimizer_resnet = optim.Adam(model_resnet.parameters(), lr=learning_rate)
scheduler_resnet = optim.lr_scheduler.StepLR(optimizer_resnet, step_size=5, gamma=0.1)
    
metrics_resnet = train_model(model_resnet, optimizer_resnet, criterion, 
                             train_loader, val_loader, num_epochs, device, scheduler_resnet)
all_metrics["ResNet50"] = metrics_resnet
    
y_true, y_pred = evaluate_model(model_resnet, val_loader, device)
print("ResNet50 Classification Report:")
print(classification_report(y_true, y_pred, target_names=full_dataset.classes))
plot_confusion_matrix(y_true, y_pred, full_dataset.classes, "ResNet50 Confusion Matrix")
    
resnet_save_path = "resnet50_best_model.pth"
torch.save(model_resnet.state_dict(), resnet_save_path)
print(f"ResNet50 model saved to {resnet_save_path}")


# --- VGG16 ---
print("Training VGG16...")
model_vgg = get_model("vgg16", num_classes, device)
optimizer_vgg = optim.Adam(model_vgg.parameters(), lr=learning_rate)
scheduler_vgg = optim.lr_scheduler.StepLR(optimizer_vgg, step_size=5, gamma=0.1)
    
metrics_vgg = train_model(model_vgg, optimizer_vgg, criterion, 
                          train_loader, val_loader, num_epochs, device, scheduler_vgg)
all_metrics["VGG16"] = metrics_vgg
    
y_true, y_pred = evaluate_model(model_vgg, val_loader, device)
print("VGG16 Classification Report:")
print(classification_report(y_true, y_pred, target_names=full_dataset.classes))
plot_confusion_matrix(y_true, y_pred, full_dataset.classes, "VGG16 Confusion Matrix")
    
vgg_save_path = "vgg16_best_model.pth"
torch.save(model_vgg.state_dict(), vgg_save_path)
print(f"VGG16 model saved to {vgg_save_path}")


# --- InceptionV3 ---
print("Training InceptionV3...")
# InceptionV3 requires 299x299 inputs; load a separate dataset using the appropriate transforms
inception_transform = get_transforms(model_type="inception_v3", train=True)
inception_train_loader, inception_val_loader, _ = load_data(
    train_dir, inception_transform, train_split, batch_size, num_workers)
model_inception = get_model("inception_v3", num_classes, device)
optimizer_inception = optim.Adam(model_inception.parameters(), lr=learning_rate)
scheduler_inception = optim.lr_scheduler.StepLR(optimizer_inception, step_size=5, gamma=0.1)
    
metrics_inception = train_model(model_inception, optimizer_inception, criterion, 
                                inception_train_loader, inception_val_loader, num_epochs, device, scheduler_inception)
all_metrics["InceptionV3"] = metrics_inception
    
y_true, y_pred = evaluate_model(model_inception, inception_val_loader, device)
print("InceptionV3 Classification Report:")
print(classification_report(y_true, y_pred, target_names=full_dataset.classes))
plot_confusion_matrix(y_true, y_pred, full_dataset.classes, "InceptionV3 Confusion Matrix")
    
inception_save_path = "inception_v3_best_model.pth"
torch.save(model_inception.state_dict(), inception_save_path)
print(f"InceptionV3 model saved to {inception_save_path}")


model_names = list(all_metrics.keys())
plot_all_models_metrics(all_metrics, model_names)




