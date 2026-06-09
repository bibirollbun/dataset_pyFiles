import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.parallel import DataParallel
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import defaultdict
import random
from tqdm import tqdm
import json

import copy

def serialize_wrong_predictions(wrong_preds):
    """Convert tuple keys to string keys for JSON serialization"""
    serialized = {}
    for (true_label, pred_label), count in wrong_preds.items():
        key = f"{true_label}_{pred_label}"  # Convert tuple to string
        serialized[key] = count
    return serialized

def deserialize_wrong_predictions(serialized_wrong_preds):
    """Convert string keys back to tuple keys"""
    deserialized = {}
    for key, count in serialized_wrong_preds.items():
        true_label, pred_label = map(int, key.split('_'))
        deserialized[(true_label, pred_label)] = count
    return deserialized


class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )
    
    def forward(self, x):
        return self.net(x)

class BYOL(nn.Module):
    def __init__(self, net, image_size, hidden_layer='avgpool', projection_size=256, 
                 projection_hidden_size=4096, moving_average_decay=0.996):
        super().__init__()
        self.moving_average_decay = moving_average_decay
        
        # Handle different encoder types
        if hasattr(net, 'feature_dim'):
            feature_dim = net.feature_dim
            self.online_encoder = net
        else:
            # For standard models (ResNet, etc.)
            if hasattr(net, 'fc'):
                feature_dim = net.fc.in_features
                net.fc = nn.Identity()
            else:
                feature_dim = 512  # fallback
            self.online_encoder = net
        
        # Online network components
        self.online_projector = MLP(feature_dim, projection_size, projection_hidden_size)
        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)
        
        # Target network - create copies
        self.target_encoder = None
        self.target_projector = None
        self.update_target_network(tau=1.0)  # Initialize target network
    
    @torch.no_grad()
    def update_target_network(self, tau=None):
        """
        Update target network using exponential moving average.
        If tau=1.0, this initializes the target network.
        """
        if tau is None:
            tau = self.moving_average_decay
            
        if self.target_encoder is None:
            # Initialize target encoder and projector
            self.target_encoder = copy.deepcopy(self.online_encoder)
            self.target_projector = copy.deepcopy(self.online_projector)
            
            # Freeze target network parameters
            for param in self.target_encoder.parameters():
                param.requires_grad = False
            for param in self.target_projector.parameters():
                param.requires_grad = False
            
            # Set BatchNorm layers in target network to eval mode
            for module in self.target_encoder.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    module.eval()
            for module in self.target_projector.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    module.eval()
        else:
            # EMA update
            for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
                target_params.data = tau * target_params.data + (1 - tau) * online_params.data
            
            for online_params, target_params in zip(self.online_projector.parameters(), self.target_projector.parameters()):
                target_params.data = tau * target_params.data + (1 - tau) * online_params.data
    
    def update_moving_average(self):
        """
        Update the target network using exponential moving average.
        This method provides compatibility with the chat implementation.
        """
        self.update_target_network()
    
    def _loss_fn(self, x, y):
        """
        BYOL loss function: negative cosine similarity
        """
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        # Calculate cosine similarity and return scalar
        cosine_sim = (x * y).sum(dim=-1)  # Shape: (batch_size,)
        loss = 2 - 2 * cosine_sim  # Shape: (batch_size,)
        return loss.mean()  # Return scalar by taking mean across batch
    
    def forward(self, im_1, im_2):
        """
        Forward pass for BYOL training.
        Can return either loss (chat implementation style) or components (pasted implementation style).
        """
        # Online network forward pass
        f_o1 = self.online_encoder(im_1)
        f_o2 = self.online_encoder(im_2)
        
        z_o1 = self.online_projector(f_o1)
        z_o2 = self.online_projector(f_o2)
        
        p_o1 = self.online_predictor(z_o1)
        p_o2 = self.online_predictor(z_o2)
        
        # Target network forward pass (no gradient)
        with torch.no_grad():
            f_t1 = self.target_encoder(im_1)
            f_t2 = self.target_encoder(im_2)
            
            z_t1 = self.target_projector(f_t1)
            z_t2 = self.target_projector(f_t2)
        
        # Calculate loss for compatibility with chat implementation
        loss_1 = self._loss_fn(p_o1, z_t2.detach())
        loss_2 = self._loss_fn(p_o2, z_t1.detach())
        loss = (loss_1 + loss_2) / 2.0
        
        # Return loss for compatibility with chat implementation
        return loss
    
    def forward_features(self, x1, x2):
        """
        Forward pass that returns all components (original pasted implementation style).
        Use this for more detailed analysis or custom loss computation.
        """
        # Online network forward pass
        f_o1 = self.online_encoder(x1)
        f_o2 = self.online_encoder(x2)
        
        z_o1 = self.online_projector(f_o1)
        z_o2 = self.online_projector(f_o2)
        
        p_o1 = self.online_predictor(z_o1)
        p_o2 = self.online_predictor(z_o2)
        
        # Target network forward pass (no gradient)
        with torch.no_grad():
            f_t1 = self.target_encoder(x1)
            f_t2 = self.target_encoder(x2)
            
            z_t1 = self.target_projector(f_t1)
            z_t2 = self.target_projector(f_t2)
        
        return p_o1, p_o2, z_t1.detach(), z_t2.detach(), f_o1, f_o2
    
    def get_features(self, x):
        """
        Extract features from the encoder (useful for downstream tasks).
        """
        return self.online_encoder(x)


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Number of GPUs available: {torch.cuda.device_count()}")

# Create directories for outputs
os.makedirs('/kaggle/working/models', exist_ok=True)
os.makedirs('/kaggle/working/plots', exist_ok=True)
os.makedirs('/kaggle/working/metrics', exist_ok=True)


class BYOLTransform:
    def __init__(self, size=224):
        def get_transform():
            return transforms.Compose([
                transforms.RandomResizedCrop(size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.transform1 = get_transform()
        self.transform2 = get_transform()

    def __call__(self, x):
        return self.transform1(x), self.transform2(x)

def get_standard_transforms(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class AffectNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        # Define AffectNet emotion classes
        # 0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt
        emotion_classes = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
        
        # Find all image paths and assign labels
        classes = sorted(os.listdir(root_dir))
        for i, class_name in enumerate(classes):
            if class_name in emotion_classes or class_name.isdigit():
                self.class_to_idx[class_name] = i
                class_dir = os.path.join(root_dir, class_name)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append(os.path.join(class_dir, img_name))
                            self.labels.append(i)
        
        print(f"Found {len(self.class_to_idx)} classes: {list(self.class_to_idx.keys())}")
        print(f"Total images: {len(self.image_paths)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class RAFDBDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        # Find all image paths and assign labels
        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        for i, class_name in enumerate(classes):
            self.class_to_idx[class_name] = i
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(i)
        
        print(f"RAF-DB Found {len(self.class_to_idx)} classes: {list(self.class_to_idx.keys())}")
        print(f"RAF-DB Total images: {len(self.image_paths)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def train_byol(model, train_loader, optimizer, scheduler, epochs, device):
    model.train()
    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler()
    
    train_losses = []
    train_accuracies = []
    learning_rates = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        total_cosine_sim = 0.0
        num_batches = 0
        
        for i, (images, _) in enumerate(train_loader):
            try:
                # Get the two views of each image
                if isinstance(images, list):
                    im_1 = images[0].to(device, non_blocking=True)
                    im_2 = images[1].to(device, non_blocking=True)
                else:
                    im_1 = images.to(device, non_blocking=True)
                    im_2 = images.to(device, non_blocking=True)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass with mixed precision
                with autocast():
                    loss = model(im_1, im_2)
                    
                    # Ensure loss is a scalar
                    if loss.dim() > 0:
                        loss = loss.mean()
                    
                    # Get representations for accuracy calculation
                    with torch.no_grad():
                        # Handle DataParallel wrapper
                        if isinstance(model, torch.nn.DataParallel):
                            repr_1 = model.module.online_encoder(im_1)
                            repr_2 = model.module.online_encoder(im_2)
                        else:
                            repr_1 = model.online_encoder(im_1)
                            repr_2 = model.online_encoder(im_2)
                        
                        # Calculate cosine similarity for accuracy metric
                        repr_1_norm = F.normalize(repr_1, dim=-1, p=2)
                        repr_2_norm = F.normalize(repr_2, dim=-1, p=2)
                        cosine_sim = (repr_1_norm * repr_2_norm).sum(dim=-1).mean()
                        total_cosine_sim += cosine_sim.item()
                        num_batches += 1
                
                # Backward pass with mixed precision
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Update target network
                if isinstance(model, torch.nn.DataParallel):
                    model.module.update_moving_average()
                else:
                    model.update_moving_average()
                    
                # Update statistics
                running_loss += loss.item()
                
                # Print progress every 100 batches
                if i % 100 == 0:
                    print(f'Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}')
                    
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                continue
        
        # Step the scheduler after each epoch
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = (total_cosine_sim / num_batches) * 100 if num_batches > 0 else 0
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, LR: {current_lr:.6f}")
    
    return train_losses, train_accuracies, learning_rates


class FinetuneModel(nn.Module):
    def __init__(self, byol_model, num_classes):
        super(FinetuneModel, self).__init__()
        self.encoder = byol_model.online_encoder
        
        # Get the feature dimension from the encoder
        # Handle case where fc layer is already replaced with Identity
        if hasattr(self.encoder, 'fc') and hasattr(self.encoder.fc, 'in_features'):
            feature_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        else:
            feature_dim = self._get_feature_dim()
            
            # Ensure fc layer is Identity if it's not already
            if hasattr(self.encoder, 'fc') and not isinstance(self.encoder.fc, nn.Identity):
                self.encoder.fc = nn.Identity()
        
        self.fc = nn.Linear(feature_dim, num_classes)
    
    def _get_feature_dim(self):
        """Dynamically determine the feature dimension by doing a forward pass"""
        # Create a dummy input to determine feature dimension
        dummy_input = torch.randn(1, 3, 224, 224)
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
            self.encoder = self.encoder.cuda()
        
        self.encoder.eval()
        with torch.no_grad():
            features = self.encoder(dummy_input)
            feature_dim = features.shape[1]
        
        return feature_dim
    
    def forward(self, x):
        features = self.encoder(x)
        return self.fc(features)

def finetune_model(byol_model, train_loader, val_loader, num_classes, epochs, device, lr=0.001):
    model = FinetuneModel(byol_model, num_classes)
    
    # Use DataParallel for multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for fine-tuning")
        model = DataParallel(model)
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    
    # Training metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    wrong_preds = defaultdict(int)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        wrong_preds.clear()
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                # Forward pass
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                
                # Update statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Track wrong predictions
                for true, pred in zip(targets.cpu().numpy(), predicted.cpu().numpy()):
                    if true != pred:
                        wrong_preds[(true, pred)] += 1
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    return model, train_losses, train_accs, val_losses, val_accs, wrong_preds


def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            with autocast():
                outputs = model(images)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Track wrong predictions
    wrong_preds = defaultdict(int)
    for true, pred in zip(all_targets, all_preds):
        if true != pred:
            wrong_preds[(true, pred)] += 1
    
    return cm, wrong_preds


def plot_loss(losses, title="Training Loss", save_path="training_loss.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_accuracy(accuracies, title="Training Accuracy", save_path="training_accuracy.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(accuracies)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_finetune_metrics(train_losses, val_losses, train_accs, val_accs, save_path="finetune_metrics.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_wrong_predictions(wrong_preds, num_classes, class_names=None, title="Wrong Predictions Distribution", save_path="wrong_preds.png"):
    # Convert wrong_preds to a more plottable format
    true_classes = []
    pred_classes = []
    counts = []
    
    for (true, pred), count in wrong_preds.items():
        true_classes.append(true)
        pred_classes.append(pred)
        counts.append(count)
    
    plt.figure(figsize=(12, 10))
    cm = np.zeros((num_classes, num_classes))
    
    for true, pred, count in zip(true_classes, pred_classes, counts):
        cm[true, pred] = count
    
    # Zero out the diagonal (correct predictions)
    np.fill_diagonal(cm, 0)
    
    # Plot confusion matrix
    ax = sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    
    if class_names:
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names, rotation=0)
    
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # Define paths
    rafdb_train_dir = '/kaggle/input/raf-db-dataset/DATASET/train'
    rafdb_test_dir = '/kaggle/input/raf-db-dataset/DATASET/test'
    
    # Initialize ResNet34 backbone
    resnet34 = models.resnet34(pretrained=True)
    
    # Initialize BYOL model (but we'll only use the encoder part)
    byol_model = BYOL(
        resnet34,
        image_size=224,
        hidden_layer='avgpool',
        projection_size=256,
        projection_hidden_size=4096,
        moving_average_decay=0.996
    )
    
    byol_model = byol_model.to(device)
    
    print("Starting fine-tuning on RAF-DB with BYOL ResNet34 structure...")
    
    # Load RAF-DB dataset for fine-tuning
    standard_transform = get_standard_transforms()
    rafdb_dataset = RAFDBDataset(rafdb_train_dir, transform=standard_transform)
    
    # Split RAF-DB into train (85%) and val (15%)
    train_size = int(0.85 * len(rafdb_dataset))
    val_size = len(rafdb_dataset) - train_size
    train_dataset, val_dataset = random_split(rafdb_dataset, [train_size, val_size])
    
    # Create data loaders for fine-tuning
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=16, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=16, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    print(f"RAF-DB dataset size: {len(rafdb_dataset)}")
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Save RAF-DB class mapping
    with open('/kaggle/working/metrics/rafdb_class_mapping.json', 'w') as f:
        json.dump(rafdb_dataset.class_to_idx, f)
    
    # Fine-tune on RAF-DB
    num_classes = len(rafdb_dataset.class_to_idx)
    
    finetuned_model, train_losses, train_accs, val_losses, val_accs, wrong_preds = finetune_model(
        byol_model, 
        train_loader, 
        val_loader, 
        num_classes, 
        epochs=50, 
        device=device
    )
    
    # Save fine-tuned model
    torch.save(finetuned_model.state_dict(), '/kaggle/working/models/finetuned_rafdb_model.pth')
    
    # Save fine-tuning metrics
    finetune_metrics = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'wrong_predictions': serialize_wrong_predictions(wrong_preds)
    }

    with open('/kaggle/working/metrics/finetune_metrics.json', 'w') as f:
        json.dump(finetune_metrics, f)
    
    # Plot fine-tuning metrics
    plot_finetune_metrics(train_losses, val_losses, train_accs, val_accs, 
                         save_path="/kaggle/working/plots/finetune_metrics.png")
    
    # Plot wrong predictions
    plot_wrong_predictions(wrong_preds, num_classes, 
                          class_names=list(rafdb_dataset.class_to_idx.keys()),
                          title="Wrong Predictions Distribution - RAF-DB",
                          save_path="/kaggle/working/plots/wrong_predictions_rafdb.png")
    
    print("Fine-tuning complete!")
    print(f"Best validation accuracy: {max(val_accs):.2f}%")
    
    print("\nEvaluating on RAF-DB test set...")
    
    # Load RAF-DB test dataset
    rafdb_test_dataset = RAFDBDataset(rafdb_test_dir, transform=standard_transform)
    
    # Create test data loader
    test_loader = DataLoader(
        rafdb_test_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=16, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    print(f"RAF-DB test dataset size: {len(rafdb_test_dataset)}")
    
    # Evaluate on test set
    test_cm, test_wrong_preds = evaluate_model(finetuned_model, test_loader, device)
    
    # Calculate test accuracy
    test_accuracy = np.trace(test_cm) / np.sum(test_cm) * 100
    print(f"Test accuracy: {test_accuracy:.2f}%")
    
    # Save test results
    test_results = {
        'test_accuracy': test_accuracy,
        'confusion_matrix': test_cm.tolist(),
        'wrong_predictions': serialize_wrong_predictions(test_wrong_preds)
    }
    with open('/kaggle/working/metrics/test_results.json', 'w') as f:
        json.dump(test_results, f)
    
    # Plot test confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(rafdb_dataset.class_to_idx.keys()),
                yticklabels=list(rafdb_dataset.class_to_idx.keys()))
    plt.title(f'Test Confusion Matrix - RAF-DB (Accuracy: {test_accuracy:.2f}%)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('/kaggle/working/plots/test_confusion_matrix.png')
    plt.close()
    
    # Plot test wrong predictions
    plot_wrong_predictions(test_wrong_preds, num_classes, 
                          class_names=list(rafdb_dataset.class_to_idx.keys()),
                          title="Wrong Predictions Distribution - RAF-DB Test Set",
                          save_path="/kaggle/working/plots/wrong_predictions_test.png")
    
    print(f"\nFinal Results:")
    print(f"Best Validation Accuracy: {max(val_accs):.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
if __name__ == "__main__":
    main()

