!tar -xvzf /kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013.tar.gz -C /kaggle/working


import os
import random
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, vgg16
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models.vgg import VGG16_Weights
import copy
import numpy as np
import random
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


import torch
import matplotlib.pyplot as plt

def test_dataloader_shapes(dataloader, expected_shape=(1, 128, 128), num_classes=5):
    images, labels = next(iter(dataloader))
    assert isinstance(images, torch.Tensor), "Images are not torch tensors."
    assert isinstance(labels, torch.Tensor), "Labels are not torch tensors."

    assert images.shape[1:] == expected_shape, f"Expected shape {expected_shape}, got {images.shape[1:]}"
    assert labels.max().item() < num_classes, f"Label exceeds expected range {num_classes - 1}"
    print(f"âœ… Batch shape: {images.shape}, Labels: {labels.tolist()}")


def test_data_module(dm):
    dm.setup()
    
    print("\nTesting Train DataLoader:")
    test_dataloader_shapes(dm.train_dataloader())
    
    print("\nTesting Validation DataLoader:")
    test_dataloader_shapes(dm.val_dataloader())
    
    print("\nTesting Test DataLoader:")
    test_dataloader_shapes(dm.test_dataloader())

    print("\nClass names:", dm.get_class_names())
    print("Class weights:", dm.get_class_weights())


def plot_samples_from_dataloader(dataloader, class_map, n=16):
    """
    Plot a grid of n samples from the dataloader with their labels.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    images, labels = next(iter(dataloader))
    images = images[:n]
    labels = labels[:n]

    # Unnormalize: (img * std + mean)
    images = images * 0.5 + 0.5

    n_cols = int(np.sqrt(n))
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
    axes = axes.flatten()

    for i in range(n):
        img = images[i].squeeze().cpu().numpy()  # shape (128, 128)
        label = labels[i].item()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(class_map.get(label, str(label)))
        axes[i].axis('off')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

class FERPlusDataset(Dataset):
    def __init__(self, csv_path, usage='Training', transform=None, emotion_map=None):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data['Usage'] == usage].reset_index(drop=True)

        # Drop 'Angry' (0) and 'Disgust' (1)
        self.data = self.data[~self.data['emotion'].isin([0, 1])].reset_index(drop=True)

        self.transform = transform

        # Remap remaining labels to 0...n-1
        original_to_new = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4}
        self.data['emotion'] = self.data['emotion'].map(original_to_new)

        self.emotion_map = {
            0: 'Fear', 1: 'Happy', 2: 'Sad', 3: 'Surprise', 4: 'Neutral'
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = int(self.data.iloc[idx]['emotion'])

        pixel_str = self.data.iloc[idx]['pixels']
        pixels = np.array(list(map(int, pixel_str.split())), dtype=np.uint8).reshape(48, 48)
        image = Image.fromarray(pixels)

        if self.transform:
            image = self.transform(image)

        return image, label


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
import random
import numpy as np
import os

class Utils:
    @staticmethod
    def _maybe_save(fig, filename, save):
        if save:
            os.makedirs(Config().CHECKPOINT_PATH, exist_ok=True)
            fig.savefig(os.path.join(Config().CHECKPOINT_PATH, filename))
        plt.close(fig)

    @staticmethod
    def plot_loss_curve(train_loss, val_loss, show=True, save=True):
        """
        Plot training and validation loss curves.
        """
        fig = plt.figure(figsize=(10, 5))
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid(True)
        if show:
            plt.show()
        Utils._maybe_save(fig, 'loss_curve.png', save)

    @staticmethod
    def plot_accuracy_curve(train_acc, val_acc, show=True, save=True):
        """
        Plot training and validation accuracy curves.
        """
        fig = plt.figure(figsize=(10, 5))
        plt.plot(train_acc, label='Train Accuracy', marker='o')
        plt.plot(val_acc, label='Validation Accuracy', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.grid(True)
        if show:
            plt.show()
        Utils._maybe_save(fig, 'accuracy_curve.png', save)

    @staticmethod
    def plot_f1_score(f1_scores, show=True, save=True):
        """
        Plot validation F1 score curve.
        """
        fig = plt.figure(figsize=(10, 5))
        plt.plot(f1_scores, label='Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('F1 Score Curve')
        plt.legend()
        plt.grid(True)
        if show:
            plt.show()
        Utils._maybe_save(fig, 'f1_score_curve.png', save)

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names, show=True, save=True):
        """
        Plot confusion matrix and print classification report.
        """
        cm = confusion_matrix(y_true, y_pred)
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        if show:
            plt.show()
        Utils._maybe_save(fig, 'confusion_matrix.png', save)

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    @staticmethod
    def set_seed(seed=None, seed_torch=True):
        """
        Set random seed for reproducibility.
        """
        if seed is None:
            seed = np.random.choice(2 ** 32)
        random.seed(seed)
        np.random.seed(seed)
        if seed_torch:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        print(f'Random seed {seed} has been set for reproducibility.')



class FERDataModule:
    def __init__(self, csv_path, batch_size=32, image_size=128, criterion=None, train_transform=None, val_transform=None):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.criterion = criterion
        self.image_size = (image_size, image_size)
        self.emotion_map = {
            0: 'Fear', 1: 'Happy', 2: 'Sad', 3: 'Surprise', 4: 'Neutral'
        }

        # Transforms
        self.train_transform = train_transform or transforms.Compose([
            transforms.Resize((int(self.image_size[0] * 1.1), int(self.image_size[1] * 1.1))),
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])

        self.val_transform = val_transform or transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def setup(self):
        self.train_dataset = FERPlusDataset(self.csv_path, usage='Training',
                                            transform=self.train_transform)
        self.val_dataset = FERPlusDataset(self.csv_path, usage='PrivateTest',
                                          transform=self.val_transform)
        self.test_dataset = FERPlusDataset(self.csv_path, usage='PublicTest',
                                           transform=self.val_transform)

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")

    def train_dataloader(self):
        num_classes = len(self.get_class_names())

        if self.criterion == 'mixup':
            mixup_fn = v2.RandomChoice([
                v2.MixUp(alpha=0.4, num_classes=num_classes),
                v2.CutMix(alpha=0.4, num_classes=num_classes)
            ])
            def collate_fn(batch):
                return mixup_fn(*torch.utils.data.default_collate(batch))
        else:
            collate_fn = None

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    def get_class_names(self):
        return list(self.train_dataset.emotion_map.values())

    def get_class_weights(self):
        labels = [label for _, label in self.train_dataset]
        labels_tensor = torch.tensor(labels)
        counts = torch.bincount(labels_tensor, minlength=len(self.get_class_names()))
        total = len(labels_tensor)
        weights = total / (counts.float() * len(counts))
        return weights


csv_path = "/kaggle/working/fer2013/fer2013.csv"
dm = FERDataModule(csv_path, batch_size=32, image_size=128)

# Run tests
test_data_module(dm)

# Plot training samples
print("\nğŸ–¼ï¸� Plotting training samples...")
plot_samples_from_dataloader(dm.train_dataloader(), class_map=dm.emotion_map)

# Plot validation samples
print("\nğŸ–¼ï¸� Plotting validation samples...")
plot_samples_from_dataloader(dm.val_dataloader(), class_map=dm.emotion_map)


import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross Entropy Loss with Label Smoothing regularization."""
    
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, outputs, targets):
        log_probs = F.log_softmax(outputs, dim=1)
        num_classes = outputs.size(1)
        
        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


class WeightedLabelSmoothingCE(nn.Module):
    """Weighted Cross Entropy Loss with Label Smoothing."""
    
    def __init__(self, class_weights, smoothing=0.1, device='cuda'):
        super(WeightedLabelSmoothingCE, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.class_weights = class_weights.to(device)
        self.device = device

    def forward(self, outputs, targets):
        log_probs = F.log_softmax(outputs, dim=1)
        num_classes = outputs.size(1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), self.confidence)
        
        # Apply class weights
        weights = self.class_weights[targets]
        weighted_loss = -torch.sum(true_dist * log_probs, dim=1) * weights
        
        return torch.mean(weighted_loss)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha=1.0, gamma=2.0, smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, outputs, targets):
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Add label smoothing if specified
        if self.smoothing > 0:
            log_probs = F.log_softmax(outputs, dim=1)
            num_classes = outputs.size(1)
            
            with torch.no_grad():
                true_dist = torch.zeros_like(log_probs)
                true_dist.fill_(self.smoothing / (num_classes - 1))
                true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - self.smoothing)
            
            smooth_loss = torch.mean(torch.sum(-true_dist * log_probs, dim=1))
            focal_loss = focal_loss.mean() * 0.7 + smooth_loss * 0.3
        else:
            focal_loss = focal_loss.mean()
            
        return focal_loss


class MixUpLoss(nn.Module):
    """Enhanced MixUp Loss with label smoothing support."""
    
    def __init__(self, alpha=0.2, smoothing=0.1):
        super(MixUpLossEnhanced, self).__init__()
        self.alpha = alpha
        self.smoothing = smoothing
        if smoothing > 0:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, targets, targets_b=None, lam=1.0):
        if targets_b is None or lam == 1.0:
            return self.criterion(outputs, targets)
        
        # MixUp: combine two targets
        return lam * self.criterion(outputs, targets) + (1 - lam) * self.criterion(outputs, targets_b)


def get_criterion(config, class_weights=None, device='cuda'):
    """Factory function to create appropriate loss function."""
    
    smoothing = getattr(config, 'LABEL_SMOOTHING', 0.0)
    criterion_type = config.CRITERION
    
    if criterion_type == "cross_entropy":
        if smoothing > 0:
            return LabelSmoothingCrossEntropy(smoothing=smoothing)
        else:
            return nn.CrossEntropyLoss()
    
    elif criterion_type == "weighted_loss":
        if class_weights is None:
            raise ValueError("Class weights required for weighted loss")
        if smoothing > 0:
            return WeightedLabelSmoothingCE(class_weights, smoothing=smoothing, device=device)
        else:
            return nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    elif criterion_type == "focal_loss":
        return FocalLoss(alpha=1.0, gamma=2.0, smoothing=smoothing)
    
    elif criterion_type == "mixup":
        return MixUpLossEnhanced(alpha=0.2, smoothing=smoothing)
    
    else:
        raise ValueError(f"Unsupported criterion: {criterion_type}")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device, criterion, optimizer, scheduler, progressive_unfreezing_frequency, EARLY_STOP_PATIENCE=None):
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'learning_rate': []
        }
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.early_stop_patience = EARLY_STOP_PATIENCE if EARLY_STOP_PATIENCE is not None else self.config.EARLY_STOP_PATIENCE
        self.criterion = criterion
        self.progressive_unfreezing_frequency = progressive_unfreezing_frequency
        # Enhanced optimizer with better regularization
        self.optimizer = optimizer or optim.AdamW(  # AdamW instead of Adam for better regularization
            model.parameters(), 
            lr=self.config.LEARNING_RATE, 
            weight_decay=self.config.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Enhanced learning rate scheduler
        self.scheduler = scheduler or ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=7, 
            verbose=True,
            min_lr=getattr(self.config, 'MIN_LR', 1e-7)
        )
        
        # Add gradient clipping
        self.gradient_clipping = getattr(self.config, 'GRADIENT_CLIPPING', None)
        
        # Track overfitting metrics
        self.overfitting_threshold = 15.0  # If train_acc - val_acc > 15%, consider overfitting

    def train(self):
        best_f1 = 0.0
        best_epoch = 0
        best_acc = 0.0
        early_stop_counter = 0
        best_val_loss = float('inf')
        best_model = None

        for epoch in range(self.config.NUM_EPOCHS):
            if (epoch+1) % self.progressive_unfreezing_frequency == 0:
                self.model.progressively_unfreeze()
            # Training phase
            train_loss, train_acc = self._train_epoch(epoch)
            
            # Validation phase
            val_loss, val_acc, val_f1, all_preds, all_labels = self._validate_epoch(epoch)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)  # val_loss from validation phase
            else:
                self.scheduler.step()  # for StepLR, CosineAnnealingLR, etc.

            # Check for overfitting
            overfitting_gap = train_acc - val_acc
            is_overfitting = overfitting_gap > self.overfitting_threshold

            # Save best model based on F1 score
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch + 1
                best_acc = val_acc
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.config.BEST_MODEL_PATH)
                best_model = copy.deepcopy(self.model.state_dict())
                early_stop_counter = 0
                print(f"âœ… [Epoch {epoch+1}] Best model saved - Val F1: {best_f1:.4f}, Acc: {best_acc:.2f}%")
            else:
                early_stop_counter += 1

            # Enhanced early stopping conditions
            should_stop = False
            
            # Standard early stopping
            if early_stop_counter >= self.early_stop_patience:
                print(f"Early stopping: No improvement for {self.early_stop_patience} epochs")
                should_stop = True
            
            # Overfitting-based early stopping
            if is_overfitting and train_acc > 85 and epoch > 10:
                print(f"Early stopping: Severe overfitting detected (gap: {overfitting_gap:.1f}%)")
                should_stop = True
            
            # Loss explosion check
            if val_loss > 10.0 and epoch > 5:
                print(f"Early stopping: Loss explosion detected (val_loss: {val_loss:.4f})")
                should_stop = True
                
            if should_stop:
                break

            # Enhanced logging with overfitting detection
            overfitting_status = "ğŸ”¥ OVERFITTING" if is_overfitting else "âœ… Normal"
            lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f} | "
                  f"Gap: {overfitting_gap:.1f}% {overfitting_status} | LR: {lr:.2e}")

            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\n{'='*60}")
        print(f"ğŸ�† BEST MODEL SUMMARY")
        print(f"{'='*60}")
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Validation F1: {best_f1:.4f}")
        print(f"Best Validation Accuracy: {best_acc:.2f}%")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        print(f"Model saved at: {self.config.BEST_MODEL_PATH}")
        print(f"{'='*60}")

        return self.history, best_epoch, best_f1, best_acc, best_model

    def _train_epoch(self, epoch):
        """Training phase for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]") \
            if self.config.USE_TQDM else self.train_loader

        for inputs, labels in train_bar:
            if isinstance(labels, tuple):  # MixUp/CutMix case
                labels_a, labels_b, lam = labels
                inputs = inputs.to(self.device)
                labels_a = labels_a.to(self.device)
                labels_b = labels_b.to(self.device)
                labels = labels_a  # Use primary label for accuracy
            else:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            if isinstance(labels, tuple):
                loss = self.criterion(outputs, labels_a, labels_b, lam)
            else:
                loss = self.criterion(outputs, labels)

            loss.backward()
            
            # Apply gradient clipping if specified
            if self.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
            
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if self.config.USE_TQDM:
                train_bar.set_postfix(
                    loss=running_loss / total, 
                    acc=100. * correct / total,
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}"
                )

        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc

    def _validate_epoch(self, epoch):
        """Validation phase for one epoch."""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        val_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]") \
            if self.config.USE_TQDM else self.val_loader

        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                
                # Use appropriate loss calculation
                if hasattr(self.criterion, 'forward') and 'lam' in self.criterion.forward.__code__.co_varnames:
                    loss = self.criterion(outputs, labels, lam=1.0)  # Standard CE for validation
                else:
                    loss = self.criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                if self.config.USE_TQDM:
                    val_bar.set_postfix(
                        loss=val_loss / val_total, 
                        acc=100. * val_correct / val_total
                    )

        epoch_val_loss = val_loss / len(self.val_loader.dataset)
        epoch_val_acc = 100. * val_correct / val_total

        # Calculate F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )

        return epoch_val_loss, epoch_val_acc, f1, all_preds, all_labels


from sklearn.metrics import precision_recall_fscore_support

class Test:
    def __init__(self, model, test_loader, criterion, config):
        """
        Initialize the Test class for model evaluation.
        
        Args:
            model (nn.Module): Trained model.
            test_loader (DataLoader): Test DataLoader.
            criterion (nn.Module): Loss function.
            config (Config): Configuration object.
        """
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test(self, xai_interpreter=None, class_names=None):
        """
        Evaluate the model on the test dataset and generate XAI heatmaps.
        
        Args:
            xai_interpreter (XAIInterpreter, optional): XAI interpreter for heatmaps.
            class_names (list, optional): List of class names for visualization.
            
        Returns:
            tuple: (test_loss, test_acc, precision, recall, f1, all_preds, all_labels)
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        test_bar = tqdm(self.test_loader, desc="Testing") if self.config.USE_TQDM else self.test_loader
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_bar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                
                if isinstance(self.criterion, MixUpLoss):
                    loss = self.criterion(outputs, labels)  # Use default lam=1.0 for testing
                else:
                    loss = self.criterion(outputs, labels)
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                if self.config.USE_TQDM:
                    test_bar.set_postfix(loss=test_loss/total, acc=100.*correct/total)

        test_loss = test_loss / len(self.test_loader.dataset)
        test_acc = 100. * correct / total
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )

        print(f"\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Score: {f1:.4f}")

        if class_names:
            Utils.plot_confusion_matrix(all_labels, all_preds, class_names)

        return test_loss, test_acc, precision, recall, f1, all_preds, all_labels


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights

class FeatureExtractor(nn.Module):
    """
    Wraps any backbone model and adapts it to output a normalized embedding.
    """
    def __init__(self, backbone, embedding_dim=128):
        """
        Args:
            backbone (nn.Module): Backbone model with a feature extractor.
            embedding_dim (int): Size of the output embedding.
        """
        super(FeatureExtractor, self).__init__()

        # Remove classifier from backbone (assumes standard torchvision format)
        if hasattr(backbone, 'fc'):  # For ResNet-type models
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            in_features = backbone.fc.in_features
        elif hasattr(backbone, 'classifier'):  # For MobileNet, etc.
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            in_features = backbone.classifier[1].in_features
        else:
            raise ValueError("Unsupported backbone architecture")

        # New embedding layer
        self.embedding = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        """
        Forward pass to extract and normalize embeddings.
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalization

def load_backbone(model_name="resnet18", pretrained=True):
    """
    Dynamically load backbone architecture with optional pre-trained weights.
    
    Args:
        model_name (str): Backbone architecture (e.g., 'resnet18').
        pretrained (bool): Whether to load ImageNet-pretrained weights.

    Returns:
        nn.Module: Backbone model
    """
    if model_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        
        # Replace first conv layer for grayscale (1 channel)
        model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        return model
    elif model_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet50(weights=weights)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    else:
        raise NotImplementedError(f"Backbone '{model_name}' is not supported yet.")



class TripletNetwork(nn.Module):
    """
    Triplet network wrapper that outputs embeddings from a backbone model.
    """
    def __init__(self, model_name="resnet18", embedding_dim=128, pretrained=True):
        """
        Args:
            model_name (str): Name of the backbone model.
            embedding_dim (int): Size of the output feature embedding.
            pretrained (bool): Use pretrained weights or not.
        """
        super(TripletNetwork, self).__init__()

        # Load the backbone and wrap it in a feature extractor
        backbone = load_backbone(model_name, pretrained)
        self.embedding_model = FeatureExtractor(backbone, embedding_dim)

    def forward(self, x):
        """
        Forward pass through the embedding network.
        """
        return self.embedding_model(x)


import torch
import torch.nn as nn


def load_triplet_model(weights_path, model_name="resnet18", embedding_dim=128, pretrained=True, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TripletNetwork(model_name=model_name, embedding_dim=embedding_dim, pretrained=pretrained)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


class TripletClassifier(nn.Module):
    def __init__(self, 
                 weights_path=None,
                 num_classes=10,
                 embedding_dim=128,
                 freeze_until=None,
                 model_name="resnet18",
                 pretrained=True,
                 device=None,
                 state_dict=None):
        super(TripletClassifier, self).__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if state_dict is not None:
            self.backbone = TripletNetwork(model_name=model_name, embedding_dim=embedding_dim, pretrained=False)
        elif weights_path is not None:
            self.backbone = load_triplet_model(
                weights_path=weights_path,
                model_name=model_name,
                embedding_dim=embedding_dim,
                pretrained=pretrained,
                device=self.device
            )
        else:
            raise ValueError("Either 'weights_path' or 'state_dict' must be provided.")

        # Expose layers for progressive unfreezing
        self.feature_layers = list(self.backbone.embedding_model.features.children())

        # Freeze layers initially
        if freeze_until is not None:
            self.freeze_layers(freeze_until)
        self.currently_frozen_until = freeze_until or 0

        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)

        if state_dict is not None:
            self.load_state_dict(state_dict)

        self.to(self.device)

    def freeze_layers(self, freeze_until):
        for i, layer in enumerate(self.feature_layers):
            requires_grad = i >= freeze_until
            for param in layer.parameters():
                param.requires_grad = requires_grad

    def progressively_unfreeze(self, step=1):
        new_freeze_until = max(0, self.currently_frozen_until - step)
        self.freeze_layers(new_freeze_until)
        self.currently_frozen_until = new_freeze_until
        print(f"[INFO] Unfroze layers from {new_freeze_until} to {len(self.feature_layers)}")

    def forward(self, x):
        embedding = self.backbone(x)
        logits = self.classifier(embedding)
        return logits


import os

class Config:
    def __init__(self):
        self.DATASET_PATH = "/kaggle/working/fer2013"
        self.MODEL_NAME = "EfficientNet"
        self.NUM_CLASSES = 5
        self.BATCH_SIZE = 128 
        self.NUM_EPOCHS = 60
        self.LEARNING_RATE = 1e-3  
        self.CRITERION = 'cross_entropy'  # Start simple, then experiment
        self.SEED = 42
        self.CHECKPOINT_PATH = "/kaggle/working/checkpoints"
        self.BEST_MODEL_PATH = os.path.join(self.CHECKPOINT_PATH, "best_model.pth")
        self.FINAL_MODEL_PATH = os.path.join(self.CHECKPOINT_PATH, "final_model.pth")
        self.USE_TQDM = True
        self.EARLY_STOP_PATIENCE = 10  # More aggressive early stopping
        self.WEIGHT_DECAY = 1e-3  # Increased L2 regularization
        self.IMAGE_SIZE = (128, 128)
        
        # New anti-overfitting parameters
        self.DROPOUT_RATE = 0.6  # Increased dropout
        self.LABEL_SMOOTHING = 0.1  # Add label smoothing
        self.GRADIENT_CLIPPING = 1.0  # Gradient clipping to prevent exploding gradients
        self.MIN_LR = 1e-7  # Minimum learning rate for scheduler
        
        os.makedirs(self.CHECKPOINT_PATH, exist_ok=True)


import os
import torch
import torch.nn as nn

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config
config = Config()

# Set random seed
Utils.set_seed(config.SEED)

# Init data module
data_module = FERDataModule(csv_path=os.path.join(config.DATASET_PATH, 'fer2013.csv'),
                            batch_size=config.BATCH_SIZE,
                            image_size=config.IMAGE_SIZE[0],
                            criterion=config.CRITERION)
data_module.setup()

# Create dataloaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()


def initialize_optimizer(model, base_lr=1e-5, mid_lr=5e-5, head_lr=1e-3, weight_decay=1e-4):
    """
    Custom optimizer setup for TripletClassifier:
    - Backbone early layers: base_lr
    - Deeper layers (e.g., layer3/layer4): mid_lr
    - Classifier head: head_lr
    """
    param_groups = []

    for name, param in model.named_parameters():
        if 'classifier' in name:
            param_groups.append({'params': param, 'lr': head_lr})
        elif 'layer4' in name or 'layer3' in name:  # deeper resnet blocks
            param_groups.append({'params': param, 'lr': mid_lr})
        elif 'embedding_model' in name or 'features' in name:
            param_groups.append({'params': param, 'lr': base_lr})
        else:
            param_groups.append({'params': param, 'lr': base_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    return optimizer


# Initialize model
model = TripletClassifier(
    weights_path="/kaggle/input/resnet5-vgg-face/pytorch/v1/1/best_triplet_model.pth",
    num_classes=config.NUM_CLASSES,
    embedding_dim=512,
    freeze_until=4,
    device='cuda',
    model_name="resnet50",
)
progressive_unfreezing_frequency = 4
optimizer = initialize_optimizer(model=model, base_lr=1e-4, mid_lr=5e-4, head_lr=1e-2, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=progressive_unfreezing_frequency*2, gamma=0.1)
criterion = get_criterion(config, data_module.get_class_weights())
# Initialize trainer
trainer = Trainer(model, train_loader, val_loader, config, device, criterion, optimizer=optimizer,scheduler=scheduler, progressive_unfreezing_frequency=progressive_unfreezing_frequency)
history, best_epoch, best_f1, best_acc, best_model = trainer.train()


# Plot training performance
Utils.plot_loss_curve(history['train_loss'], history['val_loss'], show=True, save=True)
Utils.plot_accuracy_curve(history['train_acc'], history['val_acc'], show=True, save=True)
Utils.plot_f1_score(history['val_f1'], show=True, save=True)


# Run test
best_dict = torch.load(config.BEST_MODEL_PATH, map_location=device)
best_model = TripletClassifier(
    num_classes=config.NUM_CLASSES,
    embedding_dim=512,
    freeze_until=4,
    device='cuda',
    model_name="resnet50",
    state_dict=best_dict
)
tester = Test(best_model, test_loader, criterion, config)
test_results = tester.test(class_names=data_module.get_class_names())

# Save final model
torch.save(model.state_dict(), config.FINAL_MODEL_PATH)
print(f"âœ… Final model saved at: {config.FINAL_MODEL_PATH}")

