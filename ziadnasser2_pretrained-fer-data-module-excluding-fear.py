!tar -xvzf /kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013.tar.gz -C /kaggle/working


!pip install captum


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
        self.data = self.data[~self.data['emotion'].isin([0, 1, 2])].reset_index(drop=True)

        self.transform = transform

        # Remap remaining labels to 0...n-1
        original_to_new = {3: 0, 4: 1, 5: 2, 6: 3}
        self.data['emotion'] = self.data['emotion'].map(original_to_new)

        self.emotion_map = {
            0: 'Happy', 1: 'Sad', 2: 'Surprise', 3: 'Neutral'
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


class FERDataModule:
    def __init__(self, csv_path, batch_size=32, image_size=128, criterion=None, train_transform=None, val_transform=None):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.criterion = criterion
        self.image_size = (image_size, image_size)
        self.emotion_map = {
            0: 'Happy', 1: 'Sad', 2: 'Surprise', 3: 'Neutral'
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



# csv_path = "/kaggle/working/fer2013/fer2013.csv"
# dm = FERDataModule(csv_path, batch_size=32, image_size=128)

# # Run tests
# test_data_module(dm)

# # Plot training samples
# print("\nğŸ–¼ï¸� Plotting training samples...")
# plot_samples_from_dataloader(dm.train_dataloader(), class_map=dm.emotion_map)

# # Plot validation samples
# print("\nğŸ–¼ï¸� Plotting validation samples...")
# plot_samples_from_dataloader(dm.val_dataloader(), class_map=dm.emotion_map)


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
        super(MixUpLoss, self).__init__()
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.resnet import BasicBlock, Bottleneck
import copy

def create_deeplift_compatible_resnet(model_name="resnet18", pretrained=True, input_channels=1):
    """
    Create a DeepLift-compatible ResNet by ensuring no module reuse.
    """
    if model_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
    elif model_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet50(weights=weights)
    else:
        raise NotImplementedError(f"Model '{model_name}' is not supported yet.")

    # Replace first conv layer for grayscale input
    model.conv1 = nn.Conv2d(
        in_channels=input_channels,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )
    
    # Fix the model for DeepLift compatibility
    fix_model_for_deeplift(model)
    
    return model

def fix_model_for_deeplift(model):
    """
    Comprehensive fix for DeepLift compatibility:
    1. Replace all inplace operations
    2. Ensure no module reuse
    3. Create separate ReLU instances for each use
    """
    # First pass: replace all inplace ReLUs
    replace_relu_inplace_recursive(model)
    
    # Second pass: fix residual blocks to avoid module reuse
    fix_residual_blocks(model)

def replace_relu_inplace_recursive(model):
    """Replace all inplace ReLU operations with non-inplace versions."""
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, nn.ReLU(inplace=False))
        else:
            replace_relu_inplace_recursive(module)

def fix_residual_blocks(model):
    """
    Fix residual blocks by creating separate ReLU instances and custom forward methods.
    """
    for module in model.modules():
        if isinstance(module, BasicBlock):
            fix_basic_block(module)
        elif isinstance(module, Bottleneck):
            fix_bottleneck_block(module)

def fix_basic_block(block):
    """Fix BasicBlock to be DeepLift compatible."""
    # Create separate ReLU instances
    block.relu1 = nn.ReLU(inplace=False)
    block.relu2 = nn.ReLU(inplace=False)
    
    # Store original relu for compatibility
    original_relu = block.relu
    
    # Define new forward method
    def deeplift_forward(self, x):
        identity = x.clone()  # Explicit clone to avoid in-place issues
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)  # Use separate relu instance
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Non-inplace addition
        out = torch.add(out, identity)  # Explicit non-inplace addition
        out = self.relu2(out)  # Use separate relu instance
        
        return out
    
    # Bind the new forward method
    block.forward = deeplift_forward.__get__(block, BasicBlock)

def fix_bottleneck_block(block):
    """Fix Bottleneck block to be DeepLift compatible."""
    # Create separate ReLU instances
    block.relu1 = nn.ReLU(inplace=False)
    block.relu2 = nn.ReLU(inplace=False)
    block.relu3 = nn.ReLU(inplace=False)
    block.relu4 = nn.ReLU(inplace=False)
    
    # Define new forward method
    def deeplift_forward(self, x):
        identity = x.clone()  # Explicit clone
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Non-inplace addition
        out = torch.add(out, identity)
        out = self.relu3(out)
        
        return out
    
    # Bind the new forward method
    block.forward = deeplift_forward.__get__(block, Bottleneck)

class DeepLiftCompatibleFeatureExtractor(nn.Module):
    """
    Feature extractor specifically designed for DeepLift compatibility.
    """
    def __init__(self, backbone, embedding_dim=128):
        super(DeepLiftCompatibleFeatureExtractor, self).__init__()
        
        # Remove classifier from backbone
        if hasattr(backbone, 'fc'):
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            in_features = backbone.fc.in_features
        elif hasattr(backbone, 'classifier'):
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            in_features = backbone.classifier[1].in_features
        else:
            raise ValueError("Unsupported backbone architecture")
        
        # Add global average pooling if not present
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Embedding layer
        self.embedding = nn.Linear(in_features, embedding_dim)
        
        # Separate ReLU for embedding if needed
        self.embedding_relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)

class DeepLiftCompatibleTripletNetwork(nn.Module):
    """
    Triplet network optimized for DeepLift compatibility.
    """
    def __init__(self, model_name="resnet18", embedding_dim=128, pretrained=True):
        super(DeepLiftCompatibleTripletNetwork, self).__init__()
        
        # Load DeepLift-compatible backbone
        backbone = create_deeplift_compatible_resnet(model_name, pretrained)
        self.embedding_model = DeepLiftCompatibleFeatureExtractor(backbone, embedding_dim)

    def forward(self, x):
        return self.embedding_model(x)

class DeepLiftCompatibleTripletClassifier(nn.Module):
    """
    Classifier wrapper optimized for DeepLift compatibility.
    """
    def __init__(self,
                 weights_path=None,
                 num_classes=10,
                 embedding_dim=128,
                 freeze_until=None,
                 model_name="resnet18",
                 pretrained=True,
                 device=None,
                 state_dict=None):
        super(DeepLiftCompatibleTripletClassifier, self).__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create DeepLift-compatible backbone
        if state_dict is not None:
            self.backbone = DeepLiftCompatibleTripletNetwork(
                model_name=model_name, 
                embedding_dim=embedding_dim, 
                pretrained=False
            )
        elif weights_path is not None:
            self.backbone = DeepLiftCompatibleTripletNetwork(
                model_name=model_name,
                embedding_dim=embedding_dim,
                pretrained=pretrained
            )
            # Load weights with proper handling
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.backbone.load_state_dict(checkpoint, strict=False)
        else:
            raise ValueError("Either 'weights_path' or 'state_dict' must be provided.")

        # Layer management for progressive unfreezing
        self.feature_layers = list(self.backbone.embedding_model.features.children())

        if freeze_until is not None:
            self.freeze_layers(freeze_until)
        self.currently_frozen_until = freeze_until or 0

        # Classification head with separate components
        self.classifier = nn.Linear(embedding_dim, num_classes)

        if state_dict is not None:
            self.load_state_dict(state_dict, strict=False)

        self.to(self.device)

    def freeze_layers(self, freeze_until):
        """Freeze layers up to a certain point."""
        for i, layer in enumerate(self.feature_layers):
            requires_grad = i >= freeze_until
            for param in layer.parameters():
                param.requires_grad = requires_grad

    def progressively_unfreeze(self, step=1):
        """Progressively unfreeze layers."""
        new_freeze_until = max(0, self.currently_frozen_until - step)
        self.freeze_layers(new_freeze_until)
        self.currently_frozen_until = new_freeze_until
        print(f"[INFO] Unfroze layers from {new_freeze_until} to {len(self.feature_layers)}")

    def forward(self, x):
        """Forward pass through the network."""
        embedding = self.backbone(x)
        logits = self.classifier(embedding)
        return logits

# Updated attribution function that works with the fixed model
def attribute_image_features_deeplift(model, input_tensor, target_class, baselines=None):
    """
    Perform DeepLift attribution on the fixed model.
    
    Args:
        model: DeepLift-compatible model
        input_tensor: Input tensor to attribute
        target_class: Target class for attribution
        baselines: Baseline tensor (default: zero baseline)
    
    Returns:
        Attribution tensor
    """
    from captum.attr import DeepLift
    
    # Set model to evaluation mode
    model.eval()
    
    # Create baselines if not provided
    if baselines is None:
        baselines = torch.zeros_like(input_tensor)
    
    # Initialize DeepLift
    dl = DeepLift(model)
    
    # Clear gradients
    model.zero_grad()
    
    # Compute attributions
    attributions = dl.attribute(
        input_tensor,
        baselines=baselines,
        target=target_class,
        return_convergence_delta=False
    )
    
    return attributions

# Example usage function
def create_compatible_model_from_existing(original_weights_path, model_name="resnet18", 
                                        embedding_dim=128, num_classes=10):
    """
    Create a DeepLift-compatible version of your existing model.
    
    Args:
        original_weights_path: Path to your original model weights
        model_name: Architecture name
        embedding_dim: Embedding dimension
        num_classes: Number of output classes
    
    Returns:
        DeepLift-compatible model
    """
    # Create new compatible model
    model = DeepLiftCompatibleTripletClassifier(
        weights_path=original_weights_path,
        model_name=model_name,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        pretrained=True
    )
    
    print("Created DeepLift-compatible model successfully!")
    print("You can now use DeepLift attribution without module reuse errors.")
    
    return model


import os

class Config:
    def __init__(self):
        self.DATASET_PATH = "/kaggle/working/fer2013"
        self.MODEL_NAME = "EfficientNet"
        self.NUM_CLASSES = 4
        self.BATCH_SIZE = 16 
        self.NUM_EPOCHS = 80
        self.LEARNING_RATE = 1e-6 
        self.CRITERION = 'weighted_loss'  # Start simple, then experiment
        self.SEED = 42
        self.CHECKPOINT_PATH = "/kaggle/working/checkpoints"
        self.BEST_MODEL_PATH = os.path.join(self.CHECKPOINT_PATH, "best_model.pth")
        self.FINAL_MODEL_PATH = os.path.join(self.CHECKPOINT_PATH, "final_model.pth")
        self.USE_TQDM = False
        self.EARLY_STOP_PATIENCE = 15  # More aggressive early stopping
        self.WEIGHT_DECAY = 1e-5  # Increased L2 regularization
        self.IMAGE_SIZE = (128, 128)
        
        # New anti-overfitting parameters
        self.DROPOUT_RATE = 0.0  # Increased dropout
        self.LABEL_SMOOTHING = 0.2  # Add label smoothing
        self.GRADIENT_CLIPPING = 1.0  # Gradient clipping to prevent exploding gradients
        self.MIN_LR = 1e-7  # Minimum learning rate for scheduler
        
        os.makedirs(self.CHECKPOINT_PATH, exist_ok=True)


# import os
# import torch
# import torch.nn as nn

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load config
# config = Config()

# # Set random seed
# Utils.set_seed(config.SEED)

# # Init data module
# data_module = FERDataModule(csv_path=os.path.join(config.DATASET_PATH, 'fer2013.csv'),
#                             batch_size=config.BATCH_SIZE,
#                             image_size=config.IMAGE_SIZE[0],
#                             criterion=config.CRITERION)
# data_module.setup()

# # Create dataloaders
# train_loader = data_module.train_dataloader()
# val_loader = data_module.val_dataloader()
# test_loader = data_module.test_dataloader()


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


# # Initialize model
# model = TripletClassifier(
#     weights_path="/kaggle/input/resnet5-vgg-face/pytorch/v1/1/best_triplet_model.pth",
#     num_classes=config.NUM_CLASSES,
#     embedding_dim=512,
#     freeze_until=4,
#     device='cuda',
#     model_name="resnet50",
# )
# progressive_unfreezing_frequency = 4
# optimizer = initialize_optimizer(model=model, base_lr=1e-4, mid_lr=5e-4, head_lr=1e-2, weight_decay=1e-4)
# scheduler = StepLR(optimizer, step_size=progressive_unfreezing_frequency*2, gamma=0.1)
# criterion = get_criterion(config, data_module.get_class_weights())
# # Initialize trainer
# trainer = Trainer(model, train_loader, val_loader, config, device, criterion, optimizer=optimizer,scheduler=scheduler, progressive_unfreezing_frequency=progressive_unfreezing_frequency)
# history, best_epoch, best_f1, best_acc, best_model = trainer.train()


# # Plot training performance
# Utils.plot_loss_curve(history['train_loss'], history['val_loss'], show=True, save=True)
# Utils.plot_accuracy_curve(history['train_acc'], history['val_acc'], show=True, save=True)
# Utils.plot_f1_score(history['val_f1'], show=True, save=True)


# # Run test
# best_dict = torch.load(config.BEST_MODEL_PATH, map_location=device)
# best_model = TripletClassifier(
#     num_classes=config.NUM_CLASSES,
#     embedding_dim=512,
#     freeze_until=5,
#     device='cuda',
#     model_name="resnet50",
#     state_dict=best_dict
# )
# tester = Test(best_model, test_loader, criterion, config)
# test_results = tester.test(class_names=data_module.get_class_names())

# # Save final model
# torch.save(model.state_dict(), config.FINAL_MODEL_PATH)
# print(f"âœ… Final model saved at: {config.FINAL_MODEL_PATH}")


# import os
# import torch
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader


# # Paths
# checkpoint_path = "/kaggle/input/best-model-without-least-3-classes/pytorch/default/1/checkpoints/best_model.pth"
# csv_path = "/kaggle/working/fer2013/fer2013.csv"
# output_dir = "sample_saliency_maps"
# os.makedirs(output_dir, exist_ok=True)

# # Device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load classifier state
# model_state = torch.load(checkpoint_path, map_location=device)
# model = DeepLiftCompatibleTripletClassifier(
#     num_classes=Config().NUM_CLASSES,
#     model_name='resnet50',
#     embedding_dim=512,
#     state_dict=model_state,
#     freeze_until=0
# ).to(device)

# # DataModule
# dm = FERDataModule(csv_path, batch_size=Config().BATCH_SIZE)
# dm.setup()
# test_loader = dm.test_dataloader()


from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
import torchvision


# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# dataiter = iter(test_loader)
# images, labels = next(dataiter)

# # print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(4)))


# outputs = model(images)

# _, predicted = torch.max(outputs, 1)

# print('Predicted: ', ' '.join('%5s' % predicted[j]
#                               for j in range(4)))


# ind = 8

# input = images[ind].unsqueeze(0)
# input.requires_grad = True

# def attribute_image_features(algorithm, input, **kwargs):
#     model.zero_grad()
#     tensor_attributions = algorithm.attribute(input,
#                                               target=labels[ind],
#                                               **kwargs
#                                              )
    
#     return tensor_attributions

# def perform_deeplift_attribution(model, input_tensor, target_class):
#     """
#     Perform DeepLift attribution with proper error handling.
#     """
#     try:
#         # Create baseline (zero image)
#         baselines = torch.zeros_like(input_tensor)
        
#         # Initialize DeepLift
#         dl = DeepLift(model)
        
#         # Set model to evaluation mode
#         model.eval()
        
#         # Clear gradients
#         model.zero_grad()
        
#         # Compute attributions
#         attributions = dl.attribute(
#             input_tensor,
#             baselines=baselines,
#             target=target_class,
#             return_convergence_delta=False
#         )
        
#         return attributions
        
#     except Exception as e:
#         print(f"Attribution failed: {e}")
#         return None


# saliency = Saliency(model)
# grads = saliency.attribute(input, target=labels[ind].item())
# grads.size()

# # grads: [1, 1, 128, 128]
# grads = grads.squeeze().cpu().detach().numpy()  # [128, 128]

# # Stack to simulate RGB
# grads_rgb = np.stack([grads]*3, axis=-1)  # shape: [128, 128, 3]

# ig = IntegratedGradients(model)
# attr_ig, delta = attribute_image_features(ig, input, baselines=input * 0, return_convergence_delta=True)
# attr_ig.size()

# attr_ig = attr_ig.squeeze().cpu().detach().numpy()  # [128, 128]

# # Stack to simulate RGB
# attr_ig = np.stack([grads]*3, axis=-1)  # shape: [128, 128, 3]
# print('Approximation delta: ', abs(delta))
# attr_dl = perform_deeplift_attribution(model, input, predicted[ind])

# attr_dl = attr_dl.squeeze().cpu().detach().numpy()  # [128, 128]

# # Stack to simulate RGB
# attr_dl = np.stack([grads]*3, axis=-1)  # shape: [128, 128, 3]


# import matplotlib.pyplot as plt
# from captum.attr import visualization as viz

# # Define emotion labels in correct class order
# class_names = ['happy', 'sad', 'surprise', 'neutral']

# def to_rgb(attr):
#     """
#     Converts 2D or single-channel attribution maps to 3-channel RGB for visualization.
#     """
#     if attr.ndim == 2:
#         return np.stack([attr]*3, axis=-1)
#     elif attr.ndim == 3 and attr.shape[0] == 1:  # [1, H, W]
#         attr = np.squeeze(attr, axis=0)
#         return np.stack([attr]*3, axis=-1)
#     return attr  # already 3-channel

# def visualize_deeplift(model, input_image, target_class, device, class_names=None):
#     """
#     Visualize DeepLift attributions for a given input image and target class.
#     """
#     model.eval()
#     input_image = input_image.unsqueeze(0).to(device)  # Add batch dimension
#     baselines = torch.zeros_like(input_image)  # Zero baseline for DeepLift
    
#     # Compute DeepLift attributions
#     dl = DeepLift(model)
#     try:
#         attributions = dl.attribute(input_image, baselines=baselines, target=target_class)
#         attr = attributions.squeeze().cpu().detach().numpy()
        
#         # Prepare original image for visualization
#         original_image = input_image.squeeze().cpu().numpy()
#         if original_image.ndim == 2:  # [H, W]
#             original_image = np.stack([original_image]*3, axis=-1)
#         elif original_image.shape[0] == 1:  # [1, H, W]
#             original_image = np.transpose(original_image, (1, 2, 0))
#             original_image = np.repeat(original_image, 3, axis=2)
#         else:  # [C, H, W]
#             original_image = np.transpose(original_image, (1, 2, 0))
#         original_image = (original_image / 2) + 0.5  # Unnormalize

#         # Predict probability for title
#         with torch.no_grad():
#             output = model(input_image)
#             prob = F.softmax(output, dim=1)[0, target_class].item()
        
#         # Plot original image and DeepLift attribution
#         fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#         title = f"DeepLift (Class: {class_names[target_class] if class_names else target_class}, Prob: {prob:.2f})"
        
#         # Original image
#         viz.visualize_image_attr(to_rgb(np.ones_like(original_image)), original_image,
#                                  method="original_image", title="Original",
#                                  plt_fig_axis=(fig, axes[0]), use_pyplot=False)
        
#         # DeepLift attribution
#         viz.visualize_image_attr(to_rgb(attr), original_image,
#                                  method="blended_heat_map", sign="all", show_colorbar=True,
#                                  title="DeepLift", plt_fig_axis=(fig, axes[1]), use_pyplot=False)
        
#         fig.suptitle(title, fontsize=12)
#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         plt.show()
        
#     except Exception as e:
#         print(f"DeepLift attribution failed: {e}")



# print('Original Image')
# print('Predicted:', predicted[ind], 
#       ' Probability:', torch.max(F.softmax(outputs, 1)).item())

# original_image = np.transpose((images[ind].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
# dummy_attr = np.ones_like(original_image)

# _ = viz.visualize_image_attr(dummy_attr, original_image, 
#                       method="original_image", title="Original Image")

# grads = np.expand_dims(grads, axis=-1)  # shape: [128, 128, 1]
# _ = viz.visualize_image_attr(grads, original_image, method="blended_heat_map", sign="absolute_value",
#                           show_colorbar=True, title="Overlayed Gradient Magnitudes")

# _ = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map",sign="all",
#                           show_colorbar=True, title="Overlayed Integrated Gradients")

# # _ = viz.visualize_image_attr(attr_ig_nt, original_image, method="blended_heat_map", sign="absolute_value", 
# #                              outlier_perc=10, show_colorbar=True, 
# #                              title="Overlayed Integrated Gradients \n with SmoothGrad Squared")

# _ = viz.visualize_image_attr(attr_dl, original_image, method="blended_heat_map",sign="all",show_colorbar=True, 
#                           title="Overlayed DeepLift")




# indices_to_visualize = [9, 6, 15]  # <-- change these as needed


# for ind in indices_to_visualize:
#     input_tensor = images[ind].unsqueeze(0).to(device)
#     true_class_idx = labels[ind].item()
#     true_class = class_names[true_class_idx]

#     output = model(input_tensor)
#     pred_class_idx = torch.argmax(output, dim=1).item()
#     pred_class = class_names[pred_class_idx]
#     prob = F.softmax(output, 1)[0, pred_class_idx].item()

#     print(f"Image {ind} - True: {true_class} | Predicted: {pred_class} | Prob: {prob:.4f}")

#     # DeepLift Attribution
#     attr_dl = perform_deeplift_attribution(model, input_tensor, pred_class_idx)
#     if attr_dl is None:
#         continue
#     attr_dl = attr_dl.squeeze().cpu().detach().numpy()

#     # Integrated Gradients
#     ig = IntegratedGradients(model)
#     attr_ig, delta = ig.attribute(input_tensor, baselines=torch.zeros_like(input_tensor), target=pred_class_idx, return_convergence_delta=True)
#     attr_ig = attr_ig.squeeze().cpu().detach().numpy()

#     # Saliency
#     saliency = Saliency(model)
#     grads = saliency.attribute(input_tensor, target=pred_class_idx)
#     grads = grads.squeeze().cpu().detach().numpy()

#     # Prepare original image for visualization
#     original_image = input_tensor.squeeze().cpu().numpy()
#     if original_image.ndim == 2:  # [H, W]
#         original_image = np.stack([original_image]*3, axis=-1)
#     elif original_image.shape[0] == 1:  # [1, H, W]
#         original_image = np.transpose(original_image, (1, 2, 0))
#         original_image = np.repeat(original_image, 3, axis=2)
#     else:  # [C, H, W]
#         original_image = np.transpose(original_image, (1, 2, 0))

#     # Normalize to [0, 1]
#     original_image = (original_image / 2) + 0.5

#     # Plot in 1 row with 4 columns using Captum's plotting
#     fig, axes = plt.subplots(1, 4, figsize=(20, 5))

#     viz.visualize_image_attr(to_rgb(np.ones_like(original_image)), original_image,
#                              method="original_image", title="Original",
#                              plt_fig_axis=(fig, axes[0]), use_pyplot=False)

#     viz.visualize_image_attr(to_rgb(grads), original_image,
#                              method="blended_heat_map", sign="absolute_value",
#                              show_colorbar=True, title="Saliency",
#                              plt_fig_axis=(fig, axes[1]), use_pyplot=False)

#     viz.visualize_image_attr(to_rgb(attr_ig), original_image,
#                              method="blended_heat_map", sign="all",
#                              show_colorbar=True, title="Integrated Gradients",
#                              plt_fig_axis=(fig, axes[2]), use_pyplot=False)

#     viz.visualize_image_attr(to_rgb(attr_dl), original_image,
#                              method="blended_heat_map", sign="all",
#                              show_colorbar=True, title="DeepLift",
#                              plt_fig_axis=(fig, axes[3]), use_pyplot=False)

#     fig.suptitle(f"Image {ind} | True: {true_class} | Pred: {pred_class} ({prob:.2f})", fontsize=16)
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.show()



# import torch
# from PIL import Image
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import numpy as np
# import torch.nn.functional as F
# from captum.attr import Saliency, IntegratedGradients, DeepLift, visualization as viz

# # --- helper to convert attributions to RGB format ---
# def to_rgb(attr):
#     if attr.ndim == 2:
#         return np.stack([attr]*3, axis=-1)
#     elif attr.ndim == 3 and attr.shape[0] == 1:  # [1, H, W]
#         attr = np.squeeze(attr, axis=0)
#         return np.stack([attr]*3, axis=-1)
#     return attr  # already 3-channel

# def preprocess_image(image_path):
#     image = Image.open(image_path).convert('L')  # Grayscale
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#     return transform(image)

# def predict_emotion(model, image, device, class_names):
#     model.eval()
#     image = image.unsqueeze(0).to(device)
#     with torch.no_grad():
#         output = model(image)
#         probabilities = torch.softmax(output, dim=1)
#         predicted_class = torch.argmax(probabilities, dim=1).item()
#     return class_names[predicted_class], predicted_class, probabilities[0].cpu().numpy()

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     config = Config()
#     model_state = torch.load(checkpoint_path, map_location=device)
#     model = DeepLiftCompatibleTripletClassifier(
#         num_classes=Config().NUM_CLASSES,
#         model_name='resnet50',
#         embedding_dim=512,
#         state_dict=model_state,
#         freeze_until=0
#     ).to(device)

#     class_names = {0: 'Happy', 1: 'Sad', 2: 'Surprise', 3: 'Neutral'}

#     image_paths = [
#         '/kaggle/input/happy-faces-test/Test image2/happy0.png', 
#         '/kaggle/input/happy-faces-test/Test image2/happy1.png',
#         '/kaggle/input/happy-faces-test/Test image2/happy2.png',
#         '/kaggle/input/happy-faces-test/Test image2/happy3.png'
#     ]

#     for image_path in image_paths:
#         image = preprocess_image(image_path)
#         class_name, pred_class_idx, probabilities = predict_emotion(model, image, device, class_names)

#         print(f"\nImage: {image_path}")
#         print(f"Predicted Emotion: {class_name}")
#         print("Probabilities:")
#         for i, prob in enumerate(probabilities):
#             emotion = class_names.get(i, f"Class {i}")
#             print(f"{emotion}: {prob:.4f}")

#         input_tensor = image.unsqueeze(0).to(device)

#         # DeepLift
#         try:
#             attr_dl = DeepLift(model).attribute(input_tensor, baselines=torch.zeros_like(input_tensor), target=pred_class_idx)
#             attr_dl = attr_dl.squeeze().cpu().detach().numpy()
#         except Exception as e:
#             print(f"DeepLift attribution failed: {e}")
#             continue

#         # Integrated Gradients
#         ig = IntegratedGradients(model)
#         attr_ig, delta = ig.attribute(input_tensor, baselines=torch.zeros_like(input_tensor), target=pred_class_idx, return_convergence_delta=True)
#         attr_ig = attr_ig.squeeze().cpu().detach().numpy()

#         # Saliency
#         saliency = Saliency(model)
#         grads = saliency.attribute(input_tensor, target=pred_class_idx)
#         grads = grads.squeeze().cpu().detach().numpy()

#         # Original image prep
#         original_image = image.squeeze().cpu().numpy()
#         original_image = np.transpose(np.stack([original_image]*3, axis=0), (1, 2, 0))
#         original_image = (original_image / 2) + 0.5

#         # Visualization
#         fig, axes = plt.subplots(1, 4, figsize=(20, 5))
#         viz.visualize_image_attr(to_rgb(np.ones_like(original_image)), original_image,
#                                  method="original_image", title="Original",
#                                  plt_fig_axis=(fig, axes[0]), use_pyplot=False)
#         viz.visualize_image_attr(to_rgb(grads), original_image,
#                                  method="blended_heat_map", sign="absolute_value",
#                                  show_colorbar=True, title="Saliency",
#                                  plt_fig_axis=(fig, axes[1]), use_pyplot=False)
#         viz.visualize_image_attr(to_rgb(attr_ig), original_image,
#                                  method="blended_heat_map", sign="all",
#                                  show_colorbar=True, title="Integrated Gradients",
#                                  plt_fig_axis=(fig, axes[2]), use_pyplot=False)
#         viz.visualize_image_attr(to_rgb(attr_dl), original_image,
#                                  method="blended_heat_map", sign="all",
#                                  show_colorbar=True, title="DeepLift",
#                                  plt_fig_axis=(fig, axes[3]), use_pyplot=False)

#         fig.suptitle(f"Prediction: {class_name} | Prob: {probabilities[pred_class_idx]:.2f}", fontsize=16)
#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         plt.show()

# if __name__ == "__main__":
#     main()



# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from captum.attr import DeepLift, IntegratedGradients, Saliency
# from captum.attr import visualization as viz
# from collections import defaultdict
# import torch.nn.functional as F

# class MaskAggregator:
#     def __init__(self, class_names):
#         self.class_names = class_names
#         self.num_classes = len(class_names)
        
#         # Store masks and weights for each emotion
#         self.emotion_masks = defaultdict(list)  # {emotion_idx: [masks]}
#         self.emotion_weights = defaultdict(list)  # {emotion_idx: [probabilities]}
#         self.emotion_counts = defaultdict(int)  # {emotion_idx: count}
        
#         # Store representative face images for each emotion
#         self.representative_faces = {}  # {emotion_idx: face_image}
#         self.representative_probs = {}  # {emotion_idx: probability}
        
#     def add_mask(self, mask, predicted_class, probabilities, face_image=None):
#         """
#         Add a mask to the aggregator with its prediction probabilities
        
#         Args:
#             mask: Attribution mask (numpy array)
#             predicted_class: Predicted class index
#             probabilities: Array of probabilities for all classes
#             face_image: Original face image (numpy array)
#         """
#         # Store the mask and probability for the predicted class
#         self.emotion_masks[predicted_class].append(mask.copy())
#         self.emotion_weights[predicted_class].append(probabilities[predicted_class])
#         self.emotion_counts[predicted_class] += 1
        
#         # Update representative face if this prediction has higher confidence
#         current_prob = probabilities[predicted_class]
#         if (predicted_class not in self.representative_probs or 
#             current_prob > self.representative_probs[predicted_class]):
#             if face_image is not None:
#                 self.representative_faces[predicted_class] = face_image.copy()
#                 self.representative_probs[predicted_class] = current_prob
        
#     def to_rgb_face(self, face_image):
#         """Convert face image to RGB format for visualization"""
#         if face_image.ndim == 2:  # [H, W] grayscale
#             return np.stack([face_image]*3, axis=-1)
#         elif face_image.ndim == 3 and face_image.shape[0] == 1:  # [1, H, W]
#             face_image = np.squeeze(face_image, axis=0)
#             return np.stack([face_image]*3, axis=-1)
#         elif face_image.ndim == 3 and face_image.shape[0] == 3:  # [3, H, W]
#             return np.transpose(face_image, (1, 2, 0))
#         return face_image  # already [H, W, 3]
    
#     def normalize_face(self, face_image):
#         """Normalize face image to [0, 1] range"""
#         # Assuming your images are normalized to [-1, 1], convert to [0, 1]
#         return (face_image / 2) + 0.5
#     def get_weighted_average_mask(self, emotion_idx):
#         """
#         Get weighted average mask for a specific emotion
        
#         Args:
#             emotion_idx: Index of the emotion class
            
#         Returns:
#             Weighted average mask or None if no masks exist
#         """
#         if emotion_idx not in self.emotion_masks or len(self.emotion_masks[emotion_idx]) == 0:
#             return None
            
#         masks = np.array(self.emotion_masks[emotion_idx])  # [N, H, W]
#         weights = np.array(self.emotion_weights[emotion_idx])  # [N]
        
#         # Normalize weights to sum to 1
#         weights = weights / np.sum(weights)
        
#         # Compute weighted average: sum(weight_i * mask_i)
#         weighted_mask = np.zeros_like(masks[0])
#         for i, (mask, weight) in enumerate(zip(masks, weights)):
#             weighted_mask += weight * mask
            
#         return weighted_mask
    
#     def get_all_weighted_masks(self):
#         """
#         Get weighted average masks for all emotions that have data
        
#         Returns:
#             Dictionary {emotion_idx: weighted_mask}
#         """
#         weighted_masks = {}
#         for emotion_idx in self.emotion_masks.keys():
#             mask = self.get_weighted_average_mask(emotion_idx)
#             if mask is not None:
#                 weighted_masks[emotion_idx] = mask
#         return weighted_masks
    
#     def visualize_aggregated_masks(self, method_name="DeepLift"):
#         """
#         Visualize the aggregated masks overlaid on representative faces
#         """
#         weighted_masks = self.get_all_weighted_masks()
        
#         if not weighted_masks:
#             print("No masks to visualize!")
#             return
            
#         n_emotions = len(weighted_masks)
#         fig, axes = plt.subplots(2, n_emotions, figsize=(5*n_emotions, 10))
        
#         if n_emotions == 1:
#             axes = axes.reshape(2, 1)
            
#         for idx, (emotion_idx, mask) in enumerate(weighted_masks.items()):
#             emotion_name = self.class_names.get(emotion_idx, f"Class {emotion_idx}")
#             count = self.emotion_counts[emotion_idx]
#             prob = self.representative_probs.get(emotion_idx, 0.0)
            
#             # Get representative face for this emotion
#             if emotion_idx in self.representative_faces:
#                 face_image = self.representative_faces[emotion_idx]
#                 face_rgb = self.to_rgb_face(face_image)
#                 face_normalized = self.normalize_face(face_rgb)
#             else:
#                 # Create a neutral gray background if no face available
#                 face_normalized = np.ones((128, 128, 3)) * 0.5
            
#             # Convert mask to RGB if needed
#             if mask.ndim == 2:
#                 mask_rgb = np.stack([mask]*3, axis=-1)
#             else:
#                 mask_rgb = mask
            
#             # Top row: Original representative face
#             viz.visualize_image_attr(
#                 np.ones_like(face_normalized), face_normalized,
#                 method="original_image",
#                 title=f"{emotion_name} Face\n(Prob: {prob:.3f})",
#                 plt_fig_axis=(fig, axes[0, idx]),
#                 use_pyplot=False
#             )
            
#             # Bottom row: Mask overlaid on face
#             viz.visualize_image_attr(
#                 mask_rgb, face_normalized,
#                 method="blended_heat_map", 
#                 sign="all",
#                 show_colorbar=True,
#                 title=f"{emotion_name} Mask\n({method_name}, n={count})",
#                 plt_fig_axis=(fig, axes[1, idx]),
#                 use_pyplot=False
#             )
            
#         plt.suptitle(f"Weighted Average {method_name} Masks on Representative Faces", fontsize=16)
#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         plt.show()
        
#     def visualize_mask_comparison(self, method_name="DeepLift"):
#         """
#         Create a side-by-side comparison of masks on faces vs abstract backgrounds
#         """
#         weighted_masks = self.get_all_weighted_masks()
        
#         if not weighted_masks:
#             print("No masks to visualize!")
#             return
            
#         n_emotions = len(weighted_masks)
#         fig, axes = plt.subplots(n_emotions, 3, figsize=(15, 5*n_emotions))
        
#         if n_emotions == 1:
#             axes = axes.reshape(1, 3)
            
#         for row_idx, (emotion_idx, mask) in enumerate(weighted_masks.items()):
#             emotion_name = self.class_names.get(emotion_idx, f"Class {emotion_idx}")
#             count = self.emotion_counts[emotion_idx]
#             prob = self.representative_probs.get(emotion_idx, 0.0)
            
#             # Get representative face
#             if emotion_idx in self.representative_faces:
#                 face_image = self.representative_faces[emotion_idx]
#                 face_rgb = self.to_rgb_face(face_image)
#                 face_normalized = self.normalize_face(face_rgb)
#             else:
#                 face_normalized = np.ones((128, 128, 3)) * 0.5
            
#             # Convert mask to RGB
#             if mask.ndim == 2:
#                 mask_rgb = np.stack([mask]*3, axis=-1)
#             else:
#                 mask_rgb = mask
            
#             # Column 1: Original face
#             viz.visualize_image_attr(
#                 np.ones_like(face_normalized), face_normalized,
#                 method="original_image",
#                 title=f"{emotion_name}\n(Prob: {prob:.3f})",
#                 plt_fig_axis=(fig, axes[row_idx, 0]),
#                 use_pyplot=False
#             )
            
#             # Column 2: Mask on face
#             viz.visualize_image_attr(
#                 mask_rgb, face_normalized,
#                 method="blended_heat_map", 
#                 sign="all",
#                 show_colorbar=True,
#                 title=f"Mask on Face\n(n={count})",
#                 plt_fig_axis=(fig, axes[row_idx, 1]),
#                 use_pyplot=False
#             )
            
#             # Column 3: Mask only (abstract)
#             gray_bg = np.ones_like(face_normalized) * 0.5
#             viz.visualize_image_attr(
#                 mask_rgb, gray_bg,
#                 method="blended_heat_map", 
#                 sign="all",
#                 show_colorbar=True,
#                 title="Abstract Mask",
#                 plt_fig_axis=(fig, axes[row_idx, 2]),
#                 use_pyplot=False
#             )
            
#         plt.suptitle(f"Face vs Abstract: {method_name} Attribution Masks", fontsize=16)
#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         plt.show()
        
#     def print_statistics(self):
#         """Print statistics about collected masks"""
#         print("Mask Aggregation Statistics:")
#         print("-" * 40)
#         for emotion_idx in sorted(self.emotion_masks.keys()):
#             emotion_name = self.class_names.get(emotion_idx, f"Class {emotion_idx}")
#             count = self.emotion_counts[emotion_idx]
#             avg_prob = np.mean(self.emotion_weights[emotion_idx])
#             print(f"{emotion_name}: {count} masks, avg probability: {avg_prob:.3f}")


# def collect_and_aggregate_masks(model, test_loader, device, class_names, num_samples=50):
#     """
#     Collect attribution masks from test data and create weighted aggregations
    
#     Args:
#         model: Trained model
#         test_loader: DataLoader for test data
#         device: Device to run on
#         class_names: Dictionary mapping class indices to names
#         num_samples: Number of samples to process
#     """
    
#     # Initialize aggregators for different attribution methods
#     deeplift_aggregator = MaskAggregator(class_names)
#     ig_aggregator = MaskAggregator(class_names)
#     saliency_aggregator = MaskAggregator(class_names)
    
#     model.eval()
#     sample_count = 0
    
#     print("Collecting attribution masks...")
    
#     for batch_idx, (images, labels) in enumerate(test_loader):
#         if sample_count >= num_samples:
#             break
            
#         images = images.to(device)
#         batch_size = images.shape[0]
        
#         with torch.no_grad():
#             outputs = model(images)
#             probabilities = F.softmax(outputs, dim=1)
#             predicted_classes = torch.argmax(outputs, dim=1)
        
#         # Process each image in the batch
#         for i in range(min(batch_size, num_samples - sample_count)):
#             input_tensor = images[i:i+1]  # Keep batch dimension
#             pred_class = predicted_classes[i].item()
#             probs = probabilities[i].cpu().numpy()
            
#             try:
#                 # DeepLift Attribution
#                 dl = DeepLift(model)
#                 attr_dl = dl.attribute(
#                     input_tensor, 
#                     baselines=torch.zeros_like(input_tensor), 
#                     target=pred_class
#                 )
#                 attr_dl_np = attr_dl.squeeze().cpu().detach().numpy()
#                 deeplift_aggregator.add_mask(attr_dl_np, pred_class, probs)
                
#                 # Integrated Gradients
#                 ig = IntegratedGradients(model)
#                 attr_ig, _ = ig.attribute(
#                     input_tensor, 
#                     baselines=torch.zeros_like(input_tensor), 
#                     target=pred_class,
#                     return_convergence_delta=True
#                 )
#                 attr_ig_np = attr_ig.squeeze().cpu().detach().numpy()
#                 ig_aggregator.add_mask(attr_ig_np, pred_class, probs)
                
#                 # Saliency
#                 saliency = Saliency(model)
#                 attr_sal = saliency.attribute(input_tensor, target=pred_class)
#                 attr_sal_np = attr_sal.squeeze().cpu().detach().numpy()
#                 saliency_aggregator.add_mask(attr_sal_np, pred_class, probs)
                
#                 sample_count += 1
#                 if sample_count % 10 == 0:
#                     print(f"Processed {sample_count}/{num_samples} samples")
                    
#             except Exception as e:
#                 print(f"Error processing sample {sample_count}: {e}")
#                 continue
    
#     print(f"\nCompleted processing {sample_count} samples")
    
#     # Print statistics
#     print("\nDeepLift Statistics:")
#     deeplift_aggregator.print_statistics()
    
#     print("\nIntegrated Gradients Statistics:")
#     ig_aggregator.print_statistics()
    
#     print("\nSaliency Statistics:")
#     saliency_aggregator.print_statistics()
    
#     # Visualize aggregated masks
#     print("\nVisualizing aggregated masks...")
#     deeplift_aggregator.visualize_aggregated_masks("DeepLift")
#     ig_aggregator.visualize_aggregated_masks("Integrated Gradients")
#     saliency_aggregator.visualize_aggregated_masks("Saliency")
    
#     return deeplift_aggregator, ig_aggregator, saliency_aggregator


# # Usage example (add this to your existing code):
# def run_mask_aggregation():
#     """
#     Main function to run mask aggregation analysis
#     """
#     # Your existing model setup code here...
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Load your model (use your existing model loading code)
#     model_state = torch.load(checkpoint_path, map_location=device)
#     model = DeepLiftCompatibleTripletClassifier(
#         num_classes=Config().NUM_CLASSES,
#         model_name='resnet50',
#         embedding_dim=512,
#         state_dict=model_state,
#         freeze_until=0
#     ).to(device)
    
#     # Your existing data loading code
#     dm = FERDataModule(csv_path, batch_size=Config().BATCH_SIZE)
#     dm.setup()
#     test_loader = dm.test_dataloader()
    
#     # Define class names
#     class_names = {0: 'Happy', 1: 'Sad', 2: 'Surprise', 3: 'Neutral'}
    
#     # Run aggregation
#     dl_agg, ig_agg, sal_agg = collect_and_aggregate_masks(
#         model, test_loader, device, class_names, num_samples=100
#     )
    
#     return dl_agg, ig_agg, sal_agg


# aggregators = run_mask_aggregation()


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from captum.attr import visualization as viz
from PIL import Image
import torchvision.transforms as transforms

class GradCAM:
    def __init__(self, model, target_layer_name):
        """
        Initialize GradCAM for a trained model.
        
        Args:
            model: Trained model (e.g., DeepLiftCompatibleTripletClassifier)
            target_layer_name: Name of the layer to extract gradients from (e.g., 'backbone.embedding_model.features.7.2.conv3')
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
    
        self.device = next(model.parameters()).device
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks for the target layer."""
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        # Find the target layer
        target_layer = self._find_target_layer(self.model, self.target_layer_name)
        if target_layer is None:
            raise ValueError(f"Layer '{self.target_layer_name}' not found in model")
            
        # Register hooks
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
    
    def _find_target_layer(self, model, layer_name):
        """Recursively find the target layer by name, handling nested paths."""
        target_parts = layer_name.split('.')
        
        def recursive_get_module(module, parts):
            if not parts:
                return module
            try:
                return recursive_get_module(getattr(module, parts[0]), parts[1:])
            except AttributeError:
                return None
        
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        
        module = recursive_get_module(model, target_parts)
        if module is not None:
            return module
        
        for name, module in model.named_modules():
            if layer_name in name:
                return module
                
        return None
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Class Activation Map.
        
        Args:
            input_tensor: Input image tensor [1, 1, H, W] (grayscale)
            target_class: Target class index (if None, uses predicted class)
            
        Returns:
            cam: Class activation map (numpy array)
            output: Model output logits
            target_class: Selected target class
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        
        weights = torch.mean(gradients, dim=[2, 3])
        
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32, device=self.device)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i]
        
        cam = F.relu(cam)
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy(), output, target_class
    
    def visualize_cam(self, input_tensor, target_class=None, class_names=None, alpha=0.4, colormap='jet'):
        """
        Visualize GradCAM overlay on original image using Captum visualization style.
        
        Args:
            input_tensor: Input image tensor [1, 1, H, W]
            target_class: Target class index
            class_names: Dictionary mapping class indices to names
            alpha: Transparency of overlay
            colormap: Colormap for heatmap (e.g., 'jet')
            
        Returns:
            original_img: Original image (RGB)
            cam: Raw CAM heatmap
            overlay: Overlay image
            pred_class: Predicted class
            confidence: Prediction confidence
        """
        cam, output, pred_class = self.generate_cam(input_tensor, target_class)
        
        probs = F.softmax(output, dim=1)
        confidence = probs[0, pred_class].item()
        
        original_img = input_tensor.squeeze().cpu().detach().numpy()
        original_img = (original_img / 2) + 0.5
        original_img = np.clip(original_img, 0, 1)
        original_img_rgb = np.stack([original_img] * 3, axis=-1)
        
        cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), getattr(cv2, f'COLORMAP_{colormap.upper()}'))
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255
        
        overlay = alpha * heatmap + (1 - alpha) * original_img_rgb
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        viz.visualize_image_attr(
            np.ones_like(original_img_rgb), original_img_rgb,
            method="original_image",
            title="Original Image",
            plt_fig_axis=(fig, axes[0]),
            use_pyplot=False
        )
        
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('GradCAM Heatmap')
        axes[1].axis('off')
        
        viz.visualize_image_attr(
            np.stack([cam_resized]*3, axis=-1), original_img_rgb,
            method="blended_heat_map",
            sign="all",
            show_colorbar=True,
            title=f"Overlay\nPred: {class_names.get(pred_class, pred_class)} ({confidence:.3f})",
            plt_fig_axis=(fig, axes[2]),
            use_pyplot=False
        )
        
        true_label = target_class if target_class is not None else pred_class
        plt.suptitle(f'GradCAM - True: {class_names.get(true_label, true_label)}', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        return original_img_rgb, cam_resized, overlay, pred_class, confidence

def test_gradcam_single_image(model, test_loader, device, class_names, layer_name='backbone.embedding_model.features.7.2.conv3'):
    """
    Test GradCAM on a single image for the last convolutional layer.
    """
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    input_tensor = images[0:1].to(device)
    true_label = labels[0].item()
    
    print(f"Testing GradCAM on image with true label: {class_names.get(true_label, true_label)}")
    
    gradcam = GradCAM(model, layer_name)
    original, cam, overlay, pred_class, confidence = gradcam.visualize_cam(
        input_tensor, class_names=class_names
    )
    
    return gradcam

def gradcam_batch_analysis(model, test_loader, device, class_names, layer_name='backbone.embedding_model.features.7.2.conv3', num_samples=20):
    """
    Analyze multiple images with GradCAM for the last convolutional layer.
    """
    gradcam = GradCAM(model, layer_name)
    results = []
    sample_count = 0
    
    print(f"Analyzing {num_samples} samples with GradCAM for layer: {layer_name}")
    
    for batch_idx, (images, labels) in enumerate(test_loader):
        if sample_count >= num_samples:
            break
        batch_size = images.shape[0]
        
        for i in range(min(batch_size, num_samples - sample_count)):
            input_tensor = images[i:i+1].to(device)
            true_label = labels[i].item()
            
            try:
                cam, output, pred_class = gradcam.generate_cam(input_tensor)
                probs = F.softmax(output, dim=1)
                confidence = probs[0, pred_class].item()
                
                results.append({
                    'image_idx': sample_count,
                    'true_label': true_label,
                    'pred_label': pred_class,
                    'confidence': confidence,
                    'cam': cam,
                    'input_tensor': input_tensor.cpu(),
                    'correct': true_label == pred_class
                })
                
                sample_count += 1
                if sample_count % 5 == 0:
                    print(f"Processed {sample_count}/{num_samples}")
                    
            except Exception as e:
                print(f"Error processing sample {sample_count}: {e}")
                continue
    
    return results, gradcam

def visualize_gradcam_grid(results, gradcam, class_names, rows=4, cols=5):
    """
    Visualize GradCAM results in a grid for the last convolutional layer.
    """
    num_images = min(len(results), rows * cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.flatten() if rows * cols > 1 else [axes]
    
    for i in range(num_images):
        result = results[i]
        original, cam, overlay, pred_class, confidence = gradcam.visualize_cam(
            result['input_tensor'].to(gradcam.model.device),
            class_names=class_names
        )
        
        axes[i].imshow(overlay)
        true_name = class_names.get(result['true_label'], f"Class {result['true_label']}")
        pred_name = class_names.get(result['pred_label'], f"Class {result['pred_label']}")
        color = 'green' if result['correct'] else 'red'
        axes[i].set_title(f"T: {true_name}\nP: {pred_name} ({confidence:.2f})", 
                         fontsize=10, color=color)
        axes[i].axis('off')
    
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'GradCAM Analysis Grid - Layer: {gradcam.target_layer_name.split(".")[-2:]}', fontsize=16)
    plt.tight_layout()
    plt.show()

def compare_gradcam_layers(model, input_tensor, device, class_names, layer_name='backbone.embedding_model.features.7.2.conv3'):
    """
    Visualize GradCAM for the last convolutional layer.
    """
    input_tensor = input_tensor.to(device)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    try:
        gradcam = GradCAM(model, layer_name)
        original, cam, overlay, pred_class, confidence = gradcam.visualize_cam(
            input_tensor, class_names=class_names
        )
        
        viz.visualize_image_attr(
            np.ones_like(original), original,
            method="original_image",
            title="Original Image",
            plt_fig_axis=(fig, axes[0]),
            use_pyplot=False
        )
        
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('GradCAM Heatmap')
        axes[1].axis('off')
        
        viz.visualize_image_attr(
            np.stack([cam]*3, axis=-1), original,
            method="blended_heat_map",
            sign="all",
            show_colorbar=True,
            title=f'{".".join(layer_name.split(".")[-2:])}\n{class_names.get(pred_class, pred_class)} ({confidence:.2f})',
            plt_fig_axis=(fig, axes[2]),
            use_pyplot=False
        )
        
    except Exception as e:
        print(f"Error with layer {layer_name}: {e}")
        axes[2].text(0.5, 0.5, f'Error\n{".".join(layer_name.split(".")[-2:])}', 
                     transform=axes[2].transAxes, ha='center', va='center')
        axes[2].axis('off')
    
    plt.suptitle('GradCAM for Last Conv Layer', fontsize=16)
    plt.tight_layout()
    plt.show()

def find_last_conv_layer(model):
    """
    Find the last Conv2d layer in the model.
    
    Returns:
        Layer name (e.g., 'backbone.embedding_model.features.7.2.conv3')
    """
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return name
    return 'backbone.embedding_model.features.7.2.conv3'  # Fallback to last conv layer in ResNet50 layer4

def apply_gradcam_to_image(model, image_path, device, class_names, layer_name='backbone.embedding_model.features.7.2.conv3'):
    """
    Apply GradCAM to an external test image not in the dataset.
    
    Args:
        model: Trained model (e.g., DeepLiftCompatibleTripletClassifier)
        image_path: Path to the external image file
        device: Device to run on (cuda or cpu)
        class_names: Dictionary mapping class indices to names
        layer_name: Name of the layer for GradCAM (default: last conv layer)
    
    Returns:
        None (displays the GradCAM visualization)
    """
    try:
        image = Image.open(image_path).convert('L')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    gradcam = GradCAM(model, layer_name)
    
    print(f"Applying GradCAM to external image: {image_path}")
    original, cam, overlay, pred_class, confidence = gradcam.visualize_cam(
        input_tensor=image_tensor,
        class_names=class_names
    )
    
    return gradcam

def run_gradcam_tests(model, test_loader, device, class_names):
    """
    Run a suite of GradCAM tests on the test dataset.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test dataset
        device: Device to run on
        class_names: Dictionary mapping class indices to names
    
    Returns:
        results: List of batch analysis results
        gradcam: GradCAM object from single image test
    """
    layer_name = find_last_conv_layer(model)
    
    # Test 1: Single image analysis
    print("\n1. Testing single image...")
    gradcam = test_gradcam_single_image(model, test_loader, device, class_names, layer_name)
    
    # Test 2: Batch analysis
    print("\n2. Batch analysis...")
    results, gradcam_batch = gradcam_batch_analysis(model, test_loader, device, class_names, layer_name)
    
    # Test 3: Visualize batch results
    print("\n3. Visualizing batch results...")
    visualize_gradcam_grid(results, gradcam_batch, class_names)
    
    # Test 4: Compare GradCAM for a single image
    print("\n4. Visualizing GradCAM for single image...")
    dataiter = iter(test_loader)
    images, _ = next(dataiter)
    input_tensor = images[0:1].to(device)
    compare_gradcam_layers(model, input_tensor, device, class_names, layer_name)
    
    return results, gradcam


def main():
    """
    Run GradCAM tests with a pre-trained ResNet50 model and an external image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()
    
    # Load model
    checkpoint_path = "/kaggle/input/best-model-without-least-3-classes/pytorch/default/1/checkpoints/best_model.pth"
    model_state = torch.load(checkpoint_path, map_location=device)
    model = DeepLiftCompatibleTripletClassifier(
        num_classes=config.NUM_CLASSES,
        model_name='resnet50',
        embedding_dim=512,
        state_dict=model_state,
        freeze_until=0
    ).to(device)
    
    # Load data
    csv_path = "/kaggle/working/fer2013/fer2013.csv"
    dm = FERDataModule(csv_path, batch_size=config.BATCH_SIZE)
    dm.setup()
    test_loader = dm.test_dataloader()
    
    # Class names
    class_names = {0: 'Happy', 1: 'Sad', 2: 'Surprise', 3: 'Neutral'}
    
    # Find the last convolutional layer
    layer_name = find_last_conv_layer(model)
    print(f"Using Grad-CAM layer: {layer_name}")
    
    # Run GradCAM tests on dataset
    results, gradcam_model = run_gradcam_tests(model, test_loader, device, class_names)
    
    # Apply GradCAM to external test images
    image_paths = [
        '/kaggle/input/happy-faces-test/Test image2/happy0.png', 
        '/kaggle/input/happy-faces-test/Test image2/happy1.png',
        '/kaggle/input/happy-faces-test/Test image2/happy2.png',
        '/kaggle/input/happy-faces-test/Test image2/happy3.png',
        '/kaggle/input/test-images/Test images/Screenshot 2025-07-23 212855.png',
        '/kaggle/input/test-images/Test images/happyFace.png',
        '/kaggle/input/test-images/Test images/sad face.png',
        '/kaggle/input/test-images/Test images/sadFace2.png',
        '/kaggle/input/test-images/Test images/surprisedFace1.png',
        '/kaggle/input/test-images/Test images/surprisedFace2.png',
        '/kaggle/input/sad-images-for-test/sad test images/sad0.png',
        '/kaggle/input/sad-images-for-test/sad test images/sad1.png',
        '/kaggle/input/sad-images-for-test/sad test images/sad2.png',
        '/kaggle/input/sad-images-for-test/sad test images/sad3.png',
        '/kaggle/input/sad-images-for-test/sad test images/sad4.jpg',
        '/kaggle/input/sad-images-for-test/sad test images/sad5.png',
        '/kaggle/input/sad-images-for-test/sad test images/sad6.png',
        '/kaggle/input/sad-images-for-test/sad test images/sad7.png',
        '/kaggle/input/sad-images-for-test/sad test images/sad8.png'
    ]
    for im in image_paths:
        external_image_path = im
        apply_gradcam_to_image(model, external_image_path, device, class_names, layer_name)
    
    return results, gradcam_model

if __name__ == "__main__":
    main()


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# from captum.attr import visualization as viz
# from PIL import Image
# import torchvision.transforms as transforms

# class GradCAM:
#     def __init__(self, model, target_layer_name):
#         """
#         Initialize GradCAM for a trained model.
        
#         Args:
#             model: Trained model (e.g., DeepLiftCompatibleTripletClassifier)
#             target_layer_name: Name of the layer to extract gradients from
#         """
#         self.model = model
#         self.target_layer_name = target_layer_name
#         self.gradients = None
#         self.activations = None
#         self.device = next(model.parameters()).device
#         self._register_hooks()
        
#     def _register_hooks(self):
#         """Register forward and backward hooks for the target layer."""
#         def backward_hook(module, grad_input, grad_output):
#             self.gradients = grad_output[0].detach()
            
#         def forward_hook(module, input, output):
#             self.activations = output.detach()
        
#         target_layer = self._find_target_layer(self.model, self.target_layer_name)
#         if target_layer is None:
#             raise ValueError(f"Layer '{self.target_layer_name}' not found in model")
            
#         target_layer.register_forward_hook(forward_hook)
#         target_layer.register_full_backward_hook(backward_hook)
    
#     def _find_target_layer(self, model, layer_name):
#         """Recursively find the target layer by name."""
#         target_parts = layer_name.split('.')
        
#         def recursive_get_module(module, parts):
#             if not parts:
#                 return module
#             try:
#                 return recursive_get_module(getattr(module, parts[0]), parts[1:])
#             except AttributeError:
#                 return None
        
#         for name, module in model.named_modules():
#             if name == layer_name:
#                 return module
        
#         module = recursive_get_module(model, target_parts)
#         if module is not None:
#             return module
        
#         for name, module in model.named_modules():
#             if layer_name in name:
#                 return module
                
#         return None
    
#     def generate_cam(self, input_tensor, target_class=None):
#         """
#         Generate Class Activation Map.
        
#         Args:
#             input_tensor: Input image tensor [1, 1, H, W] (grayscale)
#             target_class: Target class index (if None, uses predicted class)
            
#         Returns:
#             cam: Class activation map (numpy array)
#             output: Model output logits
#             target_class: Selected target class
#         """
#         input_tensor = input_tensor.to(self.device)
#         input_tensor.requires_grad_(True)
        
#         self.model.eval()
#         output = self.model(input_tensor)
        
#         if target_class is None:
#             target_class = output.argmax(dim=1).item()
        
#         self.model.zero_grad()
        
#         one_hot = torch.zeros_like(output)
#         one_hot[0][target_class] = 1
#         output.backward(gradient=one_hot, retain_graph=True)
        
#         gradients = self.gradients
#         activations = self.activations
        
#         weights = torch.mean(gradients, dim=[2, 3])
        
#         cam = torch.zeros(activations.shape[2:], dtype=torch.float32, device=self.device)
#         for i, w in enumerate(weights[0]):
#             cam += w * activations[0, i]
        
#         cam = F.relu(cam)
        
#         if cam.max() > 0:
#             cam = cam / cam.max()
        
#         return cam.cpu().numpy(), output, target_class
    
#     def visualize_cam(self, input_tensor, target_class=None, class_names=None, alpha=0.4, colormap='jet'):
#         """
#         Visualize GradCAM overlay on original image, showing only the overlay.
        
#         Args:
#             input_tensor: Input image tensor [1, 1, H, W]
#             target_class: Target class index
#             class_names: Dictionary mapping class indices to names
#             alpha: Transparency of overlay
#             colormap: Colormap for heatmap (e.g., 'jet')
            
#         Returns:
#             original_img: Original image (RGB)
#             cam: Raw CAM heatmap
#             heatmap: Colored heatmap
#             overlay: Overlay image
#             pred_class: Predicted class
#             confidence: Prediction confidence
#         """
#         cam, output, pred_class = self.generate_cam(input_tensor, target_class)
        
#         probs = F.softmax(output, dim=1)
#         confidence = probs[0, pred_class].item()
        
#         original_img = input_tensor.squeeze().cpu().detach().numpy()
#         original_img = (original_img / 2) + 0.5
#         original_img = np.clip(original_img, 0, 1)
#         original_img_rgb = np.stack([original_img] * 3, axis=-1)
        
#         cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
        
#         heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), getattr(cv2, f'COLORMAP_{colormap.upper()}'))
#         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#         heatmap = heatmap.astype(np.float32) / 255
        
#         overlay = alpha * heatmap + (1 - alpha) * original_img_rgb
        
#         fig, ax = plt.subplots(figsize=(4, 4))
        
#         viz.visualize_image_attr(
#             np.stack([cam_resized]*3, axis=-1), original_img_rgb,
#             method="blended_heat_map",
#             sign="all",
#             show_colorbar=True,
#             title=f"Pred: {class_names.get(pred_class, pred_class)} ({confidence:.3f})",
#             plt_fig_axis=(fig, ax),
#             use_pyplot=False
#         )
        
#         plt.tight_layout()
#         plt.show()
        
#         return original_img_rgb, cam_resized, heatmap, overlay, pred_class, confidence

# def apply_gradcam_to_image(model, image_path, device, class_names, layer_name='backbone.embedding_model.features.7.2.conv3'):
#     """
#     Apply GradCAM to an external test image not in the dataset, visualizing only the overlay.
    
#     Args:
#         model: Trained model (e.g., DeepLiftCompatibleTripletClassifier)
#         image_path: Path to the external image file
#         device: Device to run on (cuda or cpu)
#         class_names: Dictionary mapping class indices to names
#         layer_name: Name of the layer for GradCAM (default: last conv layer)
    
#     Returns:
#         gradcam: GradCAM object
#     """
#     try:
#         image = Image.open(image_path).convert('L')
#     except Exception as e:
#         print(f"Error loading image {image_path}: {e}")
#         return
    
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5], std=[0.5])
#     ])
    
#     image_tensor = transform(image).unsqueeze(0)
#     image_tensor = image_tensor.to(device)
    
#     gradcam = GradCAM(model, layer_name)
    
#     print(f"Applying GradCAM to external image: {image_path}")
#     original, cam, heatmap, overlay, pred_class, confidence = gradcam.visualize_cam(
#         input_tensor=image_tensor,
#         class_names=class_names
#     )
    
#     return gradcam

# def find_last_conv_layer(model):
#     """
#     Find the last Conv2d layer in the model.
    
#     Returns:
#         Layer name (e.g., 'backbone.embedding_model.features.7.2.conv3')
#     """
#     for name, module in reversed(list(model.named_modules())):
#         if isinstance(module, nn.Conv2d):
#             return name
#     return 'backbone.embedding_model.features.7.2.conv3'

# def main():
#     """
#     Run GradCAM tests on external images labeled as 'Sad' with a pre-trained ResNet50 model.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     config = Config()  # Assuming Config is defined elsewhere
    
#     # Load model
#     checkpoint_path = "/kaggle/input/best-model-without-least-3-classes/pytorch/default/1/checkpoints/best_model.pth"
#     model_state = torch.load(checkpoint_path, map_location=device)
#     model = DeepLiftCompatibleTripletClassifier(
#         num_classes=config.NUM_CLASSES,
#         model_name='resnet50',
#         embedding_dim=512,
#         state_dict=model_state,
#         freeze_until=0
#     ).to(device)
    
#     # Class names
#     class_names = {0: 'Happy', 1: 'Sad', 2: 'Surprise', 3: 'Neutral'}
    
#     # Find the last convolutional layer
#     layer_name = find_last_conv_layer(model)
#     print(f"Using Grad-CAM layer: {layer_name}")
    
#     # List of external images labeled as 'Sad'
#     sad_image_paths = [
#         '/kaggle/input/test-images/Test images/sad face.png',
#         '/kaggle/input/test-images/Test images/sadFace2.png',
#         # Additional placeholder paths for 'Sad' images (replace with actual paths if available)
#         '/kaggle/input/test-images/Test images/sadFace3.png',
#         '/kaggle/input/test-images/Test images/sadFace4.png',
#         '/kaggle/input/test-images/Test images/sadFace5.png'
#     ]
    
#     # Apply GradCAM to each 'Sad' image
#     for image_path in sad_image_paths:
#         apply_gradcam_to_image(model, image_path, device, class_names, layer_name)
    
#     return None, None  # Return None for consistency with original main function

# if __name__ == "__main__":
#     main()


def main():
    """
    Run GradCAM tests with a pre-trained ResNet50 model and an external image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()
    
    # Load model
    checkpoint_path = "/kaggle/input/best-model-without-least-3-classes/pytorch/default/1/checkpoints/best_model.pth"
    model_state = torch.load(checkpoint_path, map_location=device)
    model = DeepLiftCompatibleTripletClassifier(
        num_classes=config.NUM_CLASSES,
        model_name='resnet50',
        embedding_dim=512,
        state_dict=model_state,
        freeze_until=0
    ).to(device)
    
    # Load data
    csv_path = "/kaggle/working/fer2013/fer2013.csv"
    dm = FERDataModule(csv_path, batch_size=config.BATCH_SIZE)
    dm.setup()
    test_loader = dm.test_dataloader()
    
    # Class names
    class_names = {0: 'Happy', 1: 'Sad', 2: 'Surprise', 3: 'Neutral'}
    
    # Find the last convolutional layer
    layer_name = find_last_conv_layer(model)
    print(f"Using Grad-CAM layer: {layer_name}")
    
    # Run GradCAM tests on dataset
    results, gradcam_model = run_gradcam_tests(model, test_loader, device, class_names)
    
    # Apply GradCAM to external test images
    image_paths = [
        '/kaggle/input/surprise-test/surprise 1 test/Screenshot 2025-07-25 064422.png',
        '/kaggle/input/surprise-test/surprise 1 test/Screenshot 2025-07-25 064531.png',
        '/kaggle/input/surprise-test/surprise 1 test/Screenshot 2025-07-25 064611.png',
        '/kaggle/input/surprise-test/surprise 1 test/Screenshot 2025-07-25 064639.png',
        '/kaggle/input/surprise-test/surprise 1 test/Screenshot 2025-07-25 064709.png'
    ]
    for im in image_paths:
        external_image_path = im
        apply_gradcam_to_image(model, external_image_path, device, class_names, layer_name)
    
    return results, gradcam_model

if __name__ == "__main__":
    main()

