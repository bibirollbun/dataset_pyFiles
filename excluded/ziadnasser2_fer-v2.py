!tar -xvzf /kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013.tar.gz -C /kaggle/working


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
        self.transform = transform
        self.emotion_map = emotion_map or {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
                                           4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get label and convert to integer
        label = int(self.data.iloc[idx]['emotion'])
        
        # Convert pixel string to numpy array then to PIL image
        pixel_str = self.data.iloc[idx]['pixels']
        pixels = np.array(list(map(int, pixel_str.split())), dtype=np.uint8).reshape(48, 48)
        image = Image.fromarray(pixels)  # single channel image

        if self.transform:
            image = self.transform(image)

        return image, label


from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch

class FERDataModule:
    def __init__(self, csv_path, batch_size=32, image_size=128, criterion=None, train_transform=None, val_transform=None):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.criterion = criterion
        self.image_size = (image_size, image_size)
        self.emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
                            4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

        # Define transforms
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
                                            transform=self.train_transform, emotion_map=self.emotion_map)
        self.val_dataset = FERPlusDataset(self.csv_path, usage='PrivateTest',
                                          transform=self.val_transform, emotion_map=self.emotion_map)
        self.test_dataset = FERPlusDataset(self.csv_path, usage='PublicTest',
                                           transform=self.val_transform, emotion_map=self.emotion_map)

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")

    def train_dataloader(self, num_classes=7):
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
        return [self.emotion_map[i] for i in sorted(self.emotion_map.keys())]

    def get_class_weights(self):
        labels = [label for _, label in self.train_dataset]
        labels_tensor = torch.tensor(labels)
        counts = torch.bincount(labels_tensor, minlength=7)
        total = len(labels_tensor)
        weights = total / (counts.float() * 7)
        return weights


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
        return MixUpLoss(alpha=0.2, smoothing=smoothing)
    
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
    def __init__(self, model, train_loader, val_loader, config, device, criterion, EARLY_STOP_PATIENCE=None):
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
        
        # Enhanced optimizer with better regularization
        self.optimizer = optim.AdamW(  # AdamW instead of Adam for better regularization
            model.parameters(), 
            lr=self.config.LEARNING_RATE, 
            weight_decay=self.config.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Enhanced learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
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
        self.overfitting_threshold = 20.0  # If train_acc - val_acc > 15%, consider overfitting

    def train(self):
        best_f1 = 0.0
        best_epoch = 0
        best_acc = 0.0
        early_stop_counter = 0
        best_val_loss = float('inf')
        best_model = None

        for epoch in range(self.config.NUM_EPOCHS):
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

            # Update scheduler
            self.scheduler.step(val_loss)

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
                best_model = self.model
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
import torch
from tqdm import tqdm


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


import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class EmotionDetector(nn.Module):
    def __init__(self, num_classes=7, dropout_p=0.5):
        super(EmotionDetector, self).__init__()

        # Load pretrained ResNet-18
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Modify first conv layer for 1-channel grayscale input
        old_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        with torch.no_grad():
            self.model.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        # Replace the classifier with dropout + final FC
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


import os

class Config:
    def __init__(self):
        self.DATASET_PATH = "/kaggle/working/fer2013"
        self.MODEL_NAME = "EfficientNet"
        self.NUM_CLASSES = 7
        self.BATCH_SIZE = 64 
        self.NUM_EPOCHS = 60
        self.LEARNING_RATE = 1e-5  
        self.CRITERION = 'cross_entropy'  # Start simple, then experiment
        self.SEED = 42
        self.CHECKPOINT_PATH = "/kaggle/working/checkpoints"
        self.BEST_MODEL_PATH = os.path.join(self.CHECKPOINT_PATH, "best_model.pth")
        self.FINAL_MODEL_PATH = os.path.join(self.CHECKPOINT_PATH, "final_model.pth")
        self.USE_TQDM = False
        self.EARLY_STOP_PATIENCE = 15  # More aggressive early stopping
        self.WEIGHT_DECAY = 1e-4  # Increased L2 regularization
        self.IMAGE_SIZE = (128, 128)
        
        # New anti-overfitting parameters
        self.DROPOUT_RATE = 0.2  # Increased dropout
        self.LABEL_SMOOTHING = 0.1  # Add label smoothing
        self.GRADIENT_CLIPPING = 1.0  # Gradient clipping to prevent exploding gradients
        self.MIN_LR = 1e-7  # Minimum learning rate for scheduler
        
        os.makedirs(self.CHECKPOINT_PATH, exist_ok=True)


import os
import torch
import kagglehub
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
train_loader = data_module.train_dataloader(num_classes=config.NUM_CLASSES)
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()

# Initialize model
model = EmotionDetector(num_classes=config.NUM_CLASSES).to(device)

criterion = get_criterion(config, data_module.get_class_weights())

# Initialize trainer
trainer = Trainer(model, train_loader, val_loader, config, device, criterion)
history, best_epoch, best_f1, best_acc, best_model = trainer.train()

# Plot training performance
Utils.plot_loss_curve(history['train_loss'], history['val_loss'], show=True, save=True)
Utils.plot_accuracy_curve(history['train_acc'], history['val_acc'], show=True, save=True)
Utils.plot_f1_score(history['val_f1'], show=True, save=True)



tester = Test(best_model, test_loader, criterion, config)


# Run test and get predictions
test_loss, test_acc, precision, recall, f1, all_preds, all_labels = tester.test()

# Plot confusion matrix
Utils.plot_confusion_matrix(all_labels, all_preds, class_names=data_module.get_class_names())


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

# Convert to NumPy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Step 1: Filter out 'Disgust' class (label 1)
mask = all_labels != 1
filtered_preds = all_preds[mask]
filtered_labels = all_labels[mask]

# Step 2: Remap labels so class indices stay contiguous
def remap(label):
    return label - 1 if label > 1 else label

filtered_preds = np.array([remap(p) for p in filtered_preds])
filtered_labels = np.array([remap(l) for l in filtered_labels])

# Step 3: Compute metrics
acc = accuracy_score(filtered_labels, filtered_preds)
f1 = f1_score(filtered_labels, filtered_preds, average='weighted')
precision = precision_score(filtered_labels, filtered_preds, average='weighted')
recall = recall_score(filtered_labels, filtered_preds, average='weighted')

print(f"\nğŸ“Š Metrics without 'Disgust':")
print(f"Accuracy:  {acc:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

# Step 4: Plot confusion matrix for 6 classes
reduced_class_names = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
Utils.plot_confusion_matrix(filtered_labels, filtered_preds, class_names=reduced_class_names, show=True)


