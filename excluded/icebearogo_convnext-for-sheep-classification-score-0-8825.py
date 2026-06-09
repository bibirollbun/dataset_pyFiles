import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import pandas as pd
from PIL import Image
import os
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set device and random seeds
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)
print(f"Using device: {device}")



class SheepDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Define the 7 sheep breeds
        self.breeds = ['Naeimi', 'Najdi', 'Harri', 'Goat', 'Sawakni', 'Roman', 'Barbari']
        self.label_to_idx = {breed: idx for idx, breed in enumerate(self.breeds)}
        self.idx_to_label = {idx: breed for idx, breed in enumerate(self.breeds)}
        
        print(f"Dataset initialized with {len(self.data)} samples")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (288, 288), (0, 0, 0))
        
        if 'label' in self.data.columns:
            label = self.data.iloc[idx]['label']
            label_idx = self.label_to_idx[label]
        else:
            label_idx = -1
            
        if self.transform:
            image = self.transform(image)
            
        return image, label_idx, img_name



# Training transforms with aggressive augmentation
train_transform = transforms.Compose([
    transforms.Resize((320, 320)),  # Larger initial size
    transforms.RandomResizedCrop(288, scale=(0.75, 1.0)),  # 288px like winning solution
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.25),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation transforms
val_transform = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean', weight=None):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight

    def linear_combination(self, x, y, epsilon):
        return epsilon * x + (1 - epsilon) * y

    def reduce_loss(self, loss, reduction='mean'):
        return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = torch.nn.functional.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = torch.nn.functional.nll_loss(log_preds, target, reduction=self.reduction, weight=self.weight)
        return self.linear_combination(loss / n, nll, self.epsilon)



class SheepConvNeXt(nn.Module):
    def __init__(self, num_classes=7, model_size='tiny', pretrained=True):
        super(SheepConvNeXt, self).__init__()
        
        # Load different ConvNeXt variants
        if model_size == 'tiny':
            self.convnext = models.convnext_tiny(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_size == 'small':
            self.convnext = models.convnext_small(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_size == 'base':
            self.convnext = models.convnext_base(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Replace classifier
        in_features = self.convnext.classifier[2].in_features
        self.convnext.classifier[2] = nn.Linear(in_features, num_classes)
        
        # Store original parameters for freezing/unfreezing
        self.backbone_params = list(self.convnext.features.parameters())
        self.classifier_params = list(self.convnext.classifier.parameters())
        
    def forward(self, x):
        return self.convnext(x)
    
    def freeze_backbone(self):
        """Freeze backbone for Phase 1 training"""
        for param in self.backbone_params:
            param.requires_grad = False
            
    def unfreeze_backbone_partial(self, unfreeze_ratio=0.3):
        """Unfreeze top layers for Phase 2 training"""
        # Get all backbone modules
        backbone_modules = list(self.convnext.features.modules())
        
        # Calculate how many modules to unfreeze (top 30%)
        num_to_unfreeze = int(len(backbone_modules) * unfreeze_ratio)
        modules_to_unfreeze = backbone_modules[-num_to_unfreeze:]
        
        # Unfreeze selected modules
        for module in modules_to_unfreeze:
            for param in module.parameters():
                param.requires_grad = True
        
        print(f"Unfroze top {unfreeze_ratio*100}% of backbone layers ({num_to_unfreeze} modules)")

# Initialize model
model = SheepConvNeXt(num_classes=7, model_size='tiny', pretrained=True).to(device)
print(f"ConvNeXt model parameters: {sum(p.numel() for p in model.parameters()):,}")



# Load and split data
train_df = pd.read_csv(r'C:\Users\demol\Desktop\sheep_class_k\Sheep Classification Images\train_labels_og.csv')
print(f"Original dataset: {len(train_df)} samples")
print("Breed distribution:")
print(train_df['label'].value_counts())

# Create stratified split
train_data, val_data = train_test_split(
    train_df, 
    test_size=0.2, 
    stratify=train_df['label'],
    random_state=42
)

# Calculate class weights for imbalanced dataset
y_train = train_data['label'].map({breed: idx for idx, breed in enumerate(['Naeimi', 'Najdi', 'Harri', 'Goat', 'Sawakni', 'Roman', 'Barbari'])}).values
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

print(f"Class weights: {dict(zip(['Naeimi', 'Najdi', 'Harri', 'Goat', 'Sawakni', 'Roman', 'Barbari'], class_weights))}")

# Save splits
train_data.to_csv('train_split.csv', index=False)
val_data.to_csv('val_split.csv', index=False)

# Create datasets and loaders
train_dataset = SheepDataset('train_split.csv', 'train/', transform=train_transform)
val_dataset = SheepDataset('val_split.csv', 'train/', transform=val_transform)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)



def train_epoch(model, train_loader, criterion, optimizer, device, phase="Phase 1"):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (images, labels, _) in enumerate(tqdm(train_loader, desc=f"Training {phase}")):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return running_loss / len(train_loader), accuracy, f1

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return running_loss / len(val_loader), accuracy, f1



# Phase 1: Train classifier only (3 epochs)
print("ğŸš€ Starting Phase 1: Classifier Training")
print("=" * 50)

model.freeze_backbone()
criterion_phase1 = LabelSmoothingCrossEntropy(epsilon=0.1, weight=class_weights_tensor)
optimizer_phase1 = optim.Adam(model.classifier_params, lr=5e-4, weight_decay=0.01)

phase1_epochs = 3
best_val_f1_phase1 = 0.0

for epoch in range(phase1_epochs):
    print(f"\nPhase 1 - Epoch {epoch+1}/{phase1_epochs}")
    print("-" * 40)
    
    train_loss, train_acc, train_f1 = train_epoch(
        model, train_loader, criterion_phase1, optimizer_phase1, device, "Phase 1"
    )
    
    val_loss, val_acc, val_f1 = validate_epoch(model, val_loader, criterion_phase1, device)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
    
    if val_f1 > best_val_f1_phase1:
        best_val_f1_phase1 = val_f1
        torch.save(model.state_dict(), 'convnext_phase1_best.pth')
        print(f"ğŸ�¯ Phase 1 - New best val F1: {best_val_f1_phase1:.4f}")

print(f"Phase 1 completed! Best validation F1: {best_val_f1_phase1:.4f}")

# Phase 2: Fine-tune with partial backbone unfreezing
print("\nğŸ”¥ Starting Phase 2: Fine-tuning")
print("=" * 50)

model.unfreeze_backbone_partial(unfreeze_ratio=0.3)
criterion_phase2 = LabelSmoothingCrossEntropy(epsilon=0.1, weight=class_weights_tensor)
optimizer_phase2 = optim.Adam(model.parameters(), lr=3e-5, weight_decay=0.01)

# Learning rate scheduler for Phase 2
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_phase2, mode='max', factor=0.5, patience=3
)

phase2_epochs = 30
best_val_f1_phase2 = best_val_f1_phase1
patience = 7
patience_counter = 0

train_losses, train_accs, train_f1s = [], [], []
val_losses, val_accs, val_f1s = [], [], []

for epoch in range(phase2_epochs):
    print(f"\nPhase 2 - Epoch {epoch+1}/{phase2_epochs}")
    print("-" * 40)
    
    train_loss, train_acc, train_f1 = train_epoch(
        model, train_loader, criterion_phase2, optimizer_phase2, device, "Phase 2"
    )
    
    val_loss, val_acc, val_f1 = validate_epoch(model, val_loader, criterion_phase2, device)
    
    # Store metrics
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    train_f1s.append(train_f1)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    val_f1s.append(val_f1)
    
    current_lr = optimizer_phase2.param_groups[0]['lr']
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
    print(f"Learning Rate: {current_lr:.6f}")
    
    # Update scheduler
    scheduler.step(val_f1)
    
    # Save best model
    if val_f1 > best_val_f1_phase2:
        best_val_f1_phase2 = val_f1
        torch.save(model.state_dict(), 'best_convnext_sheep_model.pth')
        print(f"ğŸ�¯ Phase 2 - New best val F1: {best_val_f1_phase2:.4f} - Model saved!")
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= patience:
        print(f"â�¹ï¸� Early stopping triggered after {patience} epochs without improvement")
        break
    
    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'convnext_checkpoint_epoch_{epoch+1}.pth')

print(f"\nğŸ�� Training completed! Best validation F1 score: {best_val_f1_phase2:.4f}")
print(f"Improvement over Phase 1: {best_val_f1_phase2 - best_val_f1_phase1:.4f}")



# Test dataset for predictions
class SheepTestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = [f for f in os.listdir(img_dir) 
                         if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))]
        self.img_names.sort()
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (288, 288), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
            
        return image, -1, img_name

# Load best model and generate predictions
model.load_state_dict(torch.load('best_convnext_sheep_model.pth'))
model.eval()

test_dataset = SheepTestDataset('test/', transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

def generate_predictions(model, test_loader, device, breeds):
    model.eval()
    predictions = []
    filenames = []
    confidences = []
    
    with torch.no_grad():
        for images, _, img_names in tqdm(test_loader, desc="Generating predictions"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
            for i, pred in enumerate(predicted):
                predictions.append(breeds[pred.item()])
                filenames.append(img_names[i])
                confidences.append(confidence[i].item())
    
    return filenames, predictions, confidences

breeds = ['Naeimi', 'Najdi', 'Harri', 'Goat', 'Sawakni', 'Roman', 'Barbari']
filenames, predictions, confidences = generate_predictions(model, test_loader, device, breeds)

# Create submission
submission_df = pd.DataFrame({
    'filename': filenames,
    'label': predictions
})

submission_df = submission_df.sort_values('filename').reset_index(drop=True)
submission_df.to_csv('convnext_sheep_submission.csv', index=False)

print(f"âœ… ConvNeXt submission created with {len(submission_df)} predictions")
print(f"Average confidence: {np.mean(confidences):.3f}")


