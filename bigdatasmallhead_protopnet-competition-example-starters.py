# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


"""
ProtoPNet Competition Starter Code - FIXED VERSION
Interpretable Bird Species Classification using Prototypical Part Networks
"""

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ===================== CONFIGURATION =====================
class Config:
    # Data paths
    TRAIN_PATH = '/kaggle/input/proto-p-net-competition/cub200_cropped/train_cropped_mini'
    TEST_PATH = '/kaggle/input/proto-p-net-competition/cub200_cropped/test_cropped_mini'
    
    # Model parameters
    NUM_CLASSES = 20  # 20 bird species
    NUM_PROTOTYPES = 200  # Total number of prototypes (10 per class)
    PROTOTYPES_PER_CLASS = 10
    PROTOTYPE_SHAPE = (128, 1, 1)  # Prototype dimensions
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 20
    WARMUP_EPOCHS = 5  # Epochs before prototype projection
    
    # Image parameters
    IMG_SIZE = 224
    
    # Loss weights
    CLST_WEIGHT = 0.8  # Clustering loss weight
    SEP_WEIGHT = -0.08  # Separation loss weight
    L1_WEIGHT = 1e-4  # L1 regularization weight

config = Config()

# ===================== DATASET =====================
class BirdDataset(Dataset):
    """Custom dataset for bird images"""
    
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.images = []
        self.labels = []
        self.class_names = []
        
        # Load all images and labels
        if os.path.exists(root_dir):
            self.class_names = sorted([d for d in os.listdir(root_dir) 
                                      if os.path.isdir(os.path.join(root_dir, d))])
            
            for class_idx, class_name in enumerate(self.class_names):
                class_dir = os.path.join(root_dir, class_name)
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        self.images.append(img_path)
                        self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, idx

class TestDataset(Dataset):
    """Test dataset without labels"""
    
    def __init__(self, root_dir, class_names, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.ids = []
        
        # Load test images from all class directories
        test_id = 0
        for class_name in class_names:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in sorted(os.listdir(class_dir)):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        self.images.append(img_path)
                        self.ids.append(test_id)
                        test_id += 1
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.ids[idx]

# ===================== PROTOPNET MODEL =====================
class ProtoPNet(nn.Module):
    """Prototypical Part Network for interpretable classification"""
    
    def __init__(self, num_classes, num_prototypes, prototype_shape):
        super(ProtoPNet, self).__init__()
        
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.prototype_shape = prototype_shape
        self.epsilon = 1e-4
        
        # Feature extractor (ResNet18 backbone)
        resnet = models.resnet18(pretrained=True)
        layers = list(resnet.children())[:-2]  # Remove avgpool and fc
        self.features = nn.Sequential(*layers)
        
        # Freeze early layers initially
        for param in self.features[:6].parameters():
            param.requires_grad = False
        
        # 1x1 convolution to match prototype dimensions
        self.add_on_layers = nn.Sequential(
            nn.Conv2d(512, prototype_shape[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(prototype_shape[0], prototype_shape[0], kernel_size=1),
            nn.Sigmoid()
        )
        
        # Prototype vectors (learnable)
        self.prototype_vectors = nn.Parameter(
            torch.rand(num_prototypes, prototype_shape[0], 
                      prototype_shape[1], prototype_shape[2]),
            requires_grad=True
        )
        
        # Initialize prototypes
        nn.init.xavier_normal_(self.prototype_vectors)
        
        # Prototype class identity (which prototypes belong to which class)
        self.prototype_class_identity = torch.zeros(num_prototypes, num_classes)
        prototypes_per_class = num_prototypes // num_classes
        
        for j in range(num_prototypes):
            class_idx = j // prototypes_per_class
            self.prototype_class_identity[j, class_idx] = 1
        
        # Last layer (fully connected)
        self.last_layer = nn.Linear(num_prototypes, num_classes, bias=False)
        
        # Initialize last layer weights
        self._initialize_last_layer()
    
    def _initialize_last_layer(self):
        """Initialize last layer with positive weights for same-class prototypes"""
        positive_one_weights = torch.ones_like(self.prototype_class_identity)
        negative_one_weights = -torch.ones_like(self.prototype_class_identity)
        
        self.last_layer.weight.data.copy_(
            torch.where(self.prototype_class_identity.t() > 0,
                       positive_one_weights.t(),
                       negative_one_weights.t())
        )
    
    def _l2_distance(self, x, y):
        """Compute L2 distance between feature maps and prototypes"""
        # x: [batch_size, num_prototypes, H, W]
        # y: [num_prototypes, C, 1, 1]
        
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        
        x = x.view(n, d, -1)  # [batch, channels, H*W]
        y = y.view(m, d)  # [num_prototypes, channels]
        
        distances = torch.zeros(n, m, x.size(2)).to(x.device)
        
        for i in range(n):
            for j in range(m):
                distances[i, j] = torch.sum((x[i] - y[j].unsqueeze(1)) ** 2, dim=0)
        
        return distances
    
    def prototype_distances(self, x):
        """Compute distances between input and prototypes"""
        # Extract features
        features = self.features(x)
        features = self.add_on_layers(features)
        
        # Compute distances
        distances = self._conv_l2_distance(features, self.prototype_vectors)
        
        return features, distances
    
    def _conv_l2_distance(self, x, prototypes):
        """Compute L2 distance using convolution"""
        n, c, h, w = x.shape
        m, c_p, h_p, w_p = prototypes.shape
        
        assert c == c_p, "Number of channels must match"
        
        # Unfold operation to get patches
        distances = torch.zeros(n, m, h, w).to(x.device)
        
        for i in range(m):
            prototype = prototypes[i:i+1]  # [1, C, 1, 1]
            # Compute squared L2 distance
            diff = x - prototype
            distances[:, i] = torch.sum(diff ** 2, dim=1)
        
        return distances
    
    def distance_2_similarity(self, distances):
        """Convert distances to similarities"""
        return torch.log((distances + 1) / (distances + self.epsilon))
    
    def forward(self, x):
        """Forward pass"""
        features, distances = self.prototype_distances(x)
        
        # Get minimum distance (maximum similarity) for each prototype
        min_distances = -F.max_pool2d(-distances, kernel_size=distances.size()[2:])
        min_distances = min_distances.view(min_distances.size(0), -1)
        
        # Convert to similarity scores
        prototype_activations = self.distance_2_similarity(min_distances)
        
        # Classification
        logits = self.last_layer(prototype_activations)
        
        return logits, min_distances, features
    
    def push_forward(self, x):
        """Forward pass for prototype pushing"""
        features, distances = self.prototype_distances(x)
        return features, distances

# ===================== LOSS FUNCTIONS =====================
class ProtoPLoss(nn.Module):
    """Custom loss for ProtoPNet"""
    
    def __init__(self, class_weights=None):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights)
        
    def forward(self, logits, min_distances, labels, prototype_class_identity,
                clst_weight=0.8, sep_weight=-0.08, l1_weight=1e-4, last_layer=None):
        
        # Cross-entropy loss
        ce_loss = self.cross_entropy(logits, labels)
        
        # Cluster loss (minimize distance to same-class prototypes)
        # Get prototypes for correct class for each sample in batch
        prototypes_of_correct_class = prototype_class_identity[:, labels].t()
        
        inverted_distances = -min_distances
        clst_distances = torch.max(inverted_distances * prototypes_of_correct_class, dim=1)[0]
        clst_loss = torch.mean(clst_distances)
        
        # Separation loss (maximize distance to different-class prototypes)
        prototypes_of_wrong_class = 1 - prototypes_of_correct_class
        sep_distances = torch.max(inverted_distances * prototypes_of_wrong_class, dim=1)[0]
        sep_loss = torch.mean(sep_distances)
        
        # L1 regularization on last layer
        l1_loss = 0
        if last_layer is not None:
            l1_loss = l1_weight * torch.sum(torch.abs(last_layer.weight))
        
        # Total loss
        total_loss = ce_loss + clst_weight * clst_loss + sep_weight * sep_loss + l1_loss
        
        return total_loss, ce_loss, clst_loss, sep_loss

# ===================== PROTOTYPE PROJECTION =====================
def project_prototypes(model, dataloader, class_specific=True):
    """Project prototypes onto nearest training patches"""
    print("Projecting prototypes onto nearest training patches...")
    
    model.eval()
    
    # Collect all features and labels
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            features, _ = model.push_forward(images)
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Project each prototype
    num_prototypes = model.num_prototypes
    prototype_shape = model.prototype_shape
    
    for j in tqdm(range(num_prototypes), desc="Projecting prototypes"):
        if class_specific:
            # Find the class this prototype belongs to
            class_idx = j // (num_prototypes // model.num_classes)
            class_mask = all_labels == class_idx
            class_features = all_features[class_mask]
        else:
            class_features = all_features
        
        if len(class_features) == 0:
            continue
        
        # Reshape features for comparison
        n, c, h, w = class_features.shape
        class_features_reshaped = class_features.permute(0, 2, 3, 1).reshape(-1, c)
        
        # Find nearest patch
        prototype = model.prototype_vectors[j].view(-1).cpu()
        distances = torch.sum((class_features_reshaped - prototype) ** 2, dim=1)
        min_idx = torch.argmin(distances)
        
        # Update prototype
        model.prototype_vectors.data[j] = class_features_reshaped[min_idx].view(
            prototype_shape[0], prototype_shape[1], prototype_shape[2]
        ).to(device)
    
    return model

# ===================== TRAINING FUNCTIONS =====================
def train_epoch(model, dataloader, optimizer, criterion, epoch, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
    
    for images, labels, _ in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, min_distances, features = model(images)
        
        # Compute loss
        loss, ce_loss, clst_loss, sep_loss = criterion(
            logits, min_distances, labels,
            model.prototype_class_identity.to(device),
            clst_weight=config.CLST_WEIGHT,
            sep_weight=config.SEP_WEIGHT,
            l1_weight=config.L1_WEIGHT,
            last_layer=model.last_layer
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), correct / total

def validate(model, dataloader):
    """Validate the model"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            
            logits, _, _ = model(images)
            _, predicted = torch.max(logits.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

# ===================== VISUALIZATION =====================
def visualize_prototypes(model, dataloader, num_visualize=5):
    """Visualize learned prototypes and their activations"""
    model.eval()
    
    # Get a batch of images
    images, labels, _ = next(iter(dataloader))
    images = images.to(device)
    
    with torch.no_grad():
        logits, min_distances, features = model(images)
        _, predicted = torch.max(logits, 1)
    
    # Visualize first few images and their top activated prototypes
    fig, axes = plt.subplots(num_visualize, 4, figsize=(12, 3*num_visualize))
    
    for i in range(min(num_visualize, len(images))):
        # Original image
        img = images[i].cpu().permute(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original (Class {labels[i]})')
        axes[i, 0].axis('off')
        
        # Get top 3 activated prototypes
        similarities = -min_distances[i]
        top_prototypes = torch.topk(similarities, 3)[1]
        
        for j, proto_idx in enumerate(top_prototypes):
            # Create a heatmap showing prototype activation
            proto_class = proto_idx // (model.num_prototypes // model.num_classes)
            
            axes[i, j+1].imshow(img)
            axes[i, j+1].set_title(f'Proto {proto_idx} (Class {proto_class})')
            axes[i, j+1].axis('off')
    
    plt.tight_layout()
    plt.savefig('prototype_visualizations.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print("Prototype visualizations saved!")

# ===================== MAIN TRAINING PIPELINE =====================
def main():
    """Main training pipeline"""
    
    print("=" * 50)
    print("ProtoPNet Competition - Training Pipeline")
    print("=" * 50)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = BirdDataset(config.TRAIN_PATH, transform=train_transform, is_train=True)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Number of classes: {len(train_dataset.class_names)}")
    
    # Split train into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                          shuffle=False, num_workers=2)
    
    # Initialize model
    print("Initializing ProtoPNet model...")
    model = ProtoPNet(
        num_classes=config.NUM_CLASSES,
        num_prototypes=config.NUM_PROTOTYPES,
        prototype_shape=config.PROTOTYPE_SHAPE
    ).to(device)
    
    # Loss and optimizer
    criterion = ProtoPLoss()
    
    # Different learning rates for different parts
    optimizer_specs = [
        {'params': model.features.parameters(), 'lr': config.LEARNING_RATE / 10},
        {'params': model.add_on_layers.parameters(), 'lr': config.LEARNING_RATE},
        {'params': model.prototype_vectors, 'lr': config.LEARNING_RATE},
        {'params': model.last_layer.parameters(), 'lr': config.LEARNING_RATE}
    ]
    optimizer = optim.Adam(optimizer_specs)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    print("\nStarting training...")
    print("=" * 50)
    
    best_val_acc = 0
    train_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, 
                                           criterion, epoch, config)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_acc = validate(model, val_loader)
        val_accs.append(val_acc)
        
        print(f'\nEpoch {epoch+1}/{config.NUM_EPOCHS}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_protopnet_model.pth')
            print(f'Best model saved! (Val Acc: {val_acc:.4f})')
        
        # Prototype projection every few epochs after warmup
        if epoch >= config.WARMUP_EPOCHS and (epoch + 1) % 5 == 0:
            model = project_prototypes(model, train_loader, class_specific=True)
            print("Prototypes projected!")
            
            # Re-initialize last layer after projection
            model._initialize_last_layer()
        
        scheduler.step()
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Visualize prototypes
    print("\nVisualizing learned prototypes...")
    visualize_prototypes(model, val_loader, num_visualize=5)
    
    return model, train_dataset

# ===================== INFERENCE AND SUBMISSION =====================
def create_submission(model, test_path, train_class_names, output_file='submission.csv'):
    """Create submission file for Kaggle"""
    
    print("\n" + "=" * 50)
    print("Creating submission file...")
    
    # Test transform
    test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset
    test_dataset = TestDataset(test_path, train_class_names, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, 
                           shuffle=False, num_workers=2)
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Make predictions
    model.eval()
    all_predictions = []
    all_ids = []
    
    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            logits, _, _ = model(images)
            _, predicted = torch.max(logits, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_ids.extend(ids.numpy() if isinstance(ids, torch.Tensor) else ids)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': all_ids,
        'label': all_predictions
    })
    
    # Sort by id
    submission = submission.sort_values('id')
    
    # Save submission
    submission.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
    print(f"Shape: {submission.shape}")
    print("\nFirst few predictions:")
    print(submission.head(10))
    
    return submission

# ===================== PROTOTYPE ANALYSIS =====================
def analyze_prototypes(model, dataloader, save_path='prototype_analysis.txt'):
    """Analyze which prototypes are most activated for each class"""
    
    print("\n" + "=" * 50)
    print("Analyzing prototype activations...")
    
    model.eval()
    
    # Collect prototype activations per class
    class_prototype_activations = {i: [] for i in range(config.NUM_CLASSES)}
    
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc='Collecting activations'):
            images = images.to(device)
            logits, min_distances, _ = model(images)
            
            # Convert distances to similarities
            similarities = -min_distances
            
            for i, label in enumerate(labels):
                class_prototype_activations[label.item()].append(
                    similarities[i].cpu().numpy()
                )
    
    # Analyze activations
    analysis_results = []
    
    for class_idx in range(config.NUM_CLASSES):
        if len(class_prototype_activations[class_idx]) > 0:
            activations = np.stack(class_prototype_activations[class_idx])
            mean_activations = np.mean(activations, axis=0)
            
            # Get top 5 prototypes for this class
            top_prototypes = np.argsort(mean_activations)[-5:][::-1]
            
            analysis_results.append(f"Class {class_idx}:")
            analysis_results.append(f"  Top 5 prototypes: {top_prototypes.tolist()}")
            analysis_results.append(f"  Mean activations: {mean_activations[top_prototypes].tolist()}")
            analysis_results.append("")
    
    # Save analysis
    with open(save_path, 'w') as f:
        f.write('\n'.join(analysis_results))
    
    print(f"Prototype analysis saved to {save_path}")
    print("\nSample analysis:")
    for line in analysis_results[:12]:
        print(line)

# ===================== RUN EVERYTHING =====================
if __name__ == "__main__":
    # Train the model
    model, train_dataset = main()
    
    # Load best model for inference
    print("\nLoading best model for inference...")
    model.load_state_dict(torch.load('best_protopnet_model.pth'))
    
    # Get class names from original dataset
    original_dataset = BirdDataset(config.TRAIN_PATH)
    train_class_names = original_dataset.class_names
    
    # Create submission
    submission = create_submission(model, config.TEST_PATH, train_class_names)
    
    # Analyze prototypes
    val_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = BirdDataset(config.TRAIN_PATH, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                          shuffle=False, num_workers=2)
    
    analyze_prototypes(model, val_loader)
    
    print("\n" + "=" * 50)
    print("Pipeline completed successfully!")
    print("Files generated:")
    print("  - submission.csv (Kaggle submission)")
    print("  - best_protopnet_model.pth (Trained model)")
    print("  - training_history.png (Training plots)")
    print("  - prototype_visualizations.png (Prototype visualizations)")
    print("  - prototype_analysis.txt (Prototype activation analysis)")
    print("=" * 50)

