import numpy as np 
import pandas as pd
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchvision.models.efficientnet import efficientnet_b2, EfficientNet_B2_Weights
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_curve


base_path = '../input/alaska2-image-steganalysis'


def read_images_path(dir_name, label):
    folder_path = os.path.join(base_path, dir_name)
    return [[os.path.join(folder_path, filename), label] for filename in os.listdir(folder_path)]


def preprocess_image(image_path, size=(512, 512)):
    # Load image
    img = Image.open(image_path)
    
    # Convert to YCbCr color space (better for detecting steganography)
    img = img.convert("YCbCr")
    
    # Convert to numpy array with high precision (float64 to preserve subtle details)
    img_array = np.array(img, dtype=np.float64)
    
    # Extract DCT residuals to better detect steganography artifacts
    # This helps highlight the noise patterns that steganography introduces
    y_channel = img_array[:, :, 0]
    high_freq = y_channel - cv2.GaussianBlur(y_channel, (3, 3), 0)
    img_array[:, :, 0] = high_freq
    
    # Transpose channels from (H, W, C) to (C, H, W) for PyTorch
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Normalize to [0, 1] range but avoid standard ImageNet normalization
    # which could destroy subtle steganographic patterns
    img_array = img_array / 255.0
    
    # Convert to PyTorch tensor
    img_tensor = torch.tensor(img_array, dtype=torch.float32)
    
    return img_tensor


class SteganalysisDataset(Dataset):
    def __init__(self, image_paths, labels, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        # Get image path and label
        img_path = self.image_paths[index]
        label = self.labels[index]
        
        # Process image
        image = preprocess_image(img_path)
        
        # Apply data augmentation (only for training)
        if self.augment:
            # Flip horizontally with 50% probability (safe for steganalysis)
            if np.random.random() > 0.5:
                image = torch.flip(image, [2])
                
            # Random crop and resize back (safe for steganalysis as it preserves patterns)
            if np.random.random() > 0.5:
                i, j = torch.randint(0, 32, (2,))
                image = image[:, i:512-32+i, j:512-32+j]
                image = F.interpolate(image.unsqueeze(0), size=(512, 512), 
                                     mode='bilinear', align_corners=False).squeeze(0)
                
        return image, label


class SRMConv2D(nn.Module):
    def __init__(self):
        super(SRMConv2D, self).__init__()
        
        # Define SRM kernels (based on steganalysis research)
        kernel1 = torch.tensor([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=torch.float32)
        kernel2 = torch.tensor([[-1, 2, -1], [2, -4, 2], [0, 0, 0]], dtype=torch.float32)
        kernel3 = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        
        # Combine kernels into filter bank
        self.srm_kernels = nn.Parameter(torch.stack([kernel1, kernel2, kernel3]).unsqueeze(1), 
                                       requires_grad=False)
        
        # Define the convolution operation
        self.conv = nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=False)
        self.conv.weight = nn.Parameter(self.srm_kernels, requires_grad=False)
        
    def forward(self, x):
        # Apply SRM filters independently to each channel
        y_channel = self.conv(x[:, 0:1, :, :])
        cb_channel = self.conv(x[:, 1:2, :, :])
        cr_channel = self.conv(x[:, 2:3, :, :])
        
        # Combine results
        return torch.cat([y_channel, cb_channel, cr_channel], dim=1)


class SteganalysisModel(nn.Module):
    def __init__(self, num_classes=2):
        super(SteganalysisModel, self).__init__()
        
        # SRM preprocessing layer
        self.srm = SRMConv2D()
        
        # Load pretrained EfficientNet with updated API
        weights = EfficientNet_B2_Weights.DEFAULT  # hoặc IMAGENET1K_V1 nếu bạn cần bản cố định
        self.backbone = efficientnet_b2(weights=weights)
        
        # Modify first layer to accept 9 channels
        self.backbone.features[0][0] = nn.Conv2d(9, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Modified classifier with attention
        self.attention = nn.Sequential(
            nn.Conv2d(1408, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(1408, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.srm(x)
        features = self.backbone.features(x)
        attention_weights = self.attention(features)
        features = features * attention_weights
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        output = self.classifier(features)
        return output



def prepare_balanced_dataset(sample_size=15000):
    # Load cover images
    cover_img = read_images_path('Cover', 0)[:sample_size]
    
    # Load stego images (balanced among different techniques)
    stego_per_type = sample_size
    jmipod_img = read_images_path('JMiPOD', 1)[:stego_per_type]
    juniward_img = read_images_path('JUNIWARD', 1)[:stego_per_type]
    uerd_img = read_images_path('UERD', 1)[:stego_per_type]
    
    # Combine all images
    data = cover_img + jmipod_img + juniward_img + uerd_img
    df = pd.DataFrame(data=data, columns=['path', 'label'])
    
    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train and validation
    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size]
    val = df.iloc[train_size:]
    
    return train, val


def weighted_auc(y_true, y_score, tpr_thresholds=[0.0, 0.4, 1.0], weights=[2, 1]):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    # Calculate AUC for each region
    auc_scores = []
    for i in range(len(tpr_thresholds) - 1):
        start = tpr_thresholds[i]
        end = tpr_thresholds[i+1]
        
        # Filter points in current region
        mask = (tpr >= start) & (tpr < end)
        auc_scores.append(np.trapz(tpr[mask], fpr[mask]))
    
    # Calculate weighted AUC
    weighted_auc = np.sum(np.multiply(auc_scores, weights)) / np.sum(weights)
    
    return weighted_auc


def calculate_metrics(model, dataloader, device='cpu'):
    model.to(device).eval()
    labels, probs, preds = [], [], []

    with torch.no_grad():
        for images, batch_labels in dataloader:
            outputs = model(images.to(device))
            batch_probs = F.softmax(outputs, dim=1)[:, 1]
            batch_preds = torch.argmax(outputs, dim=1)

            labels.extend(batch_labels.cpu().numpy())
            probs.extend(batch_probs.cpu().numpy())
            preds.extend(batch_preds.cpu().numpy())

    labels = np.array(labels)
    probs = np.array(probs)
    preds = np.array(preds)

    return {
        'Weighted AUC': weighted_auc(labels, probs),
        'Accuracy': accuracy_score(labels, preds)
    }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Import OpenCV for noise extraction
import cv2

# Set hyperparameters
BATCH_SIZE = 16  # Smaller batch size for better gradient updates
EPOCHS = 10  # More epochs for better training
LR = 5e-5  # Learning rate
WEIGHT_DECAY = 1e-5  # Add regularization


# Prepare datasets
print("Preparing datasets...")
train, val = prepare_balanced_dataset(sample_size=20000)

# Create datasets
train_dataset = SteganalysisDataset(train['path'].tolist(), train['label'].tolist(), augment=True)
val_dataset = SteganalysisDataset(val['path'].tolist(), val['label'].tolist(), augment=False)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=4, pin_memory=True)

# Create model
model = SteganalysisModel(num_classes=2).to(device)

# Set up class weights to handle imbalance
# Higher weight for stego images (class 1)
WEIGHTS = torch.tensor([3, 1], dtype=torch.float32).to(device)

# Create optimizer with weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-4,
    epochs=EPOCHS,
    steps_per_epoch=len(train_dataloader),
    pct_start=0.3
)


train_losses = []
accuracies = []
weighted_aucs = []
best_auc = 0

for epoch in range(EPOCHS):
    print(f'EPOCH: {epoch+1}/{EPOCHS}')
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(0)/1e9:.2f} GB")
    
    # Training phase
    model.train()
    train_loss = 0
    
    for images, labels in tqdm(train_dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = F.cross_entropy(outputs, labels, weight=WEIGHTS)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        scheduler.step()  # <== CHUẨN OneCycleLR: step mỗi batch
        
        train_loss += loss.item()
        
    avg_train_loss = train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)
    print(f'Average Training Loss: {avg_train_loss:.4f}')
    
    # Validation
    model.eval()
    metrics = calculate_metrics(model, val_dataloader, device)
    acc = metrics['Accuracy']
    current_auc = metrics['Weighted AUC']
    
    accuracies.append(acc)
    weighted_aucs.append(current_auc)
    
    print(f'Validation Accuracy: {acc:.4f}')
    print(f'Validation Weighted AUC: {current_auc:.4f}')
    
    # Save best model
    if current_auc > best_auc:
        best_auc = current_auc
        torch.save(model.state_dict(), 'best_stego_model.pth')
        print(f"New best model saved with Weighted AUC: {best_auc:.4f}")
    
    print()

# Load best model
model.load_state_dict(torch.load('best_stego_model.pth'))
print("Best model loaded!")

# Save full model
torch.save(model, 'complete_steganalysis_model.pth')
print("Complete model saved to 'complete_steganalysis_model.pth'")



 # Plot training metrics
plt.figure(figsize=(15, 5))

# Plot Loss
plt.subplot(1, 3, 1)
plt.plot(range(1, EPOCHS+1), train_losses, marker='o', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()

# Plot Accuracy
plt.subplot(1, 3, 2)
plt.plot(range(1, EPOCHS+1), accuracies, marker='o', label='Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per Epoch')
plt.legend()

# Plot Weighted AUC
plt.subplot(1, 3, 3)
plt.plot(range(1, EPOCHS+1), weighted_aucs, marker='o', label='Weighted AUC', color='red')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('Weighted AUC per Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()


 # Prepare test predictions
test_img = read_images_path('Test', 0)
test = pd.DataFrame(data=test_img, columns=['path', 'label'])

# Create test dataset and dataloader
test_dataset = SteganalysisDataset(test['path'].tolist(), test['label'].tolist(), augment=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=4, pin_memory=True)


def predict(model, dataloader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            scores = F.softmax(outputs, dim=1)[:, 1]  # Probability of being stego
            predictions.extend(scores.cpu().numpy())
            
    return predictions

# Generate predictions
print("Generating predictions...")
test_predictions = predict(model, test_dataloader, device)
print("Predictions complete.")


# Create submission file
test_image_ids = [os.path.basename(path) for path in test_dataset.image_paths]
submission_df = pd.DataFrame({
    'Id': test_image_ids,
    'Label': test_predictions
})

# Format IDs as required by competition
submission_df['Id'] = submission_df['Id'].str.replace('.jpg', '').astype(int)
submission_df = submission_df.sort_values(by='Id')
submission_df['Id'] = submission_df['Id'].astype(str).str.zfill(4) + '.jpg'

# Save submission file
submission_df.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

