import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import timm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


class EfficientDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        if 'label' in row:
            label = torch.tensor(row['label'], dtype=torch.float32)
            return image, label
        return image,  


class LightweightModel(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b7', num_classes=1):
        super(LightweightModel, self).__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(model_name, pretrained=True)
        
        if hasattr(self.backbone, 'classifier'):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.4),  # Increased dropout
                nn.Linear(in_features, num_classes)
            )
        else:
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.4),  # Increased dropout
                nn.Linear(in_features, num_classes)
            )
        
    def forward(self, x):
        return self.backbone(x)


def get_efficient_transforms(img_size):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5),  # Replace Cutout
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    valid_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transform, valid_transform


def train_model(model, train_loader, val_loader, device, config):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        epochs=config['epochs'],
        steps_per_epoch=len(train_loader),
        div_factor=25,
        final_div_factor=1e4
    )
    
    best_auc = 0
    early_stop_counter = 0
    metrics = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_auc': []}
    save_path = f"best_model_{model.model_name}.pth"
    
    writer = SummaryWriter(log_dir='logs')
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config["epochs"]}'):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
        
        train_acc = correct_train / total_train

        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
                preds = torch.sigmoid(outputs) > 0.5
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
        
        val_acc = correct_val / total_val
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        val_auc = roc_auc_score(val_labels, val_preds)
        
        metrics['train_loss'].append(train_loss / len(train_loader))
        metrics['val_loss'].append(val_loss / len(val_loader))
        metrics['train_acc'].append(train_acc)
        metrics['val_acc'].append(val_acc)
        metrics['val_auc'].append(val_auc)
        writer.add_scalar('Loss/train', metrics['train_loss'][-1], epoch)
        writer.add_scalar('Loss/val', metrics['val_loss'][-1], epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('AUC/val', val_auc, epoch)
        
        print(f'\nEpoch {epoch + 1}:')
        print(f'Train Loss: {metrics["train_loss"][-1]:.4f}, Train Accuracy: {train_acc:.4f}')
        print(f'Val Loss: {metrics["val_loss"][-1]:.4f}, Val Accuracy: {val_acc:.4f}, Val AUC: {val_auc:.4f}')
        
        if val_auc > best_auc:
            best_auc = val_auc
            early_stop_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'epoch': epoch
            }, save_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    writer.close()
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('training_metrics.csv', index=False)
    
    return best_auc, save_path


def predict(model, test_loader, device, weights_path):
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc='Predicting'): 
            images = images.to(device)
            outputs = model(images)
            predictions.extend(torch.sigmoid(outputs).cpu().numpy())
    
    return np.array(predictions)


def main():
    config = {
        'img_size': 224,
        'batch_size': 4,
        'epochs': 50,
        'learning_rate': 1e-3,
        'patience': 5,
        'weight_decay': 1e-4,
        'train_val_split': 0.2,
        'seed': 42
    }

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_df = pd.read_csv("/kaggle/input/cidaut-ai-fake-scene-classification-2024/train.csv")
    train_df['label'] = train_df['label'].map({"editada": 0, "real": 1})
    
    train_data, val_data = train_test_split(train_df, test_size=config['train_val_split'], random_state=config['seed'])
    train_transform, valid_transform = get_efficient_transforms(config['img_size'])
    
    train_dataset = EfficientDataset(
        train_data,
        "/kaggle/input/cidaut-ai-fake-scene-classification-2024/Train",
        transform=train_transform
    )
    val_dataset = EfficientDataset(
        val_data,
        "/kaggle/input/cidaut-ai-fake-scene-classification-2024/Train",
        transform=valid_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    model_name = 'tf_efficientnet_b7'
    print(f"\nTraining {model_name}")
    model = LightweightModel(model_name=model_name).to(device)

    score, weights_path = train_model(model, train_loader, val_loader, device, config)
    print(f"{model_name} - Best AUC: {score:.4f}")
    test_df = pd.read_csv("/kaggle/input/cidaut-ai-fake-scene-classification-2024/sample_submission.csv")
    test_dataset = EfficientDataset(
        test_df,
        "/kaggle/input/cidaut-ai-fake-scene-classification-2024/Test",
        transform=valid_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    predictions = predict(model, test_loader, device, weights_path)
    
    test_df['label'] = predictions.flatten()
    test_df.to_csv('submission.csv', index=False)
    print("\nPredictions saved to submission.csv")


if __name__ == "__main__":
    main()



import os
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# Define the validation transformation (same as in training)
img_size = 224
valid_transform = A.Compose([
    A.Resize(img_size, img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Path to the image to predict
image_path = "/kaggle/input/cidaut-ai-fake-scene-classification-2024/Train/100.jpg"

# Read and preprocess the image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
augmented = valid_transform(image=image)
image_tensor = augmented["image"]
image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

# Define the model (using the LightweightModel class defined earlier)
import timm
import torch.nn as nn

class LightweightModel(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b7', num_classes=1):
        super(LightweightModel, self).__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(model_name, pretrained=True)
        
        if hasattr(self.backbone, 'classifier'):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_features, num_classes)
            )
        else:
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_features, num_classes)
            )
        
    def forward(self, x):
        return self.backbone(x)

# Set up device and load the model with saved weights.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'tf_efficientnet_b7'
model = LightweightModel(model_name=model_name).to(device)

# Path to the saved model weights (adjust the path/filename as needed)
weights_path = f"best_model_{model_name}.pth"
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Model weights not found at {weights_path}")

checkpoint = torch.load(weights_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Perform prediction
with torch.no_grad():
    output = model(image_tensor.to(device))
    # Apply sigmoid activation to get a probability (since output is raw logits)
    probability = torch.sigmoid(output)
    # Since this is a binary classification task, probability indicates the score for class 1.
    predicted_prob = probability.item()

print(f"Predicted probability for image '{image_path}': {predicted_prob:.4f}")

