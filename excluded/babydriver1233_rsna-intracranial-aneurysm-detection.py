import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pydicom
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    data_dir = '/kaggle/input/rsna-intracranial-aneurysm-detection'
    batch_size = 4
    learning_rate = 1e-4
    num_epochs = 30
    image_size = (128, 128)
    num_slices = 32
    num_classes = 14
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 2

# Weighted Loss Function (missing from previous code)
class WeightedBCELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedBCELoss, self).__init__()
        self.weights = weights
    
    def forward(self, inputs, targets):
        loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        weighted_loss = loss * self.weights
        return weighted_loss.mean()

# Optimized Dataset
class AneurysmDataset(Dataset):
    def __init__(self, df, base_path, transform=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.base_path = base_path
        self.transform = transform
        self.is_train = is_train
        self.series_info = self._precompute_series_info()
        
    def _precompute_series_info(self):
        series_info = []
        for idx, row in self.df.iterrows():
            series_info.append({
                'series_id': row['SeriesInstanceUID'],
                'labels': row[[
                    'Left Infraclinoid Internal Carotid Artery',
                    'Right Infraclinoid Internal Carotid Artery',
                    'Left Supraclinoid Internal Carotid Artery',
                    'Right Supraclinoid Internal Carotid Artery',
                    'Left Middle Cerebral Artery',
                    'Right Middle Cerebral Artery',
                    'Anterior Communicating Artery',
                    'Left Anterior Cerebral Artery',
                    'Right Anterior Cerebral Artery',
                    'Left Posterior Communicating Artery',
                    'Right Posterior Communicating Artery',
                    'Basilar Tip',
                    'Other Posterior Circulation',
                    'Aneurysm Present'
                ]].values.astype(np.float32)
            })
        return series_info
        
    def __len__(self):
        return len(self.series_info)
    
    def load_dicom_series_efficient(self, series_id):
        series_path = os.path.join(self.base_path, 'series', series_id)
        if not os.path.exists(series_path):
            return np.zeros((3, Config.num_slices, Config.image_size[0], Config.image_size[1]), dtype=np.float32)
        
        try:
            dicom_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
            if not dicom_files:
                return np.zeros((3, Config.num_slices, Config.image_size[0], Config.image_size[1]), dtype=np.float32)
            
            # Sort files by instance number
            dicom_files.sort(key=lambda x: int(pydicom.dcmread(os.path.join(series_path, x), stop_before_pixels=True).InstanceNumber))
            
            # Sample slices
            num_files = len(dicom_files)
            step = max(1, num_files // Config.num_slices)
            selected_indices = range(0, min(num_files, Config.num_slices * step), step)
            selected_files = [dicom_files[i] for i in selected_indices[:Config.num_slices]]
            
            slices = []
            for file in selected_files:
                try:
                    dicom = pydicom.dcmread(os.path.join(series_path, file), stop_before_pixels=False)
                    img = dicom.pixel_array.astype(np.float32)
                    
                    if hasattr(dicom, 'RescaleSlope') and hasattr(dicom, 'RescaleIntercept'):
                        img = img * dicom.RescaleSlope + dicom.RescaleIntercept
                    
                    # Resize if needed
                    if img.shape != Config.image_size:
                        import cv2
                        img = cv2.resize(img, Config.image_size, interpolation=cv2.INTER_AREA)
                    
                    # Normalize
                    img_min, img_max = img.min(), img.max()
                    if img_max > img_min:
                        img = (img - img_min) / (img_max - img_min)
                    else:
                        img = np.zeros_like(img)
                    
                    slices.append(img)
                except Exception as e:
                    slices.append(np.zeros(Config.image_size, dtype=np.float32))
            
            # Pad if needed
            while len(slices) < Config.num_slices:
                slices.append(np.zeros(Config.image_size, dtype=np.float32))
            
            # Stack slices and add channel dimension
            volume = np.stack(slices, axis=0)  # Shape: [depth, height, width]
            volume = np.repeat(volume[np.newaxis, :, :, :], 3, axis=0)  # Shape: [channels, depth, height, width]
            
            return volume
            
        except Exception as e:
            print(f"Error loading series {series_id}: {e}")
            return np.zeros((3, Config.num_slices, Config.image_size[0], Config.image_size[1]), dtype=np.float32)
    
    def __getitem__(self, idx):
        series_info = self.series_info[idx]
        series_id = series_info['series_id']
        labels = series_info['labels']
        
        # Load DICOM volume - already in correct format [channels, depth, height, width]
        volume = self.load_dicom_series_efficient(series_id)
        
        if self.transform:
            volume = self.transform(volume)
        
        return torch.FloatTensor(volume), torch.FloatTensor(labels)

# Fixed 3D CNN Model with correct input handling
class SimpleAneurysm3DCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleAneurysm3DCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            
            # Block 2
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            
            # Block 3
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
        )
        
        # Calculate flattened size
        self._calculate_flattened_size()
        
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def _calculate_flattened_size(self):
        with torch.no_grad():
            # Input shape: [batch, channels, depth, height, width]
            mock_input = torch.zeros(1, 3, Config.num_slices, Config.image_size[0], Config.image_size[1])
            mock_output = self.features(mock_input)
            self.flattened_size = mock_output.numel()
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Training Function
def train_model_memory_efficient(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    best_score = 0
    train_losses = []
    val_scores = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Training phase
        optimizer.zero_grad()
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            images = images.to(Config.device)
            labels = labels.to(Config.device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            running_loss += loss.item() * images.size(0)
            
            # Update weights
            optimizer.step()
            optimizer.zero_grad()
                
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(Config.device)
                labels = labels.to(Config.device)
                
                outputs = model(images)
                preds = torch.sigmoid(outputs)
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # Calculate weighted AUC
        auc_scores = []
        for i in range(Config.num_classes):
            if len(np.unique(all_labels[:, i])) > 1:
                try:
                    auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                    auc_scores.append(auc)
                except:
                    auc_scores.append(0.5)
            else:
                auc_scores.append(0.5)
        
        weighted_auc = (auc_scores[-1] * 13 + sum(auc_scores[:-1])) / (13 + len(auc_scores) - 1)
        val_scores.append(weighted_auc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val AUC: {weighted_auc:.4f}')
        
        if weighted_auc > best_score:
            best_score = weighted_auc
            torch.save(model.state_dict(), 'best_model.pth')
        
        scheduler.step()
    
    return train_losses, val_scores

# Main execution
def main():
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(Config.data_dir, 'train.csv'))
    
    # Use smaller subset for testing
    sample_size = min(500, len(train_df))  # Even smaller for testing
    train_df = train_df.sample(sample_size, random_state=42).reset_index(drop=True)
    
    print(f"Using {len(train_df)} samples for training/validation")
    
    # Split data
    train_data, val_data = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df['Aneurysm Present']
    )
    
    # Simple transform
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x),  # Already converted to tensor in dataset
    ])
    
    # Create datasets
    train_dataset = AneurysmDataset(train_data, Config.data_dir, transform=transform)
    val_dataset = AneurysmDataset(val_data, Config.data_dir, transform=transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.batch_size, 
        shuffle=True, 
        num_workers=Config.num_workers,
        pin_memory=False  # Changed to False to avoid memory issues
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.batch_size, 
        shuffle=False, 
        num_workers=Config.num_workers,
        pin_memory=False
    )
    
    # Initialize model
    model = SimpleAneurysm3DCNN(Config.num_classes).to(Config.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss weights
    loss_weights = torch.ones(Config.num_classes).to(Config.device)
    loss_weights[-1] = 13.0
    criterion = WeightedBCELoss(loss_weights)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    print("Starting training...")
    train_losses, val_scores = train_model_memory_efficient(
        model, train_loader, val_loader, criterion, optimizer, scheduler, Config.num_epochs
    )
    
    print(f'Best validation score: {max(val_scores):.4f}')

if __name__ == "__main__":
    main()




