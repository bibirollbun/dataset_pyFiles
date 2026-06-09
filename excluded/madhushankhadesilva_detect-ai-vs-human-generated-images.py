import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import roc_auc_score, accuracy_score
import cv2
import os
from tqdm import tqdm
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from typing import Optional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


class AIDetectionDataset(Dataset):
    def __init__(self, df, images_dir, transform=None, mode='train'):
        self.data = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['file_name']
        if self.mode == 'train':
            label = self.data.iloc[idx]['label']
        else:
            label = 0  # dummy for test
            
        img_name = os.path.basename(img_name)
            
        img_path = os.path.join(self.images_dir, img_name)
        
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Could not load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # fallback image
            image = np.ones((224, 224, 3), dtype=np.uint8) * 128
        
        if self.transform:
            image = self.transform(image=image)['image']
            
        if self.mode == 'test':
            return image, img_name
        else:
            return image, torch.tensor(label, dtype=torch.float32)


def get_transforms():
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.Affine(
            translate_percent=(0.05, 0.05),
            scale=(0.9, 1.1),
            rotate=(-20, 20), 
            shear=(-5, 5),
            p=0.4
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2, 
            p=0.3
        ),
        A.HueSaturationValue(  
            hue_shift_limit=0.1,
            sat_shift_limit=0.2,
            p=0.3
        ),
        A.GaussianBlur(blur_limit=7, p=0.2), 
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform


BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2


train_csv = "/kaggle/input/detect-ai-vs-human-generated-images/train.csv"
test_csv = "/kaggle/input/detect-ai-vs-human-generated-images/test.csv"
train_images_dir = '/kaggle/input/ai-vs-human-generated-dataset/train_data/'
test_images_dir = '/kaggle/input/ai-vs-human-generated-dataset/test_data_v2/'

train = pd.read_csv(train_csv, index_col = 0)
train.head(10)


train_transform, val_transform = get_transforms()


train_df, val_df = train_test_split(
    train, test_size=0.2, random_state=42, stratify=train['label']
)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

train_dataset = AIDetectionDataset(train_df, train_images_dir, train_transform)
val_dataset = AIDetectionDataset(val_df, train_images_dir, val_transform)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, 
    shuffle=True, num_workers=NUM_WORKERS
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, 
    shuffle=False, num_workers=NUM_WORKERS
)


for images, labels in train_loader:
    print(images.shape)  
    print(labels.shape)  
    break


class VariationalBottleneck(nn.Module):
    def __init__(self, in_dim, bottleneck_dim=256, beta=0.1):
        super().__init__()
        self.beta = beta
        self.mean = nn.Linear(in_dim, bottleneck_dim)
        self.logvar = nn.Linear(in_dim, bottleneck_dim)
        self.decoder = nn.Linear(bottleneck_dim, in_dim)
        self.kl_loss = torch.tensor(0.0)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, training=True):
        mu = self.mean(x)
        logvar = self.logvar(x)
        z = self.reparameterize(mu, logvar) if training else mu
        recon = self.decoder(z)
        
        if training:
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            self.kl_loss = kl * self.beta
        else:
            self.kl_loss = torch.tensor(0.0, device=x.device)
        
        return recon

class SpectralBranch(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.embed = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1),
            nn.Linear(3, embed_dim // 2)
        )
        self.mse_loss = torch.tensor(0.0)
    
    def forward(self, x: torch.Tensor, training=True) -> torch.Tensor:
        B, C, H, W = x.shape
        fft_orig = torch.fft.rfft2(x, dim=(-2, -1))  # complex [B,C,H,freq_W]
        
        freq_W = fft_orig.size(-1)
        center_h, center_w = H // 2, freq_W // 2
        low_mask = torch.zeros(1, 1, H, freq_W, device=x.device)
        low_mask[:, :, max(0, center_h-8):min(H, center_h+8), 
                 max(0, center_w-8):min(freq_W, center_w+8)] = 1.0
        high_mask = 1 - low_mask
        
        low_freq = fft_orig * low_mask
        high_freq = fft_orig * high_mask
        
        if training:
            low_recon = torch.fft.irfft2(low_freq, s=(H, W))
            high_recon = torch.fft.irfft2(high_freq, s=(H, W))
            self.mse_loss = F.mse_loss(low_recon, x) + F.mse_loss(high_recon, x)
        else:
            self.mse_loss = torch.tensor(0.0, device=x.device)
        
        low_freq_abs = torch.abs(low_freq)
        high_freq_abs = torch.abs(high_freq)
        low_emb = self.embed(low_freq_abs)
        high_emb = self.embed(high_freq_abs)
        return torch.cat([low_emb, high_emb], dim=1)
        
class AIDetector(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, beta=0.1):
        super(AIDetector, self).__init__()
        
        self.backbone = timm.create_model('vit_base_patch16_224', 
                                         pretrained=pretrained,
                                         num_classes=0)
        feature_dim = 768
        
        self.spectral_branch = SpectralBranch(feature_dim)
        
        fused_dim = feature_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.LayerNorm(fused_dim // 2),
            nn.ReLU()
        )
        
        self.vib = VariationalBottleneck(fused_dim // 2, beta=beta)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(fused_dim // 2, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )
        
        self.training_mode = True
        self.total_aux_loss = torch.tensor(0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.spectral_branch.training = self.training_mode  # Note: this sets the module's training flag
        self.vib.training = self.training_mode
        
        spec_feats = self.spectral_branch(x, training=self.training_mode)
        
        spatial_feats = self.backbone(x)
        
        fused = torch.cat([spatial_feats, spec_feats], dim=1)
        fused = self.fusion(fused)
        
        vib_out = self.vib(fused, training=self.training_mode)
        
        output = self.classifier(vib_out)
        return output
    
    def get_aux_loss(self):
        kl = self.vib.kl_loss
        spec_mse = self.spectral_branch.mse_loss
        self.total_aux_loss = kl + spec_mse
        return self.total_aux_loss


class AIDetectionTrainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=config['lr'], 
                                   weight_decay=config['weight_decay'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['epochs']
        )
        
        self.best_auc = 0
        self.train_losses = []
        self.val_metrics = []
        
    def train_epoch(self):
        self.model.train()
        self.model.training_mode = True
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.float().to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images).squeeze()
            bce_loss = self.criterion(outputs, labels)
            aux_loss = self.model.get_aux_loss()
            total_loss = bce_loss + aux_loss
            
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            running_loss += total_loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'Loss': f'{total_loss.item():.4f}'})
        
        self.model.training_mode = False
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_auc = roc_auc_score(all_labels, all_preds)
        
        return epoch_loss, epoch_auc
    
    def validate(self):
        self.model.eval()
        self.model.training_mode = False
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images).squeeze()
                
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        auc = roc_auc_score(all_labels, all_preds)
        predictions = [1 if p > 0.5 else 0 for p in all_preds]
        accuracy = accuracy_score(all_labels, predictions)
        
        return auc, accuracy
    
    def train(self):
        print("Starting training...")
        
        for epoch in range(self.config['epochs']):
            print(f'\nEpoch {epoch+1}/{self.config["epochs"]}')
            
            # Train
            train_loss, train_auc = self.train_epoch()
            
            # Validate
            val_auc, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            print(f'Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}')
            print(f'Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}')
            
            # Save best model
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_auc': self.best_auc,
                    'epoch': epoch
                }, 'best_model.pth')
                print(f'New best model saved with AUC: {val_auc:.4f}')
            
            self.train_losses.append(train_loss)
            self.val_metrics.append({'auc': val_auc, 'acc': val_acc})


config = {
    'epochs': EPOCHS,
    'lr': LR,
    'weight_decay': WEIGHT_DECAY
}

model = AIDetector(num_classes=1)
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

trainer = AIDetectionTrainer(model, train_loader, val_loader, device, config)

trainer.train()


test_df = pd.read_csv("/kaggle/input/ai-vs-human-generated-dataset/test.csv")
print("Test shape:", test_df.shape)
print(test_df.head())

test_images_dir = "/kaggle/input/ai-vs-human-generated-dataset/"

_, val_transform = get_transforms()


model = AIDetector(num_classes=1, pretrained=False)
checkpoint_path = "/kaggle/working/best_model.pth" 
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()
model.training_mode = False  # Disable aux for inference
print("Model loaded successfully from:", checkpoint_path)


class AIDetectionTestDataset(torch.utils.data.Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['id'].strip()
        img_path = os.path.join(self.images_dir, img_name)

        try:
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Fallback: Gray placeholder (224x224x3)
            image = np.ones((224, 224, 3), dtype=np.uint8) * 128

        if self.transform:
            image = self.transform(image=image)['image']

        return image, img_name


test_dataset = AIDetectionTestDataset(
    df=test_df,
    images_dir=test_images_dir,
    transform=val_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


predictions = []
ids = []

with torch.no_grad():
    for images, img_names in tqdm(test_loader, desc="Predicting"):
        images = images.to(device)
        outputs = model(images).squeeze(1)  

        
        probs = outputs.cpu().numpy()

        
        preds = (probs >= 0.50).astype(int)

        predictions.extend(preds)
        ids.extend(img_names)


submission = pd.DataFrame({
    "id": ids,
    "label": predictions
})

submission.to_csv("submission.csv", index=False)
print("Submission file created successfully!")
print("Shape:", submission.shape)
print(submission.head())


assert list(submission.columns) == ["id", "label"], "Submission format incorrect!"
assert not submission.isnull().any().any(), "Missing values detected!"
print("Ready for Kaggle submission.")


# import os
# import torch
# from torch.utils.data import DataLoader
# import pandas as pd
# from tqdm import tqdm
# import numpy as np

# # ======================
# # 1ï¸�âƒ£ Setup device
# # ======================
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Using device:", device)

# # ======================
# # 2ï¸�âƒ£ Load test CSV
# # ======================
# test_df = pd.read_csv("/kaggle/input/ai-vs-human-generated-dataset/test.csv")
# print("Test shape:", test_df.shape)
# print(test_df.head())

# # ======================
# # 3ï¸�âƒ£ Define image directory
# # ======================
# test_images_dir = "/kaggle/input/ai-vs-human-generated-dataset/"

# # ======================
# # 4ï¸�âƒ£ Load transforms
# # ======================
# _, val_transform = get_transforms()

# # ======================
# # 5ï¸�âƒ£ Load trained model
# # ======================
# model = AIDetector(num_classes=1, pretrained=False)
# checkpoint_path = "/kaggle/input/aivh/pytorch/default/1/best_model (1).pth"
# checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# if 'model_state_dict' in checkpoint:
#     model.load_state_dict(checkpoint['model_state_dict'])
# else:
#     model.load_state_dict(checkpoint)

# model.to(device)
# model.eval()
# print("âœ… Model loaded successfully from:", checkpoint_path)

# # ======================
# # 6ï¸�âƒ£ Define custom test dataset
# # ======================
# class AIDetectionTestDataset(torch.utils.data.Dataset):
#     def __init__(self, df, images_dir, transform=None):
#         self.df = df
#         self.images_dir = images_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         img_name = self.df.iloc[idx]['id'].strip()
#         img_path = os.path.join(self.images_dir, img_name)

#         image = cv2.imread(img_path)
#         if image is None:
#             raise FileNotFoundError(f"Image not found: {img_path}")

#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         if self.transform:
#             image = self.transform(image=image)['image']

#         return image, img_name

# # ======================
# # 7ï¸�âƒ£ Create dataset & loader
# # ======================
# test_dataset = AIDetectionTestDataset(
#     df=test_df,
#     images_dir=test_images_dir,
#     transform=val_transform
# )

# test_loader = DataLoader(
#     test_dataset,
#     batch_size=32,
#     shuffle=False,
#     num_workers=2,
#     pin_memory=True
# )

# # ======================
# # 8ï¸�âƒ£ Inference loop
# # ======================
# predictions = []
# ids = []

# with torch.no_grad():
#     for images, img_names in tqdm(test_loader, desc="Predicting"):
#         images = images.to(device)
#         outputs = model(images).squeeze(1)

#         # Ensure outputs are probabilities
#         probs = torch.sigmoid(outputs).cpu().numpy()

#         # Convert to binary (0 = human, 1 = AI)
#         preds = (probs >= 0.55).astype(int)

#         predictions.extend(preds)
#         ids.extend(img_names)

# # ======================
# # 9ï¸�âƒ£ Create submission
# # ======================
# submission = pd.DataFrame({
#     "id": ids,
#     "label": predictions
# })

# submission.to_csv("submission.csv", index=False)
# print("âœ… Submission file created successfully!")
# print("Shape:", submission.shape)
# print(submission.head())

# # ======================
# # âœ… Final check
# # ======================
# # Ensure correct column names and no index
# assert list(submission.columns) == ["id", "label"], "Submission format incorrect!"
# assert not submission.isnull().any().any(), "Missing values detected!"
# print("âœ… Ready for Kaggle submission.")



# import numpy as np
# import pandas as pd 
# from sklearn.model_selection import train_test_split
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
# import cv2
# import os
# from tqdm import tqdm
# import timm
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')

# import numpy.core.multiarray
# torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Using device: {device}')


# class Config:
#     # Data
#     train_csv = "/kaggle/input/detect-ai-vs-human-generated-images/train.csv"
#     test_csv = "/kaggle/input/detect-ai-vs-human-generated-images/test.csv"
#     train_images_dir = '/kaggle/input/ai-vs-human-generated-dataset/train_data/'
#     test_images_dir = '/kaggle/input/ai-vs-human-generated-dataset/test_data_v2/'
    
#     # Model
#     model_name = 'tf_efficientnet_b4'
#     img_size = 384
#     num_classes = 1
    
#     # Training
#     batch_size = 16
#     epochs = 15
#     lr = 2e-4
#     weight_decay = 1e-5
#     num_workers = 0
    
# config = Config()


# def get_transforms(phase='train'):
#     if phase == 'train':
#         return A.Compose([
#             A.Resize(config.img_size, config.img_size),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.2),
#             A.RandomRotate90(p=0.3),
#             A.Affine(
#                 translate_percent=(0.05, 0.05),
#                 scale=(0.9, 1.1),
#                 rotate=(-15, 15),
#                 shear=(-5, 5),
#                 interpolation=1,
#                 border_mode=0,
#                 p=0.4
#             ),
#             A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
#             A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
#             A.GaussianBlur(blur_limit=3, p=0.2),
#             A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.2),
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ToTensorV2(),
#         ])
#     else:
#         return A.Compose([
#             A.Resize(config.img_size, config.img_size),
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ToTensorV2(),
#         ])

# class AIDetectionDataset(Dataset):
#     def __init__(self, df, images_dir, transform=None, phase='train'):
#         self.data = df.reset_index(drop=True)
#         self.images_dir = images_dir
#         self.transform = transform
#         self.phase = phase
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         # Handle different column names for train and test
#         if 'file_name' in self.data.columns:
#             img_name = self.data.iloc[idx]['file_name']
#         elif 'id' in self.data.columns:
#             img_name = self.data.iloc[idx]['id']
#         else:
#             # Try to find any column that might contain filenames
#             for col in self.data.columns:
#                 if 'name' in col.lower() or 'file' in col.lower() or 'id' in col.lower() or 'path' in col.lower():
#                     img_name = self.data.iloc[idx][col]
#                     break
#             else:
#                 # Last resort - use first column
#                 img_name = self.data.iloc[idx].iloc[0]
        
#         # Extract filename from path
#         img_name = os.path.basename(img_name)
        
#         if self.phase == 'train':
#             label = self.data.iloc[idx]['label']
#         else:
#             label = 0
            
#         img_path = os.path.join(self.images_dir, img_name)
        
#         # Robust image loading with error handling
#         try:
#             image = cv2.imread(img_path)
#             if image is None:
#                 raise ValueError(f"Could not load image: {img_path}")
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         except Exception as e:
#             # Create fallback image
#             image = np.ones((config.img_size, config.img_size, 3), dtype=np.uint8) * 128
        
#         if self.transform:
#             image = self.transform(image=image)['image']
            
#         if self.phase == 'test':
#             return image, img_name
#         else:
#             return image, torch.tensor(label, dtype=torch.float32)


# class AdvancedAIDetector(nn.Module):
#     def __init__(self, model_name=config.model_name, num_classes=config.num_classes, pretrained=True):
#         super().__init__()
        
#         self.backbone = timm.create_model(
#             model_name, 
#             pretrained=pretrained,
#             num_classes=0,
#             global_pool=''
#         )
        
#         # Get feature dimensions
#         if 'efficientnet' in model_name:
#             feature_dim = 1792
#             self.global_pool = nn.AdaptiveAvgPool2d(1)
#         elif 'convnext' in model_name:
#             feature_dim = 1024
#             self.global_pool = nn.AdaptiveAvgPool2d(1)
#         else:
#             feature_dim = 768
#             self.global_pool = nn.Identity()
        
#         # Attention mechanism
#         self.attention = nn.Sequential(
#             nn.Linear(feature_dim, feature_dim // 4),
#             nn.ReLU(inplace=True),
#             nn.Linear(feature_dim // 4, 1),
#             nn.Sigmoid()
#         )
        
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Linear(feature_dim, 512),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm1d(512),
#             nn.Dropout(0.2),
#             nn.Linear(512, 128),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm1d(128),
#             nn.Dropout(0.1),
#             nn.Linear(128, num_classes),
#         )
        
#     def forward(self, x):
#         features = self.backbone(x)
        
#         if isinstance(features, tuple):
#             features = features[0]
            
#         # Global pooling
#         if hasattr(self, 'global_pool') and not isinstance(self.global_pool, nn.Identity):
#             features = self.global_pool(features)
#             features = features.view(features.size(0), -1)
        
#         # Apply attention
#         attention_weights = self.attention(features)
#         attended_features = features * attention_weights
        
#         # Classification
#         output = self.classifier(attended_features)
#         return output

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#         self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
        
#     def forward(self, inputs, targets):
        
#         bce_loss = self.bce_with_logits(inputs, targets.unsqueeze(1))
        
#         # Convert to probabilities
#         pt = torch.exp(-bce_loss)
        
#         # Focal loss component
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss


# class Trainer:
#     def __init__(self, model, train_loader, val_loader, device, config):
#         self.model = model.to(device)
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.device = device
#         self.config = config
        
#         self.criterion = nn.BCEWithLogitsLoss()
        
#         # Optimizer
#         self.optimizer = optim.AdamW(
#             model.parameters(), 
#             lr=config.lr, 
#             weight_decay=config.weight_decay
#         )
        
#         # Scheduler
#         self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
#             self.optimizer, T_0=config.epochs//3, T_mult=1
#         )
        
#         self.best_auc = 0
#         self.train_history = []
        
#     def train_epoch(self):
#         self.model.train()
#         running_loss = 0.0
#         all_preds = []
#         all_labels = []
        
#         pbar = tqdm(self.train_loader, desc='Training')
#         for batch_idx, (images, labels) in enumerate(pbar):
#             images, labels = images.to(self.device), labels.to(self.device)
            
#             self.optimizer.zero_grad()
            
#             # Forward pass - get raw logits
#             outputs = self.model(images).squeeze()
            
#             loss = self.criterion(outputs, labels)
            
#             loss.backward()
            
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
#             self.optimizer.step()
            
#             running_loss += loss.item()
            
#             preds = torch.sigmoid(outputs.detach())
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
            
#             pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
#         epoch_loss = running_loss / len(self.train_loader)
#         epoch_auc = roc_auc_score(all_labels, all_preds)
        
#         return epoch_loss, epoch_auc
    
#     def validate(self):
#         self.model.eval()
#         all_preds = []
#         all_labels = []
        
#         with torch.no_grad():
#             for images, labels in tqdm(self.val_loader, desc='Validation'):
#                 images, labels = images.to(self.device), labels.to(self.device)
                
#                 outputs = self.model(images).squeeze()
#                 # Apply sigmoid to get probabilities
#                 preds = torch.sigmoid(outputs)
                
#                 all_preds.extend(preds.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
        
#         auc = roc_auc_score(all_labels, all_preds)
#         predictions = [1 if p > 0.5 else 0 for p in all_preds]
#         accuracy = accuracy_score(all_labels, predictions)
#         f1 = f1_score(all_labels, predictions)
        
#         return auc, accuracy, f1
    
#     def train(self):
#         print("Starting training...")
        
#         for epoch in range(self.config.epochs):
#             print(f'\nEpoch {epoch+1}/{self.config.epochs}')
            
#             # Train
#             train_loss, train_auc = self.train_epoch()
            
#             # Validate
#             val_auc, val_acc, val_f1 = self.validate()
            
#             # Update scheduler
#             self.scheduler.step()
            
#             print(f'Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}')
#             print(f'Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
            
#             # Save best model with PyTorch 2.6 compatibility
#             if val_auc > self.best_auc:
#                 self.best_auc = val_auc
#                 self._save_model(epoch, val_auc)
#                 print(f' New best model saved with AUC: {val_auc:.4f}')
            
#             self.train_history.append({
#                 'epoch': epoch,
#                 'train_loss': train_loss,
#                 'train_auc': train_auc,
#                 'val_auc': val_auc,
#                 'val_acc': val_acc,
#                 'val_f1': val_f1
#             })
    
#     def _save_model(self, epoch, auc):
#         """Safe model saving for PyTorch 2.6"""
#         checkpoint = {
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'best_auc': self.best_auc,
#             'epoch': epoch
#         }
#         torch.save(checkpoint, 'best_model.pth', weights_only=False)


# def prepare_data():
#     """Prepare and validate data loading"""
#     print("Loading data...")
    
#     # Load CSVs
#     train_df = pd.read_csv(config.train_csv, index_col=0)
#     test_df = pd.read_csv(config.test_csv)
    
#     print(f"Train data shape: {train_df.shape}")
#     print(f"Test data shape: {test_df.shape}")
#     print(f"Train columns: {train_df.columns.tolist()}")
#     print(f"Test columns: {test_df.columns.tolist()}")
    
#     # Debug image loading
#     def debug_images(df, images_dir, num_samples=3):
#         print(f"\nğŸ”� Debugging image loading from {images_dir}:")
#         successful = 0
        
#         for i in range(min(num_samples, len(df))):
#             # Handle different column names
#             if 'file_name' in df.columns:
#                 img_name = os.path.basename(df.iloc[i]['file_name'])
#             elif 'id' in df.columns:
#                 img_name = os.path.basename(df.iloc[i]['id'])
#             else:
#                 # Try to find filename column
#                 for col in df.columns:
#                     if 'name' in col.lower() or 'file' in col.lower() or 'id' in col.lower():
#                         img_name = os.path.basename(df.iloc[i][col])
#                         break
#                 else:
#                     img_name = os.path.basename(df.iloc[i].iloc[0])
            
#             img_path = os.path.join(images_dir, img_name)
#             image = cv2.imread(img_path)
#             if image is not None:
#                 print(f" {img_name} - Shape: {image.shape}")
#                 successful += 1
#             else:
#                 print(f"{img_name} - Failed to load")
#         return successful > 0
    
#     # Test image loading
#     print("\nTesting train images:")
#     train_success = debug_images(train_df, config.train_images_dir)
    
#     print("\nTesting test images:")
#     test_success = debug_images(test_df, config.test_images_dir)
    
#     if not train_success or not test_success:
#         raise Exception("Image loading failed. Check paths and file structure.")
    
#     # Split data
#     train_data, val_data = train_test_split(
#         train_df, test_size=0.2, random_state=42, stratify=train_df['label']
#     )
    
#     print(f"\n Data split:")
#     print(f"  Training samples: {len(train_data)}")
#     print(f"  Validation samples: {len(val_data)}")
#     print(f"  Test samples: {len(test_df)}")
    
#     # Get transforms
#     train_transform = get_transforms('train')
#     val_transform = get_transforms('valid')
    
#     # Create datasets
#     train_dataset = AIDetectionDataset(train_data, config.train_images_dir, train_transform, 'train')
#     val_dataset = AIDetectionDataset(val_data, config.train_images_dir, val_transform, 'val')
#     test_dataset = AIDetectionDataset(test_df, config.test_images_dir, val_transform, 'test')
    
#     # Create data loaders
#     train_loader = DataLoader(
#         train_dataset, batch_size=config.batch_size, 
#         shuffle=True, num_workers=config.num_workers
#     )
#     val_loader = DataLoader(
#         val_dataset, batch_size=config.batch_size, 
#         shuffle=False, num_workers=config.num_workers
#     )
#     test_loader = DataLoader(
#         test_dataset, batch_size=config.batch_size, 
#         shuffle=False, num_workers=config.num_workers
#     )
    
#     return train_loader, val_loader, test_loader, test_df


# def create_submission(model, test_loader, test_df):
#     """Create submission file"""
#     print("Creating submission...")
    
#     model.eval()
#     predictions = []
#     filenames = []
    
#     with torch.no_grad():
#         for images, names in tqdm(test_loader, desc='Predicting'):
#             images = images.to(device)
#             outputs = model(images).squeeze()
#             # Apply sigmoid to get probabilities
#             preds = torch.sigmoid(outputs)
#             predictions.extend(preds.cpu().numpy())
#             filenames.extend(names)
    
#     # Create submission
#     submission = pd.DataFrame({
#         'filename': filenames,
#         'label': predictions
#     })
    
#     # Ensure we have predictions for all test samples
#     if len(submission) != len(test_df):
#         print(f"Warning: Submission length {len(submission)} doesn't match test data {len(test_df)}")
#         # Create submission from test_df with default predictions
#         if 'id' in test_df.columns:
#             submission = pd.DataFrame({
#                 'filename': test_df['id'].apply(os.path.basename),
#                 'label': [0.5] * len(test_df)
#             })
#         else:
#             submission = pd.DataFrame({
#                 'filename': [f"image_{i}.jpg" for i in range(len(test_df))],
#                 'label': [0.5] * len(test_df)
#             })
    
#     submission.to_csv('submission.csv', index=False)
#     print("Submission file created: submission.csv")
#     print(f"Submission shape: {submission.shape}")
#     print(submission.head())
    
#     return submission


# def load_model_safely(model_path='best_model.pth'):
#     """Safe model loading for PyTorch 2.6"""
#     model = AdvancedAIDetector()
    
#     try:
#         # Method 1: Direct load with weights_only=False
#         checkpoint = torch.load(model_path, map_location=device, weights_only=False)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         print("Model loaded successfully")
#         return model, checkpoint
#     except Exception as e:
#         print(f"Method 1 failed: {e}")
#         try:
#             # Method 2: Use safe globals context
#             with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
#                 checkpoint = torch.load(model_path, map_location=device)
#                 model.load_state_dict(checkpoint['model_state_dict'])
#                 print("Model loaded with safe globals")
#                 return model, checkpoint
#         except Exception as e2:
#             print(f"Method 2 failed: {e2}")
#             print("Model loading failed, returning untrained model")
#             return model, None


# def main():
#     """Complete training and submission pipeline"""
#     print("ğŸš€ Starting AI Detection Pipeline...")
    
#     try:
#         # Prepare data
#         train_loader, val_loader, test_loader, test_df = prepare_data()
        
#         # Test data loading
#         print("\n Testing data pipeline...")
#         test_batch = next(iter(train_loader))
#         images, labels = test_batch
#         print(f" Data loading test passed")
#         print(f"   Images shape: {images.shape}")
#         print(f"   Labels shape: {labels.shape}")
#         print(f"   Label range: {labels.min().item():.0f} to {labels.max().item():.0f}")
        
#         # Create and train model
#         print("\nğŸ�¯ Creating model...")
#         model = AdvancedAIDetector()
#         print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
#         # Train model
#         trainer = Trainer(model, train_loader, val_loader, device, config)
#         trainer.train()
        
#         # Create submission with best model
#         print("\n Generating submission...")
#         # Load best model
#         best_model, checkpoint = load_model_safely('best_model.pth')
#         if checkpoint:
#             print(f"Best model AUC: {checkpoint.get('best_auc', 'Unknown'):.4f}")
        
#         submission = create_submission(best_model, test_loader, test_df)
        
#         print(f"\n Pipeline completed successfully!")
#         print(f" Best validation AUC: {trainer.best_auc:.4f}")
        
#     except Exception as e:
#         print(f" Pipeline failed: {e}")
#         import traceback
#         traceback.print_exc()


# def quick_submission():
#     """Quick submission if model already trained"""
#     print("ğŸš€ Creating quick submission...")
    
#     # Load data
#     test_df = pd.read_csv(config.test_csv)
#     val_transform = get_transforms('valid')
#     test_dataset = AIDetectionDataset(test_df, config.test_images_dir, val_transform, 'test')
#     test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
#     # Load model
#     model, checkpoint = load_model_safely('best_model.pth')
#     model.to(device)
    
#     if checkpoint:
#         print(f"Loaded model with AUC: {checkpoint.get('best_auc', 'Unknown'):.4f}")
    
#     # Create submission
#     submission = create_submission(model, test_loader, test_df)
#     return submission


# if __name__ == '__main__':
#     # Check if model already exists
#     if os.path.exists('best_model.pth'):
#         print("Found existing model, creating submission...")
#         quick_submission()
#     else:
#         print("No existing model found, starting training...")
#         main()


# import numpy as np
# import pandas as pd 
# from sklearn.model_selection import train_test_split
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
# import cv2
# import os
# from tqdm import tqdm
# import timm
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')

# import numpy.core.multiarray
# torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Using device: {device}')


# class Config:
#     # Data
#     train_csv = "/kaggle/input/detect-ai-vs-human-generated-images/train.csv"
#     test_csv = "/kaggle/input/detect-ai-vs-human-generated-images/test.csv"
#     train_images_dir = '/kaggle/input/ai-vs-human-generated-dataset/train_data/'
#     test_images_dir = '/kaggle/input/ai-vs-human-generated-dataset/test_data_v2/'
    
#     # Model
#     model_name = 'tf_efficientnet_b4'
#     img_size = 384
#     num_classes = 1
    
#     # Training
#     batch_size = 16
#     epochs = 15
#     lr = 2e-4
#     weight_decay = 1e-5
#     num_workers = 0  # Set to 0 for stability
    
# config = Config()


# def get_transforms(phase='train'):
#     if phase == 'train':
#         return A.Compose([
#             A.Resize(config.img_size, config.img_size),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.2),
#             A.RandomRotate90(p=0.3),
#             A.Affine(
#                 translate_percent=(0.05, 0.05),
#                 scale=(0.9, 1.1),
#                 rotate=(-15, 15),
#                 shear=(-5, 5),
#                 interpolation=1,
#                 border_mode=0,
#                 p=0.4
#             ),
#             A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
#             A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
#             A.GaussianBlur(blur_limit=3, p=0.2),
#             A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.2),
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ToTensorV2(),
#         ])
#     else:
#         return A.Compose([
#             A.Resize(config.img_size, config.img_size),
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ToTensorV2(),
#         ])

# class AIDetectionDataset(Dataset):
#     def __init__(self, df, images_dir, transform=None, phase='train'):
#         self.data = df.reset_index(drop=True)
#         self.images_dir = images_dir
#         self.transform = transform
#         self.phase = phase
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         img_name = self.data.iloc[idx]['file_name']
        
#         # Extract filename from path
#         img_name = os.path.basename(img_name)
        
#         if self.phase == 'train':
#             label = self.data.iloc[idx]['label']
#         else:
#             label = 0
            
#         img_path = os.path.join(self.images_dir, img_name)
        
#         # Robust image loading with error handling
#         try:
#             image = cv2.imread(img_path)
#             if image is None:
#                 raise ValueError(f"Could not load image: {img_path}")
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         except Exception as e:
#             # Create fallback image
#             image = np.ones((config.img_size, config.img_size, 3), dtype=np.uint8) * 128
        
#         if self.transform:
#             image = self.transform(image=image)['image']
            
#         if self.phase == 'test':
#             return image, img_name
#         else:
#             return image, torch.tensor(label, dtype=torch.float32)


# class AdvancedAIDetector(nn.Module):
#     def __init__(self, model_name=config.model_name, num_classes=config.num_classes, pretrained=True):
#         super().__init__()
        
#         self.backbone = timm.create_model(
#             model_name, 
#             pretrained=pretrained,
#             num_classes=0,
#             global_pool=''
#         )
        
#         # Get feature dimensions
#         if 'efficientnet' in model_name:
#             feature_dim = 1792
#             self.global_pool = nn.AdaptiveAvgPool2d(1)
#         elif 'convnext' in model_name:
#             feature_dim = 1024
#             self.global_pool = nn.AdaptiveAvgPool2d(1)
#         else:
#             feature_dim = 768
#             self.global_pool = nn.Identity()
        
#         # Attention mechanism
#         self.attention = nn.Sequential(
#             nn.Linear(feature_dim, feature_dim // 4),
#             nn.ReLU(inplace=True),
#             nn.Linear(feature_dim // 4, 1),
#             nn.Sigmoid()
#         )
        
#         # Classifier
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Linear(feature_dim, 512),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm1d(512),
#             nn.Dropout(0.2),
#             nn.Linear(512, 128),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm1d(128),
#             nn.Dropout(0.1),
#             nn.Linear(128, num_classes),
#             nn.Sigmoid()
#         )
        
#     def forward(self, x):
#         features = self.backbone(x)
        
#         if isinstance(features, tuple):
#             features = features[0]
            
#         # Global pooling - FIXED SYNTAX ERROR
#         if hasattr(self, 'global_pool') and not isinstance(self.global_pool, nn.Identity):
#             features = self.global_pool(features)
#             features = features.view(features.size(0), -1)
        
#         # Apply attention
#         attention_weights = self.attention(features)
#         attended_features = features * attention_weights
        
#         # Classification
#         output = self.classifier(attended_features)
#         return output

# # Focal Loss for imbalanced data
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
        
#     def forward(self, inputs, targets):
#         BCE_loss = nn.BCEWithLogitsLoss()(inputs, targets.unsqueeze(1))
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
#         if self.reduction == 'mean':
#             return torch.mean(F_loss)
#         elif self.reduction == 'sum':
#             return torch.sum(F_loss)
#         else:
#             return F_loss


# class Trainer:
#     def __init__(self, model, train_loader, val_loader, device, config):
#         self.model = model.to(device)
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.device = device
#         self.config = config
        
#         # Loss function
#         self.criterion = FocalLoss()
        
#         # Optimizer
#         self.optimizer = optim.AdamW(
#             model.parameters(), 
#             lr=config.lr, 
#             weight_decay=config.weight_decay
#         )
        
#         # Scheduler
#         self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
#             self.optimizer, T_0=config.epochs//3, T_mult=1
#         )
        
#         self.best_auc = 0
#         self.train_history = []
        
#     def train_epoch(self):
#         self.model.train()
#         running_loss = 0.0
#         all_preds = []
#         all_labels = []
        
#         pbar = tqdm(self.train_loader, desc='Training')
#         for batch_idx, (images, labels) in enumerate(pbar):
#             images, labels = images.to(self.device), labels.to(self.device)
            
#             self.optimizer.zero_grad()
#             outputs = self.model(images).squeeze()
#             loss = self.criterion(outputs, labels)
#             loss.backward()
            
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
#             self.optimizer.step()
            
#             running_loss += loss.item()
            
#             # Collect predictions
#             preds = torch.sigmoid(outputs.detach())
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
            
#             pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
#         epoch_loss = running_loss / len(self.train_loader)
#         epoch_auc = roc_auc_score(all_labels, all_preds)
        
#         return epoch_loss, epoch_auc
    
#     def validate(self):
#         self.model.eval()
#         all_preds = []
#         all_labels = []
        
#         with torch.no_grad():
#             for images, labels in tqdm(self.val_loader, desc='Validation'):
#                 images, labels = images.to(self.device), labels.to(self.device)
                
#                 outputs = self.model(images).squeeze()
#                 preds = torch.sigmoid(outputs)
                
#                 all_preds.extend(preds.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
        
#         auc = roc_auc_score(all_labels, all_preds)
#         predictions = [1 if p > 0.5 else 0 for p in all_preds]
#         accuracy = accuracy_score(all_labels, predictions)
#         f1 = f1_score(all_labels, predictions)
        
#         return auc, accuracy, f1
    
#     def train(self):
#         print("Starting training...")
        
#         for epoch in range(self.config.epochs):
#             print(f'\nEpoch {epoch+1}/{self.config.epochs}')
            
#             # Train
#             train_loss, train_auc = self.train_epoch()
            
#             # Validate
#             val_auc, val_acc, val_f1 = self.validate()
            
#             # Update scheduler
#             self.scheduler.step()
            
#             print(f'Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}')
#             print(f'Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
            
#             # Save best model with PyTorch 2.6 compatibility
#             if val_auc > self.best_auc:
#                 self.best_auc = val_auc
#                 self._save_model(epoch, val_auc)
#                 print(f'âœ… New best model saved with AUC: {val_auc:.4f}')
            
#             self.train_history.append({
#                 'epoch': epoch,
#                 'train_loss': train_loss,
#                 'train_auc': train_auc,
#                 'val_auc': val_auc,
#                 'val_acc': val_acc,
#                 'val_f1': val_f1
#             })
    
#     def _save_model(self, epoch, auc):
#         """Safe model saving for PyTorch 2.6"""
#         checkpoint = {
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'best_auc': self.best_auc,
#             'epoch': epoch
#         }
#         torch.save(checkpoint, 'best_model.pth', weights_only=False)


# def prepare_data():
#     """Prepare and validate data loading"""
#     print("Loading data...")
    
#     # Load CSVs
#     train_df = pd.read_csv(config.train_csv, index_col=0)
#     test_df = pd.read_csv(config.test_csv)
    
#     print(f"Train data shape: {train_df.shape}")
#     print(f"Test data shape: {test_df.shape}")
#     print(f"Train columns: {train_df.columns.tolist()}")
#     print(f"Test columns: {test_df.columns.tolist()}")
    
#     # Debug image loading
#     def debug_images(df, images_dir, num_samples=3):
#         print(f"\nğŸ”� Debugging image loading from {images_dir}:")
#         successful = 0
#         for i in range(min(num_samples, len(df))):
#             img_name = os.path.basename(df.iloc[i]['file_name'])
#             img_path = os.path.join(images_dir, img_name)
#             image = cv2.imread(img_path)
#             if image is not None:
#                 print(f" {img_name} - Shape: {image.shape}")
#                 successful += 1
#             else:
#                 print(f"{img_name} - Failed to load")
#         return successful > 0
    
#     # Test image loading
#     train_success = debug_images(train_df, config.train_images_dir)
#     test_success = debug_images(test_df, config.test_images_dir)
    
#     if not train_success or not test_success:
#         raise Exception("Image loading failed. Check paths and file structure.")
    
#     # Split data
#     train_data, val_data = train_test_split(
#         train_df, test_size=0.2, random_state=42, stratify=train_df['label']
#     )
    
#     print(f"\n Data split:")
#     print(f"  Training samples: {len(train_data)}")
#     print(f"  Validation samples: {len(val_data)}")
#     print(f"  Test samples: {len(test_df)}")
    
#     # Get transforms
#     train_transform = get_transforms('train')
#     val_transform = get_transforms('valid')
    
#     # Create datasets
#     train_dataset = AIDetectionDataset(train_data, config.train_images_dir, train_transform, 'train')
#     val_dataset = AIDetectionDataset(val_data, config.train_images_dir, val_transform, 'val')
#     test_dataset = AIDetectionDataset(test_df, config.test_images_dir, val_transform, 'test')
    
#     # Create data loaders
#     train_loader = DataLoader(
#         train_dataset, batch_size=config.batch_size, 
#         shuffle=True, num_workers=config.num_workers
#     )
#     val_loader = DataLoader(
#         val_dataset, batch_size=config.batch_size, 
#         shuffle=False, num_workers=config.num_workers
#     )
#     test_loader = DataLoader(
#         test_dataset, batch_size=config.batch_size, 
#         shuffle=False, num_workers=config.num_workers
#     )
    
#     return train_loader, val_loader, test_loader, test_df


# def create_submission(model, test_loader, test_df):
#     """Create submission file"""
#     print("Creating submission...")
    
#     model.eval()
#     predictions = []
#     filenames = []
    
#     with torch.no_grad():
#         for images, names in tqdm(test_loader, desc='Predicting'):
#             images = images.to(device)
#             outputs = model(images).squeeze()
#             predictions.extend(outputs.cpu().numpy())
#             filenames.extend(names)
    
#     # Create submission
#     submission = pd.DataFrame({
#         'filename': filenames,
#         'label': predictions
#     })
    
#     # Ensure we have predictions for all test samples
#     if len(submission) != len(test_df):
#         print(f"Warning: Submission length {len(submission)} doesn't match test data {len(test_df)}")
#         # Create submission from test_df with default predictions
#         submission = pd.DataFrame({
#             'filename': test_df['file_name'].apply(os.path.basename),
#             'label': [0.5] * len(test_df)  # Default prediction
#         })
    
#     submission.to_csv('submission.csv', index=False)
#     print("Submission file created: submission.csv")
#     print(f"Submission shape: {submission.shape}")
#     print(submission.head())
    
#     return submission


# def load_model_safely(model_path='best_model.pth'):
#     """Safe model loading for PyTorch 2.6"""
#     model = AdvancedAIDetector()
    
#     try:
#         # Method 1: Direct load with weights_only=False
#         checkpoint = torch.load(model_path, map_location=device, weights_only=False)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         print("Model loaded successfully")
#         return model, checkpoint
#     except Exception as e:
#         print(f"Method 1 failed: {e}")
#         try:
#             # Method 2: Use safe globals context
#             with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
#                 checkpoint = torch.load(model_path, map_location=device)
#                 model.load_state_dict(checkpoint['model_state_dict'])
#                 print("Model loaded with safe globals")
#                 return model, checkpoint
#         except Exception as e2:
#             print(f"Method 2 failed: {e2}")
#             print("Model loading failed, returning untrained model")
#             return model, None


# def main():
#     """Complete training and submission pipeline"""
#     print("ğŸš€ Starting AI Detection Pipeline...")
    
#     try:
#         # Prepare data
#         train_loader, val_loader, test_loader, test_df = prepare_data()
        
#         # Test data loading
#         print("\n Testing data pipeline...")
#         test_batch = next(iter(train_loader))
#         images, labels = test_batch
#         print(f" Data loading test passed")
#         print(f"   Images shape: {images.shape}")
#         print(f"   Labels shape: {labels.shape}")
#         print(f"   Label range: {labels.min().item():.0f} to {labels.max().item():.0f}")
        
#         # Create and train model
#         print("\n Creating model...")
#         model = AdvancedAIDetector()
#         print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
#         # Train model
#         trainer = Trainer(model, train_loader, val_loader, device, config)
#         trainer.train()
        
#         # Create submission with best model
#         print("\n Generating submission...")
#         # Load best model
#         best_model, checkpoint = load_model_safely('best_model.pth')
#         if checkpoint:
#             print(f"Best model AUC: {checkpoint.get('best_auc', 'Unknown'):.4f}")
        
#         submission = create_submission(best_model, test_loader, test_df)
        
#         print(f"\nPipeline completed successfully!")
#         print(f" Best validation AUC: {trainer.best_auc:.4f}")
        
#     except Exception as e:
#         print(f" Pipeline failed: {e}")
#         import traceback
#         traceback.print_exc()


# def quick_submission():
#     """Quick submission if model already trained"""
#     print("Creating quick submission...")
    
#     # Load data
#     test_df = pd.read_csv(config.test_csv)
#     val_transform = get_transforms('valid')
#     test_dataset = AIDetectionDataset(test_df, config.test_images_dir, val_transform, 'test')
#     test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
#     # Load model
#     model, checkpoint = load_model_safely('best_model.pth')
#     model.to(device)
    
#     if checkpoint:
#         print(f"Loaded model with AUC: {checkpoint.get('best_auc', 'Unknown'):.4f}")
    
#     # Create submission
#     submission = create_submission(model, test_loader, test_df)
#     return submission


# if __name__ == '__main__':
#     # Check if model already exists
#     if os.path.exists('best_model.pth'):
#         print("Found existing model, creating submission...")
#         quick_submission()
#     else:
#         print("No existing model found, starting training...")
#         main()


# import numpy as np
# import pandas as pd 
# from sklearn.model_selection import train_test_split
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
# import cv2
# import os
# from tqdm import tqdm
# import timm
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')

# # Fix for PyTorch 2.6
# try:
#     import numpy.core.multiarray
#     torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
# except:
#     pass

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Using device: {device}')

# # ==================== CONFIGURATION ====================
# class Config:
#     # Data
#     train_csv = "/kaggle/input/detect-ai-vs-human-generated-images/train.csv"
#     test_csv = "/kaggle/input/detect-ai-vs-human-generated-images/test.csv"
#     train_images_dir = '/kaggle/input/ai-vs-human-generated-dataset/train_data/'
#     test_images_dir = '/kaggle/input/ai-vs-human-generated-dataset/test_data_v2/'
    
#     # Model
#     model_name = 'tf_efficientnet_b4'  # Changed to more stable model
#     img_size = 384
#     num_classes = 1
    
#     # Training
#     batch_size = 16
#     epochs = 5
#     lr = 1e-4  # Reduced learning rate for stability
#     weight_decay = 1e-5
#     num_workers = 0
#     early_stopping_patience = 5
    
# config = Config()

# # ==================== DATA PIPELINE ====================
# def get_transforms(phase='train'):
#     if phase == 'train':
#         return A.Compose([
#             A.Resize(config.img_size, config.img_size),
#             A.HorizontalFlip(p=0.5),
#             A.RandomRotate90(p=0.3),
#             A.Affine(
#                 translate_percent=(0.05, 0.05),
#                 scale=(0.9, 1.1),
#                 rotate=(-10, 10),
#                 shear=(-5, 5),
#                 interpolation=1,
#                 border_mode=0,
#                 p=0.3  # Reduced probability
#             ),
#             A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
#             A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ToTensorV2(),
#         ])
#     else:
#         return A.Compose([
#             A.Resize(config.img_size, config.img_size),
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ToTensorV2(),
#         ])

# class AIDetectionDataset(Dataset):
#     def __init__(self, df, images_dir, transform=None, phase='train'):
#         self.data = df.reset_index(drop=True)
#         self.images_dir = images_dir
#         self.transform = transform
#         self.phase = phase
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         try:
#             # Handle different column names for train and test
#             row = self.data.iloc[idx]
            
#             if 'file_name' in self.data.columns:
#                 img_name = row['file_name']
#             elif 'id' in self.data.columns:
#                 img_name = row['id']
#             else:
#                 # Try to find any column that might contain filenames
#                 for col in self.data.columns:
#                     if any(keyword in col.lower() for keyword in ['name', 'file', 'id', 'path']):
#                         img_name = row[col]
#                         break
#                 else:
#                     img_name = row.iloc[0]
            
#             # Extract filename from path
#             img_name = os.path.basename(str(img_name))
            
#             if self.phase == 'train':
#                 label = row['label']
#             else:
#                 label = 0
                
#             img_path = os.path.join(self.images_dir, img_name)
            
#             # Robust image loading with error handling
#             image = cv2.imread(img_path)
#             if image is None:
#                 raise ValueError(f"Could not load image: {img_path}")
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
#             if self.transform:
#                 image = self.transform(image=image)['image']
                
#             if self.phase == 'test':
#                 return image, img_name
#             else:
#                 return image, torch.tensor(float(label), dtype=torch.float32)
                
#         except Exception as e:
#             print(f"Error in dataset __getitem__: {e}")
#             # Create fallback image
#             image = np.ones((config.img_size, config.img_size, 3), dtype=np.uint8) * 128
#             if self.transform:
#                 image = self.transform(image=image)['image']
#             return image, torch.tensor(0.0, dtype=torch.float32)

# # ==================== MODEL ARCHITECTURE ====================
# class AdvancedAIDetector(nn.Module):
#     def __init__(self, model_name=config.model_name, num_classes=config.num_classes, pretrained=True):
#         super().__init__()
        
#         # Use a simpler model for stability
#         self.backbone = timm.create_model(
#             'tf_efficientnet_b2',  # More stable than b4
#             pretrained=pretrained,
#             num_classes=0,
#             drop_rate=0.2
#         )
        
#         feature_dim = 1408  # For EfficientNet-B2
        
#         # Simpler classifier
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Linear(feature_dim, 256),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm1d(256),
#             nn.Dropout(0.2),
#             nn.Linear(256, num_classes),
#         )
        
#     def forward(self, x):
#         features = self.backbone(x)
#         output = self.classifier(features)
#         return output

# # ==================== TRAINING ENGINE ====================
# class Trainer:
#     def __init__(self, model, train_loader, val_loader, device, config):
#         self.model = model.to(device)
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.device = device
#         self.config = config
        
#         # Loss function
#         self.criterion = nn.BCEWithLogitsLoss()
        
#         # Optimizer
#         self.optimizer = optim.AdamW(
#             model.parameters(), 
#             lr=config.lr, 
#             weight_decay=config.weight_decay
#         )
        
#         # Scheduler
#         self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             self.optimizer, mode='max', patience=2, factor=0.5, verbose=True
#         )
        
#         self.best_auc = 0
#         self.train_history = []
#         self.patience_counter = 0
        
#     def safe_auc_score(self, labels, preds):
#         """Safe AUC calculation with multiple fallbacks"""
#         try:
#             # Check if we have at least 2 classes
#             unique_labels = np.unique(labels)
#             if len(unique_labels) < 2:
#                 print(f"âš ï¸� Only one class present: {unique_labels}")
#                 # Return accuracy as fallback
#                 predictions = [1 if p > 0.5 else 0 for p in preds]
#                 return accuracy_score(labels, predictions)
            
#             return roc_auc_score(labels, preds)
#         except Exception as e:
#             print(f"âš ï¸� AUC calculation failed: {e}")
#             # Final fallback
#             return 0.5
    
#     def train_epoch(self):
#         self.model.train()
#         running_loss = 0.0
#         all_preds = []
#         all_labels = []
        
#         pbar = tqdm(self.train_loader, desc='Training')
#         for batch_idx, (images, labels) in enumerate(pbar):
#             try:
#                 images, labels = images.to(self.device), labels.to(self.device)
                
#                 self.optimizer.zero_grad()
                
#                 # Forward pass
#                 outputs = self.model(images).squeeze()
                
#                 # Ensure outputs and labels have same shape
#                 if outputs.dim() == 0:
#                     outputs = outputs.unsqueeze(0)
                
#                 # Loss computation
#                 loss = self.criterion(outputs, labels)
                
#                 loss.backward()
                
#                 # Gradient clipping
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
#                 self.optimizer.step()
                
#                 running_loss += loss.item()
                
#                 # Collect predictions
#                 with torch.no_grad():
#                     preds = torch.sigmoid(outputs)
#                     all_preds.extend(preds.cpu().numpy())
#                     all_labels.extend(labels.cpu().numpy())
                
#                 pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                
#             except Exception as e:
#                 print(f"Error in training batch {batch_idx}: {e}")
#                 continue
        
#         epoch_loss = running_loss / len(self.train_loader)
#         epoch_auc = self.safe_auc_score(all_labels, all_preds)
        
#         return epoch_loss, epoch_auc
    
#     def validate(self):
#         self.model.eval()
#         all_preds = []
#         all_labels = []
        
#         with torch.no_grad():
#             for images, labels in tqdm(self.val_loader, desc='Validation'):
#                 try:
#                     images, labels = images.to(self.device), labels.to(self.device)
                    
#                     outputs = self.model(images).squeeze()
#                     # Ensure proper shape
#                     if outputs.dim() == 0:
#                         outputs = outputs.unsqueeze(0)
                    
#                     preds = torch.sigmoid(outputs)
#                     all_preds.extend(preds.cpu().numpy())
#                     all_labels.extend(labels.cpu().numpy())
                    
#                 except Exception as e:
#                     print(f"Error in validation batch: {e}")
#                     continue
        
#         auc = self.safe_auc_score(all_labels, all_preds)
        
#         # Calculate accuracy and F1
#         try:
#             predictions = [1 if p > 0.5 else 0 for p in all_preds]
#             accuracy = accuracy_score(all_labels, predictions)
#             f1 = f1_score(all_labels, predictions, zero_division=0)
#         except:
#             accuracy = 0.5
#             f1 = 0.0
        
#         return auc, accuracy, f1
    
#     def train(self):
#         print("Starting training...")
        
#         for epoch in range(self.config.epochs):
#             print(f'\nEpoch {epoch+1}/{self.config.epochs}')
            
#             try:
#                 # Train
#                 train_loss, train_auc = self.train_epoch()
                
#                 # Validate
#                 val_auc, val_acc, val_f1 = self.validate()
                
#                 # Update scheduler
#                 self.scheduler.step(val_auc)
                
#                 print(f'Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}')
#                 print(f'Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
                
#                 # Save best model
#                 if val_auc > self.best_auc:
#                     self.best_auc = val_auc
#                     self._save_model(epoch, val_auc)
#                     self.patience_counter = 0
#                     print(f'âœ… New best model saved with AUC: {val_auc:.4f}')
#                 else:
#                     self.patience_counter += 1
#                     print(f'No improvement for {self.patience_counter} epochs')
                
#                 # Early stopping
#                 if self.patience_counter >= self.config.early_stopping_patience:
#                     print(f"ğŸ›‘ Early stopping at epoch {epoch+1}")
#                     break
                
#                 self.train_history.append({
#                     'epoch': epoch,
#                     'train_loss': train_loss,
#                     'train_auc': train_auc,
#                     'val_auc': val_auc,
#                     'val_acc': val_acc,
#                     'val_f1': val_f1
#                 })
                
#             except Exception as e:
#                 print(f"Error in epoch {epoch+1}: {e}")
#                 continue
    
#     def _save_model(self, epoch, auc):
#         """Safe model saving"""
#         checkpoint = {
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'best_auc': self.best_auc,
#             'epoch': epoch,
#             'config': vars(config)
#         }
#         torch.save(checkpoint, 'best_model.pth')
#         print(f"Model saved: best_model.pth")

# # ==================== DATA PREPARATION ====================
# def prepare_data():
#     """Prepare and validate data loading"""
#     print("Loading data...")
    
#     try:
#         # Load CSVs
#         train_df = pd.read_csv(config.train_csv, index_col=0)
#         test_df = pd.read_csv(config.test_csv)
        
#         print(f"Train data shape: {train_df.shape}")
#         print(f"Test data shape: {test_df.shape}")
#         print(f"Train columns: {train_df.columns.tolist()}")
#         print(f"Test columns: {test_df.columns.tolist()}")
        
#         # Check class distribution
#         print(f"\nğŸ“Š Class distribution in training data:")
#         class_dist = train_df['label'].value_counts().sort_index()
#         print(class_dist)
        
#         # Ensure we have both classes
#         if len(class_dist) < 2:
#             print("â�Œ WARNING: Training data has only one class!")
#             # Try to manually balance by sampling
#             class_0 = train_df[train_df['label'] == 0]
#             class_1 = train_df[train_df['label'] == 1]
#             min_samples = min(len(class_0), len(class_1))
            
#             if min_samples > 0:
#                 class_0 = class_0.sample(min_samples, random_state=42)
#                 class_1 = class_1.sample(min_samples, random_state=42)
#                 train_df = pd.concat([class_0, class_1]).sample(frac=1, random_state=42)
#                 print(f"Balanced training data: {len(train_df)} samples")
        
#         # Debug image loading
#         def debug_images(df, images_dir, num_samples=3):
#             print(f"\nğŸ”� Debugging image loading from {images_dir}:")
#             successful = 0
            
#             for i in range(min(num_samples, len(df))):
#                 try:
#                     if 'file_name' in df.columns:
#                         img_name = os.path.basename(df.iloc[i]['file_name'])
#                     elif 'id' in df.columns:
#                         img_name = os.path.basename(df.iloc[i]['id'])
#                     else:
#                         img_name = os.path.basename(df.iloc[i].iloc[0])
                    
#                     img_path = os.path.join(images_dir, img_name)
#                     image = cv2.imread(img_path)
#                     if image is not None:
#                         print(f"  âœ… {img_name} - Shape: {image.shape}")
#                         successful += 1
#                     else:
#                         print(f"  â�Œ {img_name} - Failed to load")
#                 except Exception as e:
#                     print(f"  â�Œ Error loading sample {i}: {e}")
#             return successful > 0
        
#         # Test image loading
#         print("\nTesting train images:")
#         train_success = debug_images(train_df, config.train_images_dir)
        
#         print("\nTesting test images:")
#         test_success = debug_images(test_df, config.test_images_dir)
        
#         if not train_success or not test_success:
#             raise Exception("Image loading failed. Check paths and file structure.")
        
#         # Split data with stratification
#         train_data, val_data = train_test_split(
#             train_df, 
#             test_size=0.2, 
#             random_state=42, 
#             stratify=train_df['label']
#         )
        
#         print(f"\nğŸ“Š Data split:")
#         print(f"  Training samples: {len(train_data)}")
#         print(f"  Validation samples: {len(val_data)}")
#         print(f"  Test samples: {len(test_df)}")
        
#         # Check validation set class distribution
#         print(f"\nğŸ“Š Validation set class distribution:")
#         val_dist = val_data['label'].value_counts().sort_index()
#         print(val_dist)
        
#         if len(val_dist) < 2:
#             print("â�Œ WARNING: Validation set has only one class! Using different random state.")
#             train_data, val_data = train_test_split(
#                 train_df, 
#                 test_size=0.2, 
#                 random_state=123,  # Different random state
#                 stratify=train_df['label']
#             )
#             print(f"New validation distribution: {val_data['label'].value_counts().sort_index()}")
        
#         # Get transforms
#         train_transform = get_transforms('train')
#         val_transform = get_transforms('valid')
        
#         # Create datasets
#         train_dataset = AIDetectionDataset(train_data, config.train_images_dir, train_transform, 'train')
#         val_dataset = AIDetectionDataset(val_data, config.train_images_dir, val_transform, 'val')
#         test_dataset = AIDetectionDataset(test_df, config.test_images_dir, val_transform, 'test')
        
#         # Create data loaders
#         train_loader = DataLoader(
#             train_dataset, batch_size=config.batch_size, 
#             shuffle=True, num_workers=config.num_workers,
#             pin_memory=True
#         )
#         val_loader = DataLoader(
#             val_dataset, batch_size=config.batch_size, 
#             shuffle=False, num_workers=config.num_workers,
#             pin_memory=True
#         )
#         test_loader = DataLoader(
#             test_dataset, batch_size=config.batch_size, 
#             shuffle=False, num_workers=config.num_workers,
#             pin_memory=True
#         )
        
#         return train_loader, val_loader, test_loader, test_df
        
#     except Exception as e:
#         print(f"Error in data preparation: {e}")
#         raise

# # ==================== INFERENCE ====================
# def create_submission(model, test_loader, test_df):
#     """Create submission file"""
#     print("Creating submission...")
    
#     model.eval()
#     predictions = []
#     filenames = []
    
#     with torch.no_grad():
#         for images, names in tqdm(test_loader, desc='Predicting'):
#             try:
#                 images = images.to(device)
#                 outputs = model(images).squeeze()
#                 # Ensure proper shape
#                 if outputs.dim() == 0:
#                     outputs = outputs.unsqueeze(0)
#                 preds = torch.sigmoid(outputs)
#                 predictions.extend(preds.cpu().numpy())
#                 filenames.extend(names)
#             except Exception as e:
#                 print(f"Error in prediction batch: {e}")
#                 # Add default predictions
#                 predictions.extend([0.5] * len(names))
#                 filenames.extend(names)
    
#     # Create submission
#     submission = pd.DataFrame({
#         'filename': filenames,
#         'label': predictions
#     })
    
#     # Ensure we have predictions for all test samples
#     if len(submission) != len(test_df):
#         print(f"Warning: Submission length {len(submission)} doesn't match test data {len(test_df)}")
#         # Create submission from test_df with default predictions
#         if 'id' in test_df.columns:
#             submission = pd.DataFrame({
#                 'filename': test_df['id'].apply(lambda x: os.path.basename(str(x))),
#                 'label': [0.5] * len(test_df)
#             })
#         else:
#             submission = pd.DataFrame({
#                 'filename': [f"image_{i}.jpg" for i in range(len(test_df))],
#                 'label': [0.5] * len(test_df)
#             })
    
#     submission.to_csv('submission.csv', index=False)
#     print("âœ… Submission file created: submission.csv")
#     print(f"Submission shape: {submission.shape}")
#     print("First few predictions:")
#     print(submission.head())
    
#     return submission

# # ==================== MODEL LOADING ====================
# def load_model_safely(model_path='best_model.pth'):
#     """Safe model loading"""
#     model = AdvancedAIDetector()
    
#     if not os.path.exists(model_path):
#         print(f"â�Œ Model file not found: {model_path}")
#         return model, None
    
#     try:
#         # Try multiple loading methods
#         methods = [
#             lambda: torch.load(model_path, map_location=device, weights_only=False),
#             lambda: torch.load(model_path, map_location=device),
#         ]
        
#         checkpoint = None
#         for i, method in enumerate(methods):
#             try:
#                 checkpoint = method()
#                 print(f"âœ… Model loaded with method {i+1}")
#                 break
#             except Exception as e:
#                 print(f"Method {i+1} failed: {e}")
#                 continue
        
#         if checkpoint and 'model_state_dict' in checkpoint:
#             model.load_state_dict(checkpoint['model_state_dict'])
#             return model, checkpoint
#         else:
#             print("â�Œ No valid checkpoint found")
#             return model, None
            
#     except Exception as e:
#         print(f"â�Œ Model loading failed: {e}")
#         return model, None

# # ==================== MAIN EXECUTION ====================
# def main():
#     """Complete training and submission pipeline"""
#     print("ğŸš€ Starting AI Detection Pipeline...")
    
#     try:
#         # Step 1: Prepare data
#         train_loader, val_loader, test_loader, test_df = prepare_data()
        
#         # Step 2: Test data loading
#         print("\nğŸ§ª Testing data pipeline...")
#         try:
#             test_batch = next(iter(train_loader))
#             images, labels = test_batch
#             print(f"âœ… Data loading test passed")
#             print(f"   Images shape: {images.shape}")
#             print(f"   Labels shape: {labels.shape}")
#             print(f"   Label range: {labels.min().item():.0f} to {labels.max().item():.0f}")
#         except Exception as e:
#             print(f"â�Œ Data loading test failed: {e}")
#             return
        
#         # Step 3: Create and train model
#         print("\nğŸ�¯ Creating model...")
#         model = AdvancedAIDetector()
#         print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
#         # Step 4: Train model
#         trainer = Trainer(model, train_loader, val_loader, device, config)
#         trainer.train()
        
#         # Step 5: Create submission with best model
#         print("\nğŸ“¤ Generating submission...")
#         best_model, checkpoint = load_model_safely('best_model.pth')
#         best_model.to(device)
        
#         if checkpoint:
#             print(f"Best model AUC: {checkpoint.get('best_auc', 'Unknown'):.4f}")
        
#         submission = create_submission(best_model, test_loader, test_df)
        
#         print(f"\nğŸ�‰ Pipeline completed successfully!")
#         print(f"ğŸ“Š Best validation AUC: {trainer.best_auc:.4f}")
        
#     except Exception as e:
#         print(f"â�Œ Pipeline failed: {e}")
#         import traceback
#         traceback.print_exc()

# # ==================== QUICK SUBMISSION ====================
# def quick_submission():
#     """Quick submission if model already trained"""
#     print("ğŸš€ Creating quick submission...")
    
#     try:
#         # Load data
#         test_df = pd.read_csv(config.test_csv)
#         val_transform = get_transforms('valid')
#         test_dataset = AIDetectionDataset(test_df, config.test_images_dir, val_transform, 'test')
#         test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        
#         # Load model
#         model, checkpoint = load_model_safely('best_model.pth')
#         model.to(device)
        
#         if checkpoint:
#             print(f"Loaded model with AUC: {checkpoint.get('best_auc', 'Unknown'):.4f}")
#         else:
#             print("âš ï¸� Using untrained model for submission")
        
#         # Create submission
#         submission = create_submission(model, test_loader, test_df)
#         return submission
#     except Exception as e:
#         print(f"â�Œ Quick submission failed: {e}")
#         return None

# # ==================== EXECUTE ====================
# if __name__ == '__main__':
#     print("ğŸ¤– AI vs Human Image Detection")
#     print("=" * 50)
    
#     # Check if model already exists
#     if os.path.exists('best_model.pth'):
#         print("ğŸ“� Found existing model")
#         response = input("Do you want to create submission only? (y/n): ").lower()
#         if response == 'y':
#             quick_submission()
#         else:
#             print("ğŸ†• Starting full training pipeline...")
#             main()
#     else:
#         print("ğŸ†• No existing model found, starting training...")
#         main()

