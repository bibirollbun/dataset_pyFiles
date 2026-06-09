# -*- coding: utf-8 -*-
"""
Kaggle Chest X-ray Submission Notebook
--------------------------------------
Chá»‰ load model Ä‘Ã£ Ä‘Æ°á»£c huáº¥n luyá»‡n vÃ  táº¡o file submission.
"""
import numpy as np
import pandas as pd
import cv2
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Cáº¥u hÃ¬nh cÆ¡ báº£n
# =============================================================================
# !!! QUAN TRá»ŒNG: Cáº¬P NHáº¬T Ä�Æ¯á»œNG DáºªN NÃ€Y CHO Ä�ÃšNG !!!
# Thay 'grand-x-ray-slam-baseline' báº±ng tÃªn dataset chá»©a file model cá»§a báº¡n
MODEL_PATH = '/kaggle/input/grand-x-ray-slam-baseline/best_model_optimized.pth'
# =============================================================================

IMG_SIZE = 320
BATCH_SIZE = 32  # CÃ³ thá»ƒ tÄƒng batch size náº¿u GPU cho phÃ©p Ä‘á»ƒ cháº¡y nhanh hÆ¡n
NUM_CLASSES = 14
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"Model path: {MODEL_PATH}")

# =============================================================================
# 1. Ä�á»ŠNH NGHÄ¨A Láº I KIáº¾N TRÃšC MODEL
# (Pháº£i giá»‘ng há»‡t vá»›i kiáº¿n trÃºc lÃºc huáº¥n luyá»‡n Ä‘á»ƒ load trá»�ng sá»‘)
# =============================================================================
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=14):
        super(EfficientNetModel, self).__init__()
        self.backbone = models.efficientnet_b3(weights=None) # KhÃ´ng cáº§n weights vÃ¬ sáº½ load tá»« file
        backbone_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.attention = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim // 8),
            nn.ReLU(),
            nn.Linear(backbone_dim // 8, backbone_dim),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(backbone_dim),
            nn.Dropout(0.5),
            nn.Linear(backbone_dim, backbone_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(backbone_dim // 2),
            nn.Dropout(0.3),
            nn.Linear(backbone_dim // 2, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        output = self.classifier(attended_features)
        return output

# =============================================================================
# 2. CÃ�C PHÃ‰P BIáº¾N Ä�á»”I Dá»® LIá»†U (AUGMENTATION) CHO TEST
# =============================================================================
def get_tta_transforms(img_size=320):
    """Láº¥y cÃ¡c phÃ©p biáº¿n Ä‘á»•i cho Test Time Augmentation (TTA)."""
    
    # Transform cÆ¡ báº£n
    base_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform láº­t ngang
    flipped_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return [base_transform, flipped_transform]

# =============================================================================
# 3. DATASET CHO Dá»® LIá»†U TEST
# =============================================================================
class ChestXRayDataset(Dataset):
    def __init__(self, df, img_size, tta_transforms):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.tta_transforms = tta_transforms
        self.image_dir = '/kaggle/input/grand-xray-slam-division-a/test1/'

    def __len__(self):
        return len(self.df)
    
    def load_and_preprocess_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)
        
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['Image_name'])
        img = self.load_and_preprocess_image(img_path)

        images = []
        for transform in self.tta_transforms:
            images.append(transform(img.copy()))
        return torch.stack(images)

# =============================================================================
# 4. HÃ€M Dá»° Ä�OÃ�N Vá»šI TTA
# =============================================================================
def predict_with_tta(model, test_loader, device):
    model.eval()
    model.to(device)
    predictions = []
    
    with torch.no_grad():
        for tta_images in tqdm(test_loader, desc='Generating predictions'):
            tta_images = tta_images.to(device)
            batch_size, tta_count = tta_images.shape[:2]
            
            tta_images = tta_images.view(-1, *tta_images.shape[2:])
            
            outputs = model(tta_images)
            outputs = torch.sigmoid(outputs)
            
            outputs = outputs.view(batch_size, tta_count, -1)
            tta_averaged = outputs.mean(dim=1)
            
            predictions.append(tta_averaged.cpu().numpy())
    
    return np.vstack(predictions)

# =============================================================================
# 5. HÃ€M MAIN: LOAD MODEL VÃ€ Táº O SUBMISSION
# =============================================================================
def main():
    # 1. Khá»Ÿi táº¡o model vÃ  load trá»�ng sá»‘
    print("Loading model...")
    model = EfficientNetModel(num_classes=NUM_CLASSES)
    
    # ThÃªm `weights_only=False` Ä‘á»ƒ tÆ°Æ¡ng thÃ­ch vá»›i cÃ¡ch lÆ°u checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("âœ“ Model loaded successfully.")

    # 2. Chuáº©n bá»‹ dá»¯ liá»‡u test
    print("Preparing test data...")
    try:
        sample_submission = pd.read_csv('/kaggle/input/grand-xray-slam-division-a/sample_submission_1.csv')
    except FileNotFoundError:
        print("Error: sample_submission_1.csv not found!")
        return
    
    tta_transforms = get_tta_transforms(IMG_SIZE)
    test_dataset = ChestXRayDataset(sample_submission, IMG_SIZE, tta_transforms)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    print(f"Test data ready: {len(test_dataset)} samples.")
    
    # 3. Cháº¡y dá»± Ä‘oÃ¡n
    predictions = predict_with_tta(model, test_loader, DEVICE)
    
    # 4. Táº¡o file submission
    print("Creating submission file...")
    label_columns = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
        'Lung Opacity', 'No Finding', 'Pleural Effusion',
        'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]
    
    # Ä�áº£m báº£o shape chÃ­nh xÃ¡c
    predictions = predictions[:len(sample_submission)]
    
    submission_df = sample_submission.copy()
    submission_df[label_columns] = predictions
    submission_df.to_csv('submission.csv', index=False)
    
    print("\nğŸ�‰ Submission created successfully: submission.csv")
    print(f"Shape: {submission_df.shape}")
    print("Submission head:")
    print(submission_df.head())

if __name__ == "__main__":
    main()

