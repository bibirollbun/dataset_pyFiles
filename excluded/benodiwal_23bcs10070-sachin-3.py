import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    # Paths
    TEST_DIR = '/kaggle/input/petfinder-pawpularity-score/test'
    TEST_CSV = '/kaggle/input/petfinder-pawpularity-score/test.csv'
    MODEL_DIR = '/kaggle/input/petfinder-trained-models'
    OUTPUT_DIR = '/kaggle/working'
    
    # Model
    MODEL_NAME = 'tf_efficientnet_b4_ns'
    IMG_SIZE = 384
    
    # Inference settings
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    USE_TTA = True  # Test Time Augmentation for better scores
    FOLDS_TO_USE = [0]  # Use multiple folds for ensemble: [0, 1, 2, 3, 4]
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Metadata features
    META_FEATURES = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 
                     'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 
                     'Info', 'Blur']

# Dataset for inference
class PawpularityTestDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, row['Id'] + '.jpg')
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # Get metadata - convert to float explicitly
        meta_values = row[Config.META_FEATURES].values.astype(np.float32)
        meta = torch.from_numpy(meta_values)
        
        return img, meta, row['Id']

# Transforms for TTA
def get_test_transforms(tta_type='original'):
    """
    Get transforms for different TTA variants
    """
    base_transforms = [
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    ]
    
    if tta_type == 'hflip':
        base_transforms.append(transforms.RandomHorizontalFlip(p=1.0))
    elif tta_type == 'vflip':
        base_transforms.append(transforms.RandomVerticalFlip(p=1.0))
    
    base_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(base_transforms)

# Model architecture (must match training)
class PawpularityModel(nn.Module):
    def __init__(self, model_name, num_meta_features=12, pretrained=False):
        super().__init__()
        
        # Load backbone
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        # Get number of features
        if hasattr(self.backbone, 'classifier'):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            in_features = self.backbone.num_features
        
        # Regression head
        self.head = nn.Sequential(
            nn.Linear(in_features + num_meta_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
    def forward(self, img, meta):
        img_features = self.backbone(img)
        combined = torch.cat([img_features, meta], dim=1)
        output = self.head(combined)
        return output

def load_model(model_path, device):
    """Load trained model from checkpoint"""
    model = PawpularityModel(Config.MODEL_NAME, num_meta_features=12, pretrained=False)
    
    # Load checkpoint - weights_only=False since this is our own trusted model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from: {model_path}")
    print(f"Model validation RMSE: {checkpoint['val_rmse']:.4f}")
    
    return model

def predict_with_tta(model, loader, device, use_tta=True):
    """
    Generate predictions with Test Time Augmentation
    TTA improves scores by 0.05-0.10 RMSE
    """
    predictions = []
    ids = []
    
    with torch.no_grad():
        for img, meta, batch_ids in tqdm(loader, desc='Predicting'):
            img, meta = img.to(device), meta.to(device)
            
            # Original prediction
            output = model(img, meta).squeeze()
            preds = torch.sigmoid(output) * 100.0
            
            if use_tta:
                # Horizontal flip TTA
                img_hflip = torch.flip(img, dims=[-1])
                output_hflip = model(img_hflip, meta).squeeze()
                preds_hflip = torch.sigmoid(output_hflip) * 100.0
                
                # Average predictions
                preds = (preds + preds_hflip) / 2.0
            
            predictions.extend(preds.cpu().numpy())
            ids.extend(batch_ids)
    
    return np.array(predictions), ids

def ensemble_predictions(fold_predictions):
    """
    Ensemble predictions from multiple folds
    Simple averaging works well for this competition
    """
    return np.mean(fold_predictions, axis=0)

def main():
    print("=" * 80)
    print("PetFinder Pawpularity - Inference")
    print("=" * 80)
    print(f"Device: {Config.DEVICE}")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"TTA: {Config.USE_TTA}")
    print(f"Folds: {Config.FOLDS_TO_USE}")
    print("=" * 80)
    
    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(Config.TEST_CSV)
    print(f"Test samples: {len(test_df)}")
    
    # Create dataset and dataloader
    test_dataset = PawpularityTestDataset(
        test_df, Config.TEST_DIR, 
        transform=get_test_transforms('original')
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, 
        shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    
    # Generate predictions for each fold
    all_fold_predictions = []
    
    for fold in Config.FOLDS_TO_USE:
        print(f"\n{'=' * 80}")
        print(f"Processing Fold {fold}")
        print('=' * 80)
        
        # Load model
        model_path = f'{Config.MODEL_DIR}/best_model_fold{fold}.pth'
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}")
            print("Please update Config.MODEL_DIR to point to your trained models")
            continue
        
        model = load_model(model_path, Config.DEVICE)
        
        # Generate predictions
        fold_preds, ids = predict_with_tta(model, test_loader, Config.DEVICE, Config.USE_TTA)
        all_fold_predictions.append(fold_preds)
        
        print(f"Fold {fold} - Predictions shape: {fold_preds.shape}")
        print(f"Fold {fold} - Mean prediction: {fold_preds.mean():.2f}")
        print(f"Fold {fold} - Std prediction: {fold_preds.std():.2f}")
        
        # Clean up memory
        del model
        torch.cuda.empty_cache()
    
    # Ensemble predictions from all folds
    print(f"\n{'=' * 80}")
    print("Ensembling Predictions")
    print('=' * 80)
    
    if len(all_fold_predictions) > 1:
        final_predictions = ensemble_predictions(all_fold_predictions)
        print(f"Ensembled {len(all_fold_predictions)} folds")
    else:
        final_predictions = all_fold_predictions[0]
        print("Using single fold predictions")
    
    # Optional: Clip predictions (some solutions found this helpful)
    # final_predictions = np.clip(final_predictions, 0, 85)
    
    print(f"Final predictions - Mean: {final_predictions.mean():.2f}, Std: {final_predictions.std():.2f}")
    
    # Create submission file
    submission_df = pd.DataFrame({
        'Id': ids,
        'Pawpularity': final_predictions
    })
    
    # Save submission
    submission_path = f'{Config.OUTPUT_DIR}/submission.csv'
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\n{'=' * 80}")
    print(f"Submission saved to: {submission_path}")
    print('=' * 80)
    
    # Show sample predictions
    print("\nSample predictions:")
    print(submission_df.head(10))
    
    # Verify submission format
    print("\nSubmission validation:")
    print(f"Shape: {submission_df.shape}")
    print(f"Columns: {submission_df.columns.tolist()}")
    print(f"Missing values: {submission_df.isnull().sum().sum()}")
    print(f"Prediction range: [{final_predictions.min():.2f}, {final_predictions.max():.2f}]")
    
    print("\n" + "=" * 80)
    print("Inference completed successfully!")
    print("Ready to submit to Kaggle!")
    print("=" * 80)

if __name__ == "__main__":
    main()

