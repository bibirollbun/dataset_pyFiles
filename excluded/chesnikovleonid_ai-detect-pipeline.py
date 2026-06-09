!pip install -q -U albumentations


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.models as models
from albumentations import Compose, Resize, RandomResizedCrop, HorizontalFlip, Rotate, ColorJitter, RandomBrightnessContrast, GaussianBlur, GaussNoise, CoarseDropout, ImageCompression, CenterCrop, Normalize
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

class Config:
    # Paths
    train_csv = "/kaggle/input/ai-vs-human-generated-dataset/train.csv"
    test_csv = "/kaggle/input/ai-vs-human-paired-test/test_paired.csv"
    img_dir = "/kaggle/input/ai-vs-human-generated-dataset"
    
    # Model parameters
    backbone_name = "swin_b"
    img_size = 224
    dropout = 0.3
    pretrained = True

    # Training parameters
    batch_size = 64
    epochs = 15
    backbone_lr = 1e-5
    head_lr = 1e-4
    weight_decay = 0.05
    tta_steps = 5
    threshold = 0.5
    mixup_alpha = 0.3
    label_smoothing = 0.1
    dynamic_unfreeze = True
    use_amp = True
    # Augmentations and validation
    use_augmentations = True
    use_validation = True
    val_split = 0.1
    
    # System
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42


class PairDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_train=True):
        self.df = df.sort_values('pair_id').reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train
        self.pair_ids = self.df['pair_id'].unique()
        
        # Check pair integrity: each pair_id should have exactly 2 images.
        counts = self.df['pair_id'].value_counts()
        if not (counts == 2).all():
            raise ValueError("Invalid pairs detected in the dataset.")

    def __len__(self):
        return len(self.pair_ids)
    
    def __getitem__(self, idx):
        # Get the pair for a given index
        pair_id = self.pair_ids[idx]
        pair = self.df[self.df['pair_id'] == pair_id]
        
        images = []
        labels = []
        for _, row in pair.iterrows():
            img_path = os.path.join(self.img_dir, row['id'])
            image = np.array(Image.open(img_path).convert('RGB'))
            images.append(image)
            labels.append(row['label'])
        
        # Apply pair-consistent augmentations using Albumentations.
        if self.transform:
            transformed = self.transform(image=images[0], image1=images[1])
            img1 = transformed['image']
            img2 = transformed['image1']
        else:
            img1, img2 = images
            
        # Random swap of the pair to avoid positional bias
        if self.is_train and torch.rand(1).item() > 0.5:
            img1, img2 = img2, img1
            labels = labels[::-1]  # swap corresponding labels
        
        # Return a tensor stack of both images and corresponding labels.
        return (torch.stack([img1, img2]), torch.tensor(labels, dtype=torch.float32))
    
    def get_raw_pair(self, idx):
        """
        Returns the original (raw) images for the given pair index without any transformations.
        """
        pair_id = self.pair_ids[idx]
        pair = self.df[self.df['pair_id'] == pair_id]
        raw_images = []
        for _, row in pair.iterrows():
            img_path = os.path.join(self.img_dir, row['id'])
            # Load the image in RGB and convert to a float numpy array scaled between 0 and 1.
            image = np.array(Image.open(img_path).convert('RGB')).astype(np.float32) / 255.0
            raw_images.append(image)
        return raw_images


class PairComparator(nn.Module):
    def __init__(self):
        super(PairComparator, self).__init__()
        # Initialize pretrained ConvNeXt backbone
        self.backbone = getattr(models, Config.backbone_name)(weights="DEFAULT")
        in_features = self.backbone.head.in_features

        # Initially freeze all but the last 10% of blocks for a stronger regularization.
        num_blocks = len(self.backbone.features)
        for idx, block in enumerate(self.backbone.features):
            # Freeze if index is less than 90% of total blocks.
            block.requires_grad_(idx >= int(num_blocks * 0.9))
        
        # Replace the classifier with AdaptiveAvgPool and Flatten for feature extraction.
        self.backbone.head = nn.Identity()
        
        # Comparison head: Concatenate features, absolute difference, and element-wise product.
        self.comparison_head = nn.Sequential(
            nn.Linear(in_features * 4, 512),
            nn.GELU(),
            nn.Dropout(Config.dropout),
            nn.LayerNorm(512),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        # x shape: (B, 2, C, H, W)
        batch_size = x.size(0)
        features = self.backbone(x.view(-1, *x.shape[2:]))  # (B*2, in_features)
        features = features.view(batch_size, 2, -1)           # (B, 2, in_features)
        f1 = features[:, 0]
        f2 = features[:, 1]
        diff = torch.abs(f1 - f2)
        prod = f1 * f2
        # Concatenate features, diff, and product to form final representation.
        pair_features = torch.cat([f1, f2, diff, prod], dim=1)
        out = self.comparison_head(pair_features).squeeze()
        return out


def gradual_unfreeze(model, current_epoch, total_epochs, start_fraction=0.9, target_fraction=0.75):
    """
    Gradually unfreezes backbone blocks from start_fraction to target_fraction over epochs.
    For example, if start_fraction=0.9 (only last 10% trainable) and target_fraction=0.75,
    then by the end of training, 25% of blocks are trainable.
    """
    num_blocks = len(model.backbone.features)
    # Calculate current unfreeze fraction (linearly interpolate)
    frac = start_fraction - (start_fraction - target_fraction) * (current_epoch / total_epochs)
    threshold = int(num_blocks * frac)
    for idx, block in enumerate(model.backbone.features):
        block.requires_grad_(idx >= threshold)
    print(f"[Epoch {current_epoch}] Gradual unfreeze: Unfroze {num_blocks - threshold} of {num_blocks} backbone blocks.")


def get_transforms():

    base = [
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    
    if Config.use_augmentations:
        train_tfm = Compose([
            Resize(256, 256),
            RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            HorizontalFlip(p=0.5),
            Rotate(limit=5, p=0.2),  # Small random rotation.
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.2),
            RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
            GaussianBlur(blur_limit=(3, 3), p=0.1),
            GaussNoise(p=0.1),  # Using default noise parameters.
            CoarseDropout(p=0.1),
            ImageCompression(quality_range=(75, 100), p=0.1),
            *base
        ], additional_targets={'image1': 'image'})
    else:
        train_tfm = Compose([
            Resize(256, 256),
            CenterCrop(height=224, width=224),
            *base
        ], additional_targets={'image1': 'image'})
    
    val_tfm = Compose([
        Resize(256, 256),
        CenterCrop(height=224, width=224),
        *base
    ], additional_targets={'image1': 'image'})
    
    return train_tfm, val_tfm


def visualize_augmentations_with_raw(dataset, num_samples=5):
    """
    For each sample, display the original (raw) pair and the augmented pair.
    The output grid will have 2 rows per sample:
       - Top row: Raw images (original from disk, scaled to [0,1]).
       - Bottom row: Augmented images (result of transforms).
    """
    # Create a figure with num_samples*2 rows and 2 columns.
    fig, axes = plt.subplots(num_samples * 2, 2, figsize=(10, num_samples * 4))
    for i in range(num_samples):
        # Get raw images from the dataset (using the custom method)
        raw_imgs = dataset.get_raw_pair(i)  # list of two numpy arrays in [0,1]
        
        # Get augmented images from __getitem__
        aug_imgs_tensor, _ = dataset[i]  # tensor of shape (2, 3, 224, 224)
        # Convert augmented images to numpy arrays (permute and clamp to [0,1])
        aug_img1 = aug_imgs_tensor[0].permute(1, 2, 0).cpu().numpy()
        aug_img2 = aug_imgs_tensor[1].permute(1, 2, 0).cpu().numpy()
        aug_img1 = np.clip(aug_img1, 0, 1)
        aug_img2 = np.clip(aug_img2, 0, 1)
        
        # Plot raw images (top row for sample i)
        axes[2*i, 0].imshow(raw_imgs[0])
        axes[2*i, 0].set_title(f"Raw Image 1 (Sample {i})")
        axes[2*i, 0].axis("off")
        
        axes[2*i, 1].imshow(raw_imgs[1])
        axes[2*i, 1].set_title(f"Raw Image 2 (Sample {i})")
        axes[2*i, 1].axis("off")
        
        # Plot augmented images (bottom row for sample i)
        axes[2*i+1, 0].imshow(aug_img1)
        axes[2*i+1, 0].set_title(f"Augmented Image 1 (Sample {i})")
        axes[2*i+1, 0].axis("off")
        
        axes[2*i+1, 1].imshow(aug_img2)
        axes[2*i+1, 1].set_title(f"Augmented Image 2 (Sample {i})")
        axes[2*i+1, 1].axis("off")
    
    plt.tight_layout()
    plt.show()


def train():
    # Load CSV and create pair_id column
    train_df = pd.read_csv(Config.train_csv)[['file_name', 'label']]
    train_df.columns = ['id', 'label']
    train_df["pair_id"] = train_df.index // 2

    # Split into train and validation if enabled.
    if Config.use_validation:
        pair_ids = train_df['pair_id'].unique()
        val_size = int(len(pair_ids) * Config.val_split)
        val_pairs = np.random.choice(pair_ids, val_size, replace=False)
        train_mask = ~train_df['pair_id'].isin(val_pairs)
        train_data = train_df[train_mask]
        val_data = train_df[~train_mask]
    else:
        train_data = train_df
        val_data = None

    train_tfm, val_tfm = get_transforms()
    train_ds = PairDataset(train_data, Config.img_dir, transform=train_tfm, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True, num_workers=4)
    
    if Config.use_validation:
        val_ds = PairDataset(val_data, Config.img_dir, transform=val_tfm, is_train=False)
        val_loader = DataLoader(val_ds, batch_size=Config.batch_size * 2, shuffle=False, num_workers=4)

    # Initialize the model
    model = PairComparator().to(Config.device)
    
    # Set up differential learning rates for backbone and comparison head
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "comparison_head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': Config.backbone_lr},
        {'params': head_params, 'lr': Config.head_lr}
    ], weight_decay=Config.weight_decay)
    
    # Use CosineAnnealingLR scheduler (or optionally OneCycleLR if desired)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    best_f1 = 0.0

    # Set up GradScaler for AMP if enabled
    scaler = torch.amp.GradScaler('cuda',enabled=Config.use_amp)

    for epoch in range(1, Config.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Gradual unfreezing (if enabled)
        if Config.dynamic_unfreeze:
            gradual_unfreeze(model, current_epoch=epoch, total_epochs=Config.epochs, start_fraction=0.9, target_fraction=0.75)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{Config.epochs}")
        for images, labels in pbar:
            images = images.to(Config.device)
            batch_size = images.size(0)
            
            # Apply mixup on first image in the pair (optional)
            if Config.mixup_alpha > 0:
                lam = np.random.beta(Config.mixup_alpha, Config.mixup_alpha)
                rand_index = torch.randperm(batch_size)
                images[:, 0] = lam * images[:, 0] + (1 - lam) * images[rand_index, 0]
                labels[:, 0] = lam * labels[:, 0] + (1 - lam) * labels[rand_index, 0]
            
            # Apply label smoothing
            labels = labels * (1 - Config.label_smoothing) + 0.5 * Config.label_smoothing
            # Our target is based on the first image's label (after possible swap)
            target = (labels[:, 0] > 0.5).float().to(Config.device)
            
            optimizer.zero_grad()
            # Use AMP autocast if enabled
            with torch.amp.autocast("cuda", enabled=Config.use_amp):
                outputs = model(images)
                loss = criterion(outputs, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > Config.threshold).float()
            correct += (preds == target).sum().item()
            total += batch_size
            
            avg_loss = total_loss / (pbar.n + 1)
            acc = correct / total
            pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'acc': f"{acc:.2%}"})
        
        scheduler.step()
        print(f"[Epoch {epoch}] Training Loss: {total_loss/len(train_loader):.4f}")
        
        if Config.use_validation:
            val_f1 = validate(model, val_loader)
            print(f"[Epoch {epoch}] Validation F1: {val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), "best_model.pth")
    
    return model


def validate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images = images.to(Config.device)
            # Target label is from first image of pair
            labels = labels[:, 0].cpu().numpy()
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > Config.threshold).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    f1 = f1_score(all_labels, all_preds)
    return f1


def create_submission(model):
    test_df = pd.read_csv(Config.test_csv)
    _, test_tfm = get_transforms()
    
    class TestPairDataset(Dataset):
        def __init__(self, df):
            self.df = df.sort_values('pair_id').reset_index(drop=True)
            self.pair_groups = df.groupby('pair_id')
            self.pair_ids = df['pair_id'].unique()
        
        def __len__(self):
            return len(self.pair_ids)
        
        def __getitem__(self, idx):
            pair_id = self.pair_ids[idx]
            pair = self.pair_groups.get_group(pair_id)
            return [pair.iloc[0]['id'], pair.iloc[1]['id']]
    
    test_ds = TestPairDataset(test_df)
    id_to_pred = {}
    model.eval()
    with torch.no_grad():
        for pair in tqdm(test_ds, desc="Processing Test Pairs"):
            images = []
            for img_id in pair:
                img_path = os.path.join(Config.img_dir, img_id)
                image = np.array(Image.open(img_path).convert('RGB'))
                transformed = test_tfm(image=image)['image']
                images.append(transformed)
            images = torch.stack(images).unsqueeze(0).to(Config.device)  # shape: (1, 2, C, H, W)
            
            # Apply Test Time Augmentation (TTA)
            if Config.tta_steps > 0:
                outputs = []
                for _ in range(Config.tta_steps):
                    # Horizontal flip as TTA augmentation
                    flipped = torch.flip(images, dims=[4])
                    outputs.append(model(flipped))
                pred = torch.stack(outputs).mean()
            else:
                pred = model(images)
            
            prob = torch.sigmoid(pred).item()
            pred_class = 1 if prob > Config.threshold else 0
            # If prediction indicates first image is AI, assign accordingly.
            if pred_class == 1:
                id_to_pred[pair[0]] = 1
                id_to_pred[pair[1]] = 0
            else:
                id_to_pred[pair[0]] = 0
                id_to_pred[pair[1]] = 1
    
    submission = test_df[['id']].copy()
    submission['label'] = submission['id'].map(id_to_pred)
    submission.to_csv("submission.csv", index=False)
    print("Submission created with pair-wise predictions!")


train_tfm, _ = get_transforms()
df = pd.read_csv(Config.train_csv)[['file_name', 'label']]
df.columns = ['id', 'label']
df["pair_id"] = df.index // 2
dataset = PairDataset(df, Config.img_dir, transform=train_tfm, is_train=True)
visualize_augmentations_with_raw(dataset, num_samples=5)


torch.manual_seed(Config.seed)
np.random.seed(Config.seed)

# Train the model
model = train()

# Load best model from validation if available
if Config.use_validation:
    model.load_state_dict(torch.load("best_model.pth"))

# Create submission using test dataset
create_submission(model)




