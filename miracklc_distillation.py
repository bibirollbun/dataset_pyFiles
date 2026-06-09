import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

# Machine learning
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Deep learning models
import timm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
DATA_ROOT = '/kaggle/input/birdclef-2025'  # Current directory
TAXONOMY_CSV = os.path.join(DATA_ROOT, 'taxonomy.csv')
TRAIN_CSV = os.path.join(DATA_ROOT, 'train.csv')

# Image parameters
TARGET_SHAPE = (256, 256)

class CFG:
    # Training parameters
    epochs = 25
    batch_size = 32
    
    # Model optimization
    model_name = 'vit_base_patch16_224'
    img_size = (256, 256)
    dropout_rate = 0.2
    
    # Knowledge distillation
    use_distillation = True
    distillation_temp = 3.0
    distillation_alpha = 0.5
    
    # Optimizer settings
    optimizer = 'AdamW'
    lr = 5e-4
    weight_decay = 1e-5
  
    # Learning rate scheduler
    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = epochs
    
    # Debug mode settings
    debug = False

class BirdCLEFDataset(Dataset):
    def __init__(self, df, spectrograms=None, mode="train"):
        self.df = df
        self.mode = mode
        self.spectrograms = spectrograms

        # Load taxonomy for label mapping
        taxonomy_df = pd.read_csv(TAXONOMY_CSV)
        self.species_ids = taxonomy_df['primary_label'].tolist()
        self.num_classes = len(self.species_ids)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}

        # Add sample name column for spectrogram lookup
        if 'samplename' not in self.df.columns:
            self.df['samplename'] = self.df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
                    
        # Report stats
        if self.spectrograms:
            sample_names = set(self.df['samplename'])
            found_samples = sum(1 for name in sample_names if name in self.spectrograms)
            print(f"Found {found_samples} spectrograms for {mode} dataset out of {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        samplename = row['samplename']
        
        # Get spectrogram or create blank if not found
        if self.spectrograms and samplename in self.spectrograms:
            spec = self.spectrograms[samplename]
        else:
            spec = np.zeros(TARGET_SHAPE, dtype=np.float32)
        
        # Add channel dimension: [1, H, W]
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)

        # One-hot encode label
        target = np.zeros(self.num_classes, dtype=np.float32)
        label = row['primary_label']
        if label in self.label_to_idx:
            target[self.label_to_idx[label]] = 1.0

        # Handle secondary labels if present
        if 'secondary_labels' in row and row['secondary_labels'] not in [[''], None, np.nan]:
            if isinstance(row['secondary_labels'], str):
                secondary_labels = eval(row['secondary_labels'])
            else:
                secondary_labels = row['secondary_labels']
            
            for label in secondary_labels:
                if label in self.label_to_idx:
                    target[self.label_to_idx[label]] = 1.0

        return {
            'melspec': spec,
            'target': torch.tensor(target, dtype=torch.float32),
            'filename': row['filename']
        }

class BirdCLEFModel(nn.Module):
    def __init__(self, num_classes, pretrained=True, cfg=None):
        super().__init__()
        
        # Use config if provided, otherwise use defaults
        self.cfg = cfg if cfg else CFG()
        
        # Create ViT for audio spectrograms
        self.vit = timm.create_model(
            self.cfg.model_name,
            pretrained=pretrained,
            img_size=self.cfg.img_size,
            in_chans=1,
            num_classes=0,  # Get embeddings
            drop_rate=self.cfg.dropout_rate
        )
        
        # Get the output feature dimension
        backbone_out = self.vit.embed_dim
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(backbone_out),
            nn.Dropout(0.2),
            nn.Linear(backbone_out, num_classes)
        )
            
    def forward(self, x):
        # Extract features from ViT
        features = self.vit(x)
        
        # Get logits from classifier
        logits = self.classifier(features)
            
        return logits

class AsymmetricLossMultiLabel(nn.Module):
    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=False,
        reduction="mean",
    ):
        super().__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w

        if self.reduction == "mean":
            return -loss.mean()
        if self.reduction == "sum":
            return -loss.sum()

        return -loss

def collate_fn(batch):
    """Collate function for dataloaders"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}
        
    # Collect items by key
    melspecs = []
    targets = []
    filenames = []
    
    for item in batch:
        if 'melspec' in item:
            melspecs.append(item['melspec'])
        if 'target' in item:
            targets.append(item['target'])
        if 'filename' in item:
            filenames.append(item['filename'])
    
    # Stack tensors when possible
    result = {}
    if melspecs:
        if isinstance(melspecs[0], torch.Tensor) and all(m.shape == melspecs[0].shape for m in melspecs):
            result['melspec'] = torch.stack(melspecs)
        else:
            result['melspec'] = melspecs
    
    if targets and isinstance(targets[0], torch.Tensor):
        result['target'] = torch.stack(targets)
    else:
        result['target'] = targets
        
    if filenames:
        result['filename'] = filenames
    
    return result

def load_teacher_model(model_path, num_classes):
    """Load the teacher model for distillation"""
    try:
        print(f"Loading teacher model from {model_path}")
        checkpoint = {}
        try:
            # Try with weights_only=False (has better compatibility)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except:
            # Fallback: try with weights_only=True
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            except Exception as e:
                print(f"Error loading model with standard approach: {e}")
                # Last resort: try to load directly with safe_globals
                try:
                    from torch.serialization import safe_globals
                    import numpy as np
                    with safe_globals([np.core.multiarray.scalar, np.dtype]):
                        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                except Exception as e2:
                    print(f"All loading attempts failed: {e2}")
                    return None
        
        # Create a model instance
        cfg = CFG()
        teacher = BirdCLEFModel(num_classes=num_classes, pretrained=False, cfg=cfg)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            teacher.load_state_dict(checkpoint['model_state_dict'])
            print("Teacher model loaded successfully")
        else:
            print("Warning: Could not find model_state_dict in checkpoint")
            return None
            
        # Set to eval mode
        teacher.eval()
        return teacher
            
    except Exception as e:
        print(f"Error loading teacher model: {e}")
        return None

def distillation_loss(outputs, labels, teacher_outputs, temp, alpha):
    """Knowledge distillation loss function"""
    # Hard target loss (standard cross-entropy)
    hard_loss = nn.BCEWithLogitsLoss()(outputs, labels)
    
    # Soft target loss (KL divergence between softened distributions)
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        torch.log_softmax(outputs / temp, dim=1),
        torch.softmax(teacher_outputs / temp, dim=1)
    ) * (temp * temp)
    
    # Combine losses
    return alpha * hard_loss + (1 - alpha) * soft_loss

def train_one_epoch(model, teacher_model, loader, optimizer, criterion, device, cfg):
    model.train()
    losses = []
    
    with tqdm(loader, desc="Training") as pbar:
        for batch in pbar:
            if not batch:  # Skip empty batches
                continue
                
            inputs = batch['melspec'].to(device)
            targets = batch['target'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            if cfg.use_distillation and teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                loss = distillation_loss(
                    outputs, 
                    targets, 
                    teacher_outputs, 
                    cfg.distillation_temp, 
                    cfg.distillation_alpha
                )
            else:
                loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Record loss
            losses.append(loss.item())
            
            # Update progress bar
            pbar.set_postfix({'loss': np.mean(losses[-10:]) if losses else 0})
    
    return np.mean(losses)

def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            if not batch:  # Skip empty batches
                continue
                
            inputs = batch['melspec'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Collect results
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            losses.append(loss.item())
    
    # Calculate metrics
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    
    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)
    
    return avg_loss, auc

def calculate_auc(targets, outputs):
    """Calculate AUC for multi-label classification"""
    num_classes = targets.shape[1]
    aucs = []
    
    # Convert logits to probabilities
    probs = 1 / (1 + np.exp(-outputs))
    
    for i in range(num_classes):
        # Only calculate AUC for classes that have positive samples
        if np.sum(targets[:, i]) > 0:
            class_auc = roc_auc_score(targets[:, i], probs[:, i])
            aucs.append(class_auc)
    
    return np.mean(aucs) if aucs else 0.0

def run_training(df, spectrograms, cfg):
    """Run training with knowledge distillation"""
    
    # Load taxonomy data
    taxonomy_df = pd.read_csv(TAXONOMY_CSV)
    num_classes = len(taxonomy_df['primary_label'])
    
    # Setup k-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Train for fold 0 only (simplified)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['primary_label'])):
        if fold > 0:  # Only use fold 0 for simplicity
            break
            
        print(f'\n{"="*30} Fold {fold} {"="*30}')
        
        # Split data
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        print(f'Training set: {len(train_df)} samples')
        print(f'Validation set: {len(val_df)} samples')
        
        # Create datasets
        train_dataset = BirdCLEFDataset(train_df, spectrograms=spectrograms, mode='train')
        val_dataset = BirdCLEFDataset(val_df, spectrograms=spectrograms, mode='valid')
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=True, 
            num_workers=2,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=False, 
            num_workers=2,
            collate_fn=collate_fn
        )
        
        # Create student model
        model = BirdCLEFModel(num_classes=num_classes, pretrained=True, cfg=cfg).to(device)
        
        # Load teacher model for distillation (if enabled)
        teacher_model = None
        if cfg.use_distillation:
            teacher_model = load_teacher_model('/kaggle/input/distill-vit/pytorch/default/1/model_fold4.pth', num_classes)
            if teacher_model:
                teacher_model = teacher_model.to(device)
        
        # Setup training components
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        criterion = AsymmetricLossMultiLabel()
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.min_lr)
        
        # Training loop
        best_auc = 0
        best_epoch = 0
        
        for epoch in range(cfg.epochs):
            print(f"\nEpoch {epoch+1}/{cfg.epochs}")
            
            # Train
            train_loss = train_one_epoch(
                model, 
                teacher_model, 
                train_loader, 
                optimizer, 
                criterion, 
                device,
                cfg
            )
            
            # Validate
            val_loss, val_auc = validate(model, val_loader, criterion, device)
            
            # Update learning rate
            scheduler.step()
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            
            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch + 1
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': val_auc
                }, f'distilled_model_fold{fold}.pth')
                print(f"New best AUC: {best_auc:.4f} - Model saved")
        
        print(f"\nBest AUC: {best_auc:.4f} at epoch {best_epoch}")

def main():
    # Check if required files exist
    if not os.path.exists(TAXONOMY_CSV):
        print(f"Error: {TAXONOMY_CSV} not found")
        return
    
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: {TRAIN_CSV} not found")
        return
    
    # Load data
    print("Loading data...")
    taxonomy_df = pd.read_csv(TAXONOMY_CSV)
    train_df = pd.read_csv(TRAIN_CSV)
    
    # Load spectrograms
    print("Loading spectrograms...")
    spectrograms = None
    try:
        spectrograms = np.load("/kaggle/input/birdclef-melspec-data/bird_mel_spectrograms.npy", allow_pickle=True).item()
        print(f"Loaded {len(spectrograms)} spectrograms")
    except Exception as e:
        print(f"Error loading spectrograms: {e}")
    
    # Run training
    cfg = CFG()
    run_training(train_df, spectrograms, cfg)

if __name__ == "__main__":
    main() 

