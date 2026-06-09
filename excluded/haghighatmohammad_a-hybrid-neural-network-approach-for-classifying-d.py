!pip install timm


import os
import cv2
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from tqdm.auto import tqdm
import timm
import gc
from PIL import Image
import warnings

# Suppress noise
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
CONFIG = {
    'seed': 42,
    'img_size': 608,
    'batch_size': 8,       # 4 per GPU (T4 x2)
    'accum_steps': 4,      # Effective Batch = 32
    'num_classes': 5,
    'epochs': 20,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    
    # EyePACS Paths (From your previous logs)
    'img_dir': '../input/resized-2015-2019-blindness-detection-images/resized train 15',
    'csv_path': '../input/resized-2015-2019-blindness-detection-images/labels/trainLabels15.csv',
    
    'checkpoint_path': 'eyepacs_checkpoint.pth',
    'best_model_path': 'best_eyepacs_model.pth'
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CONFIG['seed'])

# --- PREPROCESSING (Ben Graham) ---
class BenGrahamTransform:
    def __init__(self, sigmaX=10):
        self.sigmaX = sigmaX
    def __call__(self, img):
        image = np.array(img)
        # 1. Convert to Grey to find circle
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # 2. Crop to circle (remove black borders)
        mask = gray > 7
        if mask.any():
            image = image[np.ix_(mask.any(1),mask.any(0))]
        
        # 3. Resize to target (required for adding weighted)
        image = cv2.resize(image, (CONFIG['img_size'], CONFIG['img_size']))
        
        # 4. Ben Graham Method (Color)
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), self.sigmaX), -4, 128)
        return Image.fromarray(image)

# --- DATASET CLASS ---
class EyePACSDataset(Dataset):
    def __init__(self, df, base_path, extension=".jpg", transform=None):
        self.df = df
        self.base_path = base_path
        self.transform = transform
        self.extension = extension # EyePACS is usually .jpeg or .jpg

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 2015 CSV columns: 'image', 'level'
        image_id = self.df.iloc[idx]['image']
        label = self.df.iloc[idx]['level']
        
        # Construct path (Try .jpg and .jpeg)
        img_path = os.path.join(self.base_path, f"{image_id}{self.extension}")
        
        image = None
        try:
            if os.path.exists(img_path):
                image = cv2.imread(img_path)
            else:
                # Try alternative extension
                alt_ext = ".jpeg" if self.extension == ".jpg" else ".jpg"
                alt_path = os.path.join(self.base_path, f"{image_id}{alt_ext}")
                if os.path.exists(alt_path):
                    image = cv2.imread(alt_path)
        except: pass

        if image is None:
            # Return black image if corrupted/missing to prevent crash
            image = np.zeros((CONFIG['img_size'], CONFIG['img_size'], 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Basic resize before transform
        image = cv2.resize(image, (CONFIG['img_size'], CONFIG['img_size']))
        img = Image.fromarray(image)
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.long)

# --- MODEL ---
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x): return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x): return x * self.sigmoid(x)

class CoordinateAttention(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))
        return identity * a_h * a_w

class HybridModel(nn.Module):
    def __init__(self, num_classes=5, img_size=256): 
        super(HybridModel, self).__init__()
        self.effnet = timm.create_model('efficientnet_b3', pretrained=True)
        self.eff_n_features = self.effnet.classifier.in_features
        self.effnet.classifier = nn.Identity() 
        self.effnet.global_pool = nn.Identity() 
        self.eff_mhsa = nn.MultiheadAttention(embed_dim=self.eff_n_features, num_heads=8, batch_first=True)

        self.swin = timm.create_model('swinv2_tiny_window8_256', pretrained=True, img_size=img_size, strict_img_size=False)
        self.swin_n_features = self.swin.head.in_features
        self.swin.head = nn.Identity() 
        self.swin_coord_att = CoordinateAttention(inp=self.swin_n_features)
        
        fusion_dim = self.eff_n_features + self.swin_n_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        eff_feat = self.effnet.forward_features(x)
        b, c, h, w = eff_feat.shape
        eff_tokens = eff_feat.flatten(2).transpose(1, 2)
        eff_att, _ = self.eff_mhsa(eff_tokens, eff_tokens, eff_tokens)
        eff_out = torch.mean(eff_att, dim=1) 
        swin_feat = self.swin.forward_features(x)
        if swin_feat.dim() == 4: swin_feat = swin_feat.permute(0, 3, 1, 2)
        swin_att = self.swin_coord_att(swin_feat)
        swin_out = torch.mean(swin_att.flatten(2), dim=2)
        return self.classifier(torch.cat((eff_out, swin_out), dim=1))

# --- TRAINING LOOP ---
def run_eyepacs_training():
    print(f"ðŸš€ STARTING EYEPACS TRAINING: {CONFIG['img_size']}x{CONFIG['img_size']}")
    
    gpu_count = torch.cuda.device_count()
    print(f"âœ… Found {gpu_count} GPUs.")
    
    # Transforms
    train_transforms = transforms.Compose([
        BenGrahamTransform(sigmaX=10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # No extra color jitter needed because Ben Graham is already aggressive
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        BenGrahamTransform(sigmaX=10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load Data
    print("Loading CSV...")
    df = pd.read_csv(CONFIG['csv_path'])
    # Clean non-existing files logic would be good, but expensive. We rely on Dataset try-catch.
    
    # Split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['level'], random_state=CONFIG['seed'])
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    # Weighted Sampler
    class_counts = train_df['level'].value_counts().sort_index().values
    weights = 1. / class_counts
    samples_weights = torch.DoubleTensor([weights[t] for t in train_df['level']])
    # Sample 12,000 per epoch for manageable time
    sampler = WeightedRandomSampler(samples_weights, num_samples=12000, replacement=True)
    
    ds_train = EyePACSDataset(train_df, CONFIG['img_dir'], extension=".jpg", transform=train_transforms)
    ds_val = EyePACSDataset(val_df, CONFIG['img_dir'], extension=".jpg", transform=val_transforms)
    
    dl_train = DataLoader(ds_train, batch_size=CONFIG['batch_size'], sampler=sampler, num_workers=4, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    model = HybridModel(num_classes=5, img_size=CONFIG['img_size'])
    model = model.to(CONFIG['device'])
    
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer & Scheduler (Safe settings)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-4, 
        epochs=CONFIG['epochs'], 
        steps_per_epoch=len(dl_train) // CONFIG['accum_steps'],
        pct_start=0.1
    )
    
    # Checkpoint Logic
    start_epoch = 0
    best_f1 = 0.0
    if os.path.exists(CONFIG['checkpoint_path']):
        print(f"ðŸ”„ Checkpoint found! Resuming...")
        checkpoint = torch.load(CONFIG['checkpoint_path'])
        model.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint['model_state_dict'].items()})
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_f1 = checkpoint['best_f1']
        print(f"ðŸ”„ Resumed at Epoch {start_epoch+1}")

    if gpu_count > 1:
        model = nn.DataParallel(model)

    def get_metrics(targets, preds):
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='weighted')
        kappa = cohen_kappa_score(targets, preds, weights='quadratic')
        return acc, f1, kappa

    # Training Loop
    for epoch in range(start_epoch, CONFIG['epochs']):
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []
        
        optimizer.zero_grad()
        pbar = tqdm(dl_train, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / CONFIG['accum_steps']
            
            scaler.scale(loss).backward()
            
            if ((batch_idx + 1) % CONFIG['accum_steps'] == 0) or (batch_idx + 1 == len(dl_train)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            loss_val = loss.item() * CONFIG['accum_steps']
            if np.isnan(loss_val):
                optimizer.zero_grad()
                continue

            running_loss += loss_val
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.detach().cpu().numpy())
            train_targets.extend(labels.detach().cpu().numpy())
            
            pbar.set_postfix({'loss': f"{loss_val:.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})
            
        t_acc, t_f1, t_kappa = get_metrics(train_targets, train_preds)
        t_loss = running_loss / len(dl_train)

        # Validation
        gc.collect()
        torch.cuda.empty_cache()
        model.eval()
        val_loss_run = 0.0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for images, labels in dl_val:
                images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss_v = criterion(outputs, labels)
                    
                val_loss_run += loss_v.item()
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        v_acc, v_f1, v_kappa = get_metrics(val_targets, val_preds)
        v_loss = val_loss_run / len(dl_val)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  [Train] Loss: {t_loss:.4f} | Acc: {t_acc:.4f} | F1: {t_f1:.4f} | Kappa: {t_kappa:.4f}")
        print(f"  [Valid] Loss: {v_loss:.4f} | Acc: {v_acc:.4f} | F1: {v_f1:.4f} | Kappa: {v_kappa:.4f}")
        
        # Save
        model_state = model.module.state_dict() if gpu_count > 1 else model.state_dict()
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_f1': best_f1
        }
        torch.save(checkpoint, CONFIG['checkpoint_path'])
        
        if v_f1 > best_f1:
            best_f1 = v_f1
            checkpoint['best_f1'] = best_f1
            torch.save(checkpoint, CONFIG['checkpoint_path'])
            torch.save(model_state, CONFIG['best_model_path'])
            print(f"  [+] Best Model Saved (F1: {best_f1:.4f})")

run_eyepacs_training()

