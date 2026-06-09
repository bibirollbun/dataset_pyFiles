import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_auc_score,log_loss
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
import random
import copy
import librosa
from torch.cuda.amp import autocast, GradScaler
import warnings

warnings.filterwarnings("ignore")

# -------------------------
# 1. Configuration
# -------------------------
CONFIG = {
    'data_root': '/kaggle/input/celeb-df-v2',  # Update this path if needed
    'batch_size': 4,
    'num_frames': 20,
    'frame_size': 224,
    'num_epochs': 10,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'lstm_hidden_dim': 512,
    'dropout_rate': 0.5,
    'scheduler_patience': 3,
    'scheduler_factor': 0.5,
    'early_stopping_patience': 5,
    'model_save_path': './hybrid_model_checkpoints',
    'num_workers': 2,
    # Audio Params
    'audio_sr': 16000,
    'n_mfcc': 40,
    'audio_duration': 2.0,  # Duration of audio to analyze per video
}

os.makedirs(CONFIG['model_save_path'], exist_ok=True)

# -------------------------
# 2. Utilities
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# -------------------------
# 3. Data Processing (Video + Audio)
# -------------------------
def extract_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros((num_frames, CONFIG['frame_size'], CONFIG['frame_size'], 3), dtype=np.uint8)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return np.zeros((num_frames, CONFIG['frame_size'], CONFIG['frame_size'], 3), dtype=np.uint8)
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (CONFIG['frame_size'], CONFIG['frame_size']))
            frames.append(frame)
        else:
            frames.append(np.zeros((CONFIG['frame_size'], CONFIG['frame_size'], 3), dtype=np.uint8))
            
    cap.release()
    return np.array(frames)

def compute_freq_images(frames):
    n, h, w, c = frames.shape
    freq_images = np.zeros((n, h, w, 1), dtype=np.uint8)
    for i in range(n):
        img = frames[i].astype(np.float32)
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = np.log(np.abs(fshift) + 1e-8)
        norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        freq_images[i, :, :, 0] = norm.astype(np.uint8)
    return freq_images

def extract_audio_features(video_path, target_sr=16000, n_mfcc=40, duration=2.0):
    try:
        # Load audio with librosa (only first 'duration' seconds)
        y, sr = librosa.load(video_path, sr=target_sr, duration=duration, mono=True)
        
        # Pad if too short
        required_len = int(duration * target_sr)
        if len(y) < required_len:
            y = np.pad(y, (0, required_len - len(y)))
        else:
            y = y[:required_len]
            
        # Compute MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # Transpose to (Time, n_mfcc) -> (Seq_Len, 40)
        return mfcc.T 
    except Exception as e:
        # Return zeros if no audio or error
        # Time steps roughly = (Duration * SR) / Hop_Length. Default hop is 512.
        # 2.0 * 16000 / 512 ≈ 63 frames. We'll fix size later or pool it.
        return np.zeros((64, n_mfcc), dtype=np.float32)

class HybridDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]

        # 1. Video Frames
        frames = extract_frames(path, CONFIG['num_frames'])
        freq_frames = compute_freq_images(frames)
        
        # 2. Audio Features
        mfcc = extract_audio_features(path, CONFIG['audio_sr'], CONFIG['n_mfcc'], CONFIG['audio_duration'])
        
        # 3. Transform & Tensorize Video
        rgb_tensors = []
        freq_tensors = []
        
        if self.transform:
            for i in range(CONFIG['num_frames']):
                # RGB
                t_rgb = self.transform(frames[i])
                
                # Freq (Expand to 3 channels for EfficientNet)
                f_freq_3ch = np.repeat(freq_frames[i], 3, axis=2)
                t_freq = self.transform(f_freq_3ch)
                
                rgb_tensors.append(t_rgb)
                freq_tensors.append(t_freq)
        else:
            # Fallback transform
            to_tensor = transforms.ToTensor()
            for i in range(CONFIG['num_frames']):
                rgb_tensors.append(to_tensor(frames[i]))
                f_freq_3ch = np.repeat(freq_frames[i], 3, axis=2)
                freq_tensors.append(to_tensor(f_freq_3ch))

        rgb_tensor = torch.stack(rgb_tensors)   # [Seq, 3, H, W]
        freq_tensor = torch.stack(freq_tensors) # [Seq, 3, H, W]
        
        # Audio Tensor
        audio_tensor = torch.tensor(mfcc, dtype=torch.float32) # [Audio_Seq, 40]
        # Ensure fixed size for batching (truncate or pad to e.g., 64 steps)
        target_audio_len = 64
        if audio_tensor.shape[0] > target_audio_len:
            audio_tensor = audio_tensor[:target_audio_len, :]
        elif audio_tensor.shape[0] < target_audio_len:
            padding = torch.zeros((target_audio_len - audio_tensor.shape[0], CONFIG['n_mfcc']))
            audio_tensor = torch.cat([audio_tensor, padding], dim=0)

        return rgb_tensor, freq_tensor, audio_tensor, torch.tensor(label, dtype=torch.float)

# -------------------------
# 4. Model Components
# -------------------------

# --- A. Modality Attention (From your instruction) ---
class ModalityAttentionPerFrame(nn.Module):
    def __init__(self, feat_dim, hidden=256, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2)
        )

    def forward(self, feat_rgb, feat_fft):
        x = torch.cat([feat_rgb, feat_fft], dim=-1)
        logits = self.mlp(x)
        alpha = F.softmax(logits, dim=-1)
        a_rgb = alpha[..., 0:1]
        a_fft = alpha[..., 1:2]
        fused = a_rgb * feat_rgb + a_fft * feat_fft
        return fused

# --- B. The Autoencoder Branch (Paper Logic) ---
class TransformerBlock(nn.Module):
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class MutualAttentionFusion(nn.Module):
    def __init__(self, video_dim, audio_dim, fusion_dim=256):
        super().__init__()
        self.proj_v = nn.Linear(video_dim, fusion_dim)
        self.proj_a = nn.Linear(audio_dim, fusion_dim)
        self.cross_attn_v2a = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=4, batch_first=True)
        self.cross_attn_a2v = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=4, batch_first=True)
        self.norm_v = nn.LayerNorm(fusion_dim)
        self.norm_a = nn.LayerNorm(fusion_dim)

    def forward(self, x_v, x_a):
        feat_v = F.relu(self.proj_v(x_v))
        feat_a = F.relu(self.proj_a(x_a))
        attn_v, _ = self.cross_attn_v2a(query=feat_v, key=feat_a, value=feat_a)
        fused_v = self.norm_v(feat_v + attn_v)
        attn_a, _ = self.cross_attn_a2v(query=feat_a, key=feat_v, value=feat_v)
        fused_a = self.norm_a(feat_a + attn_a)
        return fused_v, fused_a

class AFMAE_Branch(nn.Module):
    def __init__(self, video_input_dim=1280, audio_input_dim=40, hidden_dim=256):
        super().__init__()
        self.encoder_v = nn.Sequential(nn.Linear(video_input_dim, hidden_dim), TransformerBlock(hidden_dim))
        self.encoder_a = nn.Sequential(nn.Linear(audio_input_dim, hidden_dim), TransformerBlock(hidden_dim))
        self.fusion = MutualAttentionFusion(hidden_dim, hidden_dim, hidden_dim)
        self.decoder_v = nn.Sequential(TransformerBlock(hidden_dim), nn.Linear(hidden_dim, video_input_dim))
        self.decoder_a = nn.Sequential(TransformerBlock(hidden_dim), nn.Linear(hidden_dim, audio_input_dim))

    def forward(self, video_feats, audio_feats):
        enc_v = self.encoder_v(video_feats)
        enc_a = self.encoder_a(audio_feats)
        latent_v, latent_a = self.fusion(enc_v, enc_a)
        recon_v = self.decoder_v(latent_v)
        recon_a = self.decoder_a(latent_a)
        return latent_v, latent_a, recon_v, recon_a

# --- C. The Original Detector (Refactored) ---
class DeepfakeDetectorDual(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Backbone: EfficientNet B0
        eff = models.efficientnet_b0(weights='DEFAULT')
        self.feature_extractor = nn.Sequential(eff.features, eff.avgpool)
        self.feature_dim = 1280
        
        # Modality Attention (RGB vs Freq)
        self.modality_attn = ModalityAttentionPerFrame(self.feature_dim, hidden=256)
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim, # Reduced from 2x because of Modality Attention
            hidden_size=config['lstm_hidden_dim'],
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=config['dropout_rate']
        )

    def get_cnn_features(self, x):
        # x: [Batch, Seq, 3, H, W]
        b, seq, c, h, w = x.size()
        x_in = x.view(b * seq, c, h, w)
        feat = self.feature_extractor(x_in)
        feat = feat.view(b, seq, -1) # [B, Seq, 1280]
        return feat

    def forward_features(self, x_rgb, x_freq):
        # 1. Extract raw features
        feat_rgb = self.get_cnn_features(x_rgb)
        feat_freq = self.get_cnn_features(x_freq)
        
        # 2. Fuse Modalities (RGB vs Freq)
        fused_seq = self.modality_attn(feat_rgb, feat_freq)
        
        # 3. Temporal Modeling (LSTM)
        lstm_out, _ = self.lstm(fused_seq) # [B, Seq, Hidden*2]
        
        # 4. Global Temporal Pooling (Last step or Attention)
        # Simple approach: Average pooling over time for stability
        lstm_pooled = torch.mean(lstm_out, dim=1) # [B, 1024]
        
        return lstm_pooled, fused_seq
        
class GatedFusion(nn.Module):
    def __init__(self, branch1_dim=1024, branch2_dim=512, bottleneck_dim=128, dropout_rate=0.3):
        super().__init__()
        # 1. The Bottleneck: Compress massive vectors to force the network to focus
        self.proj1 = nn.Sequential(
            nn.Linear(branch1_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.proj2 = nn.Sequential(
            nn.Linear(branch2_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 2. The Gate Network (now much smaller and harder to overfit)
        self.gate_net = nn.Sequential(
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(bottleneck_dim, 1),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(bottleneck_dim)

    def forward(self, x1, x2):
        h1 = self.proj1(x1)
        h2 = self.proj2(x2)
        
        combined = torch.cat([h1, h2], dim=1)
        z = self.gate_net(combined) # Gate value between 0.0 and 1.0
        
        fused = z * h1 + (1 - z) * h2
        return self.norm(fused), z


class BranchAttentionFusion(nn.Module):
    def __init__(self, branch1_dim=1024, branch2_dim=512, bottleneck_dim=128, dropout_rate=0.3):
        super().__init__()
        # 1. The Bottleneck Projections
        self.proj_q = nn.Sequential(
            nn.Linear(branch1_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.proj_kv = nn.Sequential(
            nn.Linear(branch2_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 2. Attention Mechanism (Now running on 128-dim instead of 512-dim)
        self.attn = nn.MultiheadAttention(embed_dim=bottleneck_dim, num_heads=4, batch_first=True, dropout=dropout_rate)
        self.norm = nn.LayerNorm(bottleneck_dim)
        
        # 3. Feed Forward Network (with heavy regularization)
        self.ffn = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x_lstm, x_ae):
        q = self.proj_q(x_lstm).unsqueeze(1)    # [Batch, 1, 128]
        kv = self.proj_kv(x_ae).unsqueeze(1)    # [Batch, 1, 128]
        
        attn_out, _ = self.attn(query=q, key=kv, value=kv)
        fused = attn_out.squeeze(1)             # [Batch, 128]
        
        fused = self.norm(fused + q.squeeze(1)) # Add & Norm
        fused = fused + self.ffn(fused)         # FFN + Residual
        return fused
        
# --- D. The Hybrid Super Model ---
class HybridDeepfakeDetector(nn.Module):
    def __init__(self, config, fusion_type='concat'):
        super().__init__()
        self.fusion_type = fusion_type
        
        # Branch 1: Supervised BiLSTM
        self.supervised_branch = DeepfakeDetectorDual(config)
        
        # Branch 2: Unsupervised Autoencoder
        self.ae_branch = AFMAE_Branch(video_input_dim=1280, audio_input_dim=40, hidden_dim=256)
        
        # Dimensions
        lstm_dim = 1024
        ae_dim = 512 # 256 (Video) + 256 (Audio)
        
        # --- FUSION STRATEGIES ---
        if fusion_type == 'gated':
            self.fusion_module = GatedFusion(lstm_dim, ae_dim)
            final_in_dim = 128
        elif fusion_type == 'attention':
            self.fusion_module = BranchAttentionFusion(lstm_dim, ae_dim)
            final_in_dim = 128
        else: # 'concat' (Default)
            self.fusion_module = nn.Identity()
            final_in_dim = lstm_dim + ae_dim

        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(final_in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, rgb, freq, audio):
        # 1. Get Features
        lstm_feat, cnn_fused_seq = self.supervised_branch.forward_features(rgb, freq)
        latent_v, latent_a, recon_v, recon_a = self.ae_branch(cnn_fused_seq, audio)
        
        # Pool AE features
        latent_v_mean = torch.mean(latent_v, dim=1)
        latent_a_mean = torch.mean(latent_a, dim=1)
        ae_feat = torch.cat([latent_v_mean, latent_a_mean], dim=1) # [B, 512]
        
        # 2. Apply Fusion
        if self.fusion_type == 'concat':
            combined = torch.cat([lstm_feat, ae_feat], dim=1)
        elif self.fusion_type == 'gated':
            combined, gate_val = self.fusion_module(lstm_feat, ae_feat)
        elif self.fusion_type == 'attention':
            combined = self.fusion_module(lstm_feat, ae_feat)
            
        # 3. Classify
        logits = self.classifier(combined)
        
        return logits.squeeze(-1), recon_v, recon_a, cnn_fused_seq

# -------------------------
# 5. Training Loop
# -------------------------
def train_hybrid(model, train_loader, val_loader, optimizer, device, epochs):
    scaler = GradScaler()
    # Loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for rgb, freq, audio, labels in loop:
            rgb, freq, audio, labels = rgb.to(device), freq.to(device), audio.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                logits, rec_v, rec_a, target_v = model(rgb, freq, audio)
                
                # A. Classification Loss (Supervised)
                loss_cls = bce_loss(logits, labels)
                
                # B. Reconstruction Loss (Unsupervised)
                # We want the AE to reconstruct the valid features
                loss_rec_v = mse_loss(rec_v, target_v.detach()) # Detach target to stop grad flow back to backbone
                loss_rec_a = mse_loss(rec_a, audio)
                
                # Total Loss
                loss = loss_cls + 0.5 * (loss_rec_v + loss_rec_a)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_acc += (preds == labels).sum().item()
            
            loop.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Acc: {train_acc/len(train_loader.dataset):.4f}")
        
        # Validation
        # ... inside train_hybrid function ...

        # --- VALIDATION PHASE ---
        model.eval()
        val_acc = 0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for rgb, freq, audio, labels in val_loader:
                rgb, freq, audio, labels = rgb.to(device), freq.to(device), audio.to(device), labels.to(device)
                
                # Forward pass
                logits, _, _, _ = model(rgb, freq, audio)
                
                # Calculate probabilities (0.0 to 1.0)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                # Accumulate stats
                val_acc += (preds == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # --- CALCULATE METRICS ---
        final_val_acc = val_acc / len(val_loader.dataset)
        
        # 1. Log Loss (The new metric)
        # We use a small eps (epsilon) to prevent log(0) errors if prob is exactly 0 or 1
        val_log_loss = log_loss(all_labels, all_probs, eps=1e-15)
        
        # 2. AUC (Area Under Curve)
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except:
            val_auc = 0.5 # Handle edge case with single class batch

        print(f"Validation Results - Epoch {epoch+1}:")
        print(f"  Accuracy: {final_val_acc:.4f}")
        print(f"  Log Loss: {val_log_loss:.4f}  <-- Lower is better")
        print(f"  AUC:      {val_auc:.4f}       <-- Higher is better")
        
        # Save Checkpoint (You can now choose to save based on best Log Loss if you want!)
        torch.save(model.state_dict(), os.path.join(CONFIG['model_save_path'], f"hybrid_epoch_{epoch+1}.pth"))

# -------------------------
# 6. Main Execution (FIXED)
# -------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = CONFIG['data_root']

    if not os.path.exists(data_root):
        raise ValueError(f"Dataset path not found: {data_root}")

    video_paths = []
    labels = []

    # Expected Celeb-DF-v2 structure:
    # Celeb-real
    # Celeb-synthesis
    # YouTube-real

    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov")):
                full_path = os.path.join(root, file)

                # Label logic:
                # 0 = Real
                # 1 = Fake
                root_lower = root.lower()

                if "synthesis" in root_lower:
                    label = 1
                elif "real" in root_lower:
                    label = 0
                else:
                    continue  # Skip unknown folders

                video_paths.append(full_path)
                labels.append(label)
    if len(video_paths) == 0:
        raise ValueError("No video files found. Check dataset path.")

    print(f"Total videos found: {len(video_paths)}")
    print(f"Real videos: {labels.count(0)}")
    print(f"Fake videos: {labels.count(1)}")
    # -------------------------
    # Train / Validation Split
    # -------------------------
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        video_paths,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    print(f"Train size: {len(train_paths)}")
    print(f"Validation size: {len(val_paths)}")
    # -------------------------
    # Transforms
    # -------------------------
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((CONFIG['frame_size'], CONFIG['frame_size'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    train_ds = HybridDataset(train_paths, train_labels, transform=train_transform)
    val_ds = HybridDataset(val_paths, val_labels, transform=train_transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    # -------------------------
    # Initialize Model
    # -------------------------
    model = HybridDeepfakeDetector(CONFIG, fusion_type='gated').to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    # -------------------------
    # Train
    # -------------------------
    train_hybrid(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        CONFIG['num_epochs']
    )

if __name__ == "__main__":
    main()



import torch
import numpy as np
import cv2
import librosa
from torchvision import transforms

# --- 1. Configuration ---
# Ensure these match what you used during training
TEST_CONFIG = {
    'num_frames': 20,
    'frame_size': 224,
    'audio_sr': 16000,
    'n_mfcc': 40,
    'audio_duration': 2.0,
    'lstm_hidden_dim': 512,
    'dropout_rate': 0.5,
    'use_spatial_dropout': True,
    'use_temporal_dropout': True,
    'use_attention': True
}

# --- 2. Load the Best Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# !!! CRITICAL: Set this to match your training experiment !!!
# Options: 'concat' (Baseline), 'gated', or 'attention'
FUSION_TYPE = 'concat' 

# Initialize the model structure with the correct fusion type
model = HybridDeepfakeDetector(TEST_CONFIG, fusion_type=FUSION_TYPE).to(device)

# Load the specific Epoch 4 weights (Best Model)
checkpoint_path = "/kaggle/working/hybrid_model_checkpoints/hybrid_epoch_4.pth"

print(f"Loading weights from: {checkpoint_path}")
try:
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Handle cases where the checkpoint saves just state_dict or the full info dict
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: File not found at {checkpoint_path}. Please check the file path.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

model.eval()

# --- 3. Preprocessing Helper Functions ---
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((TEST_CONFIG['frame_size'], TEST_CONFIG['frame_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def predict_video(video_path):
    print(f"Processing: {video_path}")
    
    # A. Extract Video Frames (RGB)
    # Ensure you have run the 'extract_frames' cell before this!
    try:
        frames = extract_frames(video_path, TEST_CONFIG['num_frames'])
    except ValueError as e:
        print(e)
        return "ERROR", 0.5

    # B. Compute Frequency Frames (FFT)
    freq = compute_freq_images(frames)
    freq_rgb = np.repeat(freq, 3, axis=3) # Expand to 3 channels
    
    # C. Extract Audio Features (MFCC)
    mfcc = extract_audio_features(video_path, TEST_CONFIG['audio_sr'], TEST_CONFIG['n_mfcc'], TEST_CONFIG['audio_duration'])

    # --- FIX: Force Audio Length to 64 Steps (Same as Training) ---
    target_audio_len = 64
    if mfcc.shape[0] > target_audio_len:
        mfcc = mfcc[:target_audio_len, :]
    elif mfcc.shape[0] < target_audio_len:
        padding = np.zeros((target_audio_len - mfcc.shape[0], TEST_CONFIG['n_mfcc']))
        mfcc = np.concatenate([mfcc, padding], axis=0)

    # D. Transform & Stack
    rgb_list = []
    freq_list = []
    for i in range(TEST_CONFIG['num_frames']):
        rgb_list.append(val_transform(frames[i].astype(np.uint8)))
        freq_list.append(val_transform(freq_rgb[i].astype(np.uint8)))
    
    # Add Batch Dimension [1, Seq, C, H, W]
    rgb_tensor = torch.stack(rgb_list).unsqueeze(0).to(device)
    freq_tensor = torch.stack(freq_list).unsqueeze(0).to(device)
    
    # Add Batch Dimension to Audio [1, Seq, MFCC]
    audio_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)

    # --- 4. Run Inference ---
    with torch.no_grad():
        # The model returns 4 values: (logits, recon_v, recon_a, cnn_feat)
        # We only need the first one (logits)
        logits, _, _, _ = model(rgb_tensor, freq_tensor, audio_tensor)
        
        # Convert Logits -> Probability (0.0 to 1.0)
        prob = torch.sigmoid(logits).item()

    # --- 5. Interpret Result ---
    # In this dataset: 1 = Real, 0 = Fake
    label = "REAL" if prob >= 0.5 else "FAKE"
    confidence = prob if prob >= 0.5 else 1 - prob
    
    print("-" * 30)
    print(f"Result: {label}")
    print(f"Probability Score (Realness): {prob:.4f}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("-" * 30)
    return label, prob

# --- 6. Run on a Test Video ---
# Replace this path with the video you want to test
test_video_path = "/kaggle/input/deepfake-detection-challenge/test_videos/bfdopzvxbi.mp4"

predict_video(test_video_path)

