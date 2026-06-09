import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class CFG:
    test_soundscapes = "/kaggle/input/birdclef-2025/train_soundscapes"
    taxonomy_csv     = "/kaggle/input/birdclef-2025/taxonomy.csv"
    model_dir        = "/kaggle/input/efficientnetv2-in21k-focal"
    SR = 32000
    WINDOW_SIZE = 5
    N_FFT = 1034
    HOP_LENGTH = 64
    N_MELS = 136
    FMIN = 20
    FMAX = 16000
    TARGET_SHAPE = (256, 256)
    model_name = 'tf_efficientnetv2_s.in21k_ft_in1k'
    use_all_folds = False
    in_channels = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pseudo_label_threshold = 0.93 # for fallback in case dynamic threshold fails

cfg = CFG()

taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
species_ids = taxonomy_df['primary_label'].tolist()
num_classes = len(species_ids)
print(f"Loaded taxonomy with {num_classes} species classes.")
print(f"Using device: {cfg.device}")


# Step 1: Count number of training samples per class
train_df_original = pd.read_csv("/kaggle/input/birdclef-2025/train.csv")
label_counts = train_df_original['primary_label'].value_counts().to_dict()

# Step 2: Set dynamic thresholds (less common → lower threshold)
min_thresh = 0.55
max_thresh = 0.99

max_count = max(label_counts.values())
min_count = min(label_counts.values())

per_class_thresholds = {}
for label, count in label_counts.items():
    commonness = (count - min_count) / (max_count - min_count + 1e-8)  # 0 = rare, 1 = common
    per_class_thresholds[label] = min_thresh + (commonness ** 0.5) * (max_thresh - min_thresh)  # Apply square root

# Sanity check
print("Dynamic threshold example:")
for label in sorted(label_counts, key=label_counts.get)[:5]:  # 5 least frequent
    print(f"{label} (rare): {per_class_thresholds[label]:.3f}")
for label in sorted(label_counts, key=label_counts.get, reverse=True)[:5]:  # 5 most frequent
    print(f"{label} (common): {per_class_thresholds[label]:.3f}")


import timm
import torch
import torch.nn as nn

class BirdCLEFModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=True,
            in_chans=cfg.in_channels,
            drop_rate=0.0,
            drop_path_rate=0.0,
        )
        
        if 'efficientnet' in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in cfg.model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '') 
        
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(backbone_out, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, dict):
            features = features.get('features', features.get('out', features))
        if features.dim() == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits

model = BirdCLEFModel(cfg, num_classes=num_classes)
model.to(cfg.device)
model.eval() 


import cv2
from tqdm import tqdm

def audio_to_melspec(audio_data, cfg):
    if np.isnan(audio_data).any():
        mean_val = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_val)
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data, sr=cfg.SR, n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH, n_mels=cfg.N_MELS,
        fmin=cfg.FMIN, fmax=cfg.FMAX, power=2.0
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    return mel_spec_norm

def process_audio_segment(audio_segment, cfg):
    segment_length = cfg.SR * cfg.WINDOW_SIZE
    if len(audio_segment) < segment_length:
        audio_segment = np.pad(audio_segment, (0, segment_length - len(audio_segment)), mode='constant')
    mel = audio_to_melspec(audio_segment, cfg)
    if mel.shape != cfg.TARGET_SHAPE:
        mel = cv2.resize(mel, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
    return mel.astype(np.float32)


from collections import defaultdict

class_preds = defaultdict(list)
pseudo_mels = {}
total_segments = 0
TOP_N = 3

audio_files = sorted(Path(cfg.test_soundscapes).glob("*.ogg"))
print(f"Processing {len(audio_files)} files...")

with torch.no_grad():
    for audio_path in tqdm(audio_files, desc="Audio files", dynamic_ncols=True):
        audio_name = audio_path.name
        y, sr = librosa.load(audio_path, sr=cfg.SR)
        if sr != cfg.SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=cfg.SR)

        seg_samples = cfg.SR * cfg.WINDOW_SIZE  # 5 seconds
        total_segments += 1  # only one segment per file

        if len(y) < seg_samples:
            n_repeat = int(np.ceil(seg_samples / len(y)))
            y = np.tile(y, n_repeat)
        
        start = max(0, len(y) // 2 - seg_samples // 2)
        end = start + seg_samples
        segment = y[start:end]

        if len(segment) < seg_samples:
            segment = np.pad(segment, (0, seg_samples - len(segment)), mode='constant')

        mel = process_audio_segment(segment, cfg)
        tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(cfg.device)

        probs = torch.sigmoid(model(tensor)).cpu().numpy().squeeze()
        top_n_indices = np.argsort(probs)[-TOP_N:][::-1]

        for idx in top_n_indices:
            prob = float(probs[idx])
            label = species_ids[idx]
            threshold = per_class_thresholds.get(label, cfg.pseudo_label_threshold)

            if prob >= threshold:
                segment_name = f"{audio_name.replace('.ogg', '')}_center5s.ogg"
                class_preds[label].append((prob, segment_name, mel))


# Keep only top-K per class
results = []
results_with_conf = []  # for confidence analysis
MAX_PER_CLASS = 400  # configurable cap

for label, entries in class_preds.items():
    top_entries = sorted(entries, key=lambda x: -x[0])[:MAX_PER_CLASS]
    for prob, segment_name, mel in top_entries:
        results.append([segment_name, label, prob])
        results_with_conf.append({
            "filename": segment_name,
            "primary_label": label,
            "confidence": prob,
            "threshold": per_class_thresholds.get(label, cfg.pseudo_label_threshold)
        })
        pseudo_mels[segment_name.replace('.ogg', '')] = mel

print(f"Done. Collected {len(results)} pseudo-labels from {total_segments} segments.")

# DEBUG: show that our keys line up with the filenames in results
print("First 10 filenames from results (with .ogg):")
print([row[0] for row in results[:10]])
print("First 10 keys in pseudo_mels dict:")
print(list(pseudo_mels.keys())[:10])

print("\nCheck each of the first 10:")
for fn, lbl, prob in results[:10]:
    k = fn.replace(".ogg","")
    print(f"{fn}  → key='{k}'  in dict? {k in pseudo_mels}")

np.save("pseudo_mels.npy", pseudo_mels)

# Save confidence info separately for analysis
conf_df = pd.DataFrame(results_with_conf)
conf_df.to_csv("pseudo_label_confidences.csv", index=False)
print(f"Saved {len(conf_df)} entries with confidence scores to 'pseudo_label_confidences.csv'")

# Save train.csv-compatible format
pseudo_train = pd.DataFrame({
    "filename": [row[0] for row in results],
    "primary_label": [row[1] for row in results],
    "secondary_labels": [[] for _ in results],
    "latitude": [None] * len(results),
    "longitude": [None] * len(results),
    "author": ["pseudo"] * len(results),
    "rating": [0] * len(results),
    "collection": ["pseudo"] * len(results)
})

pseudo_train.to_csv("pseudo_train.csv", index=False)
print(f"Saved {len(pseudo_train)} pseudo-labeled samples to 'pseudo_train.csv'")
pseudo_train.head()

