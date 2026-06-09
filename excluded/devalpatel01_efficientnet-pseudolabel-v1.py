# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Pseudo-Labeling Notebook

# 1) Imports & Config
import os
import ast
import torch
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from torch import nn
from tqdm.notebook import tqdm
import timm

class CFG:
    sample_rate = 32000
    duration = 5         # seconds per segment
    n_mels    = 128
    n_fft     = 2048
    hop_length= 512
    num_classes = 206
    model_name  = 'efficientnet_b0'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

UNLABELED_DIR = "/kaggle/input/birdclef-2025/train_soundscapes"
OUTPUT_CSV   = "pseudo_labels.csv"
MODEL_PATH   = "/kaggle/input/efficientnet-v4/efficientnet_b0_frozen_overall_best.pth"

# 2) Load Pretrained Model
class EfficientNetFrozen(nn.Module):
    """
    EfficientNet‐B0 backbone where:
      - All layers up through blocks.5 are frozen.
      - Only backbone.blocks.6 and backbone.bn2 are trainable.
      - The classifier head is trainable.
    """
    def __init__(self, model_name="efficientnet_b0", n_classes=206, unfreeze_blocks=["blocks.6"]):
        super().__init__()
        # 1) Load pretrained EfficientNet‐B0, no head:
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=1,
            num_classes=0
        )
        # 2) First, freeze everything:
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 3) Unfreeze only the specified blocks (e.g. "blocks.6") and final batchnorm ("bn2"):
        #    If you want to unfreeze more, add them by name in unfreeze_blocks.
        for name, param in self.backbone.named_parameters():
            # “blocks.6.” is the final MBConv block in B0; “bn2” is the last batchnorm
            if any([name.startswith(block_name) for block_name in unfreeze_blocks]) \
               or name.startswith("bn2"):
                param.requires_grad = True

        # 4) Build a new classifier head on top of pooled features:
        num_features = self.backbone.num_features  # should be 1280 for B0
        self.classifier = nn.Linear(num_features, n_classes)
        # Ensure classifier is trainable:
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        feats = self.backbone(x)     # → (B, 1280)
        logits = self.classifier(feats)  # → (B, 206)
        return logits
model = EfficientNetFrozen(CFG.model_name, CFG.num_classes)
state = torch.load(MODEL_PATH, map_location=CFG.device)
model.load_state_dict(state)
model.to(CFG.device)
model.eval()

# 3) Audio → Log-Mel
def audio_to_logmel(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=CFG.sample_rate,
        n_fft=CFG.n_fft, hop_length=CFG.hop_length,
        n_mels=CFG.n_mels
    )
    logmel = librosa.power_to_db(mel)
    return (logmel - logmel.mean(axis=1, keepdims=True)) / (logmel.std(axis=1, keepdims=True) + 1e-6)

# 4) Pseudo-Labeling Loop
rows = []
threshold = 0.9    # keep only very confident predictions

for fp in tqdm(sorted(Path(UNLABELED_DIR).glob("*.ogg"))):
    y, _ = librosa.load(str(fp), sr=CFG.sample_rate)
    seg_len = CFG.duration * CFG.sample_rate
    num_segs = len(y) // seg_len
    for i in range(num_segs):
        seg = y[i*seg_len:(i+1)*seg_len]
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len-len(seg)))
        logmel = audio_to_logmel(seg)
        x = torch.tensor(logmel).unsqueeze(0).unsqueeze(0).float().to(CFG.device)
        with torch.no_grad():
            probs = torch.sigmoid(model(x)).cpu().numpy()[0]
        # pick labels > threshold
        labels = list(np.where(probs >= threshold)[0])
        if labels:
            # store row: filepath, end_time, list of label indices (or map to codes later)
            rows.append({
                "row_id": f"{fp.stem}_{(i+1)*CFG.duration}",
                "labels": labels,
                **{f"class_{c}": float(probs[c]) for c in labels}
            })

# 5) Save Pseudo-Labels CSV
pseudo_df = pd.DataFrame(rows)
# fill missing class columns with 0
all_cls = [f"class_{i}" for i in range(CFG.num_classes)]
for c in all_cls:
    if c not in pseudo_df:
        pseudo_df[c] = 0.0
pseudo_df = pseudo_df[["row_id"] + all_cls]
pseudo_df.to_csv(OUTPUT_CSV, index=False)
print("✅ Saved pseudo-labels:", OUTPUT_CSV)


