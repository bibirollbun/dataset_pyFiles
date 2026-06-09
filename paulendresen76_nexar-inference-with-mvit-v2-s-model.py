#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import math
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from tqdm.notebook import tqdm
from pathlib import Path
from torchvision import transforms
from PIL import Image

# --- CONFIG ---
CHECKPOINT_PATH = '/kaggle/input/bestmodelnexarchallenge/pytorch/default/1/del025weight_61_acc_93.65_CP_81.80_CR_52.01_NCP_94.51_NCR_98.62.pt' 
TEST_CSV = '/kaggle/input/nexar-collision-prediction/test.csv'
TEST_DIR = '/kaggle/input/nexar-collision-prediction/test/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FRAME_SIZE = (256, 256) #Initial scaling from video
NUM_FRAMES = 16
FRAME_STEP = 4

class NewCrashPredictor(nn.Module):
    def __init__(self, num_frames=16):
        super().__init__()

        self.num_frames = num_frames

        # Load pretrained backbone with proper weights
        self.backbone_rgb = mvit_v2_s(weights=MViT_V2_S_Weights.DEFAULT)
        self.backbone_rgb.head = nn.Identity()

        # Output dimension from MViT V2 Small
        hidden_size = 768

        # Binary classifier for event occurrence
        self.event_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Binary output: will event occur or not
        )

    def forward(self, x, return_embeddings=False):
        # Process through modified backbone
        features = self.backbone_rgb(x)

        # Stage 1: Predict if event will occur (return logits, not probability)
        event_logits = self.event_classifier(features) 

        if return_embeddings:
            return event_logits,features

        return event_logits

# --- FRAME EXTRACTION (as in preprocess_videos.py) ---
def extract_last_frames(path, num_frames=16, step=4):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < (num_frames-1)*step+1:
        indices = [0]*num_frames
    else:
        indices = [total - 1 - i*step for i in reversed(range(num_frames))]

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    ])
   
    frames = []
    for idx in tqdm(indices, desc=f"Extracting frames from {os.path.basename(path)}", leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        # Stupid - Save frame to temp file and load again as jpeg to get exact same results as other/submitted inference code.
        frame_path = f"temp.jpg"
        cv2.imwrite(frame_path, frame)
        #load image again
        frame = Image.open(frame_path)
        frame = preprocess(frame)
        frames.append(frame)

    cap.release()
    frames = torch.stack(frames, dim=1)  # (C, T, H, W)
    return frames.unsqueeze(0)  # (1, C, T, H, W)

# --- LOAD MODEL
print ("Loading model...")
model = NewCrashPredictor()
checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model = model.to(DEVICE)
model.eval()

# --- INFERENCE ---
df_test = pd.read_csv(TEST_CSV)
df_test['id'] = df_test['id'].astype(str).str.zfill(5)
results = []

with torch.no_grad():
    for vid in tqdm(df_test['id'], desc="Inference"):
        video_path = os.path.join(TEST_DIR, f"{vid}.mp4")
        frames = extract_last_frames(video_path) #load_last_frames_from_folder(f"test_frames256/{vid}")
        frames = frames.to(DEVICE)
        logits = model(frames)
        prob = torch.sigmoid(logits).cpu().numpy().flatten()[0]
        print (f"Video {vid} has probability {prob} of event occurring")
        results.append({'id': vid, 'score': prob})

submission = pd.DataFrame(results)
submission.to_csv('submission.csv', index=False)
print("✅ Written → submission.csv")


