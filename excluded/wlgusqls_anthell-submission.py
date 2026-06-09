# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json, subprocess
from pathlib import Path
from collections import Counter

import cv2, dlib, torch
import numpy as np
from PIL import Image
from skimage import transform as trans
from torchvision import transforms
from tqdm.auto import tqdm
import timm
import torch.nn as nn

# ───────────────────────────── parameters ─────────────────────────────
MODEL_PATH = "/kaggle/input/pth-inception/model_epoch08.pt"
TEST_DIR   = "/kaggle/input/Deepfake_Detection_and_Generation_Challenge_Blue_Team/test_sample_videos"
OUT_CSV    = "/kaggle/working/submission.csv"
IMG_SIZE   = 128
MAX_FRAMES = 32
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])
majority = lambda xs: Counter(xs).most_common(1)[0][0] if xs else 0


IDX5 = [37, 44, 30, 49, 55]
REF5 = np.array([
    [30.2946, 51.6963], [65.5318, 51.5014],
    [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]
], np.float32)
REF5[:,0] += 8.0
REF5 *= IMG_SIZE / 112.0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "/kaggle/input/face-landmarks-81/shape_predictor_81_face_landmarks.dat"
)

def extract_aligned(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean_b = np.mean(rgb)
    if mean_b < 50:
        yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        inv_gamma = 1.0 / 1.1
        table = np.array([(i/255.0)**inv_gamma * 255 for i in range(256)], dtype='uint8')
        rgb = cv2.LUT(rgb, table)
    faces = detector(rgb, 1)
    if not faces:
        return None
    rect = max(faces, key=lambda r: r.width()*r.height())
    shp  = predictor(rgb, rect)
    kp = np.float32([[shp.part(i).x, shp.part(i).y] for i in IDX5])
    scale = 1.3
    dst = REF5.copy()
    m = IMG_SIZE * (scale - 1) / 2
    dst[:,0] += m
    dst[:,1] += m
    tform = trans.SimilarityTransform()
    tform.estimate(kp, dst)
    warped = cv2.warpAffine(
        rgb,
        tform.params[:2],
        (int(IMG_SIZE*scale), int(IMG_SIZE*scale))
    )
    return cv2.resize(warped, (IMG_SIZE, IMG_SIZE))

def get_frames(vp: Path, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total-1, max_frames, dtype=int)
    aligned, raw = [], []
    for fid in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ret, frame = cap.read()
        if not ret: continue
        raw.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        a = extract_aligned(frame)
        if a is not None:
            aligned.append(a)
    cap.release()
    return aligned if aligned else raw



class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lmbda):
        ctx.lmbda = lmbda
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad):
        return -ctx.lmbda * grad, None

class DANN(nn.Module):
    def __init__(self, num_domains, num_comp):
        super().__init__()
        self.backbone = timm.create_model(
            "inception_resnet_v2", pretrained=False, features_only=True
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        dim = self.backbone.feature_info[-1]["num_chs"]
        self.cls_head  = nn.Linear(dim, 2)
        self.dom_head  = nn.Linear(dim, num_domains)
        self.comp_head = nn.Linear(dim, num_comp)
    def forward(self, x, lmbda=1.0):
        f = self.backbone(x)[-1]
        f = self.pool(f).view(f.size(0), -1)
        y_cls = self.cls_head(f)
        f_rev = GradReverse.apply(f, lmbda)
        y_dom = self.dom_head(f_rev)
        y_comp = self.comp_head(f_rev)
        return y_cls, y_dom, y_comp





def main():
    from pathlib import Path
    # Ensure TEST_DIR is Path
    test_dir = Path(TEST_DIR)

    NUM_DOMAINS = 7  # 학습 시 동일 설정
    NUM_COMP    = 2
    device = DEVICE

    model = DANN(NUM_DOMAINS, NUM_COMP).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("모델 가중치 로드 완료")

    results = []
    for vp in tqdm(sorted(test_dir.glob("*.mp4")), desc="Inference"):
        frames = get_frames(vp)
        if not frames:
            results.append({"ID": f"{vp.stem}.mp4", "label": 1})
            continue
        batch = torch.stack([transform(Image.fromarray(f)) for f in frames]).to(device)
        with torch.no_grad():
            logits, _, _ = model(batch, lmbda=0.0)
            preds = logits.argmax(1).cpu().tolist()
        results.append({"ID": f"{vp.stem}.mp4", "label": majority(preds)})

    import pandas as pd
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"제출 파일 저장 → {OUT_CSV}")


if __name__ == "__main__":
    main()

