!cp -r /kaggle/input/kcc-dfc/* /kaggle/working/ 


!pip install -r requirements.txt


import os
import glob
import json
import sys

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import cv2

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import center_crop

# feat RetinaFace
from feat.utils.image_operations import convert_image_to_tensor, convert_color_vector_to_tensor
from feat.face_detectors.Retinaface.Retinaface_model import (
    RetinaFace as PyFeatRetina,
    postprocess_retinaface
)

from model import DeepFakeDetector

# =====================================================
# 1) 상수 세팅
# =====================================================
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RETINA_DIR   = "./models/retinaface"
ADA_DIR      = "./models/cvlface_adaface_ir50_ms1mv2"
FRAME_SIZE   = (224, 224)
REC_INPUT_SZ = (112, 112)

BATCH_SIZE   = 4
NUM_FRAMES   = 16
STRATEGY     = 'consecutive'
THRESHOLD    = 0.5

# =====================================================
# 2) RetinaFace 모델 로딩
# =====================================================
with open(os.path.join(RETINA_DIR, "config.json"), "r") as f:
    retina_cfg = json.load(f)
retina_model = PyFeatRetina(cfg=retina_cfg, phase="test")
ckpt = torch.load(
    os.path.join(RETINA_DIR, "mobilenet0.25_Final.pth"),
    map_location=DEVICE,
    weights_only=True
)
retina_model.load_state_dict(ckpt)
retina_model.eval().to(DEVICE)

_MEAN_VEC = np.array([123, 117, 104], dtype=np.float32)

def detect_faces(img: Image.Image, threshold=0.6):
    img_np       = np.array(img)
    frame_tensor = convert_image_to_tensor(img_np).to(DEVICE)
    mean_tensor  = convert_color_vector_to_tensor(_MEAN_VEC).to(DEVICE)
    x            = frame_tensor - mean_tensor

    loc, conf, landm = retina_model(x)
    out = postprocess_retinaface(loc, conf, landm, retina_cfg, x, device=DEVICE)

    boxes_full = out["boxes"]
    scores_all = out["scores"]
    keep       = scores_all >= threshold
    if keep.sum() == 0:
        # 변경: device는 키워드 인자로 전달합니다.
        return (
            torch.empty((0, 4), device=DEVICE),
            torch.empty((0,),    device=DEVICE),
            torch.empty((0, 10), device=DEVICE)
        )

    bf = boxes_full[keep]
    sf = scores_all[keep]
    x1, y2, x2, y1 = bf[:,1], bf[:,2], bf[:,3], bf[:,4]
    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    idx   = torch.argsort(sf, descending=True)
    return boxes[idx], sf[idx], out["landmarks"][keep][idx]

# =====================================================
# 3) AdaFace 모델 로딩
# =====================================================
sys.path.append(os.path.abspath(ADA_DIR))
from wrapper import ModelConfig, CVLFaceRecognitionModel

cfg = ModelConfig(model_path=os.path.join(ADA_DIR, "model.safetensors"))
rec_model = CVLFaceRecognitionModel(cfg)
rec_model.eval().to(DEVICE)

def extract_features(x: torch.Tensor) -> torch.Tensor:
    m = rec_model.model
    if hasattr(m, 'net'):
        m = m.net
    x = m.input_layer(x)
    return m.body(x)

# =====================================================
# 4) Dataset 클래스
# =====================================================
class ExternalVideoDataset(Dataset):
    def __init__(self, video_files, num_frames=NUM_FRAMES, strategy=STRATEGY):
        self.video_files = video_files
        self.num_frames  = num_frames
        self.strategy    = strategy
        self.resize      = transforms.Resize(FRAME_SIZE)
        self.to_tensor   = transforms.ToTensor()

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        vp    = self.video_files[idx]
        cap   = cv2.VideoCapture(vp)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise RuntimeError(f"Cannot read video: {vp}")

        if total < self.num_frames:
            idxs = list(range(total)) + [total-1]*(self.num_frames-total)
            idxs = idxs[:self.num_frames]
        elif self.strategy == 'uniform':
            idxs = [int(i * total / self.num_frames) for i in range(self.num_frames)]
        else:
            idxs = list(range(self.num_frames))

        feats, imgs = [], []
        for f in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, total-1)
                ret, frame = cap.read()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img       = Image.fromarray(frame_rgb)
            img224    = self.resize(img)

            boxes, _, _ = detect_faces(img224, threshold=THRESHOLD)
            side        = min(FRAME_SIZE)
            if boxes.shape[0] == 0:
                crop = center_crop(img224, (side, side)).resize(REC_INPUT_SZ)
            else:
                x1, y1, x2, y2 = boxes[0].to(torch.int).tolist()
                W, H = FRAME_SIZE
                x1 = max(0, min(x1, W)); y1 = max(0, min(y1, H))
                x2 = max(0, min(x2, W)); y2 = max(0, min(y2, H))
                if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                    crop = center_crop(img224, (side, side)).resize(REC_INPUT_SZ)
                else:
                    crop = img224.crop((x1, y1, x2, y2)).resize(REC_INPUT_SZ)

            xt = self.to_tensor(crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                fmap = extract_features(xt)
            feats.append(fmap.squeeze(0).cpu())
            imgs.append(self.to_tensor(img224))

        cap.release()
        return torch.stack(feats), torch.stack(imgs), os.path.basename(vp)

# =====================================================
# 5) 평가 함수 (threshold 고정)
# =====================================================
def evaluate_model(model, loader, device):
    model.eval()
    names, preds = [], []
    with torch.no_grad():
        for feats, frames, ids in tqdm(loader):
            feats, frames = feats.to(device), frames.to(device)
            out = model(feats, frames).sigmoid().view(-1).cpu().numpy()
            names.extend(ids)
            preds.extend((out > THRESHOLD).astype(int).tolist())
    return names, preds

# =====================================================
# 6) main: test_video_dir 만 외부에서 지정
# =====================================================
def main(test_video_dir):
    model = DeepFakeDetector().to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.join("models/dfd_model.pt"),
        map_location=DEVICE
    ))

    vids   = sorted(glob.glob(os.path.join(test_video_dir, "*.mp4")))
    ds     = ExternalVideoDataset(vids)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    names, preds = evaluate_model(model, loader, DEVICE)
    df = pd.DataFrame({"ID": names, "label": preds})
    csv_path = "result.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved results to {csv_path}")

# 여기에 비디오(.mp4) 경로 지정하시면 됩니다
TEST_VIDEO_DIR = "/kaggle/input/Deepfake_Detection_and_Generation_Challenge_Blue_Team/test_sample_videos"
main(TEST_VIDEO_DIR)


