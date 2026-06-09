# !rm -rf /kaggle/working/dataset
# !rm -rf /kaggle/working/dataset_json


import os
import json
import cv2
import numpy as np
import shutil

from typing import List
from PIL import Image


src_path = r"/kaggle/input/challenge-datasets"
dst_path = r"/kaggle/working/dataset"

shutil.copytree(src_path, dst_path)
print('Copied')

os.mkdir("/kaggle/working/dataset_json")


!pip install /kaggle/input/insightface/pytorch/default/1/insightface-0.7.3-cp311-cp311-linux_x86_64.whl
!pip install --no-index --find-links=/kaggle/input/onnxruntime-gpu/pytorch/default/1 onnxruntime-gpu==1.22.0


from insightface.app import FaceAnalysis

src_model_dir = '/kaggle/input/challenge/pytorch/default/1/preprocessing'
app = FaceAnalysis(name='antelopev2', root=src_model_dir, providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))


def get_all_video_paths(videos_dir, extensions=('.mp4', '.avi', '.mov', '.mkv')):
    video_paths = []
    
    for root, _, files in os.walk(videos_dir):
        for file in files:
            if file.lower().endswith(extensions):
                video_path = os.path.join(root, file)
                video_paths.append(video_path)
    
    return video_paths

    
def extract_frames(video_path: str) -> List[np.ndarray]:
    frames = []
    reader = cv2.VideoCapture(video_path)
    
    while reader.isOpened():
        success, frame = reader.read()
        if not success:
            break
        frames.append(frame)
    
    reader.release()

    return frames


def expand_bbox_to_square(bbox, img_shape, ratio=0.2):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    expand_w = int(w * ratio)
    expand_h = int(h * ratio)

    new_x1 = max(0, x1 - expand_w)
    new_y1 = max(0, y1 - expand_h)
    new_x2 = min(img_shape[1], x2 + expand_w)
    new_y2 = min(img_shape[0], y2 + expand_h)

    box_w = new_x2 - new_x1
    box_h = new_y2 - new_y1
    side = max(box_w, box_h)

    cx = (new_x1 + new_x2) // 2
    cy = (new_y1 + new_y2) // 2
    half_side = side // 2

    square_x1 = max(0, cx - half_side)
    square_y1 = max(0, cy - half_side)
    square_x2 = min(img_shape[1], square_x1 + side)
    square_y2 = min(img_shape[0], square_y1 + side)

    square_x1 = max(0, square_x2 - side)
    square_y1 = max(0, square_y2 - side)

    return [square_x1, square_y1, square_x2, square_y2]


def extract_faces_from_one_video(video_path, ratio=0.2, max_frames=32):
    frames = extract_frames(video_path)
    
    face_cropped_frames = []
    for frame in frames:
        faces = app.get(frame)

        if len(faces) == 0:
            continue

        else:
            face = faces[0]
            bbox = face.bbox.astype(int)
            expanded_bbox = expand_bbox_to_square(bbox, frame.shape, ratio=ratio)
            x1, y1, x2, y2 = map(int, expanded_bbox)
            face_crop = frame[y1:y2, x1:x2]

            # Convert to RGB
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            face_cropped_frames.append(face_crop)
            
        if len(face_cropped_frames) >= max_frames:
            break
    
    return face_cropped_frames


def save_cropped_faces_from_videos(videos_dir, output_dir='frames', ratio=0.2, max_frames=32, min_sequence_length=8):
    video_paths = get_all_video_paths(videos_dir)
    frames_root = os.path.join(videos_dir, output_dir)
    os.makedirs(frames_root, exist_ok=True)

    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        label = os.path.basename(os.path.dirname(video_path))

        save_dir = os.path.join(frames_root, label, video_name)
        os.makedirs(save_dir, exist_ok=True)

        face_frames = extract_faces_from_one_video(video_path, ratio=ratio, max_frames=max_frames)

        if len(face_frames) < min_sequence_length:
            print(f"Warning: {video_name} has only {len(face_frames)} face frames, minimum required is {min_sequence_length}")

        for i in range(0, len(face_frames) - max_frames + 1, max_frames):
            frame_chunk = face_frames[i:i+max_frames]

            for frame_idx, face_frame in enumerate(frame_chunk):
                if isinstance(face_frame, np.ndarray):
                    face_frame = Image.fromarray(face_frame)
                
                face_frame = face_frame.resize((224, 224))
                save_path = os.path.join(save_dir, f"{frame_idx+1:04d}.png")
                face_frame.save(save_path)


save_cropped_faces_from_videos('/kaggle/working/dataset/red_team_1')
save_cropped_faces_from_videos('/kaggle/working/dataset/red_team_2')
save_cropped_faces_from_videos('/kaggle/working/dataset/red_team_3')


def generate_dataset_file(dataset_name, dataset_root_path, output_file_path):
    dataset_dict = {}
    print(dataset_name)

    dataset_path = os.path.join(dataset_root_path, dataset_name)
    dataset_dict[dataset_name] = {'test': {}}
        
    frames_path = os.path.join(dataset_path, 'frames')
    for label_name, label_value in [('real', 0), ('fake', 1)]:
        label_dir = os.path.join(frames_path, label_name)
        if not os.path.exists(label_dir):
            continue
        for video_dir in os.scandir(label_dir):
            if video_dir.is_dir():
                video_name = video_dir.name
                frame_paths = sorted([os.path.join(video_dir.path, frame.name) 
                                    for frame in os.scandir(video_dir.path) 
                                    if frame.name.endswith('.png')])
                
                frame_paths = [os.path.join(dataset_name, 'frames', label_name, video_name, os.path.basename(path)) 
                                for path in frame_paths]
                                
                frame_paths = [path.replace('/', '\\') for path in frame_paths]
                
                dataset_dict[dataset_name]['test'][video_name] = {
                    'frames': frame_paths,
                    'label': label_value
                }

    output_file_path = os.path.join(output_file_path, dataset_name + '.json')
    with open(output_file_path, 'w') as f:
        json.dump(dataset_dict, f)
    print(f"{dataset_name}.json generated successfully.")


dataset_root_path = '/kaggle/working/dataset'
output_file_path = '/kaggle/working/dataset_json'


generate_dataset_file('red_team_1', dataset_root_path, output_file_path)
generate_dataset_file('red_team_2', dataset_root_path, output_file_path)
generate_dataset_file('red_team_3', dataset_root_path, output_file_path)


import random
import sys
import time
import cv2
import dlib
import yaml
import re
import logging
import datetime
import glob
import concurrent.futures
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from imutils import face_utils
from skimage import transform as trans


sys.path.append("/kaggle/input/challenge/pytorch/default/1")


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from training.dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from training.detectors import DETECTOR
from training.metrics.utils import get_video_level_predictions


# our code
default_path = "/kaggle/input/challenge/pytorch/default/1/training"

# vision tower weight
CLIP_path = "/kaggle/input/clipvit-l14/pytorch/default/1"

# our model weight
DETECTOR_YAML = default_path + "/config/detector/d_brain.yaml"
WEIGHTS_PATH  = "/kaggle/input/frame_acc_ckpts/pytorch/default/1/ckpt_best.pth"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

on_2060 = "2060" in (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")


def init_seed(cfg):
    if cfg.get("manualSeed") is None:
        cfg["manualSeed"] = random.randint(1, 10000)
    random.seed(cfg["manualSeed"])
    torch.manual_seed(cfg["manualSeed"])
    if cfg.get("cuda", True) and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["manualSeed"])

def prepare_testing_data(cfg):
    def _make_loader(name):
        cfg_local = cfg.copy()
        cfg_local["test_dataset"] = name
        ds = DeepfakeAbstractBaseDataset(cfg_local, mode="test")
        return torch.utils.data.DataLoader(
            ds,
            batch_size=cfg_local["test_batchSize"],
            shuffle=False,
            num_workers=int(cfg_local["workers"]),
            collate_fn=ds.collate_fn,
            drop_last=False,
        )
    return {name: _make_loader(name) for name in cfg["test_dataset"]}

def inference(model, data_dict):
    with torch.no_grad():
        return model(data_dict, inference=True)

def test_one_dataset(model, loader):
    preds, labels, feats = [], [], []
    for batch in tqdm(loader, leave=False):
        data, lbl = batch["image"].to(device), batch["label"].to(device)
        batch["image"], batch["label"] = data, torch.where(lbl != 0, 1, 0)
        if batch.get("mask") is not None:
            batch["mask"] = batch["mask"].to(device)
        if batch.get("landmark") is not None:
            batch["landmark"] = batch["landmark"].to(device)
        out = inference(model, batch)
        preds.extend(out["prob"].cpu().numpy())
        feats.extend(out["feat"].cpu().numpy())
        labels.extend(batch["label"].cpu().numpy())
    return np.asarray(preds), np.asarray(labels), np.asarray(feats)

def test_epoch(model, loaders, run_tag="run"):
    model.eval()
    all_metrics = {}
    for name, loader in loaders.items():
        print(f"\n=== Evaluating {name} ===")
        preds, labels, _ = test_one_dataset(model, loader)
        metrics = get_video_level_predictions(preds, loader.dataset.data_dict["image"], threshold=0.5)
        all_metrics[name] = metrics
    return all_metrics


video_folder_name = ['red_team_1', 'red_team_2', 'red_team_3']


# Load YAML configs
with open(DETECTOR_YAML) as f:
    cfg = yaml.safe_load(f)
with open(default_path + "/config/test_config.yaml") as f:
    cfg_test = yaml.safe_load(f)
cfg.update(cfg_test)


# Manual overrides from notebook variables
cfg["test_dataset"]  = video_folder_name
cfg["weights_path"]  = WEIGHTS_PATH
cfg["workers"]       = 0 if on_2060 else 8
cfg["cuda"]          = torch.cuda.is_available()

cfg['CLIP_path'] = CLIP_path
cfg['dataset_json_folder'] = '/kaggle/working/dataset_json'
cfg['base_path'] = '/kaggle/working/dataset/'


# Init seed & cudnn
init_seed(cfg)
if cfg.get("cudnn", True):
    cudnn.benchmark = True


# Dataloaders
loaders = prepare_testing_data(cfg)
print("Loaded test datasets:", list(loaders.keys()))


# Model
model_cls = DETECTOR[cfg["model_name"]]
model     = model_cls(cfg).to(device)
print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))


# Checkpoint
ckpt_path = cfg["weights_path"]
if ckpt_path and os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    # model.load_state_dict(ckpt, strict=True)
    model.load_state_dict(ckpt, strict=False)
    print("Checkpoint loaded.")
else:
    print("No checkpoint found, using random weights.")


run_tag = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(ckpt_path)))) if ckpt_path else "no_ckpt"
metrics = test_epoch(model, loaders, run_tag=run_tag)


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


video_folder_name[0]


label_df = pd.read_csv('/kaggle/working/dataset/red_team_1/red_team_1.csv')

pred_df = pd.DataFrame(metrics[video_folder_name[0]])
pred_df = pred_df.rename(columns={
    'video_id': 'ID',
    'pred_class': 'label'
})

# 확장자 없는 파일명 생성
label_df['video_key'] = label_df['ID'].apply(lambda x: os.path.splitext(x)[0])
pred_df['video_key'] = pred_df['ID'].apply(lambda x: os.path.splitext(x)[0])

# video_key 기준으로 merge
merged_df = pd.merge(label_df, pred_df[['video_key', 'label']], on='video_key', how='left')

# 예측이 없는 경우 기본값 설정
merged_df['label'] = merged_df['label'].fillna(1).astype(int)

# 평가용 컬럼 추출
y_pred = merged_df['label']
y_true = merged_df['true_label']

# Metric 계산
f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# 최종 제출 파일: label_df 의 ID 확장자 그대로 사용
final_df = merged_df[['ID', 'label']]
final_df.to_csv("submission_1.csv", index=False, encoding='utf-8')


video_folder_name[1]


label_df = pd.read_csv('/kaggle/working/dataset/red_team_2/red_team_2.csv')

pred_df = pd.DataFrame(metrics[video_folder_name[1]])
pred_df = pred_df.rename(columns={
    'video_id': 'ID',
    'pred_class': 'label'
})

# 확장자 없는 파일명 생성
label_df['video_key'] = label_df['ID'].apply(lambda x: os.path.splitext(x)[0])
pred_df['video_key'] = pred_df['ID'].apply(lambda x: os.path.splitext(x)[0])

# video_key 기준으로 merge
merged_df = pd.merge(label_df, pred_df[['video_key', 'label']], on='video_key', how='left')

# 예측이 없는 경우 기본값 설정
merged_df['label'] = merged_df['label'].fillna(1).astype(int)

# 평가용 컬럼 추출
y_pred = merged_df['label']
y_true = merged_df['true_label']

# Metric 계산
f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# 최종 제출 파일: label_df 의 ID 확장자 그대로 사용
final_df = merged_df[['ID', 'label']]
final_df.to_csv("submission_2.csv", index=False, encoding='utf-8')


video_folder_name[2]


label_df = pd.read_csv('/kaggle/working/dataset/red_team_3/red_team_3.csv')

pred_df = pd.DataFrame(metrics[video_folder_name[2]])
pred_df = pred_df.rename(columns={
    'video_id': 'ID',
    'pred_class': 'label'
})

# 확장자 없는 파일명 생성
label_df['video_key'] = label_df['ID'].apply(lambda x: os.path.splitext(x)[0])
pred_df['video_key'] = pred_df['ID'].apply(lambda x: os.path.splitext(x)[0])

# video_key 기준으로 merge
merged_df = pd.merge(label_df, pred_df[['video_key', 'label']], on='video_key', how='left')

# 예측이 없는 경우 기본값 설정
merged_df['label'] = merged_df['label'].fillna(1).astype(int)

# 평가용 컬럼 추출
y_pred = merged_df['label']
y_true = merged_df['true_label']

# Metric 계산
f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# 최종 제출 파일: label_df 의 ID 확장자 그대로 사용
final_df = merged_df[['ID', 'label']]
final_df.to_csv("submission_3.csv", index=False, encoding='utf-8')


# !rm -rf /kaggle/working/preprocessing_logs
# !rm -rf /kaggle/working/dataset_json
# !rm -rf /kaggle/working/datasets

