# 추론할 디렉토리 주소 설정
video_dir = '/kaggle/input/sub-test-videos/test_sample_videos'


!pip install --no-index --find-links=/kaggle/input/wheels2/wheels/wheels torch==2.0.1 torchvision==0.15.2 timm==0.6.5
!pip install --no-index --find-links=/kaggle/input/wheels2/wheels/wheels facenet-pytorch==2.5.2
!pip install --no-index --find-links=/kaggle/input/wheels2/wheels/wheels scipy scikit-learn numpy Pillow opencv-python albumentations==1.3.0
!pip install --no-index --find-links=/kaggle/input/wheels2/wheels/wheels decord==0.6.0



!pip install /kaggle/input/wheels2/face_recognition_models-0.3.0-py2.py3-none-any.whl


!pip install /kaggle/input/wheels2/face_recognition-1.4.0-py2.py3-none-any.whl


import os
import subprocess
import pandas as pd
from tqdm import tqdm
import numpy as np
import logging
import re

# ------------------- 설정 -------------------
GENCONVIT_PROJECT_ROOT = '/kaggle/input/genconvit8/GenConViT'
GENCONVIT_PREDICTION_SCRIPT = os.path.join(GENCONVIT_PROJECT_ROOT, 'prediction.py')
VIDEOS_TO_PROCESS_DIR = video_dir  #비디오 디렉토리 주소
GENCONVIT_DATASET = 'other'
GENCONVIT_FRAMES = '10'
OUTPUT_CSV_FILE = '/kaggle/working/genconvit_predictions.csv'

GENCONVIT_OUTPUT_REGEX = r"Prediction: (\d+\.?\d*)"

# ------------------- 로깅 -------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------- 유틸리티 함수 -------------------
def get_video_list(video_dir):
    return [
        (os.path.join(video_dir, fname), fname)
        for fname in sorted(os.listdir(video_dir))
        if fname.lower().endswith(('.mp4', '.mov', '.avi')) and os.path.isfile(os.path.join(video_dir, fname))
    ]

def run_genconvit_prediction():
    command = [
        'python', GENCONVIT_PREDICTION_SCRIPT,
        '--p', VIDEOS_TO_PROCESS_DIR,
        '--d', GENCONVIT_DATASET,
        '--f', GENCONVIT_FRAMES
    ]
    logging.info(f"GenConViT 실행: {' '.join(command)}")
    subprocess.run(command, cwd=GENCONVIT_PROJECT_ROOT,  check=True)
# ------------------- 실행 -------------------


run_genconvit_prediction()


import json

with open("GenConViT_result.json", "r") as f:
    prediction_dict = json.load(f)

# 이름과 확률 리스트 추출
video_names = prediction_dict["video"]["name"]
pred_probs = prediction_dict["video"]["pred"]

# 결과 딕셔너리 생성
GenConViT_results = {name: prob for name, prob in zip(video_names, pred_probs)}


GenConViT_results


import argparse, torch, numpy as np, cv2, csv, random, json
from pathlib import Path
from tqdm import tqdm
from facenet_pytorch import MTCNN
from PIL import Image
import timm
import torch.nn as nn
from safetensors.torch import load_file

NUM_FRAMES = 16
IMG_SIZE = 224

# ----- 모델 구조 -----
class CaFftDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0, global_pool="")
        self.vit.load_state_dict(torch.load("/kaggle/input/weights/vit_base_patch16_224.pth"))
        feat_dim = self.vit.num_features
        self.kv_proj = nn.Linear(1, 256, bias=False)
        self.ca = nn.MultiheadAttention(feat_dim, 8, kdim=256, vdim=256, batch_first=True)
        self.hist_fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(feat_dim + 128, 512), nn.ReLU(), nn.Linear(512, 1))

    @torch.no_grad()
    def _rgb_tok(self, imgs):
        return self.vit.forward_features(imgs)

    def forward(self, imgs, fft_mat, hist):
        rgb = self._rgb_tok(imgs)
        kv = torch.nn.functional.avg_pool2d(fft_mat.unsqueeze(1), 4)
        kv = kv.flatten(2).transpose(1, 2)
        kv = self.kv_proj(kv)
        fused, _ = self.ca(rgb, kv, kv)
        cls = fused[:, 0]
        hist = self.hist_fc(hist)
        return self.head(torch.cat([cls, hist], dim=-1)).squeeze(-1)

# ----- 얼굴 영역 정사각형 크롭 -----
def square_crop(img, bbox, margin=0.2):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + margin)
    x1, y1 = int(cx - side / 2), int(cy - side / 2)
    x2, y2 = int(cx + side / 2), int(cy + side / 2)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return img[y1:y2, x1:x2]

# ----- FFT + 히스토그램 계산 -----
def calc_fft_hist(gray):
    gray = cv2.resize(gray, (64, 64), cv2.INTER_AREA).astype(np.float32)
    gray = (gray - gray.mean()) / (gray.std() + 1e-6)
    power = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) ** 2 + 1e-8)
    yy, xx = np.mgrid[:64, :64] - 32
    r = np.round(np.sqrt(xx ** 2 + yy ** 2)).astype(np.int32)
    hist = np.bincount(r.ravel(), power.ravel(), minlength=256).astype(np.float32)
    hist /= (hist.max() + 1e-6)
    return power, hist

# ----- 프레임에서 얼굴 추출 (20번 랜덤시도 후 복제) -----
def get_face_stack_from_video(vpath, mtcnn, size=224, margin=0.2, n_frames=16, max_trials=20):
    cap = cv2.VideoCapture(str(vpath))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None

    all_idx = list(range(total))
    random.shuffle(all_idx)
    tried = 0
    faces = []

    while tried < max_trials and tried < len(all_idx):
        idx = all_idx[tried]
        tried += 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb)
        if boxes is not None and len(boxes) > 0:
            crop = square_crop(rgb, boxes[0], margin)
            if crop is not None:
                face = cv2.resize(crop.astype(np.uint8), (size, size), cv2.INTER_AREA)
                faces = [face] * n_frames
                break

    cap.release()
    return faces if faces else None

# ----- 메인 추론 파이프라인 -----
def infer_videos_in_directory(video_dir, model_path, device="cuda"):
    video_dir = Path(video_dir)
    exts = {'.mp4', '.avi', '.mov', '.mkv'}
    video_files = [p for p in video_dir.glob('**/*') if p.suffix.lower() in exts]
    print(f"Found {len(video_files)} video files.")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    mtcnn = MTCNN(keep_all=False, device=device, thresholds=[0.7, 0.8, 0.8])
    model = CaFftDetector().to(device).eval()
    model.load_state_dict(torch.load(model_path, map_location=device))
    to_tensor = lambda x: torch.from_numpy(x).permute(2, 0, 1).float() / 255.
    results = []

    for vpath in tqdm(video_files, desc="Infer"):
        fname = vpath.name
        faces = get_face_stack_from_video(vpath, mtcnn, size=IMG_SIZE, margin=0.2, n_frames=NUM_FRAMES)
        if faces is None:
            print(f"Skip {fname} (no face found)")
            results.append((fname, 0.5))
            continue

        imgs, ffts, hists = [], [], []
        for face in faces:
            img_t = to_tensor(face)
            imgs.append(img_t)
            gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            fft, hist = calc_fft_hist(gray)
            ffts.append(torch.from_numpy(fft).float())
            hists.append(torch.from_numpy(hist).float())

        imgs  = torch.stack(imgs).to(device)
        ffts  = torch.stack(ffts).to(device)
        hists = torch.stack(hists).to(device)

        with torch.no_grad():
            logits = model(imgs, ffts, hists)
            probs = torch.sigmoid(logits).cpu().numpy()
            fake_prob = float(np.mean(probs))

        results.append((fname, fake_prob))

    return results



CaFft_results = infer_videos_in_directory(
    video_dir,
    "/kaggle/input/weights/ff_dfd_test.pth"
)



CaFft_results2 = {video_id: prob for video_id, prob in CaFft_results}


CaFft_results2


import os
import subprocess
import re
import pandas as pd
from tqdm import tqdm
import numpy as np

# ------------------- 설정 -------------------
FREQNET_PROJECT_ROOT = '/kaggle/input/freqnet-deepfakedetection/FreqNet-DeepfakeDetection-main'
FREQNET_PREDICTION_SCRIPT = '/kaggle/input/test-single-video/test_single_video.py'
FREQNET_MODEL_WEIGHTS = os.path.join(FREQNET_PROJECT_ROOT, '4-classes-freqnet-v2.pth')

VIDEOS_TO_PROCESS_DIR = video_dir
OUTPUT_CSV_FILE = '/kaggle/working/FreqNet_result.csv'

# ------------------- 유틸리티 함수 -------------------
def get_video_files_from_dir(video_dir):
    video_file_paths = []
    if os.path.isdir(video_dir):
        for filename in sorted(os.listdir(video_dir)):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(video_dir, filename)
                video_file_paths.append((video_path, filename))
    else:
        print(f"Error: 비디오 디렉토리 '{video_dir}'를 찾을 수 없습니다.")

    if not video_file_paths:
        print(f"Error: {video_dir} 경로에서 비디오 파일을 찾을 수 없습니다.")
    return video_file_paths


def run_freqnet_prediction(script_path, model_weights_path, video_file_path, script_root_dir):
    command = [
        'python', script_path,
        '--model_path', model_weights_path,
        '--video_path', video_file_path
    ]
    env = os.environ.copy()
    env['PYTHONPATH'] = FREQNET_PROJECT_ROOT

    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True, cwd=script_root_dir, env=env)
        stdout = process.stdout
        match = re.search(r"Predicted score \(closer to 1 = fake\): (\d+\.?\d*)", stdout)
        if match:
            return float(match.group(1))
        else:
            print(f"Warning: 점수 파싱 실패 → {video_file_path}")
            print(f"FreqNet Output:\n{stdout.strip()}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error: FreqNet 실행 실패 → {video_file_path}")
        print(f"stderr:\n{e.stderr.strip()}")
        return None
    except Exception as e:
        print(f"Unexpected error → {video_file_path}: {e}")
        return None

# ------------------- 실행 -------------------
print("--- FreqNet 예측 시작 (metadata.json 사용 안 함) ---")

# 경로 유효성 체크
if not os.path.isdir(FREQNET_PROJECT_ROOT) or \
   not os.path.isfile(FREQNET_PREDICTION_SCRIPT) or \
   not os.path.isfile(FREQNET_MODEL_WEIGHTS):
    print("Error: 경로 설정 오류 - 프로젝트, 스크립트, 모델 가중치 중 하나가 존재하지 않음")
    exit()

# 비디오 파일 목록 불러오기
video_files = get_video_files_from_dir(VIDEOS_TO_PROCESS_DIR)
if not video_files:
    print("처리할 비디오 없음. 종료.")
    exit()

print(f"{len(video_files)}개의 비디오 예측 시작")

# 예측 수행
prediction_results = []
for video_path, video_filename in tqdm(video_files, desc="Predicting with FreqNet"):
    fake_prob = run_freqnet_prediction(
        FREQNET_PREDICTION_SCRIPT,
        FREQNET_MODEL_WEIGHTS,
        video_path,
        FREQNET_PROJECT_ROOT
    )
    prediction_results.append({
        'video_id': video_filename,
        'prob': fake_prob if fake_prob is not None else 0.5
    })

FreqNet_results = {entry['video_id']: entry['prob'] for entry in prediction_results}



FreqNet_results


import os
import pandas as pd

video_ids = [ f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) ]

weights = {
    "GenConViT": 0.971,
    "FreqNet": 0.0163,
    "CaFft": 0.127
}

submission = []
for vid in video_ids:
    genconvit_prob = GenConViT_results.get(vid, 0.5)
    freqnet_prob   = FreqNet_results.get(vid, 0.5)
    cafft_prob     = CaFft_results2.get(vid, 0.5)

    final_prob = (
        weights["GenConViT"] * genconvit_prob +
        weights["FreqNet"]   * freqnet_prob   +
        weights["CaFft"]     * cafft_prob
    )

    final_label = int(final_prob >= 0.5)

    submission.append({
        "ID": vid,
        "label": final_label
    })

df = pd.DataFrame(submission)
df.to_csv("submission.csv", index=False)
print("✅ submission.csv 생성 완료")



print(df)

