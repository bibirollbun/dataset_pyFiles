# =========================================================
# Kaggle Submission — BYU Locating Bacterial Flagellar Motors 2025
# Inference-only, minimal logs, outputs /kaggle/working/submission.csv
# Uses MHAF-YOLO (YOLOv10) 2.5D (RGB from Z-context) + WBF + 3D-NMS
# =========================================================

# --- Basic setup ---
import os, sys, io, time, logging, warnings, random
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch

# ------------- Paths (Kaggle) -------------
DATA_ROOT  = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025"
TEST_DIR   = f"{DATA_ROOT}/test"
MODEL_PATH = "/kaggle/input/mahf-yolo-train/mayolov2f.pt"  # <-- your trained weights
SUBMISSION_PATH = "/kaggle/working/submission.csv"

# ------------- Inference params -------------
SIZE = 1024
CONFIDENCE_THRESHOLD = 0.8   # filter low-conf boxes after WBF
WBF_IOU   = 0.5
NMS3D_IOU = 0.2
Z_FOR_SUBMISSION = 10        # window size for 2.5D context (use neighbors on each side)

CONCENTRATION = 1.0          # 1.0 = use all slices; <1.0 = subsample for speed

# ------------- Device tweaks -------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if device.startswith("cuda"):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# --- Pull ultralytics fork used by MHAF-YOLO ---
!cp -r /kaggle/input/mhafyolo/pytorch/default/1/MHAF-YOLO-main /kaggle/working/
os.chdir("/kaggle/working/MHAF-YOLO-main")

from ultralytics import YOLOv10
from ultralytics.utils import LOGGER
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
LOGGER.setLevel(logging.ERROR)

# --- Quiet context for noisy libs ---
class Quiet:
    def __enter__(self):
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = self._buf_out = io.StringIO()
        sys.stderr = self._buf_err = io.StringIO()
        return self
    def __exit__(self, *args):
        sys.stdout, sys.stderr = self._stdout, self._stderr

# =============== Helpers ===============
def list_tomos(root_dir: str):
    return [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

def clamp(v, a, b):
    return max(a, min(b, v))

def normalize_slice(gray_uint8: np.ndarray) -> np.ndarray:
    if gray_uint8.dtype != np.uint8:
        gray_uint8 = gray_uint8.astype(np.uint8)
    p2, p98 = np.percentile(gray_uint8, 2), np.percentile(gray_uint8, 98)
    if p98 <= p2 + 1e-6:
        return gray_uint8.copy()
    clipped = np.clip(gray_uint8, p2, p98)
    return (255.0 * (clipped - p2) / (p98 - p2)).astype(np.uint8)

def read_gray_resized(path: str, size: int) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        im = np.array(Image.open(path).convert("L"))
    if (im.shape[1], im.shape[0]) != (size, size):
        im = cv2.resize(im, (size, size))
    return im

def build_25d_rgb(tomo_dir: str, slice_files: list, idx: int, size: int, window: int) -> np.ndarray:
    n = len(slice_files)
    g = normalize_slice(read_gray_resized(os.path.join(tomo_dir, slice_files[idx]), size))
    prev = [read_gray_resized(os.path.join(tomo_dir, slice_files[clamp(idx-d,0,n-1)]), size)
            for d in range(1, window+1)]
    nxt  = [read_gray_resized(os.path.join(tomo_dir, slice_files[clamp(idx+d,0,n-1)]), size)
            for d in range(1, window+1)]
    r = normalize_slice(np.mean(prev, axis=0).astype(np.uint8)) if prev else g.copy()
    b = normalize_slice(np.mean(nxt,  axis=0).astype(np.uint8)) if nxt  else g.copy()
    return np.stack([r, g, b], axis=2)

def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/ua if ua > 0 else 0.0

def weighted_box_fusion_simple(xyxy, conf, iou_thr=WBF_IOU):
    xyxy = np.asarray(xyxy, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)
    if xyxy.size == 0:
        return []
    if xyxy.ndim == 1:
        if xyxy.shape[0] != 4:
            return []
        xyxy = xyxy[None, :]
    if conf.ndim == 0:
        conf = np.array([float(conf)], dtype=np.float32)
    conf = conf.reshape(-1)
    if conf.shape[0] != xyxy.shape[0]:
        n = min(conf.shape[0], xyxy.shape[0])
        xyxy, conf = xyxy[:n], conf[:n]
        if n == 0:
            return []
    used = np.zeros(len(xyxy), dtype=bool)
    fused = []
    for i in range(len(xyxy)):
        if used[i]:
            continue
        group = [i]; used[i] = True
        for j in range(i+1, len(xyxy)):
            if used[j]:
                continue
            if iou_xyxy(xyxy[i], xyxy[j]) >= iou_thr:
                group.append(j); used[j] = True
        w = conf[group]; w = w / (w.sum() + 1e-9)
        bb = (xyxy[group] * w[:, None]).sum(axis=0)
        cf = float(conf[group].mean())
        fused.append((bb[0], bb[1], bb[2], bb[3], cf))
    return fused

def _safe_model_infer(model, image_rgb, img_size, device):
    if not isinstance(image_rgb, np.ndarray):
        image_rgb = np.array(image_rgb)
    if image_rgb.dtype != np.uint8:
        image_rgb = image_rgb.astype(np.uint8)
    image_rgb = np.ascontiguousarray(image_rgb)
    with Quiet():
        return model(image_rgb, imgsz=img_size, device=device, verbose=False)

def predict_tta_wbf_single(model, image_rgb: np.ndarray, img_size=SIZE, conf_thres=0.0, wbf_iou=WBF_IOU, device='cuda:0'):
    H, W = image_rgb.shape[:2]
    all_boxes, all_confs, all_clss = [], [], []
    # original
    res = _safe_model_infer(model, image_rgb, img_size, device)
    for r in res:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls  = r.boxes.cls.cpu().numpy().astype(int)
        all_boxes.append(xyxy); all_confs.append(conf); all_clss.append(cls)
    # hflip
    img_f = cv2.flip(image_rgb, 1)
    resf = __safe_model_infer(model, img_f, img_size, device)
    for r in resf:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls  = r.boxes.cls.cpu().numpy().astype(int)
        inv = xyxy.copy()
        inv[:, 0] = W - xyxy[:, 2]
        inv[:, 2] = W - xyxy[:, 0]
        all_boxes.append(inv); all_confs.append(conf); all_clss.append(cls)
    if len(all_boxes) == 0:
        return []
    boxes_cat = np.concatenate(all_boxes, axis=0)
    confs_cat = np.concatenate(all_confs, axis=0)
    clss_cat  = np.concatenate(all_clss, axis=0)
    keep = confs_cat >= conf_thres
    boxes_cat, confs_cat, clss_cat = boxes_cat[keep], confs_cat[keep], clss_cat[keep]
    fused_all = []
    for c in np.unique(clss_cat):
        idx = np.where(clss_cat == c)[0]
        fused = weighted_box_fusion_simple(boxes_cat[idx], confs_cat[idx], iou_thr=wbf_iou)
        for (x1,y1,x2,y2,cf) in fused:
            fused_all.append((x1,y1,x2,y2,cf,int(c)))
    return fused_all

def perform_3d_nms(detections, iou_threshold=NMS3D_IOU):
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d['confidence'], reverse=True)
    final = []
    box_size = 24
    dist_thr = box_size * iou_threshold
    def dist3(a,b):
        return np.sqrt((a['z']-b['z'])**2 + (a['y']-b['y'])**2 + (a['x']-b['x'])**2)
    while dets:
        best = dets.pop(0)
        final.append(best)
        dets = [d for d in dets if dist3(d, best) > dist_thr]
    return final

def process_tomogram(root_dir, tomo_id, model, size=SIZE, device=device, window_z=Z_FOR_SUBMISSION):
    t0 = time.time()
    tomo_dir = os.path.join(root_dir, tomo_id)
    slice_files_all = sorted([f for f in os.listdir(tomo_dir) if f.lower().endswith('.jpg')])
    if len(slice_files_all) == 0:
        elapsed = time.time() - t0
        return {'tomo_id': tomo_id, 'Motor axis 0': -1, 'Motor axis 1': -1, 'Motor axis 2': -1,
                '_status': 'NONE', '_slices': 0, '_elapsed': elapsed}

    h0, w0 = cv2.imread(os.path.join(tomo_dir, slice_files_all[0]), cv2.IMREAD_GRAYSCALE).shape[:2]
    scale_y, scale_x = h0 / float(size), w0 / float(size)

    if CONCENTRATION < 1.0:
        take = max(1, int(len(slice_files_all) * CONCENTRATION))
        sel = np.linspace(0, len(slice_files_all)-1, take)
        slice_idx = np.round(sel).astype(int).tolist()
    else:
        slice_idx = list(range(len(slice_files_all)))

    all_dets = []
    for i in slice_idx:
        rgb = build_25d_rgb(tomo_dir, slice_files_all, i, size=size, window=window_z)
        try: znum = int(Path(slice_files_all[i]).stem.split('_')[1])
        except Exception: znum = i
        fused = predict_tta_wbf_single(model, rgb, img_size=size, conf_thres=0.0, wbf_iou=WBF_IOU, device=device)
        for (x1,y1,x2,y2,cf,cl) in fused:
            if cf >= CONFIDENCE_THRESHOLD:
                xc = 0.5*(x1+x2); yc = 0.5*(y1+y2)
                all_dets.append({'z': int(round(znum)),
                                 'y': int(round(yc * scale_y)),
                                 'x': int(round(xc * scale_x)),
                                 'confidence': float(cf)})

    final_dets = perform_3d_nms(all_dets, NMS3D_IOU)
    status = "FOUND" if final_dets else "NONE"
    best = final_dets[0] if final_dets else {'z': -1, 'y': -1, 'x': -1}
    elapsed = time.time() - t0
    return {'tomo_id': tomo_id,
            'Motor axis 0': int(best['z']), 'Motor axis 1': int(best['y']), 'Motor axis 2': int(best['x']),
            '_status': status, '_slices': len(slice_idx), '_elapsed': elapsed}

# =============== Load model once ===============
with Quiet():
    model = YOLOv10(MODEL_PATH)
model.to(device)
try:
    model.model.float()
except Exception:
    pass
with Quiet():
    _ = model(np.zeros((SIZE,SIZE,3), np.uint8), imgsz=SIZE, device=device, verbose=False)

# =============== Submission generation ===============
def generate_submission(test_dir=TEST_DIR, model=model, submission_path=SUBMISSION_PATH, z_for_submission=Z_FOR_SUBMISSION):
    test_tomos = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    total_tomos = len(test_tomos)
    print(f"\n=== Inference on {total_tomos} test tomograms (Z={z_for_submission}) ===")

    results = []
    motors_found = 0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for i, tomo_id in enumerate(test_tomos, 1):
        out = process_tomogram(test_dir, tomo_id, model, size=SIZE, device=device, window_z=z_for_submission)
        results.append(out)

        has_motor = out['Motor axis 0'] != -1
        if has_motor:
            motors_found += 1

        # Minimal per-tomo line:
        print(f"[{i:03d}/{total_tomos}] {tomo_id}: {out['_status']} — slices={out['_slices']} — {out['_elapsed']:.1f}s")

    submission_df = pd.DataFrame(results)[['tomo_id', 'Motor axis 0', 'Motor axis 1', 'Motor axis 2']]
    submission_df.to_csv(submission_path, index=False)

    print(f"\nDone. Motors detected in {motors_found}/{total_tomos} tomos "
          f"({(motors_found/total_tomos*100 if total_tomos else 0):.1f}%).")
    print(f"Saved submission to: {submission_path}")
    print("\nPreview:")
    print(submission_df.head())

    return submission_df

# ---- Run ----
submission_df = generate_submission()


