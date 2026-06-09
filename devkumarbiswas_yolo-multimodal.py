"""# submission_kaggle_safe.py
import os
import shutil
import gc
from pathlib import Path

import pydicom
import cv2
import numpy as np
import pandas as pd
import torch
import kaggle_evaluation.rsna_inference_server

# ===================== CONFIG =====================
ID_COL = 'SeriesInstanceUID'
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

LOCATION_MAP = {
    1: "Other Posterior Circulation",
    2: "Basilar Tip",
    3: "Right Posterior Communicating Artery",
    4: "Left Posterior Communicating Artery",
    5: "Right Infraclinoid Internal Carotid Artery",
    6: "Left Infraclinoid Internal Carotid Artery",
    7: "Right Supraclinoid Internal Carotid Artery",
    8: "Left Supraclinoid Internal Carotid Artery",
    9: "Right Middle Cerebral Artery",
    10: "Left Middle Cerebral Artery",
    11: "Right Anterior Cerebral Artery",
    12: "Left Anterior Cerebral Artery",
    13: "Anterior Communicating Artery",
}
ID_TO_LABEL = {i - 1: v for i, v in LOCATION_MAP.items()}

# ===================== HELPERS =====================
def adaptive_windowing(image):
    img_flat = image.flatten()
    img_flat = img_flat[img_flat > 0]
    if len(img_flat) == 0:
        return np.zeros_like(image, dtype=np.uint8)
    low_val, high_val = np.percentile(img_flat, [5, 95])
    img_windowed = np.clip(image, low_val, high_val)
    img_norm = ((img_windowed - low_val) / (high_val - low_val + 1e-8) * 255).astype(np.uint8)
    return img_norm

def process_dicom_series(series_path, image_size=512, num_slices=16):
    #Load DICOM series safely and reduce slices for memory efficiency.
    dcm_files = sorted([os.path.join(root, f)
                        for root, _, files in os.walk(series_path)
                        for f in files if f.endswith(".dcm")])
    if len(dcm_files) == 0:
        return np.zeros((num_slices, image_size, image_size), dtype=np.uint8)

    dicom_data = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f, force=True)
            img = ds.pixel_array.astype(np.float32)
            if img.ndim == 3 and img.shape[-1] == 3:
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            img_resized = cv2.resize(img, (image_size, image_size))
            dicom_data.append(img_resized)
        except Exception as e:
            print(f"Skipping file {f} due to error: {e}")

    if len(dicom_data) == 0:
        return np.zeros((num_slices, image_size, image_size), dtype=np.uint8)

    volume = np.stack(dicom_data, axis=0)
    volume = adaptive_windowing(volume)

    if volume.shape[0] > num_slices:
        idx = np.linspace(0, volume.shape[0] - 1, num_slices).astype(int)
        volume = volume[idx]
    elif volume.shape[0] < num_slices:
        pad = num_slices - volume.shape[0]
        volume = np.pad(volume, ((0, pad), (0, 0), (0, 0)), mode="edge")

    return volume

def create_multichannel_img(volume):
    depth, h, w = volume.shape
    start, end = int(depth * 0.15), int(depth * 0.85)
    mip = np.max(volume[start:end], axis=0)

    slice_means = np.mean(volume, axis=(1, 2))
    top_percentile = np.percentile(slice_means, 75)
    high_intensity = slice_means >= top_percentile
    weighted_avg = np.mean(volume[high_intensity], axis=0) if np.any(high_intensity) else np.mean(volume, axis=0)

    std_proj = np.zeros_like(volume[0])
    for i in range(depth - 4):
        window_std = np.std(volume[i:i+5], axis=0)
        std_proj = np.maximum(std_proj, window_std)

    chans = []
    for ch in [mip, weighted_avg, std_proj]:
        # Safe normalization to prevent NaNs
        if ch.max() == ch.min():
            ch_norm = np.zeros_like(ch, dtype=np.uint8)
        else:
            ch_norm = ((ch - ch.min()) / (ch.max() - ch.min()) * 255).astype(np.uint8)
        chans.append(ch_norm)
    return np.stack(chans, axis=-1)

def preprocess_image(img, target_size=640):
    img_resized = cv2.resize(img, (target_size, target_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(img_rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return torch.from_numpy(tensor).unsqueeze(0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tta_transforms(image):
    transforms = [image]
    transforms.append(cv2.flip(image, 1))
    transforms.append(cv2.flip(image, 0))
    transforms.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    return transforms

# ===================== MODEL =====================
model_path = "/kaggle/input/yolo8_multimodal_modal/pytorch/default/1/best.torchscript"
assert os.path.exists(model_path), f"Model not found at {model_path}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load(model_path, map_location=device)
model.eval()

# ===================== PREDICTION =====================
def predict(series_path: str) -> pd.DataFrame:
    try:
        series_id = os.path.basename(series_path)

        volume = process_dicom_series(series_path)
        base_image = create_multichannel_img(volume)

        scores_list = []
        for aug_img in tta_transforms(base_image):
            try:
                img_tensor = preprocess_image(aug_img).to(device)
                with torch.no_grad():
                    outputs = model(img_tensor)
                    if outputs is None or len(outputs) == 0:
                        continue
                    # Handle different output types safely
                    if isinstance(outputs, torch.Tensor):
                        outputs = outputs.cpu().numpy()
                    elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                        outputs = [o.cpu().numpy() if isinstance(o, torch.Tensor) else o for o in outputs]
                    else:
                        continue
            except Exception as e:
                print(f"Skipping TTA image due to error: {e}")
                continue

            scores = {label: 0.0 for label in LABEL_COLS}
            aneurysm_present = 0.0

            for det in outputs[0]:
                try:
                    conf_prob = sigmoid(float(det[4]))
                    cls_id = int(det[5])
                    label = ID_TO_LABEL.get(cls_id, None)
                    if label is not None:
                        scores[label] = max(scores[label], conf_prob)
                        aneurysm_present = max(aneurysm_present, conf_prob)
                except Exception:
                    continue

            scores["Aneurysm Present"] = max(aneurysm_present, max(scores.values()))
            scores_list.append(scores)

        if len(scores_list) == 0:
            final_scores = {label: 0.0 for label in LABEL_COLS}
        else:
            final_scores = {label: np.mean([s[label] for s in scores_list]) for label in LABEL_COLS}

        # Clean up memory
        del volume, base_image, scores_list, img_tensor
        gc.collect()

        row = [series_id] + [float(final_scores[c]) for c in LABEL_COLS]
        df = pd.DataFrame([row], columns=[ID_COL, *LABEL_COLS])
        return df

    except Exception as e:
        print(f"Failed series {series_path} with error: {e}")
        # Return zero predictions for failed series
        row = [os.path.basename(series_path)] + [0.0]*len(LABEL_COLS)
        return pd.DataFrame([row], columns=[ID_COL, *LABEL_COLS])

# ===================== SERVER / LOCAL =====================
shared_dir = "/kaggle/shared"
shutil.rmtree(shared_dir, ignore_errors=True)
os.makedirs(shared_dir, exist_ok=True)

inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    # Generate submission CSV
    submission_path = "/kaggle/working/submission.parquet"
    if os.path.exists(submission_path):
        df = pd.read_parquet(submission_path)
        df.to_csv("/kaggle/working/submission.csv", index=False)
        display(df)"""


"""# submission.py
import os
import shutil
import pydicom
import cv2
import numpy as np
import torch
import pandas as pd
import kaggle_evaluation.rsna_inference_server

# ===================== CONFIG =====================
ID_COL = 'SeriesInstanceUID'
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

LOCATION_MAP = {
    1: "Other Posterior Circulation",
    2: "Basilar Tip",
    3: "Right Posterior Communicating Artery",
    4: "Left Posterior Communicating Artery",
    5: "Right Infraclinoid Internal Carotid Artery",
    6: "Left Infraclinoid Internal Carotid Artery",
    7: "Right Supraclinoid Internal Carotid Artery",
    8: "Left Supraclinoid Internal Carotid Artery",
    9: "Right Middle Cerebral Artery",
    10: "Left Middle Cerebral Artery",
    11: "Right Anterior Cerebral Artery",
    12: "Left Anterior Cerebral Artery",
    13: "Anterior Communicating Artery",
}
ID_TO_LABEL = {i - 1: v for i, v in LOCATION_MAP.items()}

# ===================== HELPERS =====================
def normalize_dicom(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)
    return img

def preprocess_image(img, target_size=640):
    img_resized = cv2.resize(img, (target_size, target_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    img_tensor = np.transpose(img_rgb, (2, 0, 1)).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0)
    return img_tensor

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ===================== LOAD MODEL =====================
model_path = "/kaggle/input/yolo8_multimodal_modal/pytorch/default/1/best.torchscript"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load(model_path, map_location=device)
model.eval()

# ===================== PREDICT =====================
def predict(series_path: str) -> pd.DataFrame:
    series_id = os.path.basename(series_path)

    # Gather DICOM files
    dcm_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(series_path)
        for file in files if file.endswith(".dcm")
    ]
    dcm_files.sort()

    # Initialize scores
    scores = {label: 0.0 for label in LABEL_COLS}
    aneurysm_present = 0.0

    for filepath in dcm_files:
        ds = pydicom.dcmread(filepath, force=True)
        img = normalize_dicom(ds.pixel_array)
        img_tensor = preprocess_image(img).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)

        if outputs is not None:
            outputs = outputs.cpu().numpy()
            for det in outputs[0]:
                conf_prob = sigmoid(float(det[4]))
                cls_id = int(det[5])
                label = ID_TO_LABEL.get(cls_id, None)
                if label is not None:
                    scores[label] = max(scores[label], conf_prob)
                    aneurysm_present = max(aneurysm_present, conf_prob)

    # Consistency: aneurysm present = max of all detections
    scores['Aneurysm Present'] = max(aneurysm_present, max(scores.values()))

    # Return pandas DataFrame
    row = [series_id] + [float(scores[c]) for c in LABEL_COLS]
    df = pd.DataFrame([row], columns=[ID_COL, *LABEL_COLS])
    return df

# ===================== SERVER =====================
shared_dir = "/kaggle/shared"
if os.path.exists(shared_dir):
    shutil.rmtree(shared_dir)
os.makedirs(shared_dir, exist_ok=True)

inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    df = pd.read_parquet('/kaggle/working/submission.parquet')
    df.to_csv('/kaggle/working/submission.csv', index=False)
    display(df)"""


# submission_crashproof_tta_fast.py
import os
import shutil
import gc
import pydicom
import cv2
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import kaggle_evaluation.rsna_inference_server

# ===================== CONFIG =====================
ID_COL = 'SeriesInstanceUID'
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

LOCATION_MAP = {
    1: "Other Posterior Circulation",
    2: "Basilar Tip",
    3: "Right Posterior Communicating Artery",
    4: "Left Posterior Communicating Artery",
    5: "Right Infraclinoid Internal Carotid Artery",
    6: "Left Infraclinoid Internal Carotid Artery",
    7: "Right Supraclinoid Internal Carotid Artery",
    8: "Left Supraclinoid Internal Carotid Artery",
    9: "Right Middle Cerebral Artery",
    10: "Left Middle Cerebral Artery",
    11: "Right Anterior Cerebral Artery",
    12: "Left Anterior Cerebral Artery",
    13: "Anterior Communicating Artery",
}
ID_TO_LABEL = {i - 1: v for i, v in LOCATION_MAP.items()}

# Performance / timeout-oriented params
MAX_SLICES = 32            # sample at most this many slices per series (evenly spaced)
BATCH_SIZE = 16            # batch size for model inference
TARGET_SIZE = 640          # model input size
TTA_MODES = ['none', 'hflip']  # keep TTA small; remove 'hflip' if you want faster
THREADS = 1                # reduce CPU thread contention on Kaggle runner

# ===================== HELPERS =====================
def normalize_dicom(img):
    img = img.astype(np.float32)
    if img.max() == img.min():
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    return img.astype(np.uint8)

def preprocess_image_np(img, target_size=TARGET_SIZE):
    # expects grayscale numpy image
    img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    if img_resized.ndim == 2:
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_resized
    tensor = np.transpose(img_rgb, (2,0,1)).astype(np.float32) / 255.0
    return tensor  # CHW np float32

def sigmoid(x): 
    return 1.0/(1.0+np.exp(-x))

def sample_filepaths_evenly(filepaths, max_samples=MAX_SLICES):
    if len(filepaths) <= max_samples:
        return filepaths
    idxs = np.linspace(0, len(filepaths)-1, max_samples, dtype=int)
    return [filepaths[i] for i in idxs]

# ===================== MODEL =====================
# Update model_path to the correct path in your input dataset if needed
model_path = "/kaggle/input/yolo8_multimodal_modal/pytorch/default/1/best.torchscript"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Limit CPU threads to avoid overhead on Kaggle runners
torch.set_num_threads(THREADS)
torch.set_num_interop_threads(THREADS)

try:
    model = torch.jit.load(model_path, map_location=device)
    model.to(device)
    model.eval()
except Exception as e:
    # If model fails to load, we still want to register server (predict will return zeros)
    model = None
    print(f"[ERROR] Could not load model: {e}")

# ===================== PREDICTION =====================
def run_model_on_batch(batch_tensor):
    """
    batch_tensor: torch.Tensor shape (N,3,H,W)
    returns: list of outputs per image (each output may be numpy array or list)
    """
    with torch.no_grad():
        preds = model(batch_tensor)  # may be list or tensor depending on script
    # Normalize to list of per-image outputs (np)
    outputs_list = []
    if preds is None:
        return [None] * batch_tensor.shape[0]
    # Common ultralytics torchscript: returns list-like where preds[i] contains detections
    try:
        # If preds is iterable per image
        for p in preds:
            if isinstance(p, torch.Tensor):
                outputs_list.append(p.cpu().numpy())
            else:
                # try convert to numpy if possible
                try:
                    outputs_list.append(np.array(p))
                except:
                    outputs_list.append(p)
    except TypeError:
        # preds is a single tensor of shape (N, ...). Split by first dimension.
        if isinstance(preds, torch.Tensor):
            for i in range(preds.shape[0]):
                outputs_list.append(preds[i].unsqueeze(0).cpu().numpy())
        else:
            # unknown format - wrap and return
            outputs_list = [preds] * batch_tensor.shape[0]
    return outputs_list

def parse_detections_to_scores(detections, label_scores):
    """
    detections: numpy array-like for one image: rows of [x1,y1,x2,y2,conf,cls] or similar.
    Updates label_scores dict in place with max confidences.
    """
    if detections is None:
        return
    try:
        arr = np.array(detections)
        if arr.size == 0:
            return
        # Support shape (...,6) or list-of-lists
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 1:
            # single detection
            arr = arr.reshape(1, -1)
        for det in arr:
            if det.size < 6:
                continue
            conf_prob = float(sigmoid(float(det[4])) ) if (det[4] < -20 or det[4] > 20) else float(det[4])
            try:
                cls_id = int(det[5])
            except:
                continue
            label = ID_TO_LABEL.get(cls_id)
            if label:
                label_scores[label] = max(label_scores.get(label, 0.0), conf_prob)
    except Exception:
        # best-effort: ignore parse errors
        return

def predict(series_path: str) -> pd.DataFrame:
    """
    Must be fast. Returns one-row DataFrame with INDEX ID_COL and LABEL_COLS floats.
    """
    series_id = os.path.basename(series_path.rstrip("/"))
    final_scores = {label:0.0 for label in LABEL_COLS}

    # Early return zeros if model not loaded
    if model is None:
        row = [series_id] + [0.0 for _ in LABEL_COLS]
        return pd.DataFrame([row], columns=[ID_COL, *LABEL_COLS])

    try:
        # Gather DICOM file paths
        dcm_files = sorted([os.path.join(root, f)
                            for root, _, files in os.walk(series_path)
                            for f in files if f.lower().endswith(".dcm")])
        if len(dcm_files) == 0:
            # no dicoms -> zeros
            row = [series_id] + [0.0 for _ in LABEL_COLS]
            return pd.DataFrame([row], columns=[ID_COL, *LABEL_COLS])

        # sample evenly to limit runtime
        sampled_files = sample_filepaths_evenly(dcm_files, MAX_SLICES)

        # prepare batches for model
        tensors_for_infer = []
        # For mapping back: store indices mapping each prepared tensor to original augment type
        augment_map = []  # list of (file_idx, tta_mode)
        for fp in sampled_files:
            try:
                ds = pydicom.dcmread(fp, force=True)
                img = normalize_dicom(ds.pixel_array)
            except Exception:
                continue
            base_np = preprocess_image_np(img, TARGET_SIZE)
            for tta in TTA_MODES:
                if tta == 'none':
                    tensors_for_infer.append(base_np)
                    augment_map.append((fp, 'none'))
                elif tta == 'hflip':
                    flipped = np.flip(base_np, axis=2).copy()  # flip horizontally in W axis since CHW
                    tensors_for_infer.append(flipped)
                    augment_map.append((fp, 'hflip'))
                # keep TTA small to save time

        if len(tensors_for_infer) == 0:
            row = [series_id] + [0.0 for _ in LABEL_COLS]
            return pd.DataFrame([row], columns=[ID_COL, *LABEL_COLS])

        # Batch inference
        all_scores_per_prep = []  # list of dicts per prepared image
        idx = 0
        N = len(tensors_for_infer)
        while idx < N:
            batch_np = tensors_for_infer[idx: idx + BATCH_SIZE]
            batch_tensor = torch.from_numpy(np.stack(batch_np, axis=0)).to(device)  # (B,3,H,W)
            # Run model and parse outputs
            try:
                outputs_list = run_model_on_batch(batch_tensor)
            except Exception as e:
                # if batch inference fails, try per-image fallback to avoid total crash
                outputs_list = []
                for single in batch_np:
                    try:
                        st = torch.from_numpy(single[None]).to(device)
                        olist = run_model_on_batch(st)
                        outputs_list.extend(olist)
                    except Exception:
                        outputs_list.append(None)

            # for each output, update slice-level scores
            for out in outputs_list:
                slice_scores = {label:0.0 for label in LABEL_COLS}
                parse_detections_to_scores(out, slice_scores)
                all_scores_per_prep.append(slice_scores)

            # cleanup
            del batch_tensor
            gc.collect()
            idx += BATCH_SIZE

        # Reduce across TTA and slices: take max across all scores
        for s in all_scores_per_prep:
            for k, v in s.items():
                final_scores[k] = max(final_scores.get(k, 0.0), float(v))

        # Aneurysm Present is the max across location labels (exclude last label if already included)
        location_only_labels = [l for l in LABEL_COLS if l != 'Aneurysm Present']
        final_scores['Aneurysm Present'] = max([final_scores.get(l, 0.0) for l in location_only_labels] + [final_scores.get('Aneurysm Present', 0.0)])

    except Exception as e:
        # On any unexpected error, log and return zeros (fast)
        print(f"[ERROR] Series {series_id} failed during predict: {e}")
        final_scores = {label:0.0 for label in LABEL_COLS}

    row = [series_id] + [float(final_scores.get(c, 0.0)) for c in LABEL_COLS]
    return pd.DataFrame([row], columns=[ID_COL, *LABEL_COLS])

# ===================== SERVER =====================
shared_dir = "/kaggle/shared"
shutil.rmtree(shared_dir, ignore_errors=True)
os.makedirs(shared_dir, exist_ok=True)

server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    # In competition environment the runner will call endpoints
    server.serve()
else:
    # For local testing in notebook
    server.run_local_gateway()
    submission_path = '/kaggle/working/submission.parquet'
    if os.path.exists(submission_path):
        df = pd.read_parquet(submission_path)
        df.to_csv('/kaggle/working/submission.csv', index=False)
        display(df)



















