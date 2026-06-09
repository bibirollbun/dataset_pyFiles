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
import scipy.special
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

# -------------------- STABILIZATION HELPERS --------------------
def smooth_scores(scores_list, alpha=0.7):
    smoothed = {label: 0.0 for label in LABEL_COLS}
    for s in scores_list:
        for l in LABEL_COLS:
            smoothed[l] = alpha * smoothed[l] + (1 - alpha) * s[l]
    return smoothed

def normalize_scores(final_scores):
    loc_labels = LABEL_COLS[:-1]  # exclude Aneurysm Present
    values = np.array([final_scores[l] for l in loc_labels])

    norm_values = scipy.special.softmax(values)
    norm_scores = {label: float(val) for label, val in zip(loc_labels, norm_values)}

    norm_scores["Aneurysm Present"] = max(norm_values.max(), final_scores["Aneurysm Present"])
    return norm_scores

# ===================== MODEL =====================
model_path = "/kaggle/input/yolo8_trained_model/pytorch/default/1/best.torchscript"
assert os.path.exists(model_path), f"Model not found at {model_path}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load(model_path, map_location=device)
model.eval()

# ===================== PREDICTION =====================
def aggregate_tta(scores_list):
    #Confidence-weighted aggregation instead of simple smoothing.
    agg = {label: 0.0 for label in LABEL_COLS}
    weight_sum = 0.0
    for s in scores_list:
        w = s["Aneurysm Present"] + 1e-6  # higher weight if more confident
        for l in LABEL_COLS:
            agg[l] += s[l] * w
        weight_sum += w
    if weight_sum > 0:
        for l in LABEL_COLS:
            agg[l] /= weight_sum
    return agg


def predict(series_path: str) -> pd.DataFrame:
    try:
        series_id = os.path.basename(series_path)

        # --- preprocess dicoms ---
        volume = process_dicom_series(series_path)
        base_image = create_multichannel_img(volume)

        scores_list = []
        for aug_img in tta_transforms(base_image):
            try:
                img_tensor = preprocess_image(aug_img).to(device)
                with torch.no_grad():
                    outputs = model(img_tensor)

                # convert outputs
                if outputs is None or len(outputs) == 0:
                    continue
                if isinstance(outputs, torch.Tensor):
                    outputs = outputs.cpu().numpy()
                elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                    outputs = [o.cpu().numpy() if isinstance(o, torch.Tensor) else o for o in outputs]
                else:
                    continue

            except Exception as e:
                print(f"Skipping TTA image due to error: {e}")
                continue

            # --- score dict ---
            scores = {label: 0.0 for label in LABEL_COLS}
            aneurysm_present = 0.0

            for det in outputs[0]:
                try:
                    conf_prob = float(det[4])  # no sigmoid, model already calibrated
                    cls_id = int(det[5])
                    label = ID_TO_LABEL.get(cls_id, None)
                    if label is not None:
                        scores[label] = max(scores[label], conf_prob)
                        aneurysm_present = max(aneurysm_present, conf_prob)
                except Exception:
                    continue

            # aneurysm present is independent, not just max of others
            scores["Aneurysm Present"] = aneurysm_present
            scores_list.append(scores)

        # --- aggregation ---
        if len(scores_list) == 0:
            final_scores = {label: 0.0 for label in LABEL_COLS}
        else:
            agg_scores = aggregate_tta(scores_list)
            final_scores = normalize_scores(agg_scores)

        # --- thresholding (tune for CV score) ---
        THRESH = 0.2
        for l in LABEL_COLS:
            if final_scores[l] < THRESH:
                final_scores[l] = 0.0

        del volume, base_image, scores_list
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        row = [series_id] + [round(float(final_scores[c]), 6) for c in LABEL_COLS]
        df = pd.DataFrame([row], columns=[ID_COL, *LABEL_COLS])
        return df

    except Exception as e:
        print(f"Failed series {series_path} with error: {e}")
        row = [os.path.basename(series_path)] + [0.0] * len(LABEL_COLS)
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
    submission_path = "/kaggle/working/submission.parquet"
    if os.path.exists(submission_path):
        df = pd.read_parquet(submission_path)
        df.to_csv("/kaggle/working/submission.csv", index=False, float_format="%.6f")
        display(df.head())"""


"""# ===================== submission_kaggle_improved.py =====================
import os
import shutil
import gc
from pathlib import Path

import pydicom
import cv2
import numpy as np
import pandas as pd
import torch
import scipy.special
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
    #Multi-windowing to highlight vessels and brain structures.
    windows = [(0, 100), (50, 150), (80, 200)]
    windowed_images = []
    for low, high in windows:
        img_clip = np.clip(image, low, high)
        img_norm = ((img_clip - low) / (high - low + 1e-8) * 255).astype(np.uint8)
        windowed_images.append(img_norm)
    return np.stack(windowed_images, axis=-1).mean(axis=-1).astype(np.uint8)

def process_dicom_series(series_path, image_size=512, num_slices=16):
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

    # Sample or pad slices
    if volume.shape[0] > num_slices:
        idx = np.linspace(0, volume.shape[0] - 1, num_slices).astype(int)
        volume = volume[idx]
    elif volume.shape[0] < num_slices:
        pad = num_slices - volume.shape[0]
        volume = np.pad(volume, ((0, pad), (0, 0), (0, 0)), mode="reflect")

    return volume

def create_multichannel_img(volume):
    depth, h, w = volume.shape
    mip = np.max(volume, axis=0)
    slice_weights = np.percentile(volume, 75, axis=(1,2))
    weighted_avg = np.average(volume, axis=0, weights=slice_weights)
    std_proj = np.zeros_like(volume[0])
    for i in range(depth - 4):
        std_proj = np.maximum(std_proj, np.std(volume[i:i+5], axis=0))
    chans = []
    for ch in [mip, weighted_avg, std_proj]:
        ch_norm = ((ch - ch.min()) / (ch.max() - ch.min() + 1e-8) * 255).astype(np.uint8)
        chans.append(ch_norm)
    return np.stack(chans, axis=-1)

def preprocess_image(img, target_size=640):
    img_resized = cv2.resize(img, (target_size, target_size))
    if img_resized.ndim == 2:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    else:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(img_resized, (2,0,1)).astype(np.float32) / 255.0
    return torch.from_numpy(tensor).unsqueeze(0)

def sigmoid(x, T=1.5):
    return 1 / (1 + np.exp(-x / T))

def tta_transforms(image):
    transforms = [image]
    transforms += [cv2.flip(image, 0), cv2.flip(image, 1)]
    transforms += [cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
                   cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    transforms += [cv2.rotate(cv2.flip(image, 1), cv2.ROTATE_90_CLOCKWISE)]
    return transforms

def smooth_scores(scores_list, alpha=0.7):
    smoothed = {label: 0.0 for label in LABEL_COLS}
    for s in scores_list:
        for l in LABEL_COLS:
            smoothed[l] = alpha * smoothed[l] + (1 - alpha) * s[l]
    return smoothed

def normalize_scores(final_scores):
    loc_labels = LABEL_COLS[:-1]
    values = np.array([final_scores[l] for l in loc_labels])
    norm_values = scipy.special.softmax(values)
    norm_scores = {label: float(val) for label, val in zip(loc_labels, norm_values)}
    norm_scores["Aneurysm Present"] = max(norm_values.max(), final_scores["Aneurysm Present"])
    return norm_scores

# ===================== MODEL =====================
model_path = "/kaggle/input/yolo8_trained_model/pytorch/default/1/best.torchscript"
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
                    if isinstance(outputs, torch.Tensor):
                        outputs = outputs.cpu().numpy()
                    elif isinstance(outputs, (list, tuple)):
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
            smoothed = smooth_scores(scores_list, alpha=0.7)
            final_scores = normalize_scores(smoothed)

        del volume, base_image, scores_list
        gc.collect()

        row = [series_id] + [round(float(final_scores[c]), 6) for c in LABEL_COLS]
        df = pd.DataFrame([row], columns=[ID_COL, *LABEL_COLS])
        return df

    except Exception as e:
        print(f"Failed series {series_path} with error: {e}")
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
    submission_path = "/kaggle/working/submission.parquet"
    if os.path.exists(submission_path):
        df = pd.read_parquet(submission_path)
        df.to_csv("/kaggle/working/submission.csv", index=False, float_format="%.6f")
        display(df.head())"""


"""import os
import shutil
import gc
import pydicom
import cv2
import numpy as np
import pandas as pd
import torch
import scipy.special
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
    # Multi-windowing + mean aggregation
    windows = [(0, 100), (50, 150), (80, 200)]
    imgs = []
    for low, high in windows:
        img_clip = np.clip(image, low, high)
        img_norm = ((img_clip - low) / (high - low + 1e-8) * 255).astype(np.uint8)
        imgs.append(img_norm)
    return np.mean(np.stack(imgs, axis=-1), axis=-1).astype(np.uint8)

def process_dicom_series(series_path, image_size=512, num_slices=16):
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
            dicom_data.append(cv2.resize(img, (image_size, image_size)))
        except:
            continue
    if len(dicom_data) == 0:
        return np.zeros((num_slices, image_size, image_size), dtype=np.uint8)
    volume = np.stack(dicom_data, axis=0)
    volume = adaptive_windowing(volume)
    if volume.shape[0] > num_slices:
        idx = np.linspace(0, volume.shape[0]-1, num_slices).astype(int)
        volume = volume[idx]
    elif volume.shape[0] < num_slices:
        pad = num_slices - volume.shape[0]
        volume = np.pad(volume, ((0,pad),(0,0),(0,0)), mode="reflect")
    return volume

def create_multichannel_img(volume):
    depth = volume.shape[0]
    mip = np.max(volume, axis=0)
    central = volume[depth//2]
    slice_weights = np.percentile(volume, 75, axis=(1,2))
    weighted_avg = np.average(volume, axis=0, weights=slice_weights)
    std_proj = np.zeros_like(volume[0])
    for i in range(depth-4):
        std_proj = np.maximum(std_proj, np.std(volume[i:i+5], axis=0))
    chans = [mip, central, weighted_avg, std_proj]
    chans_norm = [((ch - ch.min()) / (ch.max()-ch.min()+1e-8) * 255).astype(np.uint8) for ch in chans]
    return np.stack(chans_norm, axis=-1)

def preprocess_image(img, target_size=640):
    img_resized = cv2.resize(img, (target_size, target_size))
    if img_resized.ndim==2: img_resized=cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    else: img_resized=cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(img_resized,(2,0,1)).astype(np.float32)/255
    return torch.from_numpy(tensor).unsqueeze(0)

def sigmoid(x, T=1.2):  # tuned temperature
    return 1 / (1 + np.exp(-x / T))

def tta_transforms(img):
    # 8 augmentations
    imgs = [img, cv2.flip(img,0), cv2.flip(img,1),
            cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE),
            cv2.rotate(cv2.flip(img,1),cv2.ROTATE_90_CLOCKWISE)]
    return imgs

def smooth_scores(scores_list, alpha=0.6):
    smoothed={l:0.0 for l in LABEL_COLS}
    for s in scores_list:
        for l in LABEL_COLS:
            smoothed[l]=alpha*smoothed[l]+(1-alpha)*s[l]
    return smoothed

def normalize_scores(final_scores):
    loc_labels = LABEL_COLS[:-1]
    values = np.array([final_scores[l] for l in loc_labels])
    norm_values = scipy.special.softmax(values)
    norm_scores = {label: float(val) for label,val in zip(loc_labels,norm_values)}
    norm_scores["Aneurysm Present"] = max(norm_values.max(), final_scores["Aneurysm Present"])
    return norm_scores

# ===================== MODEL =====================
model_path = "/kaggle/input/yolo8_trained_model/pytorch/default/1/best.torchscript"
assert os.path.exists(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load(model_path,map_location=device)
model.eval()

# ===================== PREDICTION =====================
def predict(series_path):
    try:
        series_id=os.path.basename(series_path)
        vol=process_dicom_series(series_path)
        img=create_multichannel_img(vol)
        scores_list=[]
        for aug in tta_transforms(img):
            try:
                tensor=preprocess_image(aug).to(device)
                with torch.no_grad():
                    out=model(tensor)
                    if out is None or len(out)==0: continue
                    if isinstance(out,torch.Tensor): out=[out.cpu().numpy()]
                    elif isinstance(out,(list,tuple)): out=[o.cpu().numpy() if isinstance(o,torch.Tensor) else o for o in out]
            except: continue
            scores={l:0.0 for l in LABEL_COLS}
            aneurysm=0.0
            for det in out[0]:
                try:
                    conf=sigmoid(float(det[4]))
                    cls=int(det[5])
                    label=ID_TO_LABEL.get(cls,None)
                    if label: scores[label]=max(scores[label],conf); aneurysm=max(aneurysm,conf)
                except: continue
            scores["Aneurysm Present"]=max(aneurysm,max(scores.values()))
            scores_list.append(scores)
        if not scores_list: final={l:0.0 for l in LABEL_COLS}
        else: final=normalize_scores(smooth_scores(scores_list))
        row=[series_id]+[round(float(final[c]),6) for c in LABEL_COLS]
        return pd.DataFrame([row],columns=[ID_COL,*LABEL_COLS])
    except:
        row=[os.path.basename(series_path)]+[0.0]*len(LABEL_COLS)
        return pd.DataFrame([row],columns=[ID_COL,*LABEL_COLS])

# ===================== SERVER / LOCAL =====================
shared_dir="/kaggle/shared"
shutil.rmtree(shared_dir,ignore_errors=True)
os.makedirs(shared_dir,exist_ok=True)
server=kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)
if os.getenv("KAGGLE_IS_COMPETITION_RERUN"): server.serve()
else:
    server.run_local_gateway()
    sp="/kaggle/working/submission.parquet"
    if os.path.exists(sp):
        df=pd.read_parquet(sp)
        df.to_csv("/kaggle/working/submission.csv",index=False,float_format="%.6f")
        display(df.head())"""


# ===================== submission_kaggle_diffusion.py =====================
import os
import shutil
import gc
from pathlib import Path

import pydicom
import cv2
import numpy as np
import pandas as pd
import torch
import timm
import scipy.special
import kaggle_evaluation.rsna_inference_server
import matplotlib.pyplot as plt

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== HELPERS =====================
def adaptive_windowing(image):
    windows = [(0, 100), (50, 150), (80, 200)]
    windowed_images = []
    for low, high in windows:
        img_clip = np.clip(image, low, high)
        img_norm = ((img_clip - low) / (high - low + 1e-8) * 255).astype(np.uint8)
        windowed_images.append(img_norm)
    return np.stack(windowed_images, axis=-1).mean(axis=-1).astype(np.uint8)

def process_dicom_series(series_path, image_size=512, num_slices=16):
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
        volume = np.pad(volume, ((0, pad), (0, 0), (0, 0)), mode="reflect")
    return volume

def create_multichannel_img(volume):
    depth, h, w = volume.shape
    mip = np.max(volume, axis=0)
    slice_weights = np.percentile(volume, 75, axis=(1,2))
    weighted_avg = np.average(volume, axis=0, weights=slice_weights)
    std_proj = np.zeros_like(volume[0])
    for i in range(depth - 4):
        std_proj = np.maximum(std_proj, np.std(volume[i:i+5], axis=0))
    chans = []
    for ch in [mip, weighted_avg, std_proj]:
        ch_norm = ((ch - ch.min()) / (ch.max() - ch.min() + 1e-8) * 255).astype(np.uint8)
        chans.append(ch_norm)
    return np.stack(chans, axis=-1)

def preprocess_image(img, target_size=640):
    img_resized = cv2.resize(img, (target_size, target_size))
    if img_resized.ndim == 2:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    else:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(img_resized, (2,0,1)).astype(np.float32) / 255.0
    return torch.from_numpy(tensor).unsqueeze(0)

def sigmoid(x, T=1.5):
    return 1 / (1 + np.exp(-x / T))

def tta_transforms(image):
    transforms = [image]
    transforms += [cv2.flip(image, 0), cv2.flip(image, 1)]
    transforms += [cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
                   cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    transforms += [cv2.rotate(cv2.flip(image, 1), cv2.ROTATE_90_CLOCKWISE)]
    return transforms

def smooth_scores(scores_list, alpha=0.6):
    avg_scores = {label: np.mean([s[label] for s in scores_list]) for label in LABEL_COLS}
    smoothed = {label: alpha * avg_scores[label] + (1-alpha) * max(avg_scores[label], 0.3) for label in LABEL_COLS}
    return smoothed

def normalize_scores(final_scores):
    loc_labels = LABEL_COLS[:-1]
    values = np.array([final_scores[l] for l in loc_labels])
    norm_values = scipy.special.softmax(values)
    norm_scores = {label: float(val) for label, val in zip(loc_labels, norm_values)}
    norm_scores["Aneurysm Present"] = max(norm_values.max(), final_scores["Aneurysm Present"])
    return norm_scores

# ===================== LOAD MODELS =====================
# YOLOv8 TorchScript
yolo_model_path = "/kaggle/input/yolo8_trained_model/pytorch/default/1/best.torchscript"
yolo_model = torch.jit.load(yolo_model_path, map_location=device)
yolo_model.eval()

# EfficientNet-B3 via timm + KaggleHub
import kagglehub
eff_path = kagglehub.model_download("timm/tf-efficientnet/pyTorch/tf-efficientnet-b3")
model_file = next((os.path.join(eff_path, f) for f in os.listdir(eff_path) if f.endswith(".pt") or f.endswith(".pth")), None)

# Create model with correct num_classes
efficientnet_model = timm.create_model("tf_efficientnet_b3", pretrained=False, num_classes=len(LABEL_COLS)-1)
state_dict = torch.load(model_file, map_location=device)

# Remove classifier weights to avoid size mismatch
for k in list(state_dict.keys()):
    if "classifier" in k:
        del state_dict[k]

efficientnet_model.load_state_dict(state_dict, strict=False)
efficientnet_model = efficientnet_model.to(device)
efficientnet_model.eval()

# ===================== PREDICTION =====================
YOLO_WEIGHT = 0.6
EFF_WEIGHT = 0.4
THRESHOLDS = {"Aneurysm Present": 0.3}

def predict(series_path: str) -> pd.DataFrame:
    series_id = os.path.basename(series_path)
    volume = process_dicom_series(series_path)
    base_image = create_multichannel_img(volume)
    scores_list = []

    for aug_img in tta_transforms(base_image):
        yolo_scores = {label:0.0 for label in LABEL_COLS}
        eff_scores = {label:0.0 for label in LABEL_COLS}

        # YOLOv8
        try:
            img_tensor = preprocess_image(aug_img).to(device)
            with torch.no_grad():
                outputs = yolo_model(img_tensor)
                if isinstance(outputs, torch.Tensor):
                    outputs = outputs.cpu().numpy()
                elif isinstance(outputs, (list, tuple)):
                    outputs = [o.cpu().numpy() if isinstance(o, torch.Tensor) else o for o in outputs]
                for det in outputs[0]:
                    conf_prob = sigmoid(float(det[4]))
                    cls_id = int(det[5])
                    label = ID_TO_LABEL.get(cls_id, None)
                    if label:
                        yolo_scores[label] = max(yolo_scores[label], conf_prob)
                yolo_scores["Aneurysm Present"] = max(yolo_scores.values())
        except:
            pass

        # EfficientNet
        try:
            img_tensor = preprocess_image(aug_img).to(device)
            with torch.no_grad():
                eff_out = efficientnet_model(img_tensor)
                eff_out = torch.sigmoid(eff_out).cpu().numpy()[0]
                for i, label in enumerate(LABEL_COLS[:-1]):
                    eff_scores[label] = float(eff_out[i])
                eff_scores["Aneurysm Present"] = max(eff_scores.values())
        except:
            pass

        combined_scores = {label: YOLO_WEIGHT*yolo_scores[label]+EFF_WEIGHT*eff_scores[label] for label in LABEL_COLS}
        scores_list.append(combined_scores)

    final_scores = normalize_scores(smooth_scores(scores_list))
    for label in THRESHOLDS:
        final_scores[label] = max(final_scores[label], THRESHOLDS[label])

    # Visualization (diffusion style)
    plt.figure(figsize=(8,4))
    plt.bar(range(len(LABEL_COLS)), [final_scores[l] for l in LABEL_COLS], tick_label=LABEL_COLS)
    plt.xticks(rotation=90)
    plt.title(f"Diffusion Ensemble Scores: {series_id}")
    plt.show()

    del volume, base_image, scores_list
    gc.collect()
    row = [series_id] + [round(float(final_scores[c]), 6) for c in LABEL_COLS]
    return pd.DataFrame([row], columns=[ID_COL, *LABEL_COLS])

# ===================== SERVER =====================
shared_dir = "/kaggle/shared"
shutil.rmtree(shared_dir, ignore_errors=True)
os.makedirs(shared_dir, exist_ok=True)
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)
if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    submission_path = "/kaggle/working/submission.parquet"
    if os.path.exists(submission_path):
        df = pd.read_parquet(submission_path)
        df.to_csv("/kaggle/working/submission.csv", index=False, float_format="%.6f")
        display(df.head())


"""# ===================== submission_kaggle_diffusion_ensemble.py =====================
import os
import shutil
import gc
from pathlib import Path

import pydicom
import cv2
import numpy as np
import pandas as pd
import torch
import timm
import scipy.special
import kaggle_evaluation.rsna_inference_server
import matplotlib.pyplot as plt

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== HELPERS =====================
def adaptive_windowing(image):
    windows = [(0, 100), (50, 150), (80, 200)]
    windowed_images = []
    for low, high in windows:
        img_clip = np.clip(image, low, high)
        img_norm = ((img_clip - low) / (high - low + 1e-8) * 255).astype(np.uint8)
        windowed_images.append(img_norm)
    return np.stack(windowed_images, axis=-1).mean(axis=-1).astype(np.uint8)

def process_dicom_series(series_path, image_size=512, num_slices=16):
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
        volume = np.pad(volume, ((0, pad), (0, 0), (0, 0)), mode="reflect")
    return volume

def create_multichannel_img(volume):
    depth, h, w = volume.shape
    mip = np.max(volume, axis=0)
    slice_weights = np.percentile(volume, 75, axis=(1,2))
    weighted_avg = np.average(volume, axis=0, weights=slice_weights)
    std_proj = np.zeros_like(volume[0])
    for i in range(depth - 4):
        std_proj = np.maximum(std_proj, np.std(volume[i:i+5], axis=0))
    chans = []
    for ch in [mip, weighted_avg, std_proj]:
        ch_norm = ((ch - ch.min()) / (ch.max() - ch.min() + 1e-8) * 255).astype(np.uint8)
        chans.append(ch_norm)
    return np.stack(chans, axis=-1)

def preprocess_image(img, target_size=640):
    img_resized = cv2.resize(img, (target_size, target_size))
    if img_resized.ndim == 2:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    else:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(img_resized, (2,0,1)).astype(np.float32) / 255.0
    return torch.from_numpy(tensor).unsqueeze(0)

def sigmoid(x, T=1.5):
    return 1 / (1 + np.exp(-x / T))

def tta_transforms(image):
    transforms = [image]
    transforms += [cv2.flip(image, 0), cv2.flip(image, 1)]
    transforms += [cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
                   cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    transforms += [cv2.rotate(cv2.flip(image, 1), cv2.ROTATE_90_CLOCKWISE)]
    return transforms

def smooth_scores(scores_list, alpha=0.6):
    avg_scores = {label: np.mean([s[label] for s in scores_list]) for label in LABEL_COLS}
    smoothed = {label: alpha * avg_scores[label] + (1-alpha) * max(avg_scores[label], 0.3) for label in LABEL_COLS}
    return smoothed

def normalize_scores(final_scores):
    loc_labels = LABEL_COLS[:-1]
    values = np.array([final_scores[l] for l in loc_labels])
    norm_values = scipy.special.softmax(values)
    norm_scores = {label: float(val) for label, val in zip(loc_labels, norm_values)}
    norm_scores["Aneurysm Present"] = max(norm_values.max(), final_scores["Aneurysm Present"])
    return norm_scores

# ===================== LOAD MODELS =====================
import kagglehub

# YOLOv8
yolo_model_path = "/kaggle/input/yolo8_trained_model/pytorch/default/1/best.torchscript"
yolo_model = torch.jit.load(yolo_model_path, map_location=device)
yolo_model.eval()

# Multi EfficientNet ensemble
eff_models = [
    "timm/tf-efficientnet/pyTorch/tf-efficientnet-b3",
    "dennisfong/rsna-2025-ia-ct-224-efficientnet/pyTorch/default"
]
efficientnet_models = []

for model_name in eff_models:
    eff_path = kagglehub.model_download(model_name)
    model_file = next((os.path.join(eff_path, f) for f in os.listdir(eff_path) if f.endswith(".pt") or f.endswith(".pth")), None)
    eff_model = timm.create_model("tf_efficientnet_b3", pretrained=False, num_classes=len(LABEL_COLS)-1)
    state_dict = torch.load(model_file, map_location=device)
    for k in list(state_dict.keys()):
        if "classifier" in k:
            del state_dict[k]
    eff_model.load_state_dict(state_dict, strict=False)
    eff_model = eff_model.to(device)
    eff_model.eval()
    efficientnet_models.append(eff_model)

# ===================== PREDICTION =====================
YOLO_WEIGHT = 0.5
EFF_WEIGHT = 0.5 / len(efficientnet_models)
THRESHOLDS = {"Aneurysm Present": 0.3}

def predict(series_path: str) -> pd.DataFrame:
    series_id = os.path.basename(series_path)
    volume = process_dicom_series(series_path)
    base_image = create_multichannel_img(volume)
    scores_list = []

    for aug_img in tta_transforms(base_image):
        yolo_scores = {label:0.0 for label in LABEL_COLS}
        eff_scores = {label:0.0 for label in LABEL_COLS}

        # YOLOv8
        try:
            img_tensor = preprocess_image(aug_img).to(device)
            with torch.no_grad():
                outputs = yolo_model(img_tensor)
                if isinstance(outputs, torch.Tensor):
                    outputs = outputs.cpu().numpy()
                elif isinstance(outputs, (list, tuple)):
                    outputs = [o.cpu().numpy() if isinstance(o, torch.Tensor) else o for o in outputs]
                for det in outputs[0]:
                    conf_prob = sigmoid(float(det[4]))
                    cls_id = int(det[5])
                    label = ID_TO_LABEL.get(cls_id, None)
                    if label:
                        yolo_scores[label] = max(yolo_scores[label], conf_prob)
                yolo_scores["Aneurysm Present"] = max(yolo_scores.values())
        except:
            pass

        # EfficientNet ensemble
        for model in efficientnet_models:
            try:
                img_tensor = preprocess_image(aug_img).to(device)
                with torch.no_grad():
                    eff_out = model(img_tensor)
                    eff_out = torch.sigmoid(eff_out).cpu().numpy()[0]
                    for i, label in enumerate(LABEL_COLS[:-1]):
                        eff_scores[label] += float(eff_out[i]) * EFF_WEIGHT
                    eff_scores["Aneurysm Present"] = max(eff_scores.values())
            except:
                continue

        combined_scores = {label: YOLO_WEIGHT*yolo_scores[label]+eff_scores[label] for label in LABEL_COLS}
        scores_list.append(combined_scores)

    final_scores = normalize_scores(smooth_scores(scores_list))
    for label in THRESHOLDS:
        final_scores[label] = max(final_scores[label], THRESHOLDS[label])

    # Diffusion visualization
    plt.figure(figsize=(10,5))
    plt.bar(range(len(LABEL_COLS)), [final_scores[l] for l in LABEL_COLS], tick_label=LABEL_COLS)
    plt.xticks(rotation=90)
    plt.title(f"Diffusion Ensemble Scores: {series_id}")
    plt.show()

    del volume, base_image, scores_list
    gc.collect()
    row = [series_id] + [round(float(final_scores[c]), 6) for c in LABEL_COLS]
    return pd.DataFrame([row], columns=[ID_COL, *LABEL_COLS])

# ===================== SERVER =====================
shared_dir = "/kaggle/shared"
shutil.rmtree(shared_dir, ignore_errors=True)
os.makedirs(shared_dir, exist_ok=True)
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)
if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    submission_path = "/kaggle/working/submission.parquet"
    if os.path.exists(submission_path):
        df = pd.read_parquet(submission_path)
        df.to_csv("/kaggle/working/submission.csv", index=False, float_format="%.6f")
        display(df.head())"""


"""# ===================== submission_kaggle_ensemble.py =====================
import os
import shutil
import gc
from pathlib import Path
import warnings

import pydicom
import cv2
import numpy as np
import pandas as pd
import torch
import timm
import scipy.special
import matplotlib.pyplot as plt
import kagglehub
import kaggle_evaluation.rsna_inference_server

warnings.filterwarnings("ignore")

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== MODEL LOADING =====================
# --- YOLOv8 Model ---
yolo_model_path = "/kaggle/input/yolo8_trained_model/pytorch/default/1/best.torchscript"
yolo_model = torch.jit.load(yolo_model_path, map_location=device)
yolo_model.eval()

# --- EfficientNet Model via KaggleHub ---
eff_path = kagglehub.model_download("dennisfong/rsna-2025-ia-ct-224-efficientnet/pyTorch/default")
model_file = next((os.path.join(eff_path, f) for f in os.listdir(eff_path) if f.endswith(".pt") or f.endswith(".pth")), None)
efficientnet_model = timm.create_model("tf_efficientnet_b3", pretrained=False, num_classes=len(LABEL_COLS)-1)

# Load checkpoint safely
state_dict = torch.load(model_file, map_location=device)
if "state_dict" in state_dict:  # handle kagglehub dict
    state_dict = state_dict["state_dict"]

# Remove 'classifier.' prefix if exists
new_state_dict = {}
for k, v in state_dict.items():
    key = k.replace("classifier.", "")
    new_state_dict[key] = v
efficientnet_model.load_state_dict(new_state_dict, strict=False)

efficientnet_model = efficientnet_model.to(device)
efficientnet_model.eval()

# ===================== HELPERS =====================
def adaptive_windowing(image):
    windows = [(0, 100), (50, 150), (80, 200)]
    windowed_images = []
    for low, high in windows:
        img_clip = np.clip(image, low, high)
        img_norm = ((img_clip - low) / (high - low + 1e-8) * 255).astype(np.uint8)
        windowed_images.append(img_norm)
    return np.stack(windowed_images, axis=-1).mean(axis=-1).astype(np.uint8)

def process_dicom_series(series_path, image_size=512, num_slices=16):
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
        except:
            continue

    if len(dicom_data) == 0:
        return np.zeros((num_slices, image_size, image_size), dtype=np.uint8)

    volume = np.stack(dicom_data, axis=0)
    volume = adaptive_windowing(volume)

    if volume.shape[0] > num_slices:
        idx = np.linspace(0, volume.shape[0]-1, num_slices).astype(int)
        volume = volume[idx]
    elif volume.shape[0] < num_slices:
        pad = num_slices - volume.shape[0]
        volume = np.pad(volume, ((0,pad),(0,0),(0,0)), mode="reflect")

    return volume

def create_multichannel_img(volume):
    depth, h, w = volume.shape
    mip = np.max(volume, axis=0)
    slice_weights = np.percentile(volume, 75, axis=(1,2))
    weighted_avg = np.average(volume, axis=0, weights=slice_weights)
    std_proj = np.zeros_like(volume[0])
    for i in range(depth-4):
        std_proj = np.maximum(std_proj, np.std(volume[i:i+5], axis=0))
    chans = []
    for ch in [mip, weighted_avg, std_proj]:
        ch_norm = ((ch - ch.min()) / (ch.max() - ch.min() + 1e-8) * 255).astype(np.uint8)
        chans.append(ch_norm)
    return np.stack(chans, axis=-1)

def preprocess_image(img, target_size=640):
    img_resized = cv2.resize(img, (target_size, target_size))
    if img_resized.ndim == 2:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    else:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(img_resized, (2,0,1)).astype(np.float32)/255.0
    return torch.from_numpy(tensor).unsqueeze(0)

def sigmoid(x, T=1.5):
    return 1/(1+np.exp(-x/T))

def tta_transforms(image):
    transforms = [image]
    transforms += [cv2.flip(image, 0), cv2.flip(image,1)]
    transforms += [cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    transforms += [cv2.rotate(cv2.flip(image,1), cv2.ROTATE_90_CLOCKWISE)]
    return transforms

def smooth_scores(scores_list, alpha=0.7):
    smoothed = {label: 0.0 for label in LABEL_COLS}
    for s in scores_list:
        for l in LABEL_COLS:
            smoothed[l] = alpha*smoothed[l] + (1-alpha)*s[l]
    return smoothed

def normalize_scores(final_scores):
    loc_labels = LABEL_COLS[:-1]
    values = np.array([final_scores[l] for l in loc_labels])
    norm_values = scipy.special.softmax(values)
    norm_scores = {label: float(val) for label, val in zip(loc_labels, norm_values)}
    norm_scores["Aneurysm Present"] = max(norm_values.max(), final_scores["Aneurysm Present"])
    return norm_scores

# ===================== PREDICTION =====================
def predict(series_path: str) -> pd.DataFrame:
    try:
        series_id = os.path.basename(series_path)
        volume = process_dicom_series(series_path)
        base_image = create_multichannel_img(volume)

        scores_list = []

        # TTA + Ensemble
        for aug_img in tta_transforms(base_image):
            # --- YOLOv8 ---
            try:
                img_tensor = preprocess_image(aug_img).to(device)
                with torch.no_grad():
                    outputs = yolo_model(img_tensor)
                    if outputs is None or len(outputs) == 0:
                        continue
                    if isinstance(outputs, torch.Tensor):
                        outputs = outputs.cpu().numpy()
                    elif isinstance(outputs, (list,tuple)):
                        outputs = [o.cpu().numpy() if isinstance(o, torch.Tensor) else o for o in outputs]
                    else:
                        continue
            except:
                continue

            scores = {label: 0.0 for label in LABEL_COLS}
            aneurysm_present = 0.0
            for det in outputs[0]:
                try:
                    conf_prob = sigmoid(float(det[4]))
                    cls_id = int(det[5])
                    label = ID_TO_LABEL.get(cls_id, None)
                    if label:
                        scores[label] = max(scores[label], conf_prob)
                        aneurysm_present = max(aneurysm_present, conf_prob)
                except:
                    continue
            scores["Aneurysm Present"] = max(aneurysm_present, max(scores.values()))
            scores_list.append(scores)

            # --- EfficientNet ---
            try:
                img_tensor_e = preprocess_image(aug_img, target_size=224).to(device)
                with torch.no_grad():
                    out = efficientnet_model(img_tensor_e)
                    probs = torch.sigmoid(out).cpu().numpy()[0]
                    for i, label in enumerate(LABEL_COLS[:-1]):
                        scores[label] = max(scores[label], float(probs[i]))
                    scores["Aneurysm Present"] = max(scores["Aneurysm Present"], max(probs))
            except:
                continue

        # Smooth + normalize
        if len(scores_list) == 0:
            final_scores = {label:0.0 for label in LABEL_COLS}
        else:
            smoothed = smooth_scores(scores_list)
            final_scores = normalize_scores(smoothed)

        del volume, base_image, scores_list
        gc.collect()

        row = [series_id]+[round(float(final_scores[c]),6) for c in LABEL_COLS]
        return pd.DataFrame([row], columns=[ID_COL,*LABEL_COLS])
    except:
        row = [os.path.basename(series_path)] + [0.0]*len(LABEL_COLS)
        return pd.DataFrame([row], columns=[ID_COL,*LABEL_COLS])

# ===================== SERVER / LOCAL =====================
shared_dir = "/kaggle/shared"
shutil.rmtree(shared_dir, ignore_errors=True)
os.makedirs(shared_dir, exist_ok=True)

inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    submission_path = "/kaggle/working/submission.parquet"
    if os.path.exists(submission_path):
        df = pd.read_parquet(submission_path)
        df.to_csv("/kaggle/working/submission.csv", index=False, float_format="%.6f")
        display(df.head())"""










