from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import random
import torch
import time
import cv2
import os
import gc

warnings.filterwarnings("ignore")


def get_batch_size(device, min_batch_size=8, max_batch_size=64):
    device_id = int(device.split(':')[1])
    gpu_mem = torch.cuda.get_device_properties(device_id).total_memory / 1e9  
    free_mem = gpu_mem - torch.cuda.memory_allocated(device_id) / 1e9
    batch_size = max(min_batch_size, min(max_batch_size, int(free_mem * 4)))
    torch.cuda.empty_cache()
    return batch_size


class CFG:
    dataset_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/"
    test_image_path = os.path.join(dataset_path, "test")
    model_paths = [
        "/kaggle/input/byu-2025-pretrained-models/riza_torchscript_models/reduced_redundancy_10s.torchscript",
        "/kaggle/input/byu-2025-pretrained-models/riza_torchscript_models/reduced_redundancy_11s.torchscript",
        "/kaggle/input/byu-2025-pretrained-models/riza_torchscript_models/reduced_redundancy_8m.torchscript",
        "/kaggle/input/byu-2025-pretrained-models/riza_torchscript_models/cleaned_data_box_1200_yolo8m.torchscript",
         "/kaggle/input/byu-2025-pretrained-models/riza_torchscript_models/cleaned_data_box_1200_yolo10m.torchscript",
       "/kaggle/input/byu-2025-pretrained-models/riza_torchscript_models/1440_box_11s.torchscript",
        "/kaggle/input/byu-2025-pretrained-models/home_made_mhaf_epoch29.torchscript",
        "/kaggle/input/byu-2025-pretrained-models/mayolov2f128.torchscript"
    ]
    
    model_image_sizes = [
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        960,
        # Add the rest in same order
    ]

    seed = 42    
    devices = ['cuda:0', 'cuda:1']
    box_size = 50
    concentration = 1
    confidence_threshold = 0.25
    agreement_iou_threshold = 0.1

    batch_sizes = [get_batch_size(device_id) for device_id in devices]



torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.manual_seed(CFG.seed)
random.seed(CFG.seed)
np.random.seed(CFG.seed)


class GPUProfiler:        
    def __enter__(self):
        torch.cuda.synchronize()
        return self
        
    def __exit__(self, *args):
        torch.cuda.synchronize()


def calculate_iou(det1, det2):
    """Calculate IoU between two bounding box dictionaries."""
    x1_i = max(det1['x1'], det2['x1'])
    y1_i = max(det1['y1'], det2['y1'])
    x2_i = min(det1['x2'], det2['x2'])
    y2_i = min(det1['y2'], det2['y2'])

    inter_w = max(0, x2_i - x1_i)
    inter_h = max(0, y2_i - y1_i)
    inter_area = inter_w * inter_h

    area1 = (det1['x2'] - det1['x1']) * (det1['y2'] - det1['y1'])
    area2 = (det2['x2'] - det2['x1']) * (det2['y2'] - det2['y1'])

    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0



def find_longest_consecutive_run_with_indices(numbers):
    """Find the longest run of consecutive integers in a sorted list."""
    if not numbers:
        return 0, []
    max_run_len = 1
    longest_run = [numbers[0]]
    current_run = [numbers[0]]
    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i - 1] + 1:
            current_run.append(numbers[i])
        else:
            if len(current_run) > max_run_len:
                max_run_len = len(current_run)
                longest_run = current_run[:]
            current_run = [numbers[i]]
    if len(current_run) > max_run_len:
        longest_run = current_run
        max_run_len = len(current_run)
    return max_run_len, longest_run



class TomogramDataset(Dataset):
    def __init__(self, tomo_dir, files, image_size):
        self.tomo_dir = tomo_dir
        self.files = files
        self.image_size = image_size

    def __getitem__(self, index):
        image_path = os.path.join(self.tomo_dir, self.files[index])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            image = np.array(Image.open(image_path).convert("L"), dtype=np.uint8)

        processed_image = self.preprocess(image)
        z_index = int(self.files[index].split('_')[1].split('.')[0])

        return *processed_image, z_index

    def __len__(self):
        return len(self.files)

    def preprocess(self, image):
        image, ratio, pad_w, pad_h = self.letterbox(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = image.transpose(2, 0, 1)
        image = np.ascontiguousarray(image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(image)
        return img_tensor, ratio, pad_w, pad_h

    def letterbox(self, image, color=(114, 114, 114), min_pad=0):
        new_w, new_h = self.image_size, self.image_size
        h, w = image.shape[:2]
        ratio = min(new_w / w, new_h / h)

        new_unpad_w, new_unpad_h = int(round(w * ratio) - 2 * min_pad), int(round(h * ratio) - 2 * min_pad)
        ratio = min(new_unpad_w / w, new_unpad_h / h)
        dw, dh = new_w - new_unpad_w, new_h - new_unpad_h

        dw /= 2
        dh /= 2

        image_resized = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
        image_resized = self.normalize(image_resized)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image_padded = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return image_padded, ratio, left, top

    def normalize(self, image):
        p2, p98 = np.percentile(image, [2, 98])
        clipped_data = np.clip(image, p2, p98)
        normalized = 255 * (clipped_data - p2) / (p98 - p2)
        return np.uint8(normalized)






def infer_batch(model, device_idx, img_tensor, ratios, paddings, conf_thres=CFG.confidence_threshold):
    img_tensor = img_tensor.to(device_idx)

    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.float16):
            outputs = model(img_tensor)

    outputs = outputs.cpu()
    batch_results = []

    for i, output in enumerate(outputs):
        ratio = float(ratios[i])
        pad_w = float(paddings[0][i])
        pad_h = float(paddings[1][i])

        if output.shape[0] == 5:
            # Format: [5, N]
            x_center, y_center, width, height, confidence = output

            mask = confidence > conf_thres
            x, y, w, h, conf = x_center[mask], y_center[mask], width[mask], height[mask], confidence[mask]

            x1 = (x - w / 2 - pad_w) / ratio
            y1 = (y - h / 2 - pad_h) / ratio
            x2 = (x + w / 2 - pad_w) / ratio
            y2 = (y + h / 2 - pad_h) / ratio

            boxes = np.stack((x1, y1, x2, y2), axis=1)
            batch_results.append((boxes, conf.numpy()))

        elif output.shape[1] == 6:
            # Format: [N, 6]
            boxes = output[:, :4].numpy()
            confidences = output[:, 4].numpy()
            mask = confidences > conf_thres

            boxes = boxes[mask]
            confidences = confidences[mask]

            x1 = (boxes[:, 0] - pad_w) / ratio
            y1 = (boxes[:, 1] - pad_h) / ratio
            x2 = (boxes[:, 2] - pad_w) / ratio
            y2 = (boxes[:, 3] - pad_h) / ratio

            boxes = np.stack((x1, y1, x2, y2), axis=1)
            batch_results.append((boxes, confidences))

        else:
            raise ValueError(f"Unknown output shape from model: {output.shape}")

    return batch_results



def process_tomogram(tomo_id, model_infos, device_idx):
    tomo_dir = os.path.join(CFG.test_image_path, tomo_id)
    slice_files = sorted([f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])

    # Group models by image size
    size_to_models = {}
    for model, img_size, fold_id in model_infos:
        size_to_models.setdefault(img_size, []).append((model, fold_id))

    all_detections = []

    for img_size, model_group in size_to_models.items():
        dataset = TomogramDataset(tomo_dir, slice_files, image_size=img_size)
        dataloader = DataLoader(dataset, batch_size=CFG.batch_sizes[device_idx], num_workers=os.cpu_count() // 2)

        for batch in tqdm(dataloader, tomo_id):
            images, ratios, paddings_w, paddings_h, indexes = batch

            for model, fold_id in model_group:
                results = infer_batch(model, CFG.devices[device_idx], images, ratios, (paddings_w, paddings_h), CFG.confidence_threshold)

                for i, (boxes, confs) in enumerate(results):
                    z = indexes[i].item()
                    for box, conf in zip(boxes, confs):
                        x1, y1, x2, y2 = box
                        all_detections.append({
                            'tomo_id': tomo_id,
                            'fold': fold_id,
                            'z': z,
                            'x_center': (x1 + x2) / 2,
                            'y_center': (y1 + y2) / 2,
                            'width': x2 - x1,
                            'height': y2 - y1,
                            'confidence': conf
                        })

    return all_detections



models = {}

# Assign global fold IDs for all models
model_infos = [
    (path, size, i)
    for i, (path, size) in enumerate(zip(CFG.model_paths, CFG.model_image_sizes))
]

for device in CFG.devices:
    models[device] = []
    for path, img_size, fold_id in model_infos:
        model = torch.jit.load(path, map_location=device).eval()
        models[device].append((model, img_size, fold_id))  # (model, image_size, fold_id)



%%time

all_predictions  = []
future_to_tomo = {}
test_tomos = sorted([d for d in os.listdir(CFG.test_image_path) if os.path.isdir(os.path.join(CFG.test_image_path, d))])

with ThreadPoolExecutor(max_workers=len(CFG.devices)) as executor:
    for i, tomo_id in enumerate(test_tomos):
        device_idx = i % len(CFG.devices)
        future = executor.submit(process_tomogram, tomo_id, models[CFG.devices[device_idx]], device_idx)

        future_to_tomo[future] = tomo_id
        


for future, tomo_id in future_to_tomo.items():
    try:
        dets = future.result()
        all_predictions.extend(dets)

    except Exception as e:
        print(f"[ERROR] Failed processing {tomo_id}: {e}")

        # Add a fallback detection with invalid (-1) values
        all_predictions.append({
            'tomo_id': tomo_id,
            'fold': -1,
            'z': -1,
            'x_center': -1,
            'y_center': -1,
            'width': 0,
            'height': 0,
            'confidence': 0.0
        })

    finally:
        torch.cuda.empty_cache()
        gc.collect()



all_predictions_df = pd.DataFrame(all_predictions)
all_predictions_df


def ensemble_tomogram_detections(preds, iou_thr=CFG.agreement_iou_threshold, conf_thr=CFG.confidence_threshold):
    if not preds:
        return {'tomo_id': '', 'Motor axis 0': -1, 'Motor axis 1': -1, 'Motor axis 2': -1}

    df = pd.DataFrame(preds)
    tomo_id = df['tomo_id'].iloc[0]
    
    # Step 1: get best box per fold per slice
    best = df.loc[df.groupby(['fold', 'z'])['confidence'].idxmax()].copy()

    # Step 2: compute corners
    best['x1'] = best['x_center'] - best['width'] / 2
    best['y1'] = best['y_center'] - best['height'] / 2
    best['x2'] = best['x_center'] + best['width'] / 2
    best['y2'] = best['y_center'] + best['height'] / 2

    # Step 3: IoU-based agreement per slice
    slice_groups = []
    for z, slice_df in best.groupby('z'):
        recs = slice_df.to_dict('records')
        passed = []
        for A in recs:
            for B in recs:
                if A is B:
                    continue
                if calculate_iou(A, B) >= iou_thr:
                    passed.append(A)
                    break

        if not passed:
            continue

        avg_conf = sum(d['confidence'] for d in passed) / len(CFG.model_paths)
        best_geo = max(passed, key=lambda d: d['confidence'])

        slice_groups.append({
            'z': int(z),
            'x': int(best_geo['x_center']),
            'y': int(best_geo['y_center']),
            'avg_conf': avg_conf
        })

    # Step 4: pick final slice
    passed_slices = [s for s in slice_groups if s['avg_conf'] >= conf_thr]
    if not passed_slices:
        return {'tomo_id': tomo_id, 'Motor axis 0': -1, 'Motor axis 1': -1, 'Motor axis 2': -1}

    best_slice = max(passed_slices, key=lambda s: s['avg_conf'])

    return {
        'tomo_id': tomo_id,
        'Motor axis 0': best_slice['z'],
        'Motor axis 1': best_slice['y'],
        'Motor axis 2': best_slice['x']
    }



ensemble_rows = []

# Get full list of test tomogram IDs
all_tomo_ids = test_tomos   # Or just use your top 3


# Group predictions for efficiency
grouped = all_predictions_df.groupby('tomo_id')

for tomo_id in all_tomo_ids:
    if tomo_id in grouped.groups:
        preds = grouped.get_group(tomo_id).to_dict('records')
    else:
        preds = []  # No predictions at all from any model

    result = ensemble_tomogram_detections(preds)
    result['tomo_id'] = tomo_id  # Ensure ID is preserved
    ensemble_rows.append(result)

submission_df = pd.DataFrame(ensemble_rows)
submission_df.to_csv("submission.csv", index=False)



all_predictions_df["tomo_id"].nunique()


from collections import defaultdict

debug_info = []

for tomo_id, df_t in all_predictions_df.groupby('tomo_id'):
    info = {'tomo_id': tomo_id}

    # Count predictions per model
    model_counts = df_t['fold'].value_counts().sort_index().to_dict()
    for model_id in range(len(CFG.model_paths)):
        info[f'model_{model_id}_count'] = model_counts.get(model_id, 0)

    # Highest confidence per model
    for model_id in range(len(CFG.model_paths)):
        highest = df_t[df_t['fold'] == model_id]['confidence'].max()
        info[f'model_{model_id}_max_conf'] = highest if not np.isnan(highest) else 0.0

    # Step 1: Best box per fold per slice
    best = df_t.loc[df_t.groupby(['fold', 'z'])['confidence'].idxmax()].copy()

    # Step 2: Compute corners
    best['x1'] = best['x_center'] - best['width'] / 2
    best['y1'] = best['y_center'] - best['height'] / 2
    best['x2'] = best['x_center'] + best['width'] / 2
    best['y2'] = best['y_center'] + best['height'] / 2

    slice_groups = []
    for z, slice_df in best.groupby('z'):
        recs = slice_df.to_dict('records')
        passed = []
        for A in recs:
            for B in recs:
                if A is B:
                    continue
                if calculate_iou(A, B) >= CFG.agreement_iou_threshold:
                    passed.append(A)
                    break
        if passed:
            avg_conf = sum(d['confidence'] for d in passed) / len(CFG.model_paths)
            best_geo = max(passed, key=lambda d: d['confidence'])
            slice_groups.append({
                'z': z,
                'avg_conf': avg_conf,
                'preds': passed,
                'x': best_geo['x_center'],
                'y': best_geo['y_center']
            })

    if slice_groups:
        best_slice = max(slice_groups, key=lambda s: s['avg_conf'])
        info['best_z'] = best_slice['z']
        info['best_avg_conf'] = round(best_slice['avg_conf'], 4)
        info['pred_count_in_best_z'] = len(best_slice['preds'])
        info['pred_confidences_in_best_z'] = [round(d['confidence'], 4) for d in best_slice['preds']]
        info['predicted_x'] = int(best_slice['x'])
        info['predicted_y'] = int(best_slice['y'])
    else:
        info['best_z'] = -1
        info['best_avg_conf'] = 0
        info['pred_count_in_best_z'] = 0
        info['pred_confidences_in_best_z'] = []
        info['predicted_x'] = -1
        info['predicted_y'] = -1

    debug_info.append(info)

debug_df = pd.DataFrame(debug_info)

# Print the first few rows
print(debug_df.head())




debug_df


submission_df


if len(submission_df) == 3:
    points = []
    image_paths = []

    for _, r in submission_df.iterrows():
        if (
            r['Motor axis 0'] != -1 and 
            r['Motor axis 1'] != -1 and 
            r['Motor axis 2'] != -1
        ):
            slice_str = f"{int(r['Motor axis 0']):04d}"
            image_path = f"/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test/{r['tomo_id']}/slice_{slice_str}.jpg"
            image_paths.append(image_path)
            points.append((r['Motor axis 2'], r['Motor axis 1']))

    fig, axes = plt.subplots(1, len(points), figsize=(5 * len(points), 5))
    box_size = 64
    half_box = box_size // 2

    for ax, (path, (x, y)) in zip(axes, zip(image_paths, points)):
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.scatter(x, y, color="red")

        rect = patches.Rectangle((x - half_box, y - half_box), box_size, box_size,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

        # Ensemble logic doesn't track a single confidence, so show a generic label
        ax.text(x, y - half_box - 10, "Ensemble", color='lime', fontsize=10, weight='bold', ha='center')
        ax.set_title(path.split("/")[-2] + "/" + path.split("/")[-1])
        ax.axis('off')

    plt.tight_layout()
    plt.show()






































