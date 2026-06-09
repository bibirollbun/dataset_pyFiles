!tar xfvz /kaggle/input/ultralytics-for-offline-install/archive.tar.gz
!pip install --no-index --find-links=./packages ultralytics
!rm -rf ./packages


import numpy as np
import torch

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


test_dir = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test'
submission_path = '/kaggle/working/submission.csv'
model_path = \
    '/kaggle/input/byu-yolo-train-02/yolo_weights/flagellar_motor_detector/weights/best.pt'


import os
from PIL import Image
import cv2
from ultralytics import YOLO
import threading
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor

CONFIDENCE_THRESHOLD = 0.45
NMS_IOU_THRESHOLD = 0.2
BATCH_SIZE = 8
    
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9

free_mem = gpu_mem - torch.cuda.memory_allocated(0) / 1e9
BATCH_SIZE = max(8, min(32, int(free_mem * 4)))


def normalize_slice(slice_data):
    
    p2 = np.percentile(slice_data, 2)
    p98 = np.percentile(slice_data, 98)
    clipped_data = np.clip(slice_data, p2, p98)
    normalized = 255 * (clipped_data - p2) / (p98 - p2)
    return np.uint8(normalized)

def preload_image_batch(file_paths):
    
    images = []
    for path in file_paths:
        img = cv2.imread(path)
        if img is None:
            img = np.array(Image.open(path))
        images.append(img)
    return images

def perform_3d_nms(detections, iou_threshold):
    
    if not detections:
        return []
    
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    final_detections = []
    
    def distance_3d(d1, d2):
        return np.sqrt((d1['z'] - d2['z'])**2 + 
                       (d1['y'] - d2['y'])**2 + 
                       (d1['x'] - d2['x'])**2)

    
    box_size = 24
    distance_threshold = box_size * iou_threshold
    
    while detections:
        
        best_detection = detections.pop(0)
        final_detections.append(best_detection)
        
        detections = [d for d in detections if distance_3d(d, best_detection) > distance_threshold]
    
    return final_detections


def process_tomogram(tomo_id, model):
    
    tomo_dir = os.path.join(test_dir, tomo_id)
    slice_files = sorted([f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])
    
    all_detections = []
    
    streams = [torch.cuda.Stream() for _ in range(min(4, BATCH_SIZE))]
    
    next_batch_thread = None
    
    for batch_start in range(0, len(slice_files), BATCH_SIZE):
        
        if next_batch_thread is not None:
            next_batch_thread.join()
            
        batch_end = min(batch_start + BATCH_SIZE, len(slice_files))
        batch_files = slice_files[batch_start:batch_end]
        
        next_batch_start = batch_end
        next_batch_end = min(next_batch_start + BATCH_SIZE, len(slice_files))
        next_batch_files = slice_files[next_batch_start:next_batch_end] if next_batch_start < len(slice_files) else []
        
        if next_batch_files:
            next_batch_paths = [os.path.join(tomo_dir, f) for f in next_batch_files]
            next_batch_thread = threading.Thread(target=preload_image_batch, args=(next_batch_paths,))
            next_batch_thread.start()
        else:
            next_batch_thread = None
            
        sub_batches = np.array_split(batch_files, len(streams))
        sub_batch_results = []
        
        for i, sub_batch in enumerate(sub_batches):
            if len(sub_batch) == 0:
                continue
                
            stream = streams[i % len(streams)]
            with torch.cuda.stream(stream) if stream else nullcontext():
                
                sub_batch_paths = [os.path.join(tomo_dir, slice_file) for slice_file in sub_batch]
                sub_batch_slice_nums = [int(slice_file.split('_')[1].split('.')[0]) for slice_file in sub_batch]
                sub_results = model(sub_batch_paths, verbose=False)
                    
                for j, result in enumerate(sub_results):
                    if len(result.boxes) > 0:
                        boxes = result.boxes
                        for box_idx, confidence in enumerate(boxes.conf):
                            if confidence >= CONFIDENCE_THRESHOLD:
                                
                                x1, y1, x2, y2 = boxes.xyxy[box_idx].cpu().numpy()
                                
                                x_center = (x1 + x2) / 2
                                y_center = (y1 + y2) / 2
                                
                                all_detections.append({
                                    'z': round(sub_batch_slice_nums[j]),
                                    'y': round(y_center),
                                    'x': round(x_center),
                                    'confidence': float(confidence)
                                })
                                
        torch.cuda.synchronize()
        
    if next_batch_thread is not None:
        next_batch_thread.join()
        
    final_detections = perform_3d_nms(all_detections, NMS_IOU_THRESHOLD)
    
    final_detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    if not final_detections:
        return {
            'tomo_id': tomo_id,
            'Motor axis 0': -1,
            'Motor axis 1': -1,
            'Motor axis 2': -1
        }
    best_detection = final_detections[0]
    
    return {
        'tomo_id': tomo_id,
        'Motor axis 0': round(best_detection['z']),
        'Motor axis 1': round(best_detection['y']),
        'Motor axis 2': round(best_detection['x'])
    }


import pandas as pd

test_tomos = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])

torch.cuda.empty_cache()

model = YOLO(model_path)
model.to(device)
model.fuse()

if torch.cuda.get_device_capability(0)[0] >= 7:
    model.model.half()
    
results = []
motors_found = 0

with ThreadPoolExecutor(max_workers=1) as executor:
    future_to_tomo = {}
    
    for i, tomo_id in enumerate(test_tomos, 1):
        future = executor.submit(process_tomogram, tomo_id, model)
        future_to_tomo[future] = tomo_id
        
    for future in future_to_tomo:
        tomo_id = future_to_tomo[future]
        try:
            torch.cuda.empty_cache()
                
            result = future.result()
            results.append(result)
        
        except Exception as e:
            results.append({
                'tomo_id': tomo_id,
                'Motor axis 0': -1,
                'Motor axis 1': -1,
                'Motor axis 2': -1
            })
            
submission_df = pd.DataFrame(results)
submission_df = submission_df[['tomo_id', 'Motor axis 0', 'Motor axis 1', 'Motor axis 2']]
submission_df.to_csv(submission_path, index=False)

submission_df

