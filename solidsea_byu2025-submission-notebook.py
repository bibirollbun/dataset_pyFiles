!tar xfvz /kaggle/input/ultralytics-for-offline-install/archive.tar.gz
!pip install --no-index --find-links=./packages ultralytics
!rm -rf ./packages


import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import cv2
from tqdm.notebook import tqdm
from ultralytics import YOLO
import threading
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Lock

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define paths
data_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/"
test_dir = os.path.join(data_path, "test")
submission_path = "/kaggle/working/submission.csv"

# Model path
model_path = "/kaggle/input/byu-2025/pytorch/default/22/yolov5xud.pt"

# Detection parameters
CONFIDENCE_THRESHOLD = 0.2
MAX_DETECTIONS_PER_TOMO = 1
NMS_IOU_THRESHOLD = 0.4
CONCENTRATION = 20
TARGET_SIZE = 1280  # Model's expected input size divisible by 32

# GPU settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"Found {num_gpus} GPU(s) available")

# Dynamic batch sizing
BATCH_SIZE = 8  # Fixed smaller batch size for varying input sizes
print(f"Using batch size: {BATCH_SIZE}")

class ImageLoader:
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = Lock()
        
    def load_and_resize(self, path):
        """Load image and resize to target size maintaining aspect ratio"""
        # Load as RGB
        img = cv2.imread(path)
        if img is None:
            img = np.array(Image.open(path).convert('RGB'))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize maintaining aspect ratio
        h, w = img.shape[:2]
        scale = min(TARGET_SIZE / h, TARGET_SIZE / w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        
        # Pad to target size
        top = (TARGET_SIZE - new_h) // 2
        bottom = TARGET_SIZE - new_h - top
        left = (TARGET_SIZE - new_w) // 2
        right = TARGET_SIZE - new_w - left
        img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        return img, (scale, (left, top))  # Return scaling and padding info

def process_tomogram(tomo_id, model, index=0, total=1, gpu_id=0):
    """Process tomogram with proper image resizing and scaling"""
    torch.cuda.set_device(gpu_id)
    print(f"Processing {tomo_id} ({index}/{total}) on GPU {gpu_id}")
    
    tomo_dir = os.path.join(test_dir, tomo_id)
    slice_files = sorted([f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])
    
    # Apply CONCENTRATION
    num_slices = len(slice_files)
    step = max(1, int(1 / CONCENTRATION * num_slices))
    selected_indices = range(0, num_slices, step)
    slice_files = [slice_files[i] for i in selected_indices if i < num_slices]
    
    print(f"Processing {len(slice_files)}/{num_slices} slices")
    
    all_detections = []
    loader = ImageLoader()
    
    # Process slices one by one due to varying sizes
    for slice_file in slice_files:
        slice_path = os.path.join(tomo_dir, slice_file)
        slice_num = int(slice_file.split('_')[1].split('.')[0])
        
        try:
            # Load and preprocess image
            img, (scale, (left_pad, top_pad)) = loader.load_and_resize(slice_path)
            
            # Convert to tensor
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                results = model(img_tensor)
            
            # Process results
            for result in results:
                if len(result.boxes) > 0:
                    boxes = result.boxes
                    for box_idx, confidence in enumerate(boxes.conf):
                        if confidence >= CONFIDENCE_THRESHOLD:
                            # Adjust coordinates back to original image space
                            x1, y1, x2, y2 = boxes.xyxy[box_idx].cpu().numpy()
                            
                            # Remove padding and scale back
                            x1 = (x1 - left_pad) / scale
                            y1 = (y1 - top_pad) / scale
                            x2 = (x2 - left_pad) / scale
                            y2 = (y2 - top_pad) / scale
                            
                            # Calculate center
                            x_center = (x1 + x2) / 2
                            y_center = (y1 + y2) / 2
                            
                            all_detections.append({
                                'tomo_id': tomo_id,
                                'z': slice_num,
                                'y': round(y_center),
                                'x': round(x_center),
                                'confidence': float(confidence)
                            })
        except Exception as e:
            print(f"Error processing {slice_file}: {e}")
            continue
    
    # If no detections, return default values
    if not all_detections:
        return {
            'tomo_id': tomo_id,
            'Motor axis 0': -1,
            'Motor axis 1': -1,
            'Motor axis 2': -1
        }
    
    # Find best detection
    best_detection = max(all_detections, key=lambda x: x['confidence'])
    return {
        'tomo_id': tomo_id,
        'Motor axis 0': best_detection['z'],
        'Motor axis 1': best_detection['y'],
        'Motor axis 2': best_detection['x']
    }

def initialize_models():
    """Initialize models with proper settings"""
    models = []
    if num_gpus > 0:
        for i in range(num_gpus):
            print(f"Initializing model on GPU {i}")
            torch.cuda.set_device(i)
            model = YOLO(model_path)
            model.to(f'cuda:{i}')
            model.fuse()
            model.model.eval()
            
            # Warmup
            dummy_input = torch.randn(1, 3, TARGET_SIZE, TARGET_SIZE, 
                                    device=f'cuda:{i}', dtype=torch.float32)
            with torch.no_grad():
                _ = model(dummy_input)
            
            models.append(model)
            torch.cuda.empty_cache()
    else:
        print("Initializing model on CPU")
        model = YOLO(model_path)
        model.to('cpu')
        model.model.eval()
        models.append(model)
    
    return models

def generate_submission():
    """Generate submission with error handling"""
    test_tomos = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    total_tomos = len(test_tomos)
    print(f"Found {total_tomos} tomograms")
    
    models = initialize_models()
    results = []
    
    with ThreadPoolExecutor(max_workers=num_gpus if num_gpus > 0 else 1) as executor:
        futures = []
        for idx, tomo_id in enumerate(test_tomos, 1):
            gpu_id = (idx-1) % num_gpus if num_gpus > 0 else 0
            futures.append(executor.submit(
                process_tomogram, tomo_id, models[gpu_id], idx, total_tomos, gpu_id
            ))
        
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing tomogram: {e}")
                results.append({
                    'tomo_id': tomo_id,
                    'Motor axis 0': -1,
                    'Motor axis 1': -1,
                    'Motor axis 2': -1
                })
    
    # Ensure consistent output format
    submission_data = []
    for res in results:
        submission_data.append({
            'tomo_id': res['tomo_id'],
            'Motor axis 0': res.get('Motor axis 0', -1),
            'Motor axis 1': res.get('Motor axis 1', -1),
            'Motor axis 2': res.get('Motor axis 2', -1)
        })
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    return submission_df

if __name__ == "__main__":
    start_time = time.time()
    submission = generate_submission()
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.2f}s ({elapsed/60:.2f}min)")


