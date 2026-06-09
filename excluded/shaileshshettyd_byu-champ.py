!pip install ultralytics


# Import & Setup

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from ultralytics import YOLO
import torchvision.ops as ops
import cv2
import threading
from concurrent.futures import ThreadPoolExecutor

# Enable GPU acceleration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Set confidence & IOU thresholds
CONFIDENCE_THRESHOLD = 0.45
NMS_IOU_THRESHOLD = 0.2

# Define paths
DATA_PATH = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/"
TEST_DIR = os.path.join(DATA_PATH, "test")


def preprocess_image(img_path):
    """ Load and normalize image for inference """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.array(Image.open(img_path))
    return img

def load_images_in_parallel(image_paths):
    """ Load batch of images in parallel for efficiency """
    images = []
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(preprocess_image, image_paths))
    return images


def run_inference(batch_images):
    """ Run YOLO inference asynchronously on a batch of images """
    with torch.no_grad():
        results = model(batch_images, verbose=False)
    return results


def perform_3d_nms(detections, iou_threshold=0.2):
    """ Efficient 3D NMS using PyTorch's native ops.nms() """
    if not detections:
        return []
    
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    boxes = torch.tensor([[d['x'], d['y'], d['z'], d['confidence']] for d in detections])
    scores = boxes[:, -1]
    
    keep_indices = ops.nms(boxes[:, :3], scores, iou_threshold)
    return [detections[i] for i in keep_indices.tolist()]


def process_tomogram(tomo_id):
    """ Process a tomogram, detect motors, and return best results """
    slice_dir = os.path.join(TEST_DIR, tomo_id)
    slice_files = sorted([f for f in os.listdir(slice_dir) if f.endswith('.jpg')])
    
    detections = []
    for i in range(0, len(slice_files), 8):  # Dynamic batch size
        batch_files = slice_files[i:i+8]
        batch_paths = [os.path.join(slice_dir, f) for f in batch_files]
        
        batch_images = load_images_in_parallel(batch_paths)
        results = run_inference(batch_images)
        
        for idx, result in enumerate(results):
            for box in result.boxes:
                detections.append({
                    'z': i + idx,
                    'x': (box.xyxy[0] + box.xyxy[2]) / 2,
                    'y': (box.xyxy[1] + box.xyxy[3]) / 2,
                    'confidence': box.conf
                })
    
    detections = perform_3d_nms(detections, NMS_IOU_THRESHOLD)
    
    if not detections:
        return {'tomo_id': tomo_id, 'Motor axis 0': -1, 'Motor axis 1': -1, 'Motor axis 2': -1}
    
    best_detection = detections[0]
    return {
        'tomo_id': tomo_id,
        'Motor axis 0': int(best_detection['z']),
        'Motor axis 1': int(best_detection['y']),
        'Motor axis 2': int(best_detection['x'])
    }


def generate_submission():
    """ Run inference on all tomograms and generate submission file """
    test_tomos = sorted([d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))])
    results = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(process_tomogram, test_tomos), total=len(test_tomos), desc="Processing Tomograms"))
    
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"✅ Submission saved to {SUBMISSION_PATH}")

