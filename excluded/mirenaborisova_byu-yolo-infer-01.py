!tar xfvz /kaggle/input/ultralytics-for-offline-install/archive.tar.gz
!pip install --no-index --find-links=./packages ultralytics
!rm -rf ./packages


import numpy as np
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


data_path = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/'
test_path = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test'
sample_submission_path = '/kaggle/working/submission.csv'

model_path = '/kaggle/input/byu-yolo-train-01/yolo_weights/flagellar_motor_detector/weights/best.pt'


def normalize_slice(slice_data):
    
    percentile_2th = np.percentile(slice_data, 2)
    percentile_98th = np.percentile(slice_data, 98)
    clipped_data = np.clip(slice_data, percentile_2th, percentile_98th)
    normalized = 255 * (clipped_data - percentile_2th) / (percentile_98th - percentile_2th)
    
    return np.uint8(normalized)


BOX_SIZE = 24
NMS_IOU_THRESHOLD = 0.2
DISTANCE_THRESHOLD = BOX_SIZE * NMS_IOU_THRESHOLD

def perform_3d_nms(detections):
    
    if not detections:
        return []

    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    final_detections = []
    
    def distance_3d(distance_1, distance_2):
        return np.sqrt((distance_1['z'] - distance_2['z']) ** 2 + 
                       (distance_1['y'] - distance_2['y']) ** 2 + 
                       (distance_1['x'] - distance_2['x']) ** 2)
    
    while detections:
        
        best_detection = detections.pop(0)
        final_detections.append(best_detection)
        
        detections = [distance for distance in detections if distance_3d(distance, best_detection) > DISTANCE_THRESHOLD]
    
    return final_detections


import os

CONFIDENCE_THRESHOLD = 0.45

def generalization(tomo_id, model):
    
    tomo_id_path = os.path.join(test_path, tomo_id)
    slice_jpg_files = sorted([f for f in os.listdir(tomo_id_path) if f.endswith('.jpg')])
    
    detections = []
    
    if len(slice_jpg_files) == 0:
        pass

    slice_jpg_files_paths = [os.path.join(tomo_id_path, slice_jpg_file) \
                             for slice_jpg_file in slice_jpg_files]
    slice_jpg_files_nums = [int(slice_jpg_file.split('_')[1].split('.')[0]) \
                            for slice_jpg_file in slice_jpg_files]
    
    sub_results = model(slice_jpg_files_paths, verbose=False)
        
    for i, result in enumerate(sub_results):
        if len(result.boxes) > 0:
            boxes = result.boxes
            for box_id, confidence in enumerate(boxes.conf):
                if confidence >= CONFIDENCE_THRESHOLD:
                    
                    x_start, y_start, x_end, y_end = boxes.xyxy[box_id].cpu().numpy()
                    
                    x_center = (x_start + x_end) / 2
                    y_center = (y_start + y_end) / 2
                    
                    detections.append({
                        'z': round(slice_jpg_files_nums[i]),
                        'y': round(y_center),
                        'x': round(x_center),
                        'confidence': float(confidence)
                    })
                            
        
    final_detections = perform_3d_nms(detections)
    
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


from ultralytics import YOLO
import pandas as pd

test_tomos = sorted(
    [test_tomo_id for test_tomo_id in os.listdir(test_path) \
     if os.path.isdir(os.path.join(test_path, test_tomo_id))])
    
model = YOLO(model_path)

predictions = []

for i, tomo_id in enumerate(test_tomos):
    prediction = generalization(tomo_id, model)
    predictions.append(prediction)
    
submission_df = pd.DataFrame(predictions)
submission_df = submission_df[['tomo_id', 'Motor axis 0', 'Motor axis 1', 'Motor axis 2']]
submission_df.to_csv(sample_submission_path, index=False)

submission_df




