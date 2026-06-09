# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install ultralytics supervision -q


import os
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm
import zipfile


OUTPUT_PATH = "/kaggle/working"
MODEL_NAME = "yolov8x.pt"
CONFIDENCE_THRESHOLD = 0.3


os.makedirs(f"{OUTPUT_PATH}/train/images", exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/train/labels", exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/test/images", exist_ok=True)


def prepare_yolo_dataset():
    yaml_content = f"""path: /kaggle/input/cars-object-detection/train/train
train: /kaggle/input/cars-object-detection/train/train/images
val: /kaggle/input/cars-object-detection/train/train/images
nc: 1
names: ['car']
"""
    with open(f"{OUTPUT_PATH}/data.yaml", "w") as f:
        f.write(yaml_content)
    print("\nСодержимое data.yaml:")
    with open(f"{OUTPUT_PATH}/data.yaml", "r") as f:
        print(f.read())


prepare_yolo_dataset()


def train_model():
    model = YOLO(MODEL_NAME)
    
    # Гиперпараметры с аугментациями
    results = model.train(
        data=f"{OUTPUT_PATH}/data.yaml",
        epochs=50,
        batch=16,
        imgsz=640,
        device=0,
        patience=10,
        verbose=True,
        # Дополнительные аугментации
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0001,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,
        close_mosaic=10,  # Отключить мозаику последние 10 эпох
        overlap_mask=True,
        single_cls=True   # Только один класс
    )
    return model


model = train_model()


def generate_predictions(model):
    test_images_path = "/kaggle/input/cars-object-detection/test/test/images"
    test_images = [f for f in os.listdir(test_images_path) if f.endswith('.JPG')]
    predictions = []
    
    for image_file in tqdm(test_images, desc="Обработка тестовых изображений"):
        image_path = f"{test_images_path}/{image_file}"
        image = cv2.imread(image_path)
        
        results = model.predict(image, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        preds = []
        for detection in detections:
            # Новый способ распаковки для YOLOv8
            bbox = detection[0]  # [x1, y1, x2, y2]
            confidence = detection[2]  # confidence score
            class_id = detection[3]  # class ID
            
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + width/2) / image.shape[1]
            center_y = (y1 + height/2) / image.shape[0]
            width /= image.shape[1]
            height /= image.shape[0]
            
            preds.extend([0, float(confidence), float(center_x), float(center_y), float(width), float(height)])
        
        image_id = os.path.splitext(image_file)[0]
        predictions.append({
            "id": image_id,
            "predictions": f"[{','.join(map(str, preds))}]" if preds else "[]"
        })
    
    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv(f"{OUTPUT_PATH}/submission.csv", index=False)


generate_predictions(model)




