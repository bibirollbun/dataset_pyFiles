import os
import re
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil
from glob import glob


!tar xzvf /kaggle/input/ultralytics-for-offline-install/archive.tar.gz




!pip install --no-index --no-deps --find-links=./packages ultralytics


from ultralytics import YOLO
model = YOLO('/kaggle/input/global-wheat-detection-yolov11n-v2/pytorch/default/1/custom_yolo (1).pt')


test_dir = "/kaggle/input/global-wheat-detection/test"
test_images = sorted(glob(os.path.join(test_dir, "*.jpg")))

results = []

for image_path in test_images:
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    preds = model.predict(source=image_path, conf=0.5, iou=0.5, save=False, verbose=False)
    boxes = preds[0].boxes
    prediction_string = ""
    for xyxy, score in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
        x_min, y_min, x_max, y_max = xyxy
        width = x_max - x_min
        height = y_max - y_min
        prediction_string += f"{score:.4f} {int(x_min)} {int(y_min)} {int(width)} {int(height)} "

    results.append({
        "image_id": image_id,
        "PredictionString": prediction_string.strip()
    })

print(results[:2])


submission_df = pd.DataFrame(results)
print(submission_df.head())
submission_df.to_csv("submission.csv", index=False)







