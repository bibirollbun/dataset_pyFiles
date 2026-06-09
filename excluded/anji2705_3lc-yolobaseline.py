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


!pip -q install ultralytics opencv-python-headless


import os
import cv2
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt


data_yaml = """
train: /kaggle/input/multi-class-object-detection-challenge/Dataset/train/images
val: /kaggle/input/multi-class-object-detection-challenge/Dataset/val/images

nc : 2
names : ['cheerios','soup']
"""


with open('/kaggle/working/data.yaml', 'w') as f:
    f.write(data_yaml)


model = YOLO('yolov8n.pt')


results = model.train(
    data = '/kaggle/working/data.yaml',
    epochs = 50,
    imgsz = 640,
    batch = 8,
    patience = 8,
    device = 0,
    optimizer = 'Adam',
    lr0 = 1e-4,
    weight_decay=0.0004,
    mosaic=0.5,
    mixup=0.1,
    fliplr=0.5,
    flipud=0.2,
    translate=0.1,
    scale=0.4,
    shear=0.2,
    perspective=0.001,
    val=True,
    seed=42
    
)


model = YOLO('/kaggle/working/runs/detect/train/weights/best.pt')


test_images_dir = '/kaggle/input/multi-class-object-detection-challenge/testImages/images'
image_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png'))]

results_list = []


for img_name in image_files:
    img_path = os.path.join(test_images_dir, img_name)
    
    with Image.open(img_path) as img:
        width, height = img.size

    results = model.predict(img_path, device=0, verbose=False)
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        prediction_parts = []
        for box in boxes:
            x_center, y_center, w, h = box.xywh[0].cpu().numpy()
            cls = int(box.cls.item())
            conf = float(box.conf.item())

            # Normalize bbox values
            x_center /= width
            y_center /= height
            w /= width
            h /= height

            prediction_parts.append(f"{cls} {conf:.6f} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        prediction_string = ' '.join(prediction_parts)
    else:
        prediction_string = 'no boxes'

    results_list.append({
        "image_id": os.path.splitext(img_name)[0],
        "prediction_string": prediction_string
    })



submission = pd.DataFrame(results_list)
submission.to_csv("submission09.csv", index=False)


submission.sample(5)




