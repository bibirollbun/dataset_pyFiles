!pip install ultralytics > /dev/null


import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from ultralytics import YOLO


data_yaml = """
train: /kaggle/input/multi-class-object-detection-challenge/Starter_Dataset/train/images
val: /kaggle/input/multi-class-object-detection-challenge/Starter_Dataset/val/images

nc: 2
names: ['cheerios', 'soup']
"""

with open('/kaggle/working/data.yaml', 'w') as file:
    file.write(data_yaml)


model = YOLO("yolo12m.pt")
data_yaml = '/kaggle/working/data.yaml'

results = model.train(
    data=data_yaml,
    pretrained=True,
    epochs=150,
    batch=8,
    imgsz=960,
    device=[0, 1],
    patience=20,
    lr0=0.0001,
    lrf=0.02,
    optimizer="Adam",
    weight_decay=0.0004,
    cos_lr=True,
    dropout=0.3,
    label_smoothing=0.01,
    mosaic=0.5,
    mixup=0.15,
    copy_paste=0.1,
    fliplr=0.5,
    flipud=0.4,
    hsv_h=0.02,
    hsv_s=0.2,
    hsv_v=0.4,
    translate=0.2,
    scale=0.5,
    shear=0.2,
    perspective=0.007,
    val=True,
    workers=8,
    seed=6
)


img_path = '/kaggle/input/multi-class-object-detection-challenge/testImages/images/IMG_8656.png'
model = YOLO('/kaggle/working/runs/detect/train/weights/best.pt')
results = model(img_path)


result_img = results[0].plot()
plt.imshow(result_img)
plt.axis('off')
plt.show()


test_images_dir = '/kaggle/input/multi-class-object-detection-challenge/testImages/images'
image_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png'))]

results_list = []

for img_name in image_files:
    img_path = os.path.join(test_images_dir, img_name)

    with Image.open(img_path) as img:
        width, height = img.size

    results = model.predict(img_path, conf=0.0001, device='0', verbose=False)
    boxes = results[0].boxes
    
    if boxes is not None and len(boxes) > 0:
        prediction_strings = []
        for box in boxes:
            box_data = box.xywh[0].cpu().numpy()
            
            cls = int(box.cls.item())
            conf = box.conf.item()

            x_center = box_data[0] / width
            y_center = box_data[1] / height
            w = box_data[2] / width
            h = box_data[3] / height
            
            prediction_strings.append(f"{cls} {conf:.6f} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        
        prediction_str = " ".join(prediction_strings)
    else:
        prediction_str = "no boxes"
    
    results_list.append({
        "image_id": os.path.splitext(img_name)[0],
        "prediction_string": prediction_str
    })


submission = pd.DataFrame(results_list)
submission.to_csv('submission.csv', index=False)
submission.sample(15)

