!pip install ultralytics > /dev/null


import os
import cv2
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from ultralytics import YOLO


train_images = Path('/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/train/images')
train_masks = Path('/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/train/labels')

test_images = Path("/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/testImages/images")


output_dir = Path("/kaggle/working/predictions/labels")
output_dir.mkdir(parents=True, exist_ok=True)


def visualize_random_masks(images_dir, masks_dir, counts=5):
    """
    Ğ’Ğ¸Ğ·ÑƒĞ°Ğ»Ğ¸Ğ·Ğ¸Ñ€ÑƒĞµÑ‚ Ñ�Ğ»ÑƒÑ‡Ğ°Ğ¹Ğ½Ñ‹Ğµ Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ñ� Ñ� Ğ½Ğ°Ğ»Ğ¾Ğ¶ĞµĞ½Ğ½Ñ‹Ğ¼Ğ¸ Ğ¼Ğ°Ñ�ĞºĞ°Ğ¼Ğ¸.
    Vizualize random images with masks.
    
    images_dir: Ğ¿ÑƒÑ‚ÑŒ Ğº Ğ´Ğ¸Ñ€ĞµĞºÑ‚Ğ¾Ñ€Ğ¸Ğ¸ Ñ� Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ñ�Ğ¼Ğ¸
    images_dir: path to directory with images
    
    masks_dir: Ğ¿ÑƒÑ‚ÑŒ Ğº Ğ´Ğ¸Ñ€ĞµĞºÑ‚Ğ¾Ñ€Ğ¸Ğ¸ Ñ� Ğ¼Ğ°Ñ�ĞºĞ°Ğ¼Ğ¸
    masks_dir: path to directory with masks
    
    counts: ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ğ¹ Ğ´Ğ»Ñ� Ğ²Ñ‹Ğ²Ğ¾Ğ´Ğ°
    counts: Numbers of images to show
    """

    image_paths = list(images_dir.glob("*"))
    samples = np.random.choice(image_paths, counts, replace=False)

    plt.figure(figsize=(counts * 5, 8))

    for i, img_path in enumerate(samples, 1):
        mask_path = masks_dir / img_path.with_suffix(".txt").name

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        with open(mask_path, "r") as f:
            lines = f.readlines()
    
        for line in lines:
            parts = line.strip().split()

            try:
                class_id, x_center, y_center, box_w, box_h = map(float, parts)
            except:
                class_id, confidence, x_center, y_center, box_w, box_h = map(float, parts)

            x_center *= w
            y_center *= h
            box_w *= w
            box_h *= h
    
            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 5)

        plt.subplot(1, counts, i)
        plt.imshow(image)
        plt.axis("off")
        plt.tight_layout()


visualize_random_masks(train_images, train_masks)


data_yaml = """
path: /kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2

train: train/images
val: val/images
test: testImages/images

nc: 1
names: ['object']
"""

with open('data.yaml', 'w') as file:
    file.write(data_yaml)


model = YOLO("yolo11x.pt")

results = model.train(
    data="data.yaml",         # Ğ¿ÑƒÑ‚ÑŒ Ğº Ğ´Ğ°Ğ½Ğ½Ñ‹Ğ¼ / data path
    epochs=100,               # Ñ�Ğ¿Ğ¾Ñ…Ğ¸ (ĞºĞ¾Ğ»-Ğ²Ğ¾ Ğ¿Ñ€Ğ¾Ñ…Ğ¾Ğ´Ğ¾Ğ²) / epochs (training passes)
    batch=16,                 # Ñ€Ğ°Ğ·Ğ¼ĞµÑ€ Ğ±Ğ°Ñ‚Ñ‡Ğ° / batch size
    imgsz=640,                # Ñ€Ğ°Ğ·Ğ¼ĞµÑ€ Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ñ� / image size
    device=[0,1],             # "cpu", 0 - GPU, [0, 1] - 2 GPUs
    patience=10,              # Ñ€Ğ°Ğ½Ğ½Ñ�Ñ� Ğ¾Ñ�Ñ‚Ğ°Ğ½Ğ¾Ğ²ĞºĞ° (ĞµÑ�Ğ»Ğ¸ Ğ½ĞµÑ‚ ÑƒĞ»ÑƒÑ‡ÑˆĞµĞ½Ğ¸Ğ¹) / early stopping patience

    lr0=0.0001,               # Ğ½Ğ°Ñ‡Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¹ lr / initial learning rate
    lrf=0.01,                 # ĞºĞ¾Ğ½ĞµÑ‡Ğ½Ñ‹Ğ¹ lr / result learning rate
    optimizer="SGD",          # Ğ¾Ğ¿Ñ‚Ğ¸Ğ¼Ğ¸Ğ·Ğ°Ñ‚Ğ¾Ñ€ / optimizer
    momentum=0.87,            # Ğ¼Ğ¾Ğ¼ĞµĞ½Ñ‚ÑƒĞ¼ Ğ´Ğ»Ñ� SGD / momentum for SGD
    weight_decay=0.001,       # Ñ€ĞµĞ³ÑƒĞ»Ñ�Ñ€Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ� Ğ²ĞµÑ�Ğ° / weight decay (L2 regularization)
    cos_lr=True,              # ĞºĞ¾Ñ�Ğ¸Ğ½ÑƒÑ�Ğ½Ñ‹Ğ¹ lr / cosine learning rate

    dropout=0.3,              # Ñ€ĞµĞ³ÑƒĞ»Ñ�Ñ€Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ� Ğ´Ñ€Ğ¾Ğ¿Ğ°ÑƒÑ‚ / dropout regularization
    label_smoothing=0.1,      # Ğ¿Ğ¾Ğ¼Ğ¾Ğ³Ğ°ĞµÑ‚ Ğ¿Ñ€Ğ¸ ÑˆÑƒĞ¼Ğ½Ñ‹Ñ… Ğ¸Ğ»Ğ¸ Ñ�Ğ¸Ğ½Ñ‚ĞµÑ‚Ğ¸Ñ‡ĞµÑ�ĞºĞ¸Ñ… Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ… / helps with noisy or synthetic data

    mosaic=0.65,              # Ğ°ÑƒĞ³Ğ¼ĞµĞ½Ñ‚Ğ°Ñ†Ğ¸Ñ� Ğ¼Ğ¾Ğ·Ğ°Ğ¸ĞºĞ° / mosaic augmentation
    mixup=0.15,               # Ñ�Ğ¼ĞµÑˆĞ¸Ğ²Ğ°Ğ½Ğ¸Ğµ Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ğ¹ / mixing images
    copy_paste=0.1,           # Ğ²Ñ�Ñ‚Ğ°Ğ²ĞºĞ° Ğ¾Ğ±ÑŠĞµĞºÑ‚Ğ¾Ğ² Ñ� Ğ´Ñ€ÑƒĞ³Ğ¸Ñ… Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ğ¹ / inserting objects from other images

    fliplr=0.5,               # Ğ²ĞµÑ€Ğ¾Ñ�Ñ‚Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ³Ğ¾Ñ€Ğ¸Ğ·Ğ¾Ğ½Ñ‚Ğ°Ğ»ÑŒĞ½Ğ¾Ğ³Ğ¾ Ğ¾Ñ‚Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ñ� / horizontal flip probability
    flipud=0.5,               # Ğ²ĞµÑ€Ğ¾Ñ�Ñ‚Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ²ĞµÑ€Ñ‚Ğ¸ĞºĞ°Ğ»ÑŒĞ½Ğ¾Ğ³Ğ¾ Ğ¾Ñ‚Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ñ� / vertical flip probability
    hsv_h=0.015,              # Ğ¸Ğ·Ğ¼ĞµĞ½ĞµĞ½Ğ¸Ğµ Ğ¾Ñ‚Ñ‚ĞµĞ½ĞºĞ° (Hue) / hue change
    hsv_s=0.4,                # Ğ¸Ğ·Ğ¼ĞµĞ½ĞµĞ½Ğ¸Ğµ Ğ½Ğ°Ñ�Ñ‹Ñ‰ĞµĞ½Ğ½Ğ¾Ñ�Ñ‚Ğ¸ (Saturation) / saturation change
    hsv_v=0.4,                # Ğ¸Ğ·Ğ¼ĞµĞ½ĞµĞ½Ğ¸Ğµ Ñ�Ñ€ĞºĞ¾Ñ�Ñ‚Ğ¸ (Value) / value change (brightness)
    translate=0.2,            # Ñ�Ğ´Ğ²Ğ¸Ğ³ Ğ¾Ğ±ÑŠĞµĞºÑ‚Ğ° / translation
    scale=0.5,                # Ğ¼Ğ°Ñ�ÑˆÑ‚Ğ°Ğ±Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ / scaling
    shear=0.2,                # Ğ½Ğ°ĞºĞ»Ğ¾Ğ½ / shear
    perspective=0.0005,       # Ğ´Ğ¾Ğ±Ğ°Ğ²Ğ»ĞµĞ½Ğ¸Ğµ Ñ€ĞµĞ°Ğ»Ğ¸Ñ�Ñ‚Ğ¸Ñ‡Ğ½Ğ¾Ñ�Ñ‚Ğ¸ / adding realism

    val=True,                 # Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ñ� Ğ²ĞºĞ»Ñ�Ñ‡ĞµĞ½Ğ° / validation is enabled
    save=True,                # Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�Ñ‚ÑŒ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸ / save models
    save_period=10,           # Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�Ñ‚ÑŒ ĞºĞ°Ğ¶Ğ´Ñ‹Ğµ 10 Ñ�Ğ¿Ğ¾Ñ… / save every 10 epochs
    workers=8,                # ÑƒÑ�ĞºĞ¾Ñ€Ğ¸Ñ‚ÑŒ Ğ·Ğ°Ğ³Ñ€ÑƒĞ·ĞºÑƒ Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ… / speed up data loading
    seed=42,                  # Ğ´Ğ»Ñ� Ğ²Ğ¾Ñ�Ğ¿Ñ€Ğ¾Ğ¸Ğ·Ğ²Ğ¾Ğ´Ğ¸Ğ¼Ğ¾Ñ�Ñ‚Ğ¸ / for reproducibility
)


for img_path in test_images.glob("*"):
    results = model.predict(img_path, conf=0.04, device=0, verbose=False) # 0 - GPU or "cpu"
    output_txt = output_dir / f"{img_path.stem}.txt"

    with open(output_txt, "w") as f:
        found = False
        for result in results:
            img_height, img_width = result.orig_shape
            boxes = result.boxes.data

            if boxes is None or len(boxes) == 0:
                continue

            filtered_boxes = boxes[boxes[:, 4] >= 0.05]
            if len(filtered_boxes) == 0:
                continue

            found = True
            for box in filtered_boxes:
                x1, y1, x2, y2, confidence, cls_id = box.tolist()

                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                f.write(f"0 {confidence:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        if not found:
            f.write("")


visualize_random_masks(test_images, output_dir)


rows = []
test_imgs = {p.stem for p in test_images.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
predicted = set()

for file in output_dir.glob("*.txt"):
    name = file.stem
    predicted.add(name)

    try:
        lines = [l.strip() for l in open(file) if len(l.strip().split()) == 6]
    except:
        lines = []

    rows.append({"image_id": name, "prediction_string": " ".join(lines) if lines else "no boxes"})

for name in test_imgs - predicted:
    rows.append({"image_id": name, "prediction_string": "no boxes"})


work_dir = '/kaggle/working'

for filename in os.listdir(work_dir):
    file_path = os.path.join(work_dir, filename)

    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(f'Error file: {file_path}. Cause: {e}')


rows = pd.DataFrame(rows)
rows.to_csv("submission.csv", index=False)




