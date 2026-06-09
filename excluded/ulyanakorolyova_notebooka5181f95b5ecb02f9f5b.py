import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Пропишите дополнительные библиотеки, которые потребуются для решения
import cv2
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
import random
from concurrent.futures import ThreadPoolExecutor


train_img_dir = '/kaggle/input/where-are-the-seagulls/data/train/images'
train_label_dir = '/kaggle/input/where-are-the-seagulls/data/train/labels'

def analyze_dataset(img_dir, label_dir):
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    objects_per_image = []
    images_without_objects = 0
    widths = []
    heights = []

    for img_file in img_files:
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')
        num_objects = 0

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                objects = [line.strip() for line in f if line.strip()]
                num_objects = len(objects)

                for line in objects:
                    class_id, x_center, y_center, width, height = map(float, line.split())
                    widths.append(width)
                    heights.append(height)

        objects_per_image.append(num_objects)
        if num_objects == 0:
            images_without_objects += 1

    widths = np.array(widths)
    heights = np.array(heights)

    # Среднее количество на фото
    avg_objects = np.mean(objects_per_image)
    # Максимальное количество объектов
    max_objects = max(objects_per_image)
    # Всего изображений
    total_images = len(img_files)
    # Определение минимального и максимального значений для ширины bounding box
    width_data_min = np.min(widths)
    width_data_max = np.max(widths)
    # Определение минимального и максимального значений для высоты bounding box
    height_data_min = np.min(heights)
    height_data_max = np.max(heights)

    # Вывод даннных
    print(f"Высота bounding box максимальная {height_data_max} и минимальная {height_data_min}")
    print(f"Ширина bounding box максимальная {width_data_max} и минимальная {width_data_min}")

    print(f"Изображений: {total_images}")
    print(f"Среднее количество: {avg_objects:.2f}")
    print(f"Максимально: {max_objects}")
    print(f"Без объектов: {images_without_objects}")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=widths, color='skyblue')
    plt.title('Box Plot of Width')

    plt.subplot(1, 2, 2)
    sns.boxplot(y=heights, color='lightgreen')
    plt.title('Box Plot of Height')

    plt.tight_layout()
    plt.show()

    examples(img_dir, label_dir)


def examples(img_dir, label_dir):
    fig, axes = plt.subplots(1, 4, figsize=(10, 5))
    img_files = os.listdir(img_dir)[:4]

    for i, img_file in enumerate(img_files):
        img_path = os.path.join(img_dir, img_file)
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        with open(label_path, 'r') as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.split())

                x_center *= w
                y_center *= h
                width *= w
                height *= h

                x1 = int(x_center - width/2)
                y1 = int(y_center - height/2)
                x2 = int(x_center + width/2)
                y2 = int(y_center + height/2)

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        axes[i].imshow(img)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


analyze_dataset(train_img_dir, train_label_dir)


import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def prepare_yolo_dataset(
    images_dir="/kaggle/input/where-are-the-seagulls/data/train/images",
    labels_dir="/kaggle/input/where-are-the-seagulls/data/train/labels",
    save_dir="yolo",
    val_ratio=0.2,
):
    os.makedirs(os.path.join(save_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labels", "val"), exist_ok=True)

    all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(all_images)
    
    split_idx = int(len(all_images) * (1 - val_ratio))
    train_files = all_images[:split_idx]
    val_files = all_images[split_idx:]

    def copy_files(files, subset):
        for img_file in tqdm(files, desc=f"Copying {subset} files"):
            img_src = os.path.join(images_dir, img_file)
            img_dst = os.path.join(save_dir, "images", subset, img_file)
            shutil.copy(img_src, img_dst)
            
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_src = os.path.join(labels_dir, label_file)
            if os.path.exists(label_src):
                label_dst = os.path.join(save_dir, "labels", subset, label_file)
                shutil.copy(label_src, label_dst)

    copy_files(train_files, "train")
    copy_files(val_files, "val")


prepare_yolo_dataset()


import yaml
yolo_config = {
    'train': os.path.abspath('yolo/images/train'),
    'val': os.path.abspath('yolo/images/val'),
    'nc': 1,
    'names': ['птицы']
}

os.makedirs('yolo', exist_ok=True)
with open('yolo/dataset.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(yolo_config, f, allow_unicode=True)



!pip install ultralytics
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(
    data='yolo/dataset.yaml',
    epochs=20,
    imgsz=640
)


test_dir = '/kaggle/input/where-are-the-seagulls/data/test/images'

# === Получаем список изображений из test ===
image_files = sorted([f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

# === Предсказания и формирование строк submission.csv ===
submission = []

for idx, image_name in enumerate(image_files):
    image_path = os.path.join(test_dir, image_name)
    results = model(image_path, verbose=False)[0]

    # Достаем боксы (xywhn = нормализованные координаты центра + ширина/высота)
    boxes = results.boxes
    if boxes is None or boxes.xywhn.shape[0] == 0:
        submission.append([idx, image_name, "-1"])
    else:
        yolo_str = ""
        for box in boxes.xywhn:
            x_center, y_center, w, h = box.tolist()
            yolo_str += f"0 {x_center:.4f} {y_center:.4f} {w:.4f} {h:.4f} "
        yolo_str = yolo_str.strip()
        submission.append([idx, image_name, yolo_str])

# === Сохраняем в CSV ===

df = pd.DataFrame(submission, columns=["index", "filename", "bbox"])

df.to_csv('/kaggle/working/yolo/submit.csv', index=False)
print('done')

