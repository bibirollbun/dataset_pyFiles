!pip install ultralytics


import os
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import glob
from tqdm.notebook import tqdm
import shutil
import yaml

from ultralytics import YOLO
from sklearn.model_selection import StratifiedKFold

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

os.makedirs("/kaggle/working/submission/", exist_ok=True)
os.makedirs("/kaggle/working/dataset/images/train", exist_ok=True)
os.makedirs("/kaggle/working/dataset/images/val", exist_ok=True)
os.makedirs("/kaggle/working/dataset/labels/train", exist_ok=True)
os.makedirs("/kaggle/working/dataset/labels/val", exist_ok=True)



CONFIG = {
    "data": {
        "base_path": "/kaggle/input/where-are-the-seagulls/data",
        "train_images": "/kaggle/input/where-are-the-seagulls/data/train/images",
        "train_labels": "/kaggle/input/where-are-the-seagulls/data/train/labels",
        "test_images": "/kaggle/input/where-are-the-seagulls/data/test/images",
        "yaml_file": "/kaggle/working/data.yaml",
    },
    "model": {
        "name": "yolov8x.pt", 
        "img_size": 640, 
    },
    "training": {
        "batch_size": 16,
        "max_epochs": 50,
        "patience": 10, 
        "device": 0,
    },
    "inference": {
        "conf_threshold": 0.25,
        "nms_threshold": 0.45,
    },
    "validation": {
        "n_splits": 5,
    },
    "seed": SEED,
}


image_files = sorted(glob.glob(os.path.join(CONFIG["data"]["train_images"], "*.jpg")))
label_files = sorted(glob.glob(os.path.join(CONFIG["data"]["train_labels"], "*.txt")))

print(f"Всего изображений в трейне: {len(image_files)}")
print(f"Всего файлов с разметкой: {len(label_files)}")


total_bboxes = 0
labeled_images_count = 0
for label_path in label_files:
    if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
      with open(label_path, 'r') as f:
          lines = f.readlines()
          if lines:
                labeled_images_count += 1
                total_bboxes += len(lines)

print(f"Изображений с объектами: {labeled_images_count}")
print(f"Всего объектов (bbox): {total_bboxes}")
if labeled_images_count > 0:
    print(f"Среднее количество объектов на размеченном изображении: {total_bboxes / labeled_images_count:.2f}")

def plot_image_with_boxes(image_path, label_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = img.shape

    if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
        with open(label_path, 'r') as f:
            for line in f.readlines():
                _, x_c, y_c, w_b, h_b = map(float, line.strip().split())
                x1 = int((x_c - w_b / 2) * w_img)
                y1 = int((y_c - h_b / 2) * h_img)
                x2 = int((x_c + w_b / 2) * w_img)
                y2 = int((y_c + h_b / 2) * h_img)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(os.path.basename(image_path))
    plt.axis('off')
    plt.show()

if image_files:
    random_idx = random.randint(0, len(image_files) - 1)
    image_p = image_files[random_idx]
    label_p = image_p.replace("images", "labels").replace(".jpg", ".txt")
    plot_image_with_boxes(image_p, label_p)


all_files = sorted(glob.glob(os.path.join(CONFIG["data"]["train_images"], "*.jpg")))
file_counts = [1 if os.path.exists(f.replace("images", "labels").replace(".jpg", ".txt")) and os.path.getsize(f.replace("images", "labels").replace(".jpg", ".txt")) > 0 else 0 for f in all_files]

train_files, val_files = [], []

if len(all_files) > CONFIG["validation"]["n_splits"]:
    skf = StratifiedKFold(n_splits=CONFIG["validation"]["n_splits"], shuffle=True, random_state=CONFIG["seed"])
    try:
        train_idx, val_idx = next(iter(skf.split(all_files, file_counts)))
        train_files, val_files = [all_files[i] for i in train_idx], [all_files[i] for i in val_idx]
    except Exception:
        train_files, val_files = all_files[:int(len(all_files)*0.8)], all_files[int(len(all_files)*0.8):]
else:
    train_files = all_files

print(f"Обучающая выборка: {len(train_files)} изображений")
print(f"Валидационная выборка: {len(val_files)} изображений")


def prepare_data(file_list, split_type):
    for img_path in tqdm(file_list, desc=f"Preparing {split_type} data"):
        label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
        
        shutil.copy(img_path, f"/kaggle/working/dataset/images/{split_type}/{os.path.basename(img_path)}")
        
        if os.path.exists(label_path):
            shutil.copy(label_path, f"/kaggle/working/dataset/labels/{split_type}/{os.path.basename(label_path)}")
        else:
            open(f"/kaggle/working/dataset/labels/{split_type}/{os.path.basename(label_path)}", 'w').close()

prepare_data(train_files, "train")
prepare_data(val_files, "val")

data_yaml = {
    'train': '../dataset/images/train/',
    'val': '../dataset/images/val/',
    'nc': 1,
    'names': ['seagull']
}

with open(CONFIG['data']['yaml_file'], 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print("\nСодержимое data.yaml:")
with open(CONFIG['data']['yaml_file'], 'r') as f:
    print(f.read())


model = YOLO(CONFIG["model"]["name"])


results = model.train(
    data=CONFIG["data"]["yaml_file"],
    epochs=CONFIG["training"]["max_epochs"],
    patience=CONFIG["training"]["patience"],
    batch=CONFIG["training"]["batch_size"],
    imgsz=CONFIG["model"]["img_size"],
    device=CONFIG["training"]["device"],
    seed=CONFIG["seed"],
    project="seagull_detection", # Папка для сохранения результатов
    name="yolov8n_experiment"   # Имя конкретного запуска
)


try:
    best_model_path = os.path.join(results.save_dir, 'weights/best.pt')
    if not os.path.exists(best_model_path):
         # Если обучение не дало результатов, ищем в стандартной папке
         best_model_path = 'seagull_detection/yolov8n_experiment/weights/best.pt'
    print(f"Загрузка лучшей модели из: {best_model_path}")
    best_model = YOLO(best_model_path)
except Exception as e:
    print(f"Не удалось загрузить лучшую модель, используем последнюю: {e}")
    best_model = model

test_files = sorted(glob.glob(os.path.join(CONFIG["data"]["test_images"], "*.jpg")))

predictions = best_model.predict(
    source=test_files,
    conf=CONFIG["inference"]["conf_threshold"],
    iou=CONFIG["inference"]["nms_threshold"],
    imgsz=CONFIG["model"]["img_size"],
    stream=True
)


results_dict = {}
for result in tqdm(predictions, total=len(test_files), desc="Generating submission"):
    filename = os.path.basename(result.path)
    if len(result.boxes.xywhn) == 0:
        results_dict[filename] = "-1"
    else:
        pred_strings = [f"0 {b[0]:.4f} {b[1]:.4f} {b[2]:.4f} {b[3]:.4f}" for b in result.boxes.xywhn.cpu().numpy()]
        results_dict[filename] = " ".join(pred_strings)

submission_data = [{'index': i, 'filename': os.path.basename(fp), 'bbox': results_dict.get(os.path.basename(fp), "-1")} for i, fp in enumerate(test_files)]
submission_df = pd.DataFrame(submission_data)
submission_df.to_csv('/kaggle/working/submission/submission.csv', index=False)

print("\nПервые 5 строк файла submission.csv:")
print(submission_df.head())



batch_size = 8 
test_images_dir = CONFIG["data"]["test_images"]
conf_th = CONFIG["inference"]["conf_threshold"]
nms_th  = CONFIG["inference"]["nms_threshold"]
img_size = CONFIG["model"]["img_size"]

all_files = sorted(glob.glob(os.path.join(test_images_dir, "*.jpg")))
n_batches = math.ceil(len(all_files) / batch_size)

results_dict = {}

for batch_idx in range(n_batches):
    batch_files = all_files[batch_idx * batch_size : (batch_idx + 1) * batch_size]
    print(f"Обработка партии {batch_idx + 1}/{n_batches}: {len(batch_files)} изображений")

    preds = best_model.predict(
        source=batch_files,
        conf=conf_th,
        iou=nms_th,
        imgsz=img_size,
        stream=True
    )

    for result in preds:
        filename = os.path.basename(result.path)
        if len(result.boxes.xywhn) == 0:
            results_dict[filename] = "-1"
        else:
            pred_strings = [
                f"0 {b[0]:.4f} {b[1]:.4f} {b[2]:.4f} {b[3]:.4f}"
                for b in result.boxes.xywhn.cpu().numpy()
            ]
            results_dict[filename] = " ".join(pred_strings)

submission_data = [
    {'index': i, 'filename': os.path.basename(fp), 'bbox': results_dict.get(os.path.basename(fp), "-1")}
    for i, fp in enumerate(all_files)
]
submission_df = pd.DataFrame(submission_data)
submission_df.to_csv('/kaggle/working/submission/submission.csv', index=False)

print("\nПервые 5 строк файла submission.csv:")
print(submission_df.head())



def plot_test_predictions(df, num_images=5):
    if len(df) == 0:
        print("Нет данных для визуализации.")
        return
    num_images = min(len(df), num_images)
    sample_df = df[df['bbox'] != "-1"].sample(num_images, random_state=SEED)
    
    for _, row in sample_df.iterrows():
        img_path = os.path.join(CONFIG["data"]["test_images"], row['filename'])
        if not os.path.exists(img_path):
            print(f"Изображение не найдено: {img_path}")
            continue
            
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        # Предсказываем заново для одного изображения, чтобы получить объект Result
        # Это позволяет легко использовать встроенную функцию отрисовки
        pred_result = best_model.predict(source=img_path)[0]
        img_plotted = pred_result.plot() # Получаем изображение с нарисованными боксами
        
        plt.figure(figsize=(10, 10))
        plt.imshow(img_plotted)
        plt.title(f"Предсказания для: {row['filename']}")
        plt.axis('off')
        plt.show()

print("\nПримеры предсказаний на тестовых данных:")
plot_test_predictions(submission_df)




