!pip install -q ultralytics



import os
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm

# المسارات
images_dir = "/kaggle/input/multi-class-object-detection-challenge/Starter_Dataset/train/images"
labels_dir = "/kaggle/input/multi-class-object-detection-challenge/Starter_Dataset/train/labels"
aug_images_dir = "/path/to/save/augmented/images"
aug_labels_dir = "/path/to/save/augmented/labels"

os.makedirs(aug_images_dir, exist_ok=True)
os.makedirs(aug_labels_dir, exist_ok=True)

# عدد النسخ لكل صورة
n_augmentations = 4

# حجم الصورة الثابت
IMAGE_SIZE = 640  # أو أي حجم تستخدمه في تدريبك

# تعريف الـ augmentations
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=10, p=0.7),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# التحويل والتخزين
for filename in tqdm(os.listdir(images_dir)):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(images_dir, filename)
        label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")

        # اقرأ الصورة
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        # اقرأ الـ labels
        bboxes = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, w, h = map(float, line.strip().split())
                    bboxes.append([x_center, y_center, w, h])
                    class_labels.append(int(class_id))

        for i in range(n_augmentations):
            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)

                new_img_name = f"{os.path.splitext(filename)[0]}_aug{i}.jpg"
                new_label_name = f"{os.path.splitext(filename)[0]}_aug{i}.txt"

                cv2.imwrite(os.path.join(aug_images_dir, new_img_name), aug_img)

                with open(os.path.join(aug_labels_dir, new_label_name), 'w') as f:
                    for bbox, cls in zip(augmented['bboxes'], augmented['class_labels']):
                        x, y, w, h = bbox
                        f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            except Exception as e:
                print(f"Error augmenting {filename}: {e}")



import yaml

# Set paths
base_path = "/kaggle/input/multi-class-object-detection-challenge"
yaml_path = "/kaggle/working/yolo_params.yaml"


# Build YAML dictionary
data_yaml = {
    "train": f"{base_path}/Starter_Dataset/train/images",
    "val":   f"{base_path}/Starter_Dataset/val/images",
    "test":  f"{base_path}/testImages/images",
    "nc": 2,
    "names": ["cheerios", "soup"]
}

# Write to file
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print("✅ yolo_params.yaml updated successfully at:", yaml_path)



#with open("/kaggle/working/yolo_params.yaml", "r") as f:
#    print(f.read())
#


# import os
# import cv2
# import matplotlib.pyplot as plt

# # Paths
# base_path = "/kaggle/input/multi-class-object-detection-challenge/Dataset"

# images_dir = f"{base_path}/train/images"
# labels_dir = f"{base_path}/train/labels"
# class_names = ['cheerios', 'soup']  # Make sure these match the dataset classes

# def plot_yolo_label(img_path, label_path):
#     image = cv2.imread(img_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     h, w = image.shape[:2]

#     if not os.path.exists(label_path):
#         print(f"Label not found: {label_path}")
#         return image

#     with open(label_path, 'r') as f:
#         lines = f.readlines()

#     for i, line in enumerate(lines):
#         parts = line.strip().split()
#         if len(parts) != 5:
#             print(f"Skipping invalid line {i} in {label_path}: {parts}")
#             continue

#         class_id = int(parts[0])
#         x_center, y_center, box_w, box_h = map(float, parts[1:])

#         # Convert normalized to pixel coordinates
#         x1 = int((x_center - box_w / 2) * w)
#         y1 = int((y_center - box_h / 2) * h)
#         x2 = int((x_center + box_w / 2) * w)
#         y2 = int((y_center + box_h / 2) * h)

#         # Clip to image boundaries
#         x1, y1 = max(0, x1), max(0, y1)
#         x2, y2 = min(w - 1, x2), min(h - 1, y2)

#         # Draw box and label
#         label = f"{class_names[class_id]}"
#         cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 4)
#         cv2.putText(image, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 10)

#         #print(f"[{img_path}] Box {i}: {label} → ({x1},{y1}) to ({x2},{y2})")

#     return image

# # Show 5 sample images with labels
# sample_images = os.listdir(images_dir)[:5]

# plt.figure(figsize=(15, 10))
# for i, image_file in enumerate(sample_images):
#     img_path = os.path.join(images_dir, image_file)
#     label_file = image_file.replace(".png", ".txt").replace(".jpg", ".txt")
#     label_path = os.path.join(labels_dir, label_file)

#     img = plot_yolo_label(img_path, label_path)
#     plt.subplot(1, 5, i + 1)
#     plt.imshow(img)
#     plt.axis("off")
#     plt.title(image_file)

# plt.tight_layout()
# plt.show()



zero, one, multi = 0, 0, 0
for file in os.listdir(labels_dir):
    if file.endswith(".txt"):
        with open(os.path.join(labels_dir, file)) as f:
            count = len(f.readlines())
            if count == 0:
                zero += 1
            elif count == 1:
                one += 1
            else:
                multi += 1

print(f"Images with 0 boxes: {zero}")
print(f"Images with 1 box: {one}")
print(f"Images with 2+ boxes: {multi}")



from ultralytics import YOLO

model = YOLO("yolo12m.pt")  

model.train(
    data="/kaggle/working/yolo_params.yaml",
    epochs=100,
    imgsz=640,          
    batch=16,            

    # Optimization
    optimizer='SGD',   
    lr0=0.001,
    weight_decay=0.0005,
    momentum=0.937,
    cos_lr=True,

    # Augmentations
    mosaic=1.0,
    close_mosaic=40,     
    hsv_h=0.025,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.4,
    translate=0.2,


    # General settings
    patience=30,
    workers=4,
    seed=42,
    warmup_epochs=3,
    project='runs/train',
    name='run1',
    save_period=5,
    plots=True,
    verbose=True,
)


model = YOLO("/kaggle/working/runs2/train_best/yolov12l_aug_896/weights/best.pt")
img_test = "/kaggle/input/multi-class-object-detection-challenge/testImages/images"
model.predict(
    source=img_test,
    save=True
)


import os
import cv2
import matplotlib.pyplot as plt
# Path to YOLOv8 predictions
predict_dir = "/kaggle/working/runs/detect/predict"
image_files = [f for f in os.listdir(predict_dir) if f.lower().endswith(('.jpg', '.png'))]

# Show up to 5 predictions
plt.figure(figsize=(15, 8))
for i, img_file in enumerate(image_files[:10]):
    img_path = os.path.join(predict_dir, img_file)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(1, 10, i + 1)
    plt.imshow(img)
    plt.title(img_file, fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.show()



import os
import cv2
import matplotlib.pyplot as plt

# مسار الصور المتوقعة
predict_dir = "/kaggle/working/runs/detect/predict"
image_files = [f for f in os.listdir(predict_dir) if f.lower().endswith(('.jpg', '.png'))]

# نعرض حتى 200 صورة
num_images = min(200, len(image_files))
cols = 5
rows = (num_images + cols - 1) // cols  # لحساب عدد الصفوف المطلوب

plt.figure(figsize=(20, rows * 4))  # تعديل الحجم حسب عدد الصفوف
for i in range(num_images):
    img_path = os.path.join(predict_dir, image_files[i])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(rows, cols, i + 1)
    plt.imshow(img)
    plt.title(image_files[i], fontsize=8)
    plt.axis("off")

plt.tight_layout()
plt.show()



# Define file paths
plot_dir = "/kaggle/working/runs2/train_best/yolov12l_aug_896"
plot_files = ["confusion_matrix.png", "results.png"]

plt.figure(figsize=(15, 6))

for i, plot_file in enumerate(plot_files):
    path = os.path.join(plot_dir, plot_file)
    
    # Check if file exists
    if os.path.exists(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, 2, i + 1)
        plt.imshow(img)
        plt.title(plot_file)
        plt.axis("off")
    else:
        print(f"❌ File not found: {path}")

plt.tight_layout()
plt.show()



from pathlib import Path
import cv2
from ultralytics import YOLO
model_path = "/kaggle/working/runs2/train_best/yolov12l_aug_896/weights/best.pt"
test_images_path = Path("/kaggle/input/multi-class-object-detection-challenge/testImages/images")
output_img_dir = Path("/kaggle/working/predictions/images")
output_lbl_dir = Path("/kaggle/working/predictions/labels")

output_img_dir.mkdir(parents=True, exist_ok=True)
output_lbl_dir.mkdir(parents=True, exist_ok=True)

model = YOLO(model_path)

for img_path in test_images_path.glob("*.[jp][pn]g"):
    results = model.predict(str(img_path), conf=0.5)
    result = results[0]

    img = result.plot()
    out_img_path = output_img_dir / img_path.name
    cv2.imwrite(str(out_img_path), img)

    out_txt_path = output_lbl_dir / (img_path.stem + ".txt")
    with open(out_txt_path, "w") as f:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x, y, w, h = box.xywhn[0].tolist()  # normalized
            f.write(f"{cls} {conf:.6f} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

print("✅ Prediction done and saved to /kaggle/working/predictions/")



import pandas as pd
from pathlib import Path
import csv

labels_dir = Path("/kaggle/working/predictions/labels")
test_images_dir = Path("/kaggle/input/multi-class-object-detection-challenge/testImages/images")
output_csv = "/kaggle/working/submission.csv"

image_ids = [img.stem for img in test_images_dir.glob("*.[jp][pn]g")]

submission_data = []

for image_id in image_ids:
    label_path = labels_dir / f"{image_id}.txt"
    if label_path.exists():
        with open(label_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        pred_str = " ".join(lines) if lines else "no boxes"
    else:
        pred_str = "no boxes"

    submission_data.append({
        "image_id": image_id,
        "prediction_string": pred_str
    })

df = pd.DataFrame(submission_data)
df.to_csv(output_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)

print(f"✅ Submission file created at: {output_csv}")



import shutil

shutil.make_archive("/kaggle/working/predict_results", 'zip', "/kaggle/working/runs/detect/predict")



/kaggle/working/predict_results.zip

