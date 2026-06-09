import os
os.listdir("/kaggle/input")



import os

for item in os.listdir("/kaggle/input/diabetic-retinopathy-detection"):
    print(item)



import zipfile
import os

zip_path = "/kaggle/input/diabetic-retinopathy-detection/trainLabels.csv.zip"
extract_path = "/kaggle/working/eyepacs"

os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print(os.listdir(extract_path))



import pandas as pd

eyepacs_df = pd.read_csv("/kaggle/working/eyepacs/trainLabels.csv")

eyepacs_df.head()



# EyePACS standard format
eyepacs_df.rename(
    columns={"image": "id_code", "level": "label"},
    inplace=True
)

eyepacs_df["source"] = "EyePACS"

eyepacs_df.head()



aptos_df = pd.read_csv(
    "/kaggle/input/aptos2019-blindness-detection/train.csv"
)

aptos_df.head()



aptos_df.rename(columns={"diagnosis": "label"}, inplace=True)
aptos_df["source"] = "APTOS"

aptos_df.head()



print("EyePACS:", eyepacs_df.shape)
print("APTOS:", aptos_df.shape)



#both merging

full_df = pd.concat(
    [eyepacs_df, aptos_df],
    ignore_index=True
)



print("Full dataset shape:", full_df.shape)

print("\nSource distribution:")
print(full_df["source"].value_counts())

print("\nLabel distribution:")
print(full_df["label"].value_counts())

full_df.head()



#STEP 4: IMAGE PREPROCESSING (START SLOW)
#Goal (iss step ka):

#Black borders remove

#Image readable banani

#Model ke liye ready karni (later resize)


#Pick ONE sample image (APTOS)
import cv2
import matplotlib.pyplot as plt

sample = full_df[full_df["source"] == "APTOS"].iloc[0]

img_path = (
    "/kaggle/input/aptos2019-blindness-detection/train_images/"
    + sample["id_code"]
    + ".png"
)

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.title("Original Image")
plt.axis("off")



#Black Border Crop Function
import numpy as np

def crop_black(img, tol=7):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.any():
        img = img[np.ix_(mask.any(1), mask.any(0))]
    return img



cropped = crop_black(img)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Before Crop")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(cropped)
plt.title("After Crop")
plt.axis("off")



#Resize + Normalize Function
def preprocess_image(img, target_size=(224, 224)):
    img = crop_black(img)
    img = cv2.resize(img, target_size)
    img = img / 255.0   # normalize to [0,1]
    return img



#Test Full Preprocessing Pipeline
processed = preprocess_image(img)

plt.figure(figsize=(4,4))
plt.imshow(processed)
plt.title("Preprocessed Image (224x224)")
plt.axis("off")

print("Shape:", processed.shape)
print("Pixel range:", processed.min(), processed.max())



#Image Loader Function
def load_image(row):
    if row["source"] == "APTOS":
        path = (
            "/kaggle/input/aptos2019-blindness-detection/train_images/"
            + row["id_code"]
            + ".png"
        )
    else:  # EyePACS (later, when images available)
        path = None
    
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_image(img)
    return img



#: Test Loader on 3 Images
for i in range(3):
    sample = full_df[full_df["source"] == "APTOS"].iloc[i]
    img = load_image(sample)
    plt.imshow(img)
    plt.title(f"Label: {sample['label']}")
    plt.axis("off")
    plt.show()



#TRAIN–VALIDATION SPLIT (APTOS ONLY)
from sklearn.model_selection import train_test_split

aptos_only = full_df[full_df["source"] == "APTOS"].reset_index(drop=True)

train_df, val_df = train_test_split(
    aptos_only,
    test_size=0.2,
    stratify=aptos_only["label"],
    random_state=42
)

print("Train:", train_df.shape)
print("Val:", val_df.shape)



import random
import cv2
import matplotlib.pyplot as plt

samples = aptos_df.sample(6, random_state=42)

plt.figure(figsize=(12, 8))

for i, (_, row) in enumerate(samples.iterrows()):
    img_path = (
        "/kaggle/input/aptos2019-blindness-detection/train_images/"
        + row["id_code"]
        + ".png"
    )
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 3, i+1)
    plt.imshow(img)
    plt.title(f"Label: {row['label']}")
    plt.axis("off")

plt.tight_layout()
plt.show()



plt.figure(figsize=(15, 6))

for label in range(5):
    row = aptos_df[aptos_df["label"] == label].sample(1, random_state=1).iloc[0]
    img_path = (
        "/kaggle/input/aptos2019-blindness-detection/train_images/"
        + row["id_code"]
        + ".png"
    )
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(1, 5, label+1)
    plt.imshow(img)
    plt.title(f"Class {label}")
    plt.axis("off")

plt.show()



!pip install -q ultralytics



from ultralytics import YOLO



import cv2
import numpy as np

def generate_pseudo_boxes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # threshold
    _, thresh = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)

    # find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    h, w = gray.shape

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh

        if area > 300:   # remove tiny noise
            boxes.append((x, y, bw, bh))

    return boxes



import matplotlib.pyplot as plt

sample = aptos_df.sample(1).iloc[0]
img_path = (
    "/kaggle/input/aptos2019-blindness-detection/train_images/"
    + sample["id_code"]
    + ".png"
)

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

boxes = generate_pseudo_boxes(img)

for (x, y, w, h) in boxes:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

plt.figure(figsize=(6,6))
plt.imshow(img)
plt.title("Pseudo Bounding Boxes")
plt.axis("off")
plt.show()



import matplotlib.pyplot as plt
import cv2
import random

samples = aptos_df.sample(6, random_state=42)

plt.figure(figsize=(14, 10))

for i, (_, row) in enumerate(samples.iterrows()):
    img_path = (
        "/kaggle/input/aptos2019-blindness-detection/train_images/"
        + row["id_code"]
        + ".png"
    )
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes = generate_pseudo_boxes(img)

    # draw BLACK + BOLD boxes
    for (x, y, w, h) in boxes:
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 0, 0),   # BLACK color
            4            # thickness (bold)
        )

    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.title(f"Label: {row['label']}")
    plt.axis("off")

plt.tight_layout()
plt.show()



def convert_to_yolo_format(boxes, img_w, img_h):
    yolo_labels = []

    for (x, y, w, h) in boxes:
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        bw = w / img_w
        bh = h / img_h

        # class 0 = lesion
        yolo_labels.append(f"0 {x_center} {y_center} {bw} {bh}")

    return yolo_labels



import os

base_dir = "/kaggle/working/yolo_aptos"
img_dir = os.path.join(base_dir, "images/train")
lbl_dir = os.path.join(base_dir, "labels/train")

os.makedirs(img_dir, exist_ok=True)
os.makedirs(lbl_dir, exist_ok=True)

print("Folders ready")



import shutil

subset = aptos_df.sample(200, random_state=42)

for _, row in subset.iterrows():
    img_name = row["id_code"] + ".png"
    src_path = (
        "/kaggle/input/aptos2019-blindness-detection/train_images/"
        + img_name
    )

    img = cv2.imread(src_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape
    boxes = generate_pseudo_boxes(img)
    yolo_labels = convert_to_yolo_format(boxes, w, h)

    # save image
    shutil.copy(src_path, os.path.join(img_dir, img_name))

    # save label file
    label_path = os.path.join(lbl_dir, row["id_code"] + ".txt")
    with open(label_path, "w") as f:
        for line in yolo_labels:
            f.write(line + "\n")

print("200 images + labels saved")



import os

print("Images:", len(os.listdir("/kaggle/working/yolo_aptos/images/train")))
print("Labels:", len(os.listdir("/kaggle/working/yolo_aptos/labels/train")))

# show one label file
sample_label = os.listdir("/kaggle/working/yolo_aptos/labels/train")[0]
with open("/kaggle/working/yolo_aptos/labels/train/" + sample_label) as f:
    print(f.read())



data_yaml = """
path: /kaggle/working/yolo_aptos
train: images/train

nc: 1
names: ['lesion']
"""

with open("/kaggle/working/yolo_aptos/data.yaml", "w") as f:
    f.write(data_yaml)

print("data.yaml created")



#yolo training


from ultralytics import YOLO

# load small YOLOv8 model (fast & sufficient for start)
model = YOLO("yolov8n.pt")



import os
import shutil
import random

base_dir = "/kaggle/working/yolo_aptos"

img_train = os.path.join(base_dir, "images/train")
lbl_train = os.path.join(base_dir, "labels/train")

img_val = os.path.join(base_dir, "images/val")
lbl_val = os.path.join(base_dir, "labels/val")

os.makedirs(img_val, exist_ok=True)
os.makedirs(lbl_val, exist_ok=True)

all_images = os.listdir(img_train)
random.shuffle(all_images)

val_images = all_images[:40]   # 20% for validation

for img_name in val_images:
    # move image
    shutil.move(
        os.path.join(img_train, img_name),
        os.path.join(img_val, img_name)
    )

    # move label
    label_name = img_name.replace(".png", ".txt")
    shutil.move(
        os.path.join(lbl_train, label_name),
        os.path.join(lbl_val, label_name)
    )

print("Train–Val split done")



data_yaml = """
path: /kaggle/working/yolo_aptos

train: images/train
val: images/val

nc: 1
names: ['lesion']
"""

with open("/kaggle/working/yolo_aptos/data.yaml", "w") as f:
    f.write(data_yaml)

print("data.yaml updated with val")



model.train(
    data="/kaggle/working/yolo_aptos/data.yaml",
    epochs=20,
    imgsz=640,
    batch=8,
    workers=2,
    name="yolo_aptos_lesion"
)



from ultralytics import YOLO

model = YOLO("/kaggle/working/runs/detect/yolo_aptos_lesion2/weights/best.pt")
print("Model loaded")



import matplotlib.pyplot as plt
import cv2
import random

samples = aptos_df.sample(6, random_state=1)

plt.figure(figsize=(14, 10))

for i, (_, row) in enumerate(samples.iterrows()):
    img_path = (
        "/kaggle/input/aptos2019-blindness-detection/train_images/"
        + row["id_code"]
        + ".png"
    )

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model(img, conf=0.25, iou=0.4)

    # draw YOLO predictions
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 3)

    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.title(f"GT Label: {row['label']}")
    plt.axis("off")

plt.tight_layout()
plt.show()



import os

base_crop_dir = "/kaggle/working/cropped_aptos/images"

for label in range(5):
    os.makedirs(os.path.join(base_crop_dir, str(label)), exist_ok=True)

print("Cropped dataset folders ready")



def crop_from_boxes(img, boxes, min_area=5000):
    crops = []
    h, w, _ = img.shape

    for box in boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        area = (x2 - x1) * (y2 - y1)

        if area < min_area:
            continue

        crop = img[y1:y2, x1:x2]
        crops.append(crop)

    return crops



import matplotlib.pyplot as plt
import cv2

test_samples = aptos_df.sample(3, random_state=2)

for _, row in test_samples.iterrows():
    img_path = (
        "/kaggle/input/aptos2019-blindness-detection/train_images/"
        + row["id_code"]
        + ".png"
    )

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model(img, conf=0.25)

    for r in results:
        crops = crop_from_boxes(img, r.boxes)

        plt.figure(figsize=(12, 3))
        for i, c in enumerate(crops[:4]):
            c = cv2.resize(c, (224, 224))
            plt.subplot(1, 4, i+1)
            plt.imshow(c)
            plt.axis("off")

        plt.suptitle(f"GT Label: {row['label']}")
        plt.show()



def crop_from_boxes_with_context(img, boxes, expand_ratio=0.4, min_area=5000):
    crops = []
    h, w, _ = img.shape

    for box in boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh

        if area < min_area:
            continue

        # expand box
        dx = int(bw * expand_ratio)
        dy = int(bh * expand_ratio)

        nx1 = max(0, x1 - dx)
        ny1 = max(0, y1 - dy)
        nx2 = min(w, x2 + dx)
        ny2 = min(h, y2 + dy)

        crop = img[ny1:ny2, nx1:nx2]
        crops.append(crop)

    return crops



import cv2
import numpy as np

def crop_from_boxes_with_context(
    img,
    boxes,
    expand_ratio=0.6,   # more context
    min_area=2000       # allow smaller lesions
):
    crops = []
    h, w, _ = img.shape

    # Case 1: YOLO gave boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh

            if area < min_area:
                continue

            dx = int(bw * expand_ratio)
            dy = int(bh * expand_ratio)

            nx1 = max(0, x1 - dx)
            ny1 = max(0, y1 - dy)
            nx2 = min(w, x2 + dx)
            ny2 = min(h, y2 + dy)

            crop = img[ny1:ny2, nx1:nx2]
            if crop.size > 0:
                crops.append(crop)

    # Case 2: NO valid crops → fallback center crop
    if len(crops) == 0:
        cx, cy = w // 2, h // 2
        size = min(h, w) // 3  # central retina region

        crop = img[
            cy - size : cy + size,
            cx - size : cx + size
        ]

        if crop.size > 0:
            crops.append(crop)

    return crops



import matplotlib.pyplot as plt
import cv2

# 3 images test ke liye
test_samples = aptos_df.sample(3, random_state=2)

for _, row in test_samples.iterrows():
    img_path = (
        "/kaggle/input/aptos2019-blindness-detection/train_images/"
        + row["id_code"]
        + ".png"
    )

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # YOLO prediction
    results = model(img, conf=0.25, iou=0.4)

    for r in results:
        crops = crop_from_boxes_with_context(img, r.boxes)

        if len(crops) == 0:
            print("No valid crops for this image")
            continue

        plt.figure(figsize=(12, 3))
        for i, c in enumerate(crops[:4]):  # max 4 crops dikhao
            c = cv2.resize(c, (224, 224))
            plt.subplot(1, 4, i + 1)
            plt.imshow(c)
            plt.axis("off")

        plt.suptitle(f"GT Label: {row['label']}")
        plt.show()



def generate_multi_crops(img, boxes, expand_ratio=0.6):
    h, w, _ = img.shape
    crops = []

    # 1. YOLO-based crops
    if boxes is not None and len(boxes) > 0:
        for box in boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            bw = x2 - x1
            bh = y2 - y1

            dx = int(bw * expand_ratio)
            dy = int(bh * expand_ratio)

            nx1 = max(0, x1 - dx)
            ny1 = max(0, y1 - dy)
            nx2 = min(w, x2 + dx)
            ny2 = min(h, y2 + dy)

            crop = img[ny1:ny2, nx1:nx2]
            if crop.size > 0:
                crops.append(crop)

    # 2. Fallback multi-crops (ALWAYS add)
    size = min(h, w) // 3
    cy = h // 2

    # center
    crops.append(img[cy-size:cy+size, w//2-size:w//2+size])

    # left-center
    crops.append(img[cy-size:cy+size, w//4-size:w//4+size])

    # right-center
    crops.append(img[cy-size:cy+size, 3*w//4-size:3*w//4+size])

    # clean + resize
    final_crops = []
    for c in crops:
        if c.size > 0:
            final_crops.append(cv2.resize(c, (224, 224)))

    return final_crops



import matplotlib.pyplot as plt

def visualize_multi_crops(img, boxes, gt_label=None):
    crops = generate_multi_crops(img, boxes)

    plt.figure(figsize=(15, 4))
    for i, crop in enumerate(crops):
        plt.subplot(1, len(crops), i + 1)
        plt.imshow(crop)
        plt.axis("off")
        plt.title(f"Crop {i+1}")

    if gt_label is not None:
        plt.suptitle(f"GT Label: {gt_label}", fontsize=14)

    plt.show()



# ek random image test ke liye
row = aptos_df.sample(1).iloc[0]

img_path = (
    "/kaggle/input/aptos2019-blindness-detection/train_images/"
    + row["id_code"]
    + ".png"
)

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = model(img, conf=0.25, iou=0.4)

for r in results:
    visualize_multi_crops(img, r.boxes, gt_label=row["label"])



def generate_multi_crops(img, boxes, expand_ratio=0.6):
    h, w, _ = img.shape
    crops = []

    # 1. YOLO-based crops
    if boxes is not None and len(boxes) > 0:
        for box in boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            bw = x2 - x1
            bh = y2 - y1

            dx = int(bw * expand_ratio)
            dy = int(bh * expand_ratio)

            nx1 = max(0, x1 - dx)
            ny1 = max(0, y1 - dy)
            nx2 = min(w, x2 + dx)
            ny2 = min(h, y2 + dy)

            crop = img[ny1:ny2, nx1:nx2]
            if crop.size > 0:
                crops.append(crop)

    # 2. Fallback multi-crops (center + sides + quadrants)
    size = min(h, w) // 4

    centers = [
        (w//2, h//2),           # center
        (w//4, h//2),           # left
        (3*w//4, h//2),         # right
        (w//4, h//4),           # top-left
        (3*w//4, h//4),         # top-right
        (w//4, 3*h//4),         # bottom-left
        (3*w//4, 3*h//4)        # bottom-right
    ]

    for cx, cy in centers:
        crop = img[
            cy-size:cy+size,
            cx-size:cx+size
        ]
        if crop.size > 0:
            crops.append(crop)

    # resize
    final_crops = [cv2.resize(c, (224,224)) for c in crops if c.size > 0]

    return final_crops



import matplotlib.pyplot as plt

def visualize_multi_crops_full(img, boxes, gt_label=None):
    crops = generate_multi_crops(img, boxes)

    n = len(crops)
    plt.figure(figsize=(3*n, 4))

    for i, crop in enumerate(crops):
        plt.subplot(1, n, i + 1)
        plt.imshow(crop)
        plt.axis("off")
        plt.title(f"Crop {i+1}")

    title = "Multi-Crop Visualization"
    if gt_label is not None:
        title += f" | GT Label: {gt_label}"

    plt.suptitle(title, fontsize=14)
    plt.show()



# random sample
row = aptos_df.sample(1, random_state=42).iloc[0]

img_path = (
    "/kaggle/input/aptos2019-blindness-detection/train_images/"
    + row["id_code"]
    + ".png"
)

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = model(img, conf=0.25, iou=0.4)

for r in results:
    visualize_multi_crops_full(
        img,
        r.boxes,
        gt_label=row["label"]
    )



NUM_SAMPLES = 6   # 5–8 rakho, zyada mat



import random
import matplotlib.pyplot as plt
import cv2

def visualize_multiple_images(df, num_samples=5):
    samples = df.sample(num_samples, random_state=42)

    for idx, row in samples.iterrows():
        img_path = (
            "/kaggle/input/aptos2019-blindness-detection/train_images/"
            + row["id_code"]
            + ".png"
        )

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = model(img, conf=0.25, iou=0.4)

        for r in results:
            crops = generate_multi_crops(img, r.boxes)

            plt.figure(figsize=(3 * len(crops), 4))
            for i, crop in enumerate(crops):
                plt.subplot(1, len(crops), i + 1)
                plt.imshow(crop)
                plt.axis("off")
                plt.title(f"Crop {i+1}")

            plt.suptitle(
                f"ID: {row['id_code']} | GT Label: {row['label']}",
                fontsize=14
            )
            plt.show()



visualize_multiple_images(aptos_df, NUM_SAMPLES)



import cv2
import matplotlib.pyplot as plt

def draw_yolo_boxes(img, boxes, color=(0, 0, 0), thickness=3):
    img_boxed = img.copy()

    if boxes is None or len(boxes) == 0:
        return img_boxed

    for box in boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_boxed, (x1, y1), (x2, y2), color, thickness)

    return img_boxed


NUM_IMAGES = 5
samples = aptos_df.sample(NUM_IMAGES, random_state=42)

plt.figure(figsize=(14, 4 * NUM_IMAGES))

row_idx = 1
for _, row in samples.iterrows():
    img_path = (
        "/kaggle/input/aptos2019-blindness-detection/train_images/"
        + row["id_code"]
        + ".png"
    )

    img = cv2.imread(img_path)
    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # YOLO prediction
    results = model(img, conf=0.25, iou=0.4)

    for r in results:
        img_boxed = draw_yolo_boxes(img, r.boxes)

        # original
        plt.subplot(NUM_IMAGES, 2, row_idx)
        plt.imshow(img)
        plt.title(f"Original | GT: {row['label']}")
        plt.axis("off")

        # boxed
        plt.subplot(NUM_IMAGES, 2, row_idx + 1)
        plt.imshow(img_boxed)
        plt.title("YOLO Boxes")
        plt.axis("off")

        row_idx += 2

plt.tight_layout()
plt.show()



import os

BASE_DIR = "/kaggle/working/final_cropped_dataset"

for split in ["train", "val"]:
    for cls in range(5):
        os.makedirs(f"{BASE_DIR}/{split}/{cls}", exist_ok=True)

print("✅ Final dataset folders created")



import cv2
from tqdm import tqdm

def generate_and_save_crops(df, split="train"):
    saved = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = (
            "/kaggle/input/aptos2019-blindness-detection/train_images/"
            + row["id_code"]
            + ".png"
        )

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # YOLO inference
        results = model(img, conf=0.25, iou=0.4)

        for r in results:
            crops = generate_multi_crops(img, r.boxes)

            for i, crop in enumerate(crops):
                save_path = (
                    f"{BASE_DIR}/{split}/{row['label']}/"
                    f"{row['id_code']}_{i}.jpg"
                )
                cv2.imwrite(
                    save_path,
                    cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                )
                saved += 1

    print(f"✅ {split} crops saved:", saved)



import cv2

def lightweight_crops(img, boxes, max_yolo_crops=2):
    """
    img: RGB image (H, W, 3)
    boxes: YOLO r.boxes
    returns: list of 224x224 RGB crops
    """
    h, w, _ = img.shape
    crops = []

    # 1️⃣ YOLO-based crops (LIMITED → FAST)
    if boxes is not None and len(boxes) > 0:
        for box in boxes.xyxy.cpu().numpy()[:max_yolo_crops]:
            x1, y1, x2, y2 = map(int, box)

            # safety clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                crop = cv2.resize(crop, (224, 224))
                crops.append(crop)

    # 2️⃣ Fallback CENTER crop (ALWAYS)
    size = min(h, w) // 3
    cx, cy = w // 2, h // 2
    center_crop = img[cy-size:cy+size, cx-size:cx+size]
    center_crop = cv2.resize(center_crop, (224, 224))
    crops.append(center_crop)

    return crops



import matplotlib.pyplot as plt

row = aptos_df.sample(1, random_state=0).iloc[0]

img_path = (
    "/kaggle/input/aptos2019-blindness-detection/train_images/"
    + row["id_code"] + ".png"
)

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = model(img, conf=0.25, iou=0.4, verbose=False)

for r in results:
    crops = lightweight_crops(img, r.boxes)

    plt.figure(figsize=(12,4))
    for i, c in enumerate(crops):
        plt.subplot(1, len(crops), i+1)
        plt.imshow(c)
        plt.axis("off")
        plt.title(f"Crop {i+1}")

    plt.suptitle(f"GT Label: {row['label']}")
    plt.show()



import numpy as np
import cv2
import random
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 224
NUM_CLASSES = 5   # abhi 5-class, baad me binary bhi karenge

def preprocess(img):
    img = img.astype(np.float32) / 255.0
    return img

def data_generator(df, yolo_model, batch_size=8, shuffle=True):
    idxs = np.arange(len(df))

    while True:
        if shuffle:
            np.random.shuffle(idxs)

        batch_imgs, batch_labels = [], []

        for i in idxs:
            row = df.iloc[i]
            img_path = (
                "/kaggle/input/aptos2019-blindness-detection/train_images/"
                + row["id_code"] + ".png"
            )

            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # YOLO inference (quiet)
            results = yolo_model(img, conf=0.25, iou=0.4, verbose=False)

            # lightweight crops
            for r in results:
                crops = lightweight_crops(img, r.boxes)

                # pick ONE random crop (speed + regularization)
                crop = random.choice(crops)
                crop = preprocess(crop)

                batch_imgs.append(crop)
                batch_labels.append(row["label"])

                if len(batch_imgs) == batch_size:
                    X = np.array(batch_imgs)
                    y = to_categorical(batch_labels, NUM_CLASSES)
                    yield X, y
                    batch_imgs, batch_labels = [], []



gen = data_generator(train_df, model, batch_size=8)

X, y = next(gen)
print("Batch X:", X.shape)
print("Batch y:", y.shape)
print("First label:", y[0])


