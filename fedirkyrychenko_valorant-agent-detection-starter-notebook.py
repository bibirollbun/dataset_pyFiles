# VALORANT AGENT DETECTION – STARTER NOTEBOOK

import os
import random
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw

# === Paths ===
DATA_DIR = "/kaggle/input/valorant-agent-detection"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train/images")
TRAIN_ANN_DIR = os.path.join(DATA_DIR, "train/annotations")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test/images")

# === Class names ===
CLASSES = ["Phoenix", "Jett", "Brimstone", "Sage"]



def parse_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    labels = []

    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bbox = obj.find("bndbox")
        box = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text)
        ]
        boxes.append(box)
        labels.append(name)
    return boxes, labels

image_files = sorted(os.listdir(TRAIN_IMG_DIR))
examples = random.sample(image_files, 5)

for example in examples:
    img_path = os.path.join(TRAIN_IMG_DIR, example)
    ann_path = os.path.join(TRAIN_ANN_DIR, example.replace(".jpg", ".xml"))

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    boxes, labels = parse_annotation(ann_path)

    for box, label in zip(boxes, labels):
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1] - 10), label, fill="white")

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"Example: {example}")
    plt.axis("off")
    plt.show()


test_files = sorted(os.listdir(TEST_IMG_DIR))
submission = []

for fname in test_files:
    image_id = fname.replace(".jpg", "")

    # Example: generate 1–2 fake predictions per image
    preds = []
    for _ in range(random.randint(1, 2)):
        label = random.choice(CLASSES)
        x1 = random.randint(0, 500)
        y1 = random.randint(0, 500)
        x2 = x1 + random.randint(20, 100)
        y2 = y1 + random.randint(20, 100)
        preds.append(f"{label} {x1} {y1} {x2} {y2}")

    prediction_string = " ".join(preds)
    submission.append({
        "Id": image_id,
        "PredictionString": prediction_string
    })

df_submission = pd.DataFrame(submission)
df_submission.to_csv("submission.csv", index=False)
df_submission.head()

