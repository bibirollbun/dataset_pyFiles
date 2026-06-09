# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install -q ultralytics



from ultralytics import YOLO
model=YOLO('yolov8n-cls.pt')
model.predict(source='/kaggle/input/a0-2025-object-detection/Dataset/Test/images/0026df6bcaf93ac5_jpg.rf.510f06f5d706b8924f3a8a411211dcf8.jpg')


import os

label_dir = "/kaggle/input/yolov8format/train/labels"  # your label folder path

# Output folder (optional, to avoid overwriting original labels)
output_dir = "/kaggle/working/train/labels_fixed"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        input_path = os.path.join(label_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with open(input_path, 'r') as f:
            lines = f.readlines()

        fixed_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # skip invalid lines

            class_id = int(parts[0]) - 1  # subtract 1 to start from 0
            if class_id < 0:
                # ignore or handle invalid class ids here
                continue

            fixed_line = " ".join([str(class_id)] + parts[1:])
            fixed_lines.append(fixed_line)

        # Save fixed label
        with open(output_path, 'w') as f:
            f.write("\n".join(fixed_lines) + "\n")

print(f"Fixed labels saved to {output_dir}")



import os

data_yaml = f"""
train: /kaggle/input/yolov8format/train/images
val: /kaggle/input/yolov8format/valid/images

nc: 6
names: ['Apple', 'Banana', 'Grapes', 'Orange', 'Pineapple', 'Watermelon']
"""

with open("/kaggle/working/data.yaml", "w") as f:
    f.write(data_yaml)



from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results=model.train(data='/kaggle/working/data.yaml', epochs=20, imgsz=640)



!yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml save=True



model = YOLO('/kaggle/working/runs/detect/train/weights/best.pt')
results = model.predict(source='/kaggle/input/yolov8format/test/images', save=False, conf=0.25)


import pandas as pd
import os
import json

submission = []

for r in results:
    file_name = os.path.basename(r.path).split('.')[0]  # e.g., '00123'
    file_id = "img" + file_name[-4:]  # Convert to img format like 'img123'

    bboxes = []

    for box in r.boxes:
        x_min, y_min, x_max, y_max = map(float, box.xyxy[0])
        class_id = int(box.cls[0])
        conf = float(box.conf[0])

        bboxes.append({
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "class": class_id,
            "confidence": conf
        })

    # Ensure the bounding_boxes is a proper string with double quotes escaped
    bbox_str = json.dumps(bboxes).replace('"', '""')

    submission.append({
        "ID": file_id,
        "bounding_boxes": f'"{bbox_str}"'  # Enclose entire JSON string in quotes
    })

df = pd.DataFrame(submission)
df.to_csv("submission.csv", index=False)
print("✅ submission.csv saved!")



import os

img_dir = "/kaggle/input/a0-2025-object-detection/Dataset/Train/images"
label_dir = "/kaggle/input/a0-2025-object-detection/Dataset/Train/labels"

img_files = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]
missing_labels = []
empty_labels = []
bad_format = []

for img in img_files:
    base = os.path.splitext(img)[0]
    label_file = os.path.join(label_dir, base + ".txt")

    if not os.path.exists(label_file):
        missing_labels.append(img)
        continue

    with open(label_file) as f:
        lines = f.readlines()
        if not lines:
            empty_labels.append(img)
            continue
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                bad_format.append((img, line))
                break

print(f"Total images: {len(img_files)}")
print(f"Missing labels: {len(missing_labels)}")
print(f"Empty labels: {len(empty_labels)}")
print(f"Badly formatted labels: {len(bad_format)}")





