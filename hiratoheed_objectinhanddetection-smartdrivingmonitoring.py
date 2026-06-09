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


import os
import numpy as np
import random
from tensorflow.keras.preprocessing import image

# Path to training data
train_dir = "/kaggle/input/state-farm-distracted-driver-detection/imgs/train"   

img_size = (224, 224)   # resize for MobileNet or CNN
x = []
y = []

class_names = sorted(os.listdir(train_dir))

# Loop through each class folder
for class_name in class_names:
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path):
        img_files = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        # Shuffle images randomly
        random.shuffle(img_files)

        # Drop 20% of images randomly
        keep_count = int(0.65 * len(img_files))  # keep 80%
        keep_files = img_files[:keep_count]

        for img_file in keep_files:
            img_path = os.path.join(class_path, img_file)
            try:
                # Load and preprocess image
                img = image.load_img(img_path, target_size=img_size)
                img_array = image.img_to_array(img) / 255.0
                x.append(img_array)
                y.append(class_name)   # store label
            except:
                print(f"Skipped file: {img_path}")

# Convert to numpy arrays
x = np.array(x)
y = np.array(y)

# Print dataset stats after random dropping
print("Total images after randomly dropping 35% per class:", len(x))
for class_name in class_names:
    count = sum(y == class_name)
    print(f"Class '{class_name}': {count} images")




import numpy as np

# Shuffle dataset consistently
indices = np.arange(len(x))
np.random.shuffle(indices)

x = x[indices]
y = y[indices]

print("After shuffling:")
print("X shape:", x.shape)
print("y shape:", y.shape)



from sklearn.model_selection import train_test_split
# stratify = y means to do balanced distribution of the labels 
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42, stratify=y
)      

print("Train set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)


import numpy as np

# Function to count images per class
def count_classes(y_data, name="Dataset"):
    unique_classes, counts = np.unique(y_data, return_counts=True)
    print(f"\n{name} class distribution:")
    for cls, cnt in zip(unique_classes, counts):
        print(f"Class '{cls}': {cnt} images")

# Check distribution
count_classes(y_train, "Train set")
count_classes(y_test, "Test set")


X_test


!pip install ultralytics



X_test.shape


import os
from tensorflow.keras.preprocessing.image import array_to_img

def save_dataset(X, y, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    for img_array, label in zip(X, y):
        class_dir = os.path.join(base_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)
        img = array_to_img(img_array)  # Convert numpy -> PIL image
        img.save(os.path.join(class_dir, f"{np.random.randint(1e9)}.jpg"))

# Save train and test sets
save_dataset(X_train, y_train, "driver_dataset/train")
save_dataset(X_test, y_test, "driver_dataset/val")



X_test.shape


from ultralytics import YOLO

# Load YOLOv8 classification model (nano version for speed, you can try small/medium too)
model = YOLO("yolov8n-cls.pt")

# Train
model.train(
    data="driver_dataset",   # path with train/val folders
    epochs=10,
    imgsz=224,
    batch=32,
    patience = 4
)



metrics = model.val()
print(metrics)



X_test.shape


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import cv2

y_pred = []

for img in X_test:
    # Ensure image is uint8 and correct shape
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)  # if normalized [0,1]
    
    if img.shape[-1] != 3:  
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # if grayscale
    
    # Predict with YOLOv8
    results = model.predict(img, imgsz=224, verbose=False)
    pred_class = results[0].probs.top1
    y_pred.append(pred_class)

# Convert to numpy
y_pred = np.array(y_pred)

# Convert predicted indices back to class names
y_pred_labels = [class_names[i] for i in y_pred]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_labels, labels=class_names)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - YOLOv8 Classification")
plt.show()

# Classification Report
print(classification_report(y_test, y_pred_labels, target_names=class_names))





