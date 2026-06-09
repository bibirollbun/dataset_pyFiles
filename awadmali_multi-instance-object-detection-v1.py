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


# Cell 1: Install Libraries
!pip install ultralytics -q
print("✅ Libraries installed.")


# Cell 2: Import Libraries and Define Paths
import yaml
import os
import pandas as pd
from ultralytics import YOLO
from tqdm.notebook import tqdm

# Define the CORRECT paths, paying attention to case-sensitivity ('Starter_Dataset')
train_images_path = '/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/clutter/train/images/'
val_images_path = '/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/clutter/val/images/'
test_images_path = '/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/TestImages/images/'

# Verify that the paths exist to prevent errors
assert os.path.exists(train_images_path), f"Path not found: {train_images_path}"
assert os.path.exists(val_images_path), f"Path not found: {val_images_path}"
assert os.path.exists(test_images_path), f"Path not found: {test_images_path}"

print("✅ All paths are correct and verified.")


# Cell 3: Create data.yaml Configuration File
data_config = {
    'train': train_images_path,
    'val': val_images_path,
    'nc': 1,
    'names': ['soup_can']
}

with open('data.yaml', 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)

print("✅ data.yaml file created successfully.")
!cat data.yaml # Optional: display the content of the file


# Cell 4: Train the Model
print("--- Starting Model Training ---")

# Load the pre-trained model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='data.yaml',
    epochs=15,
    imgsz=640,
    project='soup_can_final_detection',
    name='run1',
    exist_ok=True
)

print("\n--- ✅ Model Training Complete ---")


# Cell 5: Generate Submission File
print("--- Generating Submission File ---")

# Load the best trained model
trained_model = YOLO('/kaggle/working/soup_can_final_detection/run1/weights/best.pt')

# Get a list of all test image files
test_image_files = [os.path.join(test_images_path, f) for f in os.listdir(test_images_path)]

predictions = []

# Process each test image
for img_path in tqdm(test_image_files, desc="Processing Test Images"):
    res = trained_model(img_path, verbose=False) # verbose=False keeps the output clean
    
    prediction_string = []
    for box in res[0].boxes:
        xywhn = box.xywhn[0]
        conf = box.conf[0]
        class_id = int(box.cls[0])
        prediction_string.append(f"{class_id} {conf:.4f} {xywhn[0]:.4f} {xywhn[1]:.4f} {xywhn[2]:.4f} {xywhn[3]:.4f}")
        
    image_name = os.path.basename(img_path)
    predictions.append({
        'image_id': image_name,
        'prediction_string': ' '.join(prediction_string)
    })

# Create and save the submission DataFrame
submission_df = pd.DataFrame(predictions)
submission_df.to_csv('submission.csv', index=False)

print("\n✅ Submission file created successfully!")
print("Top 5 rows of your submission file:")
print(submission_df.head())

