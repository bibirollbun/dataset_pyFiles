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



import pandas as pd
import os

def convert_to_yolo(size, box):
    """
    Converts (xmin, ymin, xmax, ymax) box format to YOLO's
    (x_center, y_center, width, height) normalized format.
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    
    width = box[2] - box[0]
    height = box[3] - box[1]
    
    x_center = x_center * dw
    y_center = y_center * dh
    width = width * dw
    height = height * dh
    
    return (x_center, y_center, width, height)

csv_file_path = '/kaggle/input/the-carnival-vision-challenge/train/_annotations.csv' 

output_dir = '/kaggle/working/labels'

try:
    df = pd.read_csv("/kaggle/input/the-carnival-vision-challenge/train/_annotations.csv")
except FileNotFoundError:
    print(f"Error: '{csv_file_path}' not found. Please check the file path in the '+ Add data' section.")
    exit()

os.makedirs(output_dir, exist_ok=True)

for filename in df['filename'].unique():
    image_df = df[df['filename'] == filename]
    
    image_width = image_df.iloc[0]['width']
    image_height = image_df.iloc[0]['height']
    
    label_filename = os.path.splitext(filename)[0] + '.txt'
    output_path = os.path.join(output_dir, label_filename)
    
    with open(output_path, 'w') as f:
        for _, row in image_df.iterrows():
            class_index = 0
            
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            
            yolo_box = convert_to_yolo((image_width, image_height), (xmin, ymin, xmax, ymax))
            
            f.write(f"{class_index} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}\n")



!pip install -U ultralytics


import os
import random
import shutil

source_images_dir = '/kaggle/input/the-carnival-vision-challenge/train/images'
source_labels_dir = '/kaggle/working/labels'

output_dir = '/kaggle/working/dataset'

train_ratio = 0.8

os.makedirs(os.path.join(output_dir, 'images/train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images/val'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels/train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels/val'), exist_ok=True)

all_filenames = [f for f in os.listdir(source_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(all_filenames)

split_index = int(len(all_filenames) * train_ratio)

train_filenames = all_filenames[:split_index]
val_filenames = all_filenames[split_index:]

def copy_files(filenames, split_type):
    for filename in filenames:
        base_name = os.path.splitext(filename)[0]
        
        
        source_image_path = os.path.join(source_images_dir, filename)
        dest_image_path = os.path.join(output_dir, f'images/{split_type}', filename)
        shutil.copy(source_image_path, dest_image_path)
        
        
        source_label_path = os.path.join(source_labels_dir, f'{base_name}.txt')
        dest_label_path = os.path.join(output_dir, f'labels/{split_type}', f'{base_name}.txt')
        if os.path.exists(source_label_path):
            shutil.copy(source_label_path, dest_label_path)

copy_files(train_filenames, 'train')
copy_files(val_filenames, 'val')

print("Dataset successfully split!")
print(f"Total images: {len(all_filenames)}")
print(f"Training images: {len(train_filenames)}")
print(f"Validation images: {len(val_filenames)}")


from ultralytics import YOLO
yaml_content = """

path: "/kaggle/working/dataset"  

# Directories for training and validation images (relative to the path above)
train: "images/train"
val: "images/val"

# Class names - since you only have 'pothole', it is at index 0
names:
  0: pothole

"""

with open("data.yaml", "w") as f:
    f.write(yaml_content)

print("data.yaml file created successfully!")



model = YOLO('yolov8s.pt')

results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=780,
    project='/kaggle/working/pothole_detection',
    degrees=20.0,         
    translate=0.2,        
    scale=0.2,            
    shear=2.0,            
    perspective=0.001,    
    
    flipud=0.5,           
    fliplr=0.5,           
    
    hsv_h=0.015,          
    hsv_s=0.7,            
    hsv_v=0.7,            
    
    mosaic=1.0,           
    mixup=0.2
)


import os
import pandas as pd
from ultralytics import YOLO

model = YOLO('/kaggle/working/pothole_detection/train/weights/best.pt')
test_directory = '/kaggle/input/the-carnival-vision-challenge/test images/images'
sample_submission_path = '/kaggle/input/the-carnival-vision-challenge/sample submission.csv'
output_csv_path = '/kaggle/working/submission.csv'

submission_df = pd.read_csv(sample_submission_path)
image_filenames = submission_df['filename'].unique()
full_test_paths = [os.path.join(test_directory, f) for f in image_filenames]
results_list = model.predict(source=full_test_paths, verbose=False)
results_dict = {os.path.basename(r.path): r for r in results_list}

for index, row in submission_df.iterrows():
    filename = row['filename']
    result = results_dict.get(filename)
    
    xmin, ymin, xmax, ymax = "", "", "", ""

    if result and len(result.boxes) > 0:
        best_box = result.boxes[result.boxes.conf.argmax()]
        coords = best_box.xyxy[0].cpu().numpy()
        xmin, ymin, xmax, ymax = coords
        
    submission_df.loc[index, ['xmin', 'ymin', 'xmax', 'ymax']] = xmin, ymin, xmax, ymax
        
final_columns = ['row_id_column_name', 'filename', 'xmin', 'ymin', 'xmax', 'ymax']
submission_df_final = submission_df[final_columns]

submission_df_final.replace("", 0, inplace=True)

coord_cols = ['xmin', 'ymin', 'xmax', 'ymax']
submission_df_final[coord_cols] = submission_df_final[coord_cols].astype(int)

submission_df_final.to_csv(output_csv_path, index=False)





