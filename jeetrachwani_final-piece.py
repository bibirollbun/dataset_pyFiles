import pandas as pd

# Step 1: Load both CSVs
train_labels = pd.read_csv('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')
class_info = pd.read_csv('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv')

# Step 2: Merge on patientId
df = pd.merge(train_labels, class_info, on='patientId', how='left')

# Step 3: Check the merged DataFrame
print(df.head())



import os
from PIL import Image
import pydicom
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Paths
IMAGES_DIR = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images'
OUTPUT_DIR = '/kaggle/working/pneumonia_yolo'

# Create folders
os.makedirs(f'{OUTPUT_DIR}/images/train', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/images/val', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/labels/train', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/labels/val', exist_ok=True)

# Map classes
class_mapping = {
    'Normal': 0,
    'No Lung Opacity / Not Normal': 1,
    'Lung Opacity': 2
}

# Select unique image IDs
image_ids = df['patientId'].unique()

# [OPTIONAL]: Subsample if needed to save time (take first 6000 images)
image_ids = image_ids[:6000]

# Split into train and validation
train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=42)

# Function to process each split
def process_images(ids, split='train'):
    for img_id in tqdm(ids):
        img_path = os.path.join(IMAGES_DIR, img_id + '.dcm')
        
        try:
            # Load DICOM image
            dicom = pydicom.dcmread(img_path)
            img = dicom.pixel_array
            img = Image.fromarray(img).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_id}: {e}")
            continue
        
        # Resize image
        img = img.resize((512, 512))
        
        # Save image
        save_img_path = f'{OUTPUT_DIR}/images/{split}/{img_id}.jpg'
        img.save(save_img_path)
        
        # Prepare label lines
        records = df[df['patientId'] == img_id]
        
        label_lines = []
        for idx, row in records.iterrows():
            if row['class'] == 'Normal':
                continue  # No boxes for Normal

            x = row['x']
            y = row['y']
            w = row['width']
            h = row['height']
            
            # Coordinates are in 1024x1024 space
            x_center = (x + w/2) / 1024
            y_center = (y + h/2) / 1024
            w_norm = w / 1024
            h_norm = h / 1024
            
            class_id = class_mapping[row['class']]
            
            label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        # Save labels
        save_lbl_path = f'{OUTPUT_DIR}/labels/{split}/{img_id}.txt'
        with open(save_lbl_path, 'w') as f:
            for line in label_lines:
                f.write(line + '\n')

# Process Train and Val splits
process_images(train_ids, split='train')
process_images(val_ids, split='val')



# Create YAML content
yaml_content = """
path: /kaggle/working/pneumonia_yolo
train: images/train
val: images/val

nc: 3
names: ['Normal', 'No Opacity', 'Opacity']
"""

# Save it
with open('/kaggle/working/pneumonia_yolo/pneumonia.yaml', 'w') as f:
    f.write(yaml_content)

print("✅ pneumonia.yaml created successfully!")



!pip install -q ultralytics



from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8n.pt')  # nano model for faster training, or yolov8s.pt for small model

# Start training
model.train(
    data='/kaggle/working/pneumonia_yolo/pneumonia.yaml',  # Our YAML file
    imgsz=512,
    epochs=30,
    batch=16,
    name='pneumonia_yolov8n',
    workers=3,  # Kaggle gives 2 CPUs typically
    patience=5,
    optimizer='Adam',  # or 'Adam'
    verbose=True
)



# Assuming df is your merged CSV (from Part 1)
# and train_ids, val_ids are your validation ids.

subset_df = df[df['patientId'].isin(val_ids.tolist())]

# Create subsets
normal_ids = subset_df[subset_df['class'] == 'Normal']['patientId'].unique()
no_opacity_ids = subset_df[subset_df['class'] == 'No Lung Opacity / Not Normal']['patientId'].unique()
opacity_ids = subset_df[subset_df['class'] == 'Lung Opacity']['patientId'].unique()

print(f"Normal images available: {len(normal_ids)}")
print(f"No Opacity images available: {len(no_opacity_ids)}")
print(f"Opacity images available: {len(opacity_ids)}")



import cv2
import matplotlib.pyplot as plt
import random
import os
from ultralytics import YOLO

# Load model
model = YOLO('/kaggle/working/runs/detect/pneumonia_yolov8n/weights/best.pt')

# Custom class names
custom_names = {0: "Normal", 1: "No Opacity", 2: "Opacity"}

# Validation image directory
val_dir = '/kaggle/working/pneumonia_yolo/images/val/'

# Function to test a few images from a class
def test_class_images(patient_ids, label_name, sample_size=5):
    sampled = random.sample(list(patient_ids), min(sample_size, len(patient_ids)))
    
    for pid in sampled:
        img_path = os.path.join(val_dir, pid + '.jpg')
        
        results = model(img_path, conf=0.25)
        preds = results[0].boxes

        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw boxes manually
        for box, cls, conf in zip(preds.xyxy, preds.cls, preds.conf):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            label = f"{custom_names[class_id]} {conf:.2f}"

            # Draw red box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Put label text
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Show results
        plt.figure(figsize=(8,8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"True Label: {label_name} | ID: {pid}")
        plt.show()

# Example: Test Normal Images
print("Testing Normal Images:")
test_class_images(normal_ids, label_name="Normal")

# Example: Test No Opacity Images
print("Testing No Opacity / Not Normal Images:")
test_class_images(no_opacity_ids, label_name="No Opacity / Not Normal")

# Example: Test Opacity (Pneumonia) Images
print("Testing Opacity Images:")
test_class_images(opacity_ids, label_name="Opacity")



from IPython.display import FileLink

# Create a download link for the best.pt model
FileLink(r'/kaggle/working/runs/detect/pneumonia_yolov8n/weights/best.pt')



import pandas as pd
import matplotlib.pyplot as plt

# Load training results
results_csv = '/kaggle/working/runs/detect/pneumonia_yolov8n/results.csv'
df = pd.read_csv(results_csv)

# Plot Losses
plt.figure(figsize=(10,6))
plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
plt.plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss')
plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision (B)')
plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall (B)')
plt.title('Losses, Precision and Recall over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True)
plt.show()

# Plot mAP (50) and mAP (50-95)
plt.figure(figsize=(10,6))
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
plt.title('Mean Average Precision over Epochs')
plt.xlabel('Epoch')
plt.ylabel('mAP Value')
plt.legend()
plt.grid(True)
plt.show()

# Plot Box Loss vs Epochs separately for clarity
plt.figure(figsize=(8,5))
plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss', color='red')
plt.title('Box Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot Class Loss vs Epochs separately
plt.figure(figsize=(8,5))
plt.plot(df['epoch'], df['train/cls_loss'], label='Class Loss', color='blue')
plt.title('Classification Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()



# Plot Precision and Recall over Epochs
plt.figure(figsize=(10,6))
plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision (B)')
plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall (B)')
plt.title('Precision and Recall over Epochs (Threshold Stability Analysis)')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True)
plt.show()



plt.figure(figsize=(10,6))
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
plt.title('Threshold Stability Analysis (mAP@0.5 vs Epoch)')
plt.xlabel('Epoch')
plt.ylabel('mAP@0.5')
plt.legend()
plt.grid(True)
plt.show()





