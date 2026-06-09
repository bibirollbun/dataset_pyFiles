#0. LOAD DATASET, TESTSET.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_path="/kaggle/input/severstal-steel-defect-detection/train.csv"
test_path="/kaggle/input/severstal-steel-defect-detection/train_images/"
df=pd.read_csv(train_path)


#1.RLE DECODE

import cv2
import matplotlib.pyplot as plt

def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded (RLE) mask into a binary mask array.

    :param mask_rle: RLE string (e.g., '1 3 10 5')
    :param shape: Tuple (height, width) of the original image (e.g., (256, 1600))
    :return: NumPy array (H, W) with values 0/1.
    """
    if isinstance(mask_rle, float) and np.isnan(mask_rle):
        # Return an empty mask when EncodedPixels is NaN (no defect)
        return np.zeros(shape, dtype=np.uint8)

    # Parse RLE into start positions and lengths
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    
    # Convert to zero-based indexing
    starts -= 1

    # Create a flat mask
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    # Fill mask based on RLE
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1

    # Reshape and transpose (RLE uses column-major order)
    return mask.reshape(shape[::-1]).T



# 2. DEFECT RESEARCH

import random

IMAGE_FOLDER = '/kaggle/input/severstal-steel-defect-detection/train_images/'
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 1600
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)

# Color definitions (RGB) for each defect class
COLOR_MAP = {
    1: (255, 0, 0),    # Class 1 – Red
    2: (0, 255, 0),    # Class 2 – Green
    3: (0, 0, 255),    # Class 3 – Blue
    4: (255, 255, 0)   # Class 4 – Yellow
}

# Filter rows containing annotated defects
df_defects = df.dropna(subset=['EncodedPixels']).copy()
images_with_defects_ids = df_defects['ImageId'].unique()

if len(images_with_defects_ids) == 0:
    print("ERROR: No defect annotations found (all EncodedPixels are NaN). Check the DataFrame.")
else:
    # Select a random image that contains at least one defect
    random_image_id = random.choice(images_with_defects_ids)
    
    # Extract all rows related to the chosen image
    defect_rows = df[df['ImageId'] == random_image_id].copy().fillna(value={'EncodedPixels': ''})

    # Load the image
    image_path = os.path.join(IMAGE_FOLDER, random_image_id)
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print(f"ERROR: Could not load image from {image_path}. Check the path.")
    else:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Canvas for color overlays
        colored_mask = np.zeros(original_image.shape, dtype=np.float32)
        
        detected_classes = []

        # Decode masks for each defect class in this image
        for index, row in defect_rows.iterrows():
            class_id = row['ClassId']
            rle = row['EncodedPixels']
            
            mask = rle_decode(rle, IMAGE_SHAPE)
            
            if np.sum(mask) > 0:  # Only process non-empty masks
                detected_classes.append(class_id)
                color = COLOR_MAP.get(class_id)
                
                colored_mask[mask == 1] = color
        
        # Visualization
        plt.figure(figsize=(18, 10))
        plt.imshow(original_image)
        
        # Overlay colored mask with transparency
        if len(detected_classes) > 0:
            plt.imshow(colored_mask.astype(np.uint8), alpha=0.5)
            title = (f'Image: {random_image_id} | Detected defects: '
                     f'{", ".join([f"Class {c} {COLOR_MAP[c]}" for c in sorted(detected_classes)])}')
        else:
            title = f'Image: {random_image_id} | No defects detected (unexpected).'

        plt.title(title)
        plt.axis('off')
        plt.show()



#3. CONVERTING TO YOLO BBOXES
import os
from tqdm import tqdm

# Constants
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 1600
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)

# Assumes rle_decode(mask_rle, shape) is already defined

def mask_to_yolo_bbox(mask):
    """
    Converts a 2D binary mask into normalized YOLO bounding box coordinates.
    
    :param mask: 2D binary mask (np.uint8) decoded from RLE.
    :return: (x_center, y_center, w, h) normalized to 0–1, or None if mask is empty.
    """
    
    # Skip empty masks
    if np.sum(mask) == 0:
        return None
    
    # Find external contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    # Use all points to form a global bounding box
    all_points = np.concatenate(contours)
    
    # Compute bounding rectangle (absolute pixel values)
    x_abs, y_abs, w_abs, h_abs = cv2.boundingRect(all_points)
    
    # Normalize to YOLO format
    x_center = (x_abs + w_abs / 2) / IMAGE_WIDTH
    y_center = (y_abs + h_abs / 2) / IMAGE_HEIGHT
    w_norm = w_abs / IMAGE_WIDTH
    h_norm = h_abs / IMAGE_HEIGHT
    
    return x_center, y_center, w_norm, h_norm

# Output directory for YOLO annotation files
YOLO_LABELS_DIR = 'yolo_labels/'
os.makedirs(YOLO_LABELS_DIR, exist_ok=True)

# Container for statistical analysis
yolo_data = []

# Filter rows with defects only
df_defects = df.dropna(subset=['EncodedPixels']).copy()

# Group all defects belonging to the same image
grouped_defects = df_defects.groupby('ImageId')

print(f"Starting conversion of {len(grouped_defects)} images with defects...")

for image_id, group in tqdm(grouped_defects, desc="RLE → YOLO BBox"):
    
    label_content = []
    
    # Process all defects for this image
    for index, row in group.iterrows():
        rle = row['EncodedPixels']
        
        # YOLO classes must start from 0; dataset uses 1–4 → subtract 1
        class_id_yolo = row['ClassId'] - 1 
        
        # Decode RLE mask
        mask = rle_decode(rle, IMAGE_SHAPE)
        
        # Compute YOLO bounding box
        bbox_data = mask_to_yolo_bbox(mask)
        
        if bbox_data:
            x_center, y_center, w_norm, h_norm = bbox_data
            
            # YOLO txt format: class x_center y_center width height
            line = f"{class_id_yolo} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            label_content.append(line)
            
            # Collect statistics
            yolo_data.append({
                'ClassId_YOLO': class_id_yolo,
                'width_norm': w_norm,
                'height_norm': h_norm,
                'aspect_ratio': w_norm / h_norm
            })

    # Save YOLO label file if any defects were found
    if label_content:
        label_filename = os.path.join(YOLO_LABELS_DIR, image_id.replace('.jpg', '.txt'))
        with open(label_filename, 'w') as f:
            f.write('\n'.join(label_content))

print("\n--- Conversion complete! ---")
print(f"YOLO annotation files saved to: {YOLO_LABELS_DIR}")

# Create DataFrame for further analysis
geometry_df = pd.DataFrame(yolo_data)
print(f"\nGeometry DataFrame ready. Processed {len(geometry_df)} individual defects.")



#5. Statistics of BBOXES
import seaborn as sns

# Ensure geometry_df was created in the previous step
if 'geometry_df' not in locals():
    print("ERROR: 'geometry_df' not found. Run the RLE → YOLO conversion cell first.")
else:
    # Convert YOLO class IDs back to 1–4 for clarity in statistical analysis
    geometry_df['ClassId'] = geometry_df['ClassId_YOLO'] + 1
    
    # Filter out extreme ratios (division by zero or outliers)
    geometry_df_filtered = geometry_df[(geometry_df['aspect_ratio'] > 0.001) & 
                                       (geometry_df['aspect_ratio'] < 1000)]

    print("--- Bounding Box Size Statistics (Normalized Values 0–1) ---")
    
    # 1. Overall statistics
    overall_stats = geometry_df_filtered[['width_norm', 'height_norm', 'aspect_ratio']].agg(['mean', 'median', 'min', 'max'])
    print("\nOverall Statistics (All Classes):")
    print(overall_stats.to_markdown(floatfmt=".4f"))
    
    print("\n" + "-"*50)

    # 2. Statistics grouped by class
    class_stats = geometry_df_filtered.groupby('ClassId')[['width_norm', 'height_norm', 'aspect_ratio']].agg(['mean', 'median', 'max'])
    print("\nStatistics by Class:")
    print(class_stats.to_markdown(floatfmt=".4f"))

    print("\n" + "-"*50)

    # 3. Visualization: Normalized Width distribution
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.violinplot(x='ClassId', y='width_norm', data=geometry_df_filtered)
    plt.title('Distribution of Normalized Width (W)')
    plt.xlabel('Class ID')

    # 4. Visualization: Normalized Height distribution
    plt.subplot(1, 3, 2)
    sns.violinplot(x='ClassId', y='height_norm', data=geometry_df_filtered)
    plt.title('Distribution of Normalized Height (H)')
    plt.xlabel('Class ID')

    # 5. Visualization: Aspect ratio distribution (limited Y-axis)
    plt.subplot(1, 3, 3)
    sns.violinplot(x='ClassId', y='aspect_ratio', data=geometry_df_filtered)
    plt.ylim(0, 10)  # Limit view to typical aspect ratios
    plt.title('Aspect Ratio (W/H)')
    plt.xlabel('Class ID')
    
    plt.tight_layout()
    plt.show()

    print("\n--- Results Interpretation ---")



#6. ULTRAYTICS
!pip install ultralytics


#7. Import YOLOv8 (medium)
import torch
import os
from ultralytics import YOLO

# Constants
NUM_CLASSES = 4  # Number of defect classes
MODEL_NAME = 'yolov8m.pt'  # Pretrained YOLOv8 model
DATA_CONFIG_NAME = 'severstal_defects.yaml'  # Data config file (created later)

print(f"Loading pretrained model: {MODEL_NAME}...")

# 1. Load pretrained model (COCO)
try:
    model = YOLO(MODEL_NAME)
    print("\nModel successfully loaded (YOLOv8-m).")
except Exception as e:
    print(f"Error while loading model: {e}")
    print("Verify the 'ultralytics' installation and internet access.")

# 2. Architecture summary
print("\n--- Model Summary ---")
print(f"Number of pretrained COCO classes: {model.model.nc}")
print(f"Target number of classes for fine-tuning: {NUM_CLASSES}")

# The model will automatically adjust its final layer during training.



#8. YAML Configuration
import shutil
from sklearn.model_selection import train_test_split
import yaml
from tqdm import tqdm

# Constants from previous steps
IMAGE_FOLDER = '/kaggle/input/severstal-steel-defect-detection/train_images/'  # Directory with JPG images
LABELS_FOLDER = '/kaggle/working/yolo_labels'  # Directory with generated YOLO .txt labels
WORKING_DIR = 'Severstal_YOLO_Dataset/'  # Output dataset directory

# Train–validation split parameters
VAL_SIZE = 0.2
RANDOM_STATE = 42
NUM_CLASSES = 4  # Defect classes 1–4

# 1. Build list of labeled images (images that have a .txt annotation file)
labeled_images = [f.replace('.txt', '.jpg') for f in os.listdir(LABELS_FOLDER) if f.endswith('.txt')]

if not labeled_images:
    print("ERROR: No annotation files found in 'yolo_labels/'. Run the RLE conversion script first.")
else:
    print(f"Found {len(labeled_images)} images with bounding-box annotations.")

    # 2. Train–validation split
    train_images, val_images = train_test_split(
        labeled_images, 
        test_size=VAL_SIZE, 
        random_state=RANDOM_STATE
    )

    print(f"Training set: {len(train_images)} images.")
    print(f"Validation set: {len(val_images)} images.")

    # 3. Create YOLO directory structure
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
        
    os.makedirs(WORKING_DIR, exist_ok=True)
    
    yolo_structure = {
        'train_images': os.path.join(WORKING_DIR, 'train', 'images'),
        'train_labels': os.path.join(WORKING_DIR, 'train', 'labels'),
        'val_images': os.path.join(WORKING_DIR, 'val', 'images'),
        'val_labels': os.path.join(WORKING_DIR, 'val', 'labels')
    }

    for path in yolo_structure.values():
        os.makedirs(path, exist_ok=True)

    # 4. Copy images and labels
    def copy_files(image_list, image_dest, label_dest):
        for image_id in tqdm(image_list, desc="Copying files"):
            # Copy image
            shutil.copy(
                os.path.join(IMAGE_FOLDER, image_id), 
                os.path.join(image_dest, image_id)
            )
            # Copy label
            label_filename = image_id.replace('.jpg', '.txt')
            shutil.copy(
                os.path.join(LABELS_FOLDER, label_filename), 
                os.path.join(label_dest, label_filename)
            )

    print("\n--- Copying Training Set ---")
    copy_files(train_images, yolo_structure['train_images'], yolo_structure['train_labels'])
    
    print("\n--- Copying Validation Set ---")
    copy_files(val_images, yolo_structure['val_images'], yolo_structure['val_labels'])

    # 5. Generate YAML configuration file
    yaml_config = {
        'path': f'../{WORKING_DIR}',  # Base path to the dataset
        'train': 'train/images',      # Training images
        'val': 'val/images',          # Validation images
        
        # Number of classes and their names (YOLO always starts indexing at 0)
        'nc': NUM_CLASSES,
        'names': {
            0: 'Class_1_Porosity',
            1: 'Class_2_Patches',
            2: 'Class_3_Scratches',
            3: 'Class_4_Inclusions'
        }
    }
    
    # Save YAML config
    yaml_path = os.path.join(WORKING_DIR, 'severstal_defects.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)

    print("\n--- Dataset Structure Ready ---")
    print(f"YAML configuration saved as: {yaml_path}")
    print("\nReady to start training YOLOv8.")



import torch
from ultralytics import YOLO

# Model and dataset paths
MODEL_TO_TRAIN = 'yolov8m.pt'  # Ultralytics will download it automatically if needed
YAML_CONFIG_PATH = 'Severstal_YOLO_Dataset/severstal_defects.yaml'

# Training parameters
NEW_EPOCHS = 60
IMG_SIZE = 1024
BATCH_SIZE = 8
NAME = 'severstal_yolov8m_v1'
PATIENCE = 20

print(f"Starting YOLOv8 Medium training for {NEW_EPOCHS} epochs...")

# 1. Load the YOLOv8 Large model
try:
    model = YOLO(MODEL_TO_TRAIN)
except Exception as e:
    print(f"ERROR: Failed to load model {MODEL_TO_TRAIN}. {e}")
    raise

# 2. Run training
results = model.train(
    data=YAML_CONFIG_PATH,
    imgsz=IMG_SIZE,
    epochs=NEW_EPOCHS,
    batch=BATCH_SIZE,
    name=NAME,
    patience=PATIENCE,
    pretrained=True,
    amp=True
)



import cv2
import os
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import time  # For measuring inference FPS

# 1. Paths and model
BEST_MODEL_PATH = '/kaggle/input/yolo-v8m-v2/pytorch/default/1/best_YOLO_v8m_v2.pt'
TEST_IMAGE_DIR = test_path
OUTPUT_VIDEO_FILE = 'severstal_realtime_fps_demo_5_v8m_v2.mp4'

# 2. Virtual "strip" and motion parameters
IMAGES_PER_STRIP = 200
IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 256
VIRTUAL_STRIP_WIDTH = IMAGE_WIDTH * IMAGES_PER_STRIP
FPS = 15
PIXELS_PER_FRAME = 55  # Strip sliding speed

# Output video resolution
OUTPUT_WIDTH = IMAGE_WIDTH
OUTPUT_HEIGHT = IMAGE_HEIGHT
IMG_SIZE_INFERENCE = 1024
CONF_THRESHOLD = 0.50  # Confidence threshold for detection

# 3. Load YOLO model
print(f"Loading model from: {BEST_MODEL_PATH}...")
try:
    model = YOLO(BEST_MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load model. {e}")
    exit()

# 4. Prepare clean virtual strip
image_files = sorted([f for f in os.listdir(TEST_IMAGE_DIR) if f.endswith('.jpg')])

if len(image_files) < IMAGES_PER_STRIP:
    print(f"ERROR: Need at least {IMAGES_PER_STRIP} test images.")
    exit()

images_to_stitch = image_files[:IMAGES_PER_STRIP]
strip_parts = [cv2.imread(os.path.join(TEST_IMAGE_DIR, img_name)) for img_name in images_to_stitch]

if any(part is None for part in strip_parts):
    print("ERROR: Some test images could not be loaded.")
    exit()

virtual_strip = cv2.hconcat(strip_parts)
print(f"Created virtual strip: {VIRTUAL_STRIP_WIDTH}x{IMAGE_HEIGHT}")

# 5. Setup VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, FPS, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

if not video_writer.isOpened():
    print("ERROR: Failed to initialize VideoWriter.")
    exit()

total_frames = (VIRTUAL_STRIP_WIDTH - OUTPUT_WIDTH) // PIXELS_PER_FRAME

# 6. Inference loop and FPS measurement
print(f"\nGenerating video ({total_frames} frames)...")

for i in tqdm(range(total_frames), desc="Generating Real-Time Demo"):
    
    # Extract current frame from virtual strip
    start_x = i * PIXELS_PER_FRAME
    end_x = start_x + OUTPUT_WIDTH
    current_frame = virtual_strip[:, start_x:end_x].copy()

    # Measure inference time
    start_time = time.time()
    results = model(current_frame, imgsz=IMG_SIZE_INFERENCE, conf=CONF_THRESHOLD, verbose=False)
    end_time = time.time()

    inference_time = end_time - start_time
    try:
        inference_fps = 1.0 / inference_time
    except ZeroDivisionError:
        inference_fps = 0.0

    # Annotate frame
    annotated_frame = results[0].plot()
    fps_text = f"FPS: {inference_fps:.2f} (Inf. Time: {inference_time*1000:.1f}ms)"
    cv2.putText(
        annotated_frame, 
        fps_text, 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.8, 
        (255, 255, 255), 
        2
    )

    # Write frame to video
    video_writer.write(annotated_frame)

# 7. Release resources
video_writer.release()
print("\n--- Video saved! ---")
print(f"Output file: {OUTPUT_VIDEO_FILE}")
print("Download from the Kaggle output directory.")


