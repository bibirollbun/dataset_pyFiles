! pip install -qU "python-gdcm" pydicom pylibjpeg "opencv-python-headless" "scikit-image" "ipywidgets" "dicomsdl"


# 1. Uninstall the old, incompatible torch
!pip uninstall -y torch torchvision torchaudio


# 1. Clone YOLOv5
!git clone https://github.com/ultralytics/yolov5.git /kaggle/working/yolov5

# 2. Install its Python requirements
!pip install -r /kaggle/working/yolov5/requirements.txt


# 1. Handle datasets
import io
import os
import cv2
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import resize
from PIL import Image, ImageDraw
from pathlib import Path
from collections import Counter
from joblib import Parallel, delayed
from tqdm import tqdm
import imageio
import gc


parent_dir = "/kaggle/input/jp2000"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.hub.load(
    './yolov5', 
    'custom', 
    path="/kaggle/input/roi-rsna/rsna-roi-003.pt", 
    source='local'
)
model.to(device)


model.children()


torch.cuda.empty_cache()
file_list = glob(
    os.path.join(parent_dir, "train_image_processed_jp2000_512", "*", "*.jp2")
)


%matplotlib inline

images = []

for path in random.sample(file_list, 25):
    frame = plt.imread(path)
    
    # Convert to PIL Image
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    detections = model(frame)
    results = detections.pandas().xyxy[0].to_dict(orient="records")
    
    for result in results:
        xmin, ymin = int(result['xmin']), int(result['ymin'])
        xmax, ymax = int(result['xmax']), int(result['ymax'])
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='red', width=4)
    
    images.append(np.array(img_pil))


fig, axes = plt.subplots(5, 5, figsize=(20,20))
    
for idx, image in enumerate(images):
    i = idx % 5 
    j = idx // 5 
    axes[i, j].imshow(image, cmap="bone")
    axes[i, j].axis('off')

plt.subplots_adjust(wspace=0, hspace=.2)
plt.show()




fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes = axes.flatten()

for ax, img in zip(axes, images[:4]):
    ax.imshow(img)
    ax.axis('off')

plt.subplots_adjust(wspace=0, hspace=0.2)
plt.show()


# === CONFIG ===
RESIZE_TO = 512
SAVE_DIR = f"/kaggle/working/train_image_ROI_processed_jp2000_{RESIZE_TO}"
parent_dir = "/kaggle/input/jp2000"

# Gather all .jp2 paths
all_jp2_files = list(
    Path(os.path.join(parent_dir, "train_image_processed_jp2000_512")).rglob("*.jp2")
)
fail_counter = Counter()
model.eval()

# === CLEANING ===
def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

# === RESIZE ===
def image_resize(image, width=None, height=None):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = resize(image, (dim[1], dim[0]), preserve_range=True, anti_aliasing=True)
    return resized.astype(np.uint8)

# === MAIN PROCESSING FUNCTION ===
def extract_and_roi(path):
    try:
        if not path.exists() or not path.is_file():
            fail_counter["invalid_path"] += 1
            print(f"[ERROR] Invalid path: {path}")
            return
        
        parent_folder = path.parent.name
        save_subdir = os.path.join(parent_dir, SAVE_DIR, parent_folder)
        os.makedirs(save_subdir, exist_ok=True)

        image = plt.imread(str(path))
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Ensure image loaded correctly
        if image is None:
            raise ValueError("Could not load image")
        
        # Check if the image is valid
        if image.shape[0] == 0 or image.shape[1] == 0:
            fail_counter["invalid_image"] += 1
            print(f"[ERROR] Image is empty: {path}")
            return
        
        results = model(image)
        detections = results.pandas().xyxy[0]
        
        if len(detections) == 0:
            fail_counter["no_roi"] += 1
            return

        for i, det in detections.iterrows():
            x1, y1, x2, y2 = map(int, [det["xmin"], det["ymin"], det["xmax"], det["ymax"]])
            
            # Check for valid ROI dimensions
            if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
                fail_counter["invalid_roi"] += 1
                print(f"[ERROR] Invalid ROI for {path}: {x1}, {y1}, {x2}, {y2}")
                continue
            
            roi = image[y1:y2, x1:x2]
            roi_resized = image_resize(roi, width=RESIZE_TO)
            
            save_path = os.path.join(save_subdir, f"{path.stem}.jp2")
            try:
                imageio.imwrite(save_path, roi_resized, format='JP2')
            except Exception as e:
                print(f"[ERROR] Failed to save ROI for {path}: {e}")
                fail_counter["save_error"] += 1

            # === For sanity Check === 
            # return save_path
        
            # Clean up memory
            cleanup()

    except Exception as e:
        print(f"[ERROR] Failed processing {path} — {e}")
        fail_counter["fail"] += 1

# === Sanity Check ===
# for path in tqdm(all_jp2_files[:1], desc="Sanity Check"):
#     save_path = extract_and_roi(path)
#     img = plt.imread(save_path)
#     plt.imshow(img, cmap="turbo")
#     print("✅ Sanity check complete.")
#     print(f"image size = {img.shape}")
    
#     plt.axis('off')
#     plt.show()
#     break

# === RUN ===
Parallel(n_jobs=16, backend="loky", prefer="threads")(
    delayed(extract_and_roi)(path) for path in tqdm(all_jp2_files)
)

print(f"✅ Done! Processed {len(all_jp2_files)} images.")
print(f"❌ Failed: {fail_counter['fail']}, ",
      f"No ROI: {fail_counter['no_roi']}, ",
      f"Invalid Path: {fail_counter['invalid_path']}, ",
      f"Invalid Image: {fail_counter['invalid_image']}," 
      f"Save Errors: {fail_counter['save_error']}")

