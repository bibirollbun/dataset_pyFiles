import os
import cv2
import numpy as np
import pandas as pd
import shutil
import multiprocessing
from tqdm import tqdm
from joblib import Parallel, delayed

# Configuration
RAW_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/train.csv"
RAW_IMG_DIR = "/kaggle/input/aptos2019-blindness-detection/train_images"

# Output directory settings
SAVE_DIR = "./processed_images_224"
IMG_SIZE = 224

# Create output directory if it doesn't exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Core Image Processing Functions
def crop_image_from_gray(img, tol=7):
    """
    Crops the black background borders from the fundus images.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img > tol
        check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0): 
            return img 
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def load_ben_color(image, sigmaX=10):
    """
    Applies Ben Graham's preprocessing method to enhance vascular contrast.
    Formula: image = 4 * image - 4 * GaussianBlur(image) + 128
    """
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), sigmaX), -4, 128)
    return image

# Single Image Wrapper (For Parallelization)
def process_single_image(img_name, raw_dir, save_dir, img_size):
    """
    Pipeline: Read -> Crop -> Resize -> Ben's Color -> Save
    Returns: Boolean indicating success.
    """
    img_path = os.path.join(raw_dir, img_name + ".png")
    save_path = os.path.join(save_dir, img_name + ".png")
    
    # Optional: Skip if already exists
    # if os.path.exists(save_path): return True 

    try:
        image = cv2.imread(img_path)
        if image is None:
            return False
            
        # 1. Crop black borders
        image = crop_image_from_gray(image)
        # 2. Resize
        image = cv2.resize(image, (img_size, img_size))
        # 3. Ben's Preprocessing
        image = load_ben_color(image, sigmaX=10)
        
        # 4. Save
        cv2.imwrite(save_path, image)
        return True
        
    except Exception:
        # Fallback: Create black image to prevent dataloader errors later
        black_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        cv2.imwrite(save_path, black_img)
        return False

# Main Execution
if __name__ == "__main__":
    df = pd.read_csv(RAW_CSV_PATH)
    
    # Auto-detect available CPU cores
    num_cores = multiprocessing.cpu_count()
    
    # execute parallel processing
    # Backend 'threading' is often faster for I/O bound tasks like image saving
    results = Parallel(n_jobs=num_cores, backend="threading")(
        delayed(process_single_image)(
            row['id_code'], 
            RAW_IMG_DIR, 
            SAVE_DIR, 
            IMG_SIZE
        ) for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing")
    )
    
    # Compress the output folder to a single zip file for easy export/download
    shutil.make_archive("processed_images", 'zip', SAVE_DIR)




