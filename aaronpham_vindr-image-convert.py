import os
import pydicom
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import concurrent.futures

# Cấu hình
INPUT_DIR = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/test"
OUTPUT_DIR = "/kaggle/working/test_png_384"
IMG_SIZE = 384

os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_one_image(filename):
    if not filename.endswith('.dicom'): return
    
    file_path = os.path.join(INPUT_DIR, filename)
    save_path = os.path.join(OUTPUT_DIR, filename.replace('.dicom', '.png'))
    
    # Nếu file đã tồn tại thì bỏ qua (resume)
    if os.path.exists(save_path): return

    try:
        # Đọc DICOM
        dicom = pydicom.dcmread(file_path)
        pixel_array = dicom.pixel_array
        
        # Photometric Interpretation handling (quan trọng cho X-Ray)
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            pixel_array = np.max(pixel_array) - pixel_array
            
        # Normalize về 0-255
        pixel_array = pixel_array.astype(np.float32)
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255.0
        pixel_array = pixel_array.astype(np.uint8)
        
        # Resize
        img = cv2.resize(pixel_array, (IMG_SIZE, IMG_SIZE))
        
        # Save PNG
        cv2.imwrite(save_path, img)
    except Exception as e:
        print(f"Error converting {filename}: {e}")

# Chạy đa luồng (Kaggle CPU có 4 cores)
files = os.listdir(INPUT_DIR)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    list(tqdm(executor.map(convert_one_image, files), total=len(files)))

