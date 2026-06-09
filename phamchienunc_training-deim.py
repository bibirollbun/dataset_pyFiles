import pydicom
import numpy as np
from PIL import Image
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

dicom_dir = '/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train'
output_dir = '/kaggle/working/train'
os.makedirs(output_dir, exist_ok=True)

def convert_dicom_to_png(dicom_file):
    try:
        ds = pydicom.dcmread(dicom_file, stop_before_pixels=False)  # vẫn đọc pixel data
        arr = ds.pixel_array.astype(np.float32)
        arr -= arr.min()
        arr /= arr.max()
        arr = (arr * 255).astype(np.uint8)

        Image.fromarray(arr).save(os.path.join(output_dir, Path(dicom_file).stem + '.png'))
        return dicom_file
    except Exception as e:
        return f"Lỗi: {dicom_file} -> {e}"

# Duyệt danh sách file
dicom_files = list(Path(dicom_dir).rglob('*.dicom'))

# Dùng đa tiến trình (8 tiến trình thường nhanh nhất trên Kaggle)
with ProcessPoolExecutor(max_workers=8) as ex:
    for result in ex.map(convert_dicom_to_png, dicom_files):
        print(result)

print("✅ Hoàn thành!")


import pydicom
import numpy as np
from PIL import Image
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

dicom_dir = '/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/test'
output_dir = '/kaggle/working/test'
os.makedirs(output_dir, exist_ok=True)

def convert_dicom_to_png(dicom_file):
    try:
        ds = pydicom.dcmread(dicom_file, stop_before_pixels=False)  # vẫn đọc pixel data
        arr = ds.pixel_array.astype(np.float32)
        arr -= arr.min()
        arr /= arr.max()
        arr = (arr * 255).astype(np.uint8)

        Image.fromarray(arr).save(os.path.join(output_dir, Path(dicom_file).stem + '.png'))
        return dicom_file
    except Exception as e:
        return f"Lỗi: {dicom_file} -> {e}"

# Duyệt danh sách file
dicom_files = list(Path(dicom_dir).rglob('*.dicom'))

# Dùng đa tiến trình (8 tiến trình thường nhanh nhất trên Kaggle)
with ProcessPoolExecutor(max_workers=8) as ex:
    for result in ex.map(convert_dicom_to_png, dicom_files):
        print(result)

print("✅ Hoàn thành!")




