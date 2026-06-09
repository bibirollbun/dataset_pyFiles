import os
import pandas as pd
import pydicom
import cv2
import numpy as np
from tqdm import tqdm
import concurrent.futures

# --- CẤU HÌNH (SỬA LẠI NẾU CẦN) ---
# Đường dẫn folder chứa ảnh DICOM gốc
INPUT_DIR = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train"

# Folder đầu ra (Sẽ chứa toàn bộ file PNG)
OUTPUT_DIR = "/kaggle/working/train_png_640" 

# File CSV lọc danh sách ảnh (như trong ảnh bạn gửi)
FILTER_CSV = "/kaggle/input/vindr-train-dims/train_v2(with_dims).csv"  # Sửa đường dẫn này nếu file ở chỗ khác
# Nếu không có file CSV lọc, có thể comment dòng trên và dùng os.listdir

TARGET_SIZE = 1024 # Size ảnh mong muốn
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. HÀM XỬ LÝ ẢNH "ROBUST" (BẤT CHẤP NHIỄU) ---
def convert_dicom_robust(dicom_path, target_size=640, fix_monochrome=True):
    try:
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array.astype(np.float32)

        # Fix lỗi âm bản
        if fix_monochrome and hasattr(ds, "PhotometricInterpretation"):
            if ds.PhotometricInterpretation == "MONOCHROME1":
                img = np.amax(img) - img
        
        # Robust Scaling (Thay cho Windowing cứng nhắc)
        # Cắt bỏ 1% điểm ảnh quá sáng/quá tối để loại nhiễu
        p_low = np.percentile(img, 1)
        p_high = np.percentile(img, 99)
        img = np.clip(img, p_low, p_high)
        
        # Chuẩn hóa Min-Max về 0-1
        if p_high > p_low:
            img = (img - p_low) / (p_high - p_low)
        else:
            img = np.zeros_like(img) # Ảnh lỗi

        # Đưa về 0-255
        img = (img * 255.0).astype(np.uint8)

        # Resize + Padding (Letterbox)
        h, w = img.shape
        scale = min(target_size/h, target_size/w)
        nh, nw = int(h*scale), int(w*scale)
        img_resized = cv2.resize(img, (nw, nh))
        
        final_img = np.zeros((target_size, target_size), dtype=np.uint8)
        dy = (target_size - nh) // 2
        dx = (target_size - nw) // 2
        final_img[dy:dy+nh, dx:dx+nw] = img_resized
        
        return final_img
    except Exception as e:
        # Nếu file dicom bị lỗi đọc, trả về None
        return None

# --- 2. HÀM WRAPPER: KẾT NỐI INPUT -> XỬ LÝ -> OUTPUT ---
def process_and_save(filename):
    # Đảm bảo filename có đuôi .dicom
    if not filename.endswith('.dicom'): 
        filename = f"{filename}.dicom"
        
    # TẠO ĐƯỜNG DẪN ĐẦY ĐỦ (Fix lỗi FileNot Found)
    in_path = os.path.join(INPUT_DIR, filename)
    out_path = os.path.join(OUTPUT_DIR, filename.replace('.dicom', '.png'))
    
    # Skip nếu file đã tồn tại (để resume nếu bị ngắt)
    if os.path.exists(out_path): return

    # Gọi hàm xử lý
    processed_img = convert_dicom_robust(in_path, target_size=TARGET_SIZE)
    
    # Lưu ảnh nếu xử lý thành công
    if processed_img is not None:
        cv2.imwrite(out_path, processed_img)

# --- 3. CHƯƠNG TRÌNH CHÍNH ---

# Lấy danh sách file cần làm
if os.path.exists(FILTER_CSV):
    print(f"Đang đọc danh sách ID từ {FILTER_CSV}...")
    df = pd.read_csv(FILTER_CSV)
    # Lấy cột image_id và đảm bảo thêm đuôi .dicom
    target_files = [f"{img_id}.dicom" if not str(img_id).endswith('.dicom') else str(img_id) 
                    for img_id in df['image_id'].unique()]
else:
    print("Không tìm thấy CSV lọc, sẽ quét toàn bộ thư mục input...")
    target_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.dicom')]

# Lọc lại lần cuối để chắc chắn file input có tồn tại
print("Đang kiểm tra file input...")
files_to_process = [f for f in target_files if os.path.exists(os.path.join(INPUT_DIR, f))]

print(f"--> BẮT ĐẦU XỬ LÝ {len(files_to_process)} ẢNH...")
print(f"--> OUTPUT SẼ LƯU TẠI: {OUTPUT_DIR}")

# Chạy đa luồng (4 workers là tối ưu cho Kaggle)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    list(tqdm(executor.map(process_and_save, files_to_process), total=len(files_to_process)))
    
print("\n=== HOÀN TẤT! ===")


import shutil
import os

# --- CẤU HÌNH ---
SOURCE_DIR = "/kaggle/working/train_png_640" # Folder chứa ảnh PNG vừa tạo
OUTPUT_NAME = "/kaggle/working/train_png_640" # Tên file zip đầu ra

print("Đang nén file zip... (Bước này mất vài phút)")
# Tạo file zip: vinbigdata_preprocessed.zip
shutil.make_archive(OUTPUT_NAME, 'zip', SOURCE_DIR)

print(f"Xong! File đã lưu tại: {OUTPUT_NAME}.zip")

# (Tùy chọn) Xóa folder ảnh lẻ đi cho nhẹ Output, chỉ giữ lại file zip
# shutil.rmtree(SOURCE_DIR)

