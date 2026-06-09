!pip install ultralytics


!pip install -U numpy==1.26.4 scikit-learn==1.4.2 scipy==1.13.1 --force-reinstall


import pandas as pd
import os
import numpy as np
import shutil
import yaml
import matplotlib.pyplot as plt
import random
import cv2
import multiprocessing
from tqdm import tqdm
from glob import glob
from sklearn import model_selection
from skimage import exposure
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


# size = 1024
# TRAIN_LABELS_PATH = './vinbigdata/labels/train'
# VAL_LABELS_PATH = './vinbigdata/labels/val'
# TRAIN_IMAGES_PATH = './vinbigdata/images/train' #12000
# VAL_IMAGES_PATH = './vinbigdata/images/val' #3000
# External_DIR = f'../input/vinbigdata-{size}-image-dataset/vinbigdata/train' # 15000
# os.makedirs(TRAIN_LABELS_PATH, exist_ok = True)
# os.makedirs(VAL_LABELS_PATH, exist_ok = True)
# os.makedirs(TRAIN_IMAGES_PATH, exist_ok = True)
# os.makedirs(VAL_IMAGES_PATH, exist_ok = True)


# original_df = pd.read_csv('../input/vinbigdata-chest-xray-abnormalities-detection/train.csv')
# number_of_imageids = len(original_df['image_id'].values)
# print(f'Total number of image_ids (train + validation) {number_of_imageids}')

# number_of_images = len(os.listdir('../input/vinbigdata-chest-xray-abnormalities-detection/train'))
# print(f'Total number of images (train + validation) {number_of_images}')

# number_of_labels = len(os.listdir('../input/vinbigdata-yolo-labels-dataset/labels'))
# print(f'Total number of labels (train + validation) {number_of_labels}')


# import os
# import cv2
# import numpy as np
# import pydicom
# import multiprocessing
# from tqdm import tqdm
# from skimage import exposure

# def dicom2array(path, voi_lut=True, fix_monochrome=True):
#     dicom = pydicom.read_file(path)
#     if voi_lut:
#         data = apply_voi_lut(dicom.pixel_array, dicom)
#     else:
#         data = dicom.pixel_array
#     if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
#         data = np.amax(data) - data
#     data = data - np.min(data)
#     data = data / np.max(data)
#     data = (data * 255).astype(np.uint8)
#     return data

# def process_image(dicom_path_output_dir):
#     dicom_path, output_dir = dicom_path_output_dir
#     file_name = os.path.splitext(os.path.basename(dicom_path))[0]
#     image_array = dicom2array(dicom_path)
#     equalized_image = exposure.equalize_hist(image_array)
#     equalized_image = (equalized_image * 255).astype(np.uint8)
#     cv2.imwrite(os.path.join(output_dir, f"{file_name}.jpeg"), equalized_image)

# def saving_image(output_dir, dicom_path_list):
#     os.makedirs(output_dir, exist_ok=True)
#     dicom_path_output_dir_list = [(path, output_dir) for path in dicom_path_list]

#     # Use multiprocessing Pool for parallel processing
#     with multiprocessing.Pool() as pool:
#         list(tqdm(pool.imap(process_image, dicom_path_output_dir_list), total=len(dicom_path_list), desc="Processing Images"))


# df = pd.read_csv('/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv')
# number_of_images = len(df['image_id'].values)
# print(f'Total number of image ids (train + validation) {number_of_images}')

# df = df[df.class_id!=14].reset_index(drop = True)
# number_of_images = len(df['image_id'].values)
# print(f'Total number of image ids after dropping normal images (train + validation) {number_of_images}')

# df.head()


# df = df.drop(columns=['class_name', 'rad_id', 'x_min', 'x_max', 'y_min', 'y_max',  'class_id']) # we only need image ids, labels are pre-made
# df.head()


# import pandas as pd
# from sklearn.model_selection import GroupShuffleSplit
# import os
# from tqdm.notebook import tqdm
# import shutil # Đảm bảo đã import shutil

# # --- PHẦN 1: CODE CHIA "SẠCH" (GROUP SHUFFLE SPLIT) ---

# # 1. Tải file train.csv GỐC
# TRAIN_CSV_PATH = '/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv' 
# try:
#     df_train_full = pd.read_csv(TRAIN_CSV_PATH)
#     print(f"Đã tải thành công file: {TRAIN_CSV_PATH}")
# except FileNotFoundError:
#     print(f"LỖI: Không tìm thấy file {TRAIN_CSV_PATH}.")
#     raise

# # 2. Xác định Dữ liệu (X) và Nhóm (Groups)
# X = df_train_full.index
# groups = df_train_full['image_id'] # Dùng 'image_id' làm NHÓM
# print(f"Chuẩn bị chia {len(df_train_full)} bounding box...")
# print(f"Dựa trên {len(groups.unique())} ảnh (nhóm) duy nhất.")

# # 3. Khởi tạo công cụ chia
# gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
# train_idx, val_idx = next(gss.split(X, groups=groups))

# # 4. Lấy danh sách image_id "SẠCH" (Không rò rỉ)
# train_image_ids = df_train_full.iloc[train_idx]['image_id'].unique()
# val_image_ids = df_train_full.iloc[val_idx]['image_id'].unique()
# print(f"Số ảnh Train: {len(train_image_ids)}")
# print(f"Số ảnh Val: {len(val_image_ids)}")


# # --- PHẦN 2: CODE COPY FILE CỦA BẠN (PREPROCCESS_DATA) ---

# # (Hàm preproccess_data của bạn giữ nguyên ở đây)
# def preproccess_data(df, labels_path, images_path):
#     # (Biến 'size' phải được định nghĩa ở cell trước đó, ví dụ: size = 1024)
#     for img_id in tqdm(df.image_id.unique()):
#         label_src = f"../input/vinbigdata-yolo-labels-dataset/labels/{img_id}.txt"
#         image_src = f"/kaggle/input/vinbigdata-{size}-image-dataset/vinbigdata/train/{img_id}.png"
#         if os.path.exists(label_src) and os.path.exists(image_src):
#             shutil.copy(label_src, labels_path)
#             shutil.copy(image_src, images_path)

# # --- SỬA LỖI: Tạo DataFrame "sạch" ---
# # (Biến train_image_ids và val_image_ids BÂY GIỜ đã tồn tại)
# print("Đang tạo DataFrame 'sạch' từ danh sách ID...")
# df_train = pd.DataFrame({'image_id': train_image_ids})
# df_valid = pd.DataFrame({'image_id': val_image_ids})
# # --- HẾT PHẦN SỬA ---

# # Bây giờ các biến df_train, df_valid đã tồn tại và "sạch"
# preproccess_data(df_train, TRAIN_LABELS_PATH, TRAIN_IMAGES_PATH)
# preproccess_data(df_valid, VAL_LABELS_PATH, VAL_IMAGES_PATH)

# print("\nSố lượng file sau khi copy:")
# print("Train Labels:", len(os.listdir(TRAIN_LABELS_PATH)))
# print("Train Images:", len(os.listdir(TRAIN_IMAGES_PATH)))
# print("Val Labels:", len(os.listdir(VAL_LABELS_PATH)))
# print("Val Images:", len(os.listdir(VAL_IMAGES_PATH)))


# classes = [
#     'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
#     'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass',
#     'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis'
# ]

# data = dict(
#     train='../vinbigdata/images/train',
#     val='../vinbigdata/images/val',
#     nc=14,
#     names=classes
# )

# with open('/kaggle/working/vinbigdata.yaml', 'w') as outfile:
#     yaml.dump(data, outfile, default_flow_style=False)

# print("\nCấu hình YAML:")
# print(open('/kaggle/working/vinbigdata.yaml').read())


import os
import shutil

old_runs_dir = '/kaggle/input/runs-epoch-51-yolov11/runs'
new_runs_dir = '/kaggle/working/runs'

print(f"Chuẩn bị sao chép toàn bộ thư mục 'runs' cũ...")
try:
    if os.path.exists(old_runs_dir):
        shutil.copytree(old_runs_dir, new_runs_dir)
        print(f"\nSao chép thành công!")
        print("\nCấu trúc thư mục mới trong /kaggle/working/runs/detect:")
        os.system(f"ls -l {new_runs_dir}/detect/")
    else:
        print(f"\n--- LỖI ---")
        print(f"Không tìm thấy thư mục 'runs' cũ tại: {old_runs_dir}")
except FileExistsError:
    print(f"\n--- CẢNH BÁO ---")
    print(f"Thư mục {new_runs_dir} đã tồn tại. Bỏ qua bước sao chép.")


old_yaml_path = '/kaggle/input/runs-epoch-51-yolov11/vinbigdata.yaml'

new_yaml_path = '/kaggle/working/vinbigdata.yaml'

print(f"\nChuẩn bị sao chép file 'vinbigdata.yaml'...")
print(f"  Từ: {old_yaml_path}")
print(f"  Đến: {new_yaml_path}")

try:
    if os.path.exists(old_yaml_path):
        shutil.copyfile(old_yaml_path, new_yaml_path)
        print(f"\nSao chép 'vinbigdata.yaml' thành công!")
    else:
        print(f"\n--- LỖI ---")
        print(f"Không tìm thấy file .yaml cũ tại: {old_yaml_path}")
        print("Hãy kiểm tra lại đường dẫn.")

except FileExistsError:
    print(f"\n--- CẢNH BÁO ---")
    print(f"File {new_yaml_path} đã tồn tại. Bỏ qua bước sao chép .yaml.")

except Exception as e:
    print(f"\n--- LỖI KHÁC KHI SAO CHÉP .yaml ---")
    print(e)


import os
import shutil

# 1. Đường dẫn đến thư mục 'vinbigdata' trong output của version CŨ
old_vinbigdata_dir = '/kaggle/input/runs-epoch-51-yolov11/vinbigdata/'

# 2. Đường dẫn đến thư mục 'vinbigdata' MỚI trong /kaggle/working/
new_vinbigdata_dir = '/kaggle/working/vinbigdata'

# --- Đã sửa lại câu thông báo ---
print(f"Chuẩn bị sao chép toàn bộ thư mục 'vinbigdata' cũ...") 
print(f"  Từ: {old_vinbigdata_dir}")
print(f"  Đến: {new_vinbigdata_dir}")

# 3. Dùng shutil.copytree để copy toàn bộ thư mục
try:
    if os.path.exists(old_vinbigdata_dir):
        shutil.copytree(old_vinbigdata_dir, new_vinbigdata_dir)
        print(f"\nSao chép thành công!")
        
        # --- Đã sửa lại lệnh kiểm tra ---
        print(f"\nCấu trúc thư mục mới trong {new_vinbigdata_dir}:")
        os.system(f"ls -l {new_vinbigdata_dir}") # Liệt kê nội dung của vinbigdata
        
    else:
        print(f"\n--- LỖI ---")
        print(f"Không tìm thấy thư mục 'vinbigdata' cũ tại: {old_vinbigdata_dir}")
        print("Bạn đã '+ Add data' output của version cũ (notebook-yolov11-1024) chưa?")
        
except FileExistsError:
    print(f"\n--- CẢNH BÁO ---")
    print(f"Thư mục {new_vinbigdata_dir} đã tồn tại. Bỏ qua bước sao chép.")
    print("Điều này ổn nếu bạn chạy lại cell.")


from ultralytics import YOLO

# Load model
model = YOLO('/kaggle/input/runs-epoch-51-yolov11/runs/detect/train/weights/best.pt')

# Train YOLOv11
model.train(
    # --- Các tham số gốc của bạn ---
    data='./vinbigdata.yaml',
    imgsz=1024,
    batch=16,
    device=[0,1],
    epochs=100,
    patience=30,
    cos_lr=True,
    resume=True,

    optimizer='SGD',  
    lr0=0.0005,       
    lrf=0.001,
    
    mixup=0.2,
    cutmix=0.15,
)


# from ultralytics import YOLO

# # Load model
# model = YOLO('yolo11l.pt')

# # Train YOLOv11
# model.train(
#     # --- Các tham số gốc của bạn ---
#     data='./vinbigdata.yaml',
#     imgsz=1024,
#     batch=16,
#     device=[0,1],
#     epochs=100,
#     patience=30,
#     cos_lr=True,
#     resume=False,

#     optimizer='SGD',  
#     lr0=0.0005,       
#     lrf=0.001,
    
#     mixup=0.2,
#     cutmix=0.15,
# )

