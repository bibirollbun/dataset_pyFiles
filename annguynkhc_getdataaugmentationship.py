H = 768
WORK_DIR = "/kaggle/working/"


import pandas as pd
import numpy as np
def encode_to_bounding_box(encode_value, check_test = False):
    """Function help convert Encode Pixel value to Bounding Box for YoLo Training for a picture"""
    encode_value = list(map(int, encode_value.split()))
    x_min, y_min, x_max, y_max = 1000, 1000, -1, -1
    for i in range(0, len(encode_value), 2):
        start = encode_value[i]
        for j in range(start, start + encode_value[i+1]):
            r = (j-1) % H
            c = (j-1) // H
            if x_min > c: x_min = c
            if y_min > r: y_min = r
            if x_max < c: x_max = c
            if y_max < r: y_max = r
    if x_min > 0 and y_min > 0: 
        x_min, y_min = x_min - 1, y_min - 1
    if x_max > 0 and y_max > 0:
        x_max, y_max = x_max + 1, y_max + 1
    if check_test == True:
        return [x_min, y_min, x_max, y_max]
    x_center = (x_min + x_max) / 2 / H
    y_center = (y_min + y_max) / 2 / H
    w = (x_max - x_min) / H
    h = (y_max - y_min) / H
    bb = [0, x_center, y_center, w, h]
    return " ".join(map(str, bb))



import os
import shutil

images_path = os.path.join(WORK_DIR, "images")
img_train = os.path.join(images_path, "train")
img_val = os.path.join(images_path, "val")
img_test = os.path.join(images_path, "test")

labels_path = os.path.join(WORK_DIR, "labels")
lb_train = os.path.join(labels_path, "train")
lb_val = os.path.join(labels_path, "val")
lb_test = os.path.join(labels_path, "test")

if os.path.exists(images_path) and os.path.exists(labels_path):
    shutil.rmtree(images_path)
    shutil.rmtree(labels_path)
for path in [images_path, img_train, img_val, img_test,
         labels_path, lb_train, lb_val, lb_test]:
    os.makedirs(path, exist_ok=True)


train_list = pd.read_csv("/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv")
has_ship_train = 50000
has_ship_val = 10000
has_ship_test = 1000


import albumentations as A
import cv2
import matplotlib.pyplot as plt
satellite_transforms = A.Compose([
            # 1. Sương mù và bụi khí quyển (giảm tương phản, làm mờ)
            A.OneOf([
                # Sương mù nhẹ
                A.RandomFog(
                    fog_coef_lower=0.1, 
                    fog_coef_upper=0.4, 
                    alpha_coef=0.1,
                    p=1.0
                ),
                # Bụi khí quyển - giảm tương phản
                A.Compose([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1, 
                        contrast_limit=(-0.3, -0.1), # Giảm contrast
                        p=1.0
                    ),
                    A.GaussianBlur(blur_limit=(1, 3), p=0.8),  # Làm mờ nhẹ
                ]),
            ], p=0.4),
            
            # 2. Glint từ mặt nước (lóa sáng cục bộ)
            A.OneOf([
                # Sun flare mô phỏng phản xạ mặt nước
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 1),  # Toàn ảnh
                    angle_lower=0,
                    angle_upper=1,
                    num_flare_circles_lower=1,
                    num_flare_circles_upper=3,
                    src_radius=50,
                    p=1.0
                ),
            ], p=0.4),
            
            # 3. Motion blur do vệ tinh di chuyển
            A.OneOf([
                A.MotionBlur(blur_limit=(3, 7), p=1.0),  # Motion blur
                A.Compose([
                    A.MotionBlur(blur_limit=(2, 4), p=1.0),
                    A.GaussianBlur(blur_limit=(1, 2), p=0.5),  # Thêm blur nhẹ
                ]),
            ], p=0.2),
            
            # # 4. Nhiễu sensor và nhiễu điện tử
            A.OneOf([
                # Nhiễu Gaussian
                A.GaussNoise(
                    var_limit=(10.0, 50.0),
                    mean=0,
                    per_channel=True,
                    p=1.0
                ),
                # Nhiễu speckle (đặc trưng của ảnh vệ tinh)
                A.MultiplicativeNoise(
                    multiplier=(0.95, 1.05),
                    per_channel=False,
                    p=1.0
                ),
                # Nhiễu ISO cao
                A.ISONoise(
                    color_shift=(0.01, 0.05),
                    intensity=(0.1, 0.5),
                    p=1.0
                ),
            ], p=0.015),
            
            # 5. Vấn đề về độ phơi sáng và bão hòa
            A.OneOf([
                # Over-exposure
                A.RandomBrightnessContrast(
                    brightness_limit=(0.2, 0.4),
                    contrast_limit=(0.1, 0.3),
                    p=1.0
                ),
                # Under-exposure  
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.3, -0.1),
                    contrast_limit=(0.1, 0.3),
                    p=1.0
                ),
                # Saturation issues
                A.HueSaturationValue(
                    hue_shift_limit=5,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
                ),
            ], p=0.05),
            
            # 6. Atmospheric effects (hiệu ứng khí quyển)
            A.OneOf([
                # Haze effect
                A.Compose([
                    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=-0.2, p=1.0),
                    A.HueSaturationValue(sat_shift_limit=-20, p=1.0),  # Giảm saturation
                ]),
                # Atmospheric scattering
                A.Compose([
                    A.ToSepia(p=0.3),  # Tạo hiệu ứng scattering nhẹ
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                ]),
            ], p=0.05),  
        ])


import cv2
import matplotlib.pyplot as plt
import os

a_path = "/kaggle/input/airbus-ship-detection/train_v2/0005d01c8.jpg"

def apply_and_visualize(a_path, satellite_transforms):
    # Đọc ảnh BGR bằng OpenCV
    img = cv2.imread(a_path)
    if img is None:
        raise ValueError(f"Không đọc được ảnh từ {a_path}")
    
    # Chuyển sang RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Áp dụng transform
    transformed = satellite_transforms(image=img_rgb)
    img_transformed = transformed["image"]

    # Đường dẫn lưu
    orig_path = "/kaggle/working/original.jpg"
    trans_path = "/kaggle/working/transformed.jpg"

    # Lưu ảnh (cv2 dùng BGR nên cần convert lại)
    cv2.imwrite(orig_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(trans_path, cv2.cvtColor(img_transformed, cv2.COLOR_RGB2BGR))

    # In dung lượng file
    orig_size = os.path.getsize(orig_path) / 1024  # KB
    trans_size = os.path.getsize(trans_path) / 1024  # KB
    print(f"Dung lượng ảnh gốc: {orig_size:.2f} KB")
    print(f"Dung lượng ảnh transformed: {trans_size:.2f} KB")

    # Hiển thị
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(img_transformed)
    plt.title("Transformed")
    plt.axis("off")

    plt.show()

apply_and_visualize(a_path, satellite_transforms)



import random
from tqdm import tqdm
def offline_aug(train_path, satellite_transforms):
    """
    Mở ảnh từ đường dẫn, áp dụng satellite_transforms (albumentations),
    rồi ghi đè kết quả lên chính file đó.
    """
    files = os.listdir(train_path)
    print(len(files))
    processed = 0
    for file_name in tqdm(files, desc="Augmenting", unit="img"):
        if random.random() < 0.7: continue
        else:
            full_img_path = os.path.join(train_path, file_name)
            image = cv2.imread(full_img_path)
            os.remove(full_img_path)
            if image is None:
                raise FileNotFoundError(f"Không tìm thấy ảnh: {full_img_path}")
            # Áp dụng transform (albumentations cần đầu vào là dict)
            transformed = satellite_transforms(image=image)
            transformed_image = transformed["image"]
            # Ghi đè ảnh (giữ nguyên BGR cho OpenCV)
            cv2.imwrite(full_img_path, transformed_image)
        processed += 1
        if processed % 10000 == 0: print("Processed: ", processed)
    print("Offline Data Augmentation is succesful !")


index_seg = 0
def setup_dataset(has_ship_len, index_seg, img_path, lb_path, tag = 'train'):
    has = 0
    multiple = False
    while has < has_ship_len:
        if pd.isna(train_list.iloc[index_seg,1]) == False:
            img_name = train_list.iloc[index_seg,0]
            
            src_img_path = "/kaggle/input/airbus-ship-detection/train_v2/"+img_name
            full_img_path = os.path.join(img_path, img_name)
            shutil.copy(src_img_path, full_img_path) #copy ảnh vào thư mục cần 
            
            full_lb_path = os.path.join(lb_path, img_name.replace(".jpg", ".txt"))
            bbs = []
            while train_list.iloc[index_seg,0] == img_name:
                multiple = True
                bbs.append(encode_to_bounding_box(train_list.iloc[index_seg,1]))
                index_seg += 1
            with open(full_lb_path, "w") as f:
                for item in bbs:
                    f.write(item + "\n")
        has += 1
        if has % 2000 == 0: print("Processed ",tag,": ",has)
        if multiple == False: index_seg += 1
        else: multiple = False
    print("Check index_seg:", index_seg)
    return index_seg

index_seg = setup_dataset(has_ship_train, index_seg, img_train, lb_train, 'train') #SETUP TRAIN
index_seg = setup_dataset(has_ship_val, index_seg, img_val, lb_val, 'validation') #SETUP VALIDATION
setup_dataset(has_ship_test, index_seg, img_test, lb_test, 'test') #SETUP TEST


# offline_aug(img_train, satellite_transforms) #DATA AUGMENTATION FOR TRAIN DATA




