!pip install --upgrade gdcm pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg
!pip install gdcm
!pip install pylibjpeg pylibjpeg-libjpeg
!pip install pandas==1.3.5 numpy==1.21.6



import torch
import torch.nn as nn

# 檢查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定義一個簡單的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(128, 64)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel()  # 創建模型實例
model.to(device)  # 將模型移動到 GPU 或 CPU



import torch
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))  # 顯示可用 GPU

print(torch.cuda.is_available())  # True 表示 GPU 可用
print(torch.cuda.device_count())  # 顯示可用 GPU 數量
print(torch.cuda.get_device_name(0))  # 顯示 GPU 型號
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



import os
import pydicom
import matplotlib.pyplot as plt

# 設定影像資料夾路徑
data_dir = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images"

# 獲取第一個病人的資料夾
first_patient_folder = sorted(os.listdir(data_dir))[0]

# 取得該病人資料夾中的所有 .dcm 檔案
dcm_files = sorted([f for f in os.listdir(os.path.join(data_dir, first_patient_folder)) if f.endswith(".dcm")])

# 初始化圖像展示
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# 顯示前 8 張 CT 影像
for idx, dcm_file in enumerate(dcm_files[:8]):
    # 讀取 .dcm 檔案
    dcm_path = os.path.join(data_dir, first_patient_folder, dcm_file)
    dicom_data = pydicom.dcmread(dcm_path)
    
    # 顯示影像
    ax = axes[idx // 4, idx % 4]
    ax.imshow(dicom_data.pixel_array)
    ax.set_title(f"{dcm_file}")
    ax.axis("off")

plt.tight_layout()
plt.show()


import os
import numpy as np
import cv2
import pydicom
import pandas as pd
import matplotlib.pyplot as plt
from pydicom.pixel_data_handlers.util import convert_color_space

falsees=0
# 讀取 CSV 檔案，篩選有骨折的影像
csv_path = "/kaggle/input/final-csv/vertebrae_labels_with_fracture.csv"
df = pd.read_csv(csv_path)
fracture_df = df[(df.iloc[:, 2:9] == 1).any(axis=1)]  # 篩選有骨折的影像

# **隨機選取 20% 的資料**
fracture_df = fracture_df.sample(frac=0.2, random_state=42).reset_index(drop=True)

# 影像預處理函數
def preprocess_image(image):
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    denoised = cv2.fastNlMeansDenoising(image, None, h=20, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    gamma = 1.4
    gamma_corrected = np.power(enhanced / 255.0, gamma) * 255
    return np.clip(gamma_corrected, 0, 255).astype(np.uint8)

# 邊緣檢測函數
def edge_detection(image):
    edges = cv2.Canny(image, 50, 150)
    laplacian = np.clip(np.abs(cv2.Laplacian(image, cv2.CV_64F)), 0, 255).astype(np.uint8)
    return cv2.bitwise_or(edges, laplacian)

# 分水嶺分割
def watershed_segmentation(image, edges):
    _, binary_edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = np.uint8(cv2.dilate(opening, kernel, iterations=3))
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)
    return np.where(markers > 1, 255, 0).astype('uint8')

# GrabCut 分割
def apply_grabcut_with_watershed(image, mask):
    grabcut_mask = np.zeros(image.shape[:2], np.uint8)
    grabcut_mask[mask == 255] = cv2.GC_PR_FGD
    grabcut_mask[mask == 0] = cv2.GC_BGD
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, grabcut_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    final_mask = np.where((grabcut_mask == 2) | (grabcut_mask == 0), 0, 1).astype('uint8')
    return image * cv2.merge([final_mask, final_mask, final_mask])

# 設定輸入與輸出目錄
input_dir = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images"
output_dir = "/kaggle/working/npy_output"
os.makedirs(output_dir, exist_ok=True)

# 遍歷 CSV，處理符合條件的影像（只處理 20%）
for _, row in fracture_df.iterrows():
    spine_id = row["SpineID"]
    slice_number = row["SliceNumber"]
    dicom_path = os.path.join(input_dir, spine_id, f"{slice_number}.dcm")
    
    if os.path.exists(dicom_path):
        try:
            dicom_data = pydicom.dcmread(dicom_path)
            image = dicom_data.pixel_array

            if dicom_data.PhotometricInterpretation == "YBR_FULL":
                image = convert_color_space(image, "YBR_FULL", "RGB")

            if image.dtype != np.uint8:
                image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)

            # 預處理與分割
            preprocessed_image = preprocess_image(image)
            edges = edge_detection(preprocessed_image)
            watershed_result = watershed_segmentation(preprocessed_image, edges)
            preprocessed_bgr = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)
            grabcut_result = apply_grabcut_with_watershed(preprocessed_bgr, watershed_result)

            # 建立輸出路徑並儲存 GrabCut 處理後的影像為 .npy 檔案
            output_path = os.path.join(output_dir, spine_id)
            os.makedirs(output_path, exist_ok=True)
            npy_output_path = os.path.join(output_path, f"{slice_number}.npy")
            np.save(npy_output_path, grabcut_result)

        except Exception as e:
            falsees+=1

print("已完成 20% 影像的處理並存儲為 .npy 檔案。")
print("錯誤處理數量:",falsees)


import os
import numpy as np
import cv2
import pydicom
import pandas as pd
import matplotlib.pyplot as plt
from pydicom.pixel_data_handlers.util import convert_color_space

# 讀取 CSV 檔案，篩選有骨折的影像
csv_path = "/kaggle/input/final-csv/vertebrae_labels_with_fracture.csv"
df = pd.read_csv(csv_path)
fracture_df = df[(df.iloc[:, 2:9] == 1).any(axis=1)]  # 篩選有骨折的影像

# CPU 版本的影像預處理函數
def preprocess_image(image):
    # 降噪
    denoised = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # CLAHE 增強
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Gamma 校正 (使用 NumPy 在 CPU 上計算)
    gamma = 1.4
    gamma_corrected = np.power(enhanced / 255.0, gamma) * 255
    gamma_corrected = np.clip(gamma_corrected, 0, 255)
    return gamma_corrected.astype(np.uint8)

# CPU 版本的邊緣檢測函數
def edge_detection(image):
    # Canny 邊緣檢測
    edges = cv2.Canny(image, 50, 150)
    
    # Laplacian 邊緣運算，注意取絕對值以避免負值影響
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = np.clip(np.abs(laplacian), 0, 255).astype(np.uint8)
    
    # 組合邊緣資訊
    combined_edges = cv2.bitwise_or(edges, laplacian)
    return combined_edges

# 分水嶺分割 (CPU 執行)
def watershed_segmentation(image, edges):
    _, binary_edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = np.uint8(cv2.dilate(opening, kernel, iterations=3))
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)
    return np.where(markers > 1, 255, 0).astype('uint8')

# GrabCut 分割 (CPU 執行)
def apply_grabcut_with_watershed(image, mask):
    grabcut_mask = np.zeros(image.shape[:2], np.uint8)
    grabcut_mask[mask == 255] = cv2.GC_PR_FGD
    grabcut_mask[mask == 0] = cv2.GC_BGD
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, grabcut_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    final_mask = np.where((grabcut_mask == 2) | (grabcut_mask == 0), 0, 1).astype('uint8')
    return image * cv2.merge([final_mask, final_mask, final_mask])

# 設定輸入與輸出目錄
input_dir = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images"
output_dir = "/kaggle/working/npy_output"
os.makedirs(output_dir, exist_ok=True)

# 遍歷 CSV，處理符合條件的影像
for _, row in fracture_df.iterrows():
    spine_id = row["SpineID"]
    slice_number = row["SliceNumber"]
    dicom_path = os.path.join(input_dir, spine_id, f"{slice_number}.dcm")
    
    if os.path.exists(dicom_path):
        dicom_data = pydicom.dcmread(dicom_path)
        image = dicom_data.pixel_array

        if dicom_data.PhotometricInterpretation == "YBR_FULL":
            image = convert_color_space(image, "YBR_FULL", "RGB")

        if image.dtype != np.uint8:
            image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)

        # 預處理與分割
        preprocessed_image = preprocess_image(image)
        edges = edge_detection(preprocessed_image)
        watershed_result = watershed_segmentation(preprocessed_image, edges)
        preprocessed_bgr = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)
        grabcut_result = apply_grabcut_with_watershed(preprocessed_bgr, watershed_result)

        # 建立輸出路徑並儲存 GrabCut 處理後的影像為 .npy 檔案
        output_path = os.path.join(output_dir, spine_id)
        os.makedirs(output_path, exist_ok=True)
        npy_output_path = os.path.join(output_path, f"{slice_number}.npy")
        np.save(npy_output_path, grabcut_result)

print("所有符合條件的影像處理完成並存儲為 `.npy` 檔案。")



import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 讀取 CSV 標註檔案
csv_path = "/mnt/data/vertebrae_labels_with_fracture.csv"
df = pd.read_csv(csv_path)

# 設定影像資料夾
npy_dir = "/kaggle/working/npy_output"

# 載入影像與標籤
image_data, labels = [], []
for _, row in df.iterrows():
    spine_id = row["SpineID"]
    slice_number = row["SliceNumber"]
    npy_path = os.path.join(npy_dir, spine_id, f"{slice_number}.npy")

    if os.path.exists(npy_path):
        img = np.load(npy_path)
        image_data.append(img)
        labels.append(row.iloc[2:9].values)  # 取 C1-C7 的骨折標註

# 轉換為 NumPy 陣列
image_data = np.array(image_data).astype(np.float32) / 255.0  # 正規化
labels = np.array(labels, dtype=np.uint8)  # 轉換為整數

# One-hot 編碼標籤 (C1-C7 為 7 類 + 背景 1 類 = 共 8 類)
num_classes = 8
labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

# 確保影像維度正確
image_data = np.expand_dims(image_data, axis=-1)  # (batch, 128, 128, 1)

# 切分訓練集與驗證集 (80% 訓練，20% 驗證)
X_train, X_val, y_train, y_val = train_test_split(image_data, labels, test_size=0.2, random_state=42)

# U-Net 模型架構
def build_unet(input_shape=(128, 128, 1), num_classes=8):
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4)

    # Decoder
    u5 = UpSampling2D((2, 2))(c4)
    u5 = Concatenate()([u5, c3])
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u6 = UpSampling2D((2, 2))(c5)
    u6 = Concatenate()([u6, c2])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Concatenate()([u7, c1])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c7)

    model = Model(inputs, outputs)
    return model

# 編譯模型
model = build_unet()
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss=CategoricalCrossentropy(),
              metrics=["accuracy"])

# 設定回調函數 (儲存最佳模型)
checkpoint = ModelCheckpoint("/kaggle/working/unet_best_model.h5", save_best_only=True, monitor="val_loss")
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)

# 訓練模型
history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=16,
                    callbacks=[checkpoint, reduce_lr])

# 儲存最終模型
model.save("/kaggle/working/unet_final_model.h5")

# 繪製訓練曲線
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.title("Accuracy")
plt.legend()

plt.show()



import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def unet_classification_model(input_size=(128, 128, 1)):
    inputs = Input(input_size)
    
    # 下採樣部分 (Encoder)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # 底層
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv4)

    # 上採樣部分 (Decoder)
    up5 = UpSampling2D(size=(2, 2))(conv4)
    merge5 = Concatenate()([conv3, up5])
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    merge6 = Concatenate()([conv2, up6])
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    merge7 = Concatenate()([conv1, up7])
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(merge7)

    # 分類層 (全局池化+二元分類)
    global_avg_pool = GlobalAveragePooling2D()(conv7)
    dropout = Dropout(0.5)(global_avg_pool)
    output = Dense(1, activation='sigmoid')(dropout)

    model = Model(inputs, output)
    return model



# 構建 U-Net 二分類模型
model = unet_classification_model()

# 編譯模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 訓練模型
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_val, y_val),
    verbose=1
)


import matplotlib.pyplot as plt

# 繪製準確率曲線
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('U-Net Classification Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 繪製損失曲線
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('U-Net Classification Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

loss, accuracy = model.evaluate(X_val, y_val)
print(f"測試集準確率: {accuracy:.4f}")


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
from tqdm import tqdm

BASE_PATH = '/kaggle/input/rsna-2022-cervical-spine-fracture-detection'
SEGMENTATIONS_PATH = os.path.join(BASE_PATH, 'segmentations')

segmentation_files = os.listdir(SEGMENTATIONS_PATH)
print(f"\nNumber of segmentation files: {len(segmentation_files)}")
print("Sample segmentation file names:")
print(segmentation_files[:5])


pip install nibabel



import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# 讀取 NIfTI 檔案
nii_path = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/segmentations/1.2.826.0.1.3680043.10633.nii"
nii_image = nib.load(nii_path)
image_data = nii_image.get_fdata()

# 顯示某一切片
plt.imshow(image_data[:, :, image_data.shape[2] // 2], cmap="gray")
plt.title("NIfTI 切片影像")
plt.axis("off")
plt.show()

np.load("/mnt/data/ct_image.npz")



import nibabel as nib
import numpy as np

# 設定 NIfTI 檔案路徑
nii_file_path = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/segmentations/1.2.826.0.1.3680043.10633.nii"

# 讀取 NIfTI 檔案
nii_image = nib.load(nii_file_path)

# 取得影像數據（3D NumPy 陣列）
image_data = nii_image.get_fdata()

# 取得影像的頭部資訊
header = nii_image.header

# 顯示影像的基本資訊
print(f"影像尺寸: {image_data.shape}")
print(f"體素間距: {header.get_zooms()}")
print(f"資料類型: {image_data.dtype}")

# 檢查影像的數值範圍（可能標註區域）
print(f"影像最小值: {np.min(image_data)}, 影像最大值: {np.max(image_data)}")

# 嘗試識別可能的標註區域（如 C1~C7）
unique_values = np.unique(image_data)
print(f"影像中的唯一值: {unique_values[:20]}")  # 列出前 20 個唯一數值

# 如果影像內有離散標籤（如 1, 2, 3, 4, 5, 6, 7），可能是 C1~C7 的標註
c1_c7_labels = [val for val in unique_values if 1 <= val <= 7]
print(f"可能的 C1~C7 標註值: {c1_c7_labels}")



import nibabel as nib
import numpy as np
import pandas as pd
from collections import Counter

# 讀取 .nii 影像數據
path = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/segmentations/1.2.826.0.1.3680043.10633.nii"
imgs = nib.load(path).get_fdata()

print(f"NIfTI 影像尺寸: {imgs.shape}")  # (512, 512, 339)

# 計算像素頻率函數
def count_pixel_frequency(slice_array):
    """
    計算單張切片中各像素值的頻率，僅保留 C1~C7 (1.0 ~ 7.0)
    """
    flat_array = slice_array.flatten()
    freq_dict = dict(Counter(flat_array))
    
    # 過濾掉非 C1~C7 的值
    filtered_freq = {k: v for k, v in freq_dict.items() if 1.0 <= k <= 7.0}
    return filtered_freq

# 儲存結果
results = []

# 逐張切片計算
for idx in range(imgs.shape[2]):
    freq_result = count_pixel_frequency(imgs[:, :, idx])
    
    if freq_result:  # 只保留含有 C1~C7 的切片
        df = pd.DataFrame(list(freq_result.items()), columns=["Pixel Value", "Frequency"])
        df["Slice"] = idx + 1  # 記錄切片索引
        results.append(df)
        print(f"切片 {idx+1}: {freq_result}")
    
    # 控制批次數量，防止過多輸出
    if len(results) >= 400:
        break

# 合併所有結果為 DataFrame
final_df = pd.concat(results, ignore_index=True)

# 顯示結果
import ace_tools as tools
tools.display_dataframe_to_user(name="C1~C7 Pixel Frequency Analysis", dataframe=final_df)



import nibabel as nib
import numpy as np
import pandas as pd
import os
import cv2
import pydicom
import matplotlib.pyplot as plt
from collections import Counter

# 設定路徑
nii_path = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/segmentations/1.2.826.0.1.3680043.10633.nii"
dcm_folder = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images/1.2.826.0.1.3680043.10633"
csv_output_path = "/kaggle/working/segment_labels.csv"

# 讀取 .nii 影像數據
imgs = nib.load(nii_path).get_fdata()

print(f"NIfTI 影像尺寸: {imgs.shape}")  # (512, 512, 339)

# 計算像素頻率函數
def count_pixel_frequency(slice_array):
    """
    計算單張切片中各像素值的頻率，僅保留 C1~C7 (1.0 ~ 7.0)
    """
    flat_array = slice_array.flatten()
    freq_dict = dict(Counter(flat_array))
    
    # 過濾掉非 C1~C7 的值
    filtered_freq = {k: v for k, v in freq_dict.items() if 1.0 <= k <= 7.0}
    return filtered_freq

# 影像處理函數
def preprocess_image(image):
    denoised = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    gamma = 1.2
    gamma_corrected = np.power(enhanced / 255.0, gamma) * 255
    return gamma_corrected.astype(np.uint8)

def edge_detection(image):
    edges = cv2.Canny(image, 50, 150)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = np.clip(laplacian, 0, 255).astype(np.uint8)
    return cv2.bitwise_or(edges, laplacian)

def watershed_segmentation(image, edges):
    _, binary_edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    sure_bg = np.uint8(sure_bg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)
    return np.where(markers > 1, 255, 0).astype('uint8')

# 儲存結果
results = []

# 逐張切片計算
for idx in range(imgs.shape[2]):
    freq_result = count_pixel_frequency(imgs[:, :, idx])
    
    if freq_result:  # 只保留含有 C1~C7 的切片
        dicom_path = os.path.join(dcm_folder, f"{idx+1}.dcm")
        dicom_data = pydicom.dcmread(dicom_path)
        image = dicom_data.pixel_array
        if image.dtype != np.uint8:
            image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
        
        preprocessed_image = preprocess_image(image)
        edges = edge_detection(preprocessed_image)
        watershed_result = watershed_segmentation(preprocessed_image, edges)
        
        for label, count in freq_result.items():
            results.append({
                "DICOM Path": dicom_path,
                "Slice": idx + 1,
                "Label": int(label),
                "Pixel Count": count
            })
        print(f"切片 {idx+1}: {freq_result}")
    
    if len(results) >= 400:
        break

# 轉換為 DataFrame 並儲存為 CSV
final_df = pd.DataFrame(results)
final_df.to_csv(csv_output_path, index=False)

# 顯示 CSV 內容
import ace_tools as tools
tools.display_dataframe_to_user(name="Segment Labels", dataframe=final_df)


