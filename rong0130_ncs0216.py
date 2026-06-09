import os
import numpy as np
import pandas as pd
import tensorflow as tf
import SimpleITK as sitk
import cv2
from tensorflow.keras import layers, models, callbacks

# ====================================================
# (A) 讀取原始 CSV 並自動衍生 is_cervical 標籤
# ====================================================
# 路徑請根據你的環境調整
train_images_dir = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images"
labels_csv_path = "/kaggle/input/vertebrae-labels/vertebrae_labels.csv"  # 原始 CSV，包含 Label1~Label19 及 Fracture 標籤

df = pd.read_csv(labels_csv_path)
print("原始資料筆數:", len(df))

# 利用 Label1～Label7（假設這幾個欄位代表頸椎各部位）衍生 is_cervical
label_cols = ["Label1", "Label2", "Label3", "Label4", "Label5", "Label6", "Label7"]
df["is_cervical"] = df[label_cols].max(axis=1)
print("is_cervical 分布:")
print(df["is_cervical"].value_counts())

# ====================================================
# (B) 過濾出實際存在的 DICOM 檔案
# ====================================================
valid_rows = []
for i, row in df.iterrows():
    spine_id = row["SpineID"]
    slice_num = int(row["SliceNumber"])
    dicom_path = os.path.join(train_images_dir, spine_id, f"{slice_num}.dcm")
    if os.path.exists(dicom_path):
        valid_rows.append(i)
df_filtered = df.loc[valid_rows].reset_index(drop=True)
filtered_csv_path = "/kaggle/working/filtered_vertebrae_labels.csv"
df_filtered.to_csv(filtered_csv_path, index=False)
print("過濾後筆數:", len(df_filtered))

# ====================================================
# (C) Stage 1：建立頸椎分類器資料集
# ====================================================
# 取出所有影像與 is_cervical 標籤
spine_ids_stage1 = df_filtered["SpineID"].astype(str).values
slice_nums_stage1 = df_filtered["SliceNumber"].astype(np.int32).values
is_cervical_labels = df_filtered["is_cervical"].values.astype(np.float32)

# 定義讀取影像函式（共用）
def load_dicom_image(spine_id, slice_number):
    spine_id = spine_id.numpy().decode("utf-8")
    slice_number = int(slice_number.numpy())
    dicom_path = os.path.join(train_images_dir, spine_id, f"{slice_number}.dcm")
    try:
        dicom_data = sitk.ReadImage(dicom_path)
        image = sitk.GetArrayFromImage(dicom_data)[0]
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image = cv2.resize(image, (256,256))
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    except:
        image = np.zeros((256,256,3), dtype=np.uint8)
    return image

def load_sample_stage1(spine_id, slice_number, is_cervical):
    image = tf.py_function(func=load_dicom_image, inp=[spine_id, slice_number], Tout=tf.uint8)
    image = tf.cast(image, tf.float32) / 255.0
    image.set_shape([256,256,3])
    is_cervical = tf.cast(is_cervical, tf.float32)
    return image, is_cervical

# 在 NumPy 層面切分 80% 訓練 / 20% 驗證
dataset_size_stage1 = len(spine_ids_stage1)
train_size_stage1 = int(0.8 * dataset_size_stage1)

train_spine_ids_stage1 = spine_ids_stage1[:train_size_stage1]
train_slice_nums_stage1 = slice_nums_stage1[:train_size_stage1]
train_is_cervical = is_cervical_labels[:train_size_stage1]

val_spine_ids_stage1 = spine_ids_stage1[train_size_stage1:]
val_slice_nums_stage1 = slice_nums_stage1[train_size_stage1:]
val_is_cervical = is_cervical_labels[train_size_stage1:]

train_dataset_stage1 = tf.data.Dataset.from_tensor_slices((train_spine_ids_stage1, train_slice_nums_stage1, train_is_cervical))
train_dataset_stage1 = train_dataset_stage1.map(load_sample_stage1, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset_stage1 = train_dataset_stage1.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

val_dataset_stage1 = tf.data.Dataset.from_tensor_slices((val_spine_ids_stage1, val_slice_nums_stage1, val_is_cervical))
val_dataset_stage1 = val_dataset_stage1.map(load_sample_stage1, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset_stage1 = val_dataset_stage1.batch(32).prefetch(tf.data.AUTOTUNE)

print("Stage 1 - 訓練集 batch 數:", tf.data.experimental.cardinality(train_dataset_stage1).numpy())
print("Stage 1 - 驗證集 batch 數:", tf.data.experimental.cardinality(val_dataset_stage1).numpy())

# --------------------------------------------------
# Stage 1: 建立頸椎分類器模型 (二分類)
# --------------------------------------------------
def build_cervical_classifier(input_shape=(256,256,3)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    return model

cervical_model = build_cervical_classifier()
cervical_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cervical_model.summary()

# 訓練 Stage 1 模型
cervical_history = cervical_model.fit(train_dataset_stage1, epochs=10, validation_data=val_dataset_stage1)

# --------------------------------------------------
# (D) Stage 2：建立頸椎骨折預測模型資料集
# --------------------------------------------------
# 從 df_filtered 中只取出 is_cervical==1 的資料
df_fracture = df_filtered[df_filtered["is_cervical"] == 1].reset_index(drop=True)
print("Stage 2: 頸椎骨折資料筆數:", len(df_fracture))

target_fracture_cols = ["Fracture_C1","Fracture_C2","Fracture_C3","Fracture_C4","Fracture_C5","Fracture_C6","Fracture_C7"]
spine_ids_fracture = df_fracture["SpineID"].astype(str).values
slice_nums_fracture = df_fracture["SliceNumber"].astype(np.int32).values
fracture_labels = df_fracture[target_fracture_cols].values.astype(np.float32)

dataset_size_fracture = len(spine_ids_fracture)
train_size_fracture = int(0.8 * dataset_size_fracture)

train_spine_ids_fracture = spine_ids_fracture[:train_size_fracture]
train_slice_nums_fracture = slice_nums_fracture[:train_size_fracture]
train_fracture_labels = fracture_labels[:train_size_fracture]

val_spine_ids_fracture = spine_ids_fracture[train_size_fracture:]
val_slice_nums_fracture = slice_nums_fracture[train_size_fracture:]
val_fracture_labels = fracture_labels[train_size_fracture:]

def load_sample_fracture(spine_id, slice_number, label):
    image = tf.py_function(func=load_dicom_image, inp=[spine_id, slice_number], Tout=tf.uint8)
    image = tf.cast(image, tf.float32) / 255.0
    image.set_shape([256,256,3])
    label = tf.cast(label, tf.float32)
    return image, label

train_dataset_fracture = tf.data.Dataset.from_tensor_slices((train_spine_ids_fracture, train_slice_nums_fracture, train_fracture_labels))
train_dataset_fracture = train_dataset_fracture.map(load_sample_fracture, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset_fracture = train_dataset_fracture.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

val_dataset_fracture = tf.data.Dataset.from_tensor_slices((val_spine_ids_fracture, val_slice_nums_fracture, val_fracture_labels))
val_dataset_fracture = val_dataset_fracture.map(load_sample_fracture, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset_fracture = val_dataset_fracture.batch(32).prefetch(tf.data.AUTOTUNE)

print("Stage 2 - 訓練集 batch 數:", tf.data.experimental.cardinality(train_dataset_fracture).numpy())
print("Stage 2 - 驗證集 batch 數:", tf.data.experimental.cardinality(val_dataset_fracture).numpy())

# --------------------------------------------------
# Stage 2: 建立改進版 UNet 骨折預測模型 (7 標籤)
# --------------------------------------------------
def build_improved_unet_classifier(input_shape=(256,256,3), num_classes=7):
    inputs = layers.Input(shape=input_shape)
    # Encoder Block 1
    c1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2,2))(c1)
    
    # Encoder Block 2
    c2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2,2))(c2)
    
    # Encoder Block 3
    c3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2,2))(c3)
    
    # Encoder Block 4
    c4 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    p4 = layers.MaxPooling2D((2,2))(c4)
    
    # Bottleneck
    c5 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(p4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)
    
    # Decoder Block 1
    u6 = layers.UpSampling2D((2,2))(c5)
    u6 = layers.Concatenate()([u6, c4])
    c6 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(u6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)
    
    # Decoder Block 2
    u7 = layers.UpSampling2D((2,2))(c6)
    u7 = layers.Concatenate()([u7, c3])
    c7 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(c7)
    c7 = layers.BatchNormalization()(c7)
    
    # Decoder Block 3
    u8 = layers.UpSampling2D((2,2))(c7)
    u8 = layers.Concatenate()([u8, c2])
    c8 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(u8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c8)
    c8 = layers.BatchNormalization()(c8)
    
    # Decoder Block 4
    u9 = layers.UpSampling2D((2,2))(c8)
    u9 = layers.Concatenate()([u9, c1])
    c9 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(u9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(c9)
    c9 = layers.BatchNormalization()(c9)
    
    gap = layers.GlobalAveragePooling2D()(c9)
    d1 = layers.Dense(64, activation='relu')(gap)
    d1 = layers.Dropout(0.5)(d1)
    outputs = layers.Dense(num_classes, activation='sigmoid')(d1)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

fracture_model = build_improved_unet_classifier()
fracture_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')])
fracture_model.summary()

# 訓練 Stage 2 模型
fracture_history = fracture_model.fit(train_dataset_fracture, epochs=50, validation_data=val_dataset_fracture)

# --------------------------------------------------
# (E) 最終應用：兩階段推論函式，並顯示準確率（機率）
# --------------------------------------------------
def predict_image(spine_id, slice_number):
    # 讀取影像
    image = load_dicom_image(tf.constant(spine_id), tf.constant(slice_number))
    image = image.astype(np.float32) / 255.0
    image_exp = np.expand_dims(image, axis=0)
    
    # 第一階段：使用頸椎分類器判斷是否為頸椎
    cervical_pred = cervical_model.predict(image_exp)
    cervical_prob = cervical_pred[0,0]
    is_cervical = cervical_prob > 0.5
    print(f"頸椎分類器信心：{cervical_prob:.2f}")
    
    if is_cervical:
        # 如果是頸椎，使用骨折預測模型
        fracture_pred = fracture_model.predict(image_exp)
        # fracture_pred 為 7 個標籤的機率
        fracture_status = (fracture_pred[0] > 0.5).astype(int)
        print("該影像為頸椎。")
        print("各椎體骨折預測 (機率)：", np.round(fracture_pred[0], 2))
        print("各椎體骨折狀態 (0/1)：", fracture_status)
        return fracture_status, fracture_pred[0]
    else:
        print("該影像並非頸椎影像。")
        return None, None

# 測試推論 (請根據你的資料替換 SpineID 與 SliceNumber)
predict_image("1.2.826.0.1.3680043.780", 5)


