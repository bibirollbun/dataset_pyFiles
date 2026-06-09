pip install tqdm


import pandas as pd

def process_csv(file_path, base_path):
    """
    Đọc file CSV, tách từng dòng dựa vào khoảng trắng và lấy các cột cần thiết.
    Thêm chuỗi `base_path` vào cột `image_path`, chuyển label '3' thành '1',
    và giữ lại các cột bounding box (cột 3-6 cho box 1, cột 7-10 cho box 2).
    Nếu thiếu cột, gán giá trị bounding box = 0.

    Args:
        file_path (str): Đường dẫn tới file CSV.
        base_path (str): Chuỗi cần thêm vào đầu cột `image_path`.

    Returns:
        pandas.DataFrame: DataFrame chứa các cột `image_path`, `label`, và bounding box.
    """
    result = []

    with open(file_path, 'r') as file:
        for line in file:
            # Tách dòng thành các phần tử bằng khoảng trắng
            splitted = line.strip().split()

            # Đảm bảo có ít nhất 2 cột (image_path và label)
            if len(splitted) >= 2:
                label = splitted[1]
                
                # Nếu label là '3', chuyển thành '1'
                if label == '3':
                    label = '1'
                
                # Tạo dictionary với image_path, label và bounding box
                row = {
                    'image_path': base_path + splitted[0],
                    'label': label,
                    'bbox1_x': splitted[2] if len(splitted) > 2 else '0',  # Cột 3
                    'bbox1_y': splitted[3] if len(splitted) > 3 else '0',  # Cột 4
                    'bbox1_w': splitted[4] if len(splitted) > 4 else '0',  # Cột 5
                    'bbox1_h': splitted[5] if len(splitted) > 5 else '0',  # Cột 6
                    'bbox2_x': splitted[6] if len(splitted) > 6 else '0',  # Cột 7
                    'bbox2_y': splitted[7] if len(splitted) > 7 else '0',  # Cột 8
                    'bbox2_w': splitted[8] if len(splitted) > 8 else '0',  # Cột 9
                    'bbox2_h': splitted[9] if len(splitted) > 9 else '0'   # Cột 10
                }
                result.append(row)

    # Chuyển danh sách sang DataFrame
    data = pd.DataFrame(result)
    return data

# Gọi hàm để xử lý dữ liệu
data_train = process_csv("/kaggle/input/databee/train/train/gt_one.csv", "/kaggle/input/databee/train/train/")
data_test = process_csv("/kaggle/input/databee/test/test/gt_one.csv", "/kaggle/input/databee/test/test/")
data_val = process_csv("/kaggle/input/databee/val/val/gt_one.csv", "/kaggle/input/databee/val/val/")

# Hiển thị kết quả
print("Số lượng dòng trong data_train:")
print(data_train.count())
print("\nSố lượng dòng trong data_test:")
print(data_test.count())
print("\n5 dòng đầu tiên của data_train:")
print(data_train.head())


import pandas as pd
import numpy as np
import cv2
import os
from pathlib import Path
import zipfile
from tqdm import tqdm

# Hàm tạo mặt nạ từ bounding box
def create_mask_from_bbox(image_path, bbox1_x, bbox1_y, bbox1_w, bbox1_h, 
                         bbox2_x, bbox2_y, bbox2_w, bbox2_h):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ {image_path}")
    
    orig_height, orig_width = img.shape[:2]
    mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
    
    try:
        bbox1_x1, bbox1_y1 = int(float(bbox1_x)), int(float(bbox1_y))
        bbox1_x2, bbox1_y2 = int(float(bbox1_w)), int(float(bbox1_h))
        bbox2_x1, bbox2_y1 = int(float(bbox2_x)), int(float(bbox2_y))
        bbox2_x2, bbox2_y2 = int(float(bbox2_w)), int(float(bbox2_h))
    except ValueError:
        return mask
    
    for (x1, y1, x2, y2) in [(bbox1_x1, bbox1_y1, bbox1_x2, bbox1_y2), 
                            (bbox2_x1, bbox2_y1, bbox2_x2, bbox2_y2)]:
        if x1 != 0 or y1 != 0 or x2 != 0 or y2 != 0:
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_width, x2), min(orig_height, y2)
            if x1 < x2 and y1 < y2:
                mask[y1:y2, x1:x2] = 1
    
    return mask

# Hàm tạo mặt nạ cho dataset và lưu vào thư mục
def generate_masks_for_dataset(dataframe, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    records = []
    
    for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Tạo mask"):
        try:
            mask = create_mask_from_bbox(
                row['image_path'], row['bbox1_x'], row['bbox1_y'], row['bbox1_w'], row['bbox1_h'],
                row['bbox2_x'], row['bbox2_y'], row['bbox2_w'], row['bbox2_h']
            )
            # Lấy tên file gốc từ image_path (không bao gồm phần mở rộng)
            image_name = os.path.basename(row['image_path']).rsplit('.', 1)[0]
            mask_path = os.path.join(output_dir, f"{image_name}_mask.png")
            cv2.imwrite(mask_path, mask * 255)
            records.append({
                'image_path': row['image_path'],
                'mask_path': mask_path,
                'label': row['label']
            })
        except Exception:
            pass  # Bỏ qua lỗi, không hiển thị thông báo
    
    return pd.DataFrame(records)

# Hàm lưu DataFrame vào CSV và nén thành file zip
def save_to_csv_and_zip(df, csv_path, zip_path, mask_dir):
    df.to_csv(csv_path, index=False)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_path, os.path.basename(csv_path))
        for mask_file in Path(mask_dir).glob("*.png"):
            zipf.write(mask_file, os.path.join("masks", mask_file.name))

# Giả định rằng data_train, data_val, data_test đã được định nghĩa trước đó
# Nếu chưa, bạn cần load chúng từ file CSV gốc, ví dụ:
# data_train = pd.read_csv('path_to_train.csv')
# data_val = pd.read_csv('path_to_val.csv')
# data_test = pd.read_csv('path_to_test.csv')

# Chạy cho tập train
output_dir_train = "/kaggle/working/masks/train/"
train_info_df = generate_masks_for_dataset(data_train, output_dir_train)
train_csv_path = "/kaggle/working/train_info.csv"
train_zip_path = "/kaggle/working/train_info.zip"
save_to_csv_and_zip(train_info_df, train_csv_path, train_zip_path, output_dir_train)

# Chạy cho tập validation
output_dir_val = "/kaggle/working/masks/val/"  # Sửa từ output_dir_train thành output_dir_val
val_info_df = generate_masks_for_dataset(data_val, output_dir_val)  # Sửa từ train_info_df thành val_info_df
val_csv_path = "/kaggle/working/val_info.csv"  # Sửa từ train_csv_path thành val_csv_path
val_zip_path = "/kaggle/working/val_info.zip"  # Sửa từ train_zip_path thành val_zip_path
save_to_csv_and_zip(val_info_df, val_csv_path, val_zip_path, output_dir_val)

# Chạy cho tập test
output_dir_test = "/kaggle/working/masks/test/"
test_info_df = generate_masks_for_dataset(data_test, output_dir_test)
test_csv_path = "/kaggle/working/test_info.csv"
test_zip_path = "/kaggle/working/test_info.zip"
save_to_csv_and_zip(test_info_df, test_csv_path, test_zip_path, output_dir_test)


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Đọc file CSV cho 3 tập: train, validation, test
train_data = pd.read_csv('/kaggle/working/train_info.csv')
val_data = pd.read_csv('/kaggle/working/val_info.csv')
test_data = pd.read_csv('/kaggle/working/test_info.csv')

# Hàm tiền xử lý ảnh và mặt nạ
def preprocess_image(image_path, mask_path, target_size=(256, 256)):
    try:
        # Đọc ảnh
        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image) / 255.0  # Chuẩn hóa về [0, 1]
        
        # Đọc mặt nạ (giả sử mặt nạ là ảnh grayscale)
        mask = load_img(mask_path, target_size=target_size, color_mode="grayscale")
        mask = img_to_array(mask) / 255.0  # Chuẩn hóa về [0, 1]
        mask = np.round(mask)  # Đảm bảo giá trị nhị phân (0 hoặc 1)
        
        return image, mask
    except Exception as e:
        print(f"Error loading {image_path} or {mask_path}: {e}")
        return None, None

# Hàm generator để load dữ liệu theo batch
def data_generator(dataframe, target_size=(256, 256), batch_size=16):
    while True:
        for start in range(0, len(dataframe), batch_size):
            end = min(start + batch_size, len(dataframe))
            batch_images = []
            batch_masks = []
            for idx in range(start, end):
                image, mask = preprocess_image(dataframe['image_path'][idx], 
                                              dataframe['mask_path'][idx], 
                                              target_size)
                if image is not None and mask is not None:
                    batch_images.append(image)
                    batch_masks.append(mask)
            if batch_images:
                yield np.array(batch_images), np.array(batch_masks)

# Tạo dataset từ generator cho 3 tập
batch_size = 16
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_data, batch_size=batch_size),
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 256, 256, 3], [None, 256, 256, 1])
).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(val_data, batch_size=batch_size),
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 256, 256, 3], [None, 256, 256, 1])
).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(test_data, batch_size=batch_size),
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 256, 256, 3], [None, 256, 256, 1])
).prefetch(tf.data.AUTOTUNE)

# In số lượng mẫu trong mỗi tập để kiểm tra
print("Number of samples in Train set:", len(train_data))
print("Number of samples in Validation set:", len(val_data))
print("Number of samples in Test set:", len(test_data))

# Lấy một batch từ mỗi dataset để kiểm tra shape
train_iter = iter(train_dataset)
val_iter = iter(val_dataset)
test_iter = iter(test_dataset)

train_images, train_masks = next(train_iter)
val_images, val_masks = next(val_iter)
test_images, test_masks = next(test_iter)

print("Train batch shape (images, masks):", train_images.shape, train_masks.shape)
print("Validation batch shape (images, masks):", val_images.shape, val_masks.shape)
print("Test batch shape (images, masks):", test_images.shape, test_masks.shape)


import matplotlib.pyplot as plt

# Hàm hiển thị hình ảnh và mặt nạ
def display_image_mask(image, mask, title="Image and Mask"):
    # Chuyển EagerTensor thành NumPy array
    image = image.numpy() if isinstance(image, tf.Tensor) else image
    mask = mask.numpy() if isinstance(mask, tf.Tensor) else mask
    
    plt.figure(figsize=(10, 5))
    
    # Hiển thị hình ảnh gốc
    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(image)
    plt.axis('off')
    
    # Hiển thị mặt nạ
    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(mask.squeeze(), cmap='gray')  # squeeze() để loại bỏ chiều kênh (1) của mặt nạ
    plt.axis('off')
    
    plt.suptitle(title)
    plt.show()

# Tạo iterator để lấy dữ liệu từ các dataset
train_iter = iter(train_dataset)
val_iter = iter(val_dataset)
test_iter = iter(test_dataset)

# Lấy một batch từ mỗi dataset
train_images, train_masks = next(train_iter)
val_images, val_masks = next(val_iter)
test_images, test_masks = next(test_iter)

# Hiển thị 3 cặp từ tập train
print("Displaying 3 samples from the training set:")
for i in range(min(3, len(train_images))):  # Đảm bảo không vượt quá số mẫu trong batch
    display_image_mask(train_images[i], train_masks[i], title=f"Train Sample {i+1}")

# Hiển thị 3 cặp từ tập validation
print("Displaying 3 samples from the validation set:")
for i in range(min(3, len(val_images))):  # Đảm bảo không vượt quá số mẫu trong batch
    display_image_mask(val_images[i], val_masks[i], title=f"Validation Sample {i+1}")

# Hiển thị 3 cặp từ tập test
print("Displaying 3 samples from the test set:")
for i in range(min(3, len(test_images))):  # Đảm bảo không vượt quá số mẫu trong batch
    display_image_mask(test_images[i], test_masks[i], title=f"Test Sample {i+1}")


# Import các thư viện cần thiết
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback  # Thêm Callback và ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

# Định nghĩa các độ đo tùy chỉnh: IoU và Dice Coefficient
def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + tf.keras.backend.epsilon())

def dice_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + tf.keras.backend.epsilon())

# Xây dựng mô hình U-Net
def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    
    u4 = UpSampling2D((2, 2))(c3)
    u4 = concatenate([u4, c2])
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u4)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    
    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c1])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)
    model = Model(inputs, outputs)
    return model

# Callback để hiển thị thanh tiến trình
class TQDMProgressBar(Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.progress_bar = tqdm(total=self.epochs, desc='Training Progress', unit='epoch')

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.update(1)
        self.progress_bar.set_postfix({
            'loss': f"{logs.get('loss'):.4f}",
            'val_loss': f"{logs.get('val_loss'):.4f}",
            'iou': f"{logs.get('iou_metric'):.4f}",
            'val_iou': f"{logs.get('val_iou_metric'):.4f}",
            'dice': f"{logs.get('dice_metric'):.4f}",
            'val_dice': f"{logs.get('val_dice_metric'):.4f}"
        })

    def on_train_end(self, logs=None):
        self.progress_bar.close()

# Khởi tạo và biên dịch mô hình
model = unet_model()
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=[iou_metric, dice_metric])

# Thêm Early Stopping callback
early_stopping = EarlyStopping(
    monitor='val_iou_metric',  # Theo dõi validation IoU
    mode='max',                # Tối đa hóa IoU
    patience=25,               # Dừng sau 10 epoch nếu không cải thiện
    restore_best_weights=True, # Khôi phục trọng số tốt nhất
    verbose=1
)

# Thêm ModelCheckpoint callback để lưu mô hình tốt nhất
checkpoint = ModelCheckpoint(
    'best_unet_model.h5',      # Đường dẫn lưu mô hình tốt nhất
    monitor='val_iou_metric',  # Theo dõi validation IoU
    mode='max',                # Tối đa hóa IoU
    save_best_only=True,       # Chỉ lưu khi mô hình cải thiện
    verbose=1                  # Hiển thị thông báo khi lưu
)

# Huấn luyện mô hình với cả TQDM, Early Stopping và ModelCheckpoint
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    steps_per_epoch=len(train_data) // batch_size,
    validation_steps=len(val_data) // batch_size,
    epochs=150,
    callbacks=[TQDMProgressBar(), early_stopping, checkpoint],  # Thêm checkpoint vào callbacks
    verbose=0
)

# Lưu mô hình cuối cùng (tùy chọn)
model.save('unet_model.h5')

# Đánh giá mô hình
test_loss, test_iou, test_dice = model.evaluate(
    test_dataset,
    steps=len(test_data) // batch_size,
    verbose=1
)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test IoU: {test_iou:.4f}")
print(f"Test Dice Coefficient: {test_dice:.4f}")

# Vẽ biểu đồ
plt.figure(figsize=(18, 4))
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['iou_metric'], label='Train IoU')
plt.plot(history.history['val_iou_metric'], label='Validation IoU')
plt.title('IoU over Epochs')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history.history['dice_metric'], label='Train Dice')
plt.plot(history.history['val_dice_metric'], label='Validation Dice')
plt.title('Dice Coefficient over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Dice')
plt.legend()

plt.tight_layout()
plt.show()

# Hiển thị dự đoán
test_iterator = iter(test_dataset)
test_images, test_masks = next(test_iterator)
predictions = model.predict(test_images)

for i in range(min(3, len(test_images))):
    plt.figure(figsize=(15, 5))
    
    test_image = test_images[i].numpy() if isinstance(test_images[i], tf.Tensor) else test_images[i]
    test_mask = test_masks[i].numpy() if isinstance(test_masks[i], tf.Tensor) else test_masks[i]
    prediction = predictions[i]
    
    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(test_image)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("True Mask")
    plt.imshow(test_mask.squeeze(), cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(prediction.squeeze(), cmap='gray')
    plt.axis('off')
    
    plt.show()

