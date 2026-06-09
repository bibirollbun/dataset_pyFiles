import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split  
import cv2  
import os
from PIL import Image  
import warnings
import torch
from torch.utils.data import WeightedRandomSampler
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")



import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy

physical_devices = tf.config.list_physical_devices()
print("Available physical devices:", physical_devices)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:

        tf.config.set_visible_devices(gpus[0], 'GPU')
        
        tf.config.experimental.set_memory_growth(gpus[0], True)
        
        set_global_policy('mixed_float16')
        
        print("GPU is available and set for use:", gpus)
        print("Mixed precision enabled for faster training.")
    except RuntimeError as e:
        print("Error setting GPU:", e)
else:
    print("No GPU detected. Falling back to CPU. Please check Kaggle accelerator settings.")

# Optional: In version TF để confirm
print("TensorFlow version:", tf.__version__)


# Data 
data_path = '/kaggle/input/cassava-leaf-disease-classification/'


# Đọc và đếm dữ liệu từ tập train:
train_csv_path = os.path.join(data_path, 'train.csv')
train_df = pd.read_csv(train_csv_path)
train_df.head(10)


# Xem overview
print("Shape:", train_df.shape)
print("\nThông tin cột:")
print(train_df.info())
print("\nSố lượng unique labels:", train_df['label'].nunique())


# Kiểm tra duplicate
duplicates = train_df['image_id'].duplicated().sum()
print(f"Số duplicate image_id: {duplicates}")

if duplicates > 0:
    train_df = train_df.drop_duplicates(subset=['image_id'], keep='first')
    print("Đã drop duplicates!")


train_df['label'].value_counts()


import json
with open('../input/cassava-leaf-disease-classification/label_num_to_disease_map.json') as file:
    print(json.dumps(json.loads(file.read()), indent=4))


import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')  
sns.set_palette("viridis")  

plt.figure(figsize=(12, 6))  
ax = sns.countplot(data=train_df, x='label', palette='viridis')

plt.title('Phân phối các lớp bệnh lá sắn (Cassava Leaf Disease)', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Lớp bệnh (0: CBB, 1: CBSD, 2: CGM, 3: CMD, 4: Healthy)', fontsize=14)
plt.ylabel('Số lượng ảnh', fontsize=14)

label_names = ['CBB', 'CBSD', 'CGM', 'CMD', 'Healthy']
plt.xticks(ticks=range(5), labels=label_names, rotation=45, ha='right')

ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

# Tính số lượng từng label 
label_counts = train_df['label'].value_counts().sort_index()

# Tên lớp đơn giản
label_names = ['CBB', 'CBSD', 'CGM', 'CMD', 'Healthy']
colors = ['red', 'blue', 'green', 'orange', 'purple']

# Vẽ pie chart 
plt.figure(figsize=(8, 6))
plt.pie(label_counts.values, 
        labels=[label_names[i] for i in label_counts.index], 
        autopct='%1.1f%%', 
        startangle=90,
        colors=colors) 

plt.title('Phân phối lớp bệnh lá sắn')
plt.axis('equal')
plt.show()


# Lấy và hiển thị 6 ảnh ngẫu nhiên từ lớp bệnh Bacterial Blight

sample = train_df[train_df.label == 0].sample(6, random_state=42) 
plt.figure(figsize=(12, 8))

for ind, (image_id, label) in enumerate(zip(sample.image_id, sample.label)):
    plt.subplot(2, 3, ind + 1)
    
    img_path = os.path.join("../input/cassava-leaf-disease-classification/train_images", image_id)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image)
    plt.axis("off")
    
plt.suptitle('Cassava Bacterial Blight', fontsize=14, fontweight='bold')
plt.tight_layout()  
plt.show()


sample = train_df[train_df.label == 1].sample(6, random_state=42)  

plt.figure(figsize=(12, 8))
for ind, (image_id, label) in enumerate(zip(sample.image_id, sample.label)):
    plt.subplot(2, 3, ind + 1)
    
    img_path = os.path.join("../input/cassava-leaf-disease-classification/train_images", image_id)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image)
    plt.axis("off")

plt.suptitle('Cassava Brown Streak Disease', fontsize=14, fontweight='bold')
plt.tight_layout() 
plt.show()


sample = train_df[train_df.label == 2].sample(6, random_state=42)  

plt.figure(figsize=(12, 8))
for ind, (image_id, label) in enumerate(zip(sample.image_id, sample.label)):
    plt.subplot(2, 3, ind + 1)
    
    img_path = os.path.join("../input/cassava-leaf-disease-classification/train_images", image_id)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image)
    plt.axis("off")

plt.suptitle('Cassava Green Mottle', fontsize=14, fontweight='bold')
plt.tight_layout() 
plt.show()


sample = train_df[train_df.label == 3].sample(6, random_state=42)  

plt.figure(figsize=(12, 8))
for ind, (image_id, label) in enumerate(zip(sample.image_id, sample.label)):
    plt.subplot(2, 3, ind + 1)
    
    img_path = os.path.join("../input/cassava-leaf-disease-classification/train_images", image_id)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image)
    plt.axis("off")

plt.suptitle('Cassava Mosaic Disease', fontsize=14, fontweight='bold')
plt.tight_layout() 
plt.show()


sample = train_df[train_df.label == 4].sample(6, random_state=42)  

plt.figure(figsize=(12, 8))
for ind, (image_id, label) in enumerate(zip(sample.image_id, sample.label)):
    plt.subplot(2, 3, ind + 1)
    
    img_path = os.path.join("../input/cassava-leaf-disease-classification/train_images", image_id)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image)
    plt.axis("off")

plt.suptitle('Healthy', fontsize=14, fontweight='bold')
plt.tight_layout() 
plt.show()


train_img_dir = os.path.join(data_path, 'train_images')

# Thêm đường dẫn đầy đủ vào DataFrame để tiện dùng
train_df['file_path'] = train_df['image_id'].apply(lambda x: os.path.join(train_img_dir, x))
train_df.head()


train_df, val_df = train_test_split(
    train_df,
    test_size=0.15,
    stratify=train_df['label'],
    random_state=42
)

print(f"Số lượng ảnh train: {len(train_df)}, val: {len(val_df)}")


from sklearn.utils.class_weight import compute_class_weight

train_labels_for_weights = train_df['label'].values
classes = np.unique(train_labels_for_weights)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=train_labels_for_weights
)

class_weight_dict = dict(zip(classes, class_weights))

print("Danh sách các lớp:", classes)
print("Trọng số tương ứng cho từng lớp:", class_weights)
print("\nTừ điển Class Weight cho Keras model.fit():")
print(class_weight_dict)


import tensorflow as tf

IMAGE_SIZE = (224, 224) 
BATCH_SIZE = 32

train_filepaths = train_df['file_path'].values
train_labels = train_df['label'].astype(np.int32).values

val_filepaths = val_df['file_path'].values
val_labels = val_df['label'].astype(np.int32).values

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    tf.keras.layers.RandomContrast(0.2),
], name="data_augmentation")


def build_dataset(filepaths, labels, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    
    SHUFFLE_BUFFER_SIZE = 1024 

    def decode_image(filepath, label):
        image = tf.io.read_file(filepath)
        image = tf.image.decode_jpeg(image, channels=3)
        return image, label

    dataset = dataset.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE) 
        
        dataset = dataset.map(lambda image, label: (data_augmentation(image, training=True), label),
                              num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.map(lambda image, label: (tf.image.resize(image, IMAGE_SIZE), label),
                          num_parallel_calls=tf.data.AUTOTUNE)
                          
    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.map(lambda images, labels: (tf.keras.applications.resnet50.preprocess_input(images), labels),
                          num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

train_ds = build_dataset(train_filepaths, train_labels, is_training=True)
val_ds = build_dataset(val_filepaths, val_labels, is_training=False)



class_counts = train_df['label'].value_counts()
class_weights_map = (1.0 / class_counts).to_dict()

train_df['sample_weight'] = train_df['label'].map(class_weights_map)
simulated_balanced_df = train_df.sample(
    n=len(train_df),
    weights='sample_weight',
    replace=True,
    random_state=42
)
sampled_labels = simulated_balanced_df['label']

fig, axes = plt.subplots(1, 1, figsize=(12, 7)) 

sns.countplot(ax=axes, x=sampled_labels, palette='plasma')
axes.set_title("Phân phối nhãn Sau khi Mô phỏng Weighted Sampling", fontsize=16)
axes.set_xlabel("Nhãn", fontsize=12)
axes.set_ylabel("Số lượng mẫu", fontsize=12)

plt.tight_layout()
plt.show()

train_df = train_df.drop(columns=['sample_weight'])




from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

NUM_CLASSES = len(train_df['label'].unique())

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

model = Model(inputs=base_model.input, outputs=predictions)

print(f"Đã xây dựng mô hình thành công với {NUM_CLASSES} lớp đầu ra.")
model.summary()


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

model_checkpoint = ModelCheckpoint(
    filepath='best_resnet50_model.keras',
    save_best_only=True,
    monitor='val_sparse_categorical_accuracy',
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)

EPOCHS = 20

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[model_checkpoint, early_stopping],
    class_weight=class_weight_dict
)

print("\nHoàn tất quá trình huấn luyện!")


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

best_model = load_model('best_resnet50_model.keras')
print("Tải mô hình thành công!")

best_model.trainable = True
freeze_until = 140 

for layer in best_model.layers[:freeze_until]:
    layer.trainable = False

best_model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)

initial_epochs = 5 
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine_tune = best_model.fit(
    train_ds,
    epochs=total_epochs,
    initial_epoch=initial_epochs,
    validation_data=val_ds,
    callbacks=[model_checkpoint, early_stopping]
)


try:
    history_df1 = pd.DataFrame(history.history)
except NameError:
    history_df1 = pd.DataFrame() 

history_df2 = pd.DataFrame(history_fine_tune.history)

if not history_df1.empty:
    full_history_df = pd.concat([history_df1, history_df2], axis=0)
else:
    full_history_df = history_df2

full_history_df = full_history_df.reset_index(drop=True)

best_epoch = full_history_df['val_sparse_categorical_accuracy'].idxmax()
best_val_acc = full_history_df['val_sparse_categorical_accuracy'].max()

fig, axes = plt.subplots(1, 2, figsize=(20, 7))
fig.suptitle('Biểu đồ Toàn bộ Lịch sử Huấn luyện (Feature Extraction + Fine-Tuning)', fontsize=16)

axes[0].plot(full_history_df.index, full_history_df['sparse_categorical_accuracy'], label='Train Accuracy', color='blue', marker='o', markersize=3)
axes[0].plot(full_history_df.index, full_history_df['val_sparse_categorical_accuracy'], label='Validation Accuracy', color='orange', marker='o', markersize=3)
axes[0].set_title('Training & Validation Accuracy', fontsize=14)
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Accuracy')
axes[0].scatter(best_epoch, best_val_acc, s=150, c='red', zorder=5, label=f'Best Val Acc: {best_val_acc:.4f} at Epoch {best_epoch+1}')

if not history_df1.empty:
    axes[0].axvline(x=len(history_df1)-1, color='grey', linestyle='--', label='Start Fine-Tuning')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(full_history_df.index, full_history_df['loss'], label='Train Loss', color='blue', marker='o', markersize=3)
axes[1].plot(full_history_df.index, full_history_df['val_loss'], label='Validation Loss', color='orange', marker='o', markersize=3)
axes[1].set_title('Training & Validation Loss', fontsize=14)
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')

if not history_df1.empty:
    axes[1].axvline(x=len(history_df1)-1, color='grey', linestyle='--', label='Start Fine-Tuning')
axes[1].legend()
axes[1].grid(True)

plt.show()


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("Tải mô hình tốt nhất từ file 'best_resnet50_model.keras' để đánh giá...")
best_model = tf.keras.models.load_model('best_resnet50_model.keras')

print("\n--- Đánh giá hiệu năng tổng quan trên tập Validation ---")
loss, accuracy = best_model.evaluate(val_ds, verbose=0)
print(f"  - Validation Loss: {loss:.4f}")
print(f"  - Validation Accuracy: {accuracy:.4f} ({accuracy:.2%})")

y_true = np.concatenate([labels for images, labels in val_ds], axis=0)
y_pred_probs = best_model.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

label_map = {0: 'CBB', 1: 'CBSD', 2: 'CGM', 3: 'CMD', 4: 'Healthy'}
label_names = [label_map[i] for i in range(NUM_CLASSES)]

print("\n--- Báo cáo Phân loại Chi tiết (Precision, Recall, F1-Score) ---")
print(classification_report(y_true, y_pred, target_names=label_names))

fig, axes = plt.subplots(1, 2, figsize=(22, 8))
fig.suptitle('Trực quan hóa Đánh giá Hiệu năng Mô hình', fontsize=20)

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names, yticklabels=label_names, ax=axes[0])
axes[0].set_title('Ma trận nhầm lẫn', fontsize=16)
axes[0].set_ylabel('Nhãn thực tế (Actual Label)', fontsize=12)
axes[0].set_xlabel('Nhãn dự đoán (Predicted Label)', fontsize=12)

report_dict = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg']) 
report_df[['precision', 'recall', 'f1-score']].plot(kind='bar', ax=axes[1], colormap='viridis')
axes[1].set_title('Các chỉ số theo từng lớp', fontsize=16)
axes[1].set_xlabel('Lớp bệnh', fontsize=12)
axes[1].set_ylabel('Điểm số', fontsize=12)
axes[1].tick_params(axis='x', rotation=45) 
axes[1].grid(axis='y', linestyle='--')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

print("Đang tạo Confusion Matrix...")

y_pred_probs = model.predict(val_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)

y_true = np.concatenate([labels for images, labels in val_dataset], axis=0)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title('Confusion Matrix on Validation Set')
plt.show()

