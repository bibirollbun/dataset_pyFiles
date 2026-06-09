import kagglehub
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from IPython.display import FileLink

# 檢查 GPU 是否可用
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if tf.test.is_gpu_available():
    print("GPU is available and will be used for training.")
else:
    print("GPU is not available. Training will use CPU. Consider enabling GPU in Kaggle notebook settings.")

# 定義數據路徑
path = kagglehub.dataset_download("darkfanxing/ntutemnist")
DATA_DIR = '/kaggle/input/ntutemnist/'
TRAIN_DATA_FILE = DATA_DIR + 'emnist-byclass-train.npz'
TEST_DATA_FILE = DATA_DIR + 'emnist-byclass-test.npz'

# 加載訓練數據
data = np.load(TRAIN_DATA_FILE)
train_labels = data['training_labels']
train_images = data['training_images']

# 加載測試數據
test_images = np.load(TEST_DATA_FILE)['testing_images']

# 將圖像重塑為 (28, 28, 1) 並標準化像素值
train_images = train_images.reshape((-1, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((-1, 28, 28, 1)).astype('float32') / 255

# 將標籤轉為 one-hot 編碼
train_labels = to_categorical(train_labels, num_classes=62)

# 分割訓練數據為訓練集與驗證集（80% 訓練，20% 驗證）
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

# 定義單例預處理層（在函數外部創建，避免重複創建）
rotation_layer = layers.RandomRotation(factor=(-0.0524, 0.0524), fill_mode='nearest')  # 3 度 = 0.0524 弧度
translation_layer = layers.RandomTranslation(height_factor=(-0.03, 0.03), width_factor=(-0.03, 0.03), fill_mode='nearest')
zoom_layer = layers.RandomZoom(height_factor=(-0.03, 0.03), fill_mode='nearest')
flip_layer = layers.RandomFlip(mode='horizontal')

# 創建 tf.data.Dataset 來優化數據加載（結合數據增強）
def create_augmented_dataset(images, labels, batch_size=128, shuffle=True):
    # 確保輸入是 NumPy 陣列
    images = np.array(images)
    labels = np.array(labels) if labels is not None else None
    
    # 定義數據增強函數（重用單例預處理層）
    def augment_image(image, label):
        # 將 TensorFlow 張量直接應用數據增強
        image = tf.cast(image, tf.float32)
        
        # 應用單例預處理層
        augmented = rotation_layer(image)
        augmented = translation_layer(augmented)
        augmented = zoom_layer(augmented)
        augmented = flip_layer(augmented)

        return augmented, label

    # 創建數據集
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))
    dataset = dataset.batch(batch_size)
    
    # 應用數據增強
    dataset = dataset.map(
        augment_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # 預提取數據，優化 GPU 利用率
    return dataset

# 創建訓練和驗證數據集
train_dataset = create_augmented_dataset(train_images, train_labels, batch_size=128, shuffle=True)
val_dataset = create_augmented_dataset(val_images, val_labels, batch_size=128, shuffle=False)

# 檢查並設置 GPU 設備
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 限制 GPU 記憶體使用（按需增長）
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
    except RuntimeError as e:
        print(e)

# 使用 MirroredStrategy 確保在 GPU 上運行（適用單 GPU 或多 GPU）
strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')

# 在策略範圍內定義和訓練模型
with strategy.scope():
    # 定義改進的 CNN 架構（簡化模型，減少過擬合）
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),  # 進一步降低 Dropout 比例
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(62, activation='softmax')  # 62 類輸出層
    ])

    # 編譯模型（調整學習率）
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # 降低學習率
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 定義早停與模型檢查點
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = callbacks.ModelCheckpoint('best_model.weights.h5', save_best_only=True, save_weights_only=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-6)  # 調整學習率衰減

    # 訓練模型（使用 tf.data.Dataset，增加 epochs 並觀察收斂）
    history = model.fit(
        train_dataset,  # 使用優化的 tf.data.Dataset 與數據增強
        epochs=20,  # 增加訓練次數，允許更多收斂
        validation_data=val_dataset,  # 驗證數據集
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1
    )

# 創建測試數據集（無數據增強）
def create_dataset(images, labels, batch_size=128, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # 預提取數據，優化 GPU 利用率
    return dataset

test_dataset = create_dataset(test_images, None, batch_size=128, shuffle=False)

# 預測測試數據的類別（在 GPU 上運行）
with strategy.scope():
    predictions = model.predict(test_dataset).argmax(axis=-1)

# 將預測結果儲存為 CSV 檔案
with open('pred_results.csv', 'w') as f:
    f.write('Id,Category\n')  # 寫入標頭
    for i, pred in enumerate(predictions):
        f.write(f'{i},{pred}\n')  # 寫入每個預測結果

# 在 Kaggle 中提供下載鏈接
FileLink('pred_results.csv')

