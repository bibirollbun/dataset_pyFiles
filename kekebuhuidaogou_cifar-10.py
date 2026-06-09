import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

def simple_augmentation(image, label):
    image = tf.cast(image, tf.float32)
    # Only horizontal flip - keep it simple
    image = tf.image.random_flip_left_right(image)
    return image, label

# 创建数据集管道
def create_dataset(images, labels, batch_size, is_training=True, validation_split=0.15):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    if is_training:
        # 拆分训练数据以进行验证
        dataset_size = len(images)
        val_size = int(dataset_size * validation_split)
        train_size = dataset_size - val_size

        dataset = dataset.shuffle(10000, seed=42)
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)

        # 仅对训练数据应用最小增强
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(5000, reshuffle_each_iteration=True)
        train_dataset = train_dataset.map(simple_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        # 无增强的验证数据集
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset
    else:
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

# 创建参数高效但有效的模型
def create_simple_effective_model():
    inputs = layers.Input(shape=(32, 32, 3))

    x = layers.Conv2D(24, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(24, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.05)(x)

    x = layers.Conv2D(48, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(48, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.05)(x)

    x = layers.Conv2D(68, (3, 3), padding='same', activation='relu')(x)  # 72->68
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(68, (3, 3), padding='same', activation='relu')(x)  # 72->68
    x = layers.Dropout(0.05)(x)

    # 全局平均池化
    x = layers.GlobalAveragePooling2D()(x)

    # Slightly smaller dense layer
    x = layers.Dense(120, activation='relu')(x)  # 128->120
    x = layers.Dropout(0.15)(x)

    outputs = layers.Dense(10)(x)

    model = models.Model(inputs, outputs, name='competitive_model')
    return model

def cosine_decay_schedule(epoch, total_epochs=100):
    import math
    return 0.001 * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))

# 建立模型
model = create_simple_effective_model()
model.build(input_shape=(None, 32, 32, 3))

optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True, name='ce'),
        tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    ]
)

# 显示模型
model.summary()

# 计算总参数
total_params = model.count_params()
print(f"Total Parameters: {total_params:,}")
print(f"Parameter budget used: {total_params/122000*100:.1f}%")

callbacks = [
    tf.keras.callbacks.LearningRateScheduler(cosine_decay_schedule),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',  # 监控准确性
        patience=10,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

# 加载和预处理 CIFAR-10 数据集
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# 创建具有较大验证拆分的数据集
batch_size = 32  
train_dataset, val_dataset = create_dataset(train_images, train_labels, batch_size, is_training=True)
test_dataset = create_dataset(test_images, test_labels, batch_size, is_training=False)

# 使用更少的 epoch 训练模型
history = model.fit(
    train_dataset,
    epochs=100,  # epoch
    validation_data=val_dataset,
    callbacks=callbacks,
    verbose=1
)

# 在测试集上评估
test_results = model.evaluate(test_dataset, verbose=0)
print(f"\nTest Results:")
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test Cross-Entropy: {test_results[1]:.4f}")
print(f"Test Accuracy: {test_results[2]:.4f}")

# 绘制训练历史和详细的 CE 信息
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['ce'], label='Train CE', alpha=0.8)
plt.plot(history.history['val_ce'], label='Val CE', alpha=0.8)

# 查找并注释最低验证 CE
min_val_ce = min(history.history['val_ce'])
min_val_ce_epoch = history.history['val_ce'].index(min_val_ce)
plt.annotate(f'Lowest Val CE: {min_val_ce:.4f}\nEpoch: {min_val_ce_epoch + 1}',
             xy=(min_val_ce_epoch, min_val_ce),
             xytext=(min_val_ce_epoch + 5, min_val_ce + 0.1),
             arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
             bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
             fontsize=10)

#标记点
plt.plot(min_val_ce_epoch, min_val_ce, 'ro', markersize=8, alpha=0.8)

plt.title(f'Cross Entropy (Best Val CE: {min_val_ce:.4f})')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Train Acc', alpha=0.8)
plt.plot(history.history['val_accuracy'], label='Val Acc', alpha=0.8)

# 查找并注释最高的验证准确性
max_val_acc = max(history.history['val_accuracy'])
max_val_acc_epoch = history.history['val_accuracy'].index(max_val_acc)
plt.annotate(f'Best Val Acc: {max_val_acc:.4f}\nEpoch: {max_val_acc_epoch + 1}',
             xy=(max_val_acc_epoch, max_val_acc),
             xytext=(max_val_acc_epoch + 5, max_val_acc - 0.05),
             arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7),
             fontsize=10)

plt.plot(max_val_acc_epoch, max_val_acc, 'go', markersize=8, alpha=0.8)

plt.title(f'Accuracy (Best Val Acc: {max_val_acc:.4f})')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(history.history['loss'], label='Train Loss', alpha=0.8)
plt.plot(history.history['val_loss'], label='Val Loss', alpha=0.8)

# 查找并注释最低的验证损失
min_val_loss = min(history.history['val_loss'])
min_val_loss_epoch = history.history['val_loss'].index(min_val_loss)
plt.annotate(f'Lowest Val Loss: {min_val_loss:.4f}\nEpoch: {min_val_loss_epoch + 1}',
             xy=(min_val_loss_epoch, min_val_loss),
             xytext=(min_val_loss_epoch + 5, min_val_loss + 0.1),
             arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7),
             fontsize=10)

plt.plot(min_val_loss_epoch, min_val_loss, 'bo', markersize=8, alpha=0.8)

plt.title(f'Loss (Best Val Loss: {min_val_loss:.4f})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print
print(f"\n COMPETITION SUMMARY:")
print(f" Parameters: {total_params:,} / 122,000 ({total_params/122000*100:.1f}%)")
print(f" Best Validation CE: {min_val_ce:.4f} (Epoch {min_val_ce_epoch + 1})")
print(f" Best Validation Accuracy: {max_val_acc:.4f} (Epoch {max_val_acc_epoch + 1})")
print(f" Best Validation Loss: {min_val_loss:.4f} (Epoch {min_val_loss_epoch + 1})")
print(f" Final Test CE: {test_results[1]:.4f}")
print(f" Final Test Accuracy: {test_results[2]:.4f}")


Model: "competitive_model"

