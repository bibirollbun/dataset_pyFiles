import os
import random
import warnings
import json
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ===================== Configuration =====================
IMG_SIZE = 512  
size = (IMG_SIZE, IMG_SIZE)
NUM_CLASSES = 5
BATCH_SIZE = 16  
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 15

# ===================== Setup =====================
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Enable mixed precision for faster training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Set seed for reproducibility
def seed_everything(seed=21):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed_everything()
warnings.filterwarnings('ignore')


# ===================== Load Dataset =====================
work_dir = '../input/cassava-leaf-disease-classification/'
train_path = os.path.join(work_dir, 'train_images')
data = pd.read_csv(os.path.join(work_dir, 'train.csv'))

with open(os.path.join(work_dir, 'label_num_to_disease_map.json')) as f:
    real_labels = json.load(f)
    real_labels = {int(k): v for k, v in real_labels.items()}

# Map label numbers to disease names
data['class_name'] = data['label'].map(real_labels)
train_df, test_df = train_test_split(data, test_size=0.1, random_state=42, stratify=data['class_name'])


# ===================== Visualize Class Distribution =====================
label_map = {
    0: 'CBB',
    1: 'CBSD',
    2: 'CGM',
    3: 'CMD',
    4: 'Healthy'
}
data['short_label'] = data['label'].map(label_map)

plt.figure(figsize=(8, 5))
ax = sns.countplot(x='short_label', data=data, palette='viridis', edgecolor='black')
plt.title("Class Distribution in Original Dataset")
plt.xlabel("Class Label")
plt.ylabel("Sample Count")
for p in ax.patches:
    height = int(p.get_height())
    ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center', fontsize=11, xytext=(0, 6), textcoords='offset points')
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()


# ===================== Sample Images per Class =====================
import matplotlib.image as mpimg
sample_dir = os.path.join(work_dir, 'train_images')
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
classes = sorted(data['label'].unique())

for i, label in enumerate(classes):
    img_name = data[data['label'] == label].iloc[0]['image_id']
    img_path = os.path.join(sample_dir, img_name)
    img = mpimg.imread(img_path)
    row, col = divmod(i, 3)
    axes[row][col].imshow(img)
    axes[row][col].set_title(f"{label_map[label]}", fontsize=14)
    axes[row][col].axis('off')
for j in range(len(classes), 6):
    row, col = divmod(j, 3)
    axes[row][col].axis('off')
plt.suptitle("Example Image from Each Class", fontsize=16)
plt.tight_layout()
plt.savefig("class_samples.png")
plt.show()


# ===================== Data Generators =====================
# Use ImageDataGenerator for augmentation; reduce augmentation intensity if needed
datagen_train = ImageDataGenerator(
    validation_split=0.2,
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

train_generator = datagen_train.flow_from_dataframe(
    train_df,
    directory=train_path,
    x_col='image_id',
    y_col='class_name',
    subset='training',
    target_size=size,
    class_mode='categorical',
    shuffle=True,
    seed=42,
    batch_size=BATCH_SIZE
)

validation_datagen = ImageDataGenerator(
    validation_split=0.2,
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

validation_generator = validation_datagen.flow_from_dataframe(
    train_df,
    directory=train_path,
    x_col='image_id',
    y_col='class_name',
    subset='validation',
    target_size=size,
    class_mode='categorical',
    shuffle=True,
    seed=42,
    batch_size=BATCH_SIZE
)

test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    directory=train_path,
    x_col='image_id',
    y_col='class_name',
    target_size=size,
    class_mode='categorical',
    shuffle=False,
    seed=42,
    batch_size=BATCH_SIZE
)


# ===================== Data Augmentation Visualization =====================
sample_aug = train_df.iloc[[0]].copy()
sample_aug['class_name'] = sample_aug['label'].map(real_labels)
preview_gen = datagen_train.flow_from_dataframe(
    sample_aug,
    directory=train_path,
    x_col='image_id',
    y_col='class_name',
    target_size=size,
    class_mode='categorical',
    batch_size=1
)
aug_imgs = [preview_gen[0][0][0] / 255.0 for _ in range(4)]
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
for i in range(4):
    axs.flat[i].imshow(aug_imgs[i])
    axs.flat[i].axis('off')
plt.suptitle("Data Augmentation Examples", fontsize=16)
plt.tight_layout()
plt.savefig("augmentation_samples.png")
plt.show()


# ===================== Compute Class Weights =====================
# Compute class weights to counter class imbalance
class_counts = train_df['label'].value_counts().sort_index()
total_samples = len(train_df)
class_weights = {i: total_samples / (NUM_CLASSES * class_counts[i]) for i in range(NUM_CLASSES)}
print("Class Weights:", class_weights)


# ===================== Model Definition =====================
def create_model():
    model = Sequential([
        EfficientNetB3(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        ),
        GlobalAveragePooling2D(),
        Flatten(),
        Dense(256, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),  # Using L2 regularization
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model


# ===================== Two-Stage Training =====================
# Stage 1: Train the new classifier head with the base model frozen
model_stage1 = create_model()
# Freeze base model layers
model_stage1.layers[0].trainable = False

model_stage1.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks_stage1 = [
    EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint('Cassava_best_model_stage1.keras', save_best_only=True, monitor='val_accuracy', mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)
]

history_stage1 = model_stage1.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS_STAGE1,
    callbacks=callbacks_stage1,
    class_weight=class_weights
)


# Stage 2: Fine-tune by unfreezing part of the base model
# Unfreeze last 60 layers of the base model as an example
for layer in model_stage1.layers[0].layers[-60:]:
    layer.trainable = True

model_stage1.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks_stage2 = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint('Cassava_best_model_stage2.keras', save_best_only=True, monitor='val_accuracy', mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)
]

history_stage2 = model_stage1.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS_STAGE2,
    callbacks=callbacks_stage2,
    class_weight=class_weights
)

# Save final fine-tuned model
model_stage1.save('Cassava_model_finetuned.keras')

# ===================== Model Architecture Visualization =====================
plot_model(model_stage1, to_file="model_architecture.png", show_shapes=True, show_layer_names=True)


# ===================== Evaluation =====================

def plot_training_history(history):
    # Plot Accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


plot_training_history(history_stage2)

test_loss, test_acc = model_stage1.evaluate(test_generator, verbose=1)
print(f"\nFinal model evaluation - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

# Predict on test set and generate confusion matrix and classification report
y_true = test_df['label'].values
pred_probs = model_stage1.predict(test_generator, verbose=1)
y_pred = np.argmax(pred_probs, axis=1)

conf_mat = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("\nClassification Report:")
target_names = [real_labels[i] for i in range(NUM_CLASSES)]
print(classification_report(y_true, y_pred, target_names=target_names))

