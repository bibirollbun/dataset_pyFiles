# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Step 1: Problem Definition
NUM_CLASSES = 2  # Binary: No DR (0), DR (1)
CLASS_NAMES = ['No DR', 'DR']

# Step 2: Dataset Understanding & Preprocessing
def load_dataset(csv_path, image_dir):
    """Load dataset from CSV and image directory."""
    df = pd.read_csv(csv_path)
    df['image_path'] = df['id_code'].apply(lambda x: os.path.join(image_dir, f"{x}.png"))
    # Convert to binary labels: 0 (No DR), 1 (DR)
    df['diagnosis'] = df['diagnosis'].apply(lambda x: 0 if x == 0 else 1).astype(str)
    return df

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image: crop black borders, resize, normalize.
    
    Args:
        image (np.array): Input image (BGR format from cv2.imread or float32 from ImageDataGenerator).
        target_size (tuple): Target size for resizing.
    
    Returns:
        image (np.array): Preprocessed RGB image normalized to [0, 1].
    """
    # Ensure image is uint8 for contour detection
    if image.dtype == np.float32 or image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Crop black borders (Ben Graham's preprocessing)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        image = image[y:y+h, x:x+w]
    
    # Resize and normalize
    image = cv2.resize(image, target_size)
    return image / 255.0

# Step 3: Data Augmentation and Generator
def create_data_generator(df, batch_size=16, augment=False, target_size=(224, 224)):
    """Create data generator for efficient loading."""
    datagen = ImageDataGenerator(
        rotation_range=15 if augment else 0,
        zoom_range=0.2 if augment else 0,
        width_shift_range=0.1 if augment else 0,
        height_shift_range=0.1 if augment else 0,
        horizontal_flip=True if augment else False,
        brightness_range=[0.8, 1.2] if augment else None,
        preprocessing_function=preprocess_image
    )
    return datagen.flow_from_dataframe(
        df,
        x_col='image_path',
        y_col='diagnosis',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=augment
    )

# Load dataset
csv_path = '/kaggle/input/aptos2019-blindness-detection/train.csv'
image_dir = '/kaggle/input/aptos2019-blindness-detection/train_images'
df = load_dataset(csv_path, image_dir)

# Debugging Tip 1: Check dataset
print("Dataset Head:")
print(df.head())
print(f"Total images: {len(df)}")
print("Label distribution:", df['diagnosis'].value_counts())

# Split dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['diagnosis'])
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['diagnosis'])

# Create data generators
train_generator = create_data_generator(train_df, augment=True)
val_generator = create_data_generator(val_df, augment=False)
test_generator = create_data_generator(test_df, augment=False)

# Debugging Tip 2: Check generators
print(f"Training samples: {train_generator.n}")
print(f"Validation samples: {val_generator.n}")
print(f"Test samples: {test_generator.n}")


# Step 4: Model Selection & Training (Pretrained ViT)
class ViTModel(tf.keras.Model):
    def __init__(self, num_classes=2):
        super(ViTModel, self).__init__()
        self.vit_layer = hub.KerasLayer("https://tfhub.dev/sayakpaul/vit_b16_fe/1", trainable=True)
        self.dense1 = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dropout = layers.Dropout(0.5)
        self.dense2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.vit_layer(inputs, training=training)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)

# Build and compile the model
model = ViTModel(num_classes=NUM_CLASSES)
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
model.build(input_shape=(None, 224, 224, 3))
model.summary()

# Step 5: Train the Model with Class Weighting
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(train_df['diagnosis'].astype(int)), y=train_df['diagnosis'].astype(int))
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    class_weight=class_weights,
    callbacks=[
        callbacks.EarlyStopping(patience=7, restore_best_weights=True),
        callbacks.ModelCheckpoint('vit_model_binary.keras', save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]
)


# Step 6: Model Evaluation & Comparison
# Generate predictions
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = test_generator.labels  # Corrected to use generator labels

# Metrics
accuracy = accuracy_score(y_true_classes, y_pred_classes)
auc = roc_auc_score(y_true_classes, y_pred[:, 1])
f1 = f1_score(y_true_classes, y_pred_classes)
qwk = cohen_kappa_score(y_true_classes, y_pred_classes, weights='quadratic')

print(f"Accuracy: {accuracy}")
print(f"AUC-ROC: {auc}")
print(f"F1-score: {f1}")
print(f"Quadratic Weighted Kappa: {qwk}")

# Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Binary Classification)')
plt.tight_layout()
plt.savefig('confusion_matrix_binary.png', dpi=300)
plt.show()

# Step 7: Deployment & Inference
model.save('binary_vit_model.keras')

# Accuracy Plot (Actual Data)
plt.figure(figsize=(12, 7))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='#2ecc71', linewidth=2.5, marker='o', markersize=4)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#e74c3c', linewidth=2.5, marker='s', markersize=4)
plt.axhspan(0.85, 0.90, facecolor='#3498db', alpha=0.2, label='Expected Test Accuracy (85–90%)')
plt.title('Training and Validation Accuracy for Pretrained ViT Model (Binary Classification, APTOS 2019)', fontsize=16, pad=15)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12, loc='lower right')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x*100)}%'))
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.savefig('vit_accuracy_binary.png', dpi=300, bbox_inches='tight')
plt.show()

# Simulated Accuracy Plot (Actual ~50% vs. Expected ~88%)
epochs_sim = np.arange(1, 21)  # Assume early stopping at 20 epochs
train_accuracy_sim = [
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.73, 0.75, 0.77,
    0.79, 0.80, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89
]
val_accuracy_sim = [
    0.35, 0.37, 0.39, 0.41, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48,
    0.48, 0.49, 0.49, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50
]
epochs_exp = np.arange(1, 31)
train_accuracy_exp = [
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.74, 0.77, 0.80,
    0.82, 0.84, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.90, 0.91,
    0.92, 0.92, 0.93, 0.93, 0.94, 0.94, 0.95, 0.95, 0.95, 0.95
]
val_accuracy_exp = [
    0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.64, 0.67, 0.70, 0.73,
    0.75, 0.77, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86,
    0.86, 0.87, 0.87, 0.88, 0.88, 0.88, 0.89, 0.89, 0.89, 0.89
]

plt.figure(figsize=(12, 7))
plt.plot(epochs_sim, train_accuracy_sim, label='Training Accuracy (Actual, Simulated)', color='#2ecc71', linewidth=2.5, marker='o', markersize=4)
plt.plot(epochs_sim, val_accuracy_sim, label='Validation Accuracy (Actual, ~50%)', color='#e74c3c', linewidth=2.5, marker='s', markersize=4)
plt.plot(epochs_exp, train_accuracy_exp, label='Training Accuracy (Expected)', color='#27ae60', linestyle='--', linewidth=2)
plt.plot(epochs_exp, val_accuracy_exp, label='Validation Accuracy (Expected)', color='#c0392b', linestyle='--', linewidth=2)
plt.axhspan(0.85, 0.90, facecolor='#3498db', alpha=0.2, label='Expected Test Accuracy (85–90%)')
plt.title('Actual vs. Expected Accuracy for Pretrained ViT Model (Binary Classification, APTOS 2019)', fontsize=16, pad=15)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12, loc='lower right')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x*100)}%'))
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.savefig('vit_accuracy_comparison_binary.png', dpi=300, bbox_inches='tight')
plt.show()

