# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
from tqdm import tqdm

# Data paths (change if needed)
DATA_DIR = "../input/histopathologic-cancer-detection"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
LABELS_PATH = os.path.join(DATA_DIR, "train_labels.csv")

# Load labels
labels_df = pd.read_csv(LABELS_PATH)
print(f"Total images: {len(labels_df)}")
labels_df.head()



# Check class distribution
sns.countplot(x="label", data=labels_df)
plt.title("Label Distribution")
plt.show()

print(labels_df['label'].value_counts(normalize=True))



# Show random sample of images from each class
def plot_sample_images(df, img_dir, label, n=5):
    ids = df[df.label==label].sample(n, random_state=1)['id'].values
    plt.figure(figsize=(15,3))
    for i, img_id in enumerate(ids):
        img = Image.open(os.path.join(img_dir, img_id + ".tif"))
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.title(f"Label: {label}")
        plt.axis('off')
    plt.show()

plot_sample_images(labels_df, TRAIN_IMG_DIR, label=0)
plot_sample_images(labels_df, TRAIN_IMG_DIR, label=1)



# Sampling a balanced dataset
N_SAMPLES = 5000  # For each class

pos_df = labels_df[labels_df.label==1].sample(N_SAMPLES, random_state=42)
neg_df = labels_df[labels_df.label==0].sample(N_SAMPLES, random_state=42)
sample_df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=1).reset_index(drop=True)

print("Sample dataset shape:", sample_df.shape)



# Image loader (fast)
IMG_SIZE = 96

def load_images(df, img_dir, img_size=IMG_SIZE):
    X = []
    for img_id in tqdm(df['id']):
        img = Image.open(os.path.join(img_dir, img_id + ".tif")).resize((img_size, img_size))
        X.append(np.array(img))
    return np.array(X)

X = load_images(sample_df, TRAIN_IMG_DIR)
X = X.astype("float32") / 255.0   # <--- THIS IS CRUCIAL!
y = sample_df['label'].values
print("X shape:", X.shape)



# Train-validation split
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Train shape:", X_train.shape, "Val shape:", X_val.shape)



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_simple_cnn(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC", "accuracy"])
    return model

cnn = get_simple_cnn((IMG_SIZE, IMG_SIZE, 3))
cnn.summary()



# Data augmentation for training
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=20)



BATCH_SIZE = 32
EPOCHS = 10

# Use EarlyStopping for efficiency
callback = keras.callbacks.EarlyStopping(monitor="val_auc", patience=3, mode="max", restore_best_weights=True)

history = cnn.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    callbacks=[callback],
    class_weight={0:1, 1:1.5}  # simple positive class weight
)



plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title("Loss")
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['AUC'], label='train')      
plt.plot(history.history['val_AUC'], label='val')   
plt.title("AUC")
plt.legend()
plt.show()



# Validation performance
val_preds = cnn.predict(X_val)
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

auc = roc_auc_score(y_val, val_preds)
acc = accuracy_score(y_val, (val_preds > 0.5).astype(int))
print(f"Validation AUC: {auc:.4f} | Accuracy: {acc:.4f}")

cm = confusion_matrix(y_val, (val_preds > 0.5).astype(int))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Validation Confusion Matrix")
plt.show()



from tensorflow.keras.applications import MobileNetV2


def get_transfer_model(input_shape):
    base = MobileNetV2(
        weights="/kaggle/input/imagenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5",
        include_top=False,
        input_shape=input_shape
    )
    base.trainable = False  # freeze base
    model = keras.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC", "accuracy"])
    return model


transfer_model = get_transfer_model((IMG_SIZE, IMG_SIZE, 3))
history2 = transfer_model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    callbacks=[callback],
    class_weight={0:1, 1:1.5}
)



# Compare AUCs
val_preds2 = transfer_model.predict(X_val)
auc2 = roc_auc_score(y_val, val_preds2)
print(f"Transfer Model Validation AUC: {auc2:.4f}")

plt.plot(history2.history['val_AUC'], label="Transfer Model")
plt.plot(history.history['val_AUC'], label="Simple CNN")
plt.title("Validation AUC Comparison")
plt.legend()
plt.show()




import glob

# 1. Load test image filenames
test_path = '../input/histopathologic-cancer-detection/test/'
test_files = glob.glob(test_path + '*.tif')
test_ids = [os.path.basename(x)[:-4] for x in test_files]

# 2. Load and preprocess test images
def load_test_images(test_ids, test_path, img_size=IMG_SIZE):
    X_test = []
    for img_id in tqdm(test_ids):
        img = Image.open(os.path.join(test_path, img_id + '.tif')).resize((img_size, img_size))
        X_test.append(np.array(img))
    X_test = np.array(X_test).astype("float32") / 255.0
    return X_test

X_test = load_test_images(test_ids, test_path)

# 3. Predict probabilities (use transfer_model or your best model)
y_pred = transfer_model.predict(X_test, batch_size=32).flatten()

# 4. Create the submission DataFrame
submission = pd.DataFrame({'id': test_ids, 'label': y_pred})
submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")


