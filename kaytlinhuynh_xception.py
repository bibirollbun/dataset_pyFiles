import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = info, 2 = warning, 3 = error only
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

print('✅')


import numpy as np
import pandas as pd
import os
import random
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, cohen_kappa_score
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import Sequence, to_categorical

print('✅')


# === SEED EVERYTHING ===
def seed_everything(seed=23):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

seed_everything(23)
print('✅ Seed set')


# === IMAGE PREPROCESSING ===
IMG_SIZE = 224

def crop_black(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        if img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0] == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            return np.stack([img1, img2, img3], axis=-1)

def circle_crop(img, sigmaX=10):
    height, width, _ = img.shape
    size = max(height, width)
    img = cv2.resize(img, (size, size))
    x, y = size // 2, size // 2
    r = min(x, y)
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (x, y), r, 1, thickness=-1)
    return cv2.bitwise_and(img, img, mask=mask)

def random_crop(img, size=(0.9, 1)):
    h, w, _ = img.shape
    cut = 1 - random.uniform(size[0], size[1])
    i = random.randint(0, int(cut * h))
    j = random.randint(0, int(cut * w))
    h_end = i + int((1 - cut) * h)
    w_end = j + int((1 - cut) * w)
    return img[i:h_end, j:w_end, :]

def preprocess_image(path, sigmaX=10, do_random_crop=False):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_black(img)
    if do_random_crop:
        img = random_crop(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    img = circle_crop(img, sigmaX=sigmaX)
    img = img.astype(np.float32) / 255.0
    return img
print('✅')


class EyeDataGenerator(Sequence):
    def __init__(self, df, image_dir, batch_size, image_size, num_classes=5, is_train=True, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.is_train = is_train
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_df = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size]
        images, labels = [], []
        for _, row in batch_df.iterrows():
            image = preprocess_image(os.path.join(self.image_dir, row['id_code'] + '.png'),
                                  image_size=self.image_size,
                                  do_random_crop=self.is_train)
            images.append(image)
            labels.append(row['diagnosis'])
        images = np.array(images)
        labels = to_categorical(labels, num_classes=self.num_classes)
        return images, labels

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)


print('✅')


# === MODEL INITIALIZATION ===
def build_model():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model = EfficientNetB3(include_top=False, weights='imagenet', input_tensor=inputs)

    x = GlobalAveragePooling2D()(base_model.output)

    # Block 1
    x1 = Dense(512)(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('gelu')(x1)
    x1 = Dropout(0.4)(x1)

    # Block 2
    x2 = Dense(256)(x1)
    x2 = BatchNormalization()(x2)
    x2 = Activation('gelu')(x2)
    x2 = Dropout(0.3)(x2)

    # Block 3
    x3 = Dense(128)(x2)
    x3 = BatchNormalization()(x3)
    x3 = Activation('gelu')(x3)

    # Residual Add (x1 + x3)
    x = Add()([x1, x3])

    outputs = Dense(5, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

print('✅')


# import data
train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
train_df.columns = ['id_code','diagnosis']

train_split = train_df.sample(frac = 0.8, random_state=42)
val_split = train_df.drop(train_split.index)

train_split['id_code'] = train_split['id_code'].astype(str) + ".png"
val_split['id_code'] = val_split['id_code'].astype(str) + ".png"

train_split['diagnosis'] = train_split['diagnosis'].astype(str)
val_split['diagnosis'] = val_split['diagnosis'].astype(str)

test_df  = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
print('✅')


# Plot class distribution
plt.figure(figsize=(10, 5))
sns.countplot(data=train_df, x='diagnosis')
plt.title("Class Distribution")
plt.show()


from tqdm import tqdm
image_dir = "../input/aptos2019-blindness-detection/train_images"
plt.figure(figsize=(20, 10))
for i in range(10):
    img = preprocess_image(os.path.join(image_dir, train_df['id_code'][i] + ".png"))
    plt.subplot(2, 5, i+1)
    plt.imshow(img)
    plt.title(f"Label: {train_df['diagnosis'][i]}")
    plt.axis('off')
plt.show()

### 7. Image Size Distribution
image_stats = []
for idx in tqdm(range(len(train_df))):
    path = os.path.join(image_dir, train_df['id_code'][idx] + ".png")
    img = cv2.imread(path)
    h, w, c = img.shape
    image_stats.append((h, w, c))

image_stats_df = pd.DataFrame(image_stats, columns=["height", "width", "channels"])
image_stats_df["ratio"] = image_stats_df["width"] / image_stats_df["height"]

plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.hist(image_stats_df['width'], bins=50)
plt.title("Width Distribution")
plt.subplot(1, 3, 2)
plt.hist(image_stats_df['height'], bins=50)
plt.title("Height Distribution")
plt.subplot(1, 3, 3)
plt.hist(image_stats_df['ratio'], bins=50)
plt.title("Aspect Ratio Distribution")
plt.show()


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=360,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()


from tensorflow.keras.applications import EfficientNetB6
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling2D,
    BatchNormalization, Activation, Concatenate
)

input_tensor = Input(shape=(224, 224, 3))
base_model = EfficientNetB6(weights='imagenet', include_top=False, input_tensor=input_tensor)

x = base_model.output
x = GlobalAveragePooling2D()(x)

# Block 1
x1 = Dense(512)(x)
x1 = BatchNormalization()(x1)
x1 = Activation('gelu')(x1)
x1 = Dropout(0.4)(x1)

# Block 2
x2 = Dense(256)(x1)
x2 = BatchNormalization()(x2)
x2 = Activation('gelu')(x2)
x2 = Dropout(0.3)(x2)

# Block 3
x3 = Dense(128)(x2)
x3 = BatchNormalization()(x3)
x3 = Activation('gelu')(x3)

# Concatenate instead of Add
x = Concatenate()([x1, x3])  # Shape now (512 + 128 = 640)

# Regularize: reduce to avoid overfitting
x = Dense(256)(x)
x = BatchNormalization()(x)
x = Activation('gelu')(x)
x = Dropout(0.3)(x)

output_tensor = Dense(5, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=output_tensor)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
    ModelCheckpoint('mobilenetv2_best_model.h5', monitor='val_loss', save_best_only=True)
]


train_df = train_split
val_df = val_split

train_df['diagnosis'] = train_df['diagnosis'].astype(str)
val_df['diagnosis'] = val_df['diagnosis'].astype(str)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory="../input/aptos2019-blindness-detection/train_images",
    x_col="id_code",
    y_col="diagnosis",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory="../input/aptos2019-blindness-detection/train_images",
    x_col="id_code",
    y_col="diagnosis",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)


history = model.fit(
    train_generator,
    validation_data = val_generator,
    epochs = 40,
    callbacks = callbacks
)


import os
import json

output_dir = '/kaggle/working'

# Save training history
history_path = os.path.join(output_dir, 'history.json')
with open(history_path, 'w') as f:
    json.dump(history.history, f)

# Save model weights
weights_path = os.path.join(output_dir, 'mobilenetv2.weights.h5')  # ✅ Must end with `.weights.h5`
model.save_weights(weights_path)

# Save entire model
model_path = os.path.join(output_dir, 'mobilenetv2_model.h5')
model.save(model_path)


val_generator.reset()
y_pred = model.predict(val_generator, verbose=1)
y_true = val_generator.classes
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


print(classification_report(y_true, y_pred_classes))


print('Kappa score: ', cohen_kappa_score(y_true, y_pred_classes, weights='quadratic'))


# Plot Loss & Accuracy
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()
plt.show()


y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
plt.figure(figsize=(10, 8))
for i in range(5):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
    auc_score = roc_auc_score(y_true_bin[:, i], y_pred[:, i])
    plt.plot(fpr, tpr, label=f"Class {i} AUC = {auc_score:.2f}")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve by Class")
plt.legend()
plt.grid(True)
plt.show()


from tensorflow.keras.preprocessing import image as keras_image
import tensorflow.keras.backend as K

img_path = f"../input/aptos2019-blindness-detection/train_images/{df_val['id_code'].iloc[0]}.png"
img = keras_image.load_img(img_path, target_size=(224, 224))
x = keras_image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0

preds = model.predict(x)
class_idx = np.argmax(preds[0])
class_output = model.output[:, class_idx]
last_conv_layer = model.get_layer("top_conv")
grads = K.gradients(class_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(pooled_grads_value.shape[0]):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

import cv2
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img

plt.imshow(cv2.cvtColor(superimposed_img.astype('uint8'), cv2.COLOR_BGR2RGB))
plt.title(f"Grad-CAM for Image: {df_val['id_code'].iloc[0]}")
plt.axis('off')
plt.show()

