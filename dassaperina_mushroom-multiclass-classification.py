import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score


IMG_SIZE = (299, 299) 
BATCH_SIZE = 32
EPOCHS_STAGE_1 = 30  
EPOCHS_STAGE_2 = 20  


TRAIN_CSV = "/kaggle/input/c/mushroom-multiclass-classification/train.csv"
TEST_CSV = "/kaggle/input/c/mushroom-multiclass-classification/test.csv"
IMG_DIR = "/kaggle/input/c/mushroom-multiclass-classification/dataset/dataset"


train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print(train_df.head())


def load_images_and_labels(df, img_dir, img_size=(299, 299)):
    images = []
    labels = []
    for idx, row in df.iterrows():
        img_id = int(row["Image"])
        label  = row["Mushroom"]
        img_path = os.path.join(img_dir, f"{img_id:05d}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            print("Ошибка чтения:", img_path)
            continue
        img = cv2.resize(img, img_size)

        img = preprocess_input(img.astype(np.float32))
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)


def load_images_only(df, img_dir, img_size=(299, 299)):
    # Для inference
    images = []
    ids = []
    for idx, row in df.iterrows():
        img_id = int(row["Image"])
        img_path = os.path.join(img_dir, f"{img_id:05d}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            images.append(np.zeros((img_size[0], img_size[1], 3), dtype=np.float32))
            ids.append(img_id)
            continue
        img = cv2.resize(img, img_size)
        img = preprocess_input(img.astype(np.float32))
        images.append(img)
        ids.append(img_id)
    return np.array(images), np.array(ids)


X_all, y_all = load_images_and_labels(train_df, IMG_DIR, IMG_SIZE)


X_train, X_val, y_train_org, y_val_org = train_test_split(
    X_all, 
    y_all, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_all
)

print("X_train:", X_train.shape, "X_val:", X_val.shape)


# class_weight

classes = np.unique(y_train_org)
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train_org
)
class_weights = dict(enumerate(class_weights_array))

num_classes = 10
y_train = to_categorical(y_train_org, num_classes=num_classes)
y_val   = to_categorical(y_val_org,   num_classes=num_classes)



from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.2,
    fill_mode='nearest'
)


val_datagen = ImageDataGenerator() 

train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
val_generator   = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)


# собир модель

base_model = Xception(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), pooling='avg')
base_model.trainable = False  # Сначала замораживаем

inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(
    optimizer=Adam(learning_rate=1e-3),  #
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# callbacks

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,  #2
    min_lr=1e-7,
    verbose=1
)


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=7,  #5
    restore_best_weights=True,
    verbose=1
)


checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)


callbacks_list = [reduce_lr, early_stopping, checkpoint]



# 1 часть обучения (только голова)

history_1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_STAGE_1,
    class_weight=class_weights,
    callbacks=callbacks_list
)


# unfreezing Xception

base_model.trainable = True

fine_tune_at = 30  
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-4),  # 5e-5
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_STAGE_2,
    class_weight=class_weights,
    callbacks=callbacks_list
)


# F1 на валидации

val_preds = model.predict(val_generator)
val_pred_classes = np.argmax(val_preds, axis=1)
y_val_true = np.argmax(y_val, axis=1)

f1_val = f1_score(y_val_true, val_pred_classes, average="macro")
print("F1 (Validation) =", f1_val)

# Если F1 < 0.84


#test
X_test, _ = load_images_only(test_df, IMG_DIR, IMG_SIZE)


predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)


submission = pd.DataFrame()
submission["Id"] = range(len(test_df))  
submission["Predicted"] = predicted_labels
submission.to_csv('submission_1.csv', index=False)
print("Submission saved!")

