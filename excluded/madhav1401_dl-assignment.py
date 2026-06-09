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


# =========================================================
# COMPLETE EFFICIENTNETB0 TRAINING PIPELINE (ALL-IN-ONE)
# =========================================================

# 0. RESET WORKSPACE (Optional for Kaggle)
!rm -rf /kaggle/working/train
!rm -rf /kaggle/working/test
!mkdir -p /kaggle/working/train
!mkdir -p /kaggle/working/test

# 1. IMPORTS
import os, zipfile, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import confusion_matrix

print("TensorFlow Version:", tf.__version__)

# 2. UNZIP TRAIN & TEST ZIP FILES
print("Unzipping dataset...")
train_zip_path = "/kaggle/input/datasciencebowl/train.zip"
test_zip_path = "/kaggle/input/datasciencebowl/test.zip"

with zipfile.ZipFile(train_zip_path, 'r') as z:
    z.extractall("/kaggle/working/train")

with zipfile.ZipFile(test_zip_path, 'r') as z:
    z.extractall("/kaggle/working/test")

print("Extraction complete!")

# 3. PATHS & CONFIG
TRAIN_DIR = "/kaggle/working/train/train"
TEST_DIR = "/kaggle/working/test/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 121
EPOCHS = 10

# 4. DATA GENERATORS
print("\nLoading data generators...")

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=360,
    width_shift_range=0.10,
    height_shift_range=0.10,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.20
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    seed=42
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    seed=42
)

eval_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.20
).flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

classes = list(train_gen.class_indices.keys())
print("Classes Loaded:", len(classes))

# 5. BUILD EFFICIENTNETB0 MODEL
def build_model():
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224,224,3))
    base.trainable = False
    inputs = Input(shape=(224,224,3))
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model = build_model()
model.summary()

# 6. TRAIN MODEL
print("\nTraining EfficientNetB0...")
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, verbose=1)

# 7. PLOT TRAINING CURVES
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy Over Epochs")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss Over Epochs")
plt.legend()
plt.savefig("training_plots.png")
plt.show()

# 8. CONFUSION MATRIX
print("\nGenerating Confusion Matrix...")
pred = model.predict(eval_gen)
y_pred = np.argmax(pred, axis=1)
y_true = eval_gen.classes
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(18,18))
sns.heatmap(cm, cmap="viridis", xticklabels=False, yticklabels=False)
plt.title("Confusion Matrix (121 Classes)")
plt.savefig("confusion_matrix.png")
plt.show()

# 9. TEST PREDICTION + SUBMISSION
print("\nPreparing Submission...")
test_images = glob.glob(os.path.join(TEST_DIR, "*.jpg"))
submission_data = []

for img_path in test_images:
    img = load_img(img_path, target_size=IMG_SIZE)
    x = img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    preds = np.clip(model.predict(x, verbose=0)[0], 1e-15, 1-1e-15)
    submission_data.append([os.path.basename(img_path)] + list(preds))

submission_df = pd.DataFrame(submission_data, columns=["image"] + classes)
submission_df.to_csv("submission_final_efficientnet.csv", index=False)
print("\nðŸŽ‰ DONE! Submission saved as submission_final_efficientnet.csv")



# Assuming 'model' is already trained in memory

# Save only the weights
model.save_weights("efficientnetb0_weights.h5")
print("Weights saved as efficientnetb0_weights.h5")

# Save entire model
model.save("efficientnetb0_full_model.keras")
print("Full model saved as efficientnetb0_full_model.keras")



# =========================================================
# OPTIMIZED EFFICIENTNETB0 TRAINING PIPELINE (NO TESTING)
# =========================================================

# 0. RESET WORKSPACE (Optional for Kaggle)
!rm -rf /kaggle/working/train
!mkdir -p /kaggle/working/train

# 1. IMPORTS
import os, zipfile, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix

print("TensorFlow Version:", tf.__version__)

# 2. UNZIP TRAIN ZIP FILE ONLY
print("Unzipping training dataset...")
train_zip_path = "/kaggle/input/datasciencebowl/train.zip"

with zipfile.ZipFile(train_zip_path, 'r') as z:
    z.extractall("/kaggle/working/train")

print("Extraction complete!")

# 3. PATHS & CONFIG
TRAIN_DIR = "/kaggle/working/train/train"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 121
EPOCHS = 15   # increased for higher accuracy

# 4. DATA GENERATORS (more augmentation)
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.20
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    seed=42
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    seed=42
)

classes = list(train_gen.class_indices.keys())
print("Classes Loaded:", len(classes))

# 5. BUILD IMPROVED MODEL (with fine-tuning)
def build_model():
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224,224,3))

    # UNFREEZE LAST 20 LAYERS FOR FINE-TUNING
    for layer in base.layers[-20:]:
        layer.trainable = True

    inputs = Input(shape=(224,224,3))
    x = base(inputs)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_model()
model.summary()

# 6. CALLBACKS FOR HIGHER ACCURACY
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=2)
]

# 7. TRAIN MODEL
print("\nTraining Optimized EfficientNetB0...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# =========================================================
# SAVE MODEL WEIGHTS AFTER TRAINING
# =========================================================
model.save_weights("efficientnetb0_finetuned_weights.h5")
print("Saved weights as efficientnetb0_finetuned_weights.h5")

model.save("efficientnetb0_finetuned_full_model.keras")
print("Saved full model as efficientnetb0_finetuned_full_model.keras")

# 8. CONFUSION MATRIX (Optional but useful)
print("\nGenerating Confusion Matrix...")
eval_gen = ImageDataGenerator(preprocess_input, validation_split=0.20).flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical",
    subset="validation", shuffle=False
)

pred = model.predict(eval_gen)
y_pred = np.argmax(pred, axis=1)
y_true = eval_gen.classes
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(18,18))
sns.heatmap(cm, cmap="viridis", xticklabels=False, yticklabels=False)
plt.title("Confusion Matrix (121 Classes)")
plt.savefig("confusion_matrix_finetuned.png")
plt.show()

print("\nðŸŽ‰ Training Complete! Weights saved successfully.")



# =========================================================
# SAVE TRAINED WEIGHTS + FULL MODEL (Corrected)
# =========================================================

# Save only weights (must end with .weights.h5)
model.save_weights("efficientnetb0_finetuned.weights.h5")
print("Saved weights as efficientnetb0_finetuned.weights.h5")

# Save full model (architecture + weights + optimizer)
model.save("efficientnetb0_finetuned_model.keras")
print("Saved full model as efficientnetb0_finetuned_model.keras")



# =========================================================
# HYBRID ENSEMBLE: BASE CNN (128x128) + EfficientNetB0 (224x224)
# =========================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

print("TensorFlow:", tf.__version__)

# ---------------------------------------------------------
# 1) PATHS  (change only if your .keras files are elsewhere)
# ---------------------------------------------------------
TRAIN_DIR       = "/kaggle/working/train/train"

BASE_MODEL_PATH = "/kaggle/input/base-cnn-keras/base_cnn_model.keras"
EFF_MODEL_PATH  = "/kaggle/working/efficientnetb0_finetuned_model.keras"

# for single-image demo
# TEST_IMAGE_PATH = "/kaggle/input/zxcvbnm/22613.jpg"   # change if needed

IMG_SIZE_BASE = (128, 128)
IMG_SIZE_EFF  = (224, 224)
BATCH_BASE = 32
BATCH_EFF  = 16
SEED = 42

# ---------------------------------------------------------
# 2) VALIDATION GENERATORS (NO SHUFFLE)
#    one for Base CNN, one for EfficientNetB0
# ---------------------------------------------------------
print("\nPreparing validation generators...")

eval_gen_base = ImageDataGenerator(
    rescale=1./255.0,
    validation_split=0.20
).flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE_BASE,
    batch_size=BATCH_BASE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=SEED
)

eval_gen_eff = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.20
).flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE_EFF,
    batch_size=BATCH_EFF,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=SEED
)

# Sanity check â€“ labels must match
assert np.array_equal(eval_gen_base.classes, eval_gen_eff.classes), "Label orders do not match!"
y_true = eval_gen_base.classes
class_names = list(eval_gen_base.class_indices.keys())
print("Validation samples:", len(y_true))
print("Number of classes:", len(class_names))

# ---------------------------------------------------------
# 3) LOAD TRAINED MODELS
# ---------------------------------------------------------
print("\nLoading saved models...")
base_cnn = tf.keras.models.load_model(BASE_MODEL_PATH)
effnet   = tf.keras.models.load_model(EFF_MODEL_PATH)
print("Models loaded âœ…")

# ---------------------------------------------------------
# 4) PREDICT ON VALIDATION SET
# ---------------------------------------------------------
print("\nPredicting with Base CNN...")
p_base = base_cnn.predict(eval_gen_base, verbose=1)

print("Predicting with EfficientNetB0...")
p_eff  = effnet.predict(eval_gen_eff, verbose=1)

y_pred_base = np.argmax(p_base, axis=1)
y_pred_eff  = np.argmax(p_eff,  axis=1)

acc_base = accuracy_score(y_true, y_pred_base)
acc_eff  = accuracy_score(y_true, y_pred_eff)

print(f"\nBase CNN   Val Accuracy : {acc_base:.4f}")
print(f"EfficientNet Val Accuracy : {acc_eff:.4f}")

# ---------------------------------------------------------
# 5) HYBRID â€“ SEARCH BEST WEIGHT
# ---------------------------------------------------------
print("\nSearching best hybrid weight...")

best_acc = 0.0
best_w   = 0.0

for w in np.arange(0, 1.05, 0.05):
    hybrid_probs = w * p_base + (1 - w) * p_eff
    hybrid_preds = np.argmax(hybrid_probs, axis=1)
    acc = accuracy_score(y_true, hybrid_preds)
    print(f"w_base={w:.2f}, w_eff={1-w:.2f} -> acc={acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_w = w

print("\n========================================")
print(f"Best weight: Base={best_w:.2f}, Eff={1-best_w:.2f}")
print(f"Hybrid Validation Accuracy: {best_acc:.4f}")
print("========================================")

# ---------------------------------------------------------
# 6) CONFUSION MATRIX FOR HYBRID (OPTIONAL)
# ---------------------------------------------------------
hybrid_probs = best_w * p_base + (1 - best_w) * p_eff
hybrid_preds = np.argmax(hybrid_probs, axis=1)
cm = confusion_matrix(y_true, hybrid_preds)

plt.figure(figsize=(20,20))
sns.heatmap(cm, cmap="viridis", xticklabels=False, yticklabels=False)
plt.title("Hybrid Ensemble Confusion Matrix (Validation)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

