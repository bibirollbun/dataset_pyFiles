import os, gc, cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
import albumentations as A

# CONFIG
TEST_PATH = '/kaggle/input/srifoton-25-machine-learning-competition/test/test/'
IMG_SIZE = (380, 380)
BATCH_SIZE = 12
NUM_CLASSES = 5
USE_MIXED_PRECISION = True
TTA_ROUNDS = 4

if USE_MIXED_PRECISION:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')


# Albumentations TTA pipelines 
tta_list = [
    A.Compose([A.HorizontalFlip(p=1.0), A.Normalize()]),
    A.Compose([A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=1.0), A.Normalize()]),
    A.Compose([A.Affine(translate_percent={"x":0.02,"y":0.02}, scale={"x":(0.98,1.02),"y":(0.98,1.02)}, rotate=(-3,3), p=1.0), A.Normalize()]),
    A.Compose([A.Normalize()])
]

def read_img(path):
    arr = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
    return img

def tf_load_test(path, aug):
    def _fn(p):
        p = p.decode('utf-8') if isinstance(p, bytes) else p
        img = read_img(p)
        img = aug(image=img)["image"].astype(np.float32) * 255.0
        return img
    img = tf.numpy_function(_fn, [path], tf.float32)
    img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    return img


# Model
def build_model():
    base = EfficientNetB4(input_shape=IMG_SIZE+(3,), include_top=False, weights=None)
    inp = layers.Input(shape=IMG_SIZE+(3,))
    x = preprocess_input(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
    return Model(inp, out)

# --- Load test files ---
test_files = sorted([f for f in os.listdir(TEST_PATH) if f.lower().endswith(('.jpg','.jpeg','.png'))])
test_paths = [os.path.join(TEST_PATH,f) for f in test_files]

# Load 2 models 
model_paths = [
    "/kaggle/input/model_capek/other/default/1/best_fold1.weights.h5",
    "/kaggle/input/model_capek/other/default/1/best_fold2.weights.h5"
]

models = []
for mp in model_paths:
    m = build_model()
    m.load_weights(mp)
    models.append(m)
    print(f"Loaded model from {mp}")


# Predict with ensemble 
all_preds = []
for model in models:
    preds_accum = None
    for t in range(TTA_ROUNDS):
        aug = tta_list[t % len(tta_list)]
        test_ds = (tf.data.Dataset.from_tensor_slices(test_paths)
                   .map(lambda p: tf_load_test(p, aug), num_parallel_calls=tf.data.AUTOTUNE)
                   .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
        pr = model.predict(test_ds, verbose=0)
        preds_accum = pr if preds_accum is None else preds_accum + pr
    preds_accum /= TTA_ROUNDS
    all_preds.append(preds_accum)

# average predictions from both models
final_preds = np.mean(all_preds, axis=0)
final_labels = np.argmax(final_preds, axis=1)


submission = pd.DataFrame({'Id': test_files, 'Predicted': final_labels})
submission.to_csv('submission_ensemble_2folds.csv', index=False)
print("Saved submission_ensemble_2folds.csv")




