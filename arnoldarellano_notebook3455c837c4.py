# ====================== 1. LIBRERÍAS Y RUTAS ======================
import os, numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

BASE = "/kaggle/input/competencia-02-julio-2025"
DATA = os.path.join(BASE, "archive/pizza_steak")
TRAIN = os.path.join(DATA, "train")
TEST  = os.path.join(DATA, "test")
SAMPLE_CSV = os.path.join(BASE, "sample_submission.csv")

IMG  = 224
BATCH= 32
SEED = 42

# ====================== 2. DATA GENERATORS ======================
datagen = ImageDataGenerator(
    rescale=1/255.,
    validation_split=0.15,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    TRAIN, target_size=(IMG,IMG), batch_size=BATCH,
    class_mode="binary", subset="training", seed=SEED)

val_gen = datagen.flow_from_directory(
    TRAIN, target_size=(IMG,IMG), batch_size=BATCH,
    class_mode="binary", subset="validation", seed=SEED)

# ====================== 3. MODELO (FEATURE-EXTRACTION) ======================
base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG,IMG,3))
base.trainable = False                                   # congelado

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.25)(x)
out = Dense(1, activation="sigmoid")(x)

model = Model(base.input, out)
model.compile(Adam(1e-3), "binary_crossentropy", metrics=["accuracy"])

# ====================== 4. ENTRENAMIENTO RÁPIDO ======================
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=2, verbose=1)
]

model.fit(train_gen, epochs=5, validation_data=val_gen, callbacks=callbacks, verbose=2)

# ====================== 5. FINE-TUNING (20 capas) ======================
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(Adam(1e-5), "binary_crossentropy", metrics=["accuracy"])
model.fit(train_gen, epochs=3, validation_data=val_gen, callbacks=callbacks, verbose=2)

# ====================== 6. PREDICCIONES EXACTAS (200 IMÁGENES) ======================
sample = pd.read_csv(SAMPLE_CSV)
ids = sample["ID"].tolist()

def path_from_id(img_id):
    p_path = os.path.join(TEST, "pizza", img_id)
    return p_path if os.path.exists(p_path) else os.path.join(TEST, "steak", img_id)

paths = [path_from_id(i) for i in ids]

def load_preprocess(p):
    arr = img_to_array(load_img(p, target_size=(IMG,IMG)))/255.
    return arr

X = np.stack([load_preprocess(p) for p in paths])
pred = model.predict(X, batch_size=64).ravel()
labels = np.where(pred>0.5, "steak", "pizza")

# ====================== 7. CREAR submission.csv ======================
sub = pd.DataFrame({"ID": ids, "label": labels})
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("✅ Generado /kaggle/working/submission.csv con", len(sub), "filas")


