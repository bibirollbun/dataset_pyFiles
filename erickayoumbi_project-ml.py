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


# Reproducibility: set random seeds
import random
import numpy as np
import tensorflow as tf

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



import pandas as pd

# Charger train.csv( lire, afficher, analyser, nettoyer les données )
df = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")
df.head()
df["id_code"] = df["id_code"] + ".png"
df.head()



df['binary_diag'] = df['diagnosis'].apply(lambda x: 0 if x == 0 else 1) ## gemini


# Analysis of the distribution
df["diagnosis"].value_counts().sort_index()



# Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
df["diagnosis"].value_counts().sort_index().plot(kind="bar", color="teal")
plt.title("Distribution of DR Severity Classes")
plt.xlabel("DR Class (0 = No DR, 4 = Proliferative)")
plt.ylabel("Number of Images")
plt.show()



# Examples Images from each DR

import cv2
import matplotlib.pyplot as plt

image_path = "/kaggle/input/aptos2019-blindness-detection/train_images/"

def show_image(filename):
    img = cv2.imread(image_path + filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")

# Afficher un exemple pour chaque classe 0 → 4
for diag in range(5):
    example = df[df["diagnosis"] == diag].iloc[0]["id_code"]
    print(f"Example for class {diag}: {example}")
    show_image(example)
    plt.show()



# Import the function used to split the dataset into train/val/test
from sklearn.model_selection import train_test_split

# Split the dataset into:
# 70% training data
# 30% temporary data (which we will later split into validation + test)
# Stratification ensures each subset keeps the same class distribution
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,        # 30% of the dataset goes to temp_df
    random_state=42,       # for reproducibility
    stratify=df["diagnosis"]   # preserve class distribution
)

train_df.head()

# Split the temporary set into:
# 1/3 validation (10% of total)
# 2/3 test (20% of total)
val_df, test_df = train_test_split(
    temp_df,
    test_size=2/3,                     # 2/3 of 30% = 20% of total
    random_state=42,
    stratify=temp_df["diagnosis"]      # preserve class distribution
)

val_df.head()

# Print the number of samples in each split
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))



# Check class balance in each subset
print("Train class distribution:")
print(train_df["diagnosis"].value_counts().sort_index())

print("\nValidation class distribution:")
print(val_df["diagnosis"].value_counts().sort_index())

print("\nTest class distribution:")
print(test_df["diagnosis"].value_counts().sort_index())



# Create binary label column: 0 = No DR, 1 = DR
train_df["binary_label"] = train_df["diagnosis"].apply(lambda x: 0 if x == 0 else 1)
val_df["binary_label"] = val_df["diagnosis"].apply(lambda x: 0 if x == 0 else 1)
test_df["binary_label"] = test_df["diagnosis"].apply(lambda x: 0 if x == 0 else 1)

# Convert labels to string as required by Keras generators
train_df["binary_label"] = train_df["binary_label"].astype(str)
val_df["binary_label"] = val_df["binary_label"].astype(str)
test_df["binary_label"] = test_df["binary_label"].astype(str)

# Là on vient de créér le label binaire (DR vs No DR) et convertir en string pour keras
# On va maintenant créer les générateurs qui permettent de charger les images les redimensionner les normaliser diviser en batch 


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

# 1. Pour l'entraînement (Augmentation + Preprocess)
IMG_SIZE = 224
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3]
)

# 2. Pour Validation et Test (UNIQUEMENT Preprocess, pas de rescale !)
val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# --- GÉNÉRATEURS ---

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=image_path,
    x_col="id_code",
    y_col="binary_label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=16,
    class_mode="binary",
    shuffle=True
)

val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=image_path,
    x_col="id_code",
    y_col="binary_label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=16,
    class_mode="binary",
    shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=image_path,
    x_col="id_code",
    y_col="binary_label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=16,
    class_mode="binary",
    shuffle=False
)


# Construction of EfficientNetB3 : C'est un réseau déjà entraîné sur ImageNet
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow as tf

# 1. Charger EfficientNetB3
base_model = EfficientNetB3(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# 2. Geler TOUTES les couches d'un coup (Remplace avantageusement ton 'for')
base_model.trainable = False 

# 3. Connecter les couches
x = base_model.output
x = GlobalAveragePooling2D()(x)
outputs = Dense(1, activation="sigmoid")(x)

# 4. Créer le modèle
model = Model(inputs=base_model.input, outputs=outputs)

# 5. Compiler
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()



print(f"Nombre de couches entraînables : {len(model.trainable_weights)}")


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Callbacks to prevent overfitting and stabilize training
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=2,
        min_lr=1e-7
    )
]


# Training of the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=callbacks
)



# Unfreeze the last layers of EfficientNetB3 for fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True
    
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5), # LR = 1e-5 car autrement on détruit les poids préentrainés
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Retrain (fine-tuning)
history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=callbacks
) 



# Evaluate the model on the test set (binary classification)
test_loss, test_acc = model.evaluate(test_generator)
print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)



import matplotlib.pyplot as plt

# Accuracy curves
plt.figure(figsize=(8,4))
plt.plot(history.history["accuracy"], label="Train Accuracy (head)")
plt.plot(history.history["val_accuracy"], label="Val Accuracy (head)")
plt.plot(history_finetune.history["accuracy"], label="Train Accuracy (finetune)")
plt.plot(history_finetune.history["val_accuracy"], label="Val Accuracy (finetune)")
plt.title("Accuracy Curves")
plt.legend()
plt.show()

# Loss curves
plt.figure(figsize=(8,4))
plt.plot(history.history["loss"], label="Train Loss (head)")
plt.plot(history.history["val_loss"], label="Val Loss (head)")
plt.plot(history_finetune.history["loss"], label="Train Loss (finetune)")
plt.plot(history_finetune.history["val_loss"], label="Val Loss (finetune)")
plt.title("Loss Curves")
plt.legend()
plt.show()



# --- Step 1: Make predictions on the test set ---
y_prob = model.predict(test_generator)            # probabilities from model
y_pred = (y_prob > 0.5).astype(int).ravel()       # convert prob → class 0/1

# True labels
y_true = test_generator.classes


# --- Step 2: Compute the metrics ---
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1-score :", f1)


# --- Step 3: Confusion matrix ---
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Binary DR Detection")
plt.show()



# Preparation des labels
# Convert labels to integers
train_df["diag_int"] = train_df["diagnosis"].astype(int)
val_df["diag_int"]  = val_df["diagnosis"].astype(int)
test_df["diag_int"] = test_df["diagnosis"].astype(int)

print(train_df.head())



# Data generators
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = 224
image_path = "/kaggle/input/aptos2019-blindness-detection/train_images/"

# Train generator with augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3]
)

# Validation & Test generators — no augmentation
val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=image_path,
    x_col="id_code",
    y_col="diag_int",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=16,
    class_mode="raw",
    shuffle=True
)

val_generator = val_test_datagen.flow_from_dataframe(
    val_df,
    directory=image_path,
    x_col="id_code",
    y_col="diag_int",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=16,
    class_mode="raw",
    shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    test_df,
    directory=image_path,
    x_col="id_code",
    y_col="diag_int",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=16,
    class_mode="raw",
    shuffle=False
)



# Calcul des class weight
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

classes = np.array([0,1,2,3,4])
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["diag_int"]
)

class_weights = dict(zip(classes, weights))
print("Class weights:", class_weights)



# Construction du modèle EfficientNetB3
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load EfficientNetB3 (backbone only)
base_model = EfficientNetB3(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze all layers
base_model.trainable = False

# Build classification head
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
outputs = Dense(5, activation="softmax")(x)

model_5c = Model(inputs, outputs)

# Compile
model_5c.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model_5c.summary()



# Phase 1 : Training du head
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

history_head = model_5c.fit(
    train_generator,
    validation_data=val_generator,
    epochs=7,
    class_weight=class_weights,
    callbacks=[lr_scheduler]
)



# Phase 2 : Fine-tuning
# Unfreeze last 20 layers
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Recompile with very small LR
model_5c.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model_5c.fit(
    train_generator,
    validation_data=val_generator,
    epochs=7,
    class_weight=class_weights,
    callbacks=[lr_scheduler]
)



# Evaluation : predictions
y_prob = model_5c.predict(test_generator)
y_pred = y_prob.argmax(axis=1)
y_true = test_df["diag_int"].values



# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))



# Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - 5-Class DR Classification")
plt.show()



# QWK
from sklearn.metrics import cohen_kappa_score

qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
print("QWK:", qwk)



test_df.dtypes



for layer in model_5c.get_layer("efficientnetb3").layers[-20:]:
    print(layer.name)



test_df.head()



for layer in model_5c.get_layer("efficientnetb3").layers[::-1]:
    if isinstance(layer, tf.keras.layers.Conv2D):
        print(layer.name)
        break
# verification de la derniere couche


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

def occlusion_sensitivity(model, img_array, patch_size=20, stride=10):
    h, w, _ = img_array.shape
    heatmap = np.zeros((h, w))

    img_preprocessed = preprocess_input(img_array.copy())
    base_pred = model.predict(np.expand_dims(img_preprocessed, axis=0))[0]
    predicted_class = np.argmax(base_pred)

    for y in range(0, h - patch_size, stride):
        for x in range(0, w - patch_size, stride):

            occluded = img_array.copy()
            occluded[y:y+patch_size, x:x+patch_size, :] = 0  # Masque noir

            occluded_pre = preprocess_input(occluded.copy())
            pred = model.predict(np.expand_dims(occluded_pre, axis=0))[0]

            heatmap[y:y+patch_size, x:x+patch_size] = base_pred[predicted_class] - pred[predicted_class]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap



def display_occlusion(img_path, model, patch_size=20, stride=10):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
    img_arr = tf.keras.preprocessing.image.img_to_array(img)

    heatmap = occlusion_sensitivity(model, img_arr, patch_size, stride)

    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title("Occlusion Sensitivity Map")
    plt.axis("off")
    plt.show()



image_folder = "/kaggle/input/aptos2019-blindness-detection/train_images/"

for c in ["0","1","2","3","4"]:
    ex = test_df[test_df["diagnosis"].astype(str) == c]

    if not ex.empty:
        img_id = ex.iloc[0]["id_code"].replace(".png","")
        img_path = image_folder + img_id + ".png"

        print(f"\n=== Occlusion Map – Classe {c} ===")
        display_occlusion(img_path, model_5c)


