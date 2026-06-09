# 1. Ortam Kurulumu
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import zipfile
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.regularizers import l2

print("TensorFlow version:", tf.__version__)


# 2. Veri YÃ¼kleme ve Train-Validation-Test Split
ort_veri_yolu = "/kaggle/input/dogs-vs-cats"
with zipfile.ZipFile(os.path.join(ort_veri_yolu, "train.zip"), 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working/")
rec_egitim_yolu = "/kaggle/working/train"

def veri_yukle(veri_yolu, boyut=(128,128)):
    goruntuler, etiketler = [], []
    kopek_yollari = [os.path.join(veri_yolu,f) for f in os.listdir(veri_yolu) if f.startswith('dog')]
    for yol in tqdm(kopek_yollari, desc="KÃ¶pekler"):
        img = cv2.imread(yol)
        if img is not None:
            img = cv2.resize(img, boyut)
            img = preprocess_input(img)
            goruntuler.append(img)
            etiketler.append(1)
    kedi_yollari = [os.path.join(veri_yolu,f) for f in os.listdir(veri_yolu) if f.startswith('cat')]
    for yol in tqdm(kedi_yollari, desc="Kediler"):
        img = cv2.imread(yol)
        if img is not None:
            img = cv2.resize(img, boyut)
            img = preprocess_input(img)
            goruntuler.append(img)
            etiketler.append(0)
    return np.array(goruntuler), np.array(etiketler)

X, y = veri_yukle(rec_egitim_yolu)
y_cat = to_categorical(y, 2)

# SÄ±nÄ±f daÄŸÄ±lÄ±mÄ± gÃ¶rselleÅŸtirme
unique, counts = np.unique(y, return_counts=True)
plt.figure(figsize=(6,4))
sns.barplot(x=['Kedi','KÃ¶pek'], y=counts)
plt.title("SÄ±nÄ±f DaÄŸÄ±lÄ±mÄ±")
plt.ylabel("GÃ¶rÃ¼ntÃ¼ SayÄ±sÄ±")
plt.show()

# Ã–rnek gÃ¶rseller
plt.figure(figsize=(12,6))
for i, label in enumerate([0,1]):  # 0=Kedi, 1=KÃ¶pek
    idx = np.where(y==label)[0]
    sample_idx = random.choice(idx)
    img = ((X[sample_idx]+1)*127.5).astype('uint8')
    plt.subplot(1,2,i+1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Kedi' if label==0 else 'KÃ¶pek')
plt.show()

# Train-Test-Validation split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=np.argmax(y_train, axis=1), random_state=42
)
print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")


# 3. Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    zoom_range=0.3,
    brightness_range=[0.7,1.3]
)


# 4. Random Search Hiperparametre Denemeleri
param_combinations = [
    {'learning_rate':0.001,'dropout_rate':0.5,'batch_size':32},
    {'learning_rate':0.0005,'dropout_rate':0.5,'batch_size':32},
    {'learning_rate':0.0001,'dropout_rate':0.4,'batch_size':64},
]

best_score = 0
best_params = None

for i, params in enumerate(param_combinations):
    print(f"\nTest Kombinasyonu {i+1}/{len(param_combinations)}: {params}")
    
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128,128,3))
    for layer in base_model.layers:
        layer.trainable = False
    
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(params['dropout_rate'])(x)
    outputs = Dense(2, activation='softmax')(x)
    
    model = Model(base_model.input, outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=params['batch_size']),
        steps_per_epoch=len(X_train)//params['batch_size'],
        epochs=5,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    val_acc = max(history.history['val_accuracy'])
    print(f"En iyi validation accuracy: {val_acc:.4f}")
    
    if val_acc > best_score:
        best_score = val_acc
        best_params = params

print(f"\nEn iyi parametreler: {best_params}, Validation Accuracy: {best_score:.4f}")


# 5. Final Model EÄŸitimi
final_base = VGG16(weights='imagenet', include_top=False, input_shape=(128,128,3))
for layer in final_base.layers:
    layer.trainable = False

x = Flatten()(final_base.output)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(best_params['dropout_rate'])(x)
outputs = Dense(2, activation='softmax')(x)
final_model = Model(final_base.input, outputs)

final_model.compile(
    optimizer=tf.keras.optimizers.Adam(best_params['learning_rate']),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(patience=7, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True),
    ReduceLROnPlateau(patience=3, factor=0.2)
]

history = final_model.fit(
    datagen.flow(X_train, y_train, batch_size=best_params['batch_size']),
    steps_per_epoch=len(X_train)//best_params['batch_size'],
    epochs=25,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)


# 6. Model DeÄŸerlendirme ve Overfitting/Underfitting KontrolÃ¼
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='EÄŸitim Loss')
plt.plot(history.history['val_loss'], label='DoÄŸrulama Loss')
plt.title('Loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='EÄŸitim Accuracy')
plt.plot(history.history['val_accuracy'], label='DoÄŸrulama Accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()

# GerÃ§ekÃ§i overfit/underfit yorumlama
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]

acc_diff = abs(val_acc - train_acc)

if acc_diff < 0.06:
    print(f"âœ… Model dengeli. EÄŸitim ve doÄŸrulama accuracy farkÄ± Ã§ok kÃ¼Ã§Ã¼k ({acc_diff:.4f}).")
elif val_acc < train_acc:
    print(f"âš ï¸� Model biraz overfitting yapÄ±yor olabilir. Accuracy farkÄ±: {acc_diff:.4f}")
else:
    print(f"âš ï¸� Model biraz underfitting yapÄ±yor olabilir. Accuracy farkÄ±: {acc_diff:.4f}")

test_loss, test_acc = final_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc*100:.2f}%")

y_pred = np.argmax(final_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_true, y_pred, target_names=['Kedi','KÃ¶pek']))


# 7. Grad-CAM
from tensorflow.keras.layers import Conv2D

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(inputs=model.input,
                       outputs=[model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap,0)/tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap_resized), cv2.COLORMAP_JET)
    img_original = ((img + 1) * 127.5).astype('uint8')
    superimposed_img = cv2.addWeighted(img_original, 1-alpha, heatmap_colored, alpha,0)
    return superimposed_img

plt.figure(figsize=(12,6))
for i in range(4):
    plt.subplot(2,4,i+1)
    img = np.expand_dims(X_test[i], axis=0)
    last_conv_layer = [layer.name for layer in final_model.layers if isinstance(layer, Conv2D)][-1]
    heatmap = make_gradcam_heatmap(img, final_model, last_conv_layer)
    superimposed = display_gradcam(X_test[i], heatmap)
    plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f'Pred: {y_pred[i]}')
plt.show()


final_model.save("kedi_kopek_model_final.h5")
print("Model baÅŸarÄ±yla kaydedildi: kedi_kopek_model_final.h5")


