# =========================================================
# 1️⃣ Gerekli kütüphaneler
# =========================================================
import os, zipfile, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report


# =========================================================
# 2️⃣ Zip dosyalarını aç
# =========================================================
dataset_path = '/kaggle/input/dogs-vs-cats'

# Çalışma dizini: /kaggle/working
extract_path = '/kaggle/working/dogs-vs-cats'
os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(os.path.join(dataset_path,'train.zip'),'r') as zip_ref:
    zip_ref.extractall(extract_path)

with zipfile.ZipFile(os.path.join(dataset_path,'test1.zip'),'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Zip dosyaları açıldı.")


import os, shutil
import zipfile
from tqdm import tqdm

# 1️⃣ ZIP dosyalarını aç
input_path = '/kaggle/input/dogs-vs-cats'
work_path  = '/kaggle/working/dogs-vs-cats'
os.makedirs(work_path, exist_ok=True)

# Sadece ilk çalıştırmada unzip yap
if not os.path.exists(os.path.join(work_path, 'train')):
    with zipfile.ZipFile(os.path.join(input_path, 'train.zip'), 'r') as z:
        z.extractall(work_path)
    with zipfile.ZipFile(os.path.join(input_path, 'test1.zip'), 'r') as z:
        z.extractall(work_path)

# 2️⃣ Klasör yollarını tanımla
train_original = os.path.join(work_path, 'train')
train_dir = os.path.join(work_path, 'train_split')
cat_dir = os.path.join(train_dir, 'cat')
dog_dir = os.path.join(train_dir, 'dog')

# 3️⃣ Hücre tekrar çalıştırılırsa eski klasörleri sil
shutil.rmtree(train_dir, ignore_errors=True)
os.makedirs(cat_dir)
os.makedirs(dog_dir)

# 4️⃣ Sadece dosyaları uygun klasöre taşı
for fname in tqdm(os.listdir(train_original)):
    src = os.path.join(train_original, fname)
    if not os.path.isfile(src):   # klasör değilse
        continue
    if fname.startswith('cat'):
        shutil.move(src, os.path.join(cat_dir, fname))
    elif fname.startswith('dog'):
        shutil.move(src, os.path.join(dog_dir, fname))



import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)  # %20 validation

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary',
    subset='training')

val_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary',
    subset='validation')

# Örnek görselleri inceleme
x_batch, y_batch = next(train_gen)
plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_batch[i])
    plt.axis('off')
plt.show()



from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(512,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_gen,
    epochs=15,
    validation_data=val_gen
)



plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title('Accuracy')
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title('Loss')
plt.show()



from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

val_gen.reset()
preds = (model.predict(val_gen) > 0.5).astype("int32")

print(confusion_matrix(val_gen.classes, preds))
print(classification_report(val_gen.classes, preds))



import numpy as np, cv2, matplotlib.pyplot as plt, tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

def grad_cam(model, img_path, last_conv_layer_name="conv2d_5",
             pred_index=None, image_size=(150,150), alpha=0.4):
    # ---- 1. Görüntü hazırla
    img = image.load_img(img_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # ---- 2. Bir kez tahmin al
    preds = model.predict(img_array)
    if pred_index is None:
        pred_index = 0   # binary için tek nöron var
    class_output = preds[:, pred_index]

    # ---- 3. Grad model: son conv + model çıkışı
    grad_model = keras.models.Model(
        [model.input],
        [model.get_layer(last_conv_layer_name).output,
         model.layers[-1].output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap,0) / (np.max(heatmap)+1e-10)

    img0 = cv2.imread(img_path)
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap.numpy(), (img0.shape[1], img0.shape[0]))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img0, 1-alpha, heatmap, alpha, 0)
    return heatmap, overlay



# Daha kısa örnek: learning rate değişimi
from tensorflow.keras.optimizers import Adam

for lr in [1e-4, 1e-3]:
    print(f"\n--- Learning rate: {lr} ---")
    model_lr = models.clone_model(model)
    model_lr.compile(optimizer=Adam(learning_rate=lr),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
    model_lr.fit(train_gen, epochs=3, validation_data=val_gen)


