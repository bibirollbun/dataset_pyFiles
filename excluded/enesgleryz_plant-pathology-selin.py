import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0


base_path = "/kaggle/input/plant-pathology-2020-fgvc7"   

print("Dosyalar:", os.listdir(base_path)[:20])
csv_file = os.path.join(base_path, "train.csv")
df = pd.read_csv(csv_file)
print(df.shape)
df.head()


if 'label' not in df.columns:
    possible = {'healthy','multiple_diseases','rust','scab'}
    if possible.issubset(set(df.columns)):
        def make_label(row):
            for c in ['healthy','multiple_diseases','rust','scab']:
                if row[c] == 1:
                    return c
        df['label'] = df.apply(make_label, axis=1)
    elif 'labels' in df.columns or 'category' in df.columns:
        # farklÄ± format varsa:
        df.rename(columns={'labels':'label','category':'label'}, inplace=True)

df['image_id'] = df['image_id'].astype(str) + ".jpg"
print("SÄ±nÄ±f daÄŸÄ±lÄ±mÄ±:")
display(df['label'].value_counts())

# Ã–rnek gÃ¶rseller
sample = df.sample(6, random_state=42)
plt.figure(figsize=(12,6))
for i, row in enumerate(sample.itertuples()):
    plt.subplot(2,3,i+1)
    img = mpimg.imread(os.path.join(base_path, "images", row.image_id))
    plt.imshow(img)
    plt.title(row.label)
    plt.axis('off')
plt.tight_layout()


train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
print(train_df.shape, val_df.shape)

# Augmentation & generator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=25,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df, directory=os.path.join(base_path, "images"),
    x_col="image_id", y_col="label",
    target_size=(224,224), class_mode="categorical", batch_size=32, shuffle=True
)

val_gen = val_datagen.flow_from_dataframe(
    val_df, directory=os.path.join(base_path, "images"),
    x_col="image_id", y_col="label",
    target_size=(224,224), class_mode="categorical", batch_size=32, shuffle=False
)

class_indices = train_gen.class_indices
print("SÄ±nÄ±f indeksi:",class_indices)


from tensorflow.keras import layers, models
import tensorflow as tf

inputs = tf.keras.Input(shape=(224,224,3))
x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
x = layers.MaxPooling2D(2,2)(x)

x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.MaxPooling2D(2,2)(x)

x = layers.Conv2D(128, (3,3), activation='relu')(x)
x = layers.MaxPooling2D(2,2)(x)

x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(class_indices), activation='softmax')(x)

model = models.Model(inputs, outputs)   
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20   
)



plt.figure(); plt.plot(history.history['accuracy'], label='train_acc'); plt.plot(history.history['val_accuracy'], label='val_acc'); plt.legend(); plt.grid();
plt.figure(); plt.plot(history.history['loss'], label='train_loss'); plt.plot(history.history['val_loss'], label='val_loss'); plt.legend(); plt.grid();

# Confusion matrix
val_preds = model.predict(val_gen)
y_true = val_gen.classes
y_pred = np.argmax(val_preds, axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=list(class_indices.keys()), yticklabels=list(class_indices.keys()))
plt.xlabel('Tahmin'); plt.ylabel('GerÃ§ek'); plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_true, y_pred, target_names=list(class_indices.keys()), zero_division=0))


import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def make_gradcam_heatmap(img_array, model):
    # Son Conv2D katmanÄ±nÄ± bul
    conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layer = layer
            break

    # Grad-CAM modeli: son conv + output
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        pred_output = predictions[:, pred_index]

    grads = tape.gradient(pred_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), conv_layer.name



img_path = os.path.join(base_path, "images", val_df.sample(1).iloc[0]['image_id'])
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
img_array = tf.keras.preprocessing.image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

heatmap, used_layer = make_gradcam_heatmap(img_array, model)
print("KullanÄ±lan son conv layer:", used_layer)

# Orijinal resim
img_orig = cv2.imread(img_path)
img_orig = cv2.resize(img_orig, (224,224))

# Heatmap Ã¼stÃ¼ne bindir
heatmap = cv2.resize(heatmap, (224,224))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img_orig


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
plt.title('Original'); plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(superimposed_img.astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.title('Grad-CAM'); plt.axis('off')

plt.show()


# Modeli kaydet
model.save("plant_disease_model.h5")
print(" Model baÅŸarÄ±yla kaydedildi: plant_disease_model.h5")

# EÄŸitim eÄŸrilerini Ã§izdir
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy over epochs")
plt.show()

