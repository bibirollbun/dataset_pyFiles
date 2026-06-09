import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNet
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# IMG_SIZE = 96
IMG_SIZE = 128
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.6, 1.4],
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    '../input/state-farm-distracted-driver-detection/imgs/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    '../input/state-farm-distracted-driver-detection/imgs/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)


from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNet

input_layer = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
# x = layers.Concatenate()([input_layer, input_layer, input_layer])  # 1->3 channels

base_model = MobileNet(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet',
    alpha=0.5
)
# base_model = tf.keras.applications.MobileNetV2(
#     input_shape=(96, 96, 3),
#     include_top=False,
#     weights='imagenet', 
#     alpha=0.5  
# )
base_model.trainable = False

x = base_model(input_layer)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs=input_layer, outputs=output)


model.summary()


from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)


plot_model(
    model,                    
    to_file='model_architecture.png', 
    show_shapes=True,         
    show_layer_names=True,     
    dpi=96                      
)

print("✅ Model architecture saved as 'model_architecture.png'")


model = tf.keras.models.load_model('/kaggle/input/final-mobilnet128-h5/tensorflow2/default/1/final_MobilNet128.h5')


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# IMG_SIZE = 128
EPOCHS = 15

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)


# IMG_SIZE = 128
EPOCHS = 15

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)


# IMG_SIZE = 96
EPOCHS = 15

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)


model = tf.keras.models.load_model('/kaggle/input/new-model-v3/tensorflow2/default/1/final_MobilNet128v3.h5')


steps = val_generator.samples // BATCH_SIZE + 1

# predictions
y_pred_prob = model.predict(val_generator, steps=steps)
y_pred = np.argmax(y_pred_prob, axis=1)

# true labels
y_true = val_generator.classes


cm = confusion_matrix(y_true, y_pred)
class_names = list(val_generator.class_indices.keys())

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

plt.figure(figsize=(10, 8))
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix - Driver Distraction")
plt.show()


from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    classification_report
)

class_names = list(val_generator.class_indices.keys())

print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))


test_dir = '../input/state-farm-distracted-driver-detection/imgs/test'
import glob

for img_path in glob.glob(f"{test_dir}/*.jpg"):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=(0,-1))  # shape = (1,128,128,1)
    
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    confidence = np.max(pred)
    print(f"{img_path} -> Class: {class_idx}, Confidence: {confidence:.2f}")


model.save("driver_monitor_modelv96.h5")  


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("driver_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved successfully!")



model.save("final_MobilNet128v3.h5")


import tensorflow as tf
import numpy as np


# model = tf.keras.models.load_model('final_MobilNet128.h5')


def representative_data_gen():
    
    
    count = 0
    for imgs, labels in val_generator:
        for i in range(imgs.shape[0]):
           
            data = np.expand_dims(imgs[i], axis=0).astype(np.float32)
            yield [data]
            count += 1
            if count >= 200: 
                return


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen


converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]


converter.inference_input_type = tf.uint8 
converter.inference_output_type = tf.uint8


try:
    tflite_model_quant = converter.convert()
    with open('mobilenet128_aint8v3.tflite', 'wb') as f:
        f.write(tflite_model_quant)
    print("Converted to tflite")
except Exception as e:
    print(f"Error during conversion: {e}")

