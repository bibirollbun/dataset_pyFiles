import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input


train_images=r"/kaggle/input/paddy-disease-classification/train_images"
test_images=r"/kaggle/input/paddy-disease-classification/test_images"


datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)


train_generator = datagen.flow_from_directory(
    train_images,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)


val_generator = datagen.flow_from_directory(
    train_images,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)


# Save the inverse class mapping in memory (no JSON)
class_indices = train_generator.class_indices
inv_class_indices = {v: k for k, v in class_indices.items()}


model = models.Sequential([
    Input(shape=(64, 64, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.summary()


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


hsitory=model.fit(train_generator, validation_data=val_generator, epochs=10)


predictions = []

for fname in sorted(os.listdir(test_images)):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(test_images, fname)
        img = image.load_img(path, target_size=(64, 64))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        class_idx = np.argmax(pred)
        label = inv_class_indices[class_idx]
        predictions.append((fname, label))


print("Saving predictions to submission.csv")
df = pd.DataFrame(predictions, columns=["image_id", "label"])
df.to_csv("submission.csv", index=False)




