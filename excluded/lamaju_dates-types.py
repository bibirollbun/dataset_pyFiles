import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
import random
import math


data_dir = "/kaggle/input/open-data-day-2025-dates-types-classification"
train_csv = os.path.join(data_dir, "train_labels.csv")
image_dir = os.path.join(data_dir, "train") 


train_df = pd.read_csv(train_csv)


print("\nContents in data_dir:", os.listdir(data_dir))
print("Contents in image_dir:", os.listdir(image_dir)[:10])  


train_df["filename"] = train_df["filename"].apply(lambda x: os.path.join(image_dir, x))


print("\nعدد الملفات الصحيحة:", sum(os.path.exists(f) for f in train_df["filename"]))


train_df = pd.read_csv(train_csv)


print("First five rows train_df:")
print(train_df.head())



print(f"Total training samples: {len(train_df)}")

class_counts = train_df['label'].value_counts()
print("\nClass distribution:")
display(class_counts)

plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar')
plt.xlabel("Date Type (Class)")
plt.ylabel("Count")
plt.title("Distribution of Date Types in Training Data")
plt.show()


train_df["filename"] = train_df["filename"].str.strip()
train_df["filename"] = train_df["filename"].apply(lambda x: os.path.join(image_dir, x))


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col="filename",
    y_col="label",
    subset="training",
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical",
    shuffle=True
)

validation_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col="filename",
    y_col="label",
    subset="validation",
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

print(" train_generator and validation_generator created successfully!")



print("عدد الفئات:", len(train_generator.class_indices))


batch = next(iter(train_generator))
print(batch[0].shape, batch[1].shape)  


print("Number of images in training :", train_generator.samples)
print("Number of images in verification:", validation_generator.samples)


print("Number of training samples:", train_generator.samples)
print("Batch size:", train_generator.batch_size)
print("Calculated steps_per_epoch:", train_generator.samples // train_generator.batch_size)


print("عدد الفئات:", len(train_generator.class_indices))
print("الفئات المتاحة:", train_generator.class_indices)


from tensorflow.keras.optimizers import Adam

model = Sequential([
    Input(shape=(150,150,3)),

    Conv2D(32, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()



from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)


history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=30,
    callbacks=[early_stopping, reduce_lr]
)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

img_path = "/kaggle/input/open-data-day-2025-dates-types-classification/test/3f9b55cd.png"

img = image.load_img(img_path, target_size=(150, 150))

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  
img_array /= 255.0 

plt.imshow(img)
plt.axis('off')
plt.title(" Test image")
plt.show()



predictions = model.predict(img_array)

predicted_class_index = np.argmax(predictions, axis=1)[0]

class_labels = list(train_generator.class_indices.keys())  
predicted_class_label = class_labels[predicted_class_index]

print(f"Predict the model: {predicted_class_label}")

plt.imshow(img)
plt.axis('off')
plt.title(f"Predicted: {predicted_class_label}")
plt.show()



model.save("/kaggle/working/final_model.h5")


test_images_path = "/kaggle/input/open-data-day-2025-dates-types-classification/test/"
test_filenames = sorted(os.listdir(test_images_path))  

predictions_list = []

class_labels = list(train_generator.class_indices.keys())

for fname in test_filenames:
    img_path = os.path.join(test_images_path, fname)
    
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]

    predictions_list.append({"filename": fname, "label": predicted_class_label})

submission_df = pd.DataFrame(predictions_list)
submission_df.to_csv("/kaggle/working/submission.csv", index=False)

print(" `submission.csv` has been successfully saved!")


