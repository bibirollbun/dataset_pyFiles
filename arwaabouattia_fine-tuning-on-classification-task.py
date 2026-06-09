import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import random
import shutil


from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix


print(os.listdir("../input"))


train_zip_path = '/kaggle/input/dogs-vs-cats/train.zip'
test_zip_path = '/kaggle/input/dogs-vs-cats/test1.zip'


base_dir = '/kaggle/working/data'
os.makedirs(base_dir, exist_ok=True)


with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working')


with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working')


train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')


train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
val_cats_dir = os.path.join(val_dir, 'cats')
val_dogs_dir = os.path.join(val_dir, 'dogs')


os.makedirs(train_cats_dir, exist_ok=True)
os.makedirs(train_dogs_dir, exist_ok=True)
os.makedirs(val_cats_dir, exist_ok=True)
os.makedirs(val_dogs_dir, exist_ok=True)


extracted_train_dir = '/kaggle/working/train'


cat_files = [f for f in os.listdir(extracted_train_dir) if f.startswith('cat')]
dog_files = [f for f in os.listdir(extracted_train_dir) if f.startswith('dog')]


train_cat_files, val_cat_files = train_test_split(cat_files, test_size=0.2, random_state=42)
train_dog_files, val_dog_files = train_test_split(dog_files, test_size=0.2, random_state=42)


for fname in train_cat_files:
    src = os.path.join(extracted_train_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

for fname in val_cat_files:
    src = os.path.join(extracted_train_dir, fname)
    dst = os.path.join(val_cats_dir, fname)
    shutil.copyfile(src, dst)


for fname in train_dog_files:
    src = os.path.join(extracted_train_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

for fname in val_dog_files:
    src = os.path.join(extracted_train_dir, fname)
    dst = os.path.join(val_dogs_dir, fname)
    shutil.copyfile(src, dst)


print(f"Number of cat training images: {len(train_cat_files)}")
print(f"Number of dog training images: {len(train_dog_files)}")
print(f"Number of cat verification images: {len(val_cat_files)}")
print(f"Number of dog verification images: {len(val_dog_files)}")


img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


base_model.summary()


for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)


model = Model(inputs=base_model.input, outputs=predictions)


model.summary()


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[checkpoint, earlystop]
)


plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()


validation_generator.reset()
predictions = model.predict(validation_generator, steps=validation_generator.samples // batch_size + 1)
predicted_classes = (predictions > 0.5).astype("int32").flatten()

true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())


print("Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))


cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


sample_imgs, sample_labels = next(validation_generator)
preds_sample = model.predict(sample_imgs)
predicted_sample = (preds_sample > 0.5).astype("int32")

plt.figure(figsize=(12, 12))
for i in range(min(9, len(sample_imgs))):
    plt.subplot(3,3,i+1)
    plt.imshow(sample_imgs[i])
    true_label = 'Dog' if sample_labels[i] == 1 else 'Cat'
    pred_label = 'Dog' if predicted_sample[i] == 1 else 'Cat'
    plt.title(f"True: {true_label}\nPred: {pred_label}")
    plt.axis('off')
plt.tight_layout()
plt.show()


def predict_test_images(test_dir, model, img_size=(224, 224)):
    test_files = os.listdir(test_dir)
    results = []
    
    for i, fname in enumerate(test_files[:12]):  # Ø§Ù�ØªØ±Ø§Ø¶ÙŠØ§Ù‹ ÙŠØ¹Ø±Ø¶ Ø£ÙˆÙ„ 12 ØµÙˆØ±Ø©
        if fname.lower().endswith('.jpg'):
            img_path = os.path.join(test_dir, fname)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, img_size)
            
            img_array = img_resized / 255.0
            
            pred = model.predict(np.expand_dims(img_array, axis=0))[0][0]
            pred_label = "Dog" if pred > 0.5 else "Cat"
            confidence = pred if pred > 0.5 else 1 - pred
            
            results.append((img_resized, pred_label, confidence))
    
    if results:
        plt.figure(figsize=(15, 12))
        for i, (img, label, conf) in enumerate(results):
            plt.subplot(3, 4, i+1)
            plt.imshow(img)
            plt.title(f"Pred: {label}\nConf: {conf:.2f}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return results


test_dir = '/kaggle/working/test1'
predict_test_images(test_dir, model, img_size=(224, 224))

