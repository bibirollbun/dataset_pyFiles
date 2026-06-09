import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image

# Download label data from CSV
labels = pd.read_csv('/kaggle/input/open-data-day-2025-dates-types-classification/train_labels.csv')

# Data verification
print(labels.head())

#here Prepare image paths
image_dir = '/kaggle/input/open-data-day-2025-dates-types-classification/train'
labels['filename'] = labels['filename'].apply(lambda x: os.path.join(image_dir, x))

# Function to download and resize images
def load_images(image_dir, labels):
    images = []
    for img_name in labels['filename']:
        img_path = os.path.join(image_dir, img_name)
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img)
        images.append(img)
    return np.array(images)

# Upload images
images = load_images(image_dir, labels)
print(f"Loaded images shape: {images.shape}")
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels_encoded = le.fit_transform(labels['label'])  #Types of dates
print(le.classes_)


from sklearn.model_selection import train_test_split

# Splitting the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

# Download ResNet50 model without upper layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base layers so that their weights do not change during training.
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dense(len(le.classes_), activation='softmax')(x)  # عدد الفئات

model = Model(inputs=base_model.input, outputs=x)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()


from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False


# Building a model to classify date types
model = models.Sequential([
    layers.InputLayer(input_shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(le.classes_), activation='softmax')
])


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

#model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

#Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)  # تحسين باستخدام SGD

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

X_train, X_val, y_train, y_val = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

y_train = to_categorical(y_train, num_classes=10)  # Assuming 10 classes
y_val = to_categorical(y_val, num_classes=10)

#trining model
model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32), # Use X_train and y_train
    validation_data=(X_val, y_val), # Use X_val and y_val
    epochs=50,
    callbacks=[reduce_lr, early_stopping]
)




import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder

model = tf.keras.models.load_model('/kaggle/input/improved_model/keras/default/1/improved_model.h5')

labels = pd.read_csv('/kaggle/input/open-data-day-2025-dates-types-classification/train_labels.csv')

if 'label' not in labels.columns:
    raise ValueError("The column 'label' was not found in train_labels.csv")

le = LabelEncoder()
le.fit(labels['label'])

img_paths = [
    '/kaggle/input/open-data-day-2025-dates-types-classification/test/2c496a0c.png',
    '/kaggle/input/open-data-day-2025-dates-types-classification/test/0b59e0e8.jpg',
    '/kaggle/input/open-data-day-2025-dates-types-classification/test/0e3f2e72.jpg',
    '/kaggle/input/open-data-day-2025-dates-types-classification/test/179f6419.jpg',
    '/kaggle/input/open-data-day-2025-dates-types-classification/test/1beec184.jpg',
    '/kaggle/input/open-data-day-2025-dates-types-classification/test/3598e77f.jpg',
    '/kaggle/input/open-data-day-2025-dates-types-classification/test/40180d8b.jpg',
    '/kaggle/input/open-data-day-2025-dates-types-classification/test/5d6941a9.jpg',
    '/kaggle/input/open-data-day-2025-dates-types-classification/test/6c8ccd99.jpg',
    '/kaggle/input/open-data-day-2025-dates-types-classification/test/79418931.jpg'
]

plt.figure(figsize=(15, 5))

for i, img_path in enumerate(img_paths):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)


        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_label = le.inverse_transform(predicted_class)[0]


        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.title(predicted_label, fontsize=10)
        plt.axis('off')

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

plt.tight_layout()
plt.show()


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from google.colab import files

model = tf.keras.models.load_model('/kaggle/input/improved_model/keras/default/1/improved_model.h5')

labels = pd.read_csv('/kaggle/input/open-data-day-2025-dates-types-classification/train_labels.csv')
le = LabelEncoder()
le.fit(labels['label'])

test_folder = "/kaggle/input/open-data-day-2025-dates-types-classification/test"
test_images = [img for img in os.listdir(test_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
predictions_list = []

for img_name in test_images:
    img_path = os.path.join(test_folder, img_name)
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_label = le.inverse_transform(predicted_class)

        predictions_list.append([img_name, predicted_label[0]])
    except Exception as e:
        print(f"⚠️ خطأ في معالجة الصورة {img_name}: {e}")


submission_df = pd.DataFrame(predictions_list, columns=['image', 'label'])

submission_df = submission_df.drop_duplicates(subset=['image']).head(126)

submission_file = 'submission.csv'
submission_df.to_csv(submission_file, index=False)

files.download(submission_file)

print(f"✅ تم حفظ النتائج في {submission_file} بنجاح! عدد الصفوف النهائية: {len(submission_df)}")


