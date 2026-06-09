import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import glob
import random

# Ruta a las imágenes
DATA_PATH = '/kaggle/input/facial-recognition-7th-gen/Fotos Grupo/'
student_folders = sorted(glob.glob(DATA_PATH + '*/'))
class_names = [p.split('/')[-2] for p in student_folders]
label_map = {name: i for i, name in enumerate(class_names)}
num_classes = len(class_names)

print(f"Se encontraron {num_classes} clases (alumnos): {class_names}")

# Preparar los datos
train_paths, train_labels, test_paths, test_labels = [], [], [], []

for student_folder in student_folders:
    student_name = student_folder.split('/')[-2]
    image_paths = glob.glob(student_folder + '*.jpg')
    random.shuffle(image_paths)
    
    current_train_paths = image_paths[:20]
    current_test_paths = image_paths[20:]
    
    num_train_images = len(current_train_paths)
    num_test_images = len(current_test_paths)
    
    train_paths.extend(current_train_paths)
    test_paths.extend(current_test_paths)
    
    train_labels.extend([label_map[student_name]] * num_train_images)
    test_labels.extend([label_map[student_name]] * num_test_images)

# Parámetros
IMG_SIZE = 160
BATCH_SIZE = 32

# Función de preprocesamiento
def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = image / 255.0
    return image, label

# Crear datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_dataset = train_dataset.shuffle(len(train_paths)).map(load_and_preprocess_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
test_dataset = test_dataset.map(load_and_preprocess_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Modelo base (más rápido que EfficientNet)
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

# Construcción del modelo
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = models.Model(inputs, outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Entrenar
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset
)

# Guardar el modelo con el nombre 'arnold.h5'
model.save('arnold.h5')
print("Modelo guardado como 'arnold.h5'")


