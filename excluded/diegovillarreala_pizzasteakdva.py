from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, MobileNetV2
from tensorflow.keras.models import Model


# prompt: download competencia-02-julio-2025.zip contents

!unzip competencia-02-julio-2025.zip


#Imágnes de Pizza
!ls archive/pizza_steak/train/pizza/*.jpg  | wc -l


#Imágnes de Stek
!ls archive/pizza_steak/train/steak/*.jpg  | wc -l


#Imágnes de Stek
!ls archive/pizza_steak/test/steak/*.jpg  | wc -l


#Imágnes de Pizza
!ls archive/pizza_steak/test/pizza/*.jpg  | wc -l


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Preprocesamiento
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                  )

training_set = train_datagen.flow_from_directory(
    'archive/pizza_steak/train',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

test_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                  )

test_set = train_datagen.flow_from_directory(
    'archive/pizza_steak/test',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)




import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

cnn = Sequential()
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[128, 128, 3]))
cnn.add(MaxPooling2D(pool_size=2, strides=2))
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=2, strides=2))
cnn.add(Flatten())
cnn.add(Dense(units=256, activation='relu'))
cnn.add(Dense(units=1, activation='sigmoid'))

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history=cnn.fit(x = training_set, validation_data=test_set, epochs = 50)


loss, acc = cnn.evaluate(training_set)
print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")



import matplotlib.pyplot as plt

def plot_training_history(history):

  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(len(acc))

  plt.figure(figsize=(12, 4))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()





cnn.summary()


plot_training_history(history)


from tensorflow.keras.applications import InceptionV3

inception_model = InceptionV3(weights="imagenet", input_shape=(128, 128, 3), include_top=False)

inception_model.trainable = False


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPool2D, GlobalAveragePooling2D

model = Sequential()
model.add(inception_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(264, activation="relu")) # 64 units
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.summary()


from tensorflow.keras.optimizers import Adam
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(training_set, validation_data=test_set, epochs=30, verbose=1)


loss, acc = model.evaluate(training_set)
print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")


import pandas as pd

history_df = pd.DataFrame(history.history)
history_df.loc[:,['loss', 'val_loss']].plot()
history_df.loc[:,['accuracy', 'val_accuracy']].plot()


model.save("pizzavssteak.keras")



cnn.save("pizzavssteak_V1.keras")


from tensorflow.keras.models import load_model

model = load_model('pizzavssteak.keras')
model_v1 = load_model('pizzavssteak_V1.keras')



from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = train_datagen.flow_from_directory(
    'archive/pizza_steak/test',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle = False
)







import numpy as np

preds = model.predict(test_set, verbose=1)
preds_v1 = model_v1.predict(test_set, verbose=1)
pred_classes = (preds > 0.8).astype(int).flatten()
pred_classes_V1 = (preds_v1 > 0.8).astype(int).flatten()



image_names = [path.split('/')[-1] for path in test_set.filenames]
image_names_v1 = [path.split('/')[-1] for path in test_set.filenames]


labels = ['pizza', 'steak']
results = [labels[i] for i in pred_classes]
results_v1 = [labels[i] for i in pred_classes_V1]


import pandas as pd
import os
import numpy as np
from tensorflow.keras.preprocessing import image

# Define the paths to the test directories for pizza and steak
pizza_test_dir = 'archive/pizza_steak/test/pizza'
steak_test_dir = 'archive/pizza_steak/test/steak'

# Get and sort the filenames for each class
pizza_filenames = sorted(os.listdir(pizza_test_dir))
steak_filenames = sorted(os.listdir(steak_test_dir))

# Take the first 100 filenames from each class
first_100_pizza_filenames = pizza_filenames[:100]
first_100_steak_filenames = steak_filenames[:100]

# Combine the filenames and create corresponding labels
image_filenames_subset = first_100_pizza_filenames + first_100_steak_filenames
labels_subset = ['pizza'] * len(first_100_pizza_filenames) + ['steak'] * len(first_100_steak_filenames)

# Load and preprocess the selected images
images_subset = []
for filename in first_100_pizza_filenames:
    img_path = os.path.join(pizza_test_dir, filename)
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    images_subset.append(img_array)

for filename in first_100_steak_filenames:
    img_path = os.path.join(steak_test_dir, filename)
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    images_subset.append(img_array)

images_subset = np.array(images_subset)

# Get predictions for the subset of images using both models
preds_subset = model.predict(images_subset, verbose=1)
preds_subset_v1 = model_v1.predict(images_subset, verbose=1)

# Determine the predicted classes based on a threshold (e.g., 0.5)
pred_classes_subset = (preds_subset > 0.5).astype(int).flatten()
pred_classes_subset_v1 = (preds_subset_v1 > 0.5).astype(int).flatten()

# Map the predicted class indices to labels
labels_map = {0: 'pizza', 1: 'steak'} # Adjust this based on your model's output mapping
results_subset = [labels_map[i] for i in pred_classes_subset]
results_subset_v1 = [labels_map[i] for i in pred_classes_subset_v1]


# Create DataFrames for the results
df_subset = pd.DataFrame({
    'ID': image_filenames_subset,
    'label': results_subset
})

df_subset_v1 = pd.DataFrame({
    'ID': image_filenames_subset,
    'label': results_subset_v1
})

# Save the DataFrames to CSV files
df_subset.to_csv('resultados_pizza_steak_subset.csv', index=False)
df_subset_v1.to_csv('resultados_pizza_steak_V1_subset.csv', index=False)

print("CSV files with predictions for the first 100 pizza and 100 steak images created.")


import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np
import os

N = 10
IMG_SIZE = (128, 128)

test_root = '/kaggle/input/competencia-02-julio-2025/archive/pizza_steak/test'


image_paths = []
for class_dir in os.listdir(test_root):
    class_path = os.path.join(test_root, class_dir)
    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):
            image_paths.append(os.path.join(class_path, img_name))

random_paths = random.sample(image_paths, N)

plt.figure(figsize=(15, 3))
for i, img_path in enumerate(random_paths):
    # Cargar y preprocesar imagen
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array_exp)[0][0]
    pred_label = 'steak' if pred > 0.5 else 'pizza'

    # Mostrar imagen
    plt.subplot(1, N, i+1)
    plt.imshow(img)
    plt.title(f"Predict: {pred_label}\nFile: {os.path.basename(img_path)}")
    plt.axis('off')

plt.tight_layout()
plt.show()



import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np
import os

N = 10
IMG_SIZE = (128, 128)

test_root = '/kaggle/input/competencia-02-julio-2025/archive/pizza_steak/test'


image_paths = []
for class_dir in os.listdir(test_root):
    class_path = os.path.join(test_root, class_dir)
    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):
            image_paths.append(os.path.join(class_path, img_name))

random_paths = random.sample(image_paths, N)

plt.figure(figsize=(15, 3))
for i, img_path in enumerate(random_paths):
    # Cargar y preprocesar imagen
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array_exp)[0][0]
    pred_label = 'steak' if pred > 0.5 else 'pizza'

    # Mostrar imagen
    plt.subplot(1, N, i+1)
    plt.imshow(img)
    plt.title(f"Predict: {pred_label}\nFile: {os.path.basename(img_path)}")
    plt.axis('off')

plt.tight_layout()
plt.show()



import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pandas as pd

N = 30
IMG_SIZE = (128, 128)

test_root = '/kaggle/input/competencia-02-julio-2025/archive/pizza_steak/test'

image_paths = []
for class_dir in os.listdir(test_root):
    class_path = os.path.join(test_root, class_dir)
    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):
            image_paths.append(os.path.join(class_path, img_name))

random_paths = random.sample(image_paths, N)

results = []

plt.figure(figsize=(15, 3))
for i, img_path in enumerate(random_paths):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array_exp)[0][0]
    pred_label = 'steak' if pred > 0.5 else 'pizza'

    results.append({
        'filename': os.path.basename(img_path),
        'prediction': pred_label
    })

    plt.subplot(1, N, i+1)
    plt.imshow(img)
    plt.title(f"Predict: {pred_label}\nFile: {os.path.basename(img_path)}")
    plt.axis('off')

plt.tight_layout()
plt.show()

df_resultados = pd.DataFrame(results)
display(df_resultados)


