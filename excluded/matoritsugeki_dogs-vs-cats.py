def get_data_and_label(image_path):
    labels = []
    datas = []
    files = []
    for path in os.listdir(image_path):
        if 'dog' in path:
            labels.append(1)
        else:
            labels.append(0)

        feature = data.imread(f'{image_path}/{path}')
        feature = transform.resize(feature, (100, 100), mode='reflect')    
        datas.append(convert_image(feature))
        files.append(path)
    return datas, labels,files



import os
import random
import shutil

# ãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒªè¨­å®šï¼ˆã�‚ã�ªã�Ÿã�®å®Ÿéš›ã�®ãƒ‘ã‚¹ã�«å¤‰æ›´ã�—ã�¦ã��ã� ã�•ã�„ï¼‰
SOURCE_DIR = "/path/to/source"  # å…ƒã�®ãƒ•ã‚¡ã‚¤ãƒ«ã�Œå…¥ã�£ã�¦ã�„ã‚‹ãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒª
TRAINING_DIR = "/path/to/train"
VALIDATION_DIR = "/path/to/val"

SPLIT_SIZE = 0.9  # 90%ã‚’å­¦ç¿’ç”¨ã�«



import os

# å…¥åŠ›ãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒªã�®ä¸­èº«ã‚’ç¢ºèª�ã�™ã‚‹
print(os.listdir("/kaggle/input"))



import os

print(os.listdir("/kaggle/input/dogs-vs-cats-redux-kernels-edition"))




print(os.listdir("/kaggle/input"))



import zipfile
import os

zip_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
extract_path = "/kaggle/working/train"

# è§£å‡�
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)



SOURCE_DIR = "/kaggle/working/train/train"  # ZIPè§£å‡�å¾Œã�®æ­£ã�—ã�„ç”»åƒ�ãƒ•ã‚©ãƒ«ãƒ€
TRAINING_DIR = "/kaggle/working/split/train"
VALIDATION_DIR = "/kaggle/working/split/val"
SPLIT_SIZE = 0.9

file_list = [os.path.join(SOURCE_DIR, f) for f in os.listdir(SOURCE_DIR)
             if os.path.isfile(os.path.join(SOURCE_DIR, f)) and os.path.getsize(os.path.join(SOURCE_DIR, f)) > 0]



import os
from glob import glob
from shutil import copyfile

SOURCE_DIR = "/path/to/source/images/"
TRAINING_DIR = "/path/to/train/"
VALIDATION_DIR = "/path/to/val/"

# ã�™ã�¹ã�¦ã�®ç”»åƒ�ãƒ•ã‚¡ã‚¤ãƒ«ã‚’å�–å¾—
all_files = glob(SOURCE_DIR + "*.jpg")

# 8:2ã�§åˆ†å‰²ï¼ˆä¾‹ï¼‰
split_idx = int(len(all_files) * 0.8)
train_list = all_files[:split_idx]
val_list = all_files[split_idx:]

# ãƒ•ã‚¡ã‚¤ãƒ«ã‚’ã‚³ãƒ”ãƒ¼
for f in train_list:
    copyfile(f, os.path.join(TRAINING_DIR, os.path.basename(f)))

for f in val_list:
    copyfile(f, os.path.join(VALIDATION_DIR, os.path.basename(f)))



import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAINING_DIR = "/content/data/train"  # â†�ã�“ã�“ã‚’å®Ÿéš›ã�®ãƒ‘ã‚¹ã�«å�ˆã‚�ã�›ã�¦ä¿®æ­£

# å­˜åœ¨ãƒ�ã‚§ãƒƒã‚¯
if not os.path.exists(TRAINING_DIR):
    print("ã‚¨ãƒ©ãƒ¼: TRAINING_DIR ã�Œå­˜åœ¨ã�—ã�¾ã�›ã‚“")
else:
    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
        directory=TRAINING_DIR,
        target_size=(256, 256),
        class_mode='categorical',
        batch_size=32
    )



import os

# /content ä»¥ä¸‹ã�®ãƒ•ã‚©ãƒ«ãƒ€ä¸€è¦§ã‚’è¦‹ã‚‹
print("ç�¾åœ¨ã�®ãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒª:", os.getcwd())
print("contentç›´ä¸‹:", os.listdir("/content"))

# dataãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒªã�Œã�‚ã‚‹ã�ªã‚‰ã��ã�®ä¸­ã‚‚è¦‹ã‚‹
if os.path.exists("/content/data"):
    print("dataã�®ä¸­:", os.listdir("/content/data"))



SOURCE_DIR = "/kaggle/input/my-dataset/"
TRAINING_DIR = "/kaggle/working/train/"
VALIDATION_DIR = "/kaggle/working/val/"



import os

os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)



import os

print("inputãƒ•ã‚©ãƒ«ãƒ€ã�®ä¸­èº«:", os.listdir("/kaggle/input"))



os.listdir("/kaggle/input")


import os

path = os.path.join("/kaggle/input", "dogs-vs-cats-redux-kernels-edition")
print(os.listdir(path))



import zipfile
import os

zip_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
extract_dir = "/kaggle/working/train"

# è§£å‡�å‡¦ç�†
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# è§£å‡�ã�—ã�Ÿãƒ•ã‚©ãƒ«ãƒ€ã�®ä¸­èº«ã‚’ç¢ºèª�
print("è§£å‡�å¾Œã�®ãƒ•ã‚¡ã‚¤ãƒ«ä¾‹:", os.listdir(extract_dir)[:10])



import shutil

# åˆ†é¡�å…ˆã�®ãƒ•ã‚©ãƒ«ãƒ€
cat_dir = "/kaggle/working/sorted/train/cat"
dog_dir = "/kaggle/working/sorted/train/dog"

os.makedirs(cat_dir, exist_ok=True)
os.makedirs(dog_dir, exist_ok=True)

# trainãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒªå†…ã�®ç”»åƒ�ã‚’å�–å¾—
for filename in os.listdir(extract_dir):
    if filename.startswith("cat"):
        shutil.copy(os.path.join(extract_dir, filename), os.path.join(cat_dir, filename))
    elif filename.startswith("dog"):
        shutil.copy(os.path.join(extract_dir, filename), os.path.join(dog_dir, filename))

print("åˆ†é¡�å®Œäº†")



from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    directory="/kaggle/working/sorted/train",
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    directory="/kaggle/working/sorted/train",
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)



from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ãƒ‡ãƒ¼ã‚¿ã‚¸ã‚§ãƒ�ãƒ¬ãƒ¼ã‚¿ã�®å®šç¾©ï¼ˆå‰�å‡¦ç�†è¾¼ã�¿ï¼‰
datagen = ImageDataGenerator(rescale=1./255)

# flow_from_directory ã‚’ datagen ã�‹ã‚‰å‘¼ã�³å‡ºã�™
train_generator = datagen.flow_from_directory(
    directory="/kaggle/working/sorted/train",
    target_size=(256, 256),
    class_mode='categorical',  # 2ã‚¯ãƒ©ã‚¹ã�ªã‚‰ 'binary' ã�§ã‚‚OK
    batch_size=32
)



# GRADED FUNCTION: train_val_generators
def train_val_generators(TRAINING_DIR, VALIDATION_DIR):
  """
  Creates the training and validation data generators
  
  Args:
    TRAINING_DIR (string): directory path containing the training images
    VALIDATION_DIR (string): directory path containing the testing/validation images
    
  Returns:
    train_generator, validation_generator - tuple containing the generators
  """
  ### START CODE HERE

  # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
  train_datagen = ImageDataGenerator( rescale = 1.0/255. )

  # Pass in the appropiate arguments to the flow_from_directory method
  train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                      batch_size=50,
                                                      class_mode='binary',
                                                      target_size=(150, 150))

  # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
  validation_datagen = ImageDataGenerator( rescale = 1.0/255. )

  # Pass in the appropiate arguments to the flow_from_directory method
  validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                batch_size=50,
                                                                class_mode='binary',
                                                                target_size=(150, 150))
  ### END CODE HERE
  return train_generator, validation_generator


from tensorflow.keras.optimizers import RMSprop
# GRADED FUNCTION: create_model
def create_model():
  # DEFINE A KERAS MODEL TO CLASSIFY CATS V DOGS
  # USE AT LEAST 3 CONVOLUTION LAYERS

  ### START CODE HERE

  model = tf.keras.models.Sequential([ 
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(), 
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'), 
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(1, activation='sigmoid')  
  ])

  
  model.compile(optimizer=RMSprop(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']) 
    
  ### END CODE HERE

  return model


from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    directory="/kaggle/working/sorted/train",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    directory="/kaggle/working/sorted/train",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)



import os

extract_dir = "/kaggle/working/train/train"  # ãƒ•ã‚©ãƒ«ãƒ€ã�ŒäºŒé‡�ã�ªã�®ã�§ã�“ã�“ã�«å¤‰æ›´
print(f"{extract_dir} ã�®ä¸­èº«ã�®ãƒ•ã‚¡ã‚¤ãƒ«æ•°:", len(os.listdir(extract_dir)))
print("ãƒ•ã‚¡ã‚¤ãƒ«ä¾‹:", os.listdir(extract_dir)[:10])



import shutil

cat_dir = "/kaggle/working/sorted/train/cat"
dog_dir = "/kaggle/working/sorted/train/dog"

os.makedirs(cat_dir, exist_ok=True)
os.makedirs(dog_dir, exist_ok=True)

for filename in os.listdir(extract_dir):
    lower_name = filename.lower()
    if "cat" in lower_name:
        shutil.copy(os.path.join(extract_dir, filename), os.path.join(cat_dir, filename))
    elif "dog" in lower_name:
        shutil.copy(os.path.join(extract_dir, filename), os.path.join(dog_dir, filename))

print("æŒ¯ã‚Šåˆ†ã�‘å®Œäº†")



for folder_name in ['cat', 'dog']:
    folder_path = f"/kaggle/working/sorted/train/{folder_name}"
    files = os.listdir(folder_path)
    print(f"ãƒ•ã‚©ãƒ«ãƒ€ {folder_name} ã�®ä¸­ã�«ã�‚ã‚‹ãƒ•ã‚¡ã‚¤ãƒ«æ•°: {len(files)}")
    print(f"ãƒ•ã‚¡ã‚¤ãƒ«ä¾‹: {files[:5]}")



from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    "/kaggle/working/sorted/train",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    "/kaggle/working/sorted/train",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)



import tensorflow as tf
from tensorflow.keras import layers, models

def create_model():
    model = models.Sequential([
        layers.Input(shape=(150, 150, 3)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # 2ã‚¯ãƒ©ã‚¹åˆ†é¡�ã�ªã�®ã�§1ãƒ¦ãƒ‹ãƒƒãƒˆ+sigmoid
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model



import tensorflow as tf
from tensorflow.keras import layers, models

def create_model():
    model = models.Sequential([
        layers.Input(shape=(150, 150, 3)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # 2ã‚¯ãƒ©ã‚¹åˆ†é¡�ã�ªã�®ã�§1ãƒ¦ãƒ‹ãƒƒãƒˆ+sigmoid
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model



import tensorflow as tf
from tensorflow.keras import layers, models

def create_model():
    model = models.Sequential([
        tf.keras.Input(shape=(150, 150, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(2, activation='softmax')  # 2ã‚¯ãƒ©ã‚¹åˆ†é¡� (cat, dog)
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model



import zipfile
import os

zip_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
extract_path = "/kaggle/working/train"

os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("âœ… è§£å‡�å®Œäº†:", len(os.listdir(extract_path)), "ãƒ•ã‚¡ã‚¤ãƒ«")
print("ğŸ“„ ãƒ•ã‚¡ã‚¤ãƒ«ä¾‹:", os.listdir(extract_path)[:5])



import os

path = "/kaggle/working/train/train"
files = os.listdir(path)

print(f"ğŸ“� {path} ã�®ä¸­ã�®ãƒ•ã‚¡ã‚¤ãƒ«æ•°: {len(files)}")
print("ğŸ“„ ãƒ•ã‚¡ã‚¤ãƒ«å��ä¾‹:", files[:5])



import os, shutil
from sklearn.model_selection import train_test_split

# å…ƒã�®ç”»åƒ�ãƒ•ã‚©ãƒ«ãƒ€
original_dir = "/kaggle/working/train/train"

# æ–°ã�—ã�„ãƒ‡ãƒ¼ã‚¿æ§‹é€ ã�®ãƒ™ãƒ¼ã‚¹ãƒ‘ã‚¹
base_dir = "/kaggle/working/dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")

# ã‚¯ãƒ©ã‚¹
classes = ["cat", "dog"]

# ãƒ•ã‚©ãƒ«ãƒ€ä½œæˆ�
for category in classes:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)

# ãƒ•ã‚¡ã‚¤ãƒ«ã‚’ã‚¯ãƒ©ã‚¹ã�”ã�¨ã�«åˆ†ã�‘ã�¦ train/val ã�«åˆ†å‰²
all_files = os.listdir(original_dir)
for category in classes:
    files = [f for f in all_files if f.startswith(category)]
    train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)

    for f in train_files:
        shutil.copy(os.path.join(original_dir, f), os.path.join(train_dir, category, f))
    for f in val_files:
        shutil.copy(os.path.join(original_dir, f), os.path.join(val_dir, category, f))

print("âœ… ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã�®æŒ¯ã‚Šåˆ†ã�‘å®Œäº†")



from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ãƒ‡ãƒ¼ã‚¿æ‹¡å¼µã�®è¨­å®šï¼ˆè¨“ç·´ç”¨ï¼‰
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

# æ¤œè¨¼ç”¨ã�¯ãƒªã‚¹ã‚±ãƒ¼ãƒªãƒ³ã‚°ã�®ã�¿
val_datagen = ImageDataGenerator(rescale=1./255)

# ãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒªã�‹ã‚‰ç”»åƒ�ã‚’èª­ã�¿è¾¼ã‚€
train_generator = train_datagen.flow_from_directory(
    "/kaggle/working/dataset/train",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = val_datagen.flow_from_directory(
    "/kaggle/working/dataset/validation",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)



from tensorflow.keras import layers, models

# ãƒ¢ãƒ‡ãƒ«æ§‹ç¯‰
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # ãƒ�ã‚¤ãƒŠãƒªåˆ†é¡�
])

# ã‚³ãƒ³ãƒ‘ã‚¤ãƒ«
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# å­¦ç¿’
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator
)



import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()



import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# ãƒ†ã‚¹ãƒˆç”»åƒ�ã�®ãƒ‘ã‚¹
test_dir = '/kaggle/working/test'  # è§£å‡�æ¸ˆã�¿ã�ªã‚‰ã�“ã�®ãƒ‘ã‚¹ã�«ã�‚ã‚‹ã�¯ã�š
test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]

# å‡ºåŠ›ãƒ‡ãƒ¼ã‚¿ä¿�æŒ�ç”¨
predictions = []

for img_name in test_images:
    img_path = os.path.join(test_dir, img_name)
    img = image.load_img(img_path, target_size=(150, 150))  # å­¦ç¿’æ™‚ã�¨å�Œã�˜ã‚µã‚¤ã‚º
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # äºˆæ¸¬
    pred = model.predict(img_array)[0][0]
    predictions.append([img_name.split('.')[0], pred])  # id, label

# DataFrameã�«å¤‰æ�›ã�—ã�¦CSVä¿�å­˜
submission_df = pd.DataFrame(predictions, columns=['id', 'label'])
submission_df['id'] = submission_df['id'].astype(int)  # idåˆ—ã‚’intã�«
submission_df = submission_df.sort_values(by='id')     # idã�§ä¸¦ã�³æ›¿ã�ˆ

submission_df.to_csv('/kaggle/working/submission.csv', index=False)


