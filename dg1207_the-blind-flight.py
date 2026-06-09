# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import cv2
import os
from PIL import Image, ImageFilter
import random
from pathlib import Path
import matplotlib.pyplot as plt
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import json
from IPython.display import display
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import ImageDataGenerator


# GRID_SIZE = 20
# NUM_CLASSES = 5  # 0..4
# BATCH_SIZE = 4
# EPOCHS = 10
# LR = 1e-3

# train_images=Path('/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/train/images')
# train_labels=Path('/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/train/labels')
# train_vel=Path('/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/train/velocities')
# test_images=Path('/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/test/images')
# test_vel=Path('/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/test/velocities')


# image_paths = []
# labels = []

# for json_file in train_labels.glob("*.json"):
#     with open(json_file) as f:
#         data = json.load(f)
#     terrain = data["terrain"]

#     image_file = train_images / (json_file.stem + ".png")

#     if image_file.exists():
#         image_paths.append(str(image_file))
#         labels.append(terrain)
        
# encoder = LabelEncoder()
# labels_encoded = encoder.fit_transform(labels)
# print(encoder.classes_)


base = Image.open(r"/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/train/images/0001.png")
desert={'cactus':Image.open(r"/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/assets/desert/t1_cacti.png").resize((64,64)),
        'goal':Image.open(r"/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/assets/desert/t1_goal.png").resize((64,64)),
        'quicksand':Image.open(r"/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/assets/desert/t1_quicksand.png").resize((64,64)),
        'rocks':Image.open(r"/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/assets/desert/t1_rocks.png").resize((64,64)),
        'rover':Image.open(r"/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/assets/desert/t1_rover.png").resize((64,64)),
        'sand':Image.open(r"/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/assets/desert/t1_sand.png").resize((64,64))}  # Image to paste 400 times

width, height = base.size
base_DIR = Path('/kaggle/working/').resolve()
Base_DIR= base_DIR / "generated_maps" / "desert"
Base_DIR.mkdir(exist_ok=True, parents=True)
test_dir=Base_DIR / "test"
val_dir=Base_DIR / "validation"
train_dir=Base_DIR / "train"
tri=train_dir / "images"
trl=train_dir / "labels"
tei=test_dir / "images"
tel=test_dir / "labels"
vai=val_dir / "images"
val=val_dir / "labels"
test_dir.mkdir(exist_ok=True)
train_dir.mkdir(exist_ok=True)
val_dir.mkdir(exist_ok=True)
tri.mkdir(exist_ok=True)
trl.mkdir(exist_ok=True)
tei.mkdir(exist_ok=True)
tel.mkdir(exist_ok=True)
vai.mkdir(exist_ok=True)
val.mkdir(exist_ok=True)
bd=Path('/kaggle/working/').resolve()
t_dir=bd / 'train'
v_dir=bd / 'validation'
dt=t_dir /'desert'
dv=v_dir / 'desert'
t_dir.mkdir(exist_ok=True)
v_dir.mkdir(exist_ok=True)
dt.mkdir(exist_ok=True)
dv.mkdir(exist_ok=True)
train_i=Path('/kaggle/working/generated_maps/desert/train/images').resolve()
test_i=Path('/kaggle/working/generated_maps/desert/test/images').resolve()
val_i=Path('/kaggle/working/generated_maps/desert/validation/images').resolve()
train_l='/kaggle/working/generated_maps/desert/train/labels/'
test_l='/kaggle/working/generated_maps/desert/test/labels/'
val_l='/kaggle/working/generated_maps/desert/validation/labels/'
L=['sand','sand','sand','sand','sand','sand','sand','sand','sand','sand','cactus','rocks','cactus','rocks','quicksand','quicksand','quicksand','rover','goal']
D={'goal':0, 'rover':1, 'sand':2, 'quicksand':3, 'cactus':4, 'rocks':5}
for i in range(1000):
    if i>899:
        path=test_l+f'image{i}.txt'
        F=open(path,'w')
    elif i>699:
        path=val_l+f'image{i}.txt'
        F=open(path, 'w')
    else:
        path=train_l+f'image{i}.txt'
        F=open(path, 'w')
    x=5
    s=False
    e=False
    y=5
    while x<1385:
        y=5
        while y<1385:
            r=random.randint(0,18)
            overlay=desert[L[r]]
            if r==17 and s==False:
                s=True
            elif r==17 and s:
                continue
            elif r==18 and e==False:
                e=True
            elif r==18 and e:
                continue
            x_avg=((x-2.5) + 34.5)/1385
            y_avg=((y-2.5) + 34.5)/1385
            F.write(f'{D[L[r]]} {x_avg} {y_avg} {69/1385} {69/1385} \n')
            base.paste(overlay, (x, y))
            y+=69
        x=x+69
    d=random.randint(1,10)
    if d<=2:
        base = base.filter(ImageFilter.GaussianBlur(radius=2))
    if i>899:
        base.save(test_i /f'image{i}.png')
    elif i>699:
        base.save(val_i /f'image{i}.png')
        base.save(dv /f'image{i}.png')
    else:
        base.save(train_i /f'image{i}.png')
        base.save(dt /f'image{i}.png')
    F.close()


base = Image.open(r"/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/train/images/0001.png")
forest={'tree':Image.open(r"/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/assets/forest/t0_tree.png").resize((64,64)),
        'goal':Image.open(r"/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/assets/forest/t0_goal.png").resize((64,64)),
        'puddle':Image.open(r"/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/assets/forest/t0_puddle.png").resize((64,64)),
        'startship':Image.open(r"/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/assets/forest/t0_startship.png").resize((64,64)),
        'dirt':Image.open(r"/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/assets/forest/t0_dirt.png").resize((64,64))}  # Image to paste 400 times

width, height = base.size

# Paste 400 times at random positions
base_DIR = Path('/kaggle/working/').resolve()
Base_DIR= base_DIR / "generated_maps" / "forest"
Base_DIR.mkdir(exist_ok=True)
test_dir=Base_DIR / "test"
val_dir=Base_DIR / "validation"
train_dir=Base_DIR / "train"
tri=train_dir / "images"
trl=train_dir / "labels"
tei=test_dir / "images"
tel=test_dir / "labels"
vai=val_dir / "images"
val=val_dir / "labels"

test_dir.mkdir(exist_ok=True)
train_dir.mkdir(exist_ok=True)
val_dir.mkdir(exist_ok=True)
tri.mkdir(exist_ok=True)
trl.mkdir(exist_ok=True)
tei.mkdir(exist_ok=True)
tel.mkdir(exist_ok=True)
vai.mkdir(exist_ok=True)
val.mkdir(exist_ok=True)
ft=t_dir /'forest'
fv=v_dir / 'forest'
ft.mkdir(exist_ok=True)
fv.mkdir(exist_ok=True)
train_i=Path('/kaggle/working/generated_maps/forest/train/images').resolve()
test_i=Path('/kaggle/working/generated_maps/forest/test/images').resolve()
val_i=Path('/kaggle/working/generated_maps/forest/validation/images').resolve()
train_l='/kaggle/working/generated_maps/forest/train/labels/'
test_l='/kaggle/working/generated_maps/forest/test/labels/'
val_l='/kaggle/working/generated_maps/forest/validation/labels/'
L=['dirt','dirt','dirt','dirt','dirt','dirt','dirt','dirt','dirt','dirt','puddle','puddle','puddle','tree','tree','tree','goal','startship']
D={'goal':0, 'startship':1, 'dirt':2, 'puddle':3, 'tree':4}
for i in range(1000):
    path=path+f"image{i}.txt"
    if i>899:
        path=test_l+f'image{i}.txt'
        F=open(path,'w')
    elif i>699:
        path=val_l+f'image{i}.txt'
        F=open(path, 'w')
    else:
        path=train_l+f'image{i}.txt'
        F=open(path, 'w')
    x=5
    s=False
    e=False
    y=5
    while x<1385:
        y=5
        while y<1385:
            r=random.randint(0,17)
            overlay=forest[L[r]]
            if r==16 and s==False:
                s=True
            elif r==16 and s:
                continue
            elif r==17 and e==False:
                e=True
            elif r==17 and e:
                continue
            x_avg=((x-2.5) + 34.5)/1385
            y_avg=((y-2.5) + 34.5)/1385
            F.write(f'{D[L[r]]} {x_avg} {y_avg} {69/1385} {69/1385} \n')
            base.paste(overlay, (x, y))
            y+=69
        x=x+69
    d=random.randint(1,10)
    if d<=2:
        base = base.filter(ImageFilter.GaussianBlur(radius=2))
    if i>899:
        base.save(test_i /f'image{i}.png')
    elif i>699:
        base.save(val_i /f'image{i}.png')
        base.save(fv /f'image{i}.png')
    else:
        base.save(train_i /f'image{i}.png')
        base.save(ft /f'image{i}.png')
    F.close()


base = Image.open(r"/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/train/images/0001.png")
desert={'cactus':Image.open(r"/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/assets/lab/t2_plasma.png").resize((64,64)),
        'goal':Image.open(r"/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/assets/lab/t2_goal.png").resize((64,64)),
        'quicksand':Image.open(r"/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/assets/lab/t2_glue.png").resize((64,64)),
        'rocks':Image.open(r"/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/assets/lab/t2_wall.png").resize((64,64)),
        'rover':Image.open(r"/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/assets/lab/t2_drone.png").resize((64,64)),
        'sand':Image.open(r"/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/assets/lab/t2_floor.png").resize((64,64))}  # Image to paste 400 times

Base_DIR= base_DIR / "generated_maps" / "lab"
Base_DIR.mkdir(exist_ok=True)
test_dir=Base_DIR / "test"
val_dir=Base_DIR / "validation"
train_dir=Base_DIR / "train"
tri=train_dir / "images"
trl=train_dir / "labels"
tei=test_dir / "images"
tel=test_dir / "labels"
vai=val_dir / "images"
val=val_dir / "labels"
test_dir.mkdir(exist_ok=True)
train_dir.mkdir(exist_ok=True)
val_dir.mkdir(exist_ok=True)
tri.mkdir(exist_ok=True)
trl.mkdir(exist_ok=True)
tei.mkdir(exist_ok=True)
tel.mkdir(exist_ok=True)
vai.mkdir(exist_ok=True)
val.mkdir(exist_ok=True)
lt=t_dir /'lab'
lv=v_dir / 'lab'
lt.mkdir(exist_ok=True)
lv.mkdir(exist_ok=True)
train_i=Path('/kaggle/working/generated_maps/lab/train/images').resolve()
test_i=Path('/kaggle/working/generated_maps/lab/test/images').resolve()
val_i=Path('/kaggle/working/generated_maps/lab/validation/images').resolve()
train_l='/kaggle/working/generated_maps/lab/train/labels/'
test_l='/kaggle/working/generated_maps/lab/test/labels/'
val_l='/kaggle/working/generated_maps/lab/validation/labels/'
L=['sand','sand','sand','sand','sand','sand','sand','sand','sand','sand','cactus','rocks','cactus','rocks','quicksand','quicksand','quicksand','rover','goal']
D={'goal':0, 'rover':1, 'sand':2, 'quicksand':3, 'cactus':4, 'rocks':5}
for i in range(1000):
    path=path+f"image{i}.txt"
    if i>899:
        path=test_l+f'image{i}.txt'
        F=open(path,'w')
    elif i>699:
        path=val_l+f'image{i}.txt'
        F=open(path, 'w')
    else:
        path=train_l+f'image{i}.txt'
        F=open(path, 'w')
    x=5
    s=False
    e=False
    y=5
    while x<1385:
        y=5
        while y<1385:
            r=random.randint(0,18)
            overlay=desert[L[r]]
            if r==17 and s==False:
                s=True
            elif r==17 and s:
                continue
            elif r==18 and e==False:
                e=True
            elif r==18 and e:
                continue
            x_avg=((x-2.5) + 34.5)/1385
            y_avg=((y-2.5) + 34.5)/1385
            F.write(f'{D[L[r]]} {x_avg} {y_avg} {69/1385} {69/1385} \n')
            base.paste(overlay, (x, y))
            y+=69
        x=x+69
    d=random.randint(1,10)
    if d<=2:
        base = base.filter(ImageFilter.GaussianBlur(radius=2))
    if i>899:
        base.save(test_i /f'image{i}.png')
    elif i>699:
        base.save(val_i /f'image{i}.png')
        base.save(lv /f'image{i}.png')
    else:
        base.save(train_i /f'image{i}.png')
        base.save(lt /f'image{i}.png')
    F.close()


train_ds= tf.keras.utils.image_dataset_from_directory(
    directory='/kaggle/working/train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256,256)
)

validation_ds= tf.keras.utils.image_dataset_from_directory(
    directory='/kaggle/working/validation',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256,256)
)


def process(image, label):
  image= tf.cast(image/255, tf.float32)
  return image, label

train_ds= train_ds.map(process)
validation_ds= validation_ds.map(process)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), padding='valid', activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))

model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),padding='valid',strides=2))

model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),padding='valid',strides=2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation="softmax"))


model.summary()


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])


history=model.fit(train_ds, epochs=10, validation_data=validation_ds)


#accuracy check
plt.plot(history.history['accuracy'], color='red', label='train')
plt.plot(history.history['val_accuracy'], color='blue', label='validation')
plt.legend()
plt.show()


#overfitting check
plt.plot(history.history['loss'], color='red', label='train')
plt.plot(history.history['val_loss'], color='blue', label='validation')
plt.legend()
plt.show()


!pip list | grep torch


%pip install ultralytics





import ultralytics
ultralytics.checks()


import yaml

# Define your configuration
data_config = {
    'path': '/kaggle/working/generated_maps/desert',
    'train': 'train/images',
    'val': 'validation/images',
    'test': 'test/images',
    'nc': 6,
    'names': ['goal', 'rover', 'sand', 'quicksand','cactus', 'rocks']
}

# Write to the working directory
with open('/kaggle/working/generated_maps/desert/data.yaml', 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

print("data.yaml has been created!")


# Define your configuration
data_config = {
    'path': '/kaggle/working/generated_maps/forest',
    'train': 'train/images',
    'val': 'validation/images',
    'test': 'test/images',
    'nc': 5,
    'names': ['goal', 'startship', 'dirt', 'puddle','tree']
}

# Write to the working directory
with open('/kaggle/working/generated_maps/forest/data.yaml', 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

print("data.yaml has been created!")


# Define your configuration
data_config = {
    'path': '/kaggle/working/generated_maps/lab',
    'train': 'train/images',
    'val': 'validation/images',
    'test': 'test/images',
    'nc': 6,
    'names': ['goal', 'drone', 'floor', 'glue', 'plasma', 'wall']
}

# Write to the working directory
with open('/kaggle/working/generated_maps/lab/data.yaml', 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

print("data.yaml has been created!")


!yolo detect train data=/kaggle/working/generated_maps/desert/data.yaml model=yolo11x.pt epochs=30 imgsz=640 batch=-1 name=des


!yolo detect train data=/kaggle/working/generated_maps/forest/data.yaml model=yolo11x.pt epochs=30 imgsz=640 batch=-1 name=forest
!yolo detect train data=/kaggle/working/generated_maps/lab/data.yaml model=yolo11x.pt epochs=30 imgsz=640 batch=-1 name=lab


# labels= ['goal', 'rover', 'sand', 'quicksand','cactus', 'rocks']
# yolo=cv2.dnn.readNetFromONNX('/kaggle/working/runs/detect/des5/weights')
# yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# yolo.setPreferable.Target(cv2.dnn.DNN_TARGET_CPU)


# img= cv2.imread('')
# image= img.copy()
# row, col, d = image.shape
# wh=640
# blob= cv2.dnn.blobFromImage(image, 1/255, (wh, wh),swapRB= True, crop=False)
# yolo.setInput(blob)
# preds= yolo.forward()







