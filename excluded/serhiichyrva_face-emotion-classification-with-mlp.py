import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow import keras
from tensorflow.keras import layers
from keras import models, layers
import tqdm
from PIL import Image

import os


PATH = "/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge"

data = pd.read_csv(os.path.join(PATH, "icml_face_data.csv"))

emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}


print(data)


# Function to parse data into right format
# Output: Image in right shaped and normalized + labels
def parse_data(data):
    image_array = np.zeros(shape=(len(data), 48, 48, 1))
    image_label = np.array(list(map(int, data['emotion'])))
    
    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, ' pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48, 1))
        image_array[i] = image
        
    return image_array, image_label

# Splitting the data into train, validation and testing set thanks to Usage column
train_imgs, train_lbls = parse_data(data[data[" Usage"] == "Training"])
val_imgs, val_lbls = parse_data(data[data[" Usage"] == "PrivateTest"])
test_imgs, test_lbls = parse_data(data[data[" Usage"] == "PublicTest"])


print("train shape", np.shape(train_imgs))
print("validation shape", np.shape(val_imgs))
print("validatio shape", np.shape(val_imgs))


print(train_imgs)


import os, shutil 
os.mkdir("/kaggle/working/imgs")
data = np.array(train_imgs[:5])
i = 0
for px_map in data:
    i = i + 1
    px_map = np.reshape(px_map, (48, 48))
    image = Image.fromarray(px_map)
    image = image.convert('RGB')
    image.save('/kaggle/working/imgs/'+str(i)+'.bmp')


# Building a MLP model based on LeNet architecture 

from sklearn.model_selection import StratifiedKFold 

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_imgs, train_lbls)):
    print(f"Fold {fold}")

    X_train, X_val = train_imgs[train_idx], train_imgs[val_idx]
    y_train, y_val = train_lbls[train_idx], train_lbls[val_idx]
    
    model_mlp = keras.Sequential()

    model_mlp.add(layers.Flatten(input_shape=(48, 48, 1)))
    model_mlp.add(layers.Dense(units=120, activation='relu'))
    model_mlp.add(layers.Dense(units=84, activation='relu'))
    model_mlp.add(layers.Dense(units=7, activation = 'softmax'))
    model_mlp.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=keras.optimizers.Adam(lr=1e-3), metrics=['accuracy'])

    history = model_mlp.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=3, # по три епохи для тестовії перевірки крос-валідації, лише демонстраційної 
        batch_size=32,
        verbose=1
    )

    val_loss, val_acc = model_mlp.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy for fold {fold}: {val_acc:.4f}")


model_cnn = keras.Sequential()

# для виконання коду із прикладу треба було оновити назви класів шарів 
# та аргументи, тому що для початкового шару не вистачало кернела і інпутсайза
# а пулінг змінив назву на MaxPooling2D

model_cnn.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model_cnn.add(layers.MaxPooling2D(pool_size=(2,2)))
model_cnn.add(layers.Flatten())
model_cnn.add(layers.Dense(units=84, activation='relu'))
model_cnn.add(layers.Dense(units=7, activation = 'softmax'))
model_cnn.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=keras.optimizers.Adam(lr=1e-3), metrics=['accuracy'])

model_cnn.summary()


model_cnn.fit(train_imgs, train_lbls, epochs=5, batch_size=32, 
              validation_data=(val_imgs, val_lbls), verbose=1)

