import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow import keras
from tensorflow.keras import layers
from keras import models, layers
import tqdm
from PIL import Image
from sklearn.model_selection import KFold
import numpy as np
from tensorflow.keras import regularizers

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
if os.path.exists("/kaggle/working/imgs"):
    shutil.rmtree("/kaggle/working/imgs")
os.mkdir("/kaggle/working/imgs")
data = np.array(train_imgs[:5])
i = 0
for px_map in data:
    i = i + 1
    px_map = np.reshape(px_map, (48, 48))
    image = Image.fromarray(px_map)
    image = image.convert('RGB')
    image.save('/kaggle/working/imgs/'+str(i)+'.bmp')


# Крос-валідація
# Налаштовуємо K-Fold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []

for train_idx, val_idx in kfold.split(train_imgs, train_lbls):
    # розбиваємо дані
    X_train, X_val = train_imgs[train_idx], train_imgs[val_idx]
    y_train, y_val = train_lbls[train_idx], train_lbls[val_idx]
    
    # будуємо нову модель для кожної ітерації
    model_mlp = keras.Sequential([
    layers.Flatten(input_shape=(48, 48, 1)),
    layers.Dense(120, activation='tanh', kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(5e-4)),
    layers.Dense(84, activation='tanh', kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(5e-4)),
    layers.Dense(7, activation='softmax')
])

    
    model_mlp.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
    
    # тренуємо модель
    history = model_mlp.fit(
    X_train, y_train,
    epochs=5, batch_size=32,
    validation_data=(X_val, y_val),
    verbose=2   
)

    
    # зберігаємо точність на валідації
    scores = model_mlp.evaluate(X_val, y_val, verbose=0)
    cv_scores.append(scores[1])  # accuracy
    print(f"Fold accuracy: {scores[1]:.4f}")

print("Average CV accuracy:", np.mean(cv_scores))


model_mlp.summary()


model_cnn = keras.Sequential()

# перший згортковий шар
model_cnn.add(layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu',
                            input_shape=(48,48,1), kernel_initializer='he_normal'))
model_cnn.add(layers.BatchNormalization()) 
model_cnn.add(layers.MaxPooling2D(pool_size=(2,2)))

# другий згортковий шар
model_cnn.add(layers.Conv2D(filters=64, kernel_size=(5,5), activation='relu',
                            kernel_initializer='he_normal'))
model_cnn.add(layers.BatchNormalization())
model_cnn.add(layers.MaxPooling2D(pool_size=(2,2)))

# перетворення у вектор (міняємо Flatten на GAP)
model_cnn.add(layers.GlobalAveragePooling2D())

# повнозв’язні шари
model_cnn.add(layers.Dense(units=128, activation='relu', kernel_initializer='he_normal'))
model_cnn.add(layers.Dense(units=7, activation='softmax'))

# компіляція
model_cnn.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy']
)




history_cnn = model_cnn.fit(
    train_imgs, train_lbls,         
    epochs=5,                     
    batch_size=32,                  
    validation_data=(val_imgs, val_lbls),  
    verbose=1
)


