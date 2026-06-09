import os
import numpy as np
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Activation

from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(os.listdir("/kaggle/input/dogs-vs-cats"))


FAST_RUN = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_CHANNELS=3
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)


import zipfile

zip_files = ['test1', 'train']

for zip_file in zip_files:
    with zipfile.ZipFile(f"../input/dogs-vs-cats/{zip_file}.zip", "r") as z:
        z.extractall(".")
        print(f"{zip_file} unzipped")


import zipfile

path = "/kaggle/input/dogs-vs-cats"
for zip_file in os.listdir(path)[:2]:
    zipfile.ZipFile(f"{path}/{zip_file}", 'r').extractall(".")
    print(f"{path}/{zip_file}")
    print(f"{zip_file} unzipped")


train_images = os.listdir("/kaggle/working/train")
CATEGORIES = []

for image_name in train_images:
    class_name = image_name.split('.')[0]  # cat.588.jpg
    CATEGORIES.append(0 if class_name=='cat' else 1)

df = pd.DataFrame({"train_images": train_images, "CATEGORIES": CATEGORIES})


df.head()


df.tail()


df.describe()


df["CATEGORIES"].value_counts()


df["CATEGORIES"].value_counts().plot.bar()


shapes = []

for image in df["train_images"]:
    path = os.path.join("/kaggle/working/train",image)
    image_array = plt.imread(path)
    shapes.append(image_array.shape)
    
print(pd.Series(shapes).value_counts())

index = np.argmin(shapes)
print(f"\nThe Minimum Dimension is ==> {shapes[index]}\n")


sample = rnd.choice(train_images)
image = load_img(os.path.join("/kaggle/working/train", sample))
print(type(image), sample)
plt.imshow(image)
plt.axis("off")
plt.show()


df["CATEGORIES"] = df["CATEGORIES"].replace({0: 'cat', 1: 'dog'})


train_df, valid_df = train_test_split(df, test_size=0.20, random_state=42)

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)


train_df['CATEGORIES'].value_counts()


train_df['CATEGORIES'].value_counts().plot.bar()


valid_df['CATEGORIES'].value_counts()


valid_df['CATEGORIES'].value_counts().plot.bar()


total_train = train_df.shape[0]
total_validate = valid_df.shape[0]
batch_size=15


df.head()


train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "/kaggle/working/train", 
    x_col='train_images',
    y_col='CATEGORIES',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


valid_datagen = ImageDataGenerator(rescale=1./255)

valid_generator = valid_datagen.flow_from_dataframe(
    valid_df, 
    "/kaggle/working/train", 
    x_col='train_images',
    y_col='CATEGORIES',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


example_df = train_df.sample(n=1) # take 1 image-sample
print(example_df, "\n")

example_df= example_df.reset_index(drop=True)
print(example_df, "\n")

example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "/kaggle/working/train", 
    x_col='train_images',
    y_col='CATEGORIES',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)


plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        plt.axis('off')
        break
plt.tight_layout()
plt.show()


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)),
#     BatchNormalization(),
    MaxPool2D(pool_size=(4,4)),
    Dropout(rate=0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    # BatchNormalization(),
    MaxPool2D(pool_size=(4,4)),
    Dropout(rate=0.25),
    
    Conv2D(128, (3, 3), activation='relu'),
    # BatchNormalization(),
    MaxPool2D(pool_size=(4,4)),
    Dropout(rate=0.25),
    
    Flatten(),
    
    Dense(512, activation='relu'),
    Dropout(rate=0.5),

    Dense(2, activation='softmax') # 2 because we have cat and dog classes

])

# Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Summary of Model
model.summary()


from keras.callbacks import EarlyStopping, ReduceLROnPlateau


earlystop = EarlyStopping(patience=5)


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]


epochs = 3 if FAST_RUN else 10

history = model.fit(train_generator,
                    epochs=epochs,
                    validation_data=valid_generator,
#                     validation_steps=total_validate//batch_size, # 5000//15  = 333
#                     steps_per_epoch=total_train//batch_size,     # 20000//15 = 1333
                    callbacks=callbacks)


accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(accuracy))

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.plot(epochs, accuracy, 'bo', label="Train_Acc")
plt.plot(epochs, val_accuracy, 'r', label="val_Acc")
plt.legend(loc='best', shadow=True)

plt.subplot(1,2,2)
plt.plot(epochs, loss, 'bo', label="Train_Loss")
plt.plot(epochs, val_loss, 'r', label="val_Loss")
plt.legend(loc='best', shadow=True)

plt.show()


test_file_names = os.listdir("/kaggle/working/test1")
test_df = pd.DataFrame({'file_name': test_file_names})

total_testing = test_df.shape[0]


test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "/kaggle/working/test1", 
    x_col='file_name',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)


predictions = model.predict(test_generator)


print(predictions)
test_df["CATEGORIES"] = np.argmax(predictions, axis=1)


label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['CATEGORIES'] = test_df['CATEGORIES'].replace(label_map)


test_df['CATEGORIES'] = test_df['CATEGORIES'].replace({ 'dog': 1, 'cat': 0 })


test_df["CATEGORIES"].value_counts().plot.bar()


sample_test = test_df.head(36)

plt.figure(figsize=(15,15))

for index, row in sample_test.iterrows():
    file_name = row['file_name']
    category = row['CATEGORIES']
    path = os.path.join("/kaggle/working/test1", file_name)
    img = load_img(path, target_size=IMAGE_SIZE)
    plt.subplot(6,6,index+1)
    plt.imshow(img)
    plt.title(f"{file_name}\n is ==> {'cat' if category==0 else 'dog'} ({round(predictions[index][category] * 100, 2)}%)")
    plt.axis('off')
plt.tight_layout()
plt.show()


model.save("CNN.h5")
model.save_weights("CNN.weights.h5")


submission_df = test_df.copy()
submission_df['id'] = submission_df['file_name'].str.split('.').str[0]
submission_df['label'] = submission_df['CATEGORIES']
submission_df.drop(['file_name', 'CATEGORIES'], axis=1, inplace=True)
submission_df.to_csv("submission.csv", index=False)


!wget https://wl-genial.cf.tsp.li/resize/728x/jpg/27e/3ac/d457dd54cdbe4443445792ccb8.jpg
imagen = load_img('d457dd54cdbe4443445792ccb8.jpg', target_size=IMAGE_SIZE)
img = np.expand_dims(imagen, 0) # make 'batch' of 1
pred = model.predict(img)
#pred = labels["label_names"][np.argmax(pred)]
category = np.argmax(pred)
plt.imshow(imagen)
plt.title(f"esa imagen es \n ==> {'cat' if category==0 else 'dog'}")
plt.axis('off')
plt.tight_layout()
plt.show()


!wget https://oglebay.com/wp-content/uploads/2022/06/IMG_5670_M-e1654806092422.jpg
imagen = load_img('IMG_5670_M-e1654806092422.jpg', target_size=IMAGE_SIZE)
img = np.expand_dims(imagen, 0) # make 'batch' of 1
pred = model.predict(img)
#pred = labels["label_names"][np.argmax(pred)]
category = np.argmax(pred)
plt.imshow(imagen)
plt.title(f"esa imagen es \n ==> {'cat' if category==0 else 'dog'}")
plt.axis('off')
plt.tight_layout()
plt.show()




