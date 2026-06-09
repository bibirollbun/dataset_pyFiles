import os

print(os.listdir("../input"))
print(os.listdir("../input/dogs-vs-cats-redux-kernels-edition"))


from zipfile import ZipFile

data_path = "../input/dogs-vs-cats-redux-kernels-edition/"

# í›ˆë ¨ ì�´ë¯¸ì§€ íŒŒì�¼ ì••ì¶• í’€ê¸°
with ZipFile(data_path + 'train.zip') as zipper:
    zipper.extractall()

# í…ŒìŠ¤íŠ¸ ì�´ë¯¸ì§€ íŒŒì�¼ ì••ì¶• í’€ê¸°
with ZipFile(data_path + 'test.zip') as zipper:
    zipper.extractall()


print("í›ˆë ¨ ë�°ì�´í„° ê°œìˆ˜:", len(os.listdir('train/')))
print("í…ŒìŠ¤íŠ¸ ë�°ì�´í„° ê°œìˆ˜:", len(os.listdir('test/')))


print(os.listdir("train/")[0:5])
print(os.listdir("test/")[0:5])


import pandas as pd

filenames = []
labels = []
for filename in os.listdir("train/"):
    filenames.append(filename)
    if filename.split('.')[0] == "cat":
        labels.append(0)
    else:
        labels.append(1)

df = pd.DataFrame({
    'filename': filenames,
    'label': labels
})

df


import numpy as np

df["label"].value_counts()


import matplotlib.pyplot as plt
import random as r
import cv2

plt.figure(figsize=(15, 15))

row = 3
col = 3

for i in range(row * col):
    img_idx = r.randint(0, len(df))
    filename = df["filename"][img_idx]
    img = cv2.imread("train/" + filename)

    plt.subplot(row, col, i+1)
    plt.imshow(img)
    plt.title(filename)


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Rescaling

IMAGE_WIDTH = 112
IMAGE_HEIGHT = 112
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

model.summary()


directory = "train/"
sorted_filenames = sorted(filenames)
labels = [0 if filename.split(".")[0] == "cat" else 1 for filename in sorted_filenames]


from tensorflow.keras.utils import image_dataset_from_directory

training_data, validation_data = image_dataset_from_directory(
    directory,
    labels=labels,
    label_mode="binary",
    image_size=IMAGE_SIZE,
    seed=42,
    validation_split=0.15,
    subset="both"
)

training_data = training_data.map(lambda x, y : (x/255.0, y))
validation_data = validation_data.map(lambda x, y : (x/255.0, y))


# ë°°ì¹˜ì—�ì„œ ì�´ë¯¸ì§€ì™€ ë�¼ë²¨ ê°€ì ¸ì˜¤ê¸°
image_batch, label_batch = next(iter(training_data))

# 9ê°œ ì�´ë¯¸ì§€ ì¶œë ¥
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i])  # ì�´ë¯¸ì§€ ë³€í™˜
    plt.title(f"Label: {label_batch[i].numpy()}")  # ë�¼ë²¨ í‘œì‹œ
    plt.axis("off")
plt.show()


history = model.fit(training_data, epochs=10, validation_data=validation_data)


loss, accuracy = model.evaluate(validation_data)
print(f"ğŸ“Œ ëª¨ë�¸ í�‰ê°€ ê²°ê³¼ - ì†�ì‹¤(loss): {loss:.4f}, ì •í™•ë�„(accuracy): {accuracy:.4f}")


# ë°°ì¹˜ì—�ì„œ ì�´ë¯¸ì§€ì™€ ë�¼ë²¨ ê°€ì ¸ì˜¤ê¸°
image_batch, label_batch = next(iter(validation_data))

# 9ê°œ ì�´ë¯¸ì§€ ì¶œë ¥
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i])
    ans = "dog" if label_batch[i] else "cat"
    model_output = model(image_batch[i:i+1])
    pred = "dog" if (model_output[0][0] >= 0.5) else "cat"
    plt.title(f"Predict: {pred}({round(float(model_output[0][0]), 2)}), Answer: {ans}")  # ë�¼ë²¨ í‘œì‹œ
    plt.axis("off")
plt.show()


test_data = image_dataset_from_directory(
    "test/",
    labels=None,
    label_mode="binary",
    image_size=IMAGE_SIZE,
    shuffle=False
).map(lambda x : x/255.0)


preds = model.predict(test_data)  
pred_list = preds.flatten() 

print(pred_list)


import matplotlib.pyplot as plt

# ì²« ë²ˆì§¸ ë°°ì¹˜ ê°€ì ¸ì˜¤ê¸°
iterator = iter(test_data)
first_batch = next(iterator)

# 32ê°œ ì�´ë¯¸ì§€ ì¶œë ¥
plt.figure(figsize=(10, 20))
for i in range(32):
    plt.subplot(8, 4, i + 1)
    plt.imshow(first_batch[i])  # ì�´ë¯¸ì§€ ë³€í™˜
    plt.title(f"Predict: {'dog' if int(round(pred_list[i])) else 'cat'} ({round(float(pred_list[i]), 2)})")  # ë�¼ë²¨ í‘œì‹œ
    plt.axis("off")
plt.show()


filenames = sorted(os.listdir("test/"))  
submission_df = pd.DataFrame({
    "id": [int(f.split(".")[0]) for f in filenames],  # íŒŒì�¼ ì�´ë¦„ì—�ì„œ ID ì¶”ì¶œ
    "label": pred_list  # ì˜ˆì¸¡ ê²°ê³¼ 
})

submission_df


submission_df.to_csv("submission.csv", index=False)

