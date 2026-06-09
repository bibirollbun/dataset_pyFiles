import os, cv2, random
import numpy as np
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import to_categorical


TRAIN_DIR       = '/kaggle/working/train/'
TEST_DIR        = '/kaggle/working/test/'
TRAIN_ZIP_PATH  = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip'
TEST_ZIP_PATH   = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip'

with zipfile.ZipFile(TRAIN_ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(TRAIN_DIR)

_subdirs = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
if len(_subdirs) == 1:
    TRAIN_DIR = os.path.join(TRAIN_DIR, _subdirs[0]) + '/'

with zipfile.ZipFile(TEST_ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(TEST_DIR)

_subdirs = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
if len(_subdirs) == 1:
    TEST_DIR = os.path.join(TEST_DIR, _subdirs[0]) + '/'

train_files = [
    f for f in os.listdir(TRAIN_DIR)
    if os.path.isfile(os.path.join(TRAIN_DIR, f))
]
test_files = [
    f for f in os.listdir(TEST_DIR)
    if os.path.isfile(os.path.join(TEST_DIR, f))
]

train_dogs   = [(os.path.join(TRAIN_DIR, f), 1) for f in train_files if 'dog' in f]
train_cats   = [(os.path.join(TRAIN_DIR, f), 0) for f in train_files if 'cat' in f]
test_images  = [(os.path.join(TEST_DIR, f), -1) for f in test_files]

all_train = train_dogs + train_cats
train_images, val_images = train_test_split(
    all_train,
    train_size=200,     
    stratify=[lbl for _, lbl in all_train],  
    random_state=42
)
random.shuffle(train_images)
random.shuffle(val_images)

test_images = [(os.path.join(TEST_DIR, f), -1)
               for f in os.listdir(TEST_DIR)
               if os.path.isfile(os.path.join(TEST_DIR, f))]


# To ensure uniform image standards, we resize all the images to 64*64

ROWS = 64
COLS = 64

def read_image(tuple_set):
    file_path, label = tuple_set
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    return img, label


# Preprocess the image and turn it into a numpy array

CHANNELS = 3 # RGB

def prep_data(images):
    no_images = len(images)
    data   = np.ndarray((no_images, CHANNELS, ROWS, COLS), dtype=np.uint8)
    labels = []

    for i, image_file in enumerate(images):
        image, label = read_image(image_file)
        data[i] = image.transpose(2, 0, 1)
        labels.append(label)

    return data, labels


x_train, y_train = prep_data(train_images)
x_test, y_t = prep_data(test_images) # y_t is of no use. We don't know the label of the test set

y_train = np.array(y_train)
y_t     = np.array(y_t)


print(x_train.shape)
print(x_test.shape)


# CNN Construction (VGG CNN)

optimizer = RMSprop(learning_rate = 1e-4)
objective = 'binary_crossentropy'

model = Sequential()

model.add(Conv2D(32, (3, 3),
                 padding='same',
                 data_format='channels_first',
                 input_shape=(3, ROWS, COLS),
                 activation='relu'))
model.add(Conv2D(32, (3, 3),
                 padding='same',
                 data_format='channels_first',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first',))

model.add(Conv2D(64, (3, 3), padding='same', data_format='channels_first', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', data_format='channels_first', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first',))

model.add(Conv2D(128, (3, 3), padding='same', data_format='channels_first', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', data_format='channels_first', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first',))

model.add(Conv2D(256, (3, 3), padding='same', data_format='channels_first', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', data_format='channels_first', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first',))

model.add(Flatten(data_format='channels_first'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])


# train and forecast

nb_epoch = 10
batch_size = 10

# Write a class to store the loss after each epoch, which is convenient for drawing graphs
class LossHistory(Callback):
    def on_train_begin(self, logs = {}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs = {}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get(('val_loss')))

# overfitting is very easy in image processing, we adopt the early stop mechanism that comes with keras
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3, verbose = 1, mode = 'auto')

# run the model
history = LossHistory()

model.fit(x_train, y_train, batch_size = batch_size, epochs = nb_epoch, 
          validation_split = 0.2, verbose = 0, shuffle = True, callbacks = [history, early_stopping])

predictions = model.predict(x_test, verbose = 0)


loss = history.losses
val_loss = history.val_losses

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('VGG-16 Loss Trend')
plt.plot(loss, 'blue', label = 'Training Loss')
plt.plot(val_loss, 'green', label = 'Validation Loss')
plt.xticks(range(0, nb_epoch)[0::2])
plt.legend()
plt.show()


# Submit the result

preds = predictions.reshape(-1)

image_ids = [
    int(os.path.splitext(os.path.basename(path))[0]) 
    for path, _ in test_images
]

submission = pd.DataFrame({
    'id': image_ids,  
    'label': preds
})

submission_path = '/kaggle/working/submission.csv'
submission.to_csv(submission_path, index=False)

print(f"Submission saved to {submission_path}")




