# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import numpy as np              # For numerical operations
import pandas as pd             # For working with CSVs and DataFrames
import matplotlib.pyplot as plt # For plotting
import cv2                      # For image processing
import seaborn as sns           # For nicer plots
from sklearn.utils import shuffle                # To shuffle datasets
from sklearn.metrics import confusion_matrix     # To evaluate classification results
from sklearn.model_selection import train_test_split # For splitting data into train/test
import itertools               # Useful for looping combinations (e.g., confusion matrix plotting)
import shutil                  # File operations like copy, move


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os

data_dir = "/kaggle/input/histopathologic-cancer-detection"

print(os.listdir(data_dir))

# Total Samples Available
print('Train Images =', len(os.listdir(os.path.join(data_dir, 'train'))))
print('Test Images =', len(os.listdir(os.path.join(data_dir, 'test'))))

# Read train labels CSV
df = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))
print('Shape of DataFrame:', df.shape)
df.head()



TRAIN_DIR = '/kaggle/input/histopathologic-cancer-detection/train/'



fig = plt.figure(figsize = (20,8))
index = 1
for i in np.random.randint(low = 0, high = df.shape[0], size = 10):
    file = TRAIN_DIR + df.iloc[i]['id'] + '.tif'
    img = cv2.imread(file)
    ax = fig.add_subplot(2, 5, index)
    ax.imshow(img, cmap = 'gray')
    index = index + 1
    color = ['green' if df.iloc[i].label == 1 else 'red'][0]
    ax.set_title(df.iloc[i].label, fontsize = 18, color = color)
plt.tight_layout()
plt.show()


# Removing problematic images
df = df[df['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']  # corrupted image
df = df[df['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']  # black image

print(df.head())


labels_count = df.label.value_counts()

plt.pie(labels_count, labels=['No Cancer', 'Cancer'], startangle=180, 
        autopct='%1.1f', colors=['#00ff99','#FF96A7'], shadow=True)
plt.figure(figsize=(16,16))
plt.show()


SAMPLE_SIZE = 80000
# take a random sample of class 0 with size equal to num samples in class 1
df_0 = df[df['label'] == 0].sample(SAMPLE_SIZE, random_state = 0)
# filter out class 1
df_1 = df[df['label'] == 1].sample(SAMPLE_SIZE, random_state = 0)


# concat the dataframes
df_train = pd.concat([df_0, df_1], axis = 0).reset_index(drop = True)
# shuffle
df_train = shuffle(df_train)

print(df_train['label'].value_counts())


from sklearn.model_selection import train_test_split
import os

# Target labels for stratification
y = df_train['label']

# Stratified split
df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=0, stratify=y)

# Update base_dir path for Kaggle writable directory
base_dir = '/kaggle/working/base_dir'
train_dir = os.path.join(base_dir, 'train_dir')
val_dir = os.path.join(base_dir, 'val_dir')


import os

base_dir = '/kaggle/working/base_dir'
os.makedirs(base_dir, exist_ok=True)  # Create base_dir if it doesn't exist

train_dir = os.path.join(base_dir, 'train_dir')
os.makedirs(train_dir, exist_ok=True)

val_dir = os.path.join(base_dir, 'val_dir')
os.makedirs(val_dir, exist_ok=True)

# Create class subfolders inside train_dir
os.makedirs(os.path.join(train_dir, '0'), exist_ok=True)
os.makedirs(os.path.join(train_dir, '1'), exist_ok=True)

# Create class subfolders inside val_dir
os.makedirs(os.path.join(val_dir, '0'), exist_ok=True)
os.makedirs(os.path.join(val_dir, '1'), exist_ok=True)


print(os.listdir('/kaggle/working/base_dir/train_dir'))
print(os.listdir('/kaggle/working/base_dir/val_dir'))


df.set_index('id', inplace=True)

train_list = list(df_train['id'])
val_list = list(df_val['id'])


data_dir = '/kaggle/input/histopathologic-cancer-detection'

for image in train_list:
    file_name = image + '.tif'
    target = df.loc[image, 'label']

    label = '0' if target == 0 else '1'

    src = os.path.join(data_dir, 'train', file_name)  # Kaggle input path
    dest = os.path.join(train_dir, label, file_name)  # Kaggle working path

    shutil.copyfile(src, dest)

for image in val_list:
    file_name = image + '.tif'
    target = df.loc[image, 'label']

    label = '0' if target == 0 else '1'

    src = os.path.join(data_dir, 'train', file_name)
    dest = os.path.join(val_dir, label, file_name)

    shutil.copyfile(src, dest)

print(len(os.listdir(os.path.join(train_dir, '0'))))
print(len(os.listdir(os.path.join(train_dir, '1'))))


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

IMAGE_SIZE = 96

train_path = '/kaggle/working/base_dir/train_dir'
valid_path = '/kaggle/working/base_dir/val_dir'
test_path = '/kaggle/input/histopathologic-cancer-detection/test'

num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 32
val_batch_size = 32

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)

datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = datagen.flow_from_directory(
    train_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=train_batch_size,
    class_mode='categorical'
)

val_gen = datagen.flow_from_directory(
    valid_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=val_batch_size,
    class_mode='categorical'
)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, SeparableConv2D, Activation


class Net:
    @staticmethod
    def build(width, height, depth, classes):
            
            #initializa model
            model = Sequential()
            
            inputShape = (height, width, depth)
            
            #Add First Layer CONV => ReLU => Pooling
            model.add(Conv2D(filters = 32, kernel_size = (5,5), padding="same", activation='relu', input_shape= inputShape))
            model.add(Conv2D(filters = 32, kernel_size = (3,3), padding="same", activation='relu'))
            model.add(Conv2D(filters = 32, kernel_size = (3,3), padding="same", activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))
            
            #Add Second Layer CONV => ReLU => Pooling
            model.add(Conv2D(filters = 64, kernel_size = (3,3), padding="same", activation='relu'))
            model.add(Conv2D(filters = 64, kernel_size = (3,3), padding="same", activation='relu'))
            model.add(Conv2D(filters = 64, kernel_size = (3,3), padding="same", activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))
            
            #Add Third Layer CONV => ReLU => Pooling
            model.add(Conv2D(filters = 128, kernel_size = (3,3), padding="same", activation='relu'))
            model.add(Conv2D(filters = 128, kernel_size = (3,3), padding="same", activation='relu'))
            model.add(Conv2D(filters = 128, kernel_size = (3,3), padding="same", activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            
            #FC => ReLU
            model.add(Flatten())
            model.add(Dense(units = 500, activation = 'relu'))
            model.add(Dropout(0.2))
            #FC => Output
            model.add(Dense(classes, activation='softmax'))
            
            model.summary()
            
            return model


model = Net.build(width = 96, height = 96, depth = 3, classes = 2)


from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])



from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

filepath = "checkpoint.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, 
                              verbose=1, mode='max', min_lr=1e-5)

callbacks_list = [checkpoint, reduce_lr]

train_steps = int(np.ceil(num_train_samples / train_batch_size))
val_steps = int(np.ceil(num_val_samples / val_batch_size))

history = model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=11,
    verbose=1,
    callbacks=callbacks_list
)


# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])       # instead of 'acc'
plt.plot(history.history['val_accuracy'])   # instead of 'val_acc'
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()


# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()


from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = 96
val_batch_size = 32
valid_path = '/kaggle/working/base_dir/val_dir'

# Rescale pixel values
datagen = ImageDataGenerator(rescale=1.0 / 255)

# Define the validation generator
val_gen = datagen.flow_from_directory(
    valid_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=val_batch_size,
    class_mode='categorical',
    shuffle=False  # IMPORTANT for correct label alignment
)



# Load best weights
model.load_weights('checkpoint.h5')

# Evaluate using the generator
val_loss, val_acc = model.evaluate(val_gen, steps=val_steps, verbose=1)
print('val_loss:', val_loss)
print('val_acc:', val_acc)

# Predict using the generator
predictions = model.predict(val_gen, steps=val_steps, verbose=1)

# Convert predictions to DataFrame
df_preds = pd.DataFrame(predictions, columns=['no_tumor', 'has_tumor'])
df_preds.head()



# y_true: True labels (0 or 1)
y_true = val_gen.classes  # This is an array like [0, 1, 0, 0, 1, ...]

# y_pred: Predicted probabilities for class 1 (has_tumor)
# predictions is of shape (num_samples, 2) from model.predict
y_pred = predictions[:, 1]  # Probability of 'has_tumor'

from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# Compute ROC AUC Score
roc_auc = roc_auc_score(y_true, y_pred)
print('ROC AUC Score =', roc_auc)

# Compute ROC Curve values
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc_val = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_val:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()



from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

y_pred_binary = predictions.argmax(axis=1)
cm = confusion_matrix(y_true, y_pred_binary)

fig, ax = plot_confusion_matrix(conf_mat=cm,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True,
                                cmap='Dark2')
plt.show()

report = classification_report(y_true, y_pred_binary, target_names=['no_tumor', 'has_tumor'])
print(report)


