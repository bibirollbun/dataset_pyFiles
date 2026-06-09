# import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os, glob, cv2
from PIL import Image, ImageFile 
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.metrics import Recall
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


# import filepaths
root_dir = '../input/intel-mobileodt-cervical-cancer-screening'
train_dir = os.path.join(root_dir,'train', 'train')

type1_dir = os.path.join(train_dir, 'Type_1')
type2_dir = os.path.join(train_dir, 'Type_2')
type3_dir = os.path.join(train_dir, 'Type_3')

train_type1_files = glob.glob(type1_dir+'/*.jpg')
train_type2_files = glob.glob(type2_dir+'/*.jpg')
train_type3_files = glob.glob(type3_dir+'/*.jpg')

added_type1_files = glob.glob(os.path.join(root_dir, "additional_Type_1_v2", "Type_1")+'/*.jpg')
added_type2_files = glob.glob(os.path.join(root_dir, "additional_Type_2_v2", "Type_2")+'/*.jpg')
added_type3_files = glob.glob(os.path.join(root_dir, "additional_Type_3_v2", "Type_3")+'/*.jpg')


type1_files = train_type1_files + added_type1_files
type2_files = train_type2_files + added_type2_files
type3_files = train_type3_files + added_type3_files

print(f'''Type 1 files for training: {len(type1_files)} 
Type 2 files for training: {len(type2_files)}
Type 3 files for training: {len(type3_files)}''')


# get data for testing

test_dir = os.path.join(root_dir,'test', 'test')

test_files = glob.glob(test_dir+'/*.jpg')

print(f'Test files for training: {len(test_files)}')


# create dataframe of file and labels
files = {'filepath': type1_files + type2_files + type3_files,
          'label': ['Type 1']* len(type1_files) + ['Type 2']* len(type2_files) + ['Type 3']* len(type3_files)}

files_df = pd.DataFrame(files).sample(frac=1, random_state= 1).reset_index(drop=True)
files_df.head()


files_df.describe()


# check for damaged files
bad_files = []
for path in (files_df['filepath'].values):
    try:
        img = Image.open(path)
    except:
        index = files_df[files_df['filepath']==path].index.values[0]
        bad_files.append(index)
print(len(bad_files))


# drop the damaged files
files_df.drop(bad_files, inplace=True)


# get count of each type 
type_count = pd.DataFrame(files_df['label'].value_counts(normalize=True)*100)
type_count


# display barplot of type count
plt.figure(figsize = (15, 6))
sns.barplot(x= type_count['proportion'], y= type_count.index.to_list())
plt.title('Cervical Cancer Type Distribution')
plt.grid(True)
plt.show()


# display sample images of types
for label in ('Type 1', 'Type 2', 'Type 3'):
    filepaths = files_df[files_df['label']==label]['filepath'].values[:5]
    fig = plt.figure(figsize= (15, 6))
    for i, path in enumerate(filepaths):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (224, 224))
        fig.add_subplot(1, 5, i+1)
        plt.imshow(img)
        plt.subplots_adjust(hspace=0.5)
        plt.axis(False)
        plt.title(label)


#  split the data into train  and validation set
train_df, eval_df = train_test_split(files_df, test_size= 0.2, stratify= files_df['label'], random_state= 1)
val_df, test_df = train_test_split(eval_df, test_size= 0.5, stratify= eval_df['label'], random_state= 1)
print(len(train_df), len(val_df), len(test_df))


# loads images from dataframe
def load_images(dataframe):
    features = []
    filepaths = dataframe['filepath'].values
    labels = dataframe['label'].values
    
    for path in filepaths:
        img = cv2.imread(path)
        resized_img = cv2.resize(img, (180, 180))
        features.append(np.array(resized_img))
    return np.array(features), np.array(labels)


# load training and evaluation data
train_features, train_labels = load_images(train_df)
val_features, val_labels = load_images(val_df)
test_features, test_labels = load_images(test_df)


# check lengths of training and evaluation sets
print(f'''train features:{len(train_features)},train labels:{len(train_labels)}
    val features:{len(val_features)}, val labels:{len(val_labels)}
    test features:{len(test_features)}, test labels:{len(test_labels)}''') 


# get image shape
InputShape = train_features[766].shape
print(InputShape)


# encode the labels
le = LabelEncoder().fit(['Type 1', 'Type 2', 'Type 3'])
y_train = le.transform(train_labels)
y_val = le.transform(val_labels)
y_test = le.transform(test_labels)


# normalize the features
X_train = train_features/255
X_val  = val_features/255
X_test  = test_features/255


conv_base = VGG16(weights= 'imagenet',
                  include_top= False,
                  input_shape= (180, 180, 3))


# Load the ResNet50 base model with ImageNet weights, excluding the top classifier layers
conv_base_2 = ResNet50(weights='imagenet',
                       include_top=False,
                       input_shape=(180, 180, 3))


for layer in conv_base.layers[:-5]:
    layer.trainable= False


# Using VGG16 model
conv_base.trainable = False

model = Sequential([conv_base,
                   Flatten(),
                    Dense(180, activation='relu'),
                    Dropout(0.5),
                    Dense(3, activation='softmax')
                   ])

model.compile(optimizer= Adam(0.0001),
              loss= 'sparse_categorical_crossentropy',
              metrics= ["accuracy"]
             )
model.summary()


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True)


history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks= [early_stopping])


# Using ResNet50 model
for layer in conv_base.layers[:-5]:
    layer.trainable= False

conv_base_2.trainable = False

model_2 = Sequential([conv_base_2,
                      Flatten(),
                     Dense(180, activation='relu'),
                      Dropout(0.5),
                     Dense(3, activation='softmax')
                     ])

model_2.compile(optimizer=Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
             )
model_2.summary()


# Train the model
history_2 = model_2.fit(X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping]
)


# read training history into dataframe
history_df = pd.DataFrame(history.history)

history_df_2 = pd.DataFrame(history_2.history)


# display training and validation history

# display history of accuracy
plt.figure(figsize= (15,6))
plt.subplot(1,2,1)
plt.plot(history_df['accuracy'], label= 'accracy' )
plt.plot(history_df['val_accuracy'], label= 'val_accuracy')
# history_df[['accuracy', 'val_accuracy']]
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy History')
plt.legend()

# display history of loss
plt.subplot(1,2,2)
plt.plot(history_df['loss'], label= 'loss')
plt.plot(history_df['val_loss'], label= 'val_loss')
# history_df[['loss', 'val_loss']].plot()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss History')
plt.legend()


# display training and validation history

# display history of accurracy
plt.figure(figsize= (15,6))
plt.subplot(1,2,1)
plt.plot(history_df_2['accuracy'], label= 'accuracy' )
plt.plot(history_df_2['val_accuracy'], label= 'val_accuracy')
# history_df[['accuracy', 'val_acc']]
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy History')
plt.legend()

# display history of loss
plt.subplot(1,2,2)
plt.plot(history_df_2['loss'], label= 'loss')
plt.plot(history_df_2['val_loss'], label= 'val_loss')
# history_df[['loss', 'val_loss']].plot()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss History')
plt.legend()


model.evaluate(X_test, y_test)
model_2.evaluate(X_test, y_test)

