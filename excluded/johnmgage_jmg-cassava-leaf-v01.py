import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import seaborn as sns
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from sklearn.utils.class_weight import compute_class_weight


cassava_leaf_train = pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/train.csv')
cassava_leaf_train.shape


cassava_leaf_train.head()


cassava_leaf_proportion = (cassava_leaf_train.label.value_counts() / len(cassava_leaf_train)).to_frame()
cassava_leaf_proportion


cassava_leaf_train['label'].value_counts()


im_16 = cassava_leaf_train.sample(n=16).reset_index()

plt.figure(figsize=(6,6))
for m, row in im_16.iterrows():
    img = mpimg.imread(f'/kaggle/input/cassava-leaf-disease-classification/train_images/{row.image_id}')
    label = row.label

    plt.subplot(4,4,m+1)
    plt.imshow(img)
    plt.text(0, -5, f'Class {label}', color='k')

    plt.axis('off')
plt.tight_layout()
plt.show()


cassava_leaf_train.isna().sum(axis=0).to_frame


cassava_classes = cassava_leaf_train['label'].value_counts()
print(cassava_classes)
cassava_classes.plot(kind="bar")
plt.xlabel('label')
plt.show()


cassava_leaf_train_3 = cassava_leaf_train.loc[cassava_leaf_train['label'] == 3]
print("Class 3 Before: ", cassava_leaf_train_3.shape)
cassava_leaf_train_3 = cassava_leaf_train_3.drop(cassava_leaf_train_3.sample(frac=0.91).index)
print("Class 3 After: ", cassava_leaf_train_3.shape)

cassava_leaf_train_4 = cassava_leaf_train.loc[cassava_leaf_train['label'] == 4]
print("Class 4 Before: ", cassava_leaf_train_4.shape)
cassava_leaf_train_4 = cassava_leaf_train_4.drop(cassava_leaf_train_4.sample(frac=0.53).index)
print("Class 4 After: ", cassava_leaf_train_4.shape)

cassava_leaf_train_1 = cassava_leaf_train.loc[cassava_leaf_train['label'] == 1]
print("Class 1 Before: ", cassava_leaf_train_1.shape)
cassava_leaf_train_1 = cassava_leaf_train_1.drop(cassava_leaf_train_1.sample(frac=0.53).index)
print("Class 1 After: ", cassava_leaf_train_1.shape)

cassava_leaf_train_2 = cassava_leaf_train.loc[cassava_leaf_train['label'] == 2]
print("Class 2 Before: ", cassava_leaf_train_2.shape)
cassava_leaf_train_2 = cassava_leaf_train_2.drop(cassava_leaf_train_2.sample(frac=0.53).index)
print("Class 2 After: ", cassava_leaf_train_2.shape)


usamp_cassava_leaf_train = pd.concat([cassava_leaf_train_3, cassava_leaf_train_4, cassava_leaf_train_2, cassava_leaf_train_1, cassava_leaf_train.loc[cassava_leaf_train['label'] == 0]], axis=0)
print(usamp_cassava_leaf_train.shape)
usamp_cassava_leaf_proportion = (usamp_cassava_leaf_train.label.value_counts() / len(usamp_cassava_leaf_train)).to_frame()
usamp_cassava_leaf_proportion


usamp_cassava_leaf_train['label'] = usamp_cassava_leaf_train['label'].astype(str)
usamp_train_cassava_df, usamp_valid_cassava_df = train_test_split(usamp_cassava_leaf_train, test_size=0.2, random_state=1, stratify=usamp_cassava_leaf_train.label)
print(usamp_train_cassava_df.shape)
print(usamp_valid_cassava_df.shape)


usamp_train_dg = ImageDataGenerator(rescale=1/255)
usamp_valid_dg = ImageDataGenerator(rescale=1/255)


usamp_train_loader = usamp_train_dg.flow_from_dataframe(
    dataframe = usamp_train_cassava_df,
    directory = '/kaggle/input/cassava-leaf-disease-classification/train_images',
    x_col = 'image_id',
    y_col = 'label',
    batch_size = 64,
    seed = 1,
    shuffle = True,
    class_mode = 'categorical',
    target_size = (32,32)
)

usamp_valid_loader = usamp_valid_dg.flow_from_dataframe(
    dataframe = usamp_valid_cassava_df,
    directory = '/kaggle/input/cassava-leaf-disease-classification/train_images',
    x_col = 'image_id',
    y_col = 'label',
    batch_size = 64,
    seed = 1,
    shuffle = True,
    class_mode = 'categorical',
    target_size = (32,32)
)


usamp_train_steps = len(usamp_train_loader)
usamp_valid_steps = len(usamp_valid_loader)

print(usamp_train_steps)
print(usamp_valid_steps)


np.random.seed(1)
tf.random.set_seed(1)

usamp_cassava_model = Sequential([
    Conv2D(80, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    Conv2D(80, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Dropout(0.5),
    BatchNormalization(),

    Conv2D(160, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    Conv2D(160, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Dropout(0.5),
    BatchNormalization(),

    Flatten(),

    Dense(40, activation='relu'),
    Dropout(0.5),
    Dense(20, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(5, activation='softmax')
])

usamp_cassava_model.summary()


usamp_cassava_optimizer = tf.keras.optimizers.Adam(0.001)
usamp_cassava_model.compile(loss='categorical_crossentropy', optimizer=usamp_cassava_optimizer, metrics=['accuracy', tf.keras.metrics.AUC()])


%%time

usamp_h1 = usamp_cassava_model.fit(
    x = usamp_train_loader,
    steps_per_epoch = usamp_train_steps,
    epochs = 20,
    validation_data = usamp_valid_loader,
    validation_steps = usamp_valid_steps,
    verbose = 1
)


training_hist = pd.DataFrame(usamp_h1.history)
training_hist['epoch'] = usamp_h1.epoch
sns.lineplot(x='epoch', y='loss', data=training_hist)
sns.lineplot(x='epoch', y='val_loss', data=training_hist)
plt.legend(labels=['train_loss', 'val_loss'])


sns.lineplot(x='epoch', y='accuracy', data=training_hist)
sns.lineplot(x='epoch', y='val_accuracy', data=training_hist)
plt.legend(labels=['train_accuracy', 'val_accuracy'])


cassava_leaf_train_3 = cassava_leaf_train.loc[cassava_leaf_train['label'] == 3]
print("Class 3: ", cassava_leaf_train_3.shape)

cassava_leaf_train_4 = cassava_leaf_train.loc[cassava_leaf_train['label'] == 4]
print("Class 4 Before: ", cassava_leaf_train_4.shape)
cassava_leaf_train_4 = cassava_leaf_train_4.loc[cassava_leaf_train_4.index.repeat(5)].reset_index(drop=True)
print("Class 4 After: ", cassava_leaf_train_4.shape)

cassava_leaf_train_2 = cassava_leaf_train.loc[cassava_leaf_train['label'] == 2]
print("Class 2 Before: ", cassava_leaf_train_2.shape)
cassava_leaf_train_2 = cassava_leaf_train_2.loc[cassava_leaf_train_2.index.repeat(5)].reset_index(drop=True)
print("Class 2 After: ", cassava_leaf_train_2.shape)

cassava_leaf_train_1 = cassava_leaf_train.loc[cassava_leaf_train['label'] == 1]
print("Class 1 Before: ", cassava_leaf_train_1.shape)
cassava_leaf_train_1 = cassava_leaf_train_1.loc[cassava_leaf_train_1.index.repeat(6)].reset_index(drop=True)
print("Class 1 After: ", cassava_leaf_train_1.shape)

cassava_leaf_train_0 = cassava_leaf_train.loc[cassava_leaf_train['label'] == 0]
print("Class 0 Before: ", cassava_leaf_train_0.shape)
cassava_leaf_train_0 = cassava_leaf_train_0.loc[cassava_leaf_train_0.index.repeat(11)].reset_index(drop=True)
print("Class 0 After: ", cassava_leaf_train_0.shape)


osamp_cassava_leaf_train = pd.concat([cassava_leaf_train_3, cassava_leaf_train_4, cassava_leaf_train_2, cassava_leaf_train_1, cassava_leaf_train_0], axis=0)
print(osamp_cassava_leaf_train.shape)
osamp_cassava_leaf_proportion = (osamp_cassava_leaf_train.label.value_counts() / len(osamp_cassava_leaf_train)).to_frame()
osamp_cassava_leaf_proportion


osamp_cassava_leaf_train['label'] = osamp_cassava_leaf_train['label'].astype(str)
osamp_train_cassava_df, osamp_valid_cassava_df = train_test_split(osamp_cassava_leaf_train, test_size=0.2, random_state=1, stratify=osamp_cassava_leaf_train.label)
print(osamp_train_cassava_df.shape)
print(osamp_valid_cassava_df.shape)


osamp_train_dg = ImageDataGenerator(rescale=1/255, validation_split=0.2)
osamp_valid_dg = ImageDataGenerator(rescale=1/255)


osamp_train_loader = osamp_train_dg.flow_from_dataframe(
    dataframe = osamp_train_cassava_df,
    directory = '/kaggle/input/cassava-leaf-disease-classification/train_images',
    x_col = 'image_id',
    y_col = 'label',
    batch_size = 64,
    seed = 1,
    shuffle = True,
    class_mode = 'categorical',
    target_size = (75,100)
)

osamp_valid_loader = osamp_valid_dg.flow_from_dataframe(
    dataframe = osamp_valid_cassava_df,
    directory = '/kaggle/input/cassava-leaf-disease-classification/train_images',
    x_col = 'image_id',
    y_col = 'label',
    batch_size = 64,
    seed = 1,
    shuffle = True,
    class_mode = 'categorical',
    target_size = (75,100)
)


osamp_train_steps = len(osamp_train_loader)
osamp_valid_steps = len(osamp_valid_loader)

print(osamp_train_steps)
print(osamp_valid_steps)


np.random.seed(1)
tf.random.set_seed(1)

osamp_cassava_model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(75,100,3)),
    MaxPooling2D(2,2),
    Dropout(0.5),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(75,100,3)),
    MaxPooling2D(2,2),
    Dropout(0.5),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(75,100,3)),
    Flatten(),
    Dense(64, activation='softmax'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(5, activation='softmax')
])

osamp_cassava_model.summary()


osamp_cassava_optimizer = tf.keras.optimizers.Adam(0.0001)
osamp_cassava_model.compile(loss='categorical_crossentropy', optimizer=osamp_cassava_optimizer, metrics=['accuracy', tf.keras.metrics.AUC()])


%%time

osamp_h1 = osamp_cassava_model.fit(
    x = osamp_train_loader,
    steps_per_epoch = osamp_train_steps,
    epochs = 25,
    validation_data = osamp_valid_loader,
    validation_steps = osamp_valid_steps,
    verbose = 1
)


training_hist = pd.DataFrame(osamp_h1.history)
training_hist['epoch'] = osamp_h1.epoch
sns.lineplot(x='epoch', y='loss', data=training_hist)
sns.lineplot(x='epoch', y='val_loss', data=training_hist)
plt.legend(labels=['train_loss', 'val_loss'])


sns.lineplot(x='epoch', y='accuracy', data=training_hist)
sns.lineplot(x='epoch', y='val_accuracy', data=training_hist)
plt.legend(labels=['train_accuracy', 'val_accuracy'])


csl_cassava_leaf_train = cassava_leaf_train
csl_cassava_leaf_train['label'] = csl_cassava_leaf_train['label'].astype(str)


csl_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    #classes=np.unique(csl_cassava_leaf_train.iloc[:,1].values.flatten()),
    #y=csl_cassava_leaf_train.iloc[:,1].values.flatten()
    classes=np.unique(csl_cassava_leaf_train.iloc[:,1].to_numpy()),
    y=csl_cassava_leaf_train.iloc[:,1].to_numpy()
)

csl_weights_dict = dict(enumerate(csl_weights))
print("Weights:", csl_weights_dict)


csl_train_cassava_df, csl_valid_cassava_df = train_test_split(csl_cassava_leaf_train, test_size=0.2, random_state=1, stratify=csl_cassava_leaf_train.label)
print(csl_train_cassava_df.shape)
print(csl_valid_cassava_df.shape)


csl_train_dg = ImageDataGenerator(rescale=1/255, validation_split=0.20)
csl_valid_dg = ImageDataGenerator(rescale=1/255)


csl_train_loader = csl_train_dg.flow_from_dataframe(
    dataframe = csl_train_cassava_df,
    directory = '/kaggle/input/cassava-leaf-disease-classification/train_images',
    x_col = 'image_id',
    y_col = 'label',
    batch_size = 64,
    seed = 1,
    shuffle = True,
    class_mode = 'categorical',
    target_size = (75,100)
)

csl_valid_loader = csl_valid_dg.flow_from_dataframe(
    dataframe = csl_valid_cassava_df,
    directory = '/kaggle/input/cassava-leaf-disease-classification/train_images',
    x_col = 'image_id',
    y_col = 'label',
    batch_size = 64,
    seed = 1,
    shuffle = True,
    class_mode = 'categorical',
    target_size = (75,100)
)


csl_train_steps = len(csl_train_loader)
csl_valid_steps = len(csl_valid_loader)

print(csl_train_steps)
print(csl_valid_steps)


np.random.seed(1)
tf.random.set_seed(1)

csl_cassava_model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(75,100,3)),
    MaxPooling2D(2,2),
    Dropout(0.5),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(75,100,3)),
    MaxPooling2D(2,2),
    Dropout(0.5),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(75,100,3)),
    Flatten(),
    Dense(64, activation='softmax'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(5, activation='softmax')
])

csl_cassava_model.summary()


csl_cassava_optimizer = tf.keras.optimizers.Adam(0.0001)
csl_cassava_model.compile(loss='categorical_crossentropy', optimizer=csl_cassava_optimizer, metrics=['accuracy', tf.keras.metrics.AUC()])


%%time

csl_h1 = csl_cassava_model.fit(
    x = csl_train_loader,
    steps_per_epoch = csl_train_steps,
    epochs = 25,
    validation_data = csl_valid_loader,
    validation_steps = csl_valid_steps,
    verbose = 1,
    class_weight = csl_weights_dict
)


training_hist = pd.DataFrame(csl_h1.history)
training_hist['epoch'] = csl_h1.epoch
sns.lineplot(x='epoch', y='loss', data=training_hist)
sns.lineplot(x='epoch', y='val_loss', data=training_hist)
plt.legend(labels=['train_loss', 'val_loss'])


sns.lineplot(x='epoch', y='accuracy', data=training_hist)
sns.lineplot(x='epoch', y='val_accuracy', data=training_hist)
plt.legend(labels=['train_accuracy', 'val_accuracy'])


from tensorflow.keras import layers
cassava_da = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])


da_model = tf.keras.Seque


#import os
#file_list = os.listdir('/kaggle/input/cassava-leaf-disease-classification/train_images')
#print(file_list[0])
X = []
#print(cassava_leaf_train.iloc[:,0].shape[0])
for i in range(0,cassava_leaf_train.iloc[:,0].shape[0]):
    #print(i)
    #array_entry = np.array([cassava_leaf_train.iloc[i,0], 75, 100, 3])
    array_entry = np.array([i, 75, 100, 3])
    X.append(array_entry)
X = np.array(X)
print(X.shape)
print(X[0])


#!pip uninstall -y scikit-learn imbalanced-learn
#!pip install scikit-learn==1.3.2 imbalanced-learn==0.11.0
#!pip install --upgrade scikit-learn
#!pip install scikit-learn==1.6
#!pip install numpy==2.0
#!pip install autogluon.tabular
#!pip install --upgrade scikit-learn imbalanced-learn
#!pip install scikit-learn==1.3.2
#!pip show scikit-learn
#!pip list scikit-learn
#!pip install scikit-learn
!pip uninstall scikit-learn --yes
!pip uninstall sklearn --yes
!pip uninstall imblearn --yes


!pip install scikit-learn==1.3.2
!pip install imblearn


#!pip show imblearn
#!pip install imblearn==0.14.0
#!pip install -U imbalance-learn
!pip install -U imbalanced-learn


#!pip show scikit-learn
#!pip uninstall -y scikit-learn imbalanced-learn --yes
#!pip install scikit-learn==1.3.2 imbalanced-learn==0.11.0
!pip uninstall scikit-learn -y
#!pip install scikit-learn
!pip install scikit-learn==1.2.2


from imblearn.over_sampling import RandomOverSampler
#X = np.array([cassava_leaf_train.iloc[i,0], 75, 100, 3])
rs_X = X.reshape(X.shape[0], -1)
osamping = RandomOverSampler(sampling_strategy='minority')
osamped_X, osamped_y = osamping.fit_resample(rs_X, cassava_leaf_train.iloc[:,1].to_numpy())
print(osamped_X)

#X_2 = osamped_X.reshape(-1, 75, 100, 3)


#from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

X = cassava_leaf_train.iloc[:,0].to_numpy()
y = cassava_leaf_train.iloc[:,1].to_numpy()

smote = SMOTE(sampling_strategy='minority')
X_rs, y_res = smote.fit_resample(X, y)


da_cassava_leaf_train = cassava_leaf_train
da_cassava_leaf_train['label'] = da_cassava_leaf_train['label'].astype(str)
da_train_cassava_df, da_valid_cassava_df = train_test_split(da_cassava_leaf_train, test_size=0.2, random_state=1, stratify=da_cassava_leaf_train.label)
print(da_train_cassava_df.shape)
print(da_valid_cassava_df.shape)


da_train_dg = ImageDataGenerator(
    rescale=1/255,
    #rotation_range=20,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    #horizontal_flip=True,
    #brightness_range=[0.5, 1.5]
)
da_valid_dg = ImageDataGenerator(
    rescale=1/255,
    #rotation_range=20,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    #horizontal_flip=True,
    #brightness_range=[0.5, 1.5]
)


da_train_loader = da_train_dg.flow_from_dataframe(
    dataframe = da_train_cassava_df,
    directory = '/kaggle/input/cassava-leaf-disease-classification/train_images',
    x_col = 'image_id',
    y_col = 'label',
    batch_size = 64,
    seed = 1,
    shuffle = True,
    class_mode = 'categorical',
    target_size = (75,100)
)

da_valid_loader = da_valid_dg.flow_from_dataframe(
    dataframe = da_valid_cassava_df,
    directory = '/kaggle/input/cassava-leaf-disease-classification/train_images',
    x_col = 'image_id',
    y_col = 'label',
    batch_size = 64,
    seed = 1,
    shuffle = True,
    class_mode = 'categorical',
    target_size = (75,100)
)


da_train_steps = len(da_train_loader)
da_valid_steps = len(da_valid_loader)

print(da_train_steps)
print(da_valid_steps)


from imblearn.over_sampling import SMOTE
#import sklearn
#print(sklearn.__version__)
#!pip install scikit-learn==1.2.2
#!pip install -U imbalanced-learn
#!pip uninstall -y scikit-learn imbalanced-learn --yes
#!pip install scikit-learn==1.3.2 imbalanced-learn==0.11.0
#!pip uninstall scikit-learn -y
#!pip install scikit-learn==1.2.2
#!pip install --upgrade scikit-learn
#!pip install --upgrade imbalanced-learn
#!pip uninstall sklearn -y
#!pip uninstall scikit-learn -y
#!pip install sklearn
#!pip install scikit-learn
#!pip install --upgrade scikit-learn imbalanced-learn mlflow
#!pip install imblearn


'''
!pip uninstall scikit-learn --yes
!pip uninstall sklearn --yes
!pip uninstall imblearn --yes

!pip install scikit-learn==1.3.2
!pip install imblearn

!pip install -U imbalanced-learn

!pip uninstall scikit-learn -y
!pip install scikit-learn==1.2.2

!pip install sklearn
'''
#!pip uninstall imbalanced-learn --yes
#!pip install imbalanced-learn==0.14.0
#!pip show scikit-learn
#!pip uninstall -y scikit-learn imbalanced-learn --yes
#!pip install scikit-learn==1.3.2 imbalanced-learn==0.11.0

!pip show scikit-learn
#!pip show imbalanced-learn
#!pip install --upgrade scikit-learn imbalanced-learn


np.random.seed(1)
tf.random.set_seed(1)

da_cassava_model = Sequential([
    Conv2D(40, (3,3), activation='relu', padding='same', input_shape=(75,100,3)),
    Conv2D(40, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Dropout(0.5),
    BatchNormalization(),
    
    Conv2D(80, (3,3), activation='relu', padding='same', input_shape=(75,100,3)),
    Conv2D(80, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Dropout(0.5),
    BatchNormalization(),
    
    Conv2D(160, (3,3), activation='relu', padding='same', input_shape=(75,100,3)),
    Conv2D(160, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Dropout(0.5),
    BatchNormalization(),

    Flatten(),

    Dense(40, activation='relu'),
    Dropout(0.5),
    Dense(20, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(5, activation='softmax')
])

da_cassava_model.summary()


da_cassava_optimizer = tf.keras.optimizers.Adam(0.0001)
da_cassava_model.compile(loss='categorical_crossentropy', optimizer=da_cassava_optimizer, metrics=['accuracy', tf.keras.metrics.AUC()])


%%time

da_h1 = da_cassava_model.fit(
    x = da_train_loader,
    steps_per_epoch = da_train_steps,
    epochs = 20,
    validation_data = da_valid_loader,
    validation_steps = da_valid_steps,
    verbose = 1
)


training_hist = pd.DataFrame(da_h1.history)
training_hist['epoch'] = da_h1.epoch
sns.lineplot(x='epoch', y='loss', data=training_hist)
sns.lineplot(x='epoch', y='val_loss', data=training_hist)
plt.legend(labels=['train_loss', 'val_loss'])


sns.lineplot(x='epoch', y='accuracy', data=training_hist)
sns.lineplot(x='epoch', y='val_accuracy', data=training_hist)
plt.legend(labels=['train_accuracy', 'val_accuracy'])


#!pip uninstall numpy --yes
#!pip install numpy==2.0
!pip uninstall -y scikit-learn imbalanced-learn
#!pip install scikit-learn==1.3.2 imbalanced-learn==0.11.0
#!pip install --upgrade scikit-learn
#!pip install --upgrade imbalanced-learn
#!pip show scikit-learn
#!pip install --upgrade scikit-learn imbalanced-learn
#!pip install -U imbalanced-learn
#!pip install --upgrade --force-reinstall numpy scipy
!pip install scikit-learn==1.3.2 imbalanced-learn==0.11.0


import imblearn
from imblearn.over_sampling import KMeansSMOTE


wrs_cassava_leaf_train = cassava_leaf_train
wrs_cassava_leaf_train['label'] = wrs_cassava_leaf_train['label'].astype(str)


wrs_class_quantities = cassava_leaf_train['label'].value_counts().sort_index().values
wrs_class_quantities
wrs_weights = 1.0 / wrs_class_quantities
samples = [wrs_weights[label] for label in cassava_leaf_train['label'].tolist()]
sampler = WeightedRandomSampler(weights=samples, num_samples=len(samples), replacement=True)

#train_loader = DataLoader()


wrs_train_cassava_df, wrs_valid_cassava_df = train_test_split(cassava_leaf_train, test_size=0.2, random_state=1, stratify=cassava_leaf_train.label)
print(wrs_train_cassava_df.shape)
print(wrs_valid_cassava_df.shape)


wrs_train_cassava_df['label'] = wrs_train_cassava_df['label'].astype(str)
wrs_valid_cassava_df['label'] = wrs_valid_cassava_df['label'].astype(str)

wrs_train_dg = ImageDataGenerator(rescale=1/255)
wrs_valid_dg = ImageDataGenerator(rescale=1/255)

wrs_train_loader = wrs_train_dg.flow_from_dataframe(
    dataframe = wrs_train_cassava_df,
    directory = '/kaggle/input/cassava-leaf-disease-classification/train_images',
    x_col = 'image_id',
    y_col = 'label',
    batch_size = 64,
    seed = 1,
    shuffle = True,
    class_mode = 'categorical',
    target_size = (75,100)
)

wrs_valid_loader = wrs_valid_dg.flow_from_dataframe(
    dataframe = wrs_valid_cassava_df,
    directory = '/kaggle/input/cassava-leaf-disease-classification/train_images',
    x_col = 'image_id',
    y_col = 'label',
    batch_size = 64,
    seed = 1,
    shuffle = True,
    class_mode = 'categorical',
    target_size = (75,100)
)


#image_dataset = SimpleImageDataset('/kaggle/input/cassava-leaf-disease-classification/train_images')
wrs_train_loaded = DataLoader(wrs_train_loader, batch_size=64, sampler=sampler, num_workers=4)
wrs_valid_loaded = DataLoader(wrs_valid_loader, batch_size=64, sampler=sampler, num_workers=4)
print(type(wrs_train_loaded))
print(type(wrs_train_loader))


cassava_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(cassava_leaf_train.iloc[:,1].to_numpy()),
    y=cassava_leaf_train.iloc[:,1].to_numpy()
)

cassava_weight_dict = dict(enumerate(cassava_weights))
print(cassava_weight_dict)


np.random.seed(1)
tf.random.set_seed(1)

wrs_cassava_model = Sequential([
    Conv2D(160, (3,3), activation='relu', padding='same', input_shape=(75,100,3)),
    Conv2D(160, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Dropout(0.5),
    BatchNormalization(),

    Conv2D(80, (3,3), activation='relu', padding='same', input_shape=(75,100,3)),
    Conv2D(80, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Dropout(0.5),
    BatchNormalization(),

    Flatten(),

    Dense(40, activation='relu'),
    Dropout(0.5),
    Dense(20, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(5, activation='softmax')
])

wrs_cassava_model.summary()


wrs_cassava_optimizer = tf.keras.optimizers.Adam(0.0001)
wrs_cassava_model.compile(loss='categorical_crossentropy', optimizer=wrs_cassava_optimizer, metrics=['accuracy', tf.keras.metrics.AUC()])


%%time

wrs_h1 = wrs_cassava_model.fit(
    x = wrs_train_loaded,
    epochs = 20,
    validation_data = wrs_train_loaded,
    verbose = 1
)


train_dg = ImageDataGenerator(rescale=1/255, validation_split=0.20)
val_dg = ImageDataGenerator(rescale=1/255)


str_cassava_leaf_train = cassava_leaf_train
str_cassava_leaf_train['label'] = str_cassava_leaf_train['label'].astype(str)


csl_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(str_cassava_leaf_train.iloc[:,1].to_numpy()),
    y=str_cassava_leaf_train.iloc[:,1].to_numpy()
)

csl_weights_dict = dict(enumerate(csl_weights))
print("Weights:", csl_weights_dict)


train_loader = train_dg.flow_from_dataframe(
    dataframe = str_cassava_leaf_train,
    directory = '/kaggle/input/cassava-leaf-disease-classification/train_images',
    x_col = 'image_id',
    y_col = 'label',
    batch_size = 64,
    seed = 1,
    shuffle = True,
    class_mode = 'categorical',
    target_size = (75,100)
)

csl_valid_loader = val_dg.flow_from_dataframe(
    dataframe = str_cassava_leaf_train,
    directory = '/kaggle/input/cassava-leaf-disease-classification/train_images',
    x_col = 'image_id',
    y_col = 'label',
    batch_size = 64,
    seed = 1,
    shuffle = True,
    class_mode = 'categorical',
    target_size = (75,100)
)


csl_cassava_optimizer = tf.keras.optimizers.Adam(0.0001)
csl_cassava_model.compile(loss='categorical_crossentropy', optimizer=csl_cassava_optimizer, metrics=['accuracy', tf.keras.metrics.AUC()])


#print(csl_train_cassava_df)
#print(cassava_leaf_train)

fitting = csl_cassava_model.fit(
    train_loader,
    steps_per_epoch=len(train_loader),
    epochs=20,
    validation_data=csl_valid_loader,
    validation_steps=len(csl_valid_loader),
    verbose=1,
    class_weight = csl_weights_dict
)


training_hist = pd.DataFrame(fitting.history)
training_hist['epoch'] = fitting.epoch
sns.lineplot(x='epoch', y='loss', data=training_hist)
sns.lineplot(x='epoch', y='val_loss', data=training_hist)
plt.legend(labels=['train_loss', 'val_loss'])


sns.lineplot(x='epoch', y='accuracy', data=training_hist)
sns.lineplot(x='epoch', y='val_accuracy', data=training_hist)
plt.legend(labels=['train_accuracy', 'val_accuracy'])


import os
import shutil

labels = np.unique(cassava_leaf_train.label)
#print(labels[0])
for label in labels:
    class_folder_name = "class_" + str(label)
    #print(class_folder_name)
    class_folder_path = os.path.join("/kaggle/working/", class_folder_name)
    os.makedirs(class_folder_path, exist_ok=True)
    #print(class_folder_path)

    file_names = cassava_leaf_train.loc[cassava_leaf_train['label'] == label].iloc[:,0].tolist()
    for file_name in file_names:
        check_file_path = str(class_folder_path) + "/" + file_name
        #print(check_file_path)
        if not os.path.isfile(check_file_path):
            #print("Doesn't exist")
            shutil.copy(os.path.join("/kaggle/input/cassava-leaf-disease-classification/train_images/",file_name),class_folder_path)



'''class_zero = []
class_one = []
class_two = []
class_three = []
class_four = []

#for label in lab
shutil.copy(os.path.join("/kaggle/input/cassava-leaf-disease-classification/train_images","1000201771.jpg"),"/kaggle/working/class_0")
'''

for label in labels:
    class_folder_name = "/kaggle/working/class_" + str(label)
    print(class_folder_name)
    print(len(os.listdir(class_folder_name)))


cassava_leaf_train['label'].value_counts()


os.makedirs("/kaggle/working/training_data", exist_ok=True)
shutil.move("/kaggle/working/class_1", "/kaggle/working/training_data")
shutil.move("/kaggle/working/class_2", "/kaggle/working/training_data")
shutil.move("/kaggle/working/class_3", "/kaggle/working/training_data")
shutil.move("/kaggle/working/class_4", "/kaggle/working/training_data")


from torchvision.datasets import ImageFolder
root_folder = "/kaggle/working/training_data"
cassava_dataset = datasets.ImageFolder(root=root_folder)


wrs_class_quantities = cassava_leaf_train['label'].value_counts().sort_index().values
wrs_class_quantities
wrs_weights = 1.0 / wrs_class_quantities
samples = [wrs_weights[label] for label in cassava_leaf_train['label'].tolist()]
sampler = WeightedRandomSampler(weights=samples, num_samples=len(samples), replacement=True)


#train_ds, val_ds = torch.utils.data.random_split(train_loader, [0.8, 0.2])

#train_ds, val_ds = torch.utils.data.random_split(cassava_dataset, [0.8, 0.2])
#train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=4)
#val_loader = DataLoader(val_ds, batch_size=64, num_workers=4)

train_loader = DataLoader(cassava_dataset, batch_size=64, sampler=sampler, num_workers=4)


print(train_loader)
print(val_loader)

train_steps = len(train_loader)
val_steps = len(val_loader)

print(train_steps)
print(val_steps)


np.random.seed(1)
tf.random.set_seed(1)

wrs_model = Sequential([
    Conv2D(40, (3,3), activation='relu', padding='same', input_shape=(75,100,3)),
    Conv2D(40, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Dropout(0.5),
    BatchNormalization(),

    Conv2D(80, (3,3), activation='relu', padding='same', input_shape=(75,100,3)),
    Conv2D(80, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Dropout(0.5),
    BatchNormalization(),

    Conv2D(160, (3,3), activation='relu', padding='same', input_shape=(75,100,3)),
    Conv2D(160, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Dropout(0.5),
    BatchNormalization(),

    Flatten(),

    Dense(40, activation='relu'),
    Dropout(0.5),
    Dense(20, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(5, activation='softmax')
])

wrs_model.summary()


#train_ds, val_ds = torch.utils.data.random_split(train_loader, [17093, 4274])
#print(len(train_loader))
wrs_optimizer = tf.keras.optimizers.Adam(0.0001)
wrs_model.compile(loss='categorical_crossentropy', optimizer=wrs_optimizer, metrics=['accuracy', tf.keras.metrics.AUC()])


%%time

wrs_h1 = wrs_model.fit(
    train_loader,
    #steps_per_epoch = train_steps,
    epochs = 20
    #validation_data = val_loader,
    #validation_steps = val_steps,
    #verbose = 1,
)





class_quantities = cassava_leaf_train['label'].value_counts().to_dict()

#class_quantities = class_quantities = dict(sorted(class_quantities.items()))
#class_quantities
#top_class = max(class_quantities.values())
#top_class
#top_class = class_quantities.max()
#bottom_classes = class_quantities

top_class = max(class_quantities, key=class_quantities.get)
top_entries = class_quantities[top_class]
class_quantities.pop(top_class)
#class_quantities
top_entries



props_list = []
for classs, values in class_quantities.items():
    #print(classs)
    #print(values)
    proportions = top_entries / values
    proportions = int(np.round(proportions,0))
    #print(np.round(proportions,0))
    #print(" ")

    for i in range(0,proportions):
        copied_entries = cassava_leaf_train.loc[cassava_leaf_train['label'] == classs]
        #print(i)
    


csl_osamp_cassava_leaf_train = cassava_leaf_train
csl_osamp_cassava_leaf_train['label'] = csl_osamp_cassava_leaf_train['label'].astype(str)


csl_osamp_cassava_leaf_train_3 = csl_osamp_cassava_leaf_train.loc[csl_osamp_cassava_leaf_train['label'] == "3"]
print("Class 3: ", csl_osamp_cassava_leaf_train_3.shape)

csl_osamp_cassava_leaf_train_4 = csl_osamp_cassava_leaf_train.loc[csl_osamp_cassava_leaf_train['label'] == "4"]
print("Class 4 Before: ", csl_osamp_cassava_leaf_train_4.shape)
csl_osamp_cassava_leaf_train_4 = csl_osamp_cassava_leaf_train_4.loc[csl_osamp_cassava_leaf_train_4.index.repeat(3)].reset_index(drop=True)
print("Class 4 After: ", csl_osamp_cassava_leaf_train_4.shape)

csl_osamp_cassava_leaf_train_2 = csl_osamp_cassava_leaf_train.loc[csl_osamp_cassava_leaf_train['label'] == "2"]
print("Class 2 Before: ", csl_osamp_cassava_leaf_train_2.shape)
csl_osamp_cassava_leaf_train_2 = csl_osamp_cassava_leaf_train_2.loc[csl_osamp_cassava_leaf_train_2.index.repeat(3)].reset_index(drop=True)
print("Class 2 After: ", csl_osamp_cassava_leaf_train_2.shape)

csl_osamp_cassava_leaf_train_1 = csl_osamp_cassava_leaf_train.loc[csl_osamp_cassava_leaf_train['label'] == "1"]
print("Class 1 Before: ", csl_osamp_cassava_leaf_train_1.shape)
csl_osamp_cassava_leaf_train_1 = csl_osamp_cassava_leaf_train_1.loc[csl_osamp_cassava_leaf_train_1.index.repeat(3)].reset_index(drop=True)
print("Class 1 After: ", csl_osamp_cassava_leaf_train_1.shape)

csl_osamp_cassava_leaf_train_0 = csl_osamp_cassava_leaf_train.loc[csl_osamp_cassava_leaf_train['label'] == "0"]
print("Class 0 Before: ", csl_osamp_cassava_leaf_train_0.shape)
csl_osamp_cassava_leaf_train_0 = csl_osamp_cassava_leaf_train_0.loc[csl_osamp_cassava_leaf_train_0.index.repeat(6)].reset_index(drop=True)
print("Class 0 After: ", csl_osamp_cassava_leaf_train_0.shape)


csl_osamp_cassava_leaf_train = pd.concat([csl_osamp_cassava_leaf_train_3, csl_osamp_cassava_leaf_train_4, csl_osamp_cassava_leaf_train_2, csl_osamp_cassava_leaf_train_1, csl_osamp_cassava_leaf_train_0], axis=0)
print(csl_osamp_cassava_leaf_train.shape)
csl_osamp_cassava_leaf_proportion = (csl_osamp_cassava_leaf_train.label.value_counts() / len(csl_osamp_cassava_leaf_train)).to_frame()
csl_osamp_cassava_leaf_proportion


csl_osamp_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(csl_osamp_cassava_leaf_train.iloc[:,1].to_numpy()),
    y=csl_osamp_cassava_leaf_train.iloc[:,1].to_numpy()
)

csl_osamp_weights_dict = dict(enumerate(csl_osamp_weights))
print("Weights:", csl_osamp_weights_dict)


csl_osamp_cassava_leaf_train['label'] = csl_osamp_cassava_leaf_train['label'].astype(str)
csl_osamp_train_cassava_df, csl_osamp_valid_cassava_df = train_test_split(csl_osamp_cassava_leaf_train, test_size=0.2, random_state=1, stratify=csl_osamp_cassava_leaf_train.label)
print(csl_osamp_train_cassava_df.shape)
print(csl_osamp_valid_cassava_df.shape)


csl_osamp_train_dg = ImageDataGenerator(rescale=1/255, validation_split=0.20)
csl_osamp_valid_dg = ImageDataGenerator(rescale=1/255)


csl_osamp_train_loader = csl_osamp_train_dg.flow_from_dataframe(
    dataframe = csl_osamp_train_cassava_df,
    directory = '/kaggle/input/cassava-leaf-disease-classification/train_images',
    x_col = 'image_id',
    y_col = 'label',
    batch_size = 64,
    seed = 1,
    shuffle = True,
    class_mode = 'categorical',
    target_size = (75,100)
)

csl_osamp_valid_loader = csl_osamp_valid_dg.flow_from_dataframe(
    dataframe = csl_osamp_valid_cassava_df,
    directory = '/kaggle/input/cassava-leaf-disease-classification/train_images',
    x_col = 'image_id',
    y_col = 'label',
    batch_size = 64,
    seed = 1,
    shuffle = True,
    class_mode = 'categorical',
    target_size = (75,100)
)


csl_osamp_train_steps = len(csl_osamp_train_loader)
csl_osamp_valid_steps = len(csl_osamp_valid_loader)

print(csl_osamp_train_steps)
print(csl_osamp_valid_steps)


np.random.seed(1)
tf.random.set_seed(1)

csl_osamp_cassava_model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(75,100,3)),
    MaxPooling2D(2,2),
    Dropout(0.5),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(75,100,3)),
    MaxPooling2D(2,2),
    Dropout(0.5),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(75,100,3)),
    Flatten(),
    Dense(64, activation='softmax'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(5, activation='softmax')
])

csl_osamp_cassava_model.summary()


csl_osamp_cassava_optimizer = tf.keras.optimizers.Adam(0.0001)
csl_osamp_cassava_model.compile(loss='categorical_crossentropy', optimizer=csl_osamp_cassava_optimizer, metrics=['accuracy', tf.keras.metrics.AUC()])


%%time

csl_osamp_h1 = csl_osamp_cassava_model.fit(
    x = csl_osamp_train_loader,
    steps_per_epoch = csl_osamp_train_steps,
    epochs = 25,
    validation_data = csl_osamp_valid_loader,
    validation_steps = csl_osamp_valid_steps,
    verbose = 1,
    class_weight = csl_osamp_weights_dict
)

