import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import os
import zipfile
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


with zipfile.ZipFile('/kaggle/input/aerial-cactus-identification/train.zip', 'r') as z:
    z.extractall('.')
    
with zipfile.ZipFile('/kaggle/input/aerial-cactus-identification/test.zip', 'r') as z:
    z.extractall('.')


train_path='/kaggle/working/train/'
test_path='/kaggle/working/test/'

train_dir=os.path.join(train_path,'/train/')
test_dir=os.path.join(test_path)
train_df=pd.read_csv('/kaggle/input/aerial-cactus-identification/train.csv')


img_list1 = [os.path.join(train_path, img_id) for img_id in train_df['id']]
label_list1 = list(train_df['has_cactus'])


df = pd.DataFrame()
df['image'] = img_list1
df['label'] = label_list1


test_filenames = os.listdir(test_dir)
test_df = pd.DataFrame({
    'id': test_filenames})


df.head()


df.shape


df.info()


df.isnull().sum()


df['label'] = df['label'].astype(str)


df.info()


df['label'].value_counts()


sns.countplot(x=df['label'], palette=['salmon', 'skyblue']);


fig, ax=plt.subplots(2, 5) 
fig.set_size_inches(12, 6)
k = 0
for i in range(2):
    for j in range(5):
        img_path = img_list1[k]
        label = label_list1[k]
        img = plt.imread(img_path)
        ax[i,j].imshow(img)
        status = "Has Cactus (1)" if label == 1 else "No Cactus(0)"
        ax[i,j].set_title(status, fontsize=10)
        ax[i,j].axis('off')
        k += 1
plt.tight_layout()
plt.show()


test_df.head()


test_df.shape


sample_test = test_df.sample(10).reset_index(drop=True)
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))

for i, ax in enumerate(axes.flat):
    img_name = sample_test.loc[i, 'id']
    full_path = os.path.join(test_dir, img_name)
    img = plt.imread(full_path)
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.show()


#Providing balance by giving higher points to the lower class
unique_classes = np.sort(df['label'].unique())
class_weights_arr = class_weight.compute_class_weight(class_weight='balanced',
classes=unique_classes, y=df['label'])
weights_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights_arr)}

#Train - Validation 
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

#Data Augmentation
IMG_SIZE = (64, 64)
BATCH_SIZE = 64

train_datagen = ImageDataGenerator(
rescale=1./255,
horizontal_flip=True,
vertical_flip=True,
rotation_range=30,
zoom_range=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
directory=None,
x_col="image",
y_col="label",
target_size=IMG_SIZE,
batch_size=BATCH_SIZE,
class_mode='binary',
shuffle=True)

val_generator = val_datagen.flow_from_dataframe(dataframe=val_df,
directory=None,
x_col="image",
y_col="label",
target_size=IMG_SIZE,
batch_size=BATCH_SIZE,
class_mode='binary',
shuffle=False)


inputs = Input(shape=(64, 64, 3))
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

model.compile(optimizer=Adam(learning_rate=0.001),
loss='binary_crossentropy',
metrics=['accuracy'])

model.summary()


callbacks = [
EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

history = model.fit(
train_generator,
epochs=20,
validation_data=val_generator,
callbacks=callbacks,
class_weight=weights_dict
)


history.history['accuracy'][-1]


model.save('cactus.h5')


tf.keras.models.save_model(model, "cactus1.keras")


plt.plot(history.history['accuracy'],label='Accuracy')
plt.plot(history.history['val_accuracy'],label='Val_Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val_Loss')
plt.legend();


#Test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,
directory=test_dir,
x_col='id',
y_col=None,
class_mode=None,
shuffle=False,
target_size=IMG_SIZE,
batch_size=BATCH_SIZE)

#Prediction
predictions = model.predict(test_generator)


#Submission 
submission_df = pd.DataFrame()
submission_df['id'] = test_filenames
submission_df['has_cactus'] = predictions.flatten() 
submission_df.to_csv('submission.csv', index=False)


base_model.trainable = True

model.compile(
optimizer=Adam(learning_rate=1e-5),
loss='binary_crossentropy',
metrics=['accuracy']
)

history_fine = model.fit(
train_generator,
epochs=15,
validation_data=val_generator,
callbacks=callbacks,
class_weight=weights_dict
)


history_fine.history['accuracy'][-1]


tf.keras.models.save_model(model, "cactus2.keras")


plt.plot(history_fine.history['accuracy'],label='Accuracy')
plt.plot(history_fine.history['val_accuracy'],label='Val_Accuracy')
plt.plot(history_fine.history['loss'], label='Loss')
plt.plot(history_fine.history['val_loss'], label='Val_Loss')
plt.legend();


# Test Data Generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,directory=test_dir,
                                                  x_col='id',y_col=None,class_mode=None,
                                                  shuffle=False,target_size=(64, 64),batch_size=64)

predictions_fine = model.predict(test_generator)


#Submission
submission = pd.DataFrame()
submission ['id'] = test_filenames
submission['has_cactus'] = predictions_fine.flatten()

submission.to_csv('submission.csv', index=False)

