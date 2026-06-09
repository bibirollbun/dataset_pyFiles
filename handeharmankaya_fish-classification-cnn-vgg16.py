import os
import cv2
import zipfile
import numpy as np 
import pandas as pd 
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, Dropout, BatchNormalization, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')


with zipfile.ZipFile('/kaggle/input/the-nature-conservancy-fisheries-monitoring/train.zip', 'r') as train_zip:
    train_zip.extractall('')

base_path = '/kaggle/working'
train_path = os.path.join(base_path, 'train')


train_path = '/kaggle/working/train'
fishes = os.listdir(train_path)
fishes


labels=['ALB', 'YFT', 'SHARK', 'NoF', 'LAG', 'BET', 'OTHER', 'DOL']


img_list1=[]
label_list1=[]
for label in labels:
    for img_file in os.listdir(train_path+"/"+label):
        img_list1.append(train_path+"/"+label+"/"+img_file)
        label_list1.append(label)


train=pd.DataFrame({'img':img_list1,'label':label_list1})


label_cod={'ALB':0,'YFT':1,'SHARK':2,'NoF':3,'LAG':4,'BET':5,'OTHER':6,'DOL':7}


train['encode_label']=train['label'].map(label_cod)


train.sample(5)


ax=sns.countplot(x=train['label'], palette='Spectral', order=train['label'].value_counts().index)
ax.bar_label(ax.containers[0], fontsize=12, color='black', padding=5)
plt.title('Number of Images per Class in the Train Data')
plt.xticks(rotation=45);


train['label'].value_counts()


train_unique=train.copy().drop_duplicates(subset=["label"]).reset_index()

fig, axes=plt.subplots(nrows=2, ncols=4, figsize=(8, 4), subplot_kw={'xticks':[], 'yticks':[]})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(train_unique.img[i]))
    ax.set_title(train_unique.label[i], fontsize = 12)
plt.tight_layout(pad=0.5)
plt.show()


x = []
for img_path in train['img']: 
    img=cv2.imread(str(img_path))
    if img is None:
        print(f"Resim yüklenemedi: {img_path}") 
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=cv2.resize(img, (128, 128))
    img=img / 255.0
    x.append(img)


x=np.array(x)


y=train[['encode_label']]


#!pip install --upgrade --force-reinstall scikit-learn==1.3.2 imbalanced-learn==0.11.0


from imblearn.over_sampling import SMOTE

x_flat = x.reshape(x.shape[0], -1)  
y_flat = y['encode_label'].values 

smote = SMOTE(random_state=42)

x_resampled, y_resampled = smote.fit_resample(x_flat, y_flat)
x_resampled_images = x_resampled.reshape(-1, 128, 128, 3)


x_resampled_images.shape


y_resampled.shape


y_resampled_series = pd.Series(y_resampled)
sns.countplot(x=y_resampled_series, palette='Spectral')
plt.xticks(rotation=45);


x_train,x_test,y_train,y_test=train_test_split(x_resampled_images,y_resampled, random_state=42, test_size=0.20)


model = Sequential()
model.add(Input(shape=(128, 128, 3)))

model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(1024, kernel_size=(3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(512, activation='relu')) 
model.add(Dropout(0.5))  
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(8, activation='softmax'))

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()


early_stop=EarlyStopping(monitor='val_loss',patience=10)


history=model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=50, callbacks=[early_stop], verbose=1)


model.save('fish.h5')


history.history['accuracy'][-1]


plt.plot(history.history['accuracy'],label='Accuracy')
plt.plot(history.history['val_accuracy'],label='Val_Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val_Loss')
plt.legend();


with zipfile.ZipFile('/kaggle/input/the-nature-conservancy-fisheries-monitoring/test_stg1.zip', 'r') as test_stg1:
    test_stg1.extractall('')

base_path = '/kaggle/working'
train_path = os.path.join(base_path, 'test1')


test1_path=('/kaggle/working/test_stg1')


test_files = os.listdir(test1_path)
x_submission = []
img_names = []


for file in test_files:
    img_path = os.path.join(test1_path, file)
    img = cv2.imread(img_path)
    
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128)) 
        img = img / 255.0                 
        x_submission.append(img)
        img_names.append(file)            

x_submission = np.array(x_submission)


predictions = model.predict(x_submission, verbose=1)


submission = pd.DataFrame(predictions, columns=['ALB', 'YFT', 'SHARK', 'NoF', 'LAG', 'BET', 'OTHER', 'DOL'])


submission.insert(0, 'image', img_names)


submission.shape


submission.head()


#!pip install py7zr


from py7zr import py7zr

with py7zr.SevenZipFile('/kaggle/input/the-nature-conservancy-fisheries-monitoring/test_stg2.7z', mode='r') as z:
    z.extractall('test')


test2_path=('/kaggle/working/test/test_stg2')


test_files = os.listdir(test2_path)
x2_submission = []
img2_names = []


for file in test_files:
    img_path = os.path.join(test2_path, file)
    img = cv2.imread(img_path)
    
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128)) 
        img = img / 255.0                 
        x2_submission.append(img)
        img2_names.append(file)            

x2_submission = np.array(x2_submission)


predictions2 = model.predict(x2_submission, verbose=1)


submission2 = pd.DataFrame(predictions2, columns=['ALB', 'YFT', 'SHARK', 'NoF', 'LAG', 'BET', 'OTHER', 'DOL'])


submission2.insert(0, 'image', img2_names)


submission2.shape


submission2.head()


submission2['image'] = submission2['image'].apply(lambda x: f"test_stg2/{x}" if not x.startswith('test_stg2/') else x)


submission2.head()


submission_last = pd.concat([submission, submission2], axis=0, ignore_index=True)


submission_last.shape


submission_last.to_csv('submission.csv', index=False)


data_dir = '/kaggle/working/train/'
img_width, img_height = 224, 224
batch_size = 32
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,rotation_range=20,
                                   width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True,
                                   validation_split=0.2)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,validation_split=0.2)

train_generator = train_datagen.flow_from_directory(directory=data_dir,
                                                    target_size=(img_width, img_height),batch_size=batch_size,
                                                    class_mode='sparse',subset='training',shuffle=True)

validation_generator = val_datagen.flow_from_directory(directory=data_dir,
                                                       target_size=(img_width, img_height),batch_size=batch_size,
                                                       class_mode='sparse',subset='validation',shuffle=False)


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x) 
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x) 
predictions = Dense(8, activation='softmax')(x) 

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()


checkpoint = ModelCheckpoint("vgg16_best_fish.keras",monitor='val_accuracy',verbose=1,
                             save_best_only=True,mode='max')

early_stop = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True,verbose=1)

history = model.fit(train_generator,steps_per_epoch=train_generator.samples // batch_size,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // batch_size,
                    epochs=25,callbacks=[checkpoint, early_stop])


model.save('fishv2.h5')


history.history['accuracy'][-1]


plt.plot(history.history['accuracy'],label='Accuracy')
plt.plot(history.history['val_accuracy'],label='Val_Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val_Loss')
plt.legend();


MODEL_PATH = '/kaggle/working/fishv2.h5' 
IMG_SIZE = 224
BATCH_SIZE = 32
COLUMNS = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

def predict_with_generator(folder_path, model):
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    df = pd.DataFrame({'filename': files})
    gen = test_datagen.flow_from_dataframe(dataframe=df,directory=folder_path,x_col='filename',y_col=None,
                                           class_mode=None,target_size=(IMG_SIZE, IMG_SIZE),batch_size=BATCH_SIZE,
                                           shuffle=False)
    preds = model.predict(gen, verbose=1)
    return preds, files

#STAGE 1
test1_path = '/kaggle/working/test_stg1'
preds1, names1 = predict_with_generator(test1_path, model)
df1 = pd.DataFrame(preds1, columns=COLUMNS)
df1.insert(0, 'image', names1)

#STAGE 2
test2_path = '/kaggle/working/test/test_stg2'
preds2, names2 = predict_with_generator(test2_path, model)
df2 = pd.DataFrame(preds2, columns=COLUMNS)
df2.insert(0, 'image', names2)
df2['image'] = df2['image'].apply(lambda x: f"test_stg2/{x}" if not x.startswith('test_stg2/') else x)

submission_final = pd.concat([df1, df2], axis=0, ignore_index=True)
ordered_cols = ['image', 'ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
submission_final = submission_final[ordered_cols]
submission_final.to_csv('submission_vgg16.csv', index=False)

