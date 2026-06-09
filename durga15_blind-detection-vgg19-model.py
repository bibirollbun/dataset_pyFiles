import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as  plt
import tensorflow as tf
from tensorflow.keras import layers,models,optimizers
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout,Flatten
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from tensorflow.keras.applications import VGG16,VGG19
from tensorflow.keras.applications.efficientnet import EfficientNetB3
import warnings
warnings.simplefilter('ignore')
from PIL import Image
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

base_dir = "/kaggle/input/aptos2019-blindness-detection"


df = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")
df.head()


train_csv = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/train.csv')
counts = train_csv['diagnosis'].value_counts()
class_list = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferate']
for i,x in enumerate(class_list):
    counts[x] = counts.pop(i)

plt.figure(figsize=(10,5))
sns.barplot(x=counts.index, y=counts.values, alpha=0.8, palette='bright')
plt.title('Distribution of Output Classes')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Target Classes', fontsize=12)
plt.show()


fig = plt.figure(figsize=(30, 6))
# display 20 images
train_imgs = os.listdir(base_dir+"/train_images")
for idx, img in enumerate(np.random.choice(train_imgs, 16)):
    ax = fig.add_subplot(2, 16//2, idx+1, xticks=[], yticks=[])
    im = Image.open(base_dir+"/train_images/" + img)
    plt.imshow(im)
    lab = train_csv.loc[train_csv['id_code'] == img.split('.')[0], 'diagnosis'].values[0]
    ax.set_title('Severity: %s'%lab)


fig = plt.figure(figsize=(30, 6))
# display 20 images
test_imgs = os.listdir(base_dir+"/test_images")
for idx, img in enumerate(np.random.choice(test_imgs, 16)):
    ax = fig.add_subplot(2, 16//2, idx+1, xticks=[], yticks=[])
    im = Image.open(base_dir+"/test_images/" + img)
    plt.imshow(im)


df["id_code"] = df["id_code"].apply(lambda x: x + ".png")


train_df = df.iloc[:3000,:]
test_df = df.iloc[3000:,:]


train_df['diagnosis'] = train_df['diagnosis'].astype('str')
gen = ImageDataGenerator(
    horizontal_flip = True,
    vertical_flip = True,
    shear_range = 0.2,
    zoom_range = 0.2,
    rescale = 1/255.,
)
train_datagen = gen.flow_from_dataframe(
    train_df,
    directory = "/kaggle/input/aptos2019-blindness-detection/train_images",
    batch_size = 32,
    target_size = (224,224),
    seed = 42,
    x_col = 'id_code',
    y_col = 'diagnosis',
    class_mode = 'categorical'
)


test_df['diagnosis'] = test_df['diagnosis'].astype('str')
gen = ImageDataGenerator(
    rescale = 1/255.,
)
test_datagen = gen.flow_from_dataframe(
    test_df,
    directory="/kaggle/input/aptos2019-blindness-detection/train_images",
    batch_size = 32,
    target_size = (224,224),
    seed = 42,
    x_col = 'id_code',
    y_col = 'diagnosis',
    class_mode = 'categorical'
)


model = models.Sequential()
vgg = VGG19(include_top = False,weights = 'imagenet',input_shape=(224,224,3))
vgg.trainable = False
model.add(vgg)
model.add(GlobalAveragePooling2D())

model.add(Flatten())

model.add(Dense(256,activation='elu'))
model.add(Dropout(0.3))

model.add(Dense(5,activation='softmax'))


loss = tf.keras.losses.CategoricalCrossentropy(
    label_smoothing = 0.001,
    name = 'categorical_crossentropy'
)


optimizer = Adam(learning_rate = 1e-4)


model.compile(optimizer = optimizer,loss=loss,metrics= ['categorical_accuracy'])


model.summary()


rlrong = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    mode='min',
    min_lr = 1e-5,
    patience = 2,
    verbose=1
)
estop = EarlyStopping(
    monitor = 'val_loss',
    mode= 'min',
    patience = 3,
    verbose = 1,
    restore_best_weights = True
)


history = model.fit(train_datagen,epochs = 20,verbose=1,validation_data = test_datagen,callbacks = [rlrong,estop])


plt.plot(history.history['loss'],label='loss',color='red')
plt.plot(history.history['val_loss'],label='val loss',color='blue')
plt.legend()
plt.grid()
plt.show()


plt.plot(history.history['categorical_accuracy'],label='categorical accuracy',color='red')
plt.plot(history.history['val_categorical_accuracy'],label='val categorical accuracy',color='blue')
plt.legend()
plt.grid()
plt.show()


model.save('model_VGG16.h5')




