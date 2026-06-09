import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, BatchNormalization, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import os


train_path = '/kaggle/input/4-animal-classification/train'

X_train, y_train = [], []
mp = {}
for label_name in os.listdir(train_path):
    sub_dir = os.path.join(train_path, label_name)
    mp[label_name] = 0
    for sub_dirp in os.listdir(sub_dir):
        mp[label_name] = mp[label_name] + 1
        file = os.path.join(sub_dir, sub_dirp)
        img = imread(file)
        img = resize(img, (224, 224), preserve_range=True)
        img = img.astype(np.float32)
        X_train.append(img)
        y_train.append(label_name)


other_path = '/kaggle/input/coco-subset-for-pose-estimation/dataset/train'
mp['other'] = 0
d = 0
for sub_dir in os.listdir(other_path):
    if d == 1000:
        break
    d = d + 1
    file = os.path.join(other_path, sub_dir)
    mp['other'] = mp['other'] + 1
    img = imread(file)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    elif img.ndim != 3 or img.shape[2] != 3:
        print(f"Unexpected shape {img.shape} at {file}, skipping.")
        continue
    img = resize(img, (224, 224), preserve_range=True)
    img = img.astype(np.float32)
    X_train.append(img)
    y_train.append('other')


print(mp)


labels = list(mp.keys())
counts = list(mp.values())

plt.figure(figsize=(10, 6))
plt.bar(labels, counts)
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Number of Images per Class')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


test_path = '/kaggle/input/4-animal-classification/test/test'

X_test, id_sub = [], []
for sub_dir in os.listdir(test_path):
    file = os.path.join(test_path, sub_dir)
    img = imread(file)
    img = resize(img, (224, 224), preserve_range=True)
    img = img.astype(np.float32)
    id_sub.append(sub_dir.split('.')[0])
    X_test.append(img)


X_train = np.array(X_train)


X_test = np.array(X_test)


X_train.shape


y_train = np.array(y_train)


print(y_train)


ec = OneHotEncoder(sparse=False, categories=[['cat', 'deer', 'dog', 'horse', 'other']])

y_train = ec.fit_transform(y_train.reshape(-1, 1))


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = 19)


for i in range(10):
    plt.imshow(X_train[i] / 255.0)
    plt.show()
    print(ec.inverse_transform([y_train[i]]))


X_train.shape


y_train.shape


y_train


def model(model_name):
  model = Sequential()
  # model.add(Lambda(preprocess_input)(Input(shape=(224, 224, 3))))
  for layer in model_name.layers:
    layer.trainable = False
    model.add(layer)
  return model


vgg16 = VGG16(weights='imagenet',include_top=False, input_tensor= Lambda(preprocess_input)(Input(shape=(224, 224, 3))))
model_vgg16 = model(vgg16)
model_vgg16.add(GlobalAveragePooling2D())
model_vgg16.add(Dense(4096, activation='relu'))
model_vgg16.add(BatchNormalization())
model_vgg16.add(Dropout(0.2))

model_vgg16.add(Dense(4096, activation='relu'))
model_vgg16.add(BatchNormalization())
model_vgg16.add(Dropout(0.2))

model_vgg16.add(Dense(5, activation='softmax'))
model_vgg16.summary()


model_vgg16.compile(optimizer= Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])


early_stop = EarlyStopping(
    monitor='val_loss',   
    patience=16,      
    restore_best_weights=True 
)

history_vgg16 = model_vgg16.fit(X_train, y_train
                       , validation_data = (X_valid, y_valid)
                       , batch_size = 64, epochs = 100
                       , callbacks = [early_stop])


plt.figure(figsize=(8, 5))
plt.plot(history_vgg16.history['loss'], label='Training Loss')
if 'val_loss' in history_vgg16.history:
    plt.plot(history_vgg16.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

if 'accuracy' in history_vgg16.history or 'val_accuracy' in history_vgg16.history:
    plt.figure(figsize=(8, 5))
    if 'accuracy' in history_vgg16.history:
        plt.plot(history_vgg16.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history_vgg16.history:
        plt.plot(history_vgg16.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


y_pred = model_vgg16.predict(X_test)
y_pred = np.argmax(y_pred, axis = 1)

submission1 = pd.read_csv('/kaggle/input/4-animal-classification/Sample_submission.csv')
submission1['ID'] = id_sub
submission1['Label'] = y_pred
submission1.to_csv('submission1_1.csv', index = False)


vgg16_aug = VGG16(weights='imagenet',include_top=False, input_tensor= Lambda(preprocess_input)(Input(shape=(224, 224, 3))))
model_vgg16_aug = model(vgg16_aug)
model_vgg16_aug.add(GlobalAveragePooling2D())
model_vgg16_aug.add(Dense(4096, activation='relu'))
model_vgg16_aug.add(BatchNormalization())
model_vgg16_aug.add(Dropout(0.2))

model_vgg16_aug.add(Dense(4096, activation='relu'))
model_vgg16_aug.add(BatchNormalization())
model_vgg16_aug.add(Dropout(0.2))

model_vgg16_aug.add(Dense(5, activation='softmax'))
model_vgg16_aug.summary()


model_vgg16_aug.compile(optimizer= Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


train_generator = train_datagen.flow(
    X_train, y_train,
    batch_size=64,
    shuffle=True,
    seed=19
)


early_stop = EarlyStopping(
    monitor='val_loss',   
    patience=16,      
    restore_best_weights=True 
)

history_vgg16_aug = model_vgg16_aug.fit(train_generator
                       , validation_data = (X_valid, y_valid)
                       , batch_size = 64, epochs = 100
                       , callbacks = [early_stop])


plt.figure(figsize=(8, 5))
plt.plot(history_vgg16_aug.history['loss'], label='Training Loss')
if 'val_loss' in history_vgg16_aug.history:
    plt.plot(history_vgg16_aug.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

if 'accuracy' in history_vgg16_aug.history or 'val_accuracy' in history_vgg16_aug.history:
    plt.figure(figsize=(8, 5))
    if 'accuracy' in history_vgg16_aug.history:
        plt.plot(history_vgg16_aug.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history_vgg16_aug.history:
        plt.plot(history_vgg16_aug.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


y_pred = model_vgg16.predict(X_test)
y_pred = np.argmax(y_pred, axis = 1)

submission2 = pd.read_csv('/kaggle/input/4-animal-classification/Sample_submission.csv')
submission2['ID'] = id_sub
submission2['Label'] = y_pred
submission2.to_csv('submission2_1.csv', index = False)


class ResNet:
    def __init__(self,
                 input_shape=(224, 224, 3),
                 lr: float = 1e-3):
        
        self.input_shape = input_shape
        self.lr = lr
        self.model: tf.keras.Model = None

    def build(self):
        inputs = Input(shape=self.input_shape)

        x = Lambda(preprocess_input, name="preprocessing")(inputs)

        base = ResNet50(
            weights='imagenet',
            include_top=False,
            input_tensor=x
        )
        base.trainable = False

        x = GlobalAveragePooling2D(name="gap")(base.output)

        x = Dense(4096, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(rate = 0.2)(x)

        x = Dense(4096, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(rate = 0.2)(x)
        
        output = Dense(5, activation='softmax', name="predictions")(x)

        self.model = Model(inputs=inputs, outputs=output, name="ResNet50_4cls")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return self.model

    def summary(self):
        self.model.summary()


resnet50 = ResNet()
resnet50 = resnet50.build()
print(resnet50.summary())


early_stop = EarlyStopping(
    monitor='val_loss',   
    patience=16,      
    restore_best_weights=True 
)

history_resnet50 = resnet50.fit(X_train, y_train
                       , validation_data = (X_valid, y_valid)
                       , batch_size = 64, epochs = 100
                       , callbacks = [early_stop])


plt.figure(figsize=(8, 5))
plt.plot(history_resnet50.history['loss'], label='Training Loss')
if 'val_loss' in history_resnet50.history:
    plt.plot(history_resnet50.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

if 'accuracy' in history_resnet50.history or 'val_accuracy' in history_resnet50.history:
    plt.figure(figsize=(8, 5))
    if 'accuracy' in history_resnet50.history:
        plt.plot(history_resnet50.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history_resnet50.history:
        plt.plot(history_resnet50.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


y_pred = resnet50.predict(X_test)
y_pred = np.argmax(y_pred, axis = 1)


submission1 = pd.read_csv('/kaggle/input/4-animal-classification/Sample_submission.csv')
submission1['ID'] = id_sub
submission1['Label'] = y_pred
submission1.to_csv('submission1_2.csv', index = False)


resnet50_aug = ResNet()
resnet50_aug = resnet50_aug.build()


train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


train_generator = train_datagen.flow(
    X_train, y_train,
    batch_size=64,
    shuffle=True,
    seed=19
)


early_stop = EarlyStopping(
    monitor='val_loss',   
    patience=16,      
    restore_best_weights=True 
)

history_resnet50_aug = resnet50_aug.fit(train_generator
                           , validation_data = (X_valid, y_valid)
                           , batch_size = 64, epochs = 100
                           , callbacks = [early_stop])


plt.figure(figsize=(8, 5))
plt.plot(history_resnet50_aug.history['loss'], label='Training Loss')
if 'val_loss' in history_resnet50_aug.history:
    plt.plot(history_resnet50_aug.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

if 'accuracy' in history_resnet50_aug.history or 'val_accuracy' in history_resnet50_aug.history:
    plt.figure(figsize=(8, 5))
    if 'accuracy' in history_resnet50_aug.history:
        plt.plot(history_resnet50_aug.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history_resnet50_aug.history:
        plt.plot(history_resnet50_aug.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


y_pred = resnet50.predict(X_test)
y_pred = np.argmax(y_pred, axis = 1)


submission2 = pd.read_csv('/kaggle/input/4-animal-classification/Sample_submission.csv')
submission2['ID'] = id_sub
submission2['Label'] = y_pred
submission2.to_csv('submission2_2.csv', index = False)


import matplotlib.pyplot as plt

histories = {
    'VGG16': history_vgg16,
    'VGG16 + Aug': history_vgg16_aug,
    'ResNet50': history_resnet50,
    'ResNet50 + Aug': history_resnet50_aug
}

def plot_metric(metric, title, ylabel):
    plt.figure(figsize=(10, 6))
    for name, history in histories.items():
        if metric in history.history:
            plt.plot(history.history[metric], label=name)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

# Vẽ 4 biểu đồ
plot_metric('loss', 'Training Loss Comparison', 'Loss')
plot_metric('val_loss', 'Validation Loss Comparison', 'Loss')
plot_metric('accuracy', 'Training Accuracy Comparison', 'Accuracy')
plot_metric('val_accuracy', 'Validation Accuracy Comparison', 'Accuracy')


print('Thống kê vgg16:')
print(classification_report(y_valid, model_vgg16.predict(X_valid)))
print('Thống kê vgg16 có augmentation:')
print(classification_report(y_valid, model_vgg16_aug.predict(X_valid)))
print('Thống kê resnet:')
print(classification_report(y_valid, resnet50.predict(X_valid)))
print('Thống kê resnet50 có augmentation:')
print(classification_report(y_valid, resnet50_aug.predict(X_valid)))


def predict_image(link):
    img = imread(link)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    elif img.ndim != 3 or img.shape[2] != 3:
        print(f"Unexpected shape {img.shape} at {file}, skipping.")
        return 0
    img = resize(img, (224, 224), preserve_range=True)
    img = img.astype(np.float32)
    img = np.array(img)
    
    plt.imshow(img / 255)
    
    img = np.expand_dims(img, 0)
    
    print(img.shape)
    
    mp = {0: 'cat'
          , 1: 'deer'
          , 2:'dog'
          , 3:'horse'
          , 4: 'other'}

    pred_vgg16 = model_vgg16.predict(img)
    lab = np.argmax(pred_vgg16, axis = 1)
    lab = [mp[x] for x in lab]
    print('Predicted by VGG16')
    print('label: ', lab, ' Accuracy: ', pred_vgg16.max(axis = 1))
    
    pred_vgg16_aug = model_vgg16_aug.predict(img)
    lab = np.argmax(pred_vgg16_aug, axis = 1)
    lab = [mp[x] for x in lab]
    print('Predicted by VGG16 with augmentation')
    print('label: ', lab, ' Accuracy: ', pred_vgg16_aug.max(axis = 1))
    
    pred_resnet50 = resnet50.predict(img)
    lab = np.argmax(pred_resnet50, axis = 1)
    lab = [mp[x] for x in lab]
    print('Predicted by resnet50')
    print('label: ', lab, ' Accuracy: ', pred_resnet50.max(axis = 1))
    
    pred_resnet50_aug = resnet50_aug.predict(img)
    lab = np.argmax(pred_resnet50_aug, axis = 1)
    lab = [mp[x] for x in lab]
    print('Predicted by resnet with augmentation')
    print('label: ', lab, ' Accuracy: ', pred_resnet50_aug.max(axis = 1))
    return 0


predict_image('/kaggle/input/conho2/de4abe63-4b99-4d63-8d7b-332c44981369.jpeg')


predict_image('/kaggle/input/lmessi/images.jpeg')


predict_image('/kaggle/input/thay123/MaiTienDung-300x300.png')


predict_image('/kaggle/input/3animal/indoor-photo-dog-cat-horse-260nw-2524143355.webp')


predict_image('/kaggle/input/3horse/images (1).jpeg')


predict_image('/kaggle/input/doglikecat/shutterstock_478338919_huge.jpg')


predict_image('/kaggle/input/superdog/ba33f562-5a7e-4d20-a0e3-744746315edf.jpeg')


predict_image('/kaggle/input/pokemon/ti xung.jpeg')


predict_image('/kaggle/input/cat-pokemon/images (2).jpeg')


predict_image('/kaggle/input/deer123/White-tailed_deer_(Odocoileus_virginianus)_grazing_-_20050809.jpg')

