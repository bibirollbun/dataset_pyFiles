


%%capture 
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#!pip install scikit-image


import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
import os
import seaborn as sns
import cv2
import random
import os
import glob


# Load train data
train_df = pd.read_csv('/kaggle/input/UBC-OCEAN/train.csv')
train_df.head()


train_df['label'].value_counts()


print(train_df.shape)


train_df.isna().sum().sum()


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib.pyplot as plt
import seaborn as sns

# list of columns we want the distributions for (exclude 'label')
columns = [col for col in train_df.columns if col != 'label']

# convert boolean columns to integers to avoid histogram warnings
for col in columns:
    if train_df[col].dtype == 'bool':
        train_df[col] = train_df[col].astype(int)

for col in columns:
    
    # subplot for 3 columns
    fig, axs = plt.subplots(figsize=(15, 5), ncols=3)
    
    # 1 - tÃ¼m verinin daÄŸÄ±lÄ±mÄ±
    sns.histplot(data=train_df, x=col, kde=True, ax=axs[0])
    axs[0].set_title(f'{col} - TÃ¼m Ã–rnekler')
    
    # 2 - HGSC olanlarÄ±n daÄŸÄ±lÄ±mÄ±
    sns.histplot(data=train_df[train_df['label'] == "HGSC"], x=col, kde=True, ax=axs[1], color='orange')
    axs[1].set_title(f'{col} - HGSC')
    
    # 3 - LGSC olanlarÄ±n daÄŸÄ±lÄ±mÄ±
    sns.histplot(data=train_df[train_df['label'] == "LGSC"], x=col, kde=True, ax=axs[2], color='green')
    axs[2].set_title(f'{col} - LGSC')
    
    plt.tight_layout()
    plt.show()



train_df.info()


print(train_df.image_id.is_unique)
print(train_df.label.is_unique)


train_df[['image_height', 'image_width']].describe()


print(train_df.image_id.shape[0])
print(len(os.listdir('/kaggle/input/UBC-OCEAN/train_images')))
print(len(os.listdir('/kaggle/input/UBC-OCEAN/train_thumbnails')))


import imageio



# Temel bilimsel hesaplama ve veri analizi
import numpy as np
import pandas as pd

# GÃ¶rselleÅŸtirme
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg

# Jupyter'de ilerleme Ã§ubuÄŸu
from tqdm.notebook import tqdm

# GÃ¶rÃ¼ntÃ¼ iÅŸleme ve artÄ±rma
import albumentations as A
import imageio
import scipy.ndimage as ndi



import warnings
warnings.filterwarnings('ignore')


train_df.label.value_counts()


HGSC = train_df[train_df['label']=="HGSC"]
EC = train_df[train_df['label']=="EC"]
CC = train_df[train_df['label']=="CC"]
LGSC = train_df[train_df['label']=="LGSC"]
MC = train_df[train_df['label']=="MC"]


# Set the figure size
plt.figure(figsize=(20, 6))

# Set the font size
plt.rcParams['font.size'] = 14

# Set the colors
colors = ['lightgreen', 'lightblue', 'purple', 'blue', 'yellow']

# Plot the pie chart for the training set
plt.subplot(1, 1, 1)
plt.pie([len(HGSC), len(EC), len(CC), len(LGSC), len(MC)], labels=['HGSC', 'EC', 'CC', 'LGSC', 'MC'], autopct='%1.1f%%', colors=colors)
plt.title('Training Set')


# Add a main title to the figure
plt.suptitle('Distribution of HGSC, EC, CC, LGSC and MC Images in the Training data', fontsize=20, y=1.05)

# Show the plot
plt.show()





import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. TÃ¼m gÃ¶rselleri tek klasÃ¶rden al
all_data = glob.glob('/kaggle/input/UBC-OCEAN/train_images/*.png')

# 2. %10 test setini ayÄ±r
train_val_data, test_data = train_test_split(all_data, test_size=0.1, random_state=42)

# 3. Kalan %90'dan %10 doÄŸrulama seti ayÄ±r (1/9 â‰ˆ %10)
train_data, val_data = train_test_split(train_val_data, test_size=1/9, random_state=42)

# 4. SayÄ±larÄ± al
total_train = len(train_data)
total_val = len(val_data)
total_test = len(test_data)
total = total_train + total_val + total_test

# 5. YÃ¼zdeleri hesapla
train_pct = 100 * total_train / total
val_pct = 100 * total_val / total
test_pct = 100 * total_test / total

# 6. Pasta grafiÄŸi Ã§iz
plt.figure(figsize=(5, 5))
plt.rcParams['font.size'] = 12
colors = ['lightgreen', 'skyblue', 'salmon']

plt.pie(
    [total_train, total_val, total_test],
    labels=[
        f'Training Set ({train_pct:.1f}%)',
        f'Validation Set ({val_pct:.1f}%)',
        f'Testing Set ({test_pct:.1f}%)'
    ],
    autopct='%1.1f%%',
    colors=colors,
    startangle=90
)

plt.title('Distribution of Images in Training, Validation, and Testing Sets')
plt.axis('equal')
plt.show()



# Toplam veri sayÄ±larÄ±
total_train = len(train_data)
total_val = len(val_data)
total_test = len(test_data)

# ToplamÄ± al
total = total_train + total_val + total_test

# YÃ¼zdeleri hesapla
train_pct = 100 * total_train / total
val_pct = 100 * total_val / total
test_pct = 100 * total_test / total

# Pasta grafiÄŸi Ã§iz
plt.figure(figsize=(5, 5))
plt.rcParams['font.size'] = 12
colors = ['lightgreen', 'skyblue', 'salmon']

plt.pie(
    [total_train, total_val, total_test],
    labels=[
        f'Training Set ({train_pct:.1f}%)',
        f'Validation Set ({val_pct:.1f}%)',
        f'Testing Set ({test_pct:.1f}%)'
    ],
    autopct='%1.1f%%',
    colors=colors,
    startangle=90
)

plt.title('Distribution of Images in Training, Validation, and Testing Sets')
plt.axis('equal')  # Daireyi dÃ¼zgÃ¼n gÃ¶sterir
plt.show()



# sample training image

io.imshow('/kaggle/input/UBC-OCEAN/train_thumbnails/10642_thumbnail.png')


HGSC = train_df[train_df['label']=="HGSC"]
EC = train_df[train_df['label']=="EC"]
CC = train_df[train_df['label']=="CC"]
LGSC = train_df[train_df['label']=="LGSC"]
MC = train_df[train_df['label']=="MC"]


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# SÄ±nÄ±f adlarÄ± â€“ sabit sÄ±ralÄ± liste
class_names = ['HGSC', 'LGSC', 'MC', 'CC', 'EC']

# GerÃ§ek etiketler ve tahminler â€“ BunlarÄ± kendi model Ã§Ä±ktÄ±nla doldurman gerekiyor!
# Ã–rnek veriler (senin verinle deÄŸiÅŸtir)
y_true = ['HGSC', 'HGSC', 'LGSC', 'MC', 'EC', 'CC', 'LGSC', 'HGSC', 'MC', 'EC']
y_pred = ['HGSC', 'LGSC', 'LGSC', 'MC', 'MC', 'CC', 'HGSC', 'HGSC', 'MC', 'EC']

# Etiketleri sayÄ±ya Ã§evir
encoder = LabelEncoder()
encoder.fit(class_names)

y_true_encoded = encoder.transform(y_true)
y_pred_encoded = encoder.transform(y_pred)

# KonfÃ¼zyon matrisi oluÅŸtur
cm = confusion_matrix(y_true_encoded, y_pred_encoded)

# KonfÃ¼zyon matrisini Ã§iz
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)

plt.xlabel('Tahmin Edilen SÄ±nÄ±f')
plt.ylabel('GerÃ§ek SÄ±nÄ±f')
plt.title('KonfÃ¼zyon Matrisi')
plt.tight_layout()
plt.show()



"""!pip install torch-summary
!pip install torch-lr-finder"""


"""import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
from PIL import Image
from IPython.display import display
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from tqdm import tqdm
import torchvision.models as models
import PIL
from torchvision.models import resnet50, ResNet50_Weights
import timm
from torch.utils.data import Dataset
import glob"""


"""torch.manual_seed(0)"""


"""data_path = '/kaggle/input/UBC-OCEAN/train_images'"""


"""class_name = os.listdir(data_path)
print(class_name)"""


"""class_map = {"HGSC" : 0, "EC": 1, "CC": 2, "LGSC": 3, "MC": 4}
rev_class_map = {0 : "HGSC", 1 : "EC", 2 : "CC", 3 : "LGSC", 4 : "MC"}"""





from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import regularizers, optimizers
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


traindf=pd.read_csv('/kaggle/input/UBC-OCEAN/train.csv',dtype=str)
traindf


testdf=pd.read_csv('/kaggle/input/UBC-OCEAN/test.csv',dtype=str)
testdf


datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)
train_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="/kaggle/input/UBC-OCEAN/train_images/",
x_col="image_id",
y_col="label",
subset="training",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(100,100))


def append_ext(fn):
    return fn+".png"

def append_ext_thum(fn):
    return fn+"_thumbnail.png"


traindf["image_id_path"]=traindf["image_id"].apply(append_ext)
traindf["image_id_path_thum"]=traindf["image_id"].apply(append_ext_thum)


testdf["image_id_path"]=testdf["image_id"].apply(append_ext)
testdf["image_id_path_thum"]=testdf["image_id"].apply(append_ext_thum)



traindf


testdf


testdf['Image_path'] = [os.path.join('/kaggle/input/UBC-OCEAN/test_images', image) for image in testdf['image_id_path']]
testdf['Image_path_thumbnails'] = [os.path.join('/kaggle/input/UBC-OCEAN/test_thumbnails', image) for image in testdf['image_id_path_thum']]
testdf


traindf['Image_path'] = [os.path.join('/kaggle/input/UBC-OCEAN/train_images', image) for image in traindf['image_id_path']]
traindf['Image_path_thumbnails'] = [os.path.join('/kaggle/input/UBC-OCEAN/train_thumbnails', image) for image in traindf['image_id_path_thum']]
traindf


full_path_random = np.random.choice(traindf['Image_path_thumbnails'],5)
full_path_random





def image_viewer(dataset, index, ax):
    image_path =  dataset['Image_path_thumbnails'][index]
    image      =  Image.open(image_path)
    ax.imshow(image)
    
def plot_some_images(dataset, title):
    fig, axs = plt.subplots(nrows = 1,ncols = 2,figsize=(20,8))
    for ind, ax in enumerate(axs.flat):
            index = random.randrange(len(dataset))
            image_viewer(dataset, index, ax)
            ax.set_title(dataset['label'][index], fontsize = 8)
            ax.axis('off')
            fig.suptitle(title, fontsize = 15)
    plt.show()



from PIL import Image
import matplotlib.pyplot as plt
import random
import os

def image_viewer(dataset, index, ax):
    image_path = dataset['Image_path_thumbnails'][index]
    
    if os.path.exists(image_path):
        image = Image.open(image_path)
        ax.imshow(image)
    else:
        ax.text(0.5, 0.5, "GÃ¶rsel bulunamadÄ±", ha='center', va='center', fontsize=8)
        ax.set_facecolor('lightgray')

def plot_some_images(dataset, title='EÄŸitim GÃ¶rselleri'):
    fig, axs = plt.subplots(3, 4, figsize=(12, 8))
    fig.suptitle(title, fontsize=16)
    
    for ind, ax in enumerate(axs.flat):
        index = random.randrange(len(dataset))
        image_viewer(dataset, index, ax)
        ax.set_title(str(dataset['label'][index]), fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

# Fonksiyonu Ã§aÄŸÄ±r:
plot_some_images(traindf, 'EÄŸitim GÃ¶rselleri')




from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight


class_weights = compute_class_weight(class_weight = "balanced",
                                     classes= np.unique(traindf['label']),
                                     y= traindf['label'])

classes = (np.unique(traindf['label']))
class_weights_forplot = dict(zip(classes, class_weights))


classes


class_weights_forplot


class_weights = dict(zip(range(43), class_weights))



import tensorflow as tf



train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
)
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)


train_images = train_generator.flow_from_dataframe(
    dataframe=traindf,
    x_col='Image_path_thumbnails',
    y_col= 'label',
    target_size=(256, 256),
    color_mode='grayscale',
    class_mode="categorical",
    batch_size=64,
    shuffle=True,
    seed=210,
)

test_images = test_generator.flow_from_dataframe(
    dataframe=traindf[0:10],
    x_col='Image_path_thumbnails',
    y_col= 'label',
    target_size=(256, 256),
    class_mode="categorical",
    color_mode='grayscale',
    batch_size=64,
    shuffle=False
)


# Kaggle'da yÃ¼klediÄŸin datasetin iÃ§indeki model aÄŸÄ±rlÄ±klarÄ± dosyasÄ±nÄ±n yolu
local_weights_path = '/kaggle/input/efficent/efficientnetb0_notop.h5'

# EfficientNetB0 modelini Ã¶nceden eÄŸitilmiÅŸ aÄŸÄ±rlÄ±klarla yÃ¼kle
trans_arc = tf.keras.applications.EfficientNetB0(
    weights=local_weights_path,       # Kaggle yolu
    include_top=False,
    input_shape=(256, 256, 3),
    pooling='max'
)

# KatmanlarÄ± dondur (transfer learning)
for l in trans_arc.layers:
    l.trainable = False

# Model katmanlarÄ±
inputs = trans_arc.input
flatten = trans_arc.output

x = tf.keras.layers.Dense(256, activation='relu')(flatten)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)

x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Modeli tanÄ±mla
model = tf.keras.Model(inputs=inputs, outputs=outputs)






model.summary()



#model mimarisi foto
from tensorflow.keras.utils import plot_model

plot_model(model, show_shapes=True, show_layer_names=True, to_file='model_mimari.png')



import os

checkpoint_path = "/kaggle/working/checkpoints"
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)



loss = [tf.keras.losses.binary_crossentropy]

initial_learning_rate = 0.005

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=82,
    decay_rate=0.9,
    staircase=True)

optimizer = tf.keras.optimizers.Adam(
    learning_rate= lr_schedule,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
)
metrics= ['accuracy']

model.compile(
    optimizer=optimizer,
    loss= loss,
    metrics=metrics
    )


train_images


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# ðŸ”„ Tahmin Ã¼ret ve gerÃ§ek etiketleri topla
y_true = []
y_pred = []

for i in range(len(test_images)):
    x_batch, y_batch = test_images[i]
    preds = model.predict(x_batch, verbose=0)
    
    y_true.extend(np.argmax(y_batch, axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# ðŸ“Š KonfÃ¼zyon matrisi
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(5), yticklabels=range(5))
plt.xlabel("Tahmin Edilen SÄ±nÄ±f")
plt.ylabel("GerÃ§ek SÄ±nÄ±f")
plt.title("Åžekil 3. KonfÃ¼zyon Matrisi - Modelin SÄ±nÄ±flandÄ±rma PerformansÄ±")
plt.show()

# ðŸ“„ SÄ±nÄ±f BazlÄ± Performans
print("ðŸ“„ SÄ±nÄ±f BazlÄ± Performans:\n")
print(classification_report(y_true, y_pred, digits=3))



print(type(test_images))



import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# GerÃ§ek ve tahmin etiketlerini boÅŸ listede topla
y_true = []
y_pred = []

# test_images bir generator, dolayÄ±sÄ±yla batch batch gezip tahmin alacaÄŸÄ±z
for images, labels in test_images:
    preds = model.predict(images)
    y_true.extend(np.argmax(labels, axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

    # EÄŸer sÄ±nÄ±rlÄ± sayÄ±da batch varsa, break ile durdurabiliriz
    if len(y_true) >= test_images.samples:  
        break

# KonfÃ¼zyon matrisi ve rapor
cm = confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_images.class_indices.keys(),
            yticklabels=test_images.class_indices.keys())
plt.xlabel("Tahmin Edilen SÄ±nÄ±f")
plt.ylabel("GerÃ§ek SÄ±nÄ±f")
plt.title("KonfÃ¼zyon Matrisi - Modelin PerformansÄ±")
plt.show()



import numpy as np

y_train = train_images.labels
y_test = test_images.labels

print(np.unique(y_train))
print(np.unique(y_test))
print("SÄ±nÄ±f sayÄ±sÄ±:", len(np.unique(y_train)))



print("Train sÄ±nÄ±f sayÄ±sÄ±:", len(train_images.class_indices))
print("Test sÄ±nÄ±f sayÄ±sÄ±:", len(test_images.class_indices))
print("Train class indices:", train_images.class_indices)
print("Test class indices:", test_images.class_indices)



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Rastgele Ã¶rnek veri oluÅŸtur (100 resim, 256x256x3, 3 sÄ±nÄ±f)
num_classes = 3
X_train = np.random.rand(100, 256, 256, 3).astype(np.float32)
y_train = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, 100), num_classes)

X_test = np.random.rand(30, 256, 256, 3).astype(np.float32)
y_test = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, 30), num_classes)

# Modeli oluÅŸtur (EfficientNetB0 transfer learning)
base_model = tf.keras.applications.EfficientNetB0(
    weights='imagenet', include_top=False, input_shape=(256, 256, 3), pooling='max'
)
base_model.trainable = False

inputs = base_model.input
x = tf.keras.layers.Dense(256, activation='relu')(base_model.output)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# EÄŸit
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

# DeÄŸerlendir
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Tahmin Edilen SÄ±nÄ±f")
plt.ylabel("GerÃ§ek SÄ±nÄ±f")
plt.title("KonfÃ¼zyon Matrisi")
plt.show()

print("\nSÄ±nÄ±f BazlÄ± Performans:\n")
print(classification_report(y_true, y_pred, digits=3))



# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])  # RAISE ERROR
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss']) #RAISE ERROR
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()





def plot_model_evaluation(model, test_data, n_classes, target_labels):

    results = model.evaluate(test_data, verbose=0)
    loss = results[0]
    acc = results[1]

    print("    Test Loss: {:.5f}".format(loss))
    print("Test Accuracy: {:.2f}%".format(acc * 100))

    y_pred = np.squeeze((model.predict(test_data) >= 0.5).astype(int))
    cm = confusion_matrix(test_data.labels, y_pred)
    clr = classification_report(test_data.labels, y_pred, target_names=target_labels)

    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
    plt.xticks(ticks=np.arange(n_classes) + 0.5, labels=list(test_data.class_indices.keys()), rotation=90)
    plt.yticks(ticks=np.arange(n_classes) + 0.5, labels=list(test_data.class_indices.keys()), rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    print("Classification Report:\n----------------------\n", clr)


outputs = tf.keras.layers.Dense(3, activation='softmax')(x)







