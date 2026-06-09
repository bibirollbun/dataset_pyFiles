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


import os, gc
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import tensorflow as tf
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
print('TensorFlow version =',tf.__version__)

# USE MULTIPLE GPUS
gpus = tf.config.list_physical_devices('GPU')
if len(gpus)<=1: 
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print(f'Using {len(gpus)} GPU')
else: 
    strategy = tf.distribute.MirroredStrategy()
    print(f'Using {len(gpus)} GPUs')

VER = 5

# IF THIS EQUALS NONE, THEN WE TRAIN NEW MODELS
# IF THIS EQUALS DISK PATH, THEN WE LOAD PREVIOUSLY TRAINED MODELS
LOAD_MODELS_FROM = '/kaggle/input/brain-efficientnet-models-v3-v4-v5/'

USE_KAGGLE_SPECTROGRAMS = True
USE_EEG_SPECTROGRAMS = True


# USE MIXED PRECISION
MIX = True
if MIX:
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    print('Mixed precision enabled')
else:
    print('Using full precision')


import os, gc
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import tensorflow as tf
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
print('TensorFlow version =',tf.__version__)


df = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')
TARGETS = df.columns[-6:]
print('Train shape:', df.shape )
print('Targets', list(TARGETS))
df.head()


train = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
    {'spectrogram_id':'first','spectrogram_label_offset_seconds':'min'})
train.columns = ['spec_id','min']

tmp = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
    {'spectrogram_label_offset_seconds':'max'})
train['max'] = tmp

tmp = df.groupby('eeg_id')[['patient_id']].agg('first')
train['patient_id'] = tmp

tmp = df.groupby('eeg_id')[TARGETS].agg('sum')
for t in TARGETS:
    train[t] = tmp[t].values
    
y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1,keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby('eeg_id')[['expert_consensus']].agg('first')
train['target'] = tmp

train = train.reset_index()
print('Train non-overlapp eeg_id shape:', train.shape )
train.head()


%%time
READ_SPEC_FILES = False

# READ ALL SPECTROGRAMS
PATH = '/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/'
files = os.listdir(PATH)
print(f'There are {len(files)} spectrogram parquets')

if READ_SPEC_FILES:    
    spectrograms = {}
    for i,f in enumerate(files):
        if i%100==0: print(i,', ',end='')
        tmp = pd.read_parquet(f'{PATH}{f}')
        name = int(f.split('.')[0])
        spectrograms[name] = tmp.iloc[:,1:].values
else:
    spectrograms = np.load('/kaggle/input/brain-spectrograms/specs.npy',allow_pickle=True).item()


# %%time
# READ_SPEC_FILES = False

# # READ ALL SPECTROGRAMS
# PATH = '/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/'
# files = os.listdir(PATH)
# print(f'There are {len(files)} spectrogram parquets')

# if READ_SPEC_FILES:    
#     spectrograms = {}
#     for i,f in enumerate(files):
#         if i%100==0: print(i,', ',end='')
#         tmp = pd.read_parquet(f'{PATH}{f}')
#         name = int(f.split('.')[0])
#         spectrograms[name] = tmp.iloc[:,1:].values
# else:
#     spectrograms = np.load('/kaggle/input/brain-spectrograms/specs.npy',allow_pickle=True).item()


%%time
READ_EEG_SPEC_FILES = True

if READ_EEG_SPEC_FILES:
    all_eegs = {}
    for i,e in enumerate(train.eeg_id.values):
        if i%100==0: print(i,', ',end='')
        x = np.load(f'/kaggle/input/preprocessing/preprocessed/spec/{e}.npy')
        all_eegs[e] = x
else:
    all_eegs = np.load('/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy',allow_pickle=True).item()


# %%time
# READ_EEG_SPEC_FILES = False

# if READ_EEG_SPEC_FILES:
#     all_eegs = {}
#     for i,e in enumerate(train.eeg_id.values):
#         if i%100==0: print(i,', ',end='')
#         x = np.load(f'/kaggle/input/brain-eeg-spectrograms/EEG_Spectrograms/{e}.npy')
#         all_eegs[e] = x
# else:
#     all_eegs = np.load('/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy',allow_pickle=True).item()


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt


# import albumentations as albu
# TARS = {'Seizure':0, 'LPD':1, 'GPD':2, 'LRDA':3, 'GRDA':4, 'Other':5}
# TARS2 = {x:y for y,x in TARS.items()}

# class DataGenerator(tf.keras.utils.Sequence):
#     'Generates data for Keras'
#     def __init__(self, data, batch_size=32, shuffle=False, augment=False, mode='train',
#                  specs = spectrograms, eeg_specs = all_eegs): 

#         self.data = data
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.augment = augment
#         self.mode = mode
#         self.specs = specs
#         self.eeg_specs = eeg_specs
#         self.on_epoch_end()
        
#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         ct = int( np.ceil( len(self.data) / self.batch_size ) )
#         return ct

#     def __getitem__(self, index):
#         'Generate one batch of data'
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#         X, y = self.__data_generation(indexes)
#         if self.augment: X = self.__augment_batch(X) 
#         return X, y

#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange( len(self.data) )
#         if self.shuffle: np.random.shuffle(self.indexes)
                        
#     def __data_generation(self, indexes):
#         'Generates data containing batch_size samples' 
        
#         X = np.zeros((len(indexes),128,256,8),dtype='float32')
#         y = np.zeros((len(indexes),6),dtype='float32')
#         img = np.ones((128,256),dtype='float32')
        
#         for j,i in enumerate(indexes):
#             row = self.data.iloc[i]
#             if self.mode=='test': 
#                 r = 0
#             else: 
#                 r = int( (row['min'] + row['max'])//4 )

#             for k in range(4):
#                 # EXTRACT 300 ROWS OF SPECTROGRAM
#                 img = self.specs[row.spec_id][r:r+300,k*100:(k+1)*100].T
                
#                 # LOG TRANSFORM SPECTROGRAM
#                 img = np.clip(img,np.exp(-4),np.exp(8))
#                 img = np.log(img)
                
#                 # STANDARDIZE PER IMAGE
#                 ep = 1e-6
#                 m = np.nanmean(img.flatten())
#                 s = np.nanstd(img.flatten())
#                 img = (img-m)/(s+ep)
#                 img = np.nan_to_num(img, nan=0.0)
                
#                 # CROP TO 256 TIME STEPS
#                 X[j,14:-14,:,k] = img[:,22:-22] / 2.0
        
#             # EEG SPECTROGRAMS
#             img = self.eeg_specs[row.eeg_id]
#             X[j,:,:,4:] = img
                
#             if self.mode!='test':
#                 y[j,] = row[TARGETS]
            
#         return X,y
    
#     def __random_transform(self, img):
#         composition = albu.Compose([
#             albu.HorizontalFlip(p=0.5),
#             #albu.CoarseDropout(max_holes=8,max_height=32,max_width=32,fill_value=0,p=0.5),
#         ])
#         return composition(image=img)['image']
            
#     def __augment_batch(self, img_batch):
#         for i in range(img_batch.shape[0]):
#             img_batch[i, ] = self.__random_transform(img_batch[i, ])
#         return img_batch


import albumentations as albu
import cv2  # Required for resizing

TARS = {'Seizure':0, 'LPD':1, 'GPD':2, 'LRDA':3, 'GRDA':4, 'Other':5}
TARS2 = {x:y for y,x in TARS.items()}

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, batch_size=32, shuffle=False, augment=False, mode='train',
                 specs=spectrograms, eeg_specs=all_eegs): 
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.mode = mode
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)
        if self.augment:
            X = self.__augment_batch(X)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        X = np.zeros((len(indexes), 128, 256, 8), dtype='float32')
        y = np.zeros((len(indexes), 6), dtype='float32')
        
        for j, i in enumerate(indexes):
            row = self.data.iloc[i]
            r = 0 if self.mode == 'test' else int((row['min'] + row['max']) // 4)

            for k in range(4):
                img = self.specs[row.spec_id][r:r+300, k*100:(k+1)*100].T
                img = np.clip(img, np.exp(-4), np.exp(8))
                img = np.log(img)
                m = np.nanmean(img.flatten())
                s = np.nanstd(img.flatten())
                img = (img - m) / (s + 1e-6)
                img = np.nan_to_num(img, nan=0.0)
                X[j, 14:-14, :, k] = img[:, 22:-22] / 2.0

            # EEG SPECTROGRAMS (with resizing to (128, 256, 4))
            img = self.eeg_specs[row.eeg_id]

            if img.shape != (128, 256, 4):
                resized_img = np.zeros((128, 256, 4), dtype='float32')
                for ch in range(min(4, img.shape[-1])):
                    resized_img[:, :, ch] = cv2.resize(img[:, :, ch], (256, 128), interpolation=cv2.INTER_LINEAR)
                img = resized_img

            X[j, :, :, 4:] = img

            if self.mode != 'test':
                y[j,] = row[TARGETS]

        return X, y

    def __random_transform(self, img):
        composition = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            # albu.CoarseDropout(...) can be re-enabled here if needed
        ])
        return composition(image=img)['image']

    def __augment_batch(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i, ] = self.__random_transform(img_batch[i, ])
        return img_batch



gen = DataGenerator(train, batch_size=32, shuffle=False)
ROWS=2; COLS=3; BATCHES=2

for i,(x,y) in enumerate(gen):
    plt.figure(figsize=(20,8))
    for j in range(ROWS):
        for k in range(COLS):
            plt.subplot(ROWS,COLS,j*COLS+k+1)
            t = y[j*COLS+k]
            img = x[j*COLS+k,:,:,0][::-1,]
            mn = img.flatten().min()
            mx = img.flatten().max()
            img = (img-mn)/(mx-mn)
            plt.imshow(img)
            tars = f'[{t[0]:0.2f}'
            for s in t[1:]: tars += f', {s:0.2f}'
            eeg = train.eeg_id.values[i*32+j*COLS+k]
            plt.title(f'EEG = {eeg}\nTarget = {tars}',size=12)
            plt.yticks([])
            plt.ylabel('Frequencies (Hz)',size=14)
            plt.xlabel('Time (sec)',size=16)
    plt.show()
    if i==BATCHES-1: break


import math
LR_START = 1e-6
LR_MAX = 1e-3
LR_MIN = 1e-6
LR_RAMPUP_EPOCHS = 0
LR_SUSTAIN_EPOCHS = 0
EPOCHS2 = 10

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        decay_total_epochs = EPOCHS2 - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS - 1
        decay_epoch_index = epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        phase = math.pi * decay_epoch_index / decay_total_epochs
        cosine_decay = 0.5 * (1 + math.cos(phase))
        lr = (LR_MAX - LR_MIN) * cosine_decay + LR_MIN
    return lr

rng = [i for i in range(EPOCHS2)]
lr_y = [lrfn(x) for x in rng]
plt.figure(figsize=(10, 4))
plt.plot(rng, lr_y, '-o')
plt.xlabel('epoch',size=14); plt.ylabel('learning rate',size=14)
plt.title('Cosine Training Schedule',size=16); plt.show()

LR2 = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)


LR_START = 1e-4
LR_MAX = 1e-3
LR_RAMPUP_EPOCHS = 0
LR_SUSTAIN_EPOCHS = 1
LR_STEP_DECAY = 0.1
EVERY = 1
EPOCHS = 4

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = LR_MAX * LR_STEP_DECAY**((epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS)//EVERY)
    return lr

rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]
plt.figure(figsize=(10, 4))
plt.plot(rng, y, 'o-'); 
plt.xlabel('epoch',size=14); plt.ylabel('learning rate',size=14)
plt.title('Step Training Schedule',size=16); plt.show()

LR = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)


!pip install --no-index --find-links=/kaggle/input/tf-efficientnet-whl-files /kaggle/input/tf-efficientnet-whl-files/efficientnet-1.1.1-py3-none-any.whl


import efficientnet.tfkeras as efn

def build_model():
    
    inp = tf.keras.Input(shape=(128,256,8))
    base_model = efn.EfficientNetB0(include_top=False, weights=None, input_shape=None)
    base_model.load_weights('/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5')
    
    # RESHAPE INPUT 128x256x8 => 512x512x3 MONOTONE IMAGE
    # KAGGLE SPECTROGRAMS
    x1 = [inp[:,:,:,i:i+1] for i in range(4)]
    x1 = tf.keras.layers.Concatenate(axis=1)(x1)
    # EEG SPECTROGRAMS
    x2 = [inp[:,:,:,i+4:i+5] for i in range(4)]
    x2 = tf.keras.layers.Concatenate(axis=1)(x2)
    # MAKE 512X512X3
    if USE_KAGGLE_SPECTROGRAMS & USE_EEG_SPECTROGRAMS:
        x = tf.keras.layers.Concatenate(axis=2)([x1,x2])
    elif USE_EEG_SPECTROGRAMS: x = x2
    else: x = x1
    x = tf.keras.layers.Concatenate(axis=3)([x,x,x])
    
    # OUTPUT
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(6,activation='softmax', dtype='float32')(x)
        
    # COMPILE MODEL
    model = tf.keras.Model(inputs=inp, outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate = 1e-3)
    loss = tf.keras.losses.KLDivergence()

    model.compile(loss=loss, optimizer = opt) 
        
    return model


from sklearn.model_selection import KFold, GroupKFold
import tensorflow.keras.backend as K, gc

all_oof = []
all_true = []

gkf = GroupKFold(n_splits=5)
for i, (train_index, valid_index) in enumerate(gkf.split(train, train.target, train.patient_id)):  
    
    print('#'*25)
    print(f'### Fold {i+1}')
    
    train_gen = DataGenerator(train.iloc[train_index], shuffle=True, batch_size=32, augment=False)
    valid_gen = DataGenerator(train.iloc[valid_index], shuffle=False, batch_size=64, mode='valid')
    
    print(f'### train size {len(train_index)}, valid size {len(valid_index)}')
    print('#'*25)
    
    K.clear_session()
    with strategy.scope():
        model = build_model()
    if LOAD_MODELS_FROM is None:
        model.fit(train_gen, verbose=1,
              validation_data = valid_gen,
              epochs=EPOCHS, callbacks = [LR])
        model.save_weights(f'EffNet_v{VER}_f{i}.h5')
    else:
        model.load_weights(f'{LOAD_MODELS_FROM}EffNet_v{VER}_f{i}.h5')
        
    oof = model.predict(valid_gen, verbose=1)
    all_oof.append(oof)
    all_true.append(train.iloc[valid_index][TARGETS].values)
    
    del model, oof
    gc.collect()
    
all_oof = np.concatenate(all_oof)
all_true = np.concatenate(all_true)


!ls -l /kaggle/input/kaggle-kl-div/


import importlib.util
import sys

# Import kaggle_metric_utilities
util_path = '/kaggle/input/kaggle-kl-div/kaggle_metric_utilities.py'
util_spec = importlib.util.spec_from_file_location("kaggle_metric_utilities", util_path)
kaggle_metric_utilities = importlib.util.module_from_spec(util_spec)
sys.modules["kaggle_metric_utilities"] = kaggle_metric_utilities
util_spec.loader.exec_module(kaggle_metric_utilities)

# Now import kaggle_kl_div (after utility is loaded)
kl_path = '/kaggle/input/kaggle-kl-div/kaggle_kl_div.py'
kl_spec = importlib.util.spec_from_file_location("kaggle_kl_div", kl_path)
kaggle_kl_div = importlib.util.module_from_spec(kl_spec)
sys.modules["kaggle_kl_div"] = kaggle_kl_div
kl_spec.loader.exec_module(kaggle_kl_div)

# Grab the score function
score = kaggle_kl_div.score


oof = pd.DataFrame(all_oof.copy())
oof['id'] = np.arange(len(oof))

true = pd.DataFrame(all_true.copy())
true['id'] = np.arange(len(true))

cv = score(solution=true, submission=oof, row_id_column_name='id')
print('CV Score KL-Div for EfficientNetB2 =', cv)


import importlib.util
import sys
import numpy as np
import pandas as pd

# Load kaggle_metric_utilities
util_path = '/kaggle/input/kaggle-kl-div/kaggle_metric_utilities.py'
util_spec = importlib.util.spec_from_file_location("kaggle_metric_utilities", util_path)
kaggle_metric_utilities = importlib.util.module_from_spec(util_spec)
sys.modules["kaggle_metric_utilities"] = kaggle_metric_utilities
util_spec.loader.exec_module(kaggle_metric_utilities)

# Load kaggle_kl_div
kl_path = '/kaggle/input/kaggle-kl-div/kaggle_kl_div.py'
kl_spec = importlib.util.spec_from_file_location("kaggle_kl_div", kl_path)
kaggle_kl_div = importlib.util.module_from_spec(kl_spec)
sys.modules["kaggle_kl_div"] = kaggle_kl_div
kl_spec.loader.exec_module(kaggle_kl_div)

# Grab the score function
score = kaggle_kl_div.score

# Wrap predictions into DataFrames
oof = pd.DataFrame(all_oof.copy())
oof['id'] = np.arange(len(oof))

true = pd.DataFrame(all_true.copy())
true['id'] = np.arange(len(true))

# KL-Divergence Score
cv = score(solution=true, submission=oof, row_id_column_name='id')
print('CV Score KL-Div for EfficientNetB2 =', cv)

# Accuracy
pred_labels = np.argmax(all_oof, axis=1)
true_labels = np.argmax(all_true, axis=1)
accuracy = np.mean(pred_labels == true_labels)
print('Accuracy =', accuracy)





