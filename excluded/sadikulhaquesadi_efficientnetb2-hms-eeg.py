import os, gc
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import tensorflow as tf
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
print('TensorFlow version =',tf.__version__)

# USE MULTIPLE GPUS
gpus = tf.config.list_physical_devices('GPU')
if len(gpus)<=1: 
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print(f'Using {len(gpus)} GPU')
else: 
    strategy = tf.distribute.MirroredStrategy()
    print(f'Using {len(gpus)} GPUs')

VER = 3.3

# IF THIS EQUALS NONE, THEN WE TRAIN NEW MODELS
# IF THIS EQUALS DISK PATH, THEN WE LOAD PREVIOUSLY TRAINED MODELS
LOAD_MODELS_FROM = None #'/kaggle/input/effnet_strat_freqshift_all/tensorflow2/default/1/' #'/kaggle/input/effnet_v3.3_stratified/tensorflow2/default/1/' #'/kaggle/input/effnetb0_all_v3.2/tensorflow2/default/1/'

USE_KAGGLE_SPECTROGRAMS = True
USE_EEG_SPECTROGRAMS = True
SEED = 42


tf.random.set_seed(SEED)
np.random.seed(SEED)


# USE MIXED PRECISION
MIX = True
if MIX:
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    print('Mixed precision enabled')
else:
    print('Using full precision')


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

sum_targets = tmp.sum(axis=1)
max_vote_percentage = tmp.max(axis=1) / sum_targets
train['max_vote_percentage'] = max_vote_percentage

for t in TARGETS:
    train[t] = tmp[t].values
    
y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1,keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby('eeg_id')[['expert_consensus']].agg('first')
train['target'] = tmp

train = train.reset_index()
# train = train[train['max_vote_percentage']>=.9]
print('Train non-overlapp eeg_id shape:', train.shape )
train.head()


# # Compute class weights based on the training data
# def compute_class_weights(y_data):
#     # Convert y_data (probabilities) to class labels
#     y_classes = np.argmax(y_data, axis=1)
#     classes = np.arange(6)  # 6 classes
#     class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_classes)
#     return dict(zip(classes, class_weights))


# min_class_count = train['target'].value_counts().min()

# # Collect balanced subsets
# balanced_train_parts = []

# for cls in train['target'].unique():
#     cls_data = train[train['target'] == cls]
#     sampled_data = cls_data.sample(n=min_class_count, random_state=42)
#     balanced_train_parts.append(sampled_data)

# train = pd.concat(balanced_train_parts).sample(frac=1, random_state=42).reset_index(drop=True)

# print("Balanced class distribution:\n", train['target'].value_counts())


%%time

spectrograms = np.load('/kaggle/input/brain-spectrograms/specs.npy',allow_pickle=True).item()


%%time

all_eegs = np.load('/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy',allow_pickle=True).item()


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, batch_size=32, shuffle=False, augment=False, mode='train',
                 specs = spectrograms, eeg_specs = all_eegs): 

        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.mode = mode
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        ct = int( np.ceil( len(self.data) / self.batch_size ) )
        return ct

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)
        if self.augment:
            X = self.__augment_batch(X)
            X, y = self.__mixup(X, y)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange( len(self.data) )
        if self.shuffle: np.random.shuffle(self.indexes)
                        
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' 
        
        X = np.zeros((len(indexes),128,256,8),dtype='float32')
        y = np.zeros((len(indexes),6),dtype='float32')
        img = np.ones((128,256),dtype='float32')
        
        for j,i in enumerate(indexes):
            row = self.data.iloc[i]
            if self.mode=='test': 
                r = 0
            else: 
                r = int( (row['min'] + row['max'])//4 )

            for k in range(4):
                # EXTRACT 300 ROWS OF SPECTROGRAM
                img = self.specs[row.spec_id][r:r+300,k*100:(k+1)*100].T
                
                # LOG TRANSFORM SPECTROGRAM
                img = np.clip(img,np.exp(-4),np.exp(8))
                img = np.log(img)
                
                # STANDARDIZE PER IMAGE
                ep = 1e-6
                m = np.nanmean(img.flatten())
                s = np.nanstd(img.flatten())
                img = (img-m)/(s+ep)
                img = np.nan_to_num(img, nan=0.0)
                
                # CROP TO 256 TIME STEPS
                X[j,14:-14,:,k] = img[:,22:-22] / 2.0
        
            # EEG SPECTROGRAMS
            img = self.eeg_specs[row.eeg_id]
            X[j,:,:,4:] = img
                
            if self.mode!='test':
                y[j,] = row[TARGETS]
            
        return X,y

    def __random_transform(self, img):
        # Apply Frequency Shift (±3 bins) with 50% probability
        if np.random.rand() < 0.5:
            shift = np.random.choice([-3, 3])
            img = np.roll(img, shift=shift, axis=0)        
    
        return img

    def __mixup(self, X, y, alpha=0.2):
        # Mixup augmentation: create synthetic samples by linear interpolation
        if X.shape[0] <= 1:
            return X, y
        lam = np.random.beta(alpha, alpha, X.shape[0])
        indices = np.random.permutation(X.shape[0])
        X_mixed = X * lam[:, None, None, None] + X[indices] * (1 - lam[:, None, None, None])
        y_mixed = y * lam[:, None] + y[indices] * (1 - lam[:, None])
        
        return X_mixed, y_mixed
            
    def __augment_batch(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i, ] = self.__random_transform(img_batch[i, ])
        return img_batch


from tensorflow.keras.callbacks import ReduceLROnPlateau

LR_START = 1e-4
LR_MAX = 1e-3
LR_RAMPUP_EPOCHS = 0
LR_SUSTAIN_EPOCHS = 1
LR_STEP_DECAY = 0.1
EVERY = 1
EPOCHS = 8

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = LR_MAX * LR_STEP_DECAY**((epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS)//EVERY)
    return lr

# rng = [i for i in range(EPOCHS)]
# y = [lrfn(x) for x in rng]
# plt.figure(figsize=(10, 4))
# plt.plot(rng, y, 'o-'); 
# plt.xlabel('epoch',size=14); plt.ylabel('learning rate',size=14)
# plt.title('Step Training Schedule',size=16); plt.show()

LR = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)

# LR = ReduceLROnPlateau(
#     monitor='val_loss',  # Metric to monitor
#     factor=0.5,          # Factor by which the learning rate will be reduced. new_lr = lr * factor
#     patience=1,          # Number of epochs with no improvement after which learning rate will be reduced.
#                          # Since you only have 4 epochs, patience=1 might be more reactive, or patience=1
#                          # to give it a chance before reducing.
#     min_lr=1e-6,         # Lower bound on the learning rate.
#     verbose=1
# )


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)


# def focal_loss(gamma=2.0, alpha=0.25):
#     def focal_loss_fn(y_true, y_pred):
#         y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
#         cross_entropy = -y_true * tf.math.log(y_pred)
#         weight = alpha * y_true * tf.pow((1 - y_pred), gamma)
#         loss = weight * cross_entropy
#         return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
#     return focal_loss_fn


!pip install --no-index --find-links=/kaggle/input/tf-efficientnet-whl-files /kaggle/input/tf-efficientnet-whl-files/efficientnet-1.1.1-py3-none-any.whl


# !pip install efficientnet


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
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(6,activation='softmax', dtype='float32')(x)
        
    # COMPILE MODEL
    model = tf.keras.Model(inputs=inp, outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate = 1e-3)
    loss = tf.keras.losses.KLDivergence() #focal_loss(gamma=1.0, alpha=0.5)

    model.compile(loss=loss, optimizer = opt, metrics='accuracy') 
        
    return model


from sklearn.model_selection import KFold, GroupKFold, StratifiedGroupKFold
import tensorflow.keras.backend as K, gc

all_oof = []
all_true = []

# gkf = GroupKFold(n_splits=5)
gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
for i, (train_index, valid_index) in enumerate(gkf.split(train, train.target, train.patient_id)):  
    
    print('#'*25)
    print(f'### Fold {i+1}')

    train_fold = train.iloc[train_index].copy()
    # class_weights = compute_class_weights(train_fold[TARGETS].values)

    # Get class labels for undersampling
    y_true_fold_train = np.argmax(train_fold[TARGETS].values, axis=1)
    class_counts_fold = np.bincount(y_true_fold_train, minlength=len(TARGETS))
    min_class_size = min(class_counts_fold)  # ~733 for LRDA
    multiplier = 3
    
    class_dfs = []
    for cls_idx in range(len(TARGETS)):
        cls_df = train_fold[y_true_fold_train == cls_idx]
        target_size = min(len(cls_df), min_class_size * multiplier)  # Cap at 1,466 unless smaller
        if len(cls_df) > target_size:
            cls_df = resample(cls_df, replace=False, n_samples=target_size, random_state=SEED)
        class_dfs.append(cls_df)
    
    # Concatenate undersampled data
    undersampled_train_df = pd.concat(class_dfs).sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    print(f"Undersampled train size: {len(undersampled_train_df)}")

    
    train_gen = DataGenerator(undersampled_train_df, shuffle=True, batch_size=32, augment=True)
    valid_gen = DataGenerator(train.iloc[valid_index], shuffle=False, batch_size=64, mode='valid')
    
    print(f'### train size {len(undersampled_train_df)}, valid size {len(valid_index)}')
    print('#'*25)
    
    K.clear_session()
    with strategy.scope():
        model = build_model()
    if LOAD_MODELS_FROM is None:
        model.fit(train_gen, verbose=1,
                  validation_data = valid_gen,
                  epochs=EPOCHS, callbacks = [LR, early_stopping],
                  # class_weight = class_weights
                 )
        model.save_weights(f'EffNetV2B1_v{VER}_f{i}.h5')
    else:
        model.load_weights(f'{LOAD_MODELS_FROM}EffNet_v{VER}_f{i}.h5')
        
    oof = model.predict(valid_gen, verbose=1)
    all_oof.append(oof)
    all_true.append(train.iloc[valid_index][TARGETS].values)

    y_pred = np.argmax(oof, axis=1)
    y_true = np.argmax(train.iloc[valid_index][TARGETS].values, axis=1)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=TARGETS))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=TARGETS, yticklabels=TARGETS)
    plt.title(f'Confusion Matrix - Fold {i+1}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    del model, oof
    gc.collect()
    
all_oof = np.concatenate(all_oof)
all_true = np.concatenate(all_true)


import sys
sys.path.append('/kaggle/input/kaggle-kl-div')
from kaggle_kl_div import score

oof = pd.DataFrame(all_oof.copy())
oof['id'] = np.arange(len(oof))

true = pd.DataFrame(all_true.copy())
true['id'] = np.arange(len(true))

cv = score(solution=true, submission=oof, row_id_column_name='id')
print('CV Score KL-Div for EfficientNetB2 =',cv)


# del all_eegs, spectrograms; gc.collect()
# test = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/test.csv')
# print('Test shape',test.shape)
# test.head()


# # READ ALL SPECTROGRAMS
# PATH2 = '/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/'
# files2 = os.listdir(PATH2)
# print(f'There are {len(files2)} test spectrogram parquets')
    
# spectrograms2 = {}
# for i,f in enumerate(files2):
#     if i%100==0: print(i,', ',end='')
#     tmp = pd.read_parquet(f'{PATH2}{f}')
#     name = int(f.split('.')[0])
#     spectrograms2[name] = tmp.iloc[:,1:].values
    
# # RENAME FOR DATALOADER
# test = test.rename({'spectrogram_id':'spec_id'},axis=1)


# import pywt, librosa

# USE_WAVELET = None 

# NAMES = ['LL','LP','RP','RR']

# FEATS = [['Fp1','F7','T3','T5','O1'],
#          ['Fp1','F3','C3','P3','O1'],
#          ['Fp2','F8','T4','T6','O2'],
#          ['Fp2','F4','C4','P4','O2']]

# # DENOISE FUNCTION
# def maddest(d, axis=None):
#     return np.mean(np.absolute(d - np.mean(d, axis)), axis)

# def denoise(x, wavelet='haar', level=1):    
#     coeff = pywt.wavedec(x, wavelet, mode="per")
#     sigma = (1/0.6745) * maddest(coeff[-level])

#     uthresh = sigma * np.sqrt(2*np.log(len(x)))
#     coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

#     ret=pywt.waverec(coeff, wavelet, mode='per')
    
#     return ret

# def spectrogram_from_eeg(parquet_path, display=False):
    
#     # LOAD MIDDLE 50 SECONDS OF EEG SERIES
#     eeg = pd.read_parquet(parquet_path)
#     middle = (len(eeg)-10_000)//2
#     eeg = eeg.iloc[middle:middle+10_000]
    
#     # VARIABLE TO HOLD SPECTROGRAM
#     img = np.zeros((128,256,4),dtype='float32')
    
#     if display: plt.figure(figsize=(10,7))
#     signals = []
#     for k in range(4):
#         COLS = FEATS[k]
        
#         for kk in range(4):
        
#             # COMPUTE PAIR DIFFERENCES
#             x = eeg[COLS[kk]].values - eeg[COLS[kk+1]].values

#             # FILL NANS
#             m = np.nanmean(x)
#             if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
#             else: x[:] = 0

#             # DENOISE
#             if USE_WAVELET:
#                 x = denoise(x, wavelet=USE_WAVELET)
#             signals.append(x)

#             # RAW SPECTROGRAM
#             mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x)//256, 
#                   n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)

#             # LOG TRANSFORM
#             width = (mel_spec.shape[1]//32)*32
#             mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:width]

#             # STANDARDIZE TO -1 TO 1
#             mel_spec_db = (mel_spec_db+40)/40 
#             img[:,:,k] += mel_spec_db
                
#         # AVERAGE THE 4 MONTAGE DIFFERENCES
#         img[:,:,k] /= 4.0
        
#         if display:
#             plt.subplot(2,2,k+1)
#             plt.imshow(img[:,:,k],aspect='auto',origin='lower')
#             plt.title(f'EEG {eeg_id} - Spectrogram {NAMES[k]}')
            
#     if display: 
#         plt.show()
#         plt.figure(figsize=(10,5))
#         offset = 0
#         for k in range(4):
#             if k>0: offset -= signals[3-k].min()
#             plt.plot(range(10_000),signals[k]+offset,label=NAMES[3-k])
#             offset += signals[3-k].max()
#         plt.legend()
#         plt.title(f'EEG {eeg_id} Signals')
#         plt.show()
#         print(); print('#'*25); print()
        
#     return img


# # READ ALL EEG SPECTROGRAMS
# PATH2 = '/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/'
# DISPLAY = 1
# EEG_IDS2 = test.eeg_id.unique()
# all_eegs2 = {}

# print('Converting Test EEG to Spectrograms...'); print()
# for i,eeg_id in enumerate(EEG_IDS2):
        
#     # CREATE SPECTROGRAM FROM EEG PARQUET
#     img = spectrogram_from_eeg(f'{PATH2}{eeg_id}.parquet', i<DISPLAY)
#     all_eegs2[eeg_id] = img


# # INFER EFFICIENTNET ON TEST
# preds = []
# model = build_model()
# test_gen = DataGenerator(test, shuffle=False, batch_size=64, mode='test',
#                          specs = spectrograms2, eeg_specs = all_eegs2)

# for i in range(5):
#     print(f'Fold {i+1}')
#     if LOAD_MODELS_FROM:
#         model.load_weights(f'{LOAD_MODELS_FROM}EffNet_v{VER}_f{i}.h5')
#     else:
#         model.load_weights(f'EffNet_v{VER}_f{i}.h5')
#     pred = model.predict(test_gen, verbose=1)
#     preds.append(pred)
# pred = np.mean(preds,axis=0)
# print()
# print('Test preds shape',pred.shape)


# sub = pd.DataFrame({'eeg_id':test.eeg_id.values})
# sub[TARGETS] = pred
# sub.to_csv('submission.csv',index=False)
# print('Submissionn shape',sub.shape)
# sub.head()


# SANITY CHECK TO CONFIRM PREDICTIONS SUM TO ONE
# sub.iloc[:,-6:].sum(axis=1)

