import os, gc
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import tensorflow as tf
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
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

VER = 1

# IF THIS EQUALS NONE, THEN WE TRAIN NEW MODELS
# IF THIS EQUALS DISK PATH, THEN WE LOAD PREVIOUSLY TRAINED MODELS
LOAD_MODELS_FROM = None #'/kaggle/input/brain-efficientnet-models-v3-v4-v5/'

USE_KAGGLE_SPECTROGRAMS = False
USE_EEG_SPECTROGRAMS = True


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
for t in TARGETS:
    train[t] = tmp[t].values
    
y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1,keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby('eeg_id')[TARGETS].agg('sum')
sum_targets = tmp.sum(axis=1)
max_vote_percentage = tmp.max(axis=1) / sum_targets
train['max_vote_percentage'] = max_vote_percentage


tmp = df.groupby('eeg_id')[['expert_consensus']].agg('first')
train['target'] = tmp

train = train.reset_index()
print('Train non-overlapp eeg_id shape:', train.shape )
train.head()


train = train[train['max_vote_percentage']>=.9]


from sklearn.preprocessing import LabelEncoder

# Encode string labels to integers
le = LabelEncoder()
train['target_encoded'] = le.fit_transform(train['target'])
# print(np.unique(train['target_encoded']))


class_names = ['GPD', 'GRDA', 'LPD', 'LRDA', 'Other', 'Seizure']
class_to_idx = {name: i for i, name in enumerate(class_names)}


classes = np.unique(train['target_encoded'])
class_weights = compute_class_weight(class_weight='balanced',
                                                  classes=classes,
                                                  y=train['target_encoded'])
class_weight_dict = dict(zip(classes, class_weights))


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
# READ_EEG_SPEC_FILES = True

# if READ_EEG_SPEC_FILES:
all_eegs = {}
for i,e in enumerate(train.eeg_id.values):
    if i%100==0: print(i,', ',end='')
    # x = np.load(f'/kaggle/input/brain-eeg-spectrograms/EEG_Spectrograms/{e}.npy')
    x = np.load(f'/kaggle/input/brain-spectrogram-128-256-90/spectrograms_all_128_256/{e}.npy')
    all_eegs[e] = x
# else:
#     all_eegs = np.load('/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy',allow_pickle=True).item()


import albumentations as albu

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, batch_size=32, shuffle=False, augment=False, mode='train',
                 # specs = spectrograms, 
                 eeg_specs = all_eegs, label_col='target_encoded', one_hot=False): 

        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.mode = mode
        # self.specs = specs
        self.eeg_specs = eeg_specs
        self.label_col = label_col
        self.one_hot = one_hot
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        ct = int( np.ceil( len(self.data) / self.batch_size ) )
        return ct

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)
        if self.augment: X = self.__augment_batch(X) 
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange( len(self.data) )
        if self.shuffle: np.random.shuffle(self.indexes)
                        
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' 
        
        X = np.zeros((len(indexes),128,256,4),dtype='float32')
        y = np.zeros((len(indexes),),dtype='int32')
        # img = np.ones((128,256),dtype='float32')
        
        for j, i in enumerate(indexes):
            row = self.data.iloc[i]
            X[j] = self.eeg_specs[row.eeg_id]
            y[j] = row[self.label_col]
            # print('y_j ', y[j])

        return X,y
    
    def __random_transform(self, img):
        composition = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.GaussNoise(var_limit=(1.0, 10.0), p=0.3),
            albu.CoarseDropout(max_holes=8,max_height=32,max_width=32,fill_value=0,p=0.5),
        ])
        return composition(image=img)['image']
            
    def __augment_batch(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i, ] = self.__random_transform(img_batch[i, ])
        return img_batch


# gen = DataGenerator(train, batch_size=32, shuffle=False)
# ROWS=2; COLS=3; BATCHES=2

# for i,(x,y) in enumerate(gen):
#     plt.figure(figsize=(20,8))
#     for j in range(ROWS):
#         for k in range(COLS):
#             plt.subplot(ROWS,COLS,j*COLS+k+1)
#             t = y[j*COLS+k]
#             img = x[j*COLS+k,:,:,0][::-1,]
#             mn = img.flatten().min()
#             mx = img.flatten().max()
#             img = (img-mn)/(mx-mn)
#             plt.imshow(img)
#             tars = f'[{t[0]:0.2f}'
#             for s in t[1:]: tars += f', {s:0.2f}'
#             eeg = train.eeg_id.values[i*32+j*COLS+k]
#             plt.title(f'EEG = {eeg}\nTarget = {tars}',size=12)
#             plt.yticks([])
#             plt.ylabel('Frequencies (Hz)',size=14)
#             plt.xlabel('Time (sec)',size=16)
#     plt.show()
#     if i==BATCHES-1: break


LR_START = 3e-4
LR_MAX = 9e-4
LR_RAMPUP_EPOCHS = 0
LR_SUSTAIN_EPOCHS = 1
LR_STEP_DECAY = 0.1
EVERY = 1
EPOCHS = 5

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


def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        # Convert labels to one-hot if not already
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, depth=y_pred.shape[-1])
        
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return loss


import efficientnet.tfkeras as efn
from tensorflow.keras import metrics

def build_model():
    
    inp = tf.keras.Input(shape=(128,256,4))
    base_model = efn.EfficientNetB0(include_top=False, weights=None, input_shape=None)
    base_model.load_weights('/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5')
    
    x2 = [inp[:,:,:,i:i+1] for i in range(4)]
    x2 = tf.keras.layers.Concatenate(axis=1)(x2)
    
    x = x2
    x = tf.keras.layers.Concatenate(axis=3)([x,x,x])
    
    # OUTPUT
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(6,activation='softmax')(x)
        
    # COMPILE MODEL
    model = tf.keras.Model(inputs=inp, outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate = 1e-4)
    loss = tf.keras.losses.SparseCategoricalCrossentropy() #tf.keras.losses.KLDivergence()
    

    model.compile(loss=loss, optimizer = opt, 
                  metrics=[
                      'accuracy',
                      # metrics.Precision(name='precision'), 
                      # metrics.Recall(name='recall'),
                      # metrics.AUC(name='auc') 
                  ]) 
        
    return model


from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
import tensorflow.keras.backend as K, gc

# all_fold_preds = []
# all_fold_trues = []
# fold_metrics = []

# gkf = GroupKFold(n_splits=5)
# for i, (train_index, valid_index) in enumerate(gkf.split(train, train.target, train.patient_id)):  
    
#     print('#'*25)
#     print(f'### Fold {i+1}')
    
#     train_gen = DataGenerator(train.iloc[train_index], shuffle=True, batch_size=32, augment=True)
#     valid_gen = DataGenerator(train.iloc[valid_index], shuffle=False, batch_size=64, mode='valid')
    
#     print(f'### train size {len(train_index)}, valid size {len(valid_index)}')
#     print('#'*25)
#     for i, (x, y) in enumerate(valid_gen):
#         if i > 0: break  # just check first batch
#         print("Generator y:", y)
#         print("Direct access y:", train.iloc[valid_index]['target_encoded'].values)
    
#     K.clear_session()
#     with strategy.scope():
#         model = build_model()
#     if LOAD_MODELS_FROM is None:
#         model.fit(train_gen, verbose=1,
#               validation_data = valid_gen,
#               epochs=EPOCHS, callbacks = [LR])
#         model.save_weights(f'EffNet_v{VER}_f{i}.weights.h5')
#     else:
#         model.load_weights(f'{LOAD_MODELS_FROM}EffNet_v{VER}_f{i}.h5')
        
#     preds = model.predict(valid_gen)
#     preds_classes = np.argmax(preds, axis=1)
#     # true_classes = valid_gen['target_encoded'].values
#     true_classes = train.iloc[valid_index][TARGETS].values
#     true_classes_discrete = np.argmax(true_classes, axis=1)  # Convert back to discrete

#     cr = classification_report(true_classes_discrete, preds_classes)
#     # true_classes = train.iloc[valid_index]['target_encoded'].values
#     # print(pred_classes, true_classes)
#     # Metrics
#     # fold_auc = roc_auc_score(tf.one_hot(true_classes, depth=preds.shape[1]).numpy(), preds, average='macro')
#     fold_auc = roc_auc_score(true_classes, preds, average='macro')
    
#     print(f"Fold {i+1} Macro AUC: {fold_auc:.4f}")

#     cm = confusion_matrix(true_classes, preds_classes)
#     # cr = classification_report(true_classes, preds_classes, digits=4, output_dict=True)

#     print(f"Fold {i+1} Classification Report:")
#     print(cr)

#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.title(f'Confusion Matrix - Fold {i+1}')
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.show()

#     # Store fold-wise metrics
#     all_fold_preds.append(preds_classes)
#     all_fold_trues.append(true_classes)
#     fold_metrics.append({
#         "fold": i+1,
#         "macro_auc": fold_auc,
#         "precision_per_class": {k: v["precision"] for k, v in cr.items() if k.isdigit()},
#         "recall_per_class": {k: v["recall"] for k, v in cr.items() if k.isdigit()},
#         "f1_per_class": {k: v["f1-score"] for k, v in cr.items() if k.isdigit()}
#     })
    
#     # oof = model.predict(valid_gen, verbose=1)
#     # all_oof.append(oof)
#     # all_true.append(train.iloc[valid_index][TARGETS].values)
    
#     del model, preds
#     gc.collect()


# # After all folds
# all_fold_preds = np.concatenate(all_fold_preds)
# all_fold_trues = np.concatenate(all_fold_trues)

# overall_auc = roc_auc_score(tf.one_hot(all_fold_trues, depth=6).numpy(), 
#                             tf.one_hot(all_fold_preds, depth=6).numpy(), 
#                             average='macro')

# print(f"\nOverall Macro AUC across folds: {overall_auc:.4f}")
# print("\nOverall Classification Report:")
# print(classification_report(all_fold_trues, all_fold_preds, digits=4))

# # Summary
# for m in fold_metrics:
#     print(f"Fold {m['fold']} -> Macro AUC: {m['macro_auc']:.4f}")
#     print(f"Precision per class: {m['precision_per_class']}")
#     print(f"Recall per class: {m['recall_per_class']}")
#     print(f"F1-score per class: {m['f1_per_class']}")
#     print("-"*50)    
# all_oof = np.concatenate(all_oof)
# all_true = np.concatenate(all_true)


all_fold_preds = []
all_fold_trues = []
fold_metrics = []

# gkf = GroupKFold(n_splits=5)
gkf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
for i, (train_index, valid_index) in enumerate(gkf.split(train, train['target_encoded'], train.patient_id)):  
    
    print('#'*25)
    print(f'### Fold {i+1}')
    #one_hot false for SparseCategoricalLoss
    train_gen = DataGenerator(train.iloc[train_index], shuffle=True, batch_size=32, augment=True, label_col='target_encoded', one_hot=False)
    valid_gen = DataGenerator(train.iloc[valid_index], shuffle=False, batch_size=32, mode='valid', label_col='target_encoded', one_hot=False)
    
    print(f'### train size {len(train_index)}, valid size {len(valid_index)}')
    print('#'*25)
    
    K.clear_session()
    with strategy.scope():
        model = build_model()
        
    if LOAD_MODELS_FROM is None:
        model.fit(train_gen, verbose=1,
              validation_data=valid_gen,
              epochs=EPOCHS, callbacks=[LR], class_weight=class_weight_dict)
        model.save_weights(f'EffNet_v{VER}_f{i}.weights.h5')
    else:
        model.load_weights(f'{LOAD_MODELS_FROM}EffNet_v{VER}_f{i}.h5')

    y_true = valid_gen.data[valid_gen.label_col].values

    # Get predictions
    y_pred_probs = model.predict(valid_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Print classification report
    print(f"\nClassification Report - Fold {i+1}")
    cr = classification_report(y_true, y_pred, target_names=class_names, labels=list(range(len(class_names))), digits=4,
                               zero_division=0, output_dict=True)
    
    # Print confusion matrix
    print(f"Confusion Matrix - Fold {i+1}")
    cm = confusion_matrix(y_true, y_pred)
        
    # preds = model.predict(valid_gen)

    # # Convert predictions to class indices
    # pred_classes = np.argmax(preds, axis=1)
    
    # # Get true class indices
    # true_classes = np.array([class_to_idx[t] for t in train.iloc[valid_index]['target']])
    # cr = classification_report(true_classes, pred_classes, 
    #                            target_names=class_names, labels=list(range(len(class_names))), digits=4,
    #                            zero_division=0, output_dict=True)
    # 1. Classification Report
    cr_df = pd.DataFrame(cr).transpose()    # <-- FIX: pretty print
    display(cr_df)

    # 2. Confusion Matrix
    plt.figure(figsize=(10,8))
    # cm = confusion_matrix(true_classes, pred_classes, labels=np.arange(len(class_names)))
    sns.heatmap(cm, annot=True, fmt='d',
               xticklabels=class_names,
               yticklabels=class_names,
               cmap='Blues')
    plt.title(f'Fold {i+1} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # Store fold metrics
    fold_metrics.append({
        'fold': i+1,
        'accuracy': cr['accuracy'],
        'class_report': cr,
        'confusion_matrix': cm
    })
    
    del model, y_pred_probs
    gc.collect()


# Calculate overall metrics
print('\n' + '#'*50)
print('### FINAL OVERALL RESULTS ACROSS ALL FOLDS')
print('#'*50 + '\n')

# # Convert to arrays
# all_true = np.array(all_fold_trues)
# all_pred = np.array(all_fold_preds)

# # 1. Overall Classification Report
# print("Overall Classification Report:")
# overall_cr = classification_report(all_true, all_pred,
#                            target_names=class_names,
#                            labels=np.arange(len(class_names)),
#                            digits=4,
#                            zero_division=0)

# overall_cr_df = pd.DataFrame(overall_cr).transpose()
# display(overall_cr_df)

# # 2. Overall Confusion Matrix
# plt.figure(figsize=(10,8))
# overall_cm = confusion_matrix(all_true, all_pred, labels=np.arange(len(class_names)))
# sns.heatmap(overall_cm, annot=True, fmt='d',
#            xticklabels=class_names,
#            yticklabels=class_names,
#            cmap='Blues')

# plt.title('Overall Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()

# # 4. Print fold-wise summary
# print("\nFold-wise Performance Summary:")
# summary_df = pd.DataFrame(fold_metrics)[['fold', 'accuracy']]
# print(summary_df.to_string(index=False))

# print(f"\nMean Accuracy: {summary_df['accuracy'].mean():.4f}")


# import sys
# sys.path.append('/kaggle/input/kaggle-kl-div')
# from kaggle_kl_div import score

# oof = pd.DataFrame(all_fold_preds.copy())
# oof['id'] = np.arange(len(oof))

# true = pd.DataFrame(all_fold_trues.copy())
# true['id'] = np.arange(len(true))

# cv = score(solution=true, submission=oof, row_id_column_name='id')
# print('CV Score KL-Div for EfficientNetB2 =',cv)


# del all_eegs, spectrograms; gc.collect()
# test = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/test.csv')
# print('Test shape',test.shape)
# test.head()


# READ ALL SPECTROGRAMS
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


# INFER EFFICIENTNET ON TEST
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

