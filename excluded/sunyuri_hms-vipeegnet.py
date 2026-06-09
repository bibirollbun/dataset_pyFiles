#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Welcome to validate and use this algorithm

This code is associated with a manuscript "An Automated Classifier of Harmful Brain Activities for Clinical Usage Based on a Vision-Inspired Pre-trained Framework".

Cite as:
Sun, Y., Si, X., He, R. et al. An Automated Classifier of Harmful Brain Activities for Clinical Usage Based on a Vision-Inspired Pre-trained Framework. npj Digit. Med. 8, 768 (2025).
https://doi.org/10.1038/s41746-025-02154-4

If you have any questions or concerns regarding this code or the related manuscript, please contact syuri@tju.edu.cn.

19 December 2025
"""

# =============================================================================
# GLOBAL VARIABLES AND CONFIGURATION
# =============================================================================

# Flag to determine if model training is needed
NEEDTRAIN = True        
# Path to trained model weights for testing
LOAD_MODELS_FROM = 'modelsxxxxxxx'       

import os
# Set Keras backend to TensorFlow
os.environ["KERAS_BACKEND"] = "tensorflow"

# Determine platform (local or Kaggle)
if os.getcwd().split(os.sep)[1] == 'home':
    PLATFORM = 'local'  # Local training or online testing
    # Find the correct models directory in local input
    for dir_name in os.listdir('./input/'):
        if dir_name[:6] == 'models':
            LOAD_MODELS_FROM = dir_name
elif os.getcwd().split(os.sep)[1] == 'kaggle':
    PLATFORM = 'kaggle' # Kaggle platform
    NEEDTRAIN = False
    # Find the correct models directory in Kaggle input
    for dir_name in os.listdir('/kaggle/input/'):
        if dir_name[:6] == 'models':
            LOAD_MODELS_FROM = dir_name

# Data type used (EEG in this case)
DATATYPE = ['eeg']    
print(DATATYPE)

# Set paths based on platform
if PLATFORM == 'local':
    LOAD_MODELS_FROM = f'./input/{LOAD_MODELS_FROM}'
    LOAD_DATA_FROM = './input/hms-harmful-brain-activity-classification'
elif PLATFORM == 'kaggle':
    LOAD_MODELS_FROM = f'/kaggle/input/{LOAD_MODELS_FROM}'
    LOAD_DATA_FROM = '/kaggle/input/hms-harmful-brain-activity-classification'

# EEG sampling configuration
SFREQ = 200             
RSFREQ = 200            
EEG_LENGTH = 50         
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16 

# EEG filtering configuration
filter_range = [0.5, 45]   
SEED = 2024             
BATCHSIZE = 16        
LEARN_RATE = 1e-3
EPOCHS = 15
SPLITS = 5

# Flag for preprocessing EEG files
READ_EEG_FILES = False      
# Dictionaries to store preprocessed EEG data
eegs = {}               
eegs_test = {}             

# Brain channels configuration
BRAIN = [
         'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',   
         'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',   
         'Fz-Cz', 'Cz-Pz',
         'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',   
         'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',   
        ]

TEST_BATCHSIZE = 128

# =============================================================================
# IMPORTS
# =============================================================================

# Suppress warnings and set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES']='0, 1'
import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
import pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from scipy import signal
import time
import gc

# Set random seeds for reproducibility
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Configure TensorFlow
import tensorflow as tf
print(tf.version.VERSION)
print(tf.config.list_physical_devices('GPU'))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  
    except RuntimeError as e:
        print(e)
        
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# Configure mixed precision training
MIX = True
if MIX:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
else:
    print('Using full precision')
    
# Import itertools if needed for training
if NEEDTRAIN:
    import itertools

# =============================================================================
# LOAD TRAIN DATAFRAME
# =============================================================================

# Load training data
df = pd.read_csv(os.path.join(LOAD_DATA_FROM, 'train.csv'))
TARGETS = df.columns[-6:]
print('Train shape:', df.shape)
print('Targets', list(TARGETS))

# =============================================================================
# CREATE NON-OVERLAPPING EEG ID TRAIN DATAFRAME
# =============================================================================

if NEEDTRAIN:
    TARGETS_RAW = list()
    for i in TARGETS:
        TARGETS_RAW.append(i + '_raw')
        
    if READ_EEG_FILES:
        # Create a non-overlapping EEG ID dataframe
        train = df.drop_duplicates(['eeg_id', 'seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']).reset_index(drop=True)
        
        train['sign_id'] = train.index.values
        df['sign_id'] = df.index.values
        
        y_data = train[TARGETS].values
        train[TARGETS_RAW] = y_data
        y_data = y_data / y_data.sum(axis=1,keepdims=True)
        train[TARGETS] = y_data

        train.to_csv('train.csv', index=False)
    else:
        # Load preprocessed training dataframe
        train = pd.read_csv('train.csv')

# =============================================================================
# READ TRAIN EEGS
# =============================================================================

if filter_range != None:
    # Configure EEG filtering
    b, a = signal.butter(3, np.float32(filter_range)*2/RSFREQ, 'bandpass')
    
if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, 'train_eegs') + '/'
    if READ_EEG_FILES:
        # Read and preprocess EEG files
        time_start_time = time.time()
        
        for i, eeg_id in enumerate(train.eeg_id.unique()):
            
            if i%200==0:
                gc.collect()
                xx = (time.time() - time_start_time)
                yy = xx / (i+1) * len(train.eeg_id.unique())
                print(i, f'time: {round(xx / 60, 2)} min / {round(yy / 60, 2)} min')
            eeg_default = pd.read_parquet(os.path.join(PATH, (str(eeg_id) + '.parquet')))
            
            eeg = list()
            for channel in BRAIN:
                eeg_temp = (eeg_default.loc[:, channel.split('-')[0]] - eeg_default.loc[:, channel.split('-')[1]]).values
                eeg_temp[np.isnan(eeg_temp)] = 0
                eeg.append(np.reshape(eeg_temp, (1, -1)))
            eeg = np.concatenate(eeg, axis=0)
            
            if SFREQ != RSFREQ:
                eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

            eeg = np.clip(eeg, a_min=-1024, a_max=1024)
            
            if filter_range != None:
                eeg = signal.filtfilt(b, a, eeg, axis=1)
                
            eeg = np.array(eeg, dtype=np.float32)
            
            if 'eeg' in DATATYPE:
                eegs[eeg_id] = eeg

        # Save preprocessed EEG data
        if not os.path.exists('./input/preprocess'):
            os.makedirs('./input/preprocess')
        if 'eeg' in DATATYPE:
            np.save('./input/preprocess/eegs.npy', eegs, allow_pickle=True)

    else:
        # Load preprocessed EEG data
        if PLATFORM == 'local':
            datapath = './' + os.path.join('input', 'preprocess')
        elif PLATFORM == 'kaggle':
            datapath = '/kaggle/' + os.path.join('input', 'preprocess')

        eegs = np.load(os.path.join(datapath, 'eegs.npy'), allow_pickle=True).item()

# =============================================================================
# DATA GENERATOR
# =============================================================================

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, batch_size=32, shuffle=False, sample_weights=False, mode='train',
                 eegs=None, stage=2): 

        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_weights = sample_weights
        self.mode = mode
        self.eegs = eegs
        self.stage = stage
        self.on_epoch_end()
        
    def __len__(self):
        # Calculate number of batches
        ct = int( np.ceil( len(self.dataframe) / self.batch_size ) )
        return ct

    def __getitem__(self, index):
        # Generate one batch of data
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        x, y, sample_weights = self.__data_generation(indexes)
        return x, y, sample_weights

    def on_epoch_end(self):
        # Update indexes after each epoch
        self.nan = 0
        self.indexes = np.arange( len(self.dataframe) )
        if self.shuffle: np.random.shuffle(self.indexes)
                        
    def __data_generation(self, indexes):
        # Generate data for a batch
        x_eeg = np.zeros((len(indexes), EEG_CHANNEL_USED, round(EEG_LENGTH_USED * RSFREQ)),dtype='float32')
        y = np.zeros((len(indexes), len(TARGETS)),dtype='float32')
        sample_weights = np.zeros((len(indexes), 1),dtype='float32')
            
        for j, i in enumerate(indexes):
            row = self.dataframe.iloc[i]
            if self.mode != 'test':
                sample_weight = sum(row[TARGETS_RAW].values)/20
            
            # Process EEG data based on mode
            if self.mode == 'test':
                r_eeg = 0
            else:
                rows = df.loc[(df.eeg_id == row.eeg_id) * (df.seizure_vote == row.seizure_vote_raw) * (df.lpd_vote == row.lpd_vote_raw) * (df.gpd_vote == row.gpd_vote_raw) * (df.lrda_vote == row.lrda_vote_raw) * (df.grda_vote == row.grda_vote_raw), :].reset_index(drop=True)
                if self.mode == 'train':
                    rows = rows.iloc[np.random.permutation(len(rows))].reset_index(drop=True)
                    row = rows.loc[0, :]
                elif self.mode == 'valid':
                    row = rows.sort_values(by='eeg_sub_id').reset_index(drop=True).iloc[len(rows)//2]
                r_eeg = row.eeg_label_offset_seconds
                if (self.mode == 'train'):
                    r_eeg = r_eeg + np.random.random() * 10 - 5
                    r_eeg = max(0, r_eeg)
                    r_eeg = min(r_eeg, self.eegs[row.eeg_id].shape[1] / RSFREQ - 50)

            eeg = self.eegs[row.eeg_id][:, round(r_eeg * RSFREQ):round((r_eeg + 50) * RSFREQ)]
            eeg = np.concatenate((eeg[0:round(EEG_CHANNEL_USED/2), :], eeg[-round(EEG_CHANNEL_USED/2):, :]), axis=0)

            eeg = eeg[:, round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2):round((EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2)]
           
            if self.mode=='train':
                # Apply data augmentation for training
                if (self.stage == 2) and (np.random.rand() > 0):
                    eeg2 = eeg.copy()
                    eeg[4:8, :] = eeg2[12:16, :]
                    eeg[8:12, :] = eeg2[4:8, :]
                    eeg[12:16, :] = eeg2[8:12, :]
                else:
                    if np.random.rand() > 0.5:
                        mask = round(np.random.rand() * eeg.shape[1])
                        eeg[:, mask:round(mask + np.random.rand() * eeg.shape[1] * 0.02)] = 0
                        
                    if np.random.rand() > 0.5:
                        mask = round(np.random.rand() * eeg.shape[1])
                        eeg[:, mask:round(mask + np.random.rand() * eeg.shape[1] * 0.02)] = 0
                        
                    if np.random.rand() > 0.5:
                        mask = round(np.random.rand() * eeg.shape[1])
                        eeg[:, mask:round(mask + np.random.rand() * eeg.shape[1] * 0.02)] = 0
                    
                    
                    if np.random.rand() > 0.5:
                        eeg[np.random.permutation(eeg.shape[0])[0], :] = 0
            
                    if np.random.rand() > 0.5:
                        eeg[np.random.permutation(eeg.shape[0])[0], :] = 0
                    
                    eeg[0:round(EEG_CHANNEL_USED/2), :] = eeg[0:round(EEG_CHANNEL_USED/2), :][np.random.permutation(8), :]
                    eeg[-round(EEG_CHANNEL_USED/2):, :] = eeg[-round(EEG_CHANNEL_USED/2):, :][np.random.permutation(8), :]
                    
                    eeg2 = eeg.copy()
                    eeg[4:8, :] = eeg2[12:16, :]
                    eeg[8:12, :] = eeg2[4:8, :]
                    eeg[12:16, :] = eeg2[8:12, :]
                    
                    if np.random.rand() > 0.5:
                        eeg = eeg[::-1, :]

                    if np.random.rand() > 0.5:
                        eeg = -eeg
                    
                    if np.random.rand() > 0.5:
                        eeg = eeg[:, ::-1]
            else:
                # Process validation/test data
                eeg2 = eeg.copy()
                eeg[4:8, :] = eeg2[12:16, :]
                eeg[8:12, :] = eeg2[4:8, :]
                eeg[12:16, :] = eeg2[8:12, :]

            # Normalize EEG data
            eeg = np.clip(eeg, a_min=-1024, a_max=1024)
            eeg = eeg + 1024
            eeg = eeg / 2048 * 255
            
            x_eeg[j] = eeg
            
            if self.mode!='test':
                y[j] = row[TARGETS].values / sum(row[TARGETS].values)
                
                if self.sample_weights:
                    sample_weights[j] = sample_weight
                else:
                    sample_weights[j] = 1
        
        return x_eeg, y, sample_weights

# =============================================================================
# MODEL BUILDING
# =============================================================================

class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
        super(CosineAnnealingLRScheduler, self).__init__()
        self.total_step = total_step

        if warmth_rate == 0:
            self.warm_step = 1
        else:
            self.warm_step = int(warmth_rate)

        self.lr_max = lr_max
        self.lr_min = lr_min
        
        self.begin = 1
        
    def __call__(self, step):
        # Implement cosine annealing learning rate schedule
        if step == self.total_step:
            self.begin = 0
            self.lr_max = self.lr_max * 0.5
            self.lr_min = self.lr_min * 0.1
            
        step = step % self.total_step
        step = step + 1

        if (self.begin==1) and (step < self.warm_step):
            lr = self.lr_max / self.warm_step * step
        else:
            if self.begin==1:
                if self.total_step == 1:
                    lr = self.lr_max
                else:
                    lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + tf.cos((step - self.warm_step) / (self.total_step-self.warm_step) * np.pi))
            else:
                lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + tf.cos(step / 10 * np.pi))
        
        return np.float32(lr)

class IniToOne(tf.keras.initializers.Initializer):
    def __init__(self):
        super(IniToOne, self).__init__()

    def __call__(self, shape, dtype=None):
       # Initialize weights with ones
       assert len(shape) == 3
       filter_length, input_channel, filter_count = shape
       
       kernel = np.zeros(shape, dtype=np.float32)
       for i in range(filter_count):
           kernel[i%filter_length, 0, i] = 1.0
       kernel = tf.convert_to_tensor(kernel, dtype=dtype)
       return kernel

    def get_config(self):
        return {}

class SumToOne(tf.keras.constraints.Constraint):
    def __init__(self):
        super(SumToOne, self).__init__()

    def __call__(self, w):
        # Constrain weights to sum to one
        w = tf.abs(w)
        w_normed = w / tf.reduce_sum(w, axis=[0, 1], keepdims=True)
        return w_normed

    def get_config(self):
        return {}

class IniToOneAtten(tf.keras.initializers.Initializer):
    def __init__(self):
        super(IniToOneAtten, self).__init__()

    def __call__(self, shape, dtype=None):
       # Initialize attention weights
       assert len(shape) == 3
       filter_length, input_channel, filter_count = shape
       
       kernel = np.zeros(shape, dtype=np.float32)
       kernel[(filter_length-1)//2:(filter_length)//2+1, :, :] = 1/((filter_length)//2+1 - (filter_length-1)//2)
       kernel = tf.convert_to_tensor(kernel, dtype=dtype)
       return kernel

    def get_config(self):
        return {}

class SumToOneAtten(tf.keras.constraints.Constraint):
    def __init__(self):
        super(SumToOneAtten, self).__init__()

    def __call__(self, w):
        # Constrain attention weights
        w = tf.abs(w)
        w_normed = w / tf.reduce_sum(w, axis=[0, 1], keepdims=True)
        return w_normed

    def get_config(self):
        return {}

def build_model():
    # Build the neural network model
    inp_eeg = tf.keras.Input(shape=(EEG_CHANNEL_USED, round(EEG_LENGTH_USED * RSFREQ)), name='eeg')
    x_eeg_raw = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(inp_eeg)

    strides = 10
    if PLATFORM == 'local':
        # Configure EEG embedding layer with custom initialization
        eeg_embed = tf.keras.layers.Conv1D(filters=strides*3, kernel_size=strides, strides=strides,
                                           padding='same', use_bias=False, activation=None,
                                           kernel_initializer = IniToOne(),
                                           kernel_constraint = SumToOne(),
                                           input_shape=(None, 1)
                                           )
    else:
        # Configure standard EEG embedding layer
        eeg_embed = tf.keras.layers.Conv1D(filters=strides*3, kernel_size=strides, strides=strides,
                                            padding='same', use_bias=False, activation=None)
    
    x_eeg = tf.keras.layers.TimeDistributed(eeg_embed)(x_eeg_raw)
    
    # Reshape and prepare data for EfficientNet
    x_eeg = tf.keras.layers.Concatenate(axis=-1)([tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1, 1))(x_eeg[:, :, :, 0*strides:1*strides]),
                                                  tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1, 1))(x_eeg[:, :, :, 1*strides:2*strides]),
                                                  tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1, 1))(x_eeg[:, :, :, 2*strides:3*strides])
                                                  ])
    x_eeg = tf.keras.layers.Permute([4, 2, 1, 3])(x_eeg)
    x_eeg = tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1))(x_eeg)
    x_eeg = tf.keras.layers.Permute((3, 2, 1))(x_eeg)
    
    # Load pre-trained EfficientNetV2B3
    base_model_eeg = tf.keras.applications.EfficientNetV2B3(include_top=False, weights=None,
                                                           include_preprocessing=True)
    
    if NEEDTRAIN:
        if PLATFORM == 'local':
            # Load local pre-trained weights
            base_model_eeg.load_weights(f'./input/pre-trained-weights/{base_model_eeg.name}_notop.h5')
        if PLATFORM == 'kaggle':
            # Load Kaggle pre-trained weights
            base_model_eeg.load_weights(f'/kaggle/input/pre-trained-weights/{base_model_eeg.name}_notop.h5')

    base_model_eeg.name = 'eeg_extractor'

    x_eeg = base_model_eeg(x_eeg)
    
    # Process output features
    x_eeg = x_eeg[:, :, (x_eeg.shape[2]-1)//2:(x_eeg.shape[2])//2+1, :]
    
    x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
    x_eeg = tf.keras.layers.Dropout(0.5)(x_eeg)

    # Output layer with softmax activation
    y = tf.keras.layers.Dense(len(TARGETS), activation='softmax', dtype='float32')(x_eeg)
 
    model = tf.keras.Model(inputs=inp_eeg, outputs=y)
        
    return model

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_fold(i, stage, train_index, valid_index, df_train_stage1, df_valid_stage1, df_train_stage2, df_valid_stage2, 
               build_model, BATCHSIZE, EPOCHS, LEARN_RATE, TARGETS, TARGETS_RAW):
    
    print('#'*25)
    print(f'### Fold {i+1}')
    
    # Build model
    model = build_model()
    loss = tf.keras.losses.KLDivergence()

    # Configure data generators based on stage
    if stage == 1:
        train_gen_stage = DataGenerator(df_train_stage1, shuffle=True, sample_weights=True, batch_size=BATCHSIZE, eegs=eegs, stage=stage)
        valid_gen_stage = DataGenerator(df_valid_stage1, shuffle=False, sample_weights=True, batch_size=BATCHSIZE*2, mode='valid', eegs=eegs, stage=stage)
        opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE)
        # Configure callbacks for stage 1 training
        callbacks_stage = [
            tf.keras.callbacks.LearningRateScheduler(CosineAnnealingLRScheduler(EPOCHS, LEARN_RATE, LEARN_RATE * 0.1 * 0.1, 5)),
            tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('models', f'fold{i}_stage1.weights.h5'),
                                               monitor='val_loss', mode='min',
                                               save_weights_only=True, save_best_only=True)
        ]
    elif stage == 2:
        train_gen_stage = DataGenerator(df_train_stage2, shuffle=True, sample_weights=False, batch_size=BATCHSIZE * 2, eegs=eegs, stage=stage)
        valid_gen_stage = DataGenerator(df_valid_stage2, shuffle=False, sample_weights=False, batch_size=BATCHSIZE*2 * 2, mode='valid', eegs=eegs, stage=stage)
        opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE * 0.1 * 3)
        model.load_weights(os.path.join('models', f'fold{i}_stage1.weights.h5'))
        # Configure callbacks for stage 2 training
        callbacks_stage = [
            tf.keras.callbacks.LearningRateScheduler(CosineAnnealingLRScheduler(max(round(EPOCHS/3), 1), LEARN_RATE * 0.1 * 3, LEARN_RATE * 0.1 * 0.1 * 0.1, 0)),
            tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('models', f'fold{i}_stage2.weights.h5'),
                                               monitor='val_loss', mode='min',
                                               save_weights_only=True, save_best_only=True)
        ]

    # Compile model
    model.compile(loss=loss, optimizer=opt)
    
    # Train model
    if stage == 1:
        history = model.fit(train_gen_stage, verbose=1, validation_data=valid_gen_stage,
                            epochs=EPOCHS, callbacks=callbacks_stage)
    elif stage == 2:
        history = model.fit(train_gen_stage, verbose=1, validation_data=valid_gen_stage,
                            epochs=max(round(EPOCHS/3), 1), callbacks=callbacks_stage)

    # Load best weights
    model.load_weights(os.path.join('models', f'fold{i}_stage{stage}.weights.h5'))
 
    # Plot training history
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='loss')
    plt.plot(epochs, val_loss, 'b', label='val_loss')
    plt.title(f'loss: {round(min(loss), 4)}, val loss: {round(min(val_loss), 4)}', fontsize=12)
    plt.legend()
    plt.savefig(os.path.join('models', f'fold{i}_stage{stage}.svg'))
    plt.close()

    # Generate confusion matrix
    if stage == 1:
        valid_stage = df_valid_stage1[TARGETS].values
    elif stage == 2:
        valid_stage = df_valid_stage2[TARGETS].values
    predict_stage = model.predict(valid_gen_stage)
        
    # Clean up resources
    del train_gen_stage, valid_gen_stage, history, model
    tf.keras.backend.clear_session()
    gc.collect()

    # Calculate and plot confusion matrix
    cm = confusion_matrix(np.argmax(valid_stage, 1), np.argmax(predict_stage, 1))
    cm = cm / np.sum(cm, 1, keepdims=True)
        
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(6)
    plt.xticks(tick_marks, [f'{TARGETS[i][:-5]}' for i in [0, 1, 2, 3, 4, 5]], fontsize=10)
    plt.yticks(tick_marks, [f'{TARGETS[i][:-5]}' for i in [0, 1, 2, 3, 4, 5]], fontsize=10)
    thresh = cm.max() / 2.
    for ii, jj in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[ii, jj] > -0.1:
            plt.text(jj, ii, str(round(cm[ii, jj] * 1e4) * 1e-2)[:5], horizontalalignment="center", color="white" if cm[ii, jj] > thresh else "black", fontsize=10)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.savefig(os.path.join('models', f'fold{i}_stage{stage}_cm.svg'))
    plt.close()

    # Clean up resources
    del df_train_stage1, df_valid_stage1, df_train_stage2, df_valid_stage2
    gc.collect()

# =============================================================================
# MAIN TRAINING AND INFERENCE
# =============================================================================

if __name__ == '__main__':
    if NEEDTRAIN:
        # Create models directory if needed
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Configure cross-validation
        from sklearn.model_selection import GroupKFold
        import multiprocessing as mp
        mp.set_start_method('spawn')  # Set spawn method for multiprocessing
        
        gkf = GroupKFold(n_splits=SPLITS)
        
        # Perform k-fold cross-validation
        for i, (train_index, valid_index) in enumerate(gkf.split(train, train.expert_consensus, train.patient_id)):  
            print('#'*25)
            print(f'### Fold {i+1}')
            
            df_train_stage1 = train.iloc[train_index].reset_index(drop=True)
            df_valid_stage1 = train.iloc[valid_index].reset_index(drop=True)

            df_train_stage2 = df_train_stage1[np.sum(df_train_stage1[TARGETS_RAW].values, 1) >= 10].reset_index(drop=True)
            df_valid_stage2 = df_valid_stage1[np.sum(df_valid_stage1[TARGETS_RAW].values, 1) >= 10].reset_index(drop=True)

            # Train two stages for each fold
            for stage in [1, 2]:
                p = mp.Process(target=train_fold, args=(
                    i,
                    stage,
                    train_index,
                    valid_index,
                    df_train_stage1,
                    df_valid_stage1,
                    df_train_stage2,
                    df_valid_stage2,
                    build_model,
                    BATCHSIZE,
                    EPOCHS,
                    LEARN_RATE,
                    TARGETS,
                    TARGETS_RAW,
                ))
                p.start()
                p.join()
    
    # =============================================================================
    # INFERENCE ON TEST DATA
    # =============================================================================
    
    else:
        # Load all trained models for inference
        preds_all = []
        models = list()
        for model_i in range(999):
            if os.path.exists(os.path.join(LOAD_MODELS_FROM, f'fold{model_i}_stage2.weights.h5')):
                print(f'Fold {model_i+1}')
                model = build_model()
                model.load_weights(os.path.join(LOAD_MODELS_FROM, f'fold{model_i}_stage2.weights.h5'))
                models.append(model)

        # Load test data
        test = pd.read_csv(os.path.join(LOAD_DATA_FROM, 'test.csv'))
        test['sign_id'] = test.index.values
        print('Test shape', test.shape)

        PATH_test = os.path.join(LOAD_DATA_FROM, 'test_eegs') + '/'

        # Process test EEG data
        for i, eeg_id in enumerate(test.eeg_id):
            if i%100==0: print(i,', ',end='')
            eeg_default = pd.read_parquet(os.path.join(PATH_test, (str(eeg_id) + '.parquet')))
            
            eeg = list()
            for channel in BRAIN:
                eeg_temp = (eeg_default.loc[:, channel.split('-')[0]] - eeg_default.loc[:, channel.split('-')[1]]).values
                eeg_temp[np.isnan(eeg_temp)] = 0
                eeg.append(np.reshape(eeg_temp, (1, -1)))
            eeg = np.concatenate(eeg, axis=0)
            
            if SFREQ != RSFREQ:
                eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

            eeg = np.clip(eeg, a_min=-1024, a_max=1024)
            eegshape = eeg.shape[1]
            eeg = np.concatenate((eeg[:, ::-1], eeg, eeg[:, ::-1]), axis=1)
            if filter_range != None:
                eeg = signal.filtfilt(b, a, eeg, axis=1)
            eeg = eeg[:, eegshape:eegshape*2]
            
            eeg = np.array(eeg, dtype=np.float32)
            
            eegs_test[eeg_id] = eeg

            # Make predictions in batches
            if ((i+1)%TEST_BATCHSIZE==0) or ((i+1)==len(test.eeg_id)):
                preds = []
                test_gen = DataGenerator(test.loc[max(i-TEST_BATCHSIZE+1, len(preds_all)):i, :], shuffle=False, sample_weights=False, batch_size=TEST_BATCHSIZE, mode='test', eegs=eegs_test, stage=2)
                for model_i in range(len(models)):
                    pred = models[model_i].predict(test_gen, verbose=1)
                    preds.append(pred)
                pred = np.mean(preds, axis=0)
                del eegs_test
                gc.collect()
                eegs_test = {}
                if len(preds_all) == 0:
                    preds_all = pred.copy()
                else:
                    preds_all = np.concatenate((preds_all, pred), axis=0)

        # Prepare submission file
        sub = pd.DataFrame({'eeg_id': test.eeg_id.values})
        sub[TARGETS] = preds_all
        sub.to_csv('submission.csv', index=False)
        print('Submission shape', sub.shape)
        sub.head()




