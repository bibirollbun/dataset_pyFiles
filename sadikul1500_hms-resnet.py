# !pip install -q /kaggle/input/kerasv3-lib-ds/keras_cv-0.8.2-py3-none-any.whl --no-deps
# !pip install -q /kaggle/input/kerasv3-lib-ds/tensorflow-2.15.0.post1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-deps
# !pip install -q /kaggle/input/kerasv3-lib-ds/keras-3.0.4-py3-none-any.whl --no-deps


import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
import joblib
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
import keras
import keras_cv
import matplotlib.pyplot as plt
import math
import warnings
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import tensorflow.keras.backend as K, gc
warnings.filterwarnings('ignore')


gpus = tf.config.list_physical_devices('GPU')
if len(gpus)<=1: 
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print(f'Using {len(gpus)} GPU')
else: 
    strategy = tf.distribute.MirroredStrategy()
    print(f'Using {len(gpus)} GPUs')


class CFG:
    verbose = 1  # Verbosity
    seed = 42  # Random seed
    preset = 'efficientnetv2_b2_imagenet' #"mobilenet_v3_large_imagenet"  # Name of pretrained classifier
    image_size = [128, 256, 4]  # Input image size
    epochs = 20  # Training epochs
    batch_size = 16  # Batch size
    lr_mode = "cos"  # LR scheduler mode from one of "cos", "step", "exp"
    drop_remainder = True  # Drop incomplete batches
    num_classes = 6  # Number of classes in the dataset
    fold = 0  # Which fold to set as validation data
    class_names = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
    label2name = dict(enumerate(class_names))
    name2label = {v: k for k, v in label2name.items()}

    keras.utils.set_random_seed(seed) 


#NPY_DIR = '/kaggle/input/spectrograms-500/spectrograms_npy_balanced/'

# NPY_DIR = '/kaggle/input/spectrograms-3000/spectrograms_npy_balanced_3000/'
# NPY_DIR = '/kaggle/input/spectrograms-3000-128-512/spectrograms_npy_balanced_3000_128_512/'
# NPY_DIR = '/kaggle/input/spectrograms-500/spectrograms_3000_128_512_corrected/'
# NPY_DIR = '/kaggle/input/spectrogram-128-256-partial-90/spectrograms_all_128_256/'
NPY_DIR = '/kaggle/input/brain-spectrogram-128-256-90/spectrograms_all_128_256/'


# NPY_DIR = '/kaggle/input/brain-spectrograms-128-512/spectrograms_all_128_512_corrected/'


# def load_data():
#     BASE_PATH = "data"
#     SPEC_DIR = "/tmp/dataset/hms-hbac"
#     os.makedirs(SPEC_DIR + '/train_spectrograms', exist_ok=True)
#     os.makedirs(SPEC_DIR + '/test_spectrograms', exist_ok=True)

#     # Train + Valid
#     df = pd.read_csv(f'{BASE_PATH}/train.csv')
#     df['eeg_path'] = f'{BASE_PATH}/train_eegs/' + df['eeg_id'].astype(str) + '.parquet'
#     df['spec_path'] = f'{BASE_PATH}/train_spectrograms/' + df['spectrogram_id'].astype(str) + '.parquet'
#     df['spec2_path'] = f'{SPEC_DIR}/train_spectrograms/' + df['spectrogram_id'].astype(str) + '.npy'
#     df['class_name'] = df.expert_consensus.copy()
#     df['class_label'] = df.expert_consensus.map(CFG.name2label)
#     display(df.head(2))

#     # Test
#     test_df = pd.read_csv(f'{BASE_PATH}/test.csv')
#     test_df['eeg_path'] = f'{BASE_PATH}/test_eegs/' + test_df['eeg_id'].astype(str) + '.parquet'
#     test_df['spec_path'] = f'{BASE_PATH}/test_spectrograms/' + test_df['spectrogram_id'].astype(str) + '.parquet'
#     test_df['spec2_path'] = f'{SPEC_DIR}/test_spectrograms/' + test_df['spectrogram_id'].astype(str) + '.npy'
#     display(test_df.head(2))

#     return df, test_df


# def load_and_preprocess_meta_data(base_path):
"""Load and preprocess data with patient-aware grouping"""
# Load and group data as per your preprocessing
df = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')
TARGETS = df.columns[-6:]

# Create grouped dataframe
train = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
    {'spectrogram_id':'first','spectrogram_label_offset_seconds':'min'})
train.columns = ['spec_id','min']

tmp = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
    {'spectrogram_label_offset_seconds':'max'})
train['max'] = tmp

tmp = df.groupby('eeg_id')[['patient_id']].agg('first') 
train['patient_id'] = tmp

# train['total_evaluators'] = df[['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].sum(axis=1)

tmp = df.groupby('eeg_id')[TARGETS].agg('sum')
sum_targets = tmp.sum(axis=1)
max_vote_percentage = tmp.max(axis=1) / sum_targets
train['max_vote_percentage'] = max_vote_percentage

tmp = df.groupby('eeg_id')[TARGETS].agg('sum')
for t in TARGETS:
    train[t] = tmp[t].values
    
y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1,keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby('eeg_id')[['expert_consensus']].agg('first') 
train['target'] = tmp
train = train.reset_index()


# from sklearn.preprocessing import LabelEncoder

# # Encode string labels to integers
# le = LabelEncoder()
# train['target_encoded'] = le.fit_transform(train['target'])


train = train[train['max_vote_percentage']>=.9]


class_names = ['GPD', 'GRDA', 'LPD', 'LRDA', 'Other', 'Seizure']
class_to_idx = {name: i for i, name in enumerate(class_names)}


# print(train[train['max_vote_percentage']<1])


# npy_files = [f for f in os.listdir(NPY_DIR) if f.endswith('.npy')]
# npy_ids = {f.split('.')[0] for f in npy_files}

# # Get metadata IDs
# meta_ids = set(train['eeg_id'].astype(str))

# # Find perfect matches
# matched_ids = npy_ids & meta_ids
# print(f"Found {len(matched_ids)} perfect matches between NPY files and metadata")

# # Filter both NPY files and metadata
# matched_npy_files = [f for f in npy_files if f.split('.')[0] in matched_ids]
# matched_meta = train[train['eeg_id'].astype(str).isin(matched_ids)].copy()
# # print(matched_meta['target'].value_counts())
# # Sort both to ensure same order
# matched_npy_files.sort()
# matched_meta = matched_meta.sort_values('eeg_id')
# train = matched_meta


# meta_df, TARGETS = load_and_preprocess_meta_data("/kaggle/input/hms-harmful-brain-activity-classification/")
        
# npy_files = [f for f in os.listdir(NPY_DIR) if f.endswith('.npy')]
# npy_ids = {f.split('.')[0] for f in npy_files}

# # Get metadata IDs
# meta_ids = set(meta_df['eeg_id'].astype(str))

# # Find perfect matches
# matched_ids = npy_ids & meta_ids
# print(f"Found {len(matched_ids)} perfect matches between NPY files and metadata")

# # Filter both NPY files and metadata
# matched_npy_files = [f for f in npy_files if f.split('.')[0] in matched_ids]
# matched_meta = meta_df[meta_df['eeg_id'].astype(str).isin(matched_ids)].copy()
# # print(matched_meta['target'].value_counts())
# # Sort both to ensure same order
# matched_npy_files.sort()
# matched_meta = matched_meta.sort_values('eeg_id')
# train = matched_meta


all_eegs = {}
for i,e in enumerate(train.eeg_id.values):    
    x = np.load(f'{NPY_DIR}{e}.npy')
    all_eegs[e] = x


#!pip install -U albumentations


# spectrogram_aug = keras_cv.layers.Augmenter([
#     keras_cv.layers.RandomTimeMasking(max_mask_size=20, p=0.5),
#     keras_cv.layers.RandomFrequencyMasking(max_mask_size=10, p=0.5),
# ])

# mixup = keras_cv.layers.MixUp(alpha=0.2)
# mixup_layer = keras_cv.layers.MixUp(alpha=0.6)

# Time & frequency masking + flipping
# spectrogram_aug_layers = [
#     keras_cv.layers.RandomFlip("horizontal", seed=42),
#     keras_cv.layers.RandomCutout(
#         height_factor=0.3, width_factor=0.1, fill_mode="constant", fill_value=0.0, seed=42
#     ),
#     keras_cv.layers.RandomCutout(
#         height_factor=0.1, width_factor=0.3, fill_mode="constant", fill_value=0.0, seed=42
#     ),
# ]


import albumentations as albu
class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, batch_size=32, shuffle=False, augment=False, mode='train',
                 # specs = spectrograms,
                 eeg_specs = all_eegs): 

        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.mode = mode
        # self.specs = specs
        self.eeg_specs = eeg_specs
        self.on_epoch_end()
        
        # self.augmentation_layers = tf.keras.Sequential([
        #     tf.keras.layers.RandomFlip(mode="horizontal"),
        #     tf.keras.layers.RandomRotation(factor=0.02),  # ~2% rotation
        #     tf.keras.layers.RandomZoom(height_factor=0.05, width_factor=0.05),  # ~5% zoom
        #     tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),  # ~5% shift
        #     tf.keras.layers.GaussianNoise(stddev=0.01),  # tiny noise
        # ])
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        ct = int( np.ceil( len(self.data) / self.batch_size ) )
        return ct

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X, y = self.__data_generation(indexes)

        if self.augment:
            # X = self.augmentation_layers(X, training=True)
            X = self.__augment_batch(X)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange( len(self.data) )
        if self.shuffle: np.random.shuffle(self.indexes)
        
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' 
        
        X = np.zeros((len(indexes),CFG.image_size[0], CFG.image_size[1],4),dtype='float32')
        y = np.zeros((len(indexes),6),dtype='float32')
        img = np.ones((CFG.image_size[0], CFG.image_size[1]),dtype='float32')
        
        for j,i in enumerate(indexes):
            row = self.data.iloc[i]
            X[j] = self.eeg_specs[row.eeg_id]
            
            # Create one-hot vector from target string
            target_str = row['target']
            y[j, class_to_idx[target_str]] = 1.0
                
        return X,y

    def __random_transform(self, img):
        composition = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.CoarseDropout(max_holes=8,max_height=32,max_width=32,fill_value=0,p=0.5),
        ])
        return composition(image=img)['image']
            
    def __augment_batch(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i, ] = self.__random_transform(img_batch[i, ])
        return img_batch


#will try this augmentation next
# def __random_transform(self, img):
#     composition = albu.Compose([
#         albu.HorizontalFlip(p=0.5),  # simulates left/right brain activity variations

#         albu.RandomBrightnessContrast(p=0.5),
#         albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3, border_mode=0),

#         albu.GaussianBlur(blur_limit=(3, 5), p=0.2),  # reduce overfitting to sharp edges

#         albu.CoarseDropout(
#             max_holes=8, max_height=16, max_width=16,
#             fill_value=0, mask_fill_value=None, p=0.3
#         ),

#         albu.Normalize(),  # standardize the spectrograms
#     ])
#     return composition(image=img)['image']


def build_model():
    # Build Classifier
    model = keras_cv.models.ImageClassifier.from_preset(
        CFG.preset, num_classes=CFG.num_classes
    )

    # Compile the model  
    LOSS = keras.losses.KLDivergence()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss=LOSS)
    
    return model 


#validation loss does not improve. Stuck in 1.4
# def build_model():
#     # Get the backbone with 4 input channels
#     backbone = keras_cv.models.MobileNetV3Backbone.from_preset(
#         CFG.preset,
#         input_shape=(CFG.image_size[0], CFG.image_size[1], 4), load_weights=False
#     )
    
#     # Build classifier with this backbone
#     model = keras.Sequential([
#         backbone,
#         keras.layers.GlobalAveragePooling2D(),
#         keras.layers.Dense(CFG.num_classes, activation='softmax')
#     ])

#     # Compile the model  
#     LOSS = keras.losses.KLDivergence()
#     model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
#                 loss=LOSS)
    
#     return model


def build_model_2():
    # Get the backbone with 4 input channels
    backbone = keras_cv.models.MobileNetV3Backbone.from_preset(
        CFG.preset,
        input_shape=(CFG.image_size[0], CFG.image_size[1], 4), load_weights=False
    )

    # Freeze some layers of the backbone to prevent overfitting
    backbone.trainable = True
    for layer in backbone.layers[:100]:
        layer.trainable = False
    
    # Build classifier with this backbone
    model = keras.Sequential([
        backbone,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.3), #added to avoid overfitting
        keras.layers.Dense(CFG.num_classes, activation='softmax',
                           kernel_regularizer=keras.regularizers.l2(1e-4)) #added to avoid overfitting
    ])

    # Compile the model  
    LOSS = keras.losses.KLDivergence()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                loss=LOSS)
    
    return model


#with this model, validation loss increses and train loss fluctuates!!
def build_model_3():
    model = keras.Sequential([
        keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(CFG.image_size[0], CFG.image_size[1], 4)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),

        keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),

        keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),

        keras.layers.GlobalAveragePooling2D(),
        # keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(CFG.num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.KLDivergence(),
        metrics=['accuracy']
    )

    return model


def build_efficientnet_model():
    base = keras.applications.EfficientNetB0(
        input_shape=(CFG.image_size[0], CFG.image_size[1], 4),
        include_top=False,
        weights=None
    )
    base.trainable = True
    
    model = keras.Sequential([
        base,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(CFG.num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                 loss=keras.losses.KLDivergence(),
                 metrics=['accuracy'])
    return model


# import efficientnet.tfkeras as efn

def build_model_4():
    # Input shape: 4-channel EEG spectrogram image
    inp = tf.keras.Input(shape=(CFG.image_size[0], CFG.image_size[1], 4))  # (height, width, channels)

    # Learn a 4-to-3 channel projection using a 1x1 conv layer
    x = tf.keras.layers.Conv2D(
        filters=3,
        kernel_size=(1, 1),
        padding='same',
        activation='linear',
        use_bias=False,
        name='channel_projection'
    )(inp)

    # Load pretrained EfficientNetB0 without the top classification layer
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,  # use None here and load weights manually below
        input_shape=(128, 512, 3)  # input now has 3 channels
    )
    # Load pretrained ImageNet weights
    # base_model.load_weights('/kaggle/input/tf-efficientnet-imagenet/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5')

    # Pass through base model
    x = base_model(x)

    # Global average pooling to reduce spatial dimensions
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Classification layer (softmax for multi-class classification)
    x = tf.keras.layers.Dense(6, activation='softmax', dtype='float32')(x)

    # Define the model
    model = tf.keras.Model(inputs=inp, outputs=x)

    # Compile the model with Adam optimizer and KL divergence loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.KLDivergence(),
        metrics=['accuracy']
    )

    return model



from tensorflow import keras
from tensorflow.keras import layers, regularizers

def build_cnn_model(input_shape=(CFG.image_size[0], CFG.image_size[1], 4), num_classes=6):
    model = keras.Sequential([
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4),
                      activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4), activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')  # Output is a probability distribution
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.KLDivergence(),
        metrics=['accuracy']
    )

    return model


# class MacroF1Score(tf.keras.metrics.Metric):
#     def __init__(self, num_classes, name='macro_f1', **kwargs):
#         super(MacroF1Score, self).__init__(name=name, **kwargs)
#         self.num_classes = num_classes
#         self.precision = tf.keras.metrics.Precision(class_id=None, average='macro')
#         self.recall = tf.keras.metrics.Recall(class_id=None, average='macro')

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_true_labels = tf.argmax(y_true, axis=-1)
#         y_pred_labels = tf.argmax(y_pred, axis=-1)

#         self.precision.update_state(y_true_labels, y_pred_labels, sample_weight)
#         self.recall.update_state(y_true_labels, y_pred_labels, sample_weight)

#     def result(self):
#         p = self.precision.result()
#         r = self.recall.result()
#         return 2 * (p * r) / (p + r + 1e-7)

#     def reset_states(self):
#         self.precision.reset_states()
#         self.recall.reset_states()


#We will try this model next
import tensorflow as tf
from tensorflow.keras import layers, models

L2_REG = 8e-5  # Adjustable

def conv_block(x, filters, kernel_size=3, stride=1):
    shortcut = x

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same',
                      kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride,
                                 kernel_regularizer=regularizers.l2(L2_REG))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_custom_resnet34(input_shape=(128, 256, 4), num_classes=6):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(64, 7, strides=2, padding='same',
                      kernel_regularizer=regularizers.l2(L2_REG))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Residual blocks
    for filters, blocks, stride in [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)]:
        for i in range(blocks):
            x = conv_block(x, filters, stride=stride if i == 0 else 1)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(x)  # ⬅️ more capacity
    x = layers.Dropout(0.3)(x)  # ⬅️ small dropout before final softmax
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    
    initial_learning_rate = 8e-4      # Start slightly higher than fixed
    decay_steps = 514 * CFG.epochs    # Rule of thumb: steps_per_epoch * num_epochs
    alpha = 1e-6                      # Minimum LR value after decay

    cosine_decay_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        alpha=alpha
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=cosine_decay_schedule,
        weight_decay=1e-5
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'), 
            tf.keras.metrics.Recall(name='recall')
            # MacroF1Score(num_classes=6)
        ]
    )
    return model



def get_lr_callback(batch_size=8, mode='cos', epochs=10, plot=False):
    lr_start, lr_max, lr_min = 1e-4, 8e-4, 1e-5 #5e-5, 6e-6 * batch_size, 1e-5
    lr_ramp_ep, lr_sus_ep, lr_decay = 3, 0, 0.75

    def lrfn(epoch):  # Learning rate update function
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        elif mode == 'exp':
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        elif mode == 'step':
            lr = lr_max * lr_decay**((epoch - lr_ramp_ep - lr_sus_ep) // 2)
        elif mode == 'cos':
            decay_total_epochs = epochs - lr_ramp_ep - lr_sus_ep + 3
            decay_epoch_index = epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        return lr

    if plot:  # Plot lr curve if plot is True
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(epochs), [lrfn(epoch) for epoch in np.arange(epochs)], marker='o')
        plt.xlabel('epoch'); plt.ylabel('lr')
        plt.title('LR Scheduler')
        plt.show()

    return keras.callbacks.LearningRateScheduler(lrfn, verbose=True)  # Create lr callback


train.head()


from sklearn.utils import class_weight

classes = np.unique(train['target'])
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=classes,
                                                  y=train['target'])
class_weight_dict = dict(zip(classes, class_weights))


# def main():    
#     # Stratified Group K-Fold
#     sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CFG.seed)
#     lr_plateau_cb = keras.callbacks.ReduceLROnPlateau(
#         monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6
#     )

#     # Model Checkpoint
#     ckpt_cb = keras.callbacks.ModelCheckpoint("best_model.keras",
#                                              monitor='val_loss',
#                                              save_best_only=True,
#                                              save_weights_only=False,
#                                              mode='min')
#     #Early Stopping
#     early_stopping = keras.callbacks.EarlyStopping(
#         monitor="val_loss", patience=3, restore_best_weights=True
#     )
    
#     model = build_custom_resnet34() #build_cnn_model()
#     model.summary()
    
#     all_fold_preds = []
#     all_fold_trues = []

#     for i, (train_index, valid_index) in enumerate(sgkf.split(train, train.target, train.patient_id)):  
#         print('#'*25)
#         print(f'### Fold {i+1}')

#         # Get the patient IDs for this fold
#         train_patients = train.iloc[train_index]['patient_id'].unique()
#         valid_patients = train.iloc[valid_index]['patient_id'].unique()
        
#         # Check for patient leakage between train and validation
#         common_patients = set(train_patients) & set(valid_patients)
#         if len(common_patients) > 0:
#             print(f"WARNING: Data leakage detected! {len(common_patients)} patients appear in both train and validation sets")
#             print(f"Common patients: {common_patients}")
#         else:
#             print("No patient leakage detected between train and validation sets")
        
#         # Check class distribution
#         train_targets = train.iloc[train_index]['target']
#         valid_targets = train.iloc[valid_index]['target']
        
#         # Verify no samples from same patient are in both sets
#         assert len(set(train.iloc[train_index]['patient_id']) & 
#                set(train.iloc[valid_index]['patient_id'])) == 0, "Patient leakage detected!"
        
#         train_ds = DataGenerator(train.iloc[train_index], shuffle=True, batch_size=CFG.batch_size, augment=False)
#         valid_ds = DataGenerator(train.iloc[valid_index], shuffle=False, batch_size=CFG.batch_size, mode='valid')        
        
#         # Train Model
#         history = model.fit(
#             train_ds, 
#             epochs=CFG.epochs,
#             callbacks=[lr_plateau_cb, ckpt_cb, early_stopping],
#             validation_data=valid_ds,
#             verbose=CFG.verbose
#         )
        
#         preds = model.predict(valid_ds)
#         preds_classes = np.argmax(preds, axis=1)
#         # true_classes = train.iloc[valid_index]['target'].values
#         true_classes = train.iloc[valid_index]['target_encoded'].values
        
#         # AUC Score (multi-class, average='macro')
#         auc = roc_auc_score(tf.one_hot(true_classes, depth=preds.shape[1]).numpy(), preds, average='macro')
#         print(f"Fold {i+1} AUC Score (Macro Average): {auc:.4f}")

#         # Confusion Matrix
#         cm = confusion_matrix(true_classes, preds_classes)
#         print(f"Fold {i+1} Confusion Matrix:\n{cm}")

#         # Classification Report
#         report = classification_report(true_classes, preds_classes, digits=4)
#         print(f"Fold {i+1} Classification Report:\n{report}")

#         plt.figure(figsize=(8, 6))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                     xticklabels=[str(i) for i in range(preds.shape[1])], 
#                     yticklabels=[str(i) for i in range(preds.shape[1])])
#         plt.title(f'Confusion Matrix - Fold {i+1}')
#         plt.xlabel('Predicted Label')
#         plt.ylabel('True Label')
#         plt.show()

#         # Store for later if you want overall average
#         all_fold_preds.append(preds_classes)
#         all_fold_trues.append(true_classes)

#         # ========== NEW BLOCK ENDS HERE ==========

#     # Optionally after all folds:
#     all_fold_preds = np.concatenate(all_fold_preds)
#     all_fold_trues = np.concatenate(all_fold_trues)
    
#     overall_auc = roc_auc_score(tf.one_hot(all_fold_trues, depth=preds.shape[1]).numpy(), 
#                                 tf.one_hot(all_fold_preds, depth=preds.shape[1]).numpy(), 
#                                 average='macro')
#     print(f"Overall AUC Score across all folds: {overall_auc:.4f}")
#     print("Overall Confusion Matrix:")
#     print(confusion_matrix(all_fold_trues, all_fold_preds))
#     print("Overall Classification Report:")
#     print(classification_report(all_fold_trues, all_fold_preds, digits=4))


from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def main():    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CFG.seed)

    all_fold_preds = []
    all_fold_trues = []
    fold_metrics = []

    for i, (train_index, valid_index) in enumerate(sgkf.split(train, train.target, train.patient_id)):  
        print('#' * 25)
        print(f'### Fold {i+1}')

        # StratifiedGroupKFold splits
        train_data = train.iloc[train_index]
        valid_data = train.iloc[valid_index]

        # Double-check leakage
        assert len(set(train_data['patient_id']) & set(valid_data['patient_id'])) == 0, "Patient leakage detected!"

        # Datasets
        train_ds = DataGenerator(train_data, shuffle=True, batch_size=CFG.batch_size, augment=True)
        valid_ds = DataGenerator(valid_data, shuffle=False, batch_size=CFG.batch_size, mode='valid')

        # Model
        model = build_custom_resnet34()  # fresh model for each fold
        # model.summary()

        # Callbacks
        # lr_plateau_cb = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, 
        #                                                   min_lr=1e-6)
        
        ckpt_cb = keras.callbacks.ModelCheckpoint(f"best_model_fold{i+1}.keras", 
                                                   monitor='val_loss', save_best_only=True, mode='min')
        
        early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, 
                                                          restore_best_weights=True)

        # Train
        history = model.fit(
            train_ds,
            epochs=CFG.epochs,
            validation_data=valid_ds,
            callbacks=[ckpt_cb, early_stopping_cb],
            verbose=CFG.verbose,
            class_weight=class_weight_dict
        )

        preds = model.predict(valid_ds)

        # Convert predictions to class indices
        pred_classes = np.argmax(preds, axis=1)
        
        # Get true class indices
        true_classes = np.array([class_to_idx[t] for t in train.iloc[valid_index]['target']])
        true_classes2 = np.array([class_to_idx[t] for t in valid_data['target']])
        print(true_classes, true_classes2)
        cr = classification_report(true_classes, pred_classes, 
                                   target_names=class_names, labels=list(range(len(class_names))), digits=4,
                                   zero_division=0, output_dict=True)
        # 1. Classification Report
        cr_df = pd.DataFrame(cr).transpose()    # <-- FIX: pretty print
        display(cr_df)
    
        # 2. Confusion Matrix
        plt.figure(figsize=(10,8))
        cm = confusion_matrix(true_classes, pred_classes, labels=np.arange(len(class_names)))
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
        
        del model, preds
        gc.collect()

    # Calculate overall metrics
    print('\n' + '#'*50)
    print('### FINAL OVERALL RESULTS ACROSS ALL FOLDS')
    print('#'*50 + '\n')
    
    # Convert to arrays
    all_true = np.array(all_fold_trues)
    all_pred = np.array(all_fold_preds)
    
    # 1. Overall Classification Report
    print("Overall Classification Report:")
    overall_cr = classification_report(all_true, all_pred,
                               target_names=class_names,
                               labels=np.arange(len(class_names)),
                               digits=4,
                               zero_division=0,
                               output_dict=True)
    
    overall_cr_df = pd.DataFrame(overall_cr).transpose()
    display(overall_cr_df)
    
    # 2. Overall Confusion Matrix
    plt.figure(figsize=(10,8))
    overall_cm = confusion_matrix(all_true, all_pred, labels=np.arange(len(class_names)))
    sns.heatmap(overall_cm, annot=True, fmt='d',
               xticklabels=class_names,
               yticklabels=class_names,
               cmap='Blues')
    
    plt.title('Overall Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # 4. Print fold-wise summary
    print("\nFold-wise Performance Summary:")
    summary_df = pd.DataFrame(fold_metrics)[['fold', 'accuracy']]
    print(summary_df.to_string(index=False))
    
    print(f"\nMean Accuracy: {summary_df['accuracy'].mean():.4f}")

        


main()
#validation loss stuck at 1.37 with spectrogram-3000 data and spectrogram_3000-128_512. See version 1 of this notebook


# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix

# def plot_conf_matrix(y_true, y_pred, class_names, fold_idx):
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=class_names, yticklabels=class_names)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title(f'Fold {fold_idx+1} Confusion Matrix')
#     plt.tight_layout()
#     plt.show()



# for fold, (train_idx, valid_idx) in enumerate(
    #     sgkf.split(df, y=df["class_label"], groups=df["patient_id"])
    # ):
    #     df.loc[valid_idx, "fold"] = fold
    # df.groupby(["fold", "class_name"])[["eeg_id"]].count().T

    # # Sample from full data
    # sample_df = df.groupby("spectrogram_id").head(1).reset_index(drop=True)
    # train_df = sample_df[sample_df.fold != CFG.fold]
    # valid_df = sample_df[sample_df.fold == CFG.fold]
    # print(f"# Num Train: {len(train_df)} | Num Valid: {len(valid_df)}")

    # # Build Datasets
    # train_paths = train_df.spec2_path.values
    # train_offsets = train_df.spectrogram_label_offset_seconds.values.astype(int)
    # train_labels = train_df.class_label.values
    # train_ds = build_dataset(train_paths, train_offsets, train_labels, batch_size=CFG.batch_size,
    #                          repeat=True, shuffle=True, augment=True, cache=True)

    # valid_paths = valid_df.spec2_path.values
    # valid_offsets = valid_df.spectrogram_label_offset_seconds.values.astype(int)
    # valid_labels = valid_df.class_label.values
    # valid_ds = build_dataset(valid_paths, valid_offsets, valid_labels, batch_size=CFG.batch_size,
    #                          repeat=False, shuffle=False, augment=False, cache=True)

    # # Dataset Check
    # # plot_dataset_samples(train_ds)
    
    # # Build Model
    # model = build_model()
    # model.summary()

    # # LR Schedule
    # lr_cb = get_lr_callback(batch_size=CFG.batch_size, mode=CFG.lr_mode, plot=True)

    # # Model Checkpoint
    # ckpt_cb = keras.callbacks.ModelCheckpoint("best_model.keras",
    #                                          monitor='val_loss',
    #                                          save_best_only=True,
    #                                          save_weights_only=False,
    #                                          mode='min')

    # # Train Model
    # history = model.fit(
    #     train_ds, 
    #     epochs=CFG.epochs,
    #     callbacks=[lr_cb, ckpt_cb], 
    #     steps_per_epoch=len(train_df)//CFG.batch_size,
    #     validation_data=valid_ds, 
    #     verbose=CFG.verbose
    # )

