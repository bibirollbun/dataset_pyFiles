import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
import keras
import keras_cv
import matplotlib.pyplot as plt
import math
import warnings
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import tensorflow.keras.backend as K, gc
from sklearn.utils import class_weight
import albumentations as albu
from sklearn.utils.class_weight import compute_class_weight
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
    input_shape = (19, 1000, 1)
    epochs = 10  # Training epochs
    n_splits = 5 # Number of folds
    segments_per_eeg = 8

    sample_batch_size = 64

    num_classes = 6  # Number of classes in the dataset


NPY_DIR = '/kaggle/input/eeg-npy-unfiltered-90/eeg_npy_unfiltered/'


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
train = train[train['max_vote_percentage']>=.9]


frequency = train['target'].value_counts()
print(frequency)


class_names = ['GPD', 'GRDA', 'LPD', 'LRDA', 'Other', 'Seizure']
class_to_idx = {name: i for i, name in enumerate(class_names)}


classes = np.unique(train['target'])
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=classes,
                                                  y=train['target'])
class_weight_dict = dict(zip(classes, class_weights))


all_eegs = {}
for eeg_id in train.eeg_id.unique():    
    for i in range(1, CFG.segments_per_eeg + 1):  # segments 1 to 10
        x = np.load(f'{NPY_DIR}{eeg_id}_{i}.npy')
        all_eegs[(eeg_id, i)] = x  # key is tuple (eeg_id, segment_index)


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, batch_size=32, shuffle=False, augment=False, mode='train',
                 eeg_specs=all_eegs, segments_per_eeg=CFG.segments_per_eeg, target_map=class_to_idx):

        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.mode = mode
        self.eeg_specs = eeg_specs
        self.segments_per_eeg = segments_per_eeg
        self.target_map = target_map

        # Create a map from sample index to (original_data_index, segment_index)
        self.sample_map = []
        for i in range(len(self.data)): # Iterate through original eeg_ids
            eeg_id = self.data.iloc[i].eeg_id
            for seg in range(1, self.segments_per_eeg + 1):
                # Check if this specific segment exists in our loaded data
                if (eeg_id, seg) in self.eeg_specs:
                     self.sample_map.append({'data_idx': i, 'segment_idx': seg})

        self.on_epoch_end()

    def __len__(self):
        ct = math.ceil(len(self.sample_map) / self.batch_size)
        print('len ', ct)
        return ct

    def __getitem__(self, index):
        'Generate one batch of data'
        # Get the indices for the *samples* in this batch
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        print('start & end index ', start_idx, end_idx)
        # Get the sample map indices for this batch from the shuffled list
        batch_sample_map_indices = self.indexes[start_idx : end_idx]

        # Generate data based on these sample map indices
        X, y = self.__data_generation(batch_sample_map_indices)

        if self.augment:
            X = self.__augment_batch(X)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # Indexes now refer to the positions in self.sample_map
        self.indexes = np.arange(len(self.sample_map))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_sample_map_indices):
        'Generates data containing batch_size samples'

        # The size of X and y is determined by how many sample indices we got
        num_samples_in_batch = len(batch_sample_map_indices)
        print('num_samples_in batch ', num_samples_in_batch)
        X = np.zeros((num_samples_in_batch, CFG.input_shape[0], CFG.input_shape[1], CFG.input_shape[2]), dtype='float32')
        y = np.zeros(num_samples_in_batch, dtype='int32')

        for j, map_idx in enumerate(batch_sample_map_indices):
            # Get the actual mapping info {data_idx, segment_idx}
            sample_info = self.sample_map[map_idx]
            original_data_idx = sample_info['data_idx']
            segment_idx = sample_info['segment_idx']

            row = self.data.iloc[original_data_idx]
            eeg_id = row.eeg_id
            target_val = row['target'] 
            target_idx = self.target_map[target_val]

            x = self.eeg_specs.get((eeg_id, segment_idx))

            if x is not None:
                if x.shape == (1000, 19) and CFG.input_shape[0] == 19 and CFG.input_shape[1] == 1000:
                     X[j, :, :, 0] = np.transpose(x) # shape (19, 1000)
                elif x.shape == (19, 1000) and CFG.input_shape[0] == 19 and CFG.input_shape[1] == 1000:
                     X[j, :, :, 0] = x # shape (19, 1000)
                else:
                    print(f"Warning: Shape mismatch for eeg {eeg_id}, seg {segment_idx}. Got {x.shape}, expected (1000, 19) or (19, 1000). Skipping.")
                    y[j] = -1
                    continue 

                y[j] = target_idx
            else:
                 print(f"Warning: Data for eeg {eeg_id}, segment {segment_idx} not found in eeg_specs dict. Skipping sample.")
                 y[j] = -1

        valid_indices = np.where(y != -1)[0]
        X = X[valid_indices]
        y = y[valid_indices]

        return X, y


    def __random_transform(self, img):        
        composition = albu.Compose([            
            albu.GaussNoise(var_limit=(1.0, 10.0), p=0.3),
            albu.CoarseDropout(max_holes=8, max_height=4, max_width=100, fill_value=0, p=0.5),
        ])
        
        return composition(image=img)['image']

    def __augment_batch(self, img_batch):
        # Input img_batch shape: (batch_size, 19, 1000, 1)
        for i in range(img_batch.shape[0]):
            img_batch[i,] = self.__random_transform(img_batch[i,])
        return img_batch


def build_eeg_cnn(input_shape=(19, 1000, 1), num_classes=6):
    inputs = tf.keras.Input(shape=input_shape)
    
    l2_lambda = 1e-5
    
    x = layers.Conv2D(32, (3, 11), activation='relu', padding='same',
                      kernel_regularizer=keras.regularizers.l2(l2_lambda))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((1, 2))(x)

    x = layers.Conv2D(64, (3, 9), activation='relu', padding='same',
                      kernel_regularizer=keras.regularizers.l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((1, 2))(x)

    x = layers.Conv2D(128, (3, 7), activation='relu', padding='same',
                      kernel_regularizer=keras.regularizers.l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((1, 2))(x)

    x = layers.Conv2D(256, (3, 5), activation='relu', padding='same',
                      kernel_regularizer=keras.regularizers.l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)
    # x = layers.MaxPooling2D((1, 2))(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu', 
                     kernel_regularizer=keras.regularizers.l2(l2_lambda))(x)
    x = layers.Dropout(0.6)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    initial_learning_rate = 5e-4    
    decay_steps = 5000  
    alpha = 1e-6

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
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            'accuracy'
        ]
    )

    return model


def main():
    sgkf = StratifiedGroupKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)

    all_fold_preds = []
    all_fold_trues = []
    fold_metrics = []
    
    target_for_split = train['target'] 
    
    for i, (train_index, valid_index) in enumerate(sgkf.split(train, target_for_split, train.patient_id)):
        print('#' * 25)
        print(f'### Fold {i+1}')
    
        # StratifiedGroupKFold splits based on original train DataFrame
        train_data = train.iloc[train_index].reset_index(drop=True) # Reset index for clean iloc access later
        valid_data = train.iloc[valid_index].reset_index(drop=True) # Reset index
    
        # Double-check leakage
        assert len(set(train_data['patient_id']) & set(valid_data['patient_id'])) == 0, "Patient leakage detected!"
    
        train_ds = DataGenerator(train_data, shuffle=True, batch_size=CFG.sample_batch_size, augment=True,
                                 eeg_specs=all_eegs, segments_per_eeg=CFG.segments_per_eeg, target_map=class_to_idx)
        valid_ds = DataGenerator(valid_data, shuffle=False, batch_size=CFG.sample_batch_size, mode='valid',
                                 eeg_specs=all_eegs, segments_per_eeg=CFG.segments_per_eeg, target_map=class_to_idx)
    
        keras.backend.clear_session()
        model = build_eeg_cnn()
        model.summary()
    
        # Callbacks
        ckpt_path = f"best_model_fold{i+1}.keras"
        ckpt_cb = keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False, # Save entire model
            mode='min'
        )
    
        early_stopping_cb = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5, # Increased patience slightly
            restore_best_weights=True # Restore weights from best epoch
        )
    
        # --- Train ---
        print(f"\nTraining Fold {i+1}...")
        history = model.fit(
            train_ds,
            epochs=CFG.epochs,
            validation_data=valid_ds,
            callbacks=[ckpt_cb, early_stopping_cb],
            verbose=CFG.verbose,
            class_weight=class_weight_dict
        )
    
        print(f"\nLoading best weights for Fold {i+1} from {ckpt_path}...")
        
        best_model = keras.models.load_model(ckpt_path)
    
        print(f"Evaluating Fold {i+1} on validation data...")
        # Evaluate using the generator
        results = best_model.evaluate(valid_ds, verbose=0)
        fold_metrics.append(results)
        print(f"Fold {i+1} Validation Metrics (Loss, Acc): {results}")
    
    
        print(f'### Fold {i+1} Finished ###\n')
        if i == 1:
            break


main()

