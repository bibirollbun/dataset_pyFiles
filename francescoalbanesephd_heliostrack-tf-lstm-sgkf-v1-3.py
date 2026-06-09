import os
import random
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

SEED = 42
set_seed(SEED)


# Load the datasets
TRAIN_DF = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv",index_col=0)
TEST_DF = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv",index_col=0)
TRAIN_DEM_DF = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")
TEST_DEM_DF = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")

# Load datasets (on Colab)
# import os
# TRAIN_DF = pd.read_csv(os.path.join(cmi_detect_behavior_with_sensor_data_path, 'train.csv'))
# TEST_DF = pd.read_csv(os.path.join(cmi_detect_behavior_with_sensor_data_path, 'test.csv'))
# TRAIN_DEM_DF = pd.read_csv(os.path.join(cmi_detect_behavior_with_sensor_data_path, 'train_demographics.csv'))
# TEST_DEM_DF = pd.read_csv(os.path.join(cmi_detect_behavior_with_sensor_data_path, 'test_demographics.csv'))


# Keep only IMU features (drop thm_ and tof_ features from TRAIN and TEST)
NON_IMU_FEATURES = [col for col in TRAIN_DF.columns if col.startswith('thm_') or col.startswith('tof_')]
TRAIN_DF = TRAIN_DF.drop(NON_IMU_FEATURES, axis=1)
TEST_DF = TEST_DF.drop(NON_IMU_FEATURES, axis=1)


# Print helper function
def print_with_sep(text,sep="=",n=20):
  print("\n")
  print(sep*n)
  print(text)
  print(sep*n)

# Check shapes of all 4 datasets
datasets = {'TRAIN_DF': TRAIN_DF, 'TEST_DF': TEST_DF, 'TRAIN_DEM_DF': TRAIN_DEM_DF, 'TEST_DEM_DF': TEST_DEM_DF}

# Check shapes
print_with_sep("Shapes")
for name, df in datasets.items():
  print(f"{name} shape: {df.shape}")

# Check duplicates
print_with_sep("Duplicates")
for name, df in datasets.items():
  print(f"{name} duplicates: {df.duplicated().sum()}")

# Check nans
print_with_sep("NaNs")
for name, df in datasets.items():
  print(f"{name} NaNs: {df.isnull().sum().sum()}")

# Check col difference
print_with_sep("Columns not in test")
print(set(TRAIN_DF.columns).difference(set(TEST_DF.columns)))


# Check descriptive stats
print_with_sep("Descriptive Statistics")
for name, df in datasets.items():
  print(f"{name} Description:")
  display(df.describe(include='all').T.replace(np.nan,'-').style.background_gradient(cmap='Blues',subset='count'))
  print("\n")


# Identify target
target = 'gesture'

# Distribution of the target variable
plt.figure(figsize=(12, 5))
sns.countplot(data=TRAIN_DF, y=target, order = TRAIN_DF[target].value_counts().index, palette='viridis')
plt.title(f'Distribution of {target.capitalize()} (Target Variable)')
plt.xlabel('Count')
plt.ylabel(f'{target.capitalize()}')
plt.show()


# Relationship between numerical features and the target variable (using boxplots)

# Get the numerical features excluding the target and the index
numerical_features = TRAIN_DF.select_dtypes(include=np.number).columns.tolist()

# Set up the subplot grid
fig, axes = plt.subplots(4, 2, figsize=(18, 3 * 6))
axes = axes.flatten()

# Iterate through the numerical features and create boxplots
for i, feature in enumerate(numerical_features):
    sns.boxplot(y=target, x=feature, data=TRAIN_DF, ax=axes[i], hue = target, palette='viridis')
    axes[i].set_title(f'Target vs {feature}')
    axes[i].set_ylabel('Target')
    axes[i].set_xlabel(feature)
    axes[i].legend_.remove()

plt.tight_layout()
plt.show()


# # Perform univariate analysis for each numerical variable
# for variable in TRAIN_DF.select_dtypes(include='number'):
#     # Create subplots
#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#     annot_kws = {'xy': (0.70, 0.75), 'xycoords': 'axes fraction', 'fontsize': 10}

#     # Box plot
#     sns.boxplot(data=TRAIN_DF, x=variable, ax=axes[0])
#     axes[0].set_xlabel(variable)
#     axes[0].set_title(f"Box Plot of {variable}")

#     # Histogram
#     sns.histplot(data=TRAIN_DF, x=variable, kde=True, bins=30, ax=axes[1])
#     axes[1].set_xlabel(variable)
#     axes[1].set_ylabel("Frequency")
#     axes[1].set_title(f"Histogram of {variable} [Train]")
#     # axes[1].legend()
#     axes[1].annotate(f"Skewness (TRAIN): {TRAIN_DF[variable].skew():.2f}\nKurtosis (TRAIN): {TRAIN_DF[variable].kurt():.2f}",
#                      xy=annot_kws['xy'], xycoords=annot_kws['xycoords'], fontsize=annot_kws['fontsize'])


#     # Adjust spacing and show
#     plt.tight_layout()
#     plt.show()


# TRAIN_DF.select_dtypes(exclude='number').columns.difference([target]+['row_id', 'sequence_id'])


# # Define features to investigate
# cat_cols = TRAIN_DF.select_dtypes(exclude='number').columns.difference([target]+['row_id', 'sequence_id','subject'])

# # Visualise categorical variables
# fig, axes = plt.subplots(2,2,figsize=(15, 5))
# ax = axes.flatten()

# for i, col in enumerate(TRAIN_DF[cat_cols]):
#     sns.countplot(data=TRAIN_DF, y=col, order = TRAIN_DF[col].value_counts().index, palette='viridis', ax=ax[i])

# plt.tight_layout()


import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

def compute_angular_features(df, seq_col='sequence_id', sampling_rate=200):
    dt = 1.0 / sampling_rate
    results = []

    for _, group in df.groupby(seq_col):
        quats = group[["rot_x", "rot_y", "rot_z", "rot_w"]].to_numpy()

        omega = np.zeros((len(quats), 3))  # fill with zeros first

        for i in range(len(quats)-1):
            # Start a try loop to avoid invalid (zero norm) quaternions
            try:
                q1 = quats[i]
                q2 = quats[i+1]

                # Convert to Rotation object
                r1 = R.from_quat(q1)
                r2 = R.from_quat(q2)

                # Compute relative rotations (t -> t+1)
                delta = r2 * r1.inv()

                # Convert to rotation vectors
                omega[i+1] = delta.as_rotvec() / dt

            except Exception:
                # if anything goes wrong, keep zeros
                pass

        # angular acceleration = diff of omega
        alpha = np.zeros_like(omega)
        alpha[1:] = np.diff(omega, axis=0) / dt

        # Append
        group = group.copy()
        group[["omega_x", "omega_y", "omega_z"]] = omega
        group[["alpha_x", "alpha_y", "alpha_z"]] = alpha
        results.append(group)
        
    return pd.concat(results, ignore_index=True)



def add_physics_features(
    data,
    seq_col = "sequence_id",
    time_col = "sequence_counter",
    use_bfill = True,
    add_cross_terms = False,
    add_stats_features = False
):

    Îµ = 1e-8
    df = data.copy()

    # # sort within each sequence to ensure correct differencing
    # df = df.sort_values([seq_col, time_col])

    # Compute angular features
    df = compute_angular_features(df, seq_col=seq_col)

    # numeric columns except the counter
    num_cols = df.select_dtypes(include=np.number).columns.difference([time_col])
    
    # per-sequence imputation: forward-fill, optionally backward-fill (beware of leakage)
    if use_bfill:
        df[num_cols] = df.groupby(seq_col)[num_cols].apply(lambda x:x.ffill().bfill()).reset_index(level=0, drop=True)
        df[num_cols] = df.groupby(seq_col)[num_cols].apply(lambda x:x.fillna(x.min()).fillna(0)).reset_index(level=0, drop=True) # fill remaining nans
    else:
        df[num_cols] = df.groupby(seq_col)[num_cols].apply(lambda x:x.ffill()).reset_index(level=0, drop=True)
        df[num_cols] = df.groupby(seq_col)[num_cols].apply(lambda x:x.fillna(x.min()).fillna(0)).reset_index(level=0, drop=True) # fill remaining nans

    # --- ACCELERATION FEATURES ---
    acc_cols = [c for c in df.columns if c.startswith("acc")]
    for col in acc_cols:
        df[f"diff1_{col}"] = df.groupby(seq_col)[col].diff() # jerk
        
    # --- ROTATION FEATURES ---
    rot_cols = [c for c in df.columns if c.startswith("alpha")]
    for col in rot_cols:
        df[f"diff1_{col}"] = df.groupby(seq_col)[col].diff() # angular jerk 

    # --- ACCELERATION AND ROTATION MAGNITUDES ---
    df["acc_mag"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
    df["rot_mag"] = np.sqrt(df["alpha_x"]**2 + df["alpha_y"]**2 + df["alpha_z"]**2)
    
    # fill NaNs introduced by diff with 0
    diff_mask = [c for c in df.columns if c.startswith("diff")]
    df[diff_mask] = df[diff_mask].fillna(0.0)

    return df



# Test function on TEST_DF
display(add_physics_features(TEST_DF))


TRAIN_DF = add_physics_features(TRAIN_DF)
TEST_DF = add_physics_features(TEST_DF)


type_to_num = {
    "Target": 1,
    "Non-Target":0
    }

label_to_num = {
    'Drink from bottle/cup': 0,  # < ------- NON-TARGETS START
    'Feel around in tray and pull out an object': 0,
    'Glasses on/off': 0,
    'Pinch knee/leg skin': 0,
    'Pull air toward your face': 0,
    'Scratch knee/leg skin': 0,
    'Text on phone': 0,
    'Wave hello': 0,
    'Write name in air': 0,
    'Write name on leg': 0,  # < ------- NON-TARGETS END
    'Above ear - pull hair': 1,  # < ------- TARGETS START
    'Cheek - pinch skin': 2,
    'Eyebrow - pull hair': 3,
    'Eyelash - pull hair': 4,
    'Forehead - pull hairline': 5,
    'Forehead - scratch': 6,
    'Neck - pinch skin': 7,
    'Neck - scratch': 8,  # < ------- TARGETS END
}

# Map target (label_to_num)
TRAIN_DF['gesture'].map(label_to_num)


# Define reverse dictionaries
num_to_label = {v: k for k, v in label_to_num.items()}
num_to_type = {v: k for k, v in type_to_num.items()}

# Leave 0 value in num_to_label as a random non-target type (I don't need to trace back to specific non-target gesture. See challenge overview)
# num_to_label[0] = num_to_type[0]
num_to_label


# Encode target with map
TRAIN_DF['gesture'] = TRAIN_DF['gesture'].map(label_to_num)
TRAIN_DF['gesture'].unique()


# Segment data by sequence_id (TRAIN_DF)
features = TRAIN_DF.select_dtypes(include=np.number).columns.difference(['sequence_counter','gesture']).tolist()
train_sequences = []
train_groups = []
targets = []

for sequence_id, group_df in TRAIN_DF.groupby('sequence_id'):
    train_sequences.append(group_df[features].values)
    train_groups.append(group_df['subject'].iloc[0])
    targets.append(group_df['gesture'].iloc[0])

len(train_groups)


# Segment data by sequence_id (TEST_DF)
test_sequences = []
test_groups = []

for sequence_id, group_df in TEST_DF.groupby('sequence_id'):
    test_sequences.append(group_df[features].values)
    test_groups.append(group_df['subject'].iloc[0])

len(test_groups)


seq_lengths = [s.shape[0] for s in train_sequences]

plt.figure(figsize=(12, 5))
sns.histplot(seq_lengths, bins=100)
plt.title("Sequence Lengths")
plt.show()


# Convert to pandas Series to use quantile
seq_lengths_series = pd.Series(seq_lengths)

# Calculate the 95th percentile and assign it to MAXLEN
MAXLEN = int(seq_lengths_series.quantile(0.95))
print(f"95th percentile of sequence lengths: {MAXLEN}")


from tensorflow.keras.preprocessing.sequence import pad_sequences

# pad_sequences (train)
X_padded = pad_sequences(
    train_sequences,
    maxlen=MAXLEN,
    dtype='float32',
    padding='post',
    truncating='post',
    value=0
)


# pad_sequences (test)
X_test_padded = pad_sequences(
    test_sequences,
    maxlen=MAXLEN,
    dtype='float32',
    padding='post',
    truncating='post',
    value=0
)


# Initialise sgkf
sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Define y
y = np.array(targets)

# Define folds
folds = list(sgkf.split(X_padded, y, train_groups))


def build_model(input_shape, n_classes):
    inputs = keras.Input(shape=input_shape)

    # Mask padded timesteps
    x = layers.Masking(mask_value=0)(inputs)

    # Bidirectional LSTM
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x); x = layers.Dropout(0.1)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x); x = layers.Dropout(0.1)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x); x = layers.Dropout(0.1)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x); x = layers.Dropout(0.1)(x)

    # Global pooling â†’ converts (batch, time, features) â†’ (batch, features)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


histories = []
oof_preds = np.zeros((len(y), len(np.unique(y))))
test_preds = np.zeros((len(X_test_padded), len(np.unique(y))))

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers

n_classes = len(np.unique(y))

for fold, (train_idx, val_idx) in enumerate(folds):
    print(f"\n=== Fold {fold+1} ===")

    X_train, X_val = X_padded[train_idx], X_padded[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Fit scaler on train only
    n_features = X_train.shape[2]
    scaler = StandardScaler()

    # Flatten time axis for fitting scaler
    scaler.fit(
        X_train.reshape(-1, n_features)
    )

    # Compute class weights
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    # Convert class weights to a dictionary
    class_weight_dict = dict(zip(np.unique(y_train), weights))
    print(class_weight_dict)

    # transform and reshape back
    X_train_scaled = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test_padded.reshape(-1, n_features)).reshape(X_test_padded.shape)

    model = build_model(
        input_shape=(MAXLEN, n_features),
        n_classes=n_classes
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    histories.append(history)

    # Predict OOF and test
    oof_preds[val_idx] = model.predict(X_val_scaled, verbose=0)
    test_preds += model.predict(X_test_scaled)


# histories[0].model.predict(X_test_scaled[0])
histories[0].model.predict(X_test_scaled).argmax(axis=1)


# Plot accuracy per fold

# subplots
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
axes = ax.flatten()

for i, h in enumerate(histories):
    sns.lineplot(h.history['val_accuracy'], label=f'Fold {i}', ax=axes[0])

plt.title("Validation Accuracy per Fold")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()


# Plot val_loss per fold
for i, h in enumerate(histories):
    sns.lineplot(h.history['val_loss'], label=f'Fold {i}', ax=axes[1])

plt.title("Validation Loss per Fold")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


from sklearn.metrics import classification_report

y_oof_pred = oof_preds.argmax(axis=1)

# Get the original gesture class names from the label encoder
gesture_classes = num_to_label.values()

print(
    classification_report(y, y_oof_pred, target_names=gesture_classes)
)


test_preds = np.zeros((len(X_test_padded), len(np.unique(y))))

for i,h in enumerate(histories):
    partial_predictions = h.model.predict(X_test_padded)
    test_preds += partial_predictions

test_preds.argmax(axis=1)


import polars as pl

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:

    TEST_DF = sequence.to_pandas()
    TEST_DF = add_physics_features(TEST_DF)

    # Segment data by sequence_id (TEST_DF)
    test_sequences = []
    test_groups = []

    for sequence_id, group_df in TEST_DF.groupby('sequence_id'):
        test_sequences.append(group_df[features].values)
        test_groups.append(group_df['subject'].iloc[0])

    # pad_sequences (test)
    X_test_padded = pad_sequences(
        test_sequences,
        maxlen=MAXLEN,
        dtype='float32',
        padding='post',
        truncating='post',
        value=0
    )

    # transform and reshape back
    X_test_scaled = scaler.transform(X_test_padded.reshape(-1, n_features)).reshape(X_test_padded.shape)

    # Initialize test_preds
    test_preds = np.zeros((len(X_test_scaled), len(np.unique(y))))

    # Loop through partial models
    for i,h in enumerate(histories):
        partial_predictions = h.model.predict(X_test_scaled)
        test_preds += partial_predictions
    
        # Predict test
        test_preds += model.predict(X_test_scaled)
    
    prediction = pd.Series(test_preds.argmax(axis=1)).map(num_to_label)
    print(prediction)

    return prediction[0]


import kaggle_evaluation.cmi_inference_server
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )

