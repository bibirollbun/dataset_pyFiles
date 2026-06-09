!pip install -q polars


import math, re, os
import tensorflow as tf
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
#from kaggle_datasets import KaggleDatasets
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import kaggle_evaluation.cmi_inference_server



# copy&paste from https://www.kaggle.com/code/ryanholbrook/getting-started-with-tpus
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE

# Detect TPU, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() 
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None
    print("not running on TPU")

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() 

print("REPLICAS: ", strategy.num_replicas_in_sync)


print(tf.config.list_physical_devices('GPU'))



# Load data
train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
targets = train_df[['sequence_id', 'gesture']].drop_duplicates()



train_df.head()


train_demographics = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")
train_demographics.head()


train_df = train_df.merge(train_demographics, on='subject', how='left')
train_df.fillna(method='ffill', inplace=True)


train_df.head()


train_df.describe()



', '.join(train_df.columns.to_list())



# Encode gesture labels
label_encoder = LabelEncoder()
targets['gesture_enc'] = label_encoder.fit_transform(targets['gesture'])
gesture2id = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Features to use (IMU only)
FEATURES = [
    'acc_x', 'acc_y', 'acc_z',
    'rot_w', 'rot_x', 'rot_y', 'rot_z'
]





# Feature selection
IMU_FEATURES = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
THERMO_FEATURES = [f'thm_{i}' for i in range(1, 6)]
TOF_FEATURES = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]
DEMO_FEATURES = ['adult_child', 'age', 'sex', 'handedness', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']

FEATURES = IMU_FEATURES + THERMO_FEATURES + DEMO_FEATURES  + TOF_FEATURES # TOF excluded for now due to sparsity


FEATURES


# Prepare sequences
sequence_ids = train_df['sequence_id'].unique()
X, y = [], []
for seq_id in sequence_ids:
    df = train_df[train_df['sequence_id'] == seq_id]
    if df[FEATURES].isnull().values.any():
        continue  # skip incomplete sequences
    x = df[FEATURES].values.astype(np.float32)
    if x.shape[0] < 64:
        pad_width = 64 - x.shape[0]
        x = np.pad(x, ((0, pad_width), (0, 0)), mode='edge')
    else:
        x = x[:64]
    X.append(x)
    y.append(targets.loc[targets['sequence_id'] == seq_id, 'gesture_enc'].values[0])

X = np.stack(X)
y = np.array(y)


# Train/Val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
BATCH_SIZE = 64 * strategy.num_replicas_in_sync

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1024).batch(BATCH_SIZE).prefetch(AUTO)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(AUTO)


# Build model with TPU strategy
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, len(FEATURES))),
        tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(128, 5, padding='same', activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

model


# Train
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_class_output_loss',
    patience=5,
    restore_best_weights=True, mode="min"
)

history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[early_stop])




# Evaluate model
val_preds = model.predict(val_ds)
y_val_pred = np.argmax(val_preds, axis=1)

from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

binary_true = np.isin(y_val, [label_encoder.transform([g])[0] for g in label_encoder.classes_ if g != 'non_target']).astype(int)
binary_pred = np.isin(y_val_pred, [label_encoder.transform([g])[0] for g in label_encoder.classes_ if g != 'non_target']).astype(int)

binary_f1 = f1_score(binary_true, binary_pred)
macro_f1 = f1_score(y_val, y_val_pred, average='macro')
final_score = 0.5 * (binary_f1 + macro_f1)

print(f"Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f}, Final Score: {final_score:.4f}")

print(f"Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f}, Final Score: {final_score:.4f}")


#model.save("./model")
#pd.Series(label_encoder.classes_).to_csv("gesture_labels.csv", index=False)



# Inference function for the evaluation API
def predict(sequence, demographics) -> str:
    sequence = sequence.to_pandas()
    demographics = demographics.to_pandas()
    df = pd.merge(sequence, demographics, on='subject', how='left')

    df.fillna(method='ffill', inplace=True)
    x = df[FEATURES].values.astype(np.float32)
    if x.shape[0] < 64:
        pad_width = 64 - x.shape[0]
        x = np.pad(x, ((0, pad_width), (0, 0)), mode='edge')
    else:
        x = x[:64]
    x = np.expand_dims(x, axis=0)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = np.argmax(probs)
    return label_encoder.inverse_transform([pred_idx])[0]

# Launch evaluation server
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

