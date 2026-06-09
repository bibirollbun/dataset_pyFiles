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

FEATURES = IMU_FEATURES + THERMO_FEATURES + DEMO_FEATURES  + TOF_FEATURES 
print(f"We have {len(FEATURES)} features")


# Prepare sequences
sequence_ids = train_df['sequence_id'].unique()
X, y, y_binary = [], [], []

def is_bfrb(label_id):
    label_name = label_encoder.inverse_transform([label_id])[0]
    return int(label_name != 'non_target')

expected_feature_dim = len(FEATURES)
print(f"Expecting feature size: {expected_feature_dim}")

for seq_id in sequence_ids:
    df = train_df[train_df['sequence_id'] == seq_id]

    #print(seq_id)
    
    # Skip if any FEATURE is missing
    if df[FEATURES].isnull().values.any():
        continue

    x = df[FEATURES].values.astype(np.float32)
    
    # Pad or truncate to fixed length
    if x.shape[0] < 64:
        pad_width = 64 - x.shape[0]
        x = np.pad(x, ((0, pad_width), (0, 0)), mode='edge')
    else:
        x = x[:64]
    
    # Final shape check
    if x.shape != (64, expected_feature_dim):
        print(f"Skipping sequence {seq_id} with bad shape: {x.shape}")
        continue

    # Append class and binary labels
    X.append(x)
    label = targets.loc[targets['sequence_id'] == seq_id, 'gesture_enc'].values[0]
    y.append(label)
    y_binary.append(is_bfrb(label))

X = np.stack(X)
y = np.array(y)
y_binary = np.array(y_binary)
print(f"Final shape of X: {X.shape}")



# X_train, X_val, y_train_class, y_val_class, y_train_bin, y_val_bin = train_test_split(
#     X, y, y_binary, test_size=0.1, stratify=y, random_state=42
# )
# BATCH_SIZE = 64 * strategy.num_replicas_in_sync



# Split features into time-series and demographic parts
X_time = X[:, :, :len(IMU_FEATURES + THERMO_FEATURES + TOF_FEATURES)]
X_demo = X[:, 0, -len(DEMO_FEATURES):]  # Take demo features from first timestep

# Train/validation split
X_train_time, X_val_time, X_train_demo, X_val_demo, y_train_class, y_val_class, y_train_bin, y_val_bin = train_test_split(
    X_time, X_demo, y, y_binary, test_size=0.1, stratify=y, random_state=42
)

# Define datasets
BATCH_SIZE = 64 * strategy.num_replicas_in_sync
train_ds = tf.data.Dataset.from_tensor_slices((
    {
        "time_series": X_train_time,
        "demographics": X_train_demo
    },
    {
        "class_output": y_train_class,
        "binary_output": y_train_bin
    }
)).shuffle(1024).batch(BATCH_SIZE).prefetch(AUTO)

val_ds = tf.data.Dataset.from_tensor_slices((
    {
        "time_series": X_val_time,
        "demographics": X_val_demo
    },
    {
        "class_output": y_val_class,
        "binary_output": y_val_bin
    }
)).batch(BATCH_SIZE).prefetch(AUTO)



def build_bfrb_model(architecture: str, input_shapes: dict, num_classes: int) -> tf.keras.Model:
    def se_block(input_tensor, ratio=8):
        channel_axis = -1
        filters = input_tensor.shape[channel_axis]
        se_shape = (1, filters)

        se = tf.keras.layers.GlobalAveragePooling1D()(input_tensor)
        se = tf.keras.layers.Reshape(se_shape)(se)
        se = tf.keras.layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = tf.keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
        return tf.keras.layers.Multiply()([input_tensor, se])

    def residual_se_cnn_block(x, filters, kernel_size):
        shortcut = x
        x = tf.keras.layers.Conv1D(filters, kernel_size, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv1D(filters, kernel_size, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = se_block(x)
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.ReLU()(x)
        return x

    def attention_block(inputs):
        query = tf.keras.layers.Dense(64)(inputs)
        key = tf.keras.layers.Dense(64)(inputs)
        value = tf.keras.layers.Dense(64)(inputs)
        attention = tf.keras.layers.Attention()([query, key, value])
        return tf.keras.layers.GlobalAveragePooling1D()(attention)

    time_input = tf.keras.Input(shape=input_shapes['time_series'], name="time_series")
    demo_input = tf.keras.Input(shape=input_shapes['demographics'], name="demographics")

    # --- Time-series branch ---
    x = time_input
    if architecture == "A1":
        x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv1D(128, 5, padding='same', activation='relu')(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif architecture == "A2":
        x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv1D(128, 5, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
    elif architecture == "A3":
        x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv1D(128, 5, padding='same', activation='relu')(x)
        x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
        x = attention_block(x)
    elif architecture == "A4":
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
        x = attention_block(x)
    elif architecture == "A6":
        x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv1D(128, 5, padding='same', activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
    elif architecture == "A7":
        x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv1D(128, 5, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
    elif architecture == "A8":
        x = tf.keras.layers.Conv1D(64, 5, padding='same')(x)
        x = residual_se_cnn_block(x, 64, 5)
        x = residual_se_cnn_block(x, 64, 5)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
        x = attention_block(x)
    else:
        raise ValueError(f"Unknown architecture ID: {architecture}")

    x = tf.keras.layers.Dense(128, activation='relu')(x)

    # --- Demographic branch ---
    d = tf.keras.layers.Dense(64, activation='relu')(demo_input)
    d = tf.keras.layers.Dense(32, activation='relu')(d)

    # --- Merge branches ---
    merged = tf.keras.layers.Concatenate()([x, d])
    merged = tf.keras.layers.Dense(128, activation='relu')(merged)

    # --- Heads ---
    class_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='class_output')(merged)
    binary_output = tf.keras.layers.Dense(1, activation='sigmoid', name='binary_output')(merged)

    model = tf.keras.Model(inputs=[time_input, demo_input], outputs=[class_output, binary_output])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={
            'class_output': tf.keras.losses.SparseCategoricalCrossentropy(),
            'binary_output': tf.keras.losses.BinaryCrossentropy(),
        },
        metrics={
            'class_output': 'accuracy',
            'binary_output': 'accuracy',
        }
    )
    return model

with strategy.scope():
    model = build_bfrb_model(
        architecture="A6",
        input_shapes={
            "time_series": (64, len(IMU_FEATURES + THERMO_FEATURES + TOF_FEATURES)),  # e.g. (64, 332)
            "demographics": (len(DEMO_FEATURES),)     # e.g. (7,)
        },
        num_classes=len(label_encoder.classes_)
    )



# Train

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_class_output_loss',
    patience=10,
    restore_best_weights=True, mode="min"
)

history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[early_stop])



import matplotlib.pyplot as plt

# Plot classification (gesture) loss and accuracy
plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['class_output_loss'], label='Train Gesture Loss')
plt.plot(history.history['val_class_output_loss'], label='Val Gesture Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Gesture Classification Loss')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['class_output_accuracy'], label='Train Gesture Acc')
plt.plot(history.history['val_class_output_accuracy'], label='Val Gesture Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Gesture Classification Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Plot binary head loss and accuracy
plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['binary_output_loss'], label='Train Binary Loss')
plt.plot(history.history['val_binary_output_loss'], label='Val Binary Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Binary Detection Loss')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['binary_output_accuracy'], label='Train Binary Acc')
plt.plot(history.history['val_binary_output_accuracy'], label='Val Binary Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Binary Detection Accuracy')
plt.legend()

plt.tight_layout()
plt.show()



# Evaluate model (multi-head)
val_preds = model.predict(val_ds)
gesture_preds = val_preds[0]  # class_output
binary_preds = val_preds[1]   # binary_output

y_val_pred = np.argmax(gesture_preds, axis=1)
binary_pred = (binary_preds > 0.5).astype(int).flatten()

from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

print("Validation Classification Report:")
print(classification_report(y_val_class, y_val_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_val_class, y_val_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

binary_true = np.isin(y_val_class, [label_encoder.transform([g])[0] for g in label_encoder.classes_ if g != 'non_target']).astype(int)
binary_pred = np.isin(y_val_pred, [label_encoder.transform([g])[0] for g in label_encoder.classes_ if g != 'non_target']).astype(int)

binary_f1 = f1_score(binary_true, binary_pred)
macro_f1 = f1_score(y_val_class, y_val_pred, average='macro')
final_score = 0.5 * (binary_f1 + macro_f1)

print(f"Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f}, Final Score: {final_score:.4f}")

print(f"Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f}, Final Score: {final_score:.4f}")


# Ground truth binary labels (1 = target, 0 = non-target)
binary_true = np.isin(y_val_class, [
    label_encoder.transform([g])[0] 
    for g in label_encoder.classes_ if g != 'non_target'
]).astype(int)

# Already computed earlier: binary_pred = (binary_preds > 0.5).astype(int).flatten()

# Binary classification report
print("\nBinary Classification Report:")
print(classification_report(
    binary_true, 
    binary_pred, 
    labels=[0, 1], 
    target_names=['non_target', 'target']
))


# Binary confusion matrix
binary_cm = confusion_matrix(binary_true, binary_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(binary_cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['non_target', 'target'], 
            yticklabels=['non_target', 'target'])
plt.title("Confusion Matrix - Binary Head")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()






from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

f1_per_class = f1_score(y_val_class, y_val_pred, average=None)
plt.figure(figsize=(12, 5))
plt.bar(label_encoder.classes_, f1_per_class)
plt.xticks(rotation=45, ha='right')
plt.ylabel("F1 Score")
plt.title("Per-Class F1 Scores")
plt.tight_layout()
plt.show()



binary_f1 = f1_score(binary_true, binary_pred)
macro_f1 = f1_score(y_val_class, y_val_pred, average='macro')
final_score = 0.5 * (binary_f1 + macro_f1)

print(f"Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f}, Final Score: {final_score:.4f}")



# Visualize gesture-wise F1 scores:

from sklearn.metrics import precision_recall_fscore_support

_, _, f1s, _ = precision_recall_fscore_support(y_val_class, y_val_pred, labels=range(len(label_encoder.classes_)))
plt.figure(figsize=(12, 4))
sns.barplot(x=label_encoder.classes_, y=f1s)
plt.xticks(rotation=45)
plt.title("Per-Class F1 Score")
plt.ylabel("F1 Score")
plt.xlabel("Gesture Class")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()



#model.save("./model")
#pd.Series(label_encoder.classes_).to_csv("gesture_labels.csv", index=False)



def predict(sequence, demographics) -> str:
    # Convert to pandas if needed
    sequence = sequence.to_pandas()
    demographics = demographics.to_pandas()
    
    # Merge and fill missing
    df = pd.merge(sequence, demographics, on='subject', how='left')
    df.fillna(method='ffill', inplace=True)

    # Extract features
    x_time = df[IMU_FEATURES + THERMO_FEATURES + TOF_FEATURES].values.astype(np.float32)
    x_demo = df[DEMO_FEATURES].iloc[0].values.astype(np.float32)  # take from first row

    # Pad/crop time-series
    if x_time.shape[0] < 64:
        pad_width = 64 - x_time.shape[0]
        x_time = np.pad(x_time, ((0, pad_width), (0, 0)), mode='edge')
    else:
        x_time = x_time[:64]

    # Add batch dimension
    x_time = np.expand_dims(x_time, axis=0)
    x_demo = np.expand_dims(x_demo, axis=0)

    # Predict
    class_probs, _ = model.predict({"time_series": x_time, "demographics": x_demo}, verbose=0)
    pred_idx = np.argmax(class_probs[0])
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

