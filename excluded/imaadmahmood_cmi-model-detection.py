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


!pip install polars -q
!pip install scikit-learn -q


import os
import sys
sys.path.append("/kaggle/input/cmi-detect-behavior-with-sensor-data")

import numpy as np
import pandas as pd
import polars as pl
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, f1_score

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D, GlobalAveragePooling1D, Masking, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from kaggle_evaluation.cmi_inference_server import CMIInferenceServer


train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
test_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
test_demo = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")
train_demo = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")

display(train_df.head())
print()
display(test_df.head())

# Check Column Names
print("\nColumns in Train_df is:", train_df.columns)
print("\nColumns in Test_df is:", test_df.columns)
print("\nColumns in Test_demo is:", test_demo.columns)
print("\nColumns in Train_demo is:", train_demo.columns)

# Check Null Values
print("\nThe Null Values in Test_df is:", test_df.isnull().sum())
print("\nThe Null Values in Train_df is:", train_df.isnull().sum())
print("\nThe Null Values in Test_demo is:", test_demo.isnull().sum())
print("\nThe Null Values in Train_Demo is:", train_demo.isnull().sum())


display(test_demo.head())
display(train_demo.head())


def extract_features(df):
    df = df.copy()
    
    # Basic features
    df["acc_mag"] = np.sqrt(df.acc_x**2 + df.acc_y**2 + df.acc_z**2)
    df["rot_mag"] = np.sqrt(df.rot_x**2 + df.rot_y**2 + df.rot_z**2 + df.rot_w**2)
    df["acc_jerk"] = df["acc_mag"].diff().fillna(0)
    df["rot_jerk"] = df["rot_mag"].diff().fillna(0)
    
    # Rolling features (offline only)
    for col in ["acc_x", "acc_y", "acc_z", "rot_x", "rot_y", "rot_z"]:
        df[f"{col}_mean5"] = df[col].rolling(window=5, min_periods=1).mean()
        df[f"{col}_std5"] = df[col].rolling(window=5, min_periods=1).std().fillna(0)

    return df



train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
train_df = train_df[~train_df["gesture"].isna()]

label_df = train_df[["sequence_id", "gesture"]].drop_duplicates("sequence_id")
le = LabelEncoder()
label_df["gesture_enc"] = le.fit_transform(label_df["gesture"])
joblib.dump(le, "label_encoder.pkl")

# Define features
basic_cols = ["acc_x", "acc_y", "acc_z", "rot_x", "rot_y", "rot_z", "rot_w",
              "acc_mag", "rot_mag", "acc_jerk", "rot_jerk"]
roll_cols = [f"{col}_mean5" for col in basic_cols[:6]] + [f"{col}_std5" for col in basic_cols[:6]]
feature_cols = basic_cols + roll_cols
np.save("feature_cols.npy", feature_cols)

X_list, y_list = [], []
grouped = train_df.groupby("sequence_id")

for seq_id, group in grouped:
    group = extract_features(group)
    group = group[feature_cols].ffill().bfill().fillna(0)
    X_list.append(group.values.astype("float32"))
    y = label_df[label_df.sequence_id == seq_id]["gesture_enc"].values[0]
    y_list.append(y)

# Padding and scaling
pad_len = max(len(x) for x in X_list)
np.save("pad_len.npy", pad_len)

scaler = StandardScaler().fit(np.vstack(X_list))
joblib.dump(scaler, "scaler.pkl")

X_scaled = [scaler.transform(x) for x in X_list]
X = pad_sequences(X_scaled, maxlen=pad_len, padding="post", dtype="float32")
y = np.array(y_list)



def create_model(input_shape, n_classes):
    inp = Input(shape=input_shape)
    x = Masking(mask_value=0.0)(inp)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Conv1D(64, kernel_size=3, activation="relu", padding="same")(x)
    x = Conv1D(64, kernel_size=3, activation="relu", padding="same")(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(n_classes, activation="softmax")(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model



kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = dict(enumerate(class_weights))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
    print(f"\nTraining Fold {fold}...")
    model = create_model((pad_len, len(feature_cols)), len(le.classes_))
    
    es = EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")
    lr = ReduceLROnPlateau(patience=2, factor=0.5, verbose=1)
    
    model.fit(
        X[tr_idx], y[tr_idx],
        validation_data=(X[va_idx], y[va_idx]),
        epochs=25,
        batch_size=64,
        class_weight=class_weights,
        callbacks=[es, lr],
        verbose=2
    )
    
    # Evaluate and save model
    val_pred = np.argmax(model.predict(X[va_idx]), axis=1)
    f1 = f1_score(y[va_idx], val_pred, average="macro")
    print(f"Fold {fold} F1 Macro: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y[va_idx], val_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Fold {fold} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    model.save(f"model_fold{fold}.keras")



scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")
feature_cols = np.load("feature_cols.npy", allow_pickle=True)
pad_len = int(np.load("pad_len.npy"))
models = [load_model(f"model_fold{f}.keras") for f in range(5)]

def predict(seq: pl.DataFrame, dem: pl.DataFrame) -> str:
    df_seq = extract_features(seq.to_pandas())
    df_seq = df_seq[feature_cols].ffill().bfill().fillna(0)
    mat = scaler.transform(df_seq.values.astype("float32"))
    x = pad_sequences([mat], maxlen=pad_len, padding="post", dtype="float32")
    pred = np.mean([m.predict(x, verbose=0)[0] for m in models], axis=0)
    return le.inverse_transform([pred.argmax()])[0]

inference_server = CMIInferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway((
        "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv",
        "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv"
    ))



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# === 1. Feature Importance via Permutation ===
def estimate_feature_importance(model, X_sample, y_sample, feature_names, n_repeats=1):
    base_pred = np.argmax(model.predict(X_sample, verbose=0), axis=1)
    base_acc = accuracy_score(y_sample, base_pred)
    importance_scores = []

    for i in range(X_sample.shape[2]):
        drop_accs = []
        for _ in range(n_repeats):
            X_permuted = X_sample.copy()
            np.random.shuffle(X_permuted[:, :, i])
            perm_pred = np.argmax(model.predict(X_permuted, verbose=0), axis=1)
            perm_acc = accuracy_score(y_sample, perm_pred)
            drop_accs.append(base_acc - perm_acc)
        importance_scores.append(np.mean(drop_accs))
    
    return importance_scores

# Use model from fold 0 and its validation split
fold_to_visualize = 0
model = load_model(f"model_fold{fold_to_visualize}.keras")
va_idx = list(kf.split(X, y))[fold_to_visualize][1]
X_sample, y_sample = X[va_idx], y[va_idx]

importance_scores = estimate_feature_importance(model, X_sample, y_sample, feature_cols, n_repeats=3)

# === Feature Importance Barplot ===
plt.figure(figsize=(12, 7))
sns.barplot(x=importance_scores, y=feature_cols, palette="mako")
plt.title("ğŸ“Š Feature Importance via Permutation (Fold 0)", fontsize=16)
plt.xlabel("Drop in Accuracy (Importance)", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# === 2. Confusion Matrix ===
y_true = y_sample
y_pred = np.argmax(model.predict(X_sample, verbose=0), axis=1)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("ğŸ”� Confusion Matrix (Fold 0)", fontsize=14)
plt.tight_layout()
plt.show()

# === 3. Class Probability Heatmap for a Few Samples ===
num_examples = 10
probs = model.predict(X_sample[:num_examples], verbose=0)

plt.figure(figsize=(12, 6))
sns.heatmap(probs.T, cmap="viridis", annot=True, fmt=".2f",
            xticklabels=[f"Sample {i}" for i in range(num_examples)],
            yticklabels=le.classes_)
plt.title("ğŸ”¥ Class Probabilities per Sample", fontsize=14)
plt.xlabel("Sample")
plt.ylabel("Class")
plt.tight_layout()
plt.show()

# === 4. Feature-wise Activation Heatmap for a Sequence ===
seq_index = 0  # any index in X_sample
seq_data = X_sample[seq_index]  # shape: (pad_len, n_features)

plt.figure(figsize=(14, 6))
sns.heatmap(seq_data.T, cmap="coolwarm", xticklabels=20, yticklabels=feature_cols)
plt.title(f"ğŸ�¯ Feature Activation over Time (Sequence {seq_index})", fontsize=14)
plt.xlabel("Timestep")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


