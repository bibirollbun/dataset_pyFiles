# === Core Libraries ===
import numpy as np
import pandas as pd
import os
import kaggle_evaluation.cmi_inference_server

# === ML/DL Libraries ===
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# === Visualization ===
import matplotlib.pyplot as plt
import seaborn as sns


# Feature sets
sequence_features = ['sequence_counter']
accel_features = ['acc_x', 'acc_y', 'acc_z']
rot_features = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
thm_features = ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']
tof_features = [f"tof_{i}_v{j}" for i in range(1, 6) for j in range(64)]
demo_features = ['adult_child', 'age', 'sex', 'handedness', 'height_cm',
                 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']

# Label encoding
gesture_mapping = {
    "Cheek - pinch skin": 1,
    "Forehead - pull hairline": 2,
    "Neck - scratch": 3,
    "Neck - pinch skin": 4,
    "Eyelash - pull hair": 5,
    "Eyebrow - pull hair": 6,
    "Forehead - scratch": 7,
    "Above ear - pull hair": 8,
    "Non-Target": 0
}
inv_gesture_mapping = {v: k for k, v in gesture_mapping.items()}



def load_and_merge():
    train = pd.read_csv("../input/cmi-detect-behavior-with-sensor-data/train.csv")
    train = train[train.phase == "Gesture"]
    train_demo = pd.read_csv("../input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")
    return train.merge(train_demo, on="subject", how="left")

def encode_target(df):
    df["encoded_gesture"] = df["gesture"].map(gesture_mapping).fillna(0).astype(int)
    return df



def extract_time_series(df, group_col="sequence_id"):
    X_seq, X_accel, X_rot, X_thm, X_tof, y, demo_all = [], [], [], [], [], [], []

    grouped = df.groupby(group_col)
    for _, group in grouped:
        if len(group) < 21:
            continue
        group = group.tail(21)  # Fixed-length sequences

        X_seq.append(group[sequence_features].to_numpy())
        X_accel.append(group[accel_features].to_numpy())
        X_rot.append(group[rot_features].to_numpy())
        X_thm.append(group[thm_features].to_numpy())
        X_tof.append(group[tof_features].to_numpy())
        demo_all.append(group[demo_features].iloc[0].to_numpy())
        y.append(group["encoded_gesture"].iloc[0])

    return {
        "X_seq": np.array(X_seq),
        "X_accel": np.array(X_accel),
        "X_rot": np.array(X_rot),
        "X_thm": np.array(X_thm),
        "X_tof": np.array(X_tof),
        "X_demo": np.array(demo_all),
        "y": np.array(y)
    }



def build_model(input_shapes, demo_dim, num_classes=9):
    inputs = []
    lstm_outputs = []

    for name in ['X_seq', 'X_accel', 'X_rot', 'X_thm', 'X_tof']:
        inp = tf.keras.Input(shape=input_shapes[name], name=name)
        x = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(inp)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)
        inputs.append(inp)
        lstm_outputs.append(x)

    demo_input = tf.keras.Input(shape=(demo_dim,), name="X_demo")
    x_demo = tf.keras.layers.Dense(32, activation='relu')(demo_input)
    x_demo = tf.keras.layers.Dense(16, activation='relu')(x_demo)
    inputs.append(demo_input)

    merged = tf.keras.layers.Concatenate()(lstm_outputs + [x_demo])
    x = tf.keras.layers.Dense(64, activation='relu')(merged)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model



# Load and prepare data
df = load_and_merge()
df = encode_target(df)
data = extract_time_series(df)

# Split features
X = {k: data[k] for k in data if k != "y"}
y = data["y"]
X_train = {k: v[:int(0.8*len(y))] for k, v in X.items()}
X_val = {k: v[int(0.8*len(y)):] for k, v in X.items()}
y_train = y[:int(0.8*len(y))]
y_val = y[int(0.8*len(y)):]

# Build and train model
input_shapes = {k: X[k].shape[1:] for k in ['X_seq', 'X_accel', 'X_rot', 'X_thm', 'X_tof']}
demo_dim = X['X_demo'].shape[1]
model = build_model(input_shapes, demo_dim)
model.summary()

# Train
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32)



def plot_training_history(history):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Call it
plot_training_history(history)



def plot_classification_report(y_true, y_pred, class_names):
    print(classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Predict
y_pred = np.argmax(model.predict(X_val), axis=1)
plot_classification_report(y_val, y_pred, class_names=list(gesture_mapping.keys()))



model.save("hybrid_cnn_lstm_model.h5")


def predict(sequence, demographics) -> str:
    # âœ… Ensure pandas DataFrames
    if not isinstance(sequence, pd.DataFrame):
        sequence = sequence.to_pandas()
    if not isinstance(demographics, pd.DataFrame):
        demographics = demographics.to_pandas()

    # Merge demographics
    sequence = sequence.merge(demographics, on="subject", how="left")
    sequence = sequence.tail(21)

    # Feature extraction
    X_input = {
        'X_seq': sequence[sequence_features].to_numpy()[None, :],
        'X_accel': sequence[accel_features].to_numpy()[None, :],
        'X_rot': sequence[rot_features].to_numpy()[None, :],
        'X_thm': sequence[thm_features].to_numpy()[None, :],
        'X_tof': sequence[tof_features].to_numpy()[None, :],
        'X_demo': sequence[demo_features].iloc[0].to_numpy()[None, :]
    }

    # Load model and predict
    model = tf.keras.models.load_model("hybrid_cnn_lstm_model.h5")
    pred = np.argmax(model.predict(X_input), axis=1)[0]
    return "Text on phone" if pred == 0 else inv_gesture_mapping.get(pred, "Text on phone")





# Initialize the inference server using your predict function
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

# Automatically switch between local gateway and competition rerun
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )





