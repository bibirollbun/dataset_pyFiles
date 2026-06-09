import pandas as pd
import librosa
import os
import numpy as np
metadata = pd.read_csv('/kaggle/input/birdclef-2023/train_metadata.csv')

metadata.head()


# =========================
# Count samples per class
# =========================
print("✅ Sample counts per class:")
print(metadata["primary_label"].value_counts())

# =========================
# Optional: define class names manually
# =========================
# You can get the list of all unique classes
class_names = metadata["primary_label"].unique().tolist()
print("Class Names:", class_names)



import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio
import os

# Path to a sample audio file from BirdCLEF-2023
audio_dir = "/kaggle/input/birdclef-2023/train_audio"
sample_file = "abethr1/XC128013.ogg"  # example file
audio_file_path = os.path.join(audio_dir, sample_file)

# 1️⃣ Load audio
audio_data, sample_rate = librosa.load(audio_file_path, sr=None)

# 2️⃣ Listen to audio
Audio(audio_file_path)






# 3️⃣ Optional: plot waveform
plt.figure(figsize=(12, 4))
plt.plot(audio_data)
plt.title("Waveform of " + sample_file)
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.show()


import os
import numpy as np
import pandas as pd
import librosa
from concurrent.futures import ThreadPoolExecutor

# Load BirdCLEF-2023 metadata
meta = pd.read_csv("/kaggle/input/birdclef-2023/train_metadata.csv")

def process_audio_file(file_path, sample_rate=22050, duration=5):
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs = np.mean(mfccs.T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_audio_files(dataframe, audio_dir="/kaggle/input/birdclef-2023/train_audio", sample_rate=22050, duration=5):
    audio_data = []
    labels = []

    with ThreadPoolExecutor() as executor:
        futures = []
        for _, row in dataframe.iterrows():
            file_path = os.path.join(audio_dir, row['filename'])
            futures.append(executor.submit(process_audio_file, file_path, sample_rate, duration))
            labels.append(row['primary_label'])

        for future in futures:
            result = future.result()
            if result is not None:
                audio_data.append(result)

    return np.array(audio_data), np.array(labels)

# Load first 200 files for a quick test
X, y = load_audio_files(meta.head(200))
print("X shape:", X.shape, "y shape:", y.shape)

print(X[1])
print(y[1])


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)  # convert bird names to integer labels
num_classes = len(np.unique(y_encoded))  # this will replace 10



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))



from tensorflow.keras import layers, models

input_shape = (X_train.shape[1], X_train.shape[2])  
model = models.Sequential([
    layers.GRU(64, input_shape=input_shape, return_sequences=True),
    layers.GRU(32),
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # <- dynamically set
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()



history = model.fit(X_train, y_train, epochs=175, batch_size=32, validation_split=0.2)



from sklearn.metrics import classification_report
import numpy as np

target_names = le.classes_  # This gives the bird species codes
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = y_test

taxonomy = pd.read_csv("/kaggle/input/birdclef-2023/eBird_Taxonomy_v2021.csv")
tax_map = dict(zip(taxonomy["SPECIES_CODE"], taxonomy["PRIMARY_COM_NAME"]))

# Convert target_names codes to common names
target_names_common = [tax_map.get(code, code) for code in target_names]
report = classification_report(actual_classes, predicted_classes, target_names=target_names_common)
print(report)


import librosa
import numpy as np

def predict_bird(file_path, model, le, tax_map, duration=5, n_mfcc=13):
    """
    Predicts the bird species from an audio file using the trained GRU model.

    Args:
        file_path (str): Path to the audio file.
        model (keras.Model): Trained GRU model.
        le (LabelEncoder): Label encoder fitted on training labels.
        tax_map (dict): Mapping from species code to common name.
        duration (float): Seconds of audio to load (default 5s).
        n_mfcc (int): Number of MFCCs to extract (default 13).
    
    Returns:
        tuple: (predicted_code, predicted_name)
    """
    try:
        # Load audio
        y_audio, sr = librosa.load(file_path, sr=None, duration=duration)
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        # Reshape for GRU input (1 sample, 1 time step, features)
        mfcc_input = mfcc_mean.reshape(1, 1, n_mfcc)
        
        # Predict probabilities
        probs = model.predict(mfcc_input, verbose=0)
        pred_index = np.argmax(probs, axis=1)[0]
        
        # Get species code and name
        pred_code = le.inverse_transform([pred_index])[0]
        pred_name = tax_map.get(pred_code, "Unknown")
        
        print(f"✅ Predicted Code: {pred_code}")
        print(f"✅ Predicted Name: {pred_name}")
        return pred_code, pred_name
    except Exception as e:
        print(f"Error predicting {file_path}: {e}")
        return None, None



audio_file = "/kaggle/input/birdclef-2023/train_audio/abethr1/XC128013.ogg"
predict_bird(audio_file, model, le, tax_map)


predictions = model.predict(X_test)

predicted_classes = np.argmax(predictions, axis=1)

actual_classes = y_test

for i in range(10):
    pred_name = tax_map.get(le.inverse_transform([predicted_classes[i]])[0], "Unknown")
    actual_name = tax_map.get(le.inverse_transform([actual_classes[i]])[0], "Unknown")
    print(f'Predicted: {pred_name}, Actual: {actual_name}')



import matplotlib.pyplot as plt

# Plot training & validation accuracy and loss
plt.figure(figsize=(14, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], marker='o', label='Train Accuracy')
plt.plot(history.history['val_accuracy'], marker='o', label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], marker='o', label='Train Loss')
plt.plot(history.history['val_loss'], marker='o', label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()



from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Use your encoded classes and mapped bird names
actual_classes = y_test  # already encoded
predictions = model.predict(X_test)  # softmax outputs

# Get mapped bird names from taxonomy (optional)
taxonomy = pd.read_csv("/kaggle/input/birdclef-2023/eBird_Taxonomy_v2021.csv")
tax_map = dict(zip(taxonomy["SPECIES_CODE"], taxonomy["PRIMARY_COM_NAME"]))
class_names = [tax_map.get(code, code) for code in le.classes_]  # le from label encoding

precision = {}
recall = {}
thresholds = {}

# Compute precision-recall curve for each bird class
for i, class_name in enumerate(class_names):
    precision[i], recall[i], thresholds[i] = precision_recall_curve(
        (actual_classes == i).astype(int), predictions[:, i]
    )

# Plot all curves
plt.figure(figsize=(12, 8))
for i, class_name in enumerate(class_names):
    plt.plot(recall[i], precision[i], label=class_name)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve per Bird Species')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()



from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# True labels and predicted classes
true_labels = y_test
predicted_classes = np.argmax(model.predict(X_test), axis=1)

# Map label indices to bird names
taxonomy = pd.read_csv("/kaggle/input/birdclef-2023/eBird_Taxonomy_v2021.csv")
tax_map = dict(zip(taxonomy["SPECIES_CODE"], taxonomy["PRIMARY_COM_NAME"]))
class_names = [tax_map.get(code, code) for code in le.classes_]  # le = LabelEncoder

# Compute confusion matrix
cm = confusion_matrix(true_labels, predicted_classes)

# Plot
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix of Bird Species Predictions')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()


