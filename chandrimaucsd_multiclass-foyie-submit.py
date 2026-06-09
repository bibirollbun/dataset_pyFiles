import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from ast import literal_eval

# --- Config ---
SAMPLE_RATE = 32000
N_FFT = 1024
HOP_LENGTH = 512
MAX_FREQ = 16000
BIN_SIZE = 5
NUM_BINS = MAX_FREQ // BIN_SIZE
MAX_FILES_PER_SPECIES = 100  # Limit for each species

# --- Target Species List ---
TARGET_SPECIES = ['grekis', 'compau', 'trokin']  # Extend this list as needed
AUDIO_DIR = Path("/kaggle/input/birdclef-2025/train_audio")

# --- Load Metadata ---
df = pd.read_csv("/kaggle/input/birdclef-2025/train.csv")

# --- Parse secondary labels safely ---
def parse_labels(s):
    try:
        return literal_eval(s) if pd.notna(s) else []
    except:
        return []

df["secondary_labels"] = df["secondary_labels"].apply(parse_labels)

# --- Feature Extraction Function ---
def improved_audio_to_binary_vector(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    y_harmonic, _ = librosa.effects.hpss(y)
    S = np.abs(librosa.stft(y_harmonic, n_fft=N_FFT, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    freq_mask = freqs <= MAX_FREQ
    S = S[freq_mask, :]
    freqs = freqs[freq_mask]

    energy_time_avg = np.mean(S, axis=1)
    log_energy = librosa.amplitude_to_db(energy_time_avg, ref=np.max)

    thresholds = np.zeros_like(log_energy)
    for fmin, fmax in [(0, 2000), (2000, 6000), (6000, MAX_FREQ)]:
        band_mask = (freqs >= fmin) & (freqs < fmax)
        if np.sum(band_mask) == 0:
            continue
        band_energy = log_energy[band_mask]
        q75 = np.percentile(band_energy, 75)
        q25 = np.percentile(band_energy, 25)
        thresholds[band_mask] = q75 + 0.5 * (q75 - q25)

    binary_vec = np.zeros(NUM_BINS, dtype=int)
    for i, f in enumerate(freqs):
        bin_idx = int(f // BIN_SIZE)
        if log_energy[i] > thresholds[i]:
            if np.sum(S[i, :] > thresholds[i]) >= 3:
                binary_vec[bin_idx] = 1

    return binary_vec



from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# # --- Assuming you have these functions and variables defined ---
# SAMPLE_RATE = 32000  # Example sample rate
# NUM_BINS = 1600  # Number of frequency bins (adjust if necessary)
# num_positive = 100  # Number of positive samples (can be dynamically set)
# target_species = ['grekis', 'compau', 'nocowl']  # List of target species
# species_code = "grekis"  # Example species for processing negative samples

# --- Load Positive Data ---
df_positive = pd.read_json("/kaggle/input/data-foyie/multi_species_metadata_vectors.json", lines=True)

# --- Load Negative Data ---
df_negative = pd.read_json("/kaggle/input/data-foyie/neg_multi_species_metadata_vectors.json", lines=True)

# --- Combine Positive and Negative Data ---
df_all = pd.concat([df_positive, df_negative], ignore_index=True)

# --- Prepare Feature (X) and Label (y) ---
X = np.array([np.array(vec) for vec in df_all['vector']])
y = []

# Label encoding: 0 for negative, 1, 2, 3 for each species
label_mapping = {"negative": 0, "grekis": 1, "compau": 2, "trokin": 3}

# Add species labels to the target vector
for idx, row in df_all.iterrows():
    if row['species'] == 'negative':
        y.append(0)
    elif row['species'] == 'grekis':
        y.append(1)
    elif row['species'] == 'compau':
        y.append(2)
    elif row['species'] == 'trokin':
        y.append(3)

y = np.array(y)

# --- Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train a Random Forest Classifier ---
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# classifier=LogisticRegression()
classifier.fit(X_train, y_train)

# --- Evaluate the Model ---
y_pred = classifier.predict(X_test)

# --- Display Classification Report ---
print(classification_report(y_test, y_pred, target_names=['negative', 'grekis', 'compau', 'nocowl']))

# --- Save the Model for Later Use ---
import joblib
joblib.dump(classifier, "bird_species_classifier.pkl")
print("Model saved as 'bird_species_classifier.pkl'")




import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import load



# # --- Feature Extraction Function (Same as Training) ---
# def extract_binary_features_from_chunk(chunk, percentile=76.7):
#     S = librosa.stft(chunk, n_fft=1024, hop_length=512)
#     S_power = np.abs(S)**2
#     S_db = librosa.power_to_db(S_power, ref=np.max)
#     freqs = librosa.fft_frequencies(sr=32000, n_fft=1024)
#     bin_edges = np.linspace(0, 16000, 3201)
#     binary_vector = np.zeros(3200, dtype=int)
#     energy_per_freq = S_db.max(axis=1)
#     adaptive_threshold = np.percentile(energy_per_freq, percentile)

#     for j in range(3200):
#         f_start = bin_edges[j]
#         f_end = bin_edges[j + 1]
#         bin_mask = (freqs >= f_start) & (freqs < f_end)
#         if np.any(bin_mask):
#             max_energy = np.max(energy_per_freq[bin_mask])
#             if max_energy > adaptive_threshold:
#                 binary_vector[j] = 1
#     return binary_vector
    
# # --- Generate Predictions ---
# AUDIO_BASE_TEST = '/kaggle/input/birdclef-2025/test_soundscapes/'
# class_labels = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))

# print("\nğŸ”µ Predicting on test soundscapes...")
# submission_rows = []

# test_soundscapes = [os.path.join(AUDIO_BASE_TEST, f) 
#                     for f in sorted(os.listdir(AUDIO_BASE_TEST)) 
#                     if f.endswith('.ogg')]

# for soundscape_path in tqdm(test_soundscapes, desc="Processing Soundscapes"):
#     try:
#         sig, sr = librosa.load(soundscape_path, sr=32000)
#     except Exception as e:
#         print(f"â�Œ Error reading {soundscape_path}: {e}")
#         continue

#     samples_per_chunk = 32000 * 5  # 5-second chunks

#     for i in range(0, len(sig), samples_per_chunk):
#         chunk = sig[i:i+samples_per_chunk]
#         if len(chunk) < samples_per_chunk:
#             continue  # Skip incomplete chunks

#         try:
#             # Step 1: Extract features
#             vec = extract_binary_features_from_chunk(chunk, percentile=76.7)
            
#             # Step 2: Scale features (critical for XGBoost!)
#             vec_scaled = scaler.transform([vec])  # Use the same scaler as training
            
#             # Step 3: Predict probability for grekis
#             prob = classifier.predict_proba(vec_scaled)[0][1]  # Use XGBoost
            
#             # Step 4: Build submission row
#             row_id = f"{os.path.splitext(os.path.basename(soundscape_path))[0]}_{(i//samples_per_chunk +1)*5}"
#             row = [row_id] + [0.001] * len(class_labels)  # Default 0.001 for non-target
#             row[class_labels.index('grekis') + 1] = prob  # Set grekis probability
            
#             submission_rows.append(row)
#         except Exception as e:
#             print(f"â�Œ Error processing {soundscape_path} chunk {i}: {e}")

# # Create submission file
# submission_df = pd.DataFrame(submission_rows, columns=['row_id'] + class_labels)
# submission_df.to_csv('submission.csv', index=False)
# print("âœ… Final submission.csv created!")

# # Preview
# submission_df.head()


# import os
# import pandas as pd
# import numpy as np
# import librosa
# import joblib
# from pathlib import Path

# # --- Config ---
# SAMPLE_RATE = 32000
# N_FFT = 1024
# HOP_LENGTH = 512
# MAX_FREQ = 16000
# BIN_SIZE = 5
# NUM_BINS = MAX_FREQ // BIN_SIZE
# TARGET_SPECIES = ['grekis', 'compau', 'trokin']  # Example

# # --- Load Model ---
# # classifier = joblib.load("bird_species_classifier.pkl")

# # --- Feature Extraction Function ---
# def improved_audio_to_binary_vector(path):
#     y, sr = librosa.load(path, sr=SAMPLE_RATE)
#     y_harmonic, _ = librosa.effects.hpss(y)
#     S = np.abs(librosa.stft(y_harmonic, n_fft=N_FFT, hop_length=HOP_LENGTH))
#     freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
#     freq_mask = freqs <= MAX_FREQ
#     S = S[freq_mask, :]
#     freqs = freqs[freq_mask]

#     energy_time_avg = np.mean(S, axis=1)
#     log_energy = librosa.amplitude_to_db(energy_time_avg, ref=np.max)

#     thresholds = np.zeros_like(log_energy)
#     for fmin, fmax in [(0, 2000), (2000, 6000), (6000, MAX_FREQ)]:
#         band_mask = (freqs >= fmin) & (freqs < fmax)
#         if np.sum(band_mask) == 0:
#             continue
#         band_energy = log_energy[band_mask]
#         q75 = np.percentile(band_energy, 75)
#         q25 = np.percentile(band_energy, 25)
#         thresholds[band_mask] = q75 + 0.5 * (q75 - q25)

#     binary_vec = np.zeros(NUM_BINS, dtype=int)
#     for i, f in enumerate(freqs):
#         bin_idx = int(f // BIN_SIZE)
#         if log_energy[i] > thresholds[i]:
#             if np.sum(S[i, :] > thresholds[i]) >= 3:
#                 binary_vec[bin_idx] = 1

#     return binary_vec

# # --- Prediction on Test Soundscapes ---
# test_dir = Path("/kaggle/input/birdclef-2025/test_soundscapes")
# submission_rows = []

# for filename in os.listdir(test_dir):
#     if not filename.endswith(".ogg"):
#         continue

#     filepath = test_dir / filename
#     file_id = filename.replace(".ogg", "")

#     try:
#         y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
#         for start in range(0, 60, 5):
#             end = start + 5
#             y_clip = y[start * sr:end * sr]

#             if len(y_clip) < sr * 5:
#                 y_clip = np.pad(y_clip, (0, sr * 5 - len(y_clip)))

#             clip_path = f"/tmp/{file_id}_{end}.ogg"
#             librosa.output.write_wav(clip_path, y_clip, sr)

#             vec = improved_audio_to_binary_vector(clip_path).reshape(1, -1)
#             probs = classifier.predict_proba(vec)[0]

#             row_id = f"{file_id}_{end}"
#             row = {"row_id": row_id}

#             for i, species in enumerate(TARGET_SPECIES):
#                 row[species] = probs[i + 1] if len(probs) > i + 1 else 0.0

#             submission_rows.append(row)

#     except Exception as e:
#         print(f"â�Œ Error processing {filename}: {e}")

# # --- Create Submission DataFrame ---
# submission_df = pd.DataFrame(submission_rows)

# # Ensure all required columns are present
# expected_cols = ["row_id"] + TARGET_SPECIES
# for col in expected_cols:
#     if col not in submission_df.columns:
#         submission_df[col] = 0.0

# # submission_df = submission_df[expected_cols]

# # # Check if DataFrame is empty
# # if submission_df.empty:
# #     raise ValueError("â�Œ No predictions were added. Check test files and processing.")

# # Create submission file
# # submission_df = pd.DataFrame(submission_rows, columns=['row_id'] + class_labels)
# submission_df.to_csv('submission.csv', index=False)
# print("âœ… Final submission.csv created!")

# # Preview
# submission_df.head()
# # # --- Save Submission ---
# # submission_df.to_csv("submission.csv", index=False)
# # print("âœ… Submission file saved: submission.csv")



import os
import librosa
import numpy as np
import pandas as pd
import joblib

# --- Paths ---
AUDIO_BASE_TEST = '/kaggle/input/birdclef-2025/test_soundscapes/'
TRAIN_AUDIO_DIR = '/kaggle/input/birdclef-2025/train_audio/'

# --- Load trained model ---
# clf_rf = joblib.load("/kaggle/input/my-models/bird_species_classifier.pkl")  # update if needed

# --- Target species as per your training ---
target_species = ['grekis', 'compau', 'trokin']  # class 1, 2, 3 in your model
label_mapping = {0: "negative", 1: "grekis", 2: "compau", 3: "trokin"}

# --- All 2025 class labels from train_audio folder (required for submission) ---
class_labels = sorted(os.listdir(TRAIN_AUDIO_DIR))  # ~206 total classes

# --- Prepare submission rows ---
submission_rows = []

# --- Process test soundscapes ---
test_files = sorted(f for f in os.listdir(AUDIO_BASE_TEST) if f.endswith(".ogg"))

for filename in test_files:
    path = os.path.join(AUDIO_BASE_TEST, filename)
    try:
        y, sr = librosa.load(path, sr=32000)
        samples_per_chunk = sr * 5

        for i in range(0, len(y), samples_per_chunk):
            chunk = y[i:i+samples_per_chunk]
            if len(chunk) < samples_per_chunk:
                continue

            # Feature extraction
            S = librosa.stft(chunk, n_fft=1024, hop_length=512)
            S_power = np.abs(S)**2
            S_db = librosa.power_to_db(S_power, ref=np.max)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)

            bin_edges = np.linspace(0, 16000, 3201)
            binary_vector = np.zeros(3200, dtype=int)
            energy_per_freq = S_db.max(axis=1)
            adaptive_threshold = np.percentile(energy_per_freq, 76.7)

            for j in range(3200):
                f_start = bin_edges[j]
                f_end = bin_edges[j+1]
                bin_mask = (freqs >= f_start) & (freqs < f_end)
                if np.any(bin_mask):
                    max_energy = np.max(energy_per_freq[bin_mask])
                    if max_energy > adaptive_threshold:
                        binary_vector[j] = 1

            # Predict multi-class probabilities
            probs = classifier.predict_proba(binary_vector.reshape(1, -1))[0]  # shape: (4,)

            # Row ID format
            chunk_end = (i // samples_per_chunk + 1) * 5
            row_id = f"{filename.replace('.ogg', '')}_{chunk_end}"

            # Build submission row
            row = [row_id]
            for label in class_labels:
                if label in target_species:
                    class_index = list(label_mapping.keys())[list(label_mapping.values()).index(label)]
                    row.append(probs[class_index])
                else:
                    row.append(0.001)

            submission_rows.append(row)

    except Exception as e:
        print(f"â�Œ Error processing {filename}: {e}")

# --- Create submission DataFrame ---
submission_df = pd.DataFrame(submission_rows, columns=["row_id"] + class_labels)

# --- Save submission file ---
submission_df.to_csv("submission.csv", index=False)
print("âœ… Saved final submission as submission.csv")
submission_df.head()







