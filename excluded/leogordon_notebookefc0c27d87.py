# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

! pip install pandarallel

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/yamnet/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        pass

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


labels = {'brtpar1', '1139490', 'compau', 'chbant1', 'yehcar1', 'yecspi2', 'watjac1', 'grasal4', 'grbhaw1', 
          'yebfly1', 'neocor', '81930', 'spbwoo1', '64862', 'grepot1', 'ruther1', 'banana', 'whttro1', 
          '1462711', '42087', '66531', 'soulap1', 'amakin1', '41970', '65373', '714022', 'bafibi1', 'blcant4', 
          'rutjac1', 'plbwoo1', 'anhing', 'yehbla2', '21211', 'recwoo1', 'blbgra1', 'creoro1', 'shtfly1', 
          'amekes', 'blchaw1', '21116', '566513', 'bugtan', 'strcuc1', '1564122', '1462737', 'purgal2', 
          'socfly1', 'gohman1', 'gycwor1', 'bubwre1', 'blhpar1', '65336', 'solsan', '134933', '24292', 
          '42113', 'plukit1', 'savhaw1', 'sobtyr1', 'chfmac1', 'yebsee1', '66016', 'blbwre1', 'mastit1', 
          'smbani', 'whfant1', 'strfly1', 'roahaw', 'rumfly1', '476537', 'butsal1', 'bucmot3', 'colcha1', 
          'bobfly1', '67082', 'rebbla1', 'pavpig2', '1192948', 'whbman1', 'verfly', 'eardov1', 'norscr1', 
          'rinkin1', '67252', 'greibi1', 'greegr', 'cattyr', 'laufal1', 'trokin', 'grekis', 'crebob1', 
          'bubcur1', 'fotfly', 'palhor2', '476538', '24322', 'tropar', 'whwswa1', 'yercac1', '517119', 
          '24272', 'cocher1', 'labter1', 'bicwre1', 'compot1', 'olipic1', 'blcjay1', 'colara1', 'spepar1', 
          'cregua1', 'cargra1', '22976', 'plctan1', '715170', 'leagre', '22973', 'bkmtou1', 'yelori1', 
          'trsowl', 'strher', 'ragmac1', 'yeofly1', '548639', 'tbsfin1', '135045', '65344', 'bkcdon', 
          'stbwoo2', 'piepuf1', '868458', '963335', 'blctit1', 'saffin', 'rtlhum', 'royfly1', '66893', 
          'rutpuf1', 'linwoo1', 'wbwwre1', 'srwswa1', '126247', 'gretin1', 'grnkin', 'littin1', 'secfly1', 
          '41778', '528041', 'bbwduc', 'greani1', 'rubsee1', 'orcpar', 'rosspo1', 'yebela1', '47067', 
          'crcwoo1', '65349', 'snoegr', 'gybmar', 'thbeup1', '66578', 'turvul', 'rugdov', 'baymac', 
          'speowl1', 'cocwoo1', 'cotfly1', 'y00678', '65419', 'bobher1', '52884', '41663', '22333', 
          'piwtyr1', '21038', '787625', 'rufmot1', '65962', 'paltan1', '48124', '555142', '65547', 
          'crbtan1', '1194042', 'ywcpar', 'shghum1', 'cinbec1', 'thlsch3', '1346504', '555086', 'sahpar1', 
          'grysee1', 'blkvul', '523060', 'strowl1', 'whbant1', 'whmtyr1', '65448', 'ampkin1', 'whtdov', 
          'yectyr1', '42007', '46010', 'pirfly1', 'woosto', 'babwar', '50186'}
print(len(labels))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from IPython.display import Audio
from scipy.io import wavfile
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report
import scipy.signal
from tqdm import tqdm
flag = 0


def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=22000):
    if original_sample_rate != desired_sample_rate:
        desired_length = int(
            round(float(len(waveform))/original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform


def read_audio(filename):
    wav_data, sample_rate = sf.read(file=filename, dtype=np.int16)
    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
    return sample_rate, wav_data


import os
import pandas as pd

# Define the root directory containing the audio files
root_dir = '/kaggle/input/birdclef-2025/train_audio' # Adjust if your path is different

# List to store the data for the DataFrame
audio_data_list = []

# Walk through the directory structure
for dirname, _, filenames in os.walk(root_dir):
    # Skip the root directory itself if it doesn't contain class folders directly
    # (Adjust this condition if your structure is different)
    if dirname == root_dir:
        continue

    # Extract the class name (subdirectory name)
    # os.path.basename gets the last part of the directory path
    class_name = os.path.basename(dirname)

    # Iterate through files in the current directory
    for filename in filenames:
        # Construct the full path to the audio file
        full_path = os.path.join(dirname, filename)

        # Append the file path and its class to the list
        audio_data_list.append([full_path, class_name])
        # You can remove the print statement if you don't need it anymore
        # print(full_path) # Optional: print the path as it's processed

# Create the Pandas DataFrame
audio_dataframe = pd.DataFrame(audio_data_list, columns=["audio_path", "class"])

# Display the first few rows of the DataFrame (optional)
print(audio_dataframe.head())

# Display the shape of the DataFrame (optional)
print(f"\nDataFrame shape: {audio_dataframe.shape}")


df = audio_dataframe.iloc[0]

for i in audio_dataframe['class'].unique():
    df_sub = audio_dataframe[audio_dataframe['class'] == i]
    num_rows = 3
    # print(len(df_sub.index))
    if len(df_sub.index) < num_rows:
        num_rows = len(df_sub.index)
    # print(num_rows)
    df_sub = audio_dataframe[audio_dataframe['class'] == i].iloc[:num_rows]
    # print(df_sub)
    df = pd.concat([df,df_sub])
df['class']

# flag = 0


import pandas as pd
import concurrent.futures
from tqdm.auto import tqdm
import time # Just for mock function
import os # To get cpu count
if flag == 0:
    
    # Get the list of paths to process
    paths_to_process = df['audio_path'].tolist()

    # Determine number of workers (e.g., number of CPU cores)
    # Adjust based on memory constraints and task type (CPU vs I/O bound)
    num_workers = os.cpu_count()
    print(f"Using {num_workers} workers.")

    results = [None] * len(paths_to_process) # Preallocate results list

    print("Starting parallel processing with concurrent.futures...")
    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use executor.map to apply the function in parallel
        # Wrap executor.map with tqdm for a progress bar
        # executor.map preserves the order of the input iterable
        future_to_path = {executor.submit(read_audio, path): i for i, path in enumerate(paths_to_process)}

        for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(paths_to_process), desc="Reading audio files"):
            index = future_to_path[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                print(f'Path at index {index} generated an exception: {exc}')
                results[index] = None # Or some other error indicator
    flag = 1
    print("Finished parallel processing.")

    # Assign the results back to the DataFrame
    df['audio_data'] = results
    df.to_pickle('/kaggle/working/df.pk1')
    print(df.head())


import re

def audio_data_unfucker(fucked_tuple: tuple) -> list:
        assert(type(fucked_tuple[1]) != int)
        return fucked_tuple[1]


if flag == 1:
    pass
else:
    flag = 1
df = pd.read_pickle('/kaggle/working/df.pk1')
df = df.iloc[2:]
print(df.columns)
df['audio_data'] = df['audio_data'].apply(audio_data_unfucker)


df['audio_data']


audio_data = df['audio_data'].copy()


max_len = 1000  # Your desired fixed length
processed_audio = []
processed_labels = []

print(f"Processing audio data with max_len = {max_len}")
print(f"Original DataFrame has {len(df)} rows.")

# Iterate through the DataFrame rows using itertuples (generally faster)
# 'Index' is the DataFrame index, 'audio_data' and 'label' match column names
for row in tqdm(df.itertuples(), total=len(df), desc="Processing Audio"):
    # --- Get data for the current row ---
    # Use getattr for robustness if column names might vary slightly
    # print(row)
    original_seq = getattr(row, 'audio_data')
    original_label = getattr(row, '_3') # Fetch the label

    # Ensure the sequence is a numpy array with the desired dtype
    # This handles cases where 'audio_data' might contain lists
    seq = np.array(original_seq, dtype=np.float32)
    current_len = len(seq)

    # --- Case 1: Sequence is shorter than max_len ---
    if current_len < max_len:
        padding_needed = max_len - current_len
        # Use np.pad which can be cleaner for padding
        padded_seq = np.pad(seq, (0, padding_needed), mode='constant', constant_values=0)
        processed_audio.append(padded_seq)
        processed_labels.append(original_label) # Append the original label

    # --- Case 2: Sequence is exactly max_len ---
    elif current_len == max_len:
        processed_audio.append(seq) # No padding or truncation needed
        processed_labels.append(original_label) # Append the original label

    # --- Case 3: Sequence is longer than max_len ---
    else:
        # Calculate how many full chunks we can get
        num_full_chunks = current_len // max_len

        # Iterate through the sequence, extracting non-overlapping chunks
        for i in range(num_full_chunks):
            start_index = i * max_len
            end_index = start_index + max_len
            chunk = seq[start_index:end_index]
            processed_audio.append(chunk)
            # Append the SAME original label for EACH chunk
            processed_labels.append(original_label)

        # --- Handle the remainder (the part left over after full chunks) ---
        remainder_len = current_len % max_len
        if remainder_len > 0:
            # Extract the remainder
            remainder_start_index = num_full_chunks * max_len
            remainder_chunk = seq[remainder_start_index:]

            # Pad the remainder to max_len
            padding_needed = max_len - remainder_len
            padded_remainder = np.pad(remainder_chunk, (0, padding_needed), mode='constant', constant_values=0)

            processed_audio.append(padded_remainder)
            # Append the SAME original label for the padded remainder chunk
            processed_labels.append(original_label)


# --- Final Output ---
# Convert lists to numpy arrays (common practice for ML/DL)
final_audio_array = np.array(processed_audio)
# Labels can be kept as a list or converted to numpy array/pandas Series
final_labels = processed_labels # Or np.array(processed_labels)

print(f"\nFinished processing.")
print(f"Resulting audio array shape: {final_audio_array.shape}")
# Based on example: (1+1+2+1+1) = 6 samples -> (6, 100)
print(f"Number of resulting labels: {len(final_labels)}")
# Based on example: 6 labels -> ['Short', 'Exact', 'Long', 'Long', 'Long', 'Short_2']
# print("Example processed labels:", final_labels)


import pandas as pd
import numpy as np


print(f"Original audio array shape: {final_audio_array.shape}")
print(f"Original number of labels: {len(final_labels)}")


### VARIABLE TO CHANGE
limit_per_label = 2000

# 1. Create a DataFrame with labels and original indices
#    This avoids putting the large audio array into the DataFrame, saving memory.
print("\nCreating temporary DataFrame with labels and indices...")
indices_df = pd.DataFrame({
    'label': final_labels,
    'original_index': np.arange(len(final_labels)) # Store 0, 1, 2, ... N-1
})

# 2. Group by label and select the first 'limit_per_label' indices for each group
print(f"Grouping by label and selecting up to {limit_per_label} indices per group...")
# .head(n) conveniently takes min(n, group_size) automatically
selected_indices_df = indices_df.groupby('label', observed=True).head(limit_per_label)
# 'observed=True' can speed up grouping if labels are categorical

# --- Alternative: Random Sampling (if you don't want the *first* 100) ---
# def sample_or_head(group, n):
#     group_size = len(group)
#     # Sample if group is large enough, otherwise take all (head)
#     return group.sample(n=min(group_size, n), random_state=42) # Use random_state for reproducibility
#
# selected_indices_df = indices_df.groupby('label', observed=True).apply(sample_or_head, n=limit_per_label).reset_index(drop=True)
# print(f"Grouping by label and randomly sampling up to {limit_per_label} indices per group...")
# --- End Alternative ---


# 3. Get the selected original indices
selected_indices = selected_indices_df['original_index'].values

# Ensure the indices are sorted if you want the final array order to be somewhat grouped by label
# This is optional, .head() preserves original relative order within groups.
# selected_indices = np.sort(selected_indices)

print(f"Total indices selected: {len(selected_indices)}")

# 4. Use the selected indices to filter your original arrays
print("Filtering original audio array and labels using selected indices...")
limited_audio_array = final_audio_array[selected_indices]

# Ensure final_labels is a numpy array if it isn't already for fancy indexing
final_labels_array = np.array(final_labels)
limited_labels_array = final_labels_array[selected_indices]

# --- Verification (Optional) ---
print("\n--- Verification ---")
print(f"Limited audio array shape: {limited_audio_array.shape}")
print(f"Limited labels array length: {len(limited_labels_array)}")

# Check counts per label in the limited set
unique_labels, counts = np.unique(limited_labels_array, return_counts=True)
print("Counts per label in the limited dataset:")
for label, count in zip(unique_labels, counts):
    print(f"  Label '{label}': {count} samples")
    if count > limit_per_label:
        print(f"  WARNING: Label '{label}' has more than {limit_per_label} samples!") # Should not happen with .head()

print("\nDownsampling complete.")

# Now use 'limited_audio_array' and 'limited_labels_array' for further steps


import tensorflow_hub as hub
import os

# Define the path to the model directory provided by Kaggle
# This path will be inside '/kaggle/input/'. The exact name depends
# on how Kaggle named the model input folder (e.g., 'yamnet', 'yamnet-yamnet-v1', etc.)
# You might need to check your /kaggle/input/ directory to confirm the exact name.

# Example - Adjust 'yamnet-yamnet-v1' based on your actual input folder name
model_dir_path = '/kaggle/input/yamnet/tensorflow2/yamnet/1/' #<-- CHECK AND CHANGE THIS if needed

# Verify the path contains the saved_model.pb file
if os.path.exists(os.path.join(model_dir_path, 'saved_model.pb')):
    print(f"Loading model from: {model_dir_path}")
    # Load the model from the local SavedModel directory
    model_yamnet = hub.load(model_dir_path)
    print("Model loaded successfully!")

    # Now you can use the model as before
    # Example: Check the model's expected input signature
    # print(model_yamnet.signatures['default'].pretty_print())

else:
    print(f"Error: 'saved_model.pb' not found in {model_dir_path}")
    print("Please check the directory path and the contents of /kaggle/input/")
    print("Contents of /kaggle/input/:")
    print(os.system('ls /kaggle/input/'))


audio_embeddings = []
for i in tqdm(limited_audio_array, desc="Generating Audio Embeddings"):
    waveform = i / tf.int16.max
    scores, embeddings, spectrogram = model_yamnet(waveform)
    audio_embeddings.append(embeddings)


X = np.array(audio_embeddings) # Shape will be (N, 1024)
X_reshaped = np.squeeze(X, axis=1) # Specify axis=1 to remove the middle dimension
print(f"Generated embeddings array X with shape: {X_reshaped.shape}")

# Ensure you have the corresponding labels (length N)
y = limited_labels_array


from sklearn.preprocessing import LabelEncoder
# --- Label Encoding ---
print("\nEncoding labels...")
# 1. Encode string/object labels to integers (0, 1, 2, ...)
label_encoder = LabelEncoder()
y_integer_encoded = label_encoder.fit_transform(y)

# 2. Determine the number of unique classes
num_classes = len(label_encoder.classes_)
print(f"Found {num_classes} unique classes: {label_encoder.classes_}")

# 3. Convert integer labels to one-hot encoding
y_one_hot = tf.keras.utils.to_categorical(y_integer_encoded, num_classes=num_classes)
print(f"One-hot encoded labels shape: {y_one_hot.shape}") # Should be (N, num_classes)

# --- Train/Test Split ---
print("\nSplitting data into training and testing sets...")
xtrain, xtest, ytrain_one_hot, ytest_one_hot = train_test_split(
    X_reshaped,                      # Your embeddings array (N, 1024)
    y_one_hot,              # Your one-hot encoded labels (N, num_classes)
    test_size=0.2,          # Fraction for the test set
    random_state=42,        # For reproducibility
    stratify=y_integer_encoded # IMPORTANT: Stratify based on original integer labels
                               # to ensure class balance in train/test sets
)

print(f"xtrain shape: {xtrain.shape}") # (N * 0.8, 1024)
print(f"ytrain_one_hot shape: {ytrain_one_hot.shape}") # (N * 0.8, num_classes)
print(f"xtest shape: {xtest.shape}") # (N * 0.2, 1024)
print(f"ytest_one_hot shape: {ytest_one_hot.shape}") # (N * 0.2, num_classes)


print("\nDefining and compiling the model...")
# --- Define Model ---
model = models.Sequential([
    # Input shape is now (1024,) after averaging embeddings
    layers.Input(shape=(1024,)),
    # Flatten layer is not needed if input is 1D before Dense
    # layers.Flatten(), # Remove this
    layers.Dense(300, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(300, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(200, activation='relu'),
    # Output layer must have 'num_classes' units
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

# --- Compile Model ---
# Using categorical_crossentropy because ytrain is one-hot encoded
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Model compiled successfully.")


from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor (usually validation loss or accuracy)
    patience=3,         # Number of epochs with no improvement after which training will be stopped
    verbose=1,           # Set to 1 to print messages when stopping happens
    mode='min',          # 'min' for loss/error metrics, 'max' for accuracy metrics
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity.
)


# Check for GPU availability
gpu_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpu_devices))

if len(gpu_devices) > 0:
    print("GPU(s) found:")
    for gpu in gpu_devices:
        print(f"- {gpu.name}")
        # Try to print details (might require specific CUDA versions)
        try:
            tf.config.experimental.get_device_details(gpu)
            print("  Details:", tf.config.experimental.get_device_details(gpu))
        except: # Handle cases where details might not be fetchable
             print("  Could not fetch device details.")
    # TensorFlow automatically places operations on the GPU if available
    print("\nTensorFlow should automatically use the GPU for training.")
else:
    print("\nNo GPU found. TensorFlow will use the CPU.")
    print("Ensure NVIDIA drivers, CUDA Toolkit, and cuDNN are installed correctly and compatible.")


# --- Train Model ---
print("\nStarting model training...")
epochs = 100
history = model.fit(
    xtrain,
    ytrain_one_hot,
    epochs=epochs,
    validation_split=0.1, # Optional: use part of training data for validation during training
    callbacks=[early_stopping]
    # Or use validation_data=(xtest, ytest_one_hot) - be careful not to "tune" on test set
)

print("Model training finished.")

# --- Evaluate Model (Optional) ---
print("\nEvaluating model on the test set...")
loss, accuracy = model.evaluate(xtest, ytest_one_hot, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


import os
import pandas as pd

# Define the root directory containing the audio files
root_dir = '/kaggle/input/birdclef-2025/test_audio' # Adjust if your path is different

# List to store the data for the DataFrame
audio_data_list = []

# Walk through the directory structure
for dirname, _, filenames in os.walk(root_dir):
    # Skip the root directory itself if it doesn't contain class folders directly
    # (Adjust this condition if your structure is different)
    if dirname == root_dir:
        continue

    # Extract the class name (subdirectory name)
    # os.path.basename gets the last part of the directory path
    class_name = os.path.basename(dirname)

    # Iterate through files in the current directory
    for filename in filenames:
        # Construct the full path to the audio file
        full_path = os.path.join(dirname, filename)

        # Append the file path and its class to the list
        audio_data_list.append([full_path, class_name])
        # You can remove the print statement if you don't need it anymore
        # print(full_path) # Optional: print the path as it's processed

# Create the Pandas DataFrame
audio_dataframe = pd.DataFrame(audio_data_list, columns=["audio_path", "class"])

# Display the first few rows of the DataFrame (optional)
print(audio_dataframe.head())

# Display the shape of the DataFrame (optional)
print(f"\nDataFrame shape: {audio_dataframe.shape}")


import soundfile as sf
import numpy as np

def segment_audio(audio, sample_rate, segment_length=5):
    """Split audio into non-overlapping segments of segment_length seconds."""
    segment_samples = int(segment_length * sample_rate)
    total_segments = len(audio) // segment_samples
    segments = []
    for i in range(total_segments):
        start = i * segment_samples
        end = start + segment_samples
        segments.append(audio[start:end])
    return segments

def get_yamnet_embedding(segment, model_yamnet):
    # YAMNet expects float32 waveform in [-1.0, 1.0]
    waveform = segment.astype(np.float32) / np.abs(segment).max()
    scores, embeddings, spectrogram = model_yamnet(waveform)
    # Average embeddings over time axis (axis=0)
    return np.mean(embeddings.numpy(), axis=0)

import pandas as pd
import os

test_dir = '/kaggle/input/birdclef-2025/test_soundscapes'
submission_rows = []

for filename in os.listdir(test_dir):
    if not filename.endswith('.ogg'):
        continue
    filepath = os.path.join(test_dir, filename)
    audio, sr = sf.read(filepath)
    segments = segment_audio(audio, sr, segment_length=5)
    for i, segment in enumerate(segments):
        embedding = get_yamnet_embedding(segment, model_yamnet)
        # Model expects shape (1, 1024)
        pred = model.predict(embedding.reshape(1, -1))
        # Format row_id as required: soundscape_xxxxxx_[end_time]
        end_time = (i + 1) * 5
        row_id = f"{filename.replace('.ogg','')}_{end_time}"
        # pred is shape (1, 206)
        submission_rows.append([row_id] + pred.flatten().tolist())

# Get species columns from sample_submission.csv
sample_sub = pd.read_csv('/kaggle/input/birdclef-2025/sample_submission.csv')
species_cols = sample_sub.columns[1:]

# Build DataFrame
submission_df = pd.DataFrame(submission_rows, columns=['row_id'] + list(species_cols))
submission_df.to_csv('submission.csv', index=False)

