import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import librosa
import numpy as np
import pandas as pd

import gc
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable, Tuple, List
import traceback  # Import traceback module


# Define file paths 
test_data = "/kaggle/input/birdclef-2025/test_soundscapes"
submission = "/kaggle/input/birdclef-2025/sample_submission.csv"
train_csv = "/kaggle/input/birdclef-2025/train.csv"
taxonomy_csv = "/kaggle/input/birdclef-2025/taxonomy.csv"

transform: Optional[Callable] = None  # Type hint for transform
audio_transform: Optional[Callable] = None # Type hint for audio_transform

@dataclasses.dataclass
class AudioParam:
    SR: int = 32_000  # Sample rate
    NFFT: int = 2048  # Number of FFT points
    NMEL: int = 128   # Number of Mel bands
    FMAX: int = 16_000 # Maximum frequency
    FMIN: int = 20   # Minimum frequency
    HOP_LENGTH: int = NFFT // 4  # Hop length

audio_param = AudioParam()

# Load submission CSV to get class names
try:
    sub_csv = pd.read_csv(submission)
    idx2cls = sub_csv.columns.drop("row_id").tolist()  # List of bird species (class names)
    cls2idx = {c: i for i, c in enumerate(idx2cls)} # Class name to index mapping
except FileNotFoundError as e:
    print(f"Error: sample_submission.csv not found! {e}")
    idx2cls = [] # Provide a default for testing, but the code will likely fail
    cls2idx = {}


DEBUG = True # Enable Debugging
file_names = [os.path.join(test_data, fp) for fp in os.listdir(test_data) if fp.endswith(".ogg")]

# Use a single file for debugging.  This makes the matrix dimension calculations easier.
if len(file_names) == 0:
    file_names = [
        "/kaggle/input/birdclef-2025/train_soundscapes/H02_20230420_074000.ogg",
    ]
    DEBUG = True



#  a simpler, randomly initialized CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        # Calculate the input size to the linear layer dynamically
        self._to_linear = None  # Placeholder, will be calculated during the first forward pass
        self.fc1 = nn.Linear(1, num_classes)  # Placeholder Linear layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.flatten(x)

            # Dynamically determine the input size of the linear layer
            if self._to_linear is None:
                self._to_linear = x.shape[1]
                if self._to_linear == 0:
                   print("Error: self._to_linear is zero!")
                   #Handle this better - e.g., skip or set a min size
                   return torch.zeros((1, len(idx2cls)))  # or a zero tensor of the right size
                self.fc1 = nn.Linear(self._to_linear, len(idx2cls))  # Update the linear layer
            x = self.fc1(x)
            return x
        except Exception as e:
            print(f"Error in SimpleCNN.forward: {e}")
            return torch.zeros((1, len(idx2cls)))



# Instantiate the SimpleCNN model.
model = SimpleCNN(num_classes=len(idx2cls))
model.eval() # Set the model to evaluation mode.

def pipeline(x: np.ndarray) -> np.ndarray:
    """
    Converts audio data to a mel spectrogram and then to a dB scale.
    """
    try:
        mels = librosa.feature.melspectrogram(
            y=x,
            sr=audio_param.SR,
            n_fft=audio_param.NFFT,
            n_mels=audio_param.NMEL,
            fmax=audio_param.FMAX,
            fmin=audio_param.FMIN,
            hop_length=audio_param.HOP_LENGTH,
        )
        db_map = librosa.power_to_db(mels, ref=np.max)
        db_map = (db_map + 80) / (80 + 1e-6)  # Normalize to [0, 1] - Added small constant
        if np.isnan(db_map).any():
            print("Warning: NaN values detected in db_map!")
            db_map = np.nan_to_num(db_map) #Replace with 0

        return db_map[None, :, :]  # Add a channel dimension (1, height, width)
    except Exception as e:
        print(f"Error in pipeline: {e}")
        return np.zeros((1, audio_param.NMEL, 1)) # return a zero array



@torch.no_grad()
def predict(fp: str) -> Tuple[np.ndarray, List[str]]:
    """
    Predicts bird calls in a given audio file.

    Args:
        fp (str): File path of the audio file.

    Returns:
        Tuple[np.ndarray, List[str]]: Tuple containing the model output and the list of row IDs.
    """
    try:
        x, _ = librosa.load(fp, sr=audio_param.SR)  # Load the audio file.

        if x.size == 0:
            print(f"Warning: Audio file {fp} is empty!")
            return np.array([]), [] #return empty arrays
    except Exception as e:
        print(f"Error loading file {fp}: {e}")
        return np.array([]), []

    # Number of 5-second segments
    num_segments = int(np.floor(len(x) / audio_param.SR / 5))
    all_outs = []
    all_row_ids = []
    for i in range(num_segments):
        start = i * audio_param.SR * 5
        end = (i + 1) * audio_param.SR * 5
        segment = x[start:end]


        if audio_transform is not None:
            try:
                segment = audio_transform(sample=segment, sample_rate=audio_param.SR) #Apply audio transform
            except Exception as e:
                print(f"Audio Transform Failed {e}")

        try:
            segment = pipeline(segment)  #Convert waveform to mel spectrogram.
        except Exception as e:
            print(f"Pipeline failed {e}")
            continue

        if transform is not None:
            try:
                segment = transform(image=segment)["image"] #Apply image transform.
            except Exception as e:
                print(f"Transform failed {e}")
                continue

        try:
            segment = torch.from_numpy(segment).float().unsqueeze(0)  # Convert to tensor and add batch dimension.
            out = model(segment).sigmoid().detach().cpu().numpy() # Get the model output.
            all_outs.append(out[0])

            fp_name = os.path.basename(fp).split(".")[0] #Extract the base filename.
            row_id = f"{fp_name}_{(i + 1) * 5}" #Create row IDs.  Correct the slice name
            all_row_ids.append(row_id)
        except Exception as e:
            print(f"Error during processing of segment {i} in {fp}: {e}  {traceback.format_exc()}") #Print trace

    return np.array(all_outs), all_row_ids # return all values



row_id = []
matrix = []

#Using a ThreadPoolExecutor to parallelize the predictions
with ThreadPoolExecutor(max_workers=4) as executor:
    for fp_idx, (fp) in enumerate(file_names): #Enumerate so you know the file index
        try:
            out, rid = predict(fp)
            if len(rid) > 0:  # Only extend if there are valid results
                row_id.extend(rid)  # Changed append to extend to unpack the list of strings
                matrix.extend(out)  # Append the output, which should have shape (num_classes,)
            else:
                print(f"Warning: No predictions generated for file: {fp}")
        except Exception as e:
            print(f"Failed to run predict for file {fp} {e}") #Major problem.
        gc.collect() #Collect after each file
        print(f"Finished {fp_idx+1}/{len(file_names)}") #Track progress

try:
    matrix = np.array(matrix).reshape(-1, len(idx2cls))  # Reshape to (num_segments, num_classes)
    row_id = np.array(row_id).reshape(-1, 1)  # Ensure row_id is a 2D array
    matrix = np.hstack([row_id, matrix])  # Now both arrays have the same dimensions

    # Create a Pandas DataFrame from the results.
    sub = pd.DataFrame(matrix, columns=["row_id", *idx2cls])
    sub.to_csv('submission.csv', index=False)

    print(sub.head())
except Exception as e:
    print(f"Error creating submission file {e}") #Most likely problem.

print("Finished!") #If you see this, then great!
gc.collect()


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import IPython.display as ipd
import soundfile as sf
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings('ignore')

# Define paths
DATA_PATH = '/kaggle/input/birdclef-2025'
TRAIN_AUDIO_PATH = os.path.join(DATA_PATH, 'train_audio')
TEST_SOUNDSCAPES_PATH = os.path.join(DATA_PATH, 'test_soundscapes')
TRAIN_SOUNDSCAPES_PATH = os.path.join(DATA_PATH, 'train_soundscapes')

# Load datasets
train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
taxonomy_df = pd.read_csv(os.path.join(DATA_PATH, 'taxonomy.csv'))

# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------


# --- 1.4. Recording Location Data ---
try:
    with open(os.path.join(DATA_PATH, 'recording_location.txt'), 'r') as f:
        print("\nRecording Location:")
        print(f.read())
except FileNotFoundError:
    print("\nrecording_location.txt not found.")

# --- 1.5. Train Audio Examples ---

def plot_audio_example(filename, title):
    """Plots waveform and spectrogram of an audio file."""
    file_path = os.path.join(TRAIN_AUDIO_PATH, filename)
    try:
        y, sr = librosa.load(file_path)
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        librosa.display.waveshow(y, sr=sr)
        plt.title(f'Waveform: {title}')

        plt.subplot(1, 2, 2)
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram: {title}')
        plt.tight_layout()
        plt.show()

        print(f"Playing audio: {title}")
        ipd.display(ipd.Audio(file_path))

    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("\nTrain Audio Examples:")
for i in range(4):
    example_filename = train_df['filename'].iloc[i]
    example_common_name = train_df['common_name'].iloc[i]
    plot_audio_example(example_filename, example_common_name)


# --- 1.6. Soundscape Examples ---
def plot_soundscape_example(filename, title):
    """Plots waveform and spectrogram of a soundscape audio file."""
    file_path = os.path.join(TRAIN_SOUNDSCAPES_PATH, filename)
    try:
        y, sr = librosa.load(file_path)
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        librosa.display.waveshow(y, sr=sr)
        plt.title(f'Waveform: {title}')

        plt.subplot(1, 2, 2)
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram: {title}')
        plt.tight_layout()
        plt.show()

        print(f"Playing audio: {title}")
        ipd.display(ipd.Audio(file_path))

    except Exception as e:
        print(f"Error processing {filename}: {e}")


print("\nSoundscape Examples:")
soundscape_files = [f for f in os.listdir(TRAIN_SOUNDSCAPES_PATH) if f.endswith('.ogg')]
for i in range(min(4, len(soundscape_files))):  # Display up to 4 examples
    plot_soundscape_example(soundscape_files[i], f"Soundscape {i+1}")



# --- 1.7. Species Distribution ---
plt.figure(figsize=(12, 6))
species_counts = train_df['common_name'].value_counts().nlargest(20)
sns.barplot(x=species_counts.index, y=species_counts.values, palette="viridis")
plt.xticks(rotation=90)
plt.title('Top 20 Most Frequent Species')
plt.xlabel('Species')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# --- 1.8. Geographical Distribution (Map) ---

try:
    import folium

    # Create a map centered around the average latitude and longitude
    m = folium.Map(location=[train_df['latitude'].mean(), train_df['longitude'].mean()], zoom_start=6)

    # Add markers for each recording location
    for index, row in train_df.iterrows():
        folium.Marker([row['latitude'], row['longitude']],
                      popup=f"{row['common_name']} ({row['primary_label']})").add_to(m)

    # Display the map (you may need to save it to an HTML file and display that in Kaggle)
    m  # Display the map in the output.  If it doesn't render, save to HTML and display that.
    m.save("recording_locations.html") # Save the map to an HTML file.
    print("Map saved to recording_locations.html.  Display this file to see the map.")


except ImportError:
    print("Folium is not installed. Install it to visualize the map: pip install folium")
except Exception as e:
    print(f"Error creating map: {e}")


# --- 1.9. Taxonomy Visualization ---

plt.figure(figsize=(12,6))
class_counts = taxonomy_df['class_name'].value_counts()
sns.barplot(x=class_counts.index, y=class_counts.values, palette="magma")
plt.xticks(rotation=45)
plt.title("Distribution of Classes in Taxonomy")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()




