import librosa
import librosa.display
import matplotlib.pyplot as plt


audio_path = "/kaggle/input/birdclef-2025/train_audio/1139490/CSA36385.ogg"
y, sr = librosa.load(audio_path, sr=32000)  # Sample rate = 32 kHz


plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr, alpha=0.5)
plt.title("Raw Audio Waveform (CSA36385.ogg)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()


import librosa
import librosa.display
import matplotlib.pyplot as plt


audio_path = "/kaggle/input/birdclef-2025/train_soundscapes/H02_20230420_074000.ogg"
y, sr = librosa.load(audio_path, sr=32000)  


plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr, alpha=0.5)
plt.title("Raw Audio Waveform (H02_20230420_074000.ogg)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()


import numpy as np
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=16000)
S_dB = librosa.power_to_db(S, ref=np.max)  # Convert to dB

# Plot
plt.figure(figsize=(12, 6))
librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr, fmax=16000)
plt.colorbar(format="%+2.0f dB")
plt.title("Mel-Spectrogram (XC12345.ogg)")
plt.show()


import seaborn as sns
import pandas as pd

train_df = pd.read_csv("/kaggle/input/birdclef-2025/train.csv")

plt.figure(figsize=(12, 8))
sns.countplot(data=train_df, y="primary_label", order=train_df["primary_label"].value_counts().index[:20])  # Top 20 species
plt.title("Top 20 Species by Training Samples")
plt.show()


sns.countplot(data=train_df, x="rating")
plt.title("Distribution of Recording Quality Ratings (1-5)")
plt.show()


sns.scatterplot(data=train_df, x="longitude", y="latitude", alpha=0.5, hue="collection")
plt.title("Recording Locations (XC vs. iNat vs. CSA)")
plt.show()





import pandas as pd
import librosa
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load metadata
train_df = pd.read_csv("/kaggle/input/birdclef-2025/train.csv")

def get_duration_safe(row):
    audio_path = os.path.join("/kaggle/input/birdclef-2025/train_audio",  
                            row['filename'])
    try:
        return librosa.get_duration(path=audio_path)  # Use 'path' instead of 'filename'
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None  # Return None for missing files

train_df["duration"] = train_df.apply(get_duration_safe, axis=1)

# Drop rows with missing durations
train_df = train_df.dropna(subset=['duration'])

# Plot top 10 species by duration
plt.figure(figsize=(12, 6))
top_species = train_df["primary_label"].value_counts().index[:10]
sns.boxplot(
    data=train_df[train_df["primary_label"].isin(top_species)],
    x="duration",
    y="primary_label",
    order=top_species
)
plt.title("Call Duration by Species (Top 10)")
plt.tight_layout()  # Prevent label cutoff
plt.show()


# Load a soundscape
soundscape, sr = librosa.load("/kaggle/input/birdclef-2025/train_soundscapes/H02_20230420_074000.ogg", sr=32000)

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
librosa.display.waveshow(soundscape, sr=sr)
plt.title("Soundscape Waveform (Background Noise)")

plt.subplot(2, 1, 2)
S_soundscape = librosa.feature.melspectrogram(y=soundscape, sr=sr)
librosa.display.specshow(librosa.power_to_db(S_soundscape), x_axis="time", y_axis="mel")
plt.title("Soundscape Spectrogram")
plt.tight_layout()
plt.show()


# Load a soundscape
soundscape, sr = librosa.load("/kaggle/input/birdclef-2025/train_audio/1139490/CSA36385.ogg", sr=32000)


plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
librosa.display.waveshow(soundscape, sr=sr)
plt.title("Soundscape Waveform (Background Noise)")

plt.subplot(2, 1, 2)
S_soundscape = librosa.feature.melspectrogram(y=soundscape, sr=sr)
librosa.display.specshow(librosa.power_to_db(S_soundscape), x_axis="time", y_axis="mel")
plt.title("Soundscape Spectrogram")
plt.tight_layout()
plt.show()


import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # For saving models

# Paths
TRAIN_AUDIO_PATH = '/kaggle/input/birdclef-2025/train_audio'
TRAIN_CSV_PATH = '/kaggle/input/birdclef-2025/train.csv'
TRAIN_SOUNDSCAPE_PATH = '/kaggle/input/birdclef-2025/train_soundscapes'

# Step 1: Load Data
print("Loading metadata...")
train_df = pd.read_csv(TRAIN_CSV_PATH)

print(f"Number of samples: {len(train_df)}")
print(train_df.head())


# Step 2: Feature Extraction
def extract_features(file_path, n_mfcc=13, fixed_length=100):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # Pad or truncate to fixed length
        if mfcc.shape[1] < fixed_length:
            mfcc = np.pad(mfcc, ((0, 0), (0, fixed_length - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :fixed_length]
        
        return mfcc.flatten()  # Flatten into a 1D array
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros(n_mfcc * fixed_length)  # Return consistent-sized zero array



print("Extracting features (dummy setup with 10 samples)...")
dummy_train_df = train_df.head(10).copy()
dummy_train_df['file_path'] = dummy_train_df['filename'].apply(lambda x: os.path.join(TRAIN_AUDIO_PATH, x))
dummy_train_df['features'] = dummy_train_df['file_path'].apply(extract_features)

# Update the main DataFrame
train_df.loc[dummy_train_df.index, 'features'] = dummy_train_df['features']





#    print("Extracting features...for all Images ")
#    train_df['file_path'] = train_df['filename'].apply(lambda x: os.path.join(TRAIN_AUDIO_PATH, x))
#  train_df['features'] = train_df['file_path'].apply(extract_features)



# Prepare input features
X = np.vstack(dummy_train_df['features'].values)
y = dummy_train_df['scientific_name']

# Encode target labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)



# Step 3: Train-Test Split
print("Splitting data...")
X_train, X_valid, y_train, y_valid = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 4: Model Training
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Step 5: Evaluation
print("Evaluating model...")
y_pred = model.predict(X_valid)
accuracy = accuracy_score(y_valid, y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")

# Save the model
joblib.dump(model, 'birdclef_model.pkl')

# Step 6: Generate Submission
def generate_submission(test_audio_folder, model, encoder):
    test_files = os.listdir(test_audio_folder)
    submission = []
    for file in test_files:
        file_path = os.path.join(test_audio_folder, file)
        features = extract_features(file_path).reshape(1, -1)
        pred = model.predict(features)
        label = encoder.inverse_transform(pred)[0]
        submission.append({'row_id': file.split('.')[0], 'birds': label})
    return pd.DataFrame(submission)

print("Generating submission...")
test_audio_path = '/kaggle/input/birdclef-2025/train_audio' 
submission_df = generate_submission(test_audio_path, model, le)
# Save file by removing # from below 
# submission_df.to_csv('submission.csv', index=False)

print("Submission file created.")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway, pearsonr, spearmanr

# Load the dataset
data = pd.read_csv("/kaggle/input/birdclef-2025/train.csv")

print("Descriptive Statistics:")
print(data.describe(include='all')) 
print("\nMissing Values:")
print(data.isnull().sum()) 


# Distribution of ratings
plt.figure(figsize=(8, 6))
sns.histplot(data['rating'], kde=True, bins=20)
plt.title("Distribution of Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()


# Frequency of scientific names
top_scientific_names = data['scientific_name'].value_counts().head(10)
plt.figure(figsize=(10, 6))
top_scientific_names.plot(kind='bar', color='skyblue')
plt.title("Top 10 Most Common Birds")
plt.xlabel("Scientific Name")
plt.ylabel("Frequency")
plt.show()


# Geospatial Distribution
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data['longitude'], y=data['latitude'], hue=data['rating'], palette="viridis")
plt.title("Geospatial Distribution of Bird Occurrences")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()




collections = data['collection'].unique()
ratings_by_collection = [data[data['collection'] == col]['rating'] for col in collections]

if len(collections) > 2:
    # ANOVA for more than two groups
    f_stat, p_value = f_oneway(*ratings_by_collection)
    print("\nANOVA Test Results:")
    print(f"F-statistic: {f_stat}, P-value: {p_value}")
else:
    # T-test for two groups
    t_stat, p_value = ttest_ind(ratings_by_collection[0], ratings_by_collection[1])
    print("\nT-Test Results:")
    print(f"T-statistic: {t_stat}, P-value: {p_value}")



pearson_corr_lat, _ = pearsonr(data['latitude'], data['rating'])
pearson_corr_lon, _ = pearsonr(data['longitude'], data['rating'])
print("\nCorrelation Analysis:")
print(f"Pearson Correlation (Latitude vs Rating): {pearson_corr_lat}")
print(f"Pearson Correlation (Longitude vs Rating): {pearson_corr_lon}")




