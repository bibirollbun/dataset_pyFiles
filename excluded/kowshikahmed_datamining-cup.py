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


import pandas as pd
import os
import numpy as np
from tqdm import tqdm
tqdm.pandas()




# Define path to dataset
base_path = '/kaggle/input/birdclef-2025'
audio_path = 'train_audio/'

print(os.listdir(base_path))  # to see available files and folders



meta= pd.read_csv(os.path.join(base_path, 'train.csv'))
meta['id'] = range(1, len(meta) + 1)
meta.head()


df = meta[['id' , 'filename' , 'primary_label']]

df['filename'] = df['filename'].apply(lambda x: os.path.join(base_path, audio_path, x))
file_path = df['filename'][0]  # now this is a correct full path


#sampling
df =df.head(500)


# load audio

def load_audio_from_filename(filename):
    try:
        y, sr = librosa.load(filename, sr=None)
        return y, sr
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return None, None
# Apply and split the results
df[['waveform', 'sample_rate']] = df['filename'].progress_apply(
    lambda filename: pd.Series(load_audio_from_filename(filename))
)



df.head(2)


def extract_mel_spectrogram(y, sr):
    if y is not None and sr is not None:
        try:
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            return mel_db
        except Exception as e:
            print(f"Error processing Mel Spectrogram: {e}")
            return None
    else:
        return None
        
def extract_mfcc(y, sr):
    if y is not None and sr is not None:
        try:
            mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            return mfcc_feat
        except Exception as e:
            print(f"Error processing MFCC: {e}")
            return None
    else:
        return None



# extracting spectrogram
df['mel_spectrogram'] = df.progress_apply(lambda row: extract_mel_spectrogram(row['waveform'], row['sample_rate']), axis=1)
df['mfcc'] = df.progress_apply(lambda row: extract_mfcc(row['waveform'], row['sample_rate']), axis=1)



df.head(2)


# Function to calculate the mean of Mel spectrogram
def calculate_mel_mean(mel_spectrogram):
    return np.mean(mel_spectrogram, axis=1)

# Function to calculate the variance of Mel spectrogram
def calculate_mel_var(mel_spectrogram):
    return np.var(mel_spectrogram, axis=1)

# Function to calculate the median of Mel spectrogram
def calculate_mel_median(mel_spectrogram):
    return np.median(mel_spectrogram, axis=1)

# Function to calculate the max of Mel spectrogram
def calculate_mel_max(mel_spectrogram):
    return np.max(mel_spectrogram, axis=1)

# Function to calculate the min of Mel spectrogram
def calculate_mel_min(mel_spectrogram):
    return np.min(mel_spectrogram, axis=1)

# Function to calculate the range (max - min) of Mel spectrogram
def calculate_mel_range(mel_spectrogram):
    mel_max = np.max(mel_spectrogram, axis=1)
    mel_min = np.min(mel_spectrogram, axis=1)
    return mel_max - mel_min

# Now, assuming the 'mel_spectrogram' column exists in the dataframe 'df', 
# we will apply each of these functions to the dataframe.
features_columns = ['mel_mean', 'mel_var', 'mel_median', 'mel_max', 'mel_min', 'mel_range']

df['mel_mean'] = df['mel_spectrogram'].progress_apply(calculate_mel_mean)
df['mel_var'] = df['mel_spectrogram'].progress_apply(calculate_mel_var)
# df['mel_median'] = df['mel_spectrogram'].progress_apply(calculate_mel_median)
# df['mel_max'] = df['mel_spectrogram'].progress_apply(calculate_mel_max)
# df['mel_min'] = df['mel_spectrogram'].progress_apply(calculate_mel_min)
# df['mel_range'] = df['mel_spectrogram'].progress_apply(calculate_mel_range)



df.head(2)


import pandas as pd
import numpy as np

# Function to expand the feature column into separate bin columns
def expand_feature_column(df, feature_name):
    # Check the first row to confirm the format (ensure it's a numpy array)
    first_row = df[feature_name].iloc[0]
    if not isinstance(first_row, np.ndarray):
        raise ValueError(f"Values in column '{feature_name}' are not numpy arrays.")
    
    # Get the number of bins (assuming each row contains the same number of bins)
    n_bins = len(first_row)  # The length of the array in the first row

    # Create a list of new columns for the bins
    bin_columns = []
    for bin_idx in range(n_bins):
        # Create a new column for each bin (e.g., mel_mean_bin_1, mel_mean_bin_2, ...)
        bin_columns.append(
            df[feature_name].apply(lambda x: x[bin_idx] if isinstance(x, np.ndarray) else None)
        )
        
    # Combine the list of bin columns into a DataFrame
    bin_df = pd.concat(bin_columns, axis=1)
    
    # Rename the new columns appropriately (e.g., mel_mean_bin_1, mel_mean_bin_2, ...)
    bin_df.columns = [f'{feature_name}_bin_{bin_idx + 1}' for bin_idx in range(n_bins)]
    

    # Drop the original feature column (optional)
    #df.drop(columns=[feature_name], inplace=True)
    
    return bin_df

features_columns = ['mel_mean', 'mel_var']

df_expanded = df
features = [] 
for feature in features_columns : 
    
    expandeddf = expand_feature_column(df, feature)
    features = features + list(expandeddf.columns)
    df = pd.concat([df, expandeddf], axis=1)

df.head(2)






from sklearn.preprocessing import LabelEncoder

# Create the encoder
le = LabelEncoder()

# Fit and transform the column
df['primary_label_encoded'] = le.fit_transform(df['primary_label'])



from sklearn.model_selection import train_test_split

# Suppose you have features X and labels y
X_train, X_test, y_train, y_test = train_test_split(
    df[features],          # your features
    df['primary_label_encoded'],          # your labels
    test_size=0.2,  # 20% for test set
    random_state=42 # (optional) to make it reproducible
)



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



from xgboost import XGBClassifier 

# Create the XGBClassifier
model = XGBClassifier(
    n_estimators=10,   # number of trees
    learning_rate=0.1,  # step size shrinkage
    max_depth=6,        # maximum depth of a tree
    random_state=42     # for reproducibility
)

# Fit the model on training data
model.fit(X_train_scaled, y_train)

# Predict on test data
y_pred = model.predict(X_test_scaled)



import xgboost as xgb
booster = model.get_booster()

# Predict with feature contributions
model_pred_detail = booster.predict(xgb.DMatrix(X_test_scaled), pred_contribs=True)
print(model_pred_detail.shape)
model_pred_detail


import numpy as np
import matplotlib.pyplot as plt

# Calculate the mean contribution for each feature across all samples
mean_contribution_per_feature = np.mean(model_pred_detail[:, :, :-1], axis=(0, 1))  # Exclude the last column (class probabilities)

# Plot the mean contributions per feature using a box plot
plt.figure(figsize=(12, 8))
plt.boxplot(mean_contribution_per_feature.T, vert=False)  # Transpose to have features as boxes
plt.xlabel("Mean Feature Contribution")
plt.ylabel("Feature Index")
plt.title("Box Plot of Mean Feature Contributions")
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# If the mean_contribution_per_feature is 1D, you can reshape it into a 2D array
mean_contribution_per_feature = np.mean(model_pred_detail[:, :, :-1], axis=(0, 1))

# Reshaping to make it 2D (for heatmap compatibility, 1 feature per row)
mean_contribution_per_feature_2d = mean_contribution_per_feature.reshape(1, -1)

# Plot the heatmap
plt.figure(figsize=(10, 2))  # Adjust the figure size as needed
sns.heatmap(mean_contribution_per_feature_2d, cmap="YlGnBu")
plt.title("Mean Feature Contribution Heatmap")
plt.xlabel("Features")
plt.ylabel("Mean Contribution")
plt.show()



len(np.mean(model_pred_detail, axis=0)[0] ) # axis=0 means the mean along rows (across samples)



import matplotlib.pyplot as plt
import librosa.display

def plot_mel_and_mfcc(df, index=0):
    mel = df.loc[index, 'mel_spectrogram']
    mfcc = df.loc[index, 'mfcc']
    sr = df.loc[index, 'sample_rate']

    plt.figure(figsize=(12, 6))

    # Plot Mel Spectrogram
    plt.subplot(1, 2, 1)
    librosa.display.specshow(mel, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')

    # Plot MFCC
    plt.subplot(1, 2, 2)
    librosa.display.specshow(mfcc, sr=sr, x_axis='time', cmap='coolwarm')
    plt.colorbar()
    plt.title('MFCC')

    plt.tight_layout()
    plt.show()
plot_mel_and_mfcc(df, index=0)

