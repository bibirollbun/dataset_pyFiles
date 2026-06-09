import pandas as pd
import numpy as np


df_train  = pd.read_csv("/kaggle/input/birdclef-2025/train.csv")
df_train.head(5)


df_train.info()


!pip install reverse_geocoder




import reverse_geocoder as rg

df_train.dropna(inplace=True)

# Convert DataFrame to a list of (lat, lon) tuples
coordinates = list(zip(df_train['latitude'], df_train['longitude']))

# Perform reverse geocoding
results = rg.search(coordinates)

# Extract city names from the results
df_train['city'] = [result['name'] for result in results]






df_train.city.value_counts()


!pip install resampy


from folium.plugins import HeatMap

# Initialize map
folium_map = folium.Map(location=map_center, zoom_start=2)

# Add heatmap layer
HeatMap(df_train[['latitude', 'longitude']].values).add_to(folium_map)

# Save map
folium_map.save('map.html')



## heatmap visulization ## 
from IPython.core.display import display

display(folium_map)


!pip install  resampy


import gc
gc.collect()
del df_train


import os 
class_labels = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))



## creating Dataframe which contains class names and filename paths 
import os
import pandas as pd

file_paths = []  # List to store file paths
base_path = "/kaggle/input/birdclef-2025/train_audio"
sub_dir = []
# Iterate over each subdirectory and list its files
for subdir in os.listdir(base_path):
    subdir_path = os.path.join(base_path, subdir)  # Full path to subdirectory
    if os.path.isdir(subdir_path):  # Ensure it's a directory
        for file in os.listdir(subdir_path):  # Iterate over files in subdir
            file_paths.append(f"{subdir}/{file}")  # Store relative path
            sub_dir.append(subdir)

# Convert list to DataFrame
df = pd.DataFrame({'class_names' : sub_dir , 'filename' : file_paths})

print(df.head())  # Display first few rows



## Using Librosa to convert audio files to numerical values 

import numpy as np
from tqdm import tqdm
import os
import librosa
from joblib import Parallel, delayed

# Function to extract MFCC features
def features_extractor(file):
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
        
        # Extract MFCC features
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        
        return mfccs_scaled_features
    except Exception as e:
        # Handle errors (e.g., missing or corrupted files)
        print(f"Error processing file {file}: {e}")
        return None

# Function to process a single file
def process_file(filename, base_path):
    file_path = os.path.join(base_path, filename)
    features = features_extractor(file_path)
    return features

# Main function to extract features in parallel
def extract_features_parallel(df_train, base_path, n_jobs=4):
    # Use joblib to parallelize feature extraction
    extracted_features = Parallel(n_jobs=n_jobs)(
        delayed(process_file)(filename, base_path) for filename in tqdm(df['filename'])
    )
    
    # Filter out None values (failed extractions)
    extracted_features = [features for features in extracted_features if features is not None]
    
    return extracted_features

# Example usage
base_path = '/kaggle/input/birdclef-2025/train_audio'
extracted_features = extract_features_parallel(df, base_path, n_jobs=4)





import pandas as pd
import numpy as np

# Assuming extracted_features is a list of arrays, each with 40 MFCC features
extracted_features = np.array(extracted_features)  # Shape: (27755, 40)

# Create a DataFrame with 40 columns
extracted_features_df = pd.DataFrame(
    extracted_features,
    columns=[f'mfcc_{i+1}' for i in range(extracted_features.shape[1])]  # Column names: mfcc_1, mfcc_2, ..., mfcc_40
)

# Display the first few rows
extracted_features_df.head()


extracted_features_df['class_names'] = df['class_names']
extracted_features_df.to_csv("df_melfrequencies.csv" , index = False)


data = pd.read_csv("/kaggle/input/birdclef-2025-melfreq-data/df_melfrequencies.csv")
data.head()










