# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import os
import librosa
import numpy as np
import glob
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.decomposition import PCA
import librosa.display
import pandas as pd
from tqdm.auto import tqdm
from functools import lru_cache
import concurrent.futures
import warnings
import lightgbm as lgb
from xgboost import XGBClassifier
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score


train_audio_dir = '/kaggle/input/birdclef-2025/train_audio/'
train, taxonomy = pd.read_csv('/kaggle/input/birdclef-2025/train.csv'), pd.read_csv('/kaggle/input/birdclef-2025/taxonomy.csv')
submission =  pd.read_csv('/kaggle/input/birdclef-2025/sample_submission.csv')
submission.head()


selected_features = ['primary_label','filename','latitude','longitude','scientific_name','common_name']
train = train[selected_features]
# Define your unique labels
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

# Create a dictionary to store DataFrames
dfs = {label: train[train['primary_label'] == label] for label in labels}


%%time

# Suppress librosa warnings that waste time printing
warnings.filterwarnings('ignore')

def extract_low_amplitude_regions(stft, threshold_db=-40):
    # Take the absolute value of the spectrogram
    magnitude = np.abs(stft)
    
    # Convert to power (energy) spectrogram
    power_spectrogram = np.abs(magnitude) ** 2
    
    # Convert to decibels
    db_magnitude = librosa.power_to_db(power_spectrogram)
    
    # Create mask for low amplitude regions
    mask = db_magnitude > threshold_db
    
    # Replace low amplitude regions with a very low value
    low_amp_spectrogram = np.where(mask, db_magnitude, -np.inf)
    
    return mask, power_spectrogram  # Return power_spectrogram directly for efficiency

# Add caching to expensive function calls
@lru_cache(maxsize=128)
def cached_load(filename_key, sr=32000, duration=5):
    """Cache-friendly loader that uses a string key"""
    return librosa.load(filename_key, sr=sr, duration=duration)

def extract_features_fast(y, sr):
    """Feature extraction with low amplitude region detection"""
    # Skip HPSS which is very expensive (still using raw signal)
    y_percussive = y
    
    # Compute spectrogram (using faster n_fft if possible)
    n_fft = min(2048, len(y)//4)  # Smaller FFT is faster
    stft = librosa.stft(y_percussive, n_fft=n_fft, hop_length=n_fft//4)
    
    # Apply low amplitude region extraction (integrated into pipeline)
    mask, power_spectrogram = extract_low_amplitude_regions(stft)
    
    # Extract features using the power spectrogram from low amplitude extraction
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(
        S=power_spectrogram, sr=sr), axis=1)
    
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(
        S=power_spectrogram, sr=sr), axis=1)
    
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(
        y_percussive), axis=1)

    # More efficient concatenation (still using the three features requested)
    return np.concatenate([spectral_centroid, spectral_rolloff, zero_crossing_rate])

def process_file(args):
    """Standalone function for parallel processing"""
    filename, label = args
    try:
        y, sr = cached_load(filename, sr=32000, duration=5)
        features = extract_features_fast(y, sr)
        return (features, label)
    except Exception as e:
        # Print error for debugging (needed in Kaggle)
        print(f"Error processing {filename}: {str(e)}")
        return None

def prepare_data_optimized_parallel(dfs, labels, train_audio_dir, max_files_per_directory=50, n_components=64):
    # Prepare arguments for parallel processing
    all_args = []
    for label in labels:
        df = dfs[label]
        n_files = min(max_files_per_directory, len(df))
        filenames = [os.path.join(train_audio_dir, df.iloc[i]['filename']) for i in range(n_files)]
        all_args.extend([(filename, label) for filename in filenames])
    
    # For Kaggle - don't use too many workers
    # Kaggle typically has 2-4 CPU cores available
    num_workers = min(2, os.cpu_count() or 1)
    
    # Display total number of files
    print(f"Processing {len(all_args)} files with {num_workers} workers")
    
    # Process in parallel - but with a smaller chunksize for Kaggle
    x_p = []
    y = []
    
    # Use ThreadPoolExecutor instead of ProcessPoolExecutor for Kaggle compatibility
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(process_file, all_args, chunksize=5), 
            total=len(all_args),
            desc="Extracting features",
            unit="file"
        ))
    
    # Filter out None results (errors) and unpack features and labels
    results = [r for r in results if r is not None]
    if results:
        x_p, y = zip(*results)
    
    # Convert to numpy arrays
    if not x_p:
        print("\nWARNING: No audio files were successfully processed")
        return np.array([]), np.array([]), None, None
    
    x_p = np.array(x_p)
    y = np.array(y)
    
    print(f"\nTotal processed data shape: x_p={x_p.shape}, y={y.shape}")
    
    # Fit and transform with scaler
    scaler_p = StandardScaler()
    x_p_scaled = scaler_p.fit_transform(x_p)
    
    # Fit and transform with PCA (with minimal valid components)
    valid_n_components = min(n_components, *x_p_scaled.shape)
    if valid_n_components <= 0:
        return x_p_scaled, y, scaler_p, None
    
    pca_p = PCA(n_components=valid_n_components)
    x_p_embedded = pca_p.fit_transform(x_p_scaled)
    
    return x_p_embedded, y, scaler_p, pca_p


# Example usage in Kaggle
from sklearn.model_selection import train_test_split

# Assuming dfs, labels, train_audio_dir are already defined
X, y, scaler, pca = prepare_data_optimized_parallel(
    dfs, 
    labels, 
    train_audio_dir, 
    max_files_per_directory=40, 
    n_components=20
)

# Split and model training as usual
X_train_val, test_data_features, y_train_val, y_test_final = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Added stratify for balanced classes


%%time
test_soundscapes = glob.glob('/kaggle/input/birdclef-2025/train_soundscapes/*')

# Define efficient processing function with caching
@lru_cache(maxsize=128)
def cached_load(filename_key, sr=32000, duration=5):
    """Cache-friendly loader that uses a string key"""
    return librosa.load(filename_key, sr=sr, duration=duration)

def process_file(filename):
    """Process a single file efficiently"""
    try:
        # Use cached loading
        y, sr = cached_load(filename, sr=32000, duration=5)
        
        # Use our optimized feature extraction
        y_percussive = y  # Skip HPSS
        
        # Compute spectrogram efficiently
        n_fft = min(2048, len(y)//4)
        stft = librosa.stft(y_percussive, n_fft=n_fft, hop_length=n_fft//4)
        
        # Apply low amplitude region extraction
        magnitude = np.abs(stft)
        power_spectrogram = np.abs(magnitude) ** 2
        db_magnitude = librosa.power_to_db(power_spectrogram)
        mask = db_magnitude > -40  # threshold_db
        
        # Extract features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(
            S=power_spectrogram, sr=sr), axis=1)
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(
            S=power_spectrogram, sr=sr), axis=1)
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(
            y_percussive), axis=1)
        
        # Create feature vector
        features = np.concatenate([
            spectral_centroid, spectral_rolloff, zero_crossing_rate
        ])
        
        return filename, features
    except Exception as e:
        print(f"Error processing file {filename}: {str(e)}")
        return None

def process_files_parallel(files, max_workers=2):
    """Process files in parallel with better error handling"""
    data_features = []
    data_filename = []
    
    # Process in parallel - optimized for Kaggle
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_file, files),
            total=len(files),
            desc='Processing test files'
        ))
    
    # Filter out None results from failures
    results = [r for r in results if r is not None]
    
    # Unpack results
    if results:
        filenames, features = zip(*results)
        data_filename.extend(filenames)
        data_features.extend(features)
    
    return np.array(data_filename), np.array(data_features)

# Process test files in chunks
max_workers = min(2, os.cpu_count() or 1)
print(f"Processing test files with {max_workers} workers")

n = len(test_soundscapes)
files1 = test_soundscapes[:n//3]
files2 = test_soundscapes[n//3:2*n//3]
files3 = test_soundscapes[2*n//3:]

# Process each chunk in parallel
test_data_filename1, test_data_features1 = process_files_parallel(files1, max_workers=max_workers)
test_data_filename2, test_data_features2 = process_files_parallel(files2, max_workers=max_workers)
test_data_filename3, test_data_features3 = process_files_parallel(files3, max_workers=max_workers)

# Combine results
test_data_filename = np.concatenate([test_data_filename1, test_data_filename2, test_data_filename3])
test_data_features = np.concatenate([test_data_features1, test_data_features2, test_data_features3])

print(f"Combined test data: {len(test_data_filename)} files processed")


%%time
# Encode labels (ensure LabelEncoder is imported)
from sklearn.metrics import log_loss


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

print(f"Shapes after split: X_train={X_train.shape}, X_val={X_val.shape}, test_data_features={test_data_features.shape}")
print(f"Shapes after split: y_train={y_train.shape}, y_val={y_val.shape}, y_test_final={y_test_final.shape}")

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
bird_labels = list(label_encoder.classes_)

# Define the full set of encoded labels for log_loss
num_classes = len(bird_labels)
full_labels = list(range(num_classes))  # Corresponds to encoded labels [0, 1, ..., num_classes-1]
print(f"Number of classes in bird_labels: {len(bird_labels)}")

# Encode validation labels
y_val_encoded = label_encoder.transform(y_val)

# Apply the same transformations to test data that were used on training data
# Check if test_data_features is empty before transformation
test_vectors = None
if test_data_features.shape[0] > 0:
    print("Processing test data features...")
    if pca is not None:
        # Apply the same scaling (scaler should be fitted on X_train or X_train_val)
        test_data_scaled = scaler.transform(test_data_features)
        # Apply the same PCA transformation (pca should be fitted on scaled X_train or X_train_val)
        test_vectors = pca.transform(test_data_scaled)
    else:
        # Only scale without PCA (scaler should be fitted on X_train or X_train_val)
        test_vectors = scaler.transform(test_data_features)
else:
    print("WARNING: test_data_features is empty (shape: {}). Skipping test data transformation and predictions.".format(test_data_features.shape))

# Train LightGBM model
print("\nTraining LightGBM model...")
lgb_params = {
    'objective': 'multiclass',
    'num_class': len(bird_labels),
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'n_jobs': -1,  # Use all available cores
    'seed': 42  # for reproducibility
}

lgb_train = lgb.Dataset(X_train, y_train_encoded)
lgb_val = lgb.Dataset(X_val, y_val_encoded, reference=lgb_train)

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=5000,  # Increased rounds, relying on early stopping
    valid_sets=[lgb_train, lgb_val],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]  # Increased stopping rounds, make verbose=False cleaner
)

print(f"LightGBM best iteration: {lgb_model.best_iteration}")

# Train XGBoost model
print("\nTraining XGBoost model...")
xgb_model = XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=5000,  # Increased estimators, relying on early stopping
    use_label_encoder=False,  # Still include for potential older versions
    verbosity=1,
    n_jobs=-1,  # Use all available cores
    random_state=42  # for reproducibility
)

# Using y_val_encoded for evaluation set
xgb_model.fit(
    X_train, y_train_encoded,
    eval_set=[(X_val, y_val_encoded)],
    early_stopping_rounds=100,  # Increased early stopping rounds
    verbose=False  # Make verbose=False for cleaner output during fit
)

print(f"XGBoost best iteration: {xgb_model.best_iteration}")

# Get predictions on validation set
lgb_val_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
xgb_val_pred = xgb_model.predict_proba(X_val)

# Debugging shapes and unique classes before log_loss
print(f"Number of unique classes in y_train_encoded: {len(np.unique(y_train_encoded))}")
print(f"Number of unique classes in y_val_encoded: {len(np.unique(y_val_encoded))}")
print(f"Shape of lgb_val_pred: {lgb_val_pred.shape}")
print(f"Shape of xgb_val_pred: {xgb_val_pred.shape}")
print(f"Length of full_labels: {len(full_labels)}")

# Calculate and print validation metrics with labels parameter
print(f"\nLightGBM validation log loss: {log_loss(y_val_encoded, lgb_val_pred, labels=full_labels):.4f}")
print(f"XGBoost validation log loss: {log_loss(y_val_encoded, xgb_val_pred, labels=full_labels):.4f}")

# Get predictions on test set only if test_vectors is not None (i.e., test_data_features was not empty)
lgb_probabilities = None
xgb_probabilities = None
blended_probabilities = None
if test_vectors is not None and test_vectors.shape[0] > 0:
    print("Generating predictions for test set...")
    lgb_probabilities = lgb_model.predict(test_vectors, num_iteration=lgb_model.best_iteration)
    xgb_probabilities = xgb_model.predict_proba(test_vectors)  # predict_proba automatically uses best iteration with early stopping
    # Blend predictions (simple average)
    blended_probabilities = (lgb_probabilities + xgb_probabilities) / 2
else:
    print("WARNING: No test data available for predictions. Skipping test set predictions and submission creation.")

# --------------------- STEP 4: CREATE SUBMISSION ---------------------
# Only proceed with submission if test data is available
if test_vectors is not None and test_vectors.shape[0] > 0:
    # Function to create row_id from filenames (ensure test_data_filename is defined)
    def create_row_id(filename):
        parts = filename.split('/')[-1].split('.')[0].split('_')
        # Handle potential variations or ensure consistent format
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return filename  # Fallback (less likely needed for competition data)

    # Create row_ids from the test filenames
    # Assuming test_data_filename is a list or pandas Series matching test_data_features rows
    test_df = pd.DataFrame({'filename': test_data_filename})
    test_df['row_id'] = test_df['filename'].apply(create_row_id)

    # Create a new DataFrame with just row_ids
    submission = pd.DataFrame({'row_id': test_df['row_id']})

    # Add probability columns for each bird class
    for i, bird_label in enumerate(bird_labels):
        submission[bird_label] = blended_probabilities[:, i]

    # Display the head of the submission DataFrame for verification
    print("\nBlended Submission Head:")
    print(submission.head())

    # Save the submission to a CSV file
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission file 'blended_submission.csv' has been created.")

    # Also save individual model predictions (optional)
    lgb_submission = pd.DataFrame({'row_id': test_df['row_id']})
    xgb_submission = pd.DataFrame({'row_id': test_df['row_id']})

    for i, bird_label in enumerate(bird_labels):
        lgb_submission[bird_label] = lgb_probabilities[:, i]
        xgb_submission[bird_label] = xgb_probabilities[:, i]

    #lgb_submission.to_csv('submission.csv', index=False)
    #xgb_submission.to_csv('submission.csv', index=False)
    print("Individual model submissions also saved.")
else:
    print("WARNING: Submission files not created due to empty test data.")




