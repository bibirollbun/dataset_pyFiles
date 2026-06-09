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


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, lfilter
from scipy.stats import skew, kurtosis
import json # To handle potential JSON-like sensor data if it's embedded


# --- 1. Data Loading (Conceptual Placeholder) ---
# This section provides a conceptual outline for loading data.
# In a real competition, you would have specific file paths and formats
# (e.g., CSV, parquet, HDF5, JSON).

def load_sensor_data(data_path, sensor_type='all'):
    """
    Conceptual function to load sensor data.
    In a real scenario, this would parse actual files (e.g., CSVs per sequence_id).
    
    Args:
        data_path (str): Path to the directory containing sensor data files.
        sensor_type (str): 'all' to load all sensors, 'imu' to load only IMU data.
                           This simulates the competition's evaluation setup.
    
    Returns:
        dict: A dictionary where keys are sequence_ids and values are pandas DataFrames
              containing sensor readings over time for that sequence.
        
        Example structure (for a single sequence_id):
        {
            'sequence_id_1': pd.DataFrame({
                'time': [...],
                'acc_x': [...], 'acc_y': [...], 'acc_z': [...], # Accelerometer
                'gyro_x': [...], 'gyro_y': [...], 'gyro_z': [...], # Gyroscope
                'mag_x': [...], 'mag_y': [...], 'mag_z': [...], # Magnetometer (if available)
                'therm_0': [...], 'therm_1': [...], ..., 'therm_4': [...], # Thermopiles
                'tof_0': [...], 'tof_1': [...], ..., 'tof_4': [...] # Time-of-flight
            }),
            'sequence_id_2': ...
        }
    """
    print(f"Conceptual: Loading sensor data from {data_path} with sensor type '{sensor_type}'...")
    
    # --- Mock Data Generation ---
    # Since we don't have actual files, we generate some mock data to make
    # the subsequent feature engineering and model training conceptually viable.
    mock_data = {}
    num_sequences = 100
    sequence_length = 200 # Number of sensor readings per sequence

    imu_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    therm_cols = [f'therm_{i}' for i in range(5)]
    tof_cols = [f'tof_{i}' for i in range(5)]
    
    for i in range(num_sequences):
        sequence_id = f'seq_{i:03d}'
        df_cols = ['time'] + imu_cols
        if sensor_type == 'all':
            df_cols.extend(therm_cols)
            df_cols.extend(tof_cols)

        data = {col: np.random.rand(sequence_length) * (100 if 'therm' in col or 'tof' in col else 10) - 5
                for col in df_cols}
        data['time'] = np.arange(sequence_length) * 0.01 # Assuming 10ms sampling

        mock_data[sequence_id] = pd.DataFrame(data)
    
    print(f"Conceptual: Generated {num_sequences} mock sequences.")
    return mock_data

def load_labels(labels_path):
    """
    Conceptual function to load gesture labels.
    In a real scenario, this would load a CSV mapping sequence_id to gesture label.
    
    Args:
        labels_path (str): Path to the labels file (e.g., CSV).
    
    Returns:
        pd.DataFrame: A DataFrame with 'sequence_id' and 'gesture' columns.
        
        Example structure:
        pd.DataFrame({
            'sequence_id': ['seq_000', 'seq_001', ...],
            'gesture': ['Above ear - Pull hair', 'Drink from bottle/cup', ...]
        })
    """
    print(f"Conceptual: Loading labels from {labels_path}...")
    
    # --- Mock Label Generation ---
    bfrb_gestures = [
        'Above ear - Pull hair', 'Forehead - Pull hairline', 'Forehead - Scratch',
        'Eyebrow - Pull hair', 'Eyelash - Pull hair', 'Neck - Pinch skin',
        'Neck - Scratch', 'Cheek - Pinch skin'
    ]
    non_bfrb_gestures = [
        'Drink from bottle/cup', 'Glasses on/off', 'Pull air toward your face',
        'Pinch knee/leg skin', 'Scratch knee/leg skin', 'Write name on leg',
        'Text on phone', 'Feel around in tray and pull out an object',
        'Write name in air', 'Wave hello'
    ]
    all_gestures = bfrb_gestures + non_bfrb_gestures
    
    num_sequences = 100 # Match num_sequences in load_sensor_data
    mock_labels = {
        'sequence_id': [f'seq_{i:03d}' for i in range(num_sequences)],
        'gesture': np.random.choice(all_gestures, size=num_sequences)
    }
    
    print(f"Conceptual: Generated {num_sequences} mock labels.")
    return pd.DataFrame(mock_labels)


# --- 2. Feature Engineering ---
# This section defines how to extract meaningful features from raw sensor data.
# This is crucial for time-series data and often involves statistical measures,
# frequency domain analysis, and domain-specific insights.

class FeatureExtractor:
    def __init__(self, sample_rate=100.0):
        """
        Initializes the FeatureExtractor.
        Args:
            sample_rate (float): Sampling rate of the sensor data in Hz.
        """
        self.sample_rate = sample_rate
        self.imu_features = ['mean', 'std', 'max', 'min', 'range', 'median', 'rms', 'skew', 'kurtosis']
        self.freq_features = ['energy_bands', 'peak_freq'] # Conceptual: FFT-based features
        self.therm_tof_features = ['mean', 'std', 'diff_mean', 'diff_std'] # Statistical for therm/tof

    def _butter_lowpass(self, cutoff, order=5):
        nyq = 0.5 * self.sample_rate
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def _apply_filter(self, data, cutoff=5.0):
        b, a = self._butter_lowpass(cutoff)
        y = lfilter(b, a, data)
        return y

    def _extract_statistical_features(self, series, prefix):
        """Extracts basic statistical features from a time series."""
        features = {}
        features[f'{prefix}_mean'] = np.mean(series)
        features[f'{prefix}_std'] = np.std(series)
        features[f'{prefix}_max'] = np.max(series)
        features[f'{prefix}_min'] = np.min(series)
        features[f'{prefix}_range'] = np.max(series) - np.min(series)
        features[f'{prefix}_median'] = np.median(series)
        features[f'{prefix}_rms'] = np.sqrt(np.mean(series**2))
        
        # Avoid errors if series is too short or constant
        try:
            features[f'{prefix}_skew'] = skew(series)
        except:
            features[f'{prefix}_skew'] = 0.0 # Default if calculation fails
        try:
            features[f'{prefix}_kurtosis'] = kurtosis(series)
        except:
            features[f'{prefix}_kurtosis'] = 0.0 # Default if calculation fails

        return features

    def _extract_imu_features(self, df):
        """Extracts features from IMU data (accelerometer and gyroscope)."""
        imu_features = {}
        acc_cols = ['acc_x', 'acc_y', 'acc_z']
        gyro_cols = ['gyro_x', 'gyro_y', 'gyro_z']
        
        # Apply low-pass filter (optional, but common for noise reduction)
        for col in acc_cols + gyro_cols:
            if col in df.columns:
                df[f'{col}_filtered'] = self._apply_filter(df[col].values)
        
        # Magnitude features
        if all(col in df.columns for col in acc_cols):
            df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
            imu_features.update(self._extract_statistical_features(df['acc_mag'], 'acc_mag'))
        
        if all(col in df.columns for col in gyro_cols):
            df['gyro_mag'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
            imu_features.update(self._extract_statistical_features(df['gyro_mag'], 'gyro_mag'))

        # Features per axis
        for col in acc_cols + gyro_cols:
            if f'{col}_filtered' in df.columns: # Use filtered data for features
                imu_features.update(self._extract_statistical_features(df[f'{col}_filtered'], col))
            elif col in df.columns: # Fallback to unfiltered if no filtered version
                 imu_features.update(self._extract_statistical_features(df[col], col))

        # Conceptual: Add more advanced IMU features like orientation estimates,
        # integral of angular velocity, etc., if needed.
        return imu_features

    def _extract_therm_tof_features(self, df):
        """Extracts features from thermopile and time-of-flight data."""
        other_features = {}
        therm_cols = [f'therm_{i}' for i in range(5)]
        tof_cols = [f'tof_{i}' for i in range(5)]

        # Individual sensor statistics
        for col in therm_cols + tof_cols:
            if col in df.columns:
                other_features.update(self._extract_statistical_features(df[col], col))
        
        # Differences between sensors (e.g., for proximity/temperature gradients)
        if all(col in df.columns for col in therm_cols):
            diff_therm = np.diff(df[therm_cols].values, axis=1) # Differences between adjacent thermopiles
            other_features['therm_diff_mean'] = np.mean(diff_therm)
            other_features['therm_diff_std'] = np.std(diff_therm)
            
        if all(col in df.columns for col in tof_cols):
            diff_tof = np.diff(df[tof_cols].values, axis=1) # Differences between adjacent ToF sensors
            other_features['tof_diff_mean'] = np.mean(diff_tof)
            other_features['tof_diff_std'] = np.std(diff_tof)

        # Conceptual: More complex features like dynamic changes,
        # patterns over time for thermopiles (e.g., approaching hand),
        # or specific ToF patterns indicating a target object.
        return other_features

    def extract_features_for_sequence(self, sequence_df, sensor_type='all'):
        """
        Extracts all relevant features from a single sequence's sensor data.
        
        Args:
            sequence_df (pd.DataFrame): DataFrame for a single sequence.
            sensor_type (str): 'all' or 'imu' to determine which features to extract.
            
        Returns:
            dict: A dictionary of extracted features.
        """
        features = {}
        
        # IMU features are always extracted if data is present
        features.update(self._extract_imu_features(sequence_df.copy())) # Use copy to avoid modifying original DF
        
        # Add thermopile and ToF features if sensor_type is 'all'
        if sensor_type == 'all':
            features.update(self._extract_therm_tof_features(sequence_df.copy()))
        
        return features

    def extract_features_from_dataset(self, data_dict, sensor_type='all'):
        """
        Extracts features from an entire dataset (dictionary of DataFrames).
        
        Args:
            data_dict (dict): Dictionary of sequence_ids to DataFrames.
            sensor_type (str): 'all' or 'imu'.
            
        Returns:
            pd.DataFrame: A DataFrame where each row is a sequence_id
                          and columns are the extracted features.
        """
        all_features = []
        for seq_id, df in data_dict.items():
            features = self.extract_features_for_sequence(df, sensor_type=sensor_type)
            features['sequence_id'] = seq_id # Add sequence_id for merging later
            all_features.append(features)
        
        # Fill NaN values that might arise if some features couldn't be computed for short sequences.
        # A common strategy is to fill with 0 or the mean/median of the column.
        features_df = pd.DataFrame(all_features).fillna(0) # Or fillna(features_df.mean()) after computing
        return features_df




# --- 3. Model Definition (Conceptual Placeholder) ---
# This section defines a simple machine learning model.
# In a real scenario, you might use more complex models like LSTMs, Transformers,
# or sophisticated ensemble methods.

class BehaviorClassifier:
    def __init__(self, random_state=42):
        """
        Initializes the classifier.
        Using RandomForestClassifier as an example.
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
        self.scaler = StandardScaler()
        self.label_encoder = {} # To map string labels to integers

    def preprocess_features(self, X_df, fit_scaler=True):
        """
        Scales features.
        Args:
            X_df (pd.DataFrame): DataFrame of features.
            fit_scaler (bool): Whether to fit the scaler (True for training data).
        Returns:
            np.ndarray: Scaled features.
        """
        # Drop sequence_id if present, as it's not a feature
        X_numeric = X_df.drop(columns=['sequence_id'], errors='ignore')

        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X_numeric)
            # Store feature names seen during fit for consistent processing during transform
            self.fitted_feature_names = list(X_numeric.columns)
        else:
            # Align test features with the features seen during training
            missing_cols = set(self.fitted_feature_names) - set(X_numeric.columns)
            for col in missing_cols:
                X_numeric[col] = 0.0 # Fill missing columns (e.g., thermopile/ToF for IMU-only data) with zeros

            # Ensure the order of columns matches the training data
            X_aligned = X_numeric[self.fitted_feature_names]
            X_scaled = self.scaler.transform(X_aligned)
        return X_scaled

    def encode_labels(self, y_series, fit_encoder=True):
        """
        Encodes string labels to integers.
        Args:
            y_series (pd.Series): Series of string labels.
            fit_encoder (bool): Whether to fit the encoder (True for training data).
        Returns:
            np.ndarray: Integer-encoded labels.
        """
        if fit_encoder:
            unique_labels = sorted(y_series.unique())
            self.label_encoder = {label: i for i, label in enumerate(unique_labels)}
            self.inverse_label_encoder = {i: label for label, i in self.label_encoder.items()}
        
        return y_series.map(self.label_encoder).values

    def decode_predictions(self, y_pred_encoded):
        """
        Decodes integer predictions back to string labels.
        Args:
            y_pred_encoded (np.ndarray): Integer-encoded predictions.
        Returns:
            pd.Series: String labels.
        """
        return pd.Series(y_pred_encoded).map(self.inverse_label_encoder)

    def train(self, X_train_df, y_train_series):
        """
        Trains the classifier.
        Args:
            X_train_df (pd.DataFrame): Training features.
            y_train_series (pd.Series): Training labels.
        """
        X_train_scaled = self.preprocess_features(X_train_df, fit_scaler=True)
        y_train_encoded = self.encode_labels(y_train_series, fit_encoder=True)
        
        print("Conceptual: Training model...")
        self.model.fit(X_train_scaled, y_train_encoded)
        print("Conceptual: Model training complete.")

    def predict(self, X_test_df):
        """
        Makes predictions on new data.
        Args:
            X_test_df (pd.DataFrame): Test features.
        Returns:
            pd.Series: Predicted string labels.
        """
        X_test_scaled = self.preprocess_features(X_test_df, fit_scaler=False)
        y_pred_encoded = self.model.predict(X_test_scaled)
        return self.decode_predictions(y_pred_encoded)

    def predict_proba(self, X_test_df):
        """
        Makes probabilistic predictions.
        Args:
            X_test_df (pd.DataFrame): Test features.
        Returns:
            np.ndarray: Predicted probabilities for each class.
        """
        X_test_scaled = self.preprocess_features(X_test_df, fit_scaler=False)
        return self.model.predict_proba(X_test_scaled)


# --- 4. Evaluation Metric ---
# Implementation of the custom macro F1 score described in the competition.

def custom_f1_score(y_true, y_pred, bfrb_gestures):
    """
    Calculates the custom evaluation metric for the competition.
    
    The final score is the average of two components:
    1. Binary F1 on whether the gesture is one of the target (BFRB) or non-target (non-BFRB) types.
    2. Macro F1 on gesture, where all non-target sequences are collapsed into a single 'non_target' class.
    
    Args:
        y_true (pd.Series): True gesture labels.
        y_pred (pd.Series): Predicted gesture labels.
        bfrb_gestures (list): List of gesture names considered BFRB-like.
        
    Returns:
        float: The final custom F1 score.
    """
    
    # 1. Binary F1 for BFRB vs Non-BFRB
    # Create binary labels: 1 for BFRB, 0 for Non-BFRB
    y_true_binary = y_true.apply(lambda x: 1 if x in bfrb_gestures else 0)
    y_pred_binary = y_pred.apply(lambda x: 1 if x in bfrb_gestures else 0)
    
    binary_f1 = f1_score(y_true_binary, y_pred_binary, average='binary', pos_label=1)
    print(f"  Binary F1 (BFRB vs Non-BFRB): {binary_f1:.4f}")

    # 2. Macro F1 for specific gestures, collapsing non-targets
    y_true_collapsed = y_true.apply(lambda x: x if x in bfrb_gestures else 'non_target')
    y_pred_collapsed = y_pred.apply(lambda x: x if x in bfrb_gestures else 'non_target')
    
    # Ensure all labels in y_true_collapsed are present in y_pred_collapsed's unique values
    # to avoid errors if a predicted class is missing.
    # We pass `labels` explicitly to f1_score to handle cases where a class might not appear in y_pred.
    all_possible_collapsed_classes = sorted(list(set(y_true_collapsed.unique()).union(set(y_pred_collapsed.unique()))))

    macro_f1_collapsed = f1_score(y_true_collapsed, y_pred_collapsed, 
                                  average='macro', labels=all_possible_collapsed_classes, zero_division=0)
    print(f"  Macro F1 (Collapsed Gestures): {macro_f1_collapsed:.4f}")

    final_score = (binary_f1 + macro_f1_collapsed) / 2
    print(f"  Final Custom F1 Score: {final_score:.4f}")
    
    return final_score


# --- Main Execution Flow (Conceptual) ---

if __name__ == "__main__":
    print("Conceptual CMI Behavior Detection Model")
    print("---------------------------------------")
    print("NOTE: This code is conceptual and uses mock data.")
    print("It requires actual sensor data and labels to run meaningfully.")
    
    # Define BFRB gestures for evaluation
    bfrb_like_gestures = [
        'Above ear - Pull hair', 'Forehead - Pull hairline', 'Forehead - Scratch',
        'Eyebrow - Pull hair', 'Eyelash - Pull hair', 'Neck - Pinch skin',
        'Neck - Scratch', 'Cheek - Pinch skin'
    ]


    # --- Step 1: Load Data ---
    # Replace with actual data loading paths and logic in a real scenario
    mock_sensor_data = load_sensor_data(data_path="/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv", sensor_type='all')
    mock_labels = load_labels(labels_path="/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")

    # Align data and labels based on sequence_id
    # In a real scenario, you'd ensure consistent sequence_ids.
    available_sequences = list(mock_sensor_data.keys())
    mock_labels = mock_labels[mock_labels['sequence_id'].isin(available_sequences)].set_index('sequence_id')
    
    # Split data into training and validation sets
    train_seq_ids, val_seq_ids = train_test_split(
        list(mock_sensor_data.keys()), test_size=0.2, random_state=42
    )

    X_train_raw = {seq_id: mock_sensor_data[seq_id] for seq_id in train_seq_ids}
    X_val_raw = {seq_id: mock_sensor_data[seq_id] for seq_id in val_seq_ids}

    y_train = mock_labels.loc[train_seq_ids]['gesture']
    y_val = mock_labels.loc[val_seq_ids]['gesture']

    print(f"\nTraining sequences: {len(X_train_raw)}")
    print(f"Validation sequences: {len(X_val_raw)}")


    # --- Step 2: Feature Engineering ---
    print("\nExtracting features...")
    feature_extractor = FeatureExtractor(sample_rate=100.0) # Assuming 100 Hz sampling rate
    
    X_train_features = feature_extractor.extract_features_from_dataset(X_train_raw, sensor_type='all')
    X_val_features = feature_extractor.extract_features_from_dataset(X_val_raw, sensor_type='all')

    # Ensure feature DataFrames are aligned with labels based on sequence_id
    X_train_features = X_train_features.set_index('sequence_id').loc[y_train.index].reset_index()
    X_val_features = X_val_features.set_index('sequence_id').loc[y_val.index].reset_index()
    
    print(f"Training features shape: {X_train_features.shape}")
    print(f"Validation features shape: {X_val_features.shape}")


    # --- Step 3: Model Training ---
    print("\nInitializing and training the classifier...")
    classifier = BehaviorClassifier()
    classifier.train(X_train_features, y_train)


    # --- Step 4: Make Predictions on Validation Set ---
    print("\nMaking predictions on the validation set...")
    y_val_pred = classifier.predict(X_val_features)


    # --- Step 5: Evaluate Model ---
    print("\nEvaluating model performance on validation set:")
    validation_score = custom_f1_score(y_val, y_val_pred, bfrb_like_gestures)
    print(f"\nFinal Validation Score: {validation_score:.4f}")


    # --- Conceptual Test Set Prediction (simulating submission) ---
    print("\n--- Conceptual Test Set Prediction ---")
    print("In a real competition, you would receive test data in two phases:")
    print("1. Half with IMU data only.")
    print("2. Half with all sensor data (IMU, thermopile, ToF).")
    print("Your model needs to handle both scenarios.")


    # Mock test data (IMU only)
    mock_test_imu_data = load_sensor_data(data_path="/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv", sensor_type='imu')
    test_imu_features = feature_extractor.extract_features_from_dataset(mock_test_imu_data, sensor_type='imu')
    test_imu_predictions = classifier.predict(test_imu_features)
    print(f"Conceptual: Predicted for {len(test_imu_predictions)} IMU-only test sequences.")
    print(test_imu_predictions.head()) # Uncomment to see sample predictions


    # Mock test data (All sensors)
    mock_test_all_data = load_sensor_data(data_path="/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv", sensor_type='all')
    test_all_features = feature_extractor.extract_features_from_dataset(mock_test_all_data, sensor_type='all')
    test_all_predictions = classifier.predict(test_all_features)
    print(f"Conceptual: Predicted for {len(test_all_predictions)} all-sensor test sequences.")
    print(test_all_predictions.head()) # Uncomment to see sample predictions

    print("\n--- Submission File Format (Conceptual) ---")
    print("You would typically combine these predictions into a single DataFrame")
    print("and save it to a CSV file in the required format (sequence_id, gesture).")


    # Example of conceptual submission file structure
    all_test_predictions = pd.concat([
         pd.DataFrame({'sequence_id': test_imu_features['sequence_id'], 'gesture': test_imu_predictions}),
         pd.DataFrame({'sequence_id': test_all_features['sequence_id'], 'gesture': test_all_predictions})
     ])
    all_test_predictions.to_csv("submission.parquet", index=False)
    print("\nConceptual: 'submission.parquet' would be created.")

