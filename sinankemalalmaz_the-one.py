import numpy as np
import pandas as pd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
import sklearn # Import sklearn itself to get its version

# Print versions for reproducibility and debugging
print(f"numpy version: {np.__version__}")
print(f"pandas version: {pd.__version__}")
print(f"librosa version: {librosa.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")



# IMPORTANT: Adjust BASE_INPUT_PATH if your data is not in the same directory as your notebook.
# On Kaggle, it's typically '../input/birdclef-2025/'. Locally, it might be './data/' or similar.
BASE_INPUT_PATH = '/kaggle/input/birdclef-2025/'  # Updated for Kaggle environment

# Construct full paths to specific data directories
TRAIN_AUDIO_PATH = os.path.join(BASE_INPUT_PATH, 'train_audio')
TEST_SOUNDSCAPES_PATH = os.path.join(BASE_INPUT_PATH, 'test_soundscapes')
TRAIN_SOUNDSCAPES_PATH = os.path.join(BASE_INPUT_PATH, 'train_soundscapes') # Unlabeled data

# Define constants for audio processing and feature extraction
SR = 32000 
DURATION = 5  
N_MFCC = 20   

# Print the defined paths and constants for verification
print("File paths and constants defined:")
print(f"  Base input path: {BASE_INPUT_PATH}")
print(f"  Train audio path: {TRAIN_AUDIO_PATH}")
print(f"  Test soundscapes path: {TEST_SOUNDSCAPES_PATH}")
print(f"  Train soundscapes path: {TRAIN_SOUNDSCAPES_PATH}")
print(f"Audio Processing Parameters:")
print(f"  Sample Rate (SR): {SR} Hz")
print(f"  Snippet Duration (DURATION): {DURATION} seconds")
print(f"  Number of MFCCs (N_MFCC): {N_MFCC}")



# These files provide information about the training audio, species, and submission format.

train_metadata_df = pd.read_csv(os.path.join(BASE_INPUT_PATH, 'train.csv'))
taxonomy_df = pd.read_csv(os.path.join(BASE_INPUT_PATH, 'taxonomy.csv'))
sample_submission_df = pd.read_csv(os.path.join(BASE_INPUT_PATH, 'sample_submission.csv'))
print("Metadata loaded successfully from specified paths.")


# Display the first few rows of each DataFrame to verify successful loading and inspect data structure.
print("\n--- train_metadata_df (first 5 rows) ---")
print(train_metadata_df.head())
print(f"\nShape of train_metadata_df: {train_metadata_df.shape}")

print("\n--- taxonomy_df (first 5 rows) ---")
print(taxonomy_df.head())
print(f"\nShape of taxonomy_df: {taxonomy_df.shape}")

print("\n--- sample_submission_df (first 5 rows) ---")
print(sample_submission_df.head())
print(f"\nShape of sample_submission_df: {sample_submission_df.shape}")


# This list will define the target columns for our multi-label classification.

initial_species_candidates = []

# Strategy: Prioritize getting the species list from sample_submission.csv columns.
if 'row_id' in sample_submission_df.columns:
    # Exclude 'row_id' as it's not a species column
    initial_species_candidates = [col for col in sample_submission_df.columns if col != 'row_id']
    print("Strategy: Species list successfully derived from sample_submission.csv columns.")
elif 'species_code' in taxonomy_df.columns:
    # Fallback 1: If sample_submission didn't provide species columns, use taxonomy.csv
    initial_species_candidates = taxonomy_df['species_code'].unique().tolist()
    print("Strategy: Species list derived from taxonomy.csv 'species_code' column (fallback).")
elif 'primary_label' in train_metadata_df.columns:
    # Fallback 2: If neither of the above worked, use primary_label from train.csv
    initial_species_candidates = train_metadata_df['primary_label'].unique().tolist()
    print("Strategy: Species list derived from train.csv 'primary_label' column (secondary fallback).")
else:
    # Ultimate fallback: If no species columns found, initialize as empty.
    # This will trigger the hardcoded dummy species later if still empty.
    initial_species_candidates = []
    print("Warning: No clear source for species list found in standard dataframes.")


# Clean and finalize ALL_SPECIES list:

ALL_SPECIES = sorted(list(set(str(s).strip() for s in initial_species_candidates if pd.notna(s) and str(s).strip())))

# Critical check: If after all attempts, the species list is still empty,

if not ALL_SPECIES:
    ALL_SPECIES = ['dummy_species_1', 'dummy_species_2', 'dummy_species_3']
    print("CRITICAL WARNING: ALL_SPECIES list is empty after all attempts. Using hardcoded dummy species.")
    print("Please verify your data files and their contents.")

# Create a set version of ALL_SPECIES for efficient membership checking later (e.g., when parsing secondary labels).
ALL_SPECIES_SET = set(ALL_SPECIES)

# Print summary information about the identified species list
print(f"\n--- Species List Summary ---")
print(f"Total unique species identified for prediction: {len(ALL_SPECIES)}")
print(f"First 10 species in the list: {ALL_SPECIES[:10]}")
print(f"Last 10 species in the list: {ALL_SPECIES[-10:]}")



def extract_features(file_path, sr=SR, duration=DURATION, n_mfcc=N_MFCC):

    try:
        # Load the audio file. 'sr=sr' resamples the audio to our target sample rate.

        y, current_sr = librosa.load(file_path, sr=sr, mono=True)

        # Ensure the audio snippet has a fixed length.

        target_len_samples = duration * sr
        if len(y) > target_len_samples:
            # If audio is longer than target duration, truncate it.
            y = y[:target_len_samples]
        else:
            # If audio is shorter, pad with zeros to reach the target duration.

            y = np.pad(y, (0, target_len_samples - len(y)), 'constant')


        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)


        return np.mean(mfccs.T, axis=0)

    except Exception as e:
        # Handle any errors during audio processing (e.g., corrupted file, invalid format).
        # Print an error message and return an array of zeros to maintain consistent feature dimensions.
        print(f"Error processing {file_path}: {e}")
        return np.zeros(n_mfcc)

print("Feature extraction function 'extract_features' defined.")



from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

print("Preparing data for model training...")

features_list = [] # To store the extracted features for each audio file
labels_list = []   # To store the corresponding labels (species codes) for each audio file

# Helper function to safely parse and clean secondary_labels
# This handles the string representation of lists and filters for valid species
def parse_and_clean_labels(label_str_list_repr, all_species_set):
    """
    Parses a string representation of a list of labels and cleans them.

    Args:
        label_str_list_repr (str): A string that might represent a Python list (e.g., "['speciesA', 'speciesB']").
        all_species_set (set): A set of all known valid species codes for filtering.

    Returns:
        list: A list of cleaned and validated species codes. Returns an empty list if parsing fails
              or no valid labels are found.
    """
    if isinstance(label_str_list_repr, str) and label_str_list_repr.startswith('[') and label_str_list_repr.endswith(']'):
        try:
            # Safely evaluate the string as a Python literal (list).
            # Using ast.literal_eval is safer than eval() for untrusted input,
            # but for competition data, eval() is often used for convenience.
            # Here, we'll use eval() as it's common in Kaggle notebooks for this specific format.
            parsed_list = eval(label_str_list_repr)
            # Ensure parsed_list is actually a list before iterating
            if not isinstance(parsed_list, list):
                return []
            # Clean each label and keep only those present in our ALL_SPECIES_SET
            return [str(s).strip() for s in parsed_list if str(s).strip() and str(s).strip() in all_species_set]
        except Exception as e:
            # print(f"Warning: Could not parse secondary_labels '{label_str_list_repr}': {e}")
            return []
    return [] # Return empty list if input is not a string list representation

# --- Data Processing Loop ---
# IMPORTANT CHANGE: Now processing the FULL dataset for better performance.
# This will take significantly longer than processing a small subset.
MAX_FILES_TO_PROCESS = len(train_metadata_df) # Process all available training files

processed_count = 0
# Iterate through the entire training metadata DataFrame
for index, row in train_metadata_df.iterrows(): # Removed .head(MAX_FILES_TO_PROCESS)
    file_name = row['filename']
    primary_label = str(row['primary_label']).strip() # Ensure primary label is clean string

    current_file_labels = set() # Use a set to collect unique labels for the current file

    # Add primary label if it's valid and in our ALL_SPECIES_SET
    if primary_label and primary_label in ALL_SPECIES_SET:
        current_file_labels.add(primary_label)

    # Process secondary labels
    secondary_labels_str = row.get('secondary_labels', '[]') # Get secondary labels, default to '[]' if missing
    parsed_secondary_labels = parse_and_clean_labels(secondary_labels_str, ALL_SPECIES_SET)
    for sec_label in parsed_secondary_labels:
        current_file_labels.add(sec_label) # Add cleaned and validated secondary labels

    # Construct the full path to the audio file
    full_audio_path = os.path.join(TRAIN_AUDIO_PATH, file_name)

    # Extract features for the current audio file
    if os.path.exists(full_audio_path):
        current_features = extract_features(full_audio_path)
    else:
        # If audio file not found, print a warning and return a zero feature vector.
        # This is important for robustness, especially with large datasets.
        print(f"Warning: Audio file not found at {full_audio_path}. Using zero features.")
        current_features = np.zeros(N_MFCC) # Use N_MFCC as the feature dimension

    # Only add features and labels if there's at least one valid label.
    # This prevents training on samples with no relevant species information.
    if current_file_labels:
        features_list.append(current_features)
        labels_list.append(list(current_file_labels)) # Convert set to list for MultiLabelBinarizer
    else:
        print(f"Skipping {file_name}: No valid primary or secondary labels found.")

    processed_count += 1
    if processed_count % 1000 == 0: # Print progress every 1000 files now
        print(f"Processed {processed_count}/{MAX_FILES_TO_PROCESS} files...")

print(f"\nFinished processing {processed_count} files for feature extraction.")

# Convert lists of features and labels into NumPy arrays
X = np.array(features_list)

# Initialize MultiLabelBinarizer with our comprehensive list of ALL_SPECIES
mlb = MultiLabelBinarizer(classes=ALL_SPECIES)
# Transform the list of lists of labels into a binary matrix (one-hot encoded for multi-label)
y_transformed = mlb.fit_transform(labels_list)

# Scale the features using StandardScaler
# This is important for many ML algorithms (e.g., Logistic Regression, SVM, Neural Networks)
# to ensure features with larger values don't dominate the learning process.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and validation sets
# This allows us to evaluate model performance on unseen data during development.
# test_size=0.2 means 20% of the data will be used for validation.
# random_state for reproducibility of the split.
if X_scaled.shape[0] > 1: # Ensure there's enough data to split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_transformed, test_size=0.2, random_state=42)
    print("\nData split into training and validation sets.")
else:
    # If not enough data for splitting (e.g., only 1 file processed),
    # use the entire dataset as training and create dummy validation sets.
    X_train, y_train = X_scaled, y_transformed
    X_val, y_val = X_scaled.copy(), y_transformed.copy() # Use copies to avoid modifying original
    print("Warning: Insufficient data for train-validation split. Using all data for training and dummy validation.")


# Print the shapes of the resulting datasets to verify their dimensions
print(f"\nShape of X (all features): {X.shape}")
print(f"Shape of y (all transformed labels): {y_transformed.shape}")
print(f"Shape of X_scaled (all scaled features): {X_scaled.shape}")
print(f"Shape of X_train: {X_train.shape}, Shape of y_train: {y_train.shape}")
print(f"Shape of X_val: {X_val.shape}, Shape of y_val: {y_val.shape}")
print(f"Number of features (MFCCs): {X_train.shape[1]}")
print(f"Number of output classes (species): {y_train.shape[1]}")



from sklearn.metrics import roc_auc_score # make_scorer is no longer needed as RandomizedSearchCV is removed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

# RandomizedSearchCV, uniform, randint are no longer needed as manual comparison is used
from sklearn.model_selection import train_test_split # Still needed for X_val, y_val split in Cell 6

import warnings
import sqlite3

# ONLY Suppress the OperationalError related to readonly database and history writing.
warnings.filterwarnings('ignore', category=UserWarning, message='.*attempt to write a readonly database.*')
warnings.filterwarnings('ignore', category=UserWarning, message='History will not be written to the database.')
warnings.filterwarnings('ignore', category=UserWarning, module='IPython.core.history')
warnings.filterwarnings('ignore', category=UserWarning, module='sqlite3')


print("Starting model training and hyperparameter comparison setup...")

# --- Helper function for safe AUC calculation ---
def calculate_safe_auc(y_true, y_pred_proba, class_names):
    """
    Calculates the macro-averaged ROC AUC score, safely handling classes
    that do not have both positive and negative samples in y_true.
    """
    class_auc_scores = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            try:
                class_auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
                class_auc_scores.append(class_auc)
            except ValueError as ve:
                pass # Skip this class for AUC calculation

    if not class_auc_scores:
        print("  No classes were eligible for ROC AUC calculation. Returning 0.0.")
        return 0.0
    return np.mean(class_auc_scores)

# Dictionaries to store trained models and their scores
models = {}
validation_auc_scores = {}
training_auc_scores = {}
best_hyperparameters = {} # To store the best params found for each model type

print("\n--- Model Training and Evaluation Summary (Best Hyperparameters) ---")
print("---------------------------------------------")
print(f"{'Model':<25} | {'Training AUC':<15} | {'Validation AUC':<15} | {'Best Params':<50}")
print("---------------------------------------------")
for model_name in models.keys():
    train_auc = training_auc_scores.get(model_name, 'N/A')
    val_auc = validation_auc_scores.get(model_name, 'N/A')
    
    # Retrieve the best params from the best_hyperparameters dictionary
    params_to_display = str(best_hyperparameters.get(model_name, 'N/A'))

    train_auc_str = f"{train_auc:.4f}" if isinstance(train_auc, float) else str(train_auc)
    val_auc_str = f"{val_auc:.4f}" if isinstance(val_auc, float) else str(val_auc)

    print(f"{model_name:<25} | {train_auc_str:<15} | {val_auc_str:<15} | {params_to_display:<50}")
print("---------------------------------------------")


# Find the best overall model based on validation AUC
if validation_auc_scores:
    best_overall_model_name = max(validation_auc_scores, key=validation_auc_scores.get)
    print(f"\nBest overall performing model on validation set: {best_overall_model_name} (AUC: {validation_auc_scores[best_overall_model_name]:.4f})")
else:
    print("\nNo models were trained or evaluated successfully.")



import warnings

# Suppress ALL warnings in this specific cell, with more targeted filters for common sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.multiclass')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.model_selection._validation')
warnings.filterwarnings('ignore', category=UserWarning, message='Only one class present in y_true')
warnings.filterwarnings('ignore', category=UserWarning, message='Label not .* is present in all training examples')

print("\n--- Tuning and Training Logistic Regression (OneVsRest) ---")

# Define three different sets of hyperparameters to compare manually
log_reg_param_sets = [
    # Set 1: Good all-rounder, balanced for imbalanced classes
    {'solver': 'liblinear', 'C': 1.0, 'class_weight': 'balanced', 'max_iter': 2000},
    # Set 2: Slightly less regularization, different solver, no class weight
    {'solver': 'saga', 'C': 0.7, 'class_weight': None, 'max_iter': 2000},
    # Set 3: More regularization, different solver, balanced class weight
    {'solver': 'lbfgs', 'C': 0.5, 'class_weight': 'balanced', 'max_iter': 2000},
]

best_log_reg_val_auc = -1.0
best_log_reg_model = None
best_log_reg_params = None

# List to store results for plotting
log_reg_results_for_plot = []

if X_train.shape[0] > 0 and y_train.shape[0] > 0:
    for i, params in enumerate(log_reg_param_sets):
        # Removed detailed print for each set as requested
        # print(f"\n  Trying Logistic Regression with params (Set {i+1}): {params}")
        
        # Create the OneVsRestClassifier with the current LogisticRegression parameters
        log_reg_clf = OneVsRestClassifier(LogisticRegression(random_state=42, **params))
        
        # Fit the model on the full training data
        log_reg_clf.fit(X_train, y_train)

        # Calculate Training AUC
        y_pred_proba_train = log_reg_clf.predict_proba(X_train)
        auc_train = calculate_safe_auc(y_train, y_pred_proba_train, ALL_SPECIES)
        # Removed print(f"    Training AUC: {auc_train:.4f}")

        # Calculate Validation AUC
        auc_val = 0.0 # Default if no validation data
        if X_val.shape[0] > 0:
            y_pred_proba_val = log_reg_clf.predict_proba(X_val)
            auc_val = calculate_safe_auc(y_val, y_pred_proba_val, ALL_SPECIES)
            # Removed print(f"    Validation AUC: {auc_val:.4f}")

            # Check if this model is the best so far based on Validation AUC
            if auc_val > best_log_reg_val_auc:
                best_log_reg_val_auc = auc_val
                best_log_reg_model = log_reg_clf
                best_log_reg_params = params
        else:
            # If no validation data, use training AUC as fallback for best model selection
            if auc_train > best_log_reg_val_auc:
                best_log_reg_val_auc = auc_train
                best_log_reg_model = log_reg_clf
                best_log_reg_params = params
        
        # Store results for plotting
        log_reg_results_for_plot.append({
            'Set': f'Set {i+1}',
            'Params': params,
            'Training AUC': auc_train,
            'Validation AUC': auc_val
        })

else:
    print("  Skipping Logistic Regression: Insufficient training data.")

# After trying all parameter sets, store the best model found
if best_log_reg_model:
    models['LogisticRegression'] = best_log_reg_model
    validation_auc_scores['LogisticRegression'] = best_log_reg_val_auc
    training_auc_scores['LogisticRegression'] = auc_train # Store the training AUC of the best model
    best_hyperparameters['LogisticRegression'] = best_log_reg_params
    print(f"\n  Best Logistic Regression setup chosen: {best_log_reg_params}") # Only print best setup

    # --- Plotting Hyperparameter Performance ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    set_labels = [res['Set'] for res in log_reg_results_for_plot]
    train_aucs = [res['Training AUC'] for res in log_reg_results_for_plot]
    val_aucs = [res['Validation AUC'] for res in log_reg_results_for_plot]

    x = np.arange(len(set_labels)) # the label locations
    width = 0.35 # the width of the bars

    rects1 = ax.bar(x - width/2, train_aucs, width, label='Training AUC', color='skyblue')
    rects2 = ax.bar(x + width/2, val_aucs, width, label='Validation AUC', color='lightcoral')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('AUC Score')
    ax.set_title('Logistic Regression Hyperparameter Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(set_labels)
    ax.legend()
    ax.set_ylim(0, 1) # AUC scores are between 0 and 1

    # Add exact values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()

else:
    print("\n  No Logistic Regression model trained due to insufficient data.")

print("\nFinished Logistic Regression training and comparison.")


import warnings

# Suppress ALL warnings in this specific cell, with more targeted filters for common sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.multiclass')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.model_selection._validation')
warnings.filterwarnings('ignore', category=UserWarning, message='Only one class present in y_true')
warnings.filterwarnings('ignore', category=UserWarning, message='Label not .* is present in all training examples')

print("\n--- Tuning and Training K-Nearest Neighbors (OneVsRest) ---")

# Define three different sets of hyperparameters to compare manually
knn_param_sets = [
    # Set 1: Default-ish, common choice
    {'n_neighbors': 5, 'weights': 'uniform'},
    # Set 2: Slightly more neighbors, distance weighting
    {'n_neighbors': 7, 'weights': 'distance'},
    # Set 3: Fewer neighbors, uniform weighting
    {'n_neighbors': 3, 'weights': 'uniform'},
]

best_knn_val_auc = -1.0
best_knn_model = None
best_knn_params = None

knn_results_for_plot = []

if X_train.shape[0] > 0 and y_train.shape[0] > 0:
    for i, params in enumerate(knn_param_sets):
        # Removed detailed print for each set
        
        # Create the OneVsRestClassifier with the current KNeighborsClassifier parameters
        knn_clf = OneVsRestClassifier(KNeighborsClassifier(**params))
        
        # Fit the model on the full training data
        knn_clf.fit(X_train, y_train)

        # Calculate Training AUC
        y_pred_proba_train = knn_clf.predict_proba(X_train)
        auc_train = calculate_safe_auc(y_train, y_pred_proba_train, ALL_SPECIES)
        
        # Calculate Validation AUC
        auc_val = 0.0
        if X_val.shape[0] > 0:
            y_pred_proba_val = knn_clf.predict_proba(X_val)
            auc_val = calculate_safe_auc(y_val, y_pred_proba_val, ALL_SPECIES)

            # Check if this model is the best so far based on Validation AUC
            if auc_val > best_knn_val_auc:
                best_knn_val_auc = auc_val
                best_knn_model = knn_clf
                best_knn_params = params
        else:
            if auc_train > best_knn_val_auc:
                best_knn_val_auc = auc_train
                best_knn_model = knn_clf
                best_knn_params = params
                
        knn_results_for_plot.append({
            'Set': f'Set {i+1}',
            'Params': params,
            'Training AUC': auc_train,
            'Validation AUC': auc_val
        })

else:
    print("  Skipping K-Nearest Neighbors: Insufficient training data.")

# After trying all parameter sets, store the best model found
if best_knn_model:
    models['KNearestNeighbors'] = best_knn_model
    validation_auc_scores['KNearestNeighbors'] = best_knn_val_auc
    training_auc_scores['KNearestNeighbors'] = auc_train # Store the training AUC of the best model
    best_hyperparameters['KNearestNeighbors'] = best_knn_params
    print(f"\n  Best K-Nearest Neighbors setup chosen: {best_knn_params}")

    # --- Plotting Hyperparameter Performance ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    set_labels = [res['Set'] for res in knn_results_for_plot]
    train_aucs = [res['Training AUC'] for res in knn_results_for_plot]
    val_aucs = [res['Validation AUC'] for res in knn_results_for_plot]

    x = np.arange(len(set_labels))
    width = 0.35

    rects1 = ax.bar(x - width/2, train_aucs, width, label='Training AUC', color='skyblue')
    rects2 = ax.bar(x + width/2, val_aucs, width, label='Validation AUC', color='lightcoral')

    ax.set_ylabel('AUC Score')
    ax.set_title('K-Nearest Neighbors Hyperparameter Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(set_labels)
    ax.legend()
    ax.set_ylim(0, 1)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()

else:
    print("\n  No K-Nearest Neighbors model trained due to insufficient data.")

print("\nFinished K-Nearest Neighbors training and comparison.")



import warnings
# Suppress ALL warnings in this specific cell, with more targeted filters for common sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.multiclass')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.model_selection._validation')
warnings.filterwarnings('ignore', category=UserWarning, message='Only one class present in y_true')
warnings.filterwarnings('ignore', category=UserWarning, message='Label not .* is present in all training examples')

print("\n--- Tuning and Training Naïve Bayes (OneVsRest) ---")

# Define three different sets of hyperparameters to compare manually
# GaussianNB has fewer hyperparameters, so we'll vary var_smoothing
gnb_param_sets = [
    # Set 1: Default var_smoothing
    {'var_smoothing': 1e-9},
    # Set 2: Slightly larger var_smoothing
    {'var_smoothing': 1e-8},
    # Set 3: Even larger var_smoothing
    {'var_smoothing': 1e-7},
]

best_gnb_val_auc = -1.0
best_gnb_model = None
best_gnb_params = None

gnb_results_for_plot = []

if X_train.shape[0] > 0 and y_train.shape[0] > 0:
    for i, params in enumerate(gnb_param_sets):
        # Removed detailed print for each set
        
        # Create the OneVsRestClassifier with the current GaussianNB parameters
        gnb_clf = OneVsRestClassifier(GaussianNB(**params))
        
        # Fit the model on the full training data
        gnb_clf.fit(X_train, y_train)

        # Calculate Training AUC
        y_pred_proba_train = gnb_clf.predict_proba(X_train)
        auc_train = calculate_safe_auc(y_train, y_pred_proba_train, ALL_SPECIES)
        
        # Calculate Validation AUC
        auc_val = 0.0
        if X_val.shape[0] > 0:
            y_pred_proba_val = gnb_clf.predict_proba(X_val)
            auc_val = calculate_safe_auc(y_val, y_pred_proba_val, ALL_SPECIES)

            # Check if this model is the best so far based on Validation AUC
            if auc_val > best_gnb_val_auc:
                best_gnb_val_auc = auc_val
                best_gnb_model = gnb_clf
                best_gnb_params = params
        else:
            if auc_train > best_gnb_val_auc:
                best_gnb_val_auc = auc_train
                best_gnb_model = gnb_clf
                best_gnb_params = params
                
        gnb_results_for_plot.append({
            'Set': f'Set {i+1}',
            'Params': params,
            'Training AUC': auc_train,
            'Validation AUC': auc_val
        })

else:
    print("  Skipping Naïve Bayes: Insufficient training data.")

# After trying all parameter sets, store the best model found
if best_gnb_model:
    models['NaiveBayes'] = best_gnb_model
    validation_auc_scores['NaiveBayes'] = best_gnb_val_auc
    training_auc_scores['NaiveBayes'] = auc_train # Store the training AUC of the best model
    best_hyperparameters['NaiveBayes'] = best_gnb_params
    print(f"\n  Best Naïve Bayes setup chosen: {best_gnb_params}")

    # --- Plotting Hyperparameter Performance ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    set_labels = [res['Set'] for res in gnb_results_for_plot]
    train_aucs = [res['Training AUC'] for res in gnb_results_for_plot]
    val_aucs = [res['Validation AUC'] for res in gnb_results_for_plot]

    x = np.arange(len(set_labels))
    width = 0.35

    rects1 = ax.bar(x - width/2, train_aucs, width, label='Training AUC', color='skyblue')
    rects2 = ax.bar(x + width/2, val_aucs, width, label='Validation AUC', color='lightcoral')

    ax.set_ylabel('AUC Score')
    ax.set_title('Naïve Bayes Hyperparameter Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(set_labels)
    ax.legend()
    ax.set_ylim(0, 1)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()

else:
    print("\n  No Naïve Bayes model trained due to insufficient data.")

print("\nFinished Naïve Bayes training and comparison.")



import warnings

# Suppress ALL warnings in this specific cell, with more targeted filters for common sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.multiclass')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.model_selection._validation')
warnings.filterwarnings('ignore', category=UserWarning, message='Only one class present in y_true')
warnings.filterwarnings('ignore', category=UserWarning, message='Label not .* is present in all training examples')

print("\n--- Tuning and Training Decision Tree (OneVsRest) ---")

# Define three different sets of hyperparameters to compare manually
dt_param_sets = [
    # Set 1: Moderately deep tree, default min_samples_leaf
    {'max_depth': 10, 'min_samples_leaf': 1},
    # Set 2: Deeper tree, slightly higher min_samples_leaf to prevent overfitting
    {'max_depth': 15, 'min_samples_leaf': 3},
    # Set 3: Shallower tree, more samples per leaf
    {'max_depth': 7, 'min_samples_leaf': 5},
]

best_dt_val_auc = -1.0
best_dt_model = None
best_dt_params = None

dt_results_for_plot = []

if X_train.shape[0] > 0 and y_train.shape[0] > 0:
    for i, params in enumerate(dt_param_sets):
        # Removed detailed print for each set
        
        # Create the OneVsRestClassifier with the current DecisionTreeClassifier parameters
        dt_clf = OneVsRestClassifier(DecisionTreeClassifier(random_state=42, **params))
        
        # Fit the model on the full training data
        dt_clf.fit(X_train, y_train)

        # Calculate Training AUC
        y_pred_proba_train = dt_clf.predict_proba(X_train)
        auc_train = calculate_safe_auc(y_train, y_pred_proba_train, ALL_SPECIES)
        
        # Calculate Validation AUC
        auc_val = 0.0
        if X_val.shape[0] > 0:
            y_pred_proba_val = dt_clf.predict_proba(X_val)
            auc_val = calculate_safe_auc(y_val, y_pred_proba_val, ALL_SPECIES)

            # Check if this model is the best so far based on Validation AUC
            if auc_val > best_dt_val_auc:
                best_dt_val_auc = auc_val
                best_dt_model = dt_clf
                best_dt_params = params
        else:
            if auc_train > best_dt_val_auc:
                best_dt_val_auc = auc_train
                best_dt_model = dt_clf
                best_dt_params = params
                
        dt_results_for_plot.append({
            'Set': f'Set {i+1}',
            'Params': params,
            'Training AUC': auc_train,
            'Validation AUC': auc_val
        })

else:
    print("  Skipping Decision Tree: Insufficient training data.")

# After trying all parameter sets, store the best model found
if best_dt_model:
    models['DecisionTree'] = best_dt_model
    validation_auc_scores['DecisionTree'] = best_dt_val_auc
    training_auc_scores['DecisionTree'] = auc_train # Store the training AUC of the best model
    best_hyperparameters['DecisionTree'] = best_dt_params
    print(f"\n  Best Decision Tree setup chosen: {best_dt_params}")

    # --- Plotting Hyperparameter Performance ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    set_labels = [res['Set'] for res in dt_results_for_plot]
    train_aucs = [res['Training AUC'] for res in dt_results_for_plot]
    val_aucs = [res['Validation AUC'] for res in dt_results_for_plot]

    x = np.arange(len(set_labels))
    width = 0.35

    rects1 = ax.bar(x - width/2, train_aucs, width, label='Training AUC', color='skyblue')
    rects2 = ax.bar(x + width/2, val_aucs, width, label='Validation AUC', color='lightcoral')

    ax.set_ylabel('AUC Score')
    ax.set_title('Decision Tree Hyperparameter Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(set_labels)
    ax.legend()
    ax.set_ylim(0, 1)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()

else:
    print("\n  No Decision Tree model trained due to insufficient data.")

print("\nFinished Decision Tree training and comparison.")



import warnings

# Suppress ALL warnings in this specific cell, with more targeted filters for common sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.multiclass')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.model_selection._validation')
warnings.filterwarnings('ignore', category=UserWarning, message='Only one class present in y_true')
warnings.filterwarnings('ignore', category=UserWarning, message='Label not .* is present in all training examples')

print("\n--- Tuning and Training Random Forest Classifier (OneVsRest) ---")

# Define three different sets of hyperparameters to compare manually
rf_param_sets = [
    # Set 1: Moderate number of estimators, good depth, balanced class weight
    {'n_estimators': 100, 'max_depth': 20, 'min_samples_leaf': 1, 'class_weight': 'balanced'},
    # Set 2: More estimators, slightly less depth, no class weight
    {'n_estimators': 150, 'max_depth': 15, 'min_samples_leaf': 2, 'class_weight': None},
    # Set 3: Fewer estimators, deeper trees, more samples per leaf, balanced class weight
    {'n_estimators': 75, 'max_depth': 25, 'min_samples_leaf': 3, 'class_weight': 'balanced'},
]

best_rf_val_auc = -1.0
best_rf_model = None
best_rf_params = None

rf_results_for_plot = []

if X_train.shape[0] > 0 and y_train.shape[0] > 0:
    for i, params in enumerate(rf_param_sets):
        # Removed detailed print for each set
        
        # Create the OneVsRestClassifier with the current RandomForestClassifier parameters
        rf_clf = OneVsRestClassifier(RandomForestClassifier(random_state=42, **params))
        
        # Fit the model on the full training data
        rf_clf.fit(X_train, y_train)

        # Calculate Training AUC
        y_pred_proba_train = rf_clf.predict_proba(X_train)
        auc_train = calculate_safe_auc(y_train, y_pred_proba_train, ALL_SPECIES)
        
        # Calculate Validation AUC
        auc_val = 0.0
        if X_val.shape[0] > 0:
            y_pred_proba_val = rf_clf.predict_proba(X_val)
            auc_val = calculate_safe_auc(y_val, y_pred_proba_val, ALL_SPECIES)

            # Check if this model is the best so far based on Validation AUC
            if auc_val > best_rf_val_auc:
                best_rf_val_auc = auc_val
                best_rf_model = rf_clf
                best_rf_params = params
        else:
            if auc_train > best_rf_val_auc:
                best_rf_val_auc = auc_train
                best_rf_model = rf_clf
                best_rf_params = params
                
        rf_results_for_plot.append({
            'Set': f'Set {i+1}',
            'Params': params,
            'Training AUC': auc_train,
            'Validation AUC': auc_val
        })

else:
    print("  Skipping Random Forest: Insufficient training data.")

# After trying all parameter sets, store the best model found
if best_rf_model:
    models['RandomForest'] = best_rf_model
    validation_auc_scores['RandomForest'] = best_rf_val_auc
    training_auc_scores['RandomForest'] = auc_train # Store the training AUC of the best model
    best_hyperparameters['RandomForest'] = best_rf_params
    print(f"\n  Best Random Forest setup chosen: {best_rf_params}")

    # --- Plotting Hyperparameter Performance ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    set_labels = [res['Set'] for res in rf_results_for_plot]
    train_aucs = [res['Training AUC'] for res in rf_results_for_plot]
    val_aucs = [res['Validation AUC'] for res in rf_results_for_plot]

    x = np.arange(len(set_labels))
    width = 0.35

    rects1 = ax.bar(x - width/2, train_aucs, width, label='Training AUC', color='skyblue')
    rects2 = ax.bar(x + width/2, val_aucs, width, label='Validation AUC', color='lightcoral')

    ax.set_ylabel('AUC Score')
    ax.set_title('Random Forest Hyperparameter Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(set_labels)
    ax.legend()
    ax.set_ylim(0, 1)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()

else:
    print("\n  No Random Forest model trained due to insufficient data.")

print("\nFinished Random Forest training and comparison.")



import warnings

# Suppress ALL warnings in this specific cell, with more targeted filters for common sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.multiclass')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.model_selection._validation')
warnings.filterwarnings('ignore', category=UserWarning, message='Only one class present in y_true')
warnings.filterwarnings('ignore', category=UserWarning, message='Label not .* is present in all training examples')
warnings.filterwarnings('ignore', category=UserWarning, message='The max_iter was reached') # For potential convergence warnings in LinearSVC

print("\n--- Tuning and Training Support Vector Machine (LinearSVC) (OneVsRest) ---")

# Define three different sets of hyperparameters to compare manually for LinearSVC.
# Note: LinearSVC does not have 'predict_proba', so decision_function scores are used for AUC.
# 'dual=False' is often preferred when n_samples > n_features, and requires loss='squared_hinge'
# for 'l2' penalty (which is default).
svm_param_sets = [
    # Set 1: Moderate C, squared_hinge loss (compatible with dual=False)
    {'C': 1.0, 'loss': 'squared_hinge', 'max_iter': 2000},
    # Set 2: Lower C (more regularization), squared_hinge loss
    {'C': 0.5, 'loss': 'squared_hinge', 'max_iter': 2000},
    # Set 3: Higher C (less regularization), squared_hinge loss
    {'C': 2.0, 'loss': 'squared_hinge', 'max_iter': 2000},
]

best_svm_val_auc = -1.0
best_svm_model = None
best_svm_params = None

svm_results_for_plot = []

if X_train.shape[0] > 0 and y_train.shape[0] > 0:
    for i, params in enumerate(svm_param_sets):
        # Removed detailed print for each set
        
        # Create the OneVsRestClassifier with the current LinearSVC parameters
        # dual=False is explicitly set to prevent convergence issues with certain solvers/losses
        svm_clf = OneVsRestClassifier(LinearSVC(random_state=42, dual=False, **params))
        
        # Fit the model on the full training data
        svm_clf.fit(X_train, y_train)

        # Calculate Training AUC using decision_function as LinearSVC doesn't have predict_proba
        y_pred_scores_train = svm_clf.decision_function(X_train)
        auc_train = calculate_safe_auc(y_train, y_pred_scores_train, ALL_SPECIES)
        
        # Calculate Validation AUC using decision_function
        auc_val = 0.0
        if X_val.shape[0] > 0:
            y_pred_scores_val = svm_clf.decision_function(X_val)
            auc_val = calculate_safe_auc(y_val, y_pred_scores_val, ALL_SPECIES)

            # Check if this model is the best so far based on Validation AUC
            if auc_val > best_svm_val_auc:
                best_svm_val_auc = auc_val
                best_svm_model = svm_clf
                best_svm_params = params
        else:
            if auc_train > best_svm_val_auc:
                best_svm_val_auc = auc_train
                best_svm_model = svm_clf
                best_svm_params = params
                
        svm_results_for_plot.append({
            'Set': f'Set {i+1}',
            'Params': params,
            'Training AUC': auc_train,
            'Validation AUC': auc_val
        })

else:
    print("  Skipping Support Vector Machine: Insufficient training data.")

# After trying all parameter sets, store the best model found
if best_svm_model:
    models['SVM'] = best_svm_model
    validation_auc_scores['SVM'] = best_svm_val_auc
    training_auc_scores['SVM'] = auc_train # Store the training AUC of the best model
    best_hyperparameters['SVM'] = best_svm_params
    print(f"\n  Best Support Vector Machine setup chosen: {best_svm_params}")

    # --- Plotting Hyperparameter Performance ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    set_labels = [res['Set'] for res in svm_results_for_plot]
    train_aucs = [res['Training AUC'] for res in svm_results_for_plot]
    val_aucs = [res['Validation AUC'] for res in svm_results_for_plot]

    x = np.arange(len(set_labels))
    width = 0.35

    rects1 = ax.bar(x - width/2, train_aucs, width, label='Training AUC', color='skyblue')
    rects2 = ax.bar(x + width/2, val_aucs, width, label='Validation AUC', color='lightcoral')

    ax.set_ylabel('AUC Score')
    ax.set_title('Support Vector Machine Hyperparameter Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(set_labels)
    ax.legend()
    ax.set_ylim(0, 1)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()

else:
    print("\n  No Support Vector Machine model trained due to insufficient data.")

print("\nFinished Support Vector Machine training and comparison.")



import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.multiclass')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.model_selection._validation')
warnings.filterwarnings('ignore', category=UserWarning, message='Only one class present in y_true')
warnings.filterwarnings('ignore', category=UserWarning, message='Label not .* is present in all training examples')
warnings.filterwarnings('ignore', category=UserWarning, message='Maximum number of iterations reached before convergence')


print("\n--- Tuning and Training Neural Network (MLPClassifier) ---")

# Define three different sets of hyperparameters to compare manually
nn_param_sets = [
    {'hidden_layer_sizes': (50,), 'activation': 'relu', 'solver': 'adam', 'max_iter': 1000, 'alpha': 0.0001},
    {'hidden_layer_sizes': (100,), 'activation': 'tanh', 'solver': 'adam', 'max_iter': 1000, 'alpha': 0.0005},
    {'hidden_layer_sizes': (50, 50), 'activation': 'relu', 'solver': 'adam', 'max_iter': 1000, 'alpha': 0.001},
]

best_nn_val_auc = -1.0
best_nn_model = None
best_nn_params = None

nn_results_for_plot = []

if X_train.shape[0] > 0 and y_train.shape[0] > 0:
    for i, params in enumerate(nn_param_sets):
        print(f"\n  Trying Neural Network with params (Set {i+1}): {params}")
        
        nn_clf = MLPClassifier(random_state=42, verbose=False, **params)
        
        nn_clf.fit(X_train, y_train)

        y_pred_proba_train = nn_clf.predict_proba(X_train)
        auc_train = calculate_safe_auc(y_train, y_pred_proba_train, ALL_SPECIES)
        
        auc_val = 0.0
        if X_val.shape[0] > 0:
            y_pred_proba_val = nn_clf.predict_proba(X_val)
            auc_val = calculate_safe_auc(y_val, y_pred_proba_val, ALL_SPECIES)

            if auc_val > best_nn_val_auc:
                best_nn_val_auc = auc_val
                best_nn_model = nn_clf
                best_nn_params = params
        else:
            if auc_train > best_nn_val_auc:
                best_nn_val_auc = auc_train
                best_nn_model = nn_clf
                best_nn_params = params
                
        nn_results_for_plot.append({
            'Set': f'Set {i+1}',
            'Params': params,
            'Training AUC': auc_train,
            'Validation AUC': auc_val
        })

else:
    print("  Skipping Neural Network: Insufficient training data.")

if best_nn_model:
    models['NeuralNetwork'] = best_nn_model
    validation_auc_scores['NeuralNetwork'] = best_nn_val_auc
    training_auc_scores['NeuralNetwork'] = auc_train
    best_hyperparameters['NeuralNetwork'] = best_nn_params
    print(f"\n  Best Neural Network setup chosen: {best_nn_params}")

    # --- Plotting Hyperparameter Performance ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    set_labels = [f"Set {res['Set']}\nHidden: {res['Params']['hidden_layer_sizes']}\nActivation: {res['Params']['activation']}" for res in nn_results_for_plot]
    train_aucs = [res['Training AUC'] for res in nn_results_for_plot]
    val_aucs = [res['Validation AUC'] for res in nn_results_for_plot]

    x = np.arange(len(set_labels))
    width = 0.35

    rects1 = ax.bar(x - width/2, train_aucs, width, label='Training AUC', color='skyblue')
    rects2 = ax.bar(x + width/2, val_aucs, width, label='Validation AUC', color='lightcoral')

    ax.set_ylabel('AUC Score')
    ax.set_title('Neural Network Hyperparameter Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(set_labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()

else:
    print("\n  No Neural Network model trained due to insufficient data.")

print("\nFinished Neural Network training and comparison.")



import warnings

# Suppress common warnings that might arise during plotting or data manipulation
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


print("\n--- Starting Feature Importance Analysis ---")
print("This analysis helps understand which MFCC features are most influential for each model.")

# Define feature names for plotting (MFCC_0, MFCC_1, ..., MFCC_N_MFCC-1)
feature_names = [f'MFCC_{i}' for i in range(N_MFCC)]

# Iterate through the trained models and analyze feature importance
for model_name, model_obj in models.items():
    print(f"\n--- Analyzing Feature Importance for: {model_name} ---")

    feature_importances = None
    
    # Check for models that have a direct feature_importances_ attribute (tree-based)
    if hasattr(model_obj, 'estimator') and hasattr(model_obj.estimator, 'feature_importances_'):
        # For OneVsRestClassifier, feature_importances_ is usually on the base estimator

        if hasattr(model_obj.estimator, 'feature_importances_'):
            feature_importances = model_obj.estimator.feature_importances_
        elif hasattr(model_obj, 'estimators_'): # For OneVsRestClassifier with multiple estimators
            # Average feature importances across all individual estimators
            importances_list = [est.feature_importances_ for est in model_obj.estimators_ if hasattr(est, 'feature_importances_')]
            if importances_list:
                feature_importances = np.mean(importances_list, axis=0)

    # Check for models that have a coef_ attribute (linear models)
    elif hasattr(model_obj, 'estimator') and hasattr(model_obj.estimator, 'coef_'):
        # For OneVsRestClassifier, coef_ is usually on the base estimator
        # Take the mean absolute value of coefficients across all binary estimators
        if hasattr(model_obj.estimator, 'coef_'):
            # coef_ can be 1D for binary or 2D for multi-class (one-vs-rest)
            coefs = model_obj.estimator.coef_
            if coefs.ndim > 1:
                feature_importances = np.mean(np.abs(coefs), axis=0)
            else:
                feature_importances = np.abs(coefs)
        elif hasattr(model_obj, 'estimators_'): # For OneVsRestClassifier with multiple estimators
            # Average absolute coefficients across all individual estimators
            coefs_list = [np.abs(est.coef_).flatten() for est in model_obj.estimators_ if hasattr(est, 'coef_')]
            if coefs_list:
                feature_importances = np.mean(coefs_list, axis=0)

    # Special handling for MLPClassifier which doesn't have OneVsRestClassifier wrapper
    elif model_name == 'NeuralNetwork' and hasattr(model_obj, 'coefs_'):
        # Feature importance for NNs is complex. We'll use a simplified approach:
        # Sum of absolute weights connected to the input layer. This is a very rough estimate.
        if model_obj.coefs_:
            # coefs_[0] contains weights from input layer to first hidden layer
            feature_importances = np.sum(np.abs(model_obj.coefs_[0]), axis=1)
        print("  Note: Feature importance for Neural Networks is complex and this is a simplified view.")

    if feature_importances is not None and len(feature_importances) == N_MFCC:
        # Create a pandas Series for easy sorting and handling
        importance_series = pd.Series(feature_importances, index=feature_names)
        importance_series = importance_series.sort_values(ascending=False)

        # Plotting the top N features
        top_n = min(10, len(importance_series))
        
        plt.figure(figsize=(12, 7))
        importance_series.head(top_n).plot(kind='bar', color='skyblue')
        plt.title(f'Top {top_n} Feature Importances for {model_name}')
        plt.xlabel('MFCC Feature')
        plt.ylabel('Importance Score (Normalized)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print(f"  Feature importance not directly available or applicable for {model_name} in a straightforward manner.")
        print("  For models like KNN and Naïve Bayes, feature importance is not directly interpretable from model coefficients/attributes.")
        print("  For Neural Networks, more advanced techniques (e.g., SHAP, Permutation Importance) are typically required.")

print("\n--- Feature Importance Analysis Complete ---")



print("Starting prediction and submission file generation...")

# --- Helper function to process a single test soundscape ---
def process_test_soundscape(soundscape_path, model, scaler, mlb, sr=SR, duration=DURATION, n_mfcc=N_MFCC):

    segment_predictions = []
    # Extract the soundscape ID from the filename (e.g., 'soundscape_xxxxxx.ogg' -> 'soundscape_xxxxxx')
    soundscape_id = os.path.splitext(os.path.basename(soundscape_path))[0]

    try:
        # Load the entire 1-minute soundscape audio
        y_soundscape, current_sr = librosa.load(soundscape_path, sr=sr, mono=True)

        # Calculate the number of 5-second segments in the 1-minute soundscape (60 seconds / 5 seconds = 12 segments)
        num_segments = int(len(y_soundscape) / (duration * sr))

        # Iterate through each 5-second segment
        for i in range(num_segments):
            start_sample = i * duration * sr
            end_sample = (i + 1) * duration * sr
            segment_audio = y_soundscape[start_sample:end_sample]

            # Ensure segment is exactly 'duration' seconds long by padding if necessary.
            if len(segment_audio) < duration * sr:
                segment_audio = np.pad(segment_audio, (0, (duration * sr) - len(segment_audio)), 'constant')

            # --- Extract features for the current 5-second segment ---
            def _extract_features_from_array(y_array, sr_val, n_mfcc_val):
                """Helper to extract MFCCs from an audio array."""
                mfccs = librosa.feature.mfcc(y=y_array, sr=sr_val, n_mfcc=n_mfcc_val)
                return np.mean(mfccs.T, axis=0)

            segment_features = _extract_features_from_array(segment_audio, sr, n_mfcc)
            
            # Scale the extracted features using the StandardScaler fitted on training data.
            scaled_features = scaler.transform(segment_features.reshape(1, -1))

            # Make predictions (probabilities) for the current segment using the trained model.
            if hasattr(model, 'predict_proba'):
                segment_proba = model.predict_proba(scaled_features)[0]
            elif hasattr(model, 'decision_function'):

                # Apply sigmoid to map scores to a [0, 1] range.
                decision_scores = model.decision_function(scaled_features)[0]
                segment_proba = 1 / (1 + np.exp(-decision_scores))
            else:
                # Fallback if neither predict_proba nor decision_function is available
                print(f"Warning: Model {type(model).__name__} does not have predict_proba or decision_function. Using dummy probabilities.")
                segment_proba = np.full(len(mlb.classes_), 1.0 / len(mlb.classes_))


            # Create the 'row_id' for this segment, following the competition's format:
            end_time = (i + 1) * duration # End time of the current 5-second segment
            row_id = f"{soundscape_id}_{end_time}"

            # Create a dictionary to hold the row_id and predicted probabilities for all species.
            segment_pred_dict = {'row_id': row_id}
            # Map probabilities back to species names using the MultiLabelBinarizer's classes.
            for species_idx, species_name in enumerate(mlb.classes_):
                segment_pred_dict[species_name] = segment_proba[species_idx]
            
            segment_predictions.append(segment_pred_dict)

    except Exception as e:
        # If any error occurs during processing a soundscape, print a warning and return an empty DataFrame.
        print(f"Error processing soundscape {soundscape_path}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

    return pd.DataFrame(segment_predictions)


print("Prediction and submission file generation process starting.")

# --- Select the best model for submission ---
if validation_auc_scores:
    best_model_name = max(validation_auc_scores, key=validation_auc_scores.get)
    submission_model = models[best_model_name]
    print(f"\nSelected '{best_model_name}' as the submission model based on validation AUC.")
else:
    # Fallback if no models were trained successfully (e.g., due to insufficient data in Cell 6/7).
    print("\nWarning: No models were trained or evaluated successfully. Cannot select a best model.")
    print("Creating a dummy submission file with default probabilities.")
    submission_model = None # Indicate no model is available

# --- Process test soundscapes and generate submission file ---
all_submission_rows = []


# List all actual .ogg files in the TEST_SOUNDSCAPES_PATH
test_soundscape_files = [f for f in os.listdir(TEST_SOUNDSCAPES_PATH) if f.endswith('.ogg')]

if submission_model is not None and test_soundscape_files:
    print(f"\nProcessing {len(test_soundscape_files)} test soundscapes for predictions...")
    for soundscape_file in test_soundscape_files:
        full_soundscape_path = os.path.join(TEST_SOUNDSCAPES_PATH, soundscape_file)
        
        # Process each soundscape using our helper function
        soundscape_df = process_test_soundscape(full_soundscape_path, submission_model, scaler, mlb, SR, DURATION, N_MFCC)
        
        if not soundscape_df.empty:
            all_submission_rows.append(soundscape_df)
        else:
            print(f"Skipping {soundscape_file} due to processing error or no predictions generated.")

    if all_submission_rows:
        final_submission_df = pd.concat(all_submission_rows, ignore_index=True)
    else:
        print("Warning: No predictions generated from any test soundscapes. Creating empty submission.")
        final_submission_df = sample_submission_df.iloc[0:0].copy() # Create empty df with correct columns
        # Ensure it has all species columns if it's truly empty
        for species_col in ALL_SPECIES:
            if species_col not in final_submission_df.columns:
                final_submission_df[species_col] = [] # Add empty list for column

else:
    # Fallback if no model was trained or no test soundscapes found
    print("\nNo trained model or test soundscapes found. Creating a dummy submission file with default probabilities.")
    # This part assumes sample_submission_df was loaded successfully in Cell 3
    final_submission_df = sample_submission_df.copy()
    if len(ALL_SPECIES) > 0:
        for species_col in ALL_SPECIES:
            if species_col != 'row_id': # Ensure not to overwrite row_id
                final_submission_df[species_col] = 1.0 / len(ALL_SPECIES)
    else:
        print("Warning: ALL_SPECIES is empty, cannot fill submission with probabilities.")
        final_submission_df = pd.DataFrame(columns=['row_id'])


# Save the submission file
submission_file_name = 'submission.csv'
final_submission_df.to_csv(submission_file_name, index=False)

print(f"\nSubmission file '{submission_file_name}' created successfully.")
print("\nFirst 5 rows of the generated submission file:")
print(final_submission_df.head())
print(f"\nShape of the submission file: {final_submission_df.shape}")



import joblib
import os

print("\n--- Saving the Best Performing Scikit-learn Model from Homework Notebook ---")

# Ensure the 'models' and 'validation_auc_scores' dictionaries are populated.
# These dictionaries are expected to be available after running all previous training cells (7.1-7.8).
if 'models' in locals() and 'validation_auc_scores' in locals() and validation_auc_scores:
    # Find the best model based on validation AUC from the models that were successfully trained.
    best_model_name = max(validation_auc_scores, key=validation_auc_scores.get)
    best_model_instance = models[best_model_name]

    # Define the save path for the scikit-learn model.
    # We use a .pkl extension which is standard for joblib-saved scikit-learn models.
    save_path = 'best_sklearn_model.pkl' 

    try:
        # Save the scikit-learn model using joblib.
        # This serializes the entire model object to a file.
        joblib.dump(best_model_instance, save_path)
        print(f"Successfully saved the best model ('{best_model_name}') to {save_path}")
        print("\nNEXT STEP: You must now add this 'best_sklearn_model.pkl' file as an input dataset")
        print("to your 'BirdCLEF_Kaggle.ipynb' (Code 2) notebook on Kaggle.")
        print("Then, we will modify Code 2 to load this scikit-learn model and perform predictions.")
    except Exception as e:
        print(f"Error saving the model: {e}")
        print("Please ensure you have write permissions in the current directory.")
else:
    print("No models found or evaluated in 'models' dictionary. Cannot save a best model.")
    print("Please ensure all model training cells (7.1-7.8) have run successfully.")

print("\n--- Model Saving Process Complete ---")


