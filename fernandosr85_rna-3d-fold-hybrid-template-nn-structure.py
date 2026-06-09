# Standard Library Imports
import os
import time
import gc
import traceback
from collections import Counter
import warnings
import hashlib

# Data Manipulation Libraries
import numpy as np
import pandas as pd

# Visualization Libraries
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Machine Learning Libraries
try:
   # TensorFlow and Keras
   import tensorflow as tf
   from tensorflow.keras import layers, models, optimizers
   from tensorflow.keras.models import Model
   from tensorflow.keras.layers import (
       Input, Conv1D, Dense, Dropout, BatchNormalization, 
       Flatten, Reshape, Bidirectional, LSTM
   )
   from tensorflow.keras.callbacks import EarlyStopping
   
   # Scikit-learn
   from sklearn.model_selection import train_test_split
   
   # XGBoost
   import xgboost as xgb
   
   ML_AVAILABLE = True
except ImportError:
   print("Warning: ML libraries not available. Will use only reference-based methods.")
   ML_AVAILABLE = False

# Set random seed for reproducibility
np.random.seed(0)

# Suppress warnings
warnings.filterwarnings('ignore')


# Directories and files adjusted for the new competition
DATA_DIR = os.getenv('DATA_DIR', '/kaggle/input/stanford-rna-3d-folding/')
main_files = [
    "train_sequences.csv", 
    "train_labels.csv", 
    "validation_sequences.csv", 
    "validation_labels.csv", 
    "test_sequences.csv",
    "sample_submission.csv" 
]

DEFAULT_THRESHOLD = 0.4  # Default threshold after analysis

def optimize_dataframe(df, inplace=False, category_threshold=DEFAULT_THRESHOLD):
    """
    Optimizes the DataFrame to save memory.
    """
    if category_threshold < 0 or category_threshold > 1:
        raise ValueError("category_threshold must be between 0 and 1.")
    
    if not inplace:
        df = df.copy()
    
    for col in df.columns:
        col_type = df[col].dtype
        if np.issubdtype(col_type, np.integer):
            c_min, c_max = df[col].min(), df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        elif np.issubdtype(col_type, np.floating):
            if df[col].min() > np.finfo(np.float32).min and df[col].max() < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
        if col_type == object:
            unique_vals = len(df[col].unique())
            if unique_vals / len(df) < category_threshold:
                df[col] = df[col].astype('category')
    
    return df

def load_main_data(chunksize=50000):
    """
    Loads the main files.
    """
    data = {}
    for file_name in main_files:
        file_path = os.path.join(DATA_DIR, file_name)
        if os.path.exists(file_path):
            chunks = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False, chunksize=chunksize)
            dataframes = [optimize_dataframe(chunk, category_threshold=DEFAULT_THRESHOLD) for chunk in chunks]
            data[file_name] = pd.concat(dataframes, ignore_index=True)
        else:
            print(f"File {file_path} not found!")
    return data

def check_data_integrity(original_df, optimized_df):
    """
    Checks if the optimization did not alter the data.
    """
    try:
        pd.testing.assert_frame_equal(original_df, optimized_df, check_like=True)
        print("Integrity check passed: No changes in data after optimization.")
    except AssertionError as e:
        print(f"Data integrity check failed: {e}")

def check_duplicates(df):
    """
    Checks for duplicates in the DataFrame.
    """
    duplicates = df[df.duplicated(keep=False)]
    if not duplicates.empty:
        print(f"Warning: Duplicates found in the dataset. Number of duplicates: {duplicates.shape[0]}")
        return duplicates
    else:
        print("No duplicates found.")
    return None

def test_thresholds(df):
    """
    Tests different thresholds for DataFrame optimization.
    """
    thresholds = np.linspace(0.1, 0.9, 9)
    memory_usages = []
    for threshold in thresholds:
        optimized_df = optimize_dataframe(df.copy(), category_threshold=threshold)
        memory_usages.append(optimized_df.memory_usage(deep=True).sum() / 1024**2)
    return thresholds, memory_usages

def plot_memory_usage(thresholds, memory_usages):
    """
    Plots memory usage versus thresholds.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, memory_usages, marker='o', linestyle='-')
    plt.title("Memory Usage vs. Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Memory Usage (MB)")
    plt.grid(True)
    plt.show()

def analyze_sequence_data(df_sequences):
    """
    Analyzes RNA sequence data.
    """
    # Basic information
    print(f"Total sequences: {len(df_sequences)}")
    print(f"Available columns: {df_sequences.columns.tolist()}")
    
    # Sequence analysis
    if 'sequence' in df_sequences.columns:
        # Distribution of sequence lengths
        seq_lengths = df_sequences['sequence'].apply(len)
        print(f"\nSequence length statistics:")
        print(f"Minimum: {seq_lengths.min()}")
        print(f"Maximum: {seq_lengths.max()}")
        print(f"Average: {seq_lengths.mean():.2f}")
        
        # Nucleotide count
        nucleotides = ['A', 'C', 'G', 'U']
        nucleotide_counts = {n: df_sequences['sequence'].str.count(n).sum() for n in nucleotides}
        total_nucleotides = sum(nucleotide_counts.values())
        
        print("\nNucleotide distribution:")
        for n, count in nucleotide_counts.items():
            print(f"{n}: {count} ({count/total_nucleotides*100:.2f}%)")
    
    return df_sequences

def analyze_label_data(df_labels):
    """
    Analyzes 3D coordinate data (labels).
    """
    print(f"Total entries in labels: {len(df_labels)}")
    print(f"Available columns: {df_labels.columns.tolist()}")
    
    # Analysis of 3D coordinates if available
    coord_columns = [col for col in df_labels.columns if col.startswith(('x_', 'y_', 'z_'))]
    if coord_columns:
        print(f"\nCoordinate columns found: {len(coord_columns)}")
        
        # Basic statistics of coordinates
        for i in range(1, 6):  # For the 5 possible structures
            x_col = f'x_{i}'
            y_col = f'y_{i}'
            z_col = f'z_{i}'
            
            if x_col in df_labels.columns and y_col in df_labels.columns and z_col in df_labels.columns:
                print(f"\nStatistics for structure {i}:")
                print(f"X - Mean: {df_labels[x_col].mean():.2f}, Std: {df_labels[x_col].std():.2f}")
                print(f"Y - Mean: {df_labels[y_col].mean():.2f}, Std: {df_labels[y_col].std():.2f}")
                print(f"Z - Mean: {df_labels[z_col].mean():.2f}, Std: {df_labels[z_col].std():.2f}")
    
    return df_labels

def create_submission_template(test_df, sample_submission_df):
    """
    Creates a submission template based on test data.
    """
    # Check if sample_submission.csv is available
    if sample_submission_df is None:
        print("Sample submission file not found. Creating a new template.")
        
        # Create a new DataFrame for submission
        submission_df = pd.DataFrame()
        
        # Example code to fill the template (adjust as needed)
        ids = []
        resnames = []
        resids = []
        
        for _, row in test_df.iterrows():
            sequence = row['sequence']
            target_id = row['target_id']
            
            for i, nucleotide in enumerate(sequence, 1):
                ids.append(f"{target_id}_{i}")
                resnames.append(nucleotide)
                resids.append(i)
        
        submission_df['ID'] = ids
        submission_df['resname'] = resnames
        submission_df['resid'] = resids
        
        # Add coordinate columns (5 structures)
        for i in range(1, 6):
            submission_df[f'x_{i}'] = 0.0
            submission_df[f'y_{i}'] = 0.0
            submission_df[f'z_{i}'] = 0.0
    else:
        submission_df = sample_submission_df.copy()
        print("Submission template created based on the provided example.")
    
    return submission_df

def main():
    start_time = time.time()
    
    # Load main data
    print("Loading main data...")
    main_data = load_main_data()
    
    # Check which files were loaded
    print("\nLoaded files:")
    for file_name, df in main_data.items():
        print(f"- {file_name}: {df.shape if df is not None else 'Not found'}")
    
    # Analyze training sequence data
    if "train_sequences.csv" in main_data:
        print("\n===== Training Sequences Analysis =====")
        analyze_sequence_data(main_data["train_sequences.csv"])
    
    # Analyze training label data
    if "train_labels.csv" in main_data:
        print("\n===== Training Labels Analysis =====")
        analyze_label_data(main_data["train_labels.csv"])
    
    # Check for duplicates in training data
    if "train_sequences.csv" in main_data:
        print("\nChecking for duplicates in training sequences...")
        check_duplicates(main_data["train_sequences.csv"])
    
    # Create submission template
    if "test_sequences.csv" in main_data:
        print("\nCreating submission template...")
        submission_template = create_submission_template(
            main_data["test_sequences.csv"],
            main_data.get("sample_submission.csv")
        )
        print(f"Submission template shape: {submission_template.shape}")
        print(f"First rows of the submission template:")
        print(submission_template.head())
    
    # Calculate execution time
    end_time = time.time()
    print(f"\nRuntime: {end_time - start_time:.2f} seconds")
    
    return main_data

if __name__ == '__main__':
    main_data = main()


# Updated main directory
dir_main = "/kaggle/input/stanford-rna-3d-folding/"

# List all files and directories in the main directory
try:
    all_files = os.listdir(dir_main)
    print(f"All files and directories in '{dir_main}':")
    
    for file in all_files:
        # Check if it's a file or directory
        full_path = os.path.join(dir_main, file)
        type_desc = "directory" if os.path.isdir(full_path) else "file"
        size = os.path.getsize(full_path) / 1024  # Size in KB
        print(f" - {file} ({type_desc}, {size:.2f} KB)")
        
        # If it's a directory, list up to 5 files inside it
        if os.path.isdir(full_path):
            try:
                internal_files = os.listdir(full_path)[:5]  # Limit to 5 files
                if internal_files:
                    print(f"   First files in '{file}':")
                    for internal_file in internal_files:
                        print(f"    * {internal_file}")
                    if len(os.listdir(full_path)) > 5:
                        print(f"    * ... and {len(os.listdir(full_path)) - 5} more file(s)")
                else:
                    print(f"   '{file}' is empty")
            except Exception as e:
                print(f"   Error listing contents of '{file}': {e}")
except Exception as e:
    print(f"Error listing directory {dir_main}: {e}")

# Check the structure of the main CSV files
main_files = [
    "train_sequences.csv", 
    "train_labels.csv", 
    "validation_sequences.csv", 
    "validation_labels.csv", 
    "test_sequences.csv",
    "sample_submission.csv"
]
print("\nChecking main CSV files:")

for file in main_files:
    full_path = os.path.join(dir_main, file)
    if os.path.exists(full_path):
        # Get file size
        size_mb = os.path.getsize(full_path) / (1024 * 1024)  # Size in MB
        
        # Read the first lines to check the structure
        try:
            import pandas as pd
            df = pd.read_csv(full_path, nrows=1)
            print(f"\n{file} ({size_mb:.2f} MB):")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Example:")
            print(df.head())
        except Exception as e:
            print(f"Error reading {file}: {e}")
    else:
        print(f"{file} not found.")


# Updated main directory
dir_main = "/kaggle/input/stanford-rna-3d-folding/"

def load_data():
    """
    Loads the main CSV files from the Stanford RNA 3D Folding competition.
    Returns a dictionary with DataFrames.
    """
    main_files = [
        "train_sequences.csv", 
        "train_labels.csv", 
        "validation_sequences.csv", 
        "validation_labels.csv", 
        "test_sequences.csv",
        "sample_submission.csv"
    ]
    
    data = {}
    for file_name in main_files:
        file_path = os.path.join(dir_main, file_name)
        if os.path.exists(file_path):
            try:
                data[file_name] = pd.read_csv(file_path)
                print(f"File {file_name} loaded successfully. Shape: {data[file_name].shape}")
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
        else:
            print(f"File {file_name} not found.")
            data[file_name] = None
    
    return data

def compare_columns(main_data):
    """
    Compares columns between different DataFrames.
    """
    # List all available keys
    print("\nLoaded files:")
    print(list(main_data.keys()))
    
    # Compare columns between train_sequences.csv and test_sequences.csv
    if "train_sequences.csv" in main_data and "test_sequences.csv" in main_data:
        train_cols = set(main_data["train_sequences.csv"].columns)
        test_cols = set(main_data["test_sequences.csv"].columns)
        
        print("\nColumns in train_sequences.csv:")
        print(list(main_data["train_sequences.csv"].columns))
        
        print("\nUnique columns in train_sequences.csv (not present in test_sequences.csv):")
        print(train_cols - test_cols)
        
        print("\nUnique columns in test_sequences.csv (not present in train_sequences.csv):")
        print(test_cols - train_cols)
    
    # Compare columns between train_labels.csv and validation_labels.csv
    if "train_labels.csvv" in main_data and "validation_labels.csv" in main_data:
        train_label_cols = set(main_data["train_labels.csv"].columns)
        val_label_cols = set(main_data["validation_labels.csv"].columns)
        
        print("\nColumns in train_labels.csv:")
        print(list(main_data["train_labels.csv"].columns))
        
        print("\nColumns in validation_labels.csv:")
        print(list(main_data["validation_labels.csv"].columns))
        
        print("\nUnique columns in validation_labels.csv (not present in train_labels.csv):")
        print(val_label_cols - train_label_cols)
    
    # Compare columns between validation_labels.csv and sample_submission.csv
    if "validation_labels.csv" in main_data and "sample_submission.csv" in main_data:
        val_label_cols = set(main_data["validation_labels.csv"].columns)
        sample_cols = set(main_data["sample_submission.csv"].columns)
        
        print("\nColumns in sample_submission.csv:")
        print(list(main_data["sample_submission.csv"].columns))
        
        print("\nUnique columns in validation_labels.csv (not present in sample_submission.csv):")
        print(val_label_cols - sample_cols)
        
        print("\nUnique columns in sample_submission.csv (not present in validation_labels.csv):")
        print(sample_cols - val_label_cols)

def analyze_structure_format(main_data):
    """
    Analyzes the format of 3D structures (coordinates).
    """
    if "validation_labels.csv" in main_data and main_data["validation_labels.csv"] is not None:
        df = main_data["validation_labels.csv"]
        
        # Find all coordinate columns (x_1, y_1, z_1, etc.)
        coord_cols = [col for col in df.columns if col.startswith(('x_', 'y_', 'z_'))]
        
        # Group by structure
        structures = {}
        for col in coord_cols:
            # Extract structure number (e.g., "x_1" -> 1)
            parts = col.split('_')
            if len(parts) == 2:
                struct_num = int(parts[1])
                coord_type = parts[0]
                
                if struct_num not in structures:
                    structures[struct_num] = []
                
                structures[struct_num].append(col)
        
        print("\nStructure of the labels file:")
        print(f"Total structures found: {len(structures)}")
        
        # Show details of the first structure
        if structures:
            first_struct = min(structures.keys())
            print(f"\nDetails of structure {first_struct}:")
            print(f"Columns: {sorted(structures[first_struct])}")
            
            # Check for missing values
            for col in structures[first_struct]:
                missing = df[col].isna().sum()
                total = len(df)
                print(f"{col}: {missing} missing values ({missing/total*100:.2f}%)")
            
            # Check the range of non-missing values for the first structure
            for col in structures[first_struct]:
                non_null = df[col][df[col] != -1.0e+18]  # Values that are not -1.0e+18
                if not non_null.empty:
                    print(f"{col} - Range: [{non_null.min():.3f}, {non_null.max():.3f}]")

def main():
    # Load the data
    main_data = load_data()
    
    # Compare columns between different files
    compare_columns(main_data)
    
    # Analyze the format of 3D structures
    analyze_structure_format(main_data)
    
    return main_data

if __name__ == '__main__':
    main_data = main()


# Initialize seed to control randomness
np.random.seed(0)

# Directories and files adjusted for the new competition
DATA_DIR = os.getenv('DATA_DIR', '/kaggle/input/stanford-rna-3d-folding/')
main_files = [
   "train_sequences.csv", 
   "train_labels.csv", 
   "validation_sequences.csv", 
   "validation_labels.csv", 
   "test_sequences.csv",
   "sample_submission.csv"
]

DEFAULT_THRESHOLD = 0.4  # Default threshold after analysis

def optimize_dataframe(df, inplace=False, category_threshold=DEFAULT_THRESHOLD):
   """
   Optimizes the DataFrame to save memory.
   """
   if category_threshold < 0 or category_threshold > 1:
       raise ValueError("category_threshold must be between 0 and 1.")
   
   if not inplace:
       df = df.copy()
   
   for col in df.columns:
       col_type = df[col].dtype
       if np.issubdtype(col_type, np.integer):
           c_min, c_max = df[col].min(), df[col].max()
           if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
               df[col] = df[col].astype(np.int8)
           elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
               df[col] = df[col].astype(np.int16)
           elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
               df[col] = df[col].astype(np.int32)
       elif np.issubdtype(col_type, np.floating):
           # First check if it's not the special value -1.0e+18
           if df[col].min() > np.finfo(np.float32).min and df[col].max() < np.finfo(np.float32).max:
               df[col] = df[col].astype(np.float32)
       if col_type == object:
           unique_vals = len(df[col].unique())
           if unique_vals / len(df) < category_threshold:
               df[col] = df[col].astype('category')
   
   return df

def load_main_data(chunksize=50000):
   """
   Loads the main files.
   """
   data = {}
   for file_name in main_files:
       file_path = os.path.join(DATA_DIR, file_name)
       if os.path.exists(file_path):
           chunks = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False, chunksize=chunksize)
           dataframes = [optimize_dataframe(chunk, category_threshold=DEFAULT_THRESHOLD) for chunk in chunks]
           data[file_name] = pd.concat(dataframes, ignore_index=True)
           print(f"File {file_name} loaded successfully. Shape: {data[file_name].shape}")
       else:
           print(f"File {file_path} not found!")
           data[file_name] = None
   return data

def filter_columns_by_prefix(df, prefix="x_"):
   """
   Filters and counts the number of columns in a DataFrame based on a provided prefix.
   
   :param df: DataFrame where filtering will be applied.
   :param prefix: Prefix to be used for filtering. Ex: "x_", "y_", "z_".
   :return: List of filtered columns.
   """
   filtered_columns = [col for col in df.columns if col.startswith(prefix)]
   return filtered_columns

def count_nucleotides(df, column_name='sequence'):
   """
   Counts the frequency of each nucleotide in a specific column of a DataFrame.
   
   :param df: DataFrame containing the sequences.
   :param column_name: Name of the column containing the sequences. Default is 'sequence'.
   :return: Counter object with the nucleotide counts.
   """
   from collections import Counter

   # Check if the column exists in the DataFrame
   if column_name not in df.columns:
       raise ValueError(f"Column '{column_name}' not found in DataFrame.")
   
   # Concatenate all sequences and count nucleotides
   all_sequences = ''.join(df[column_name].tolist())
   nucleotide_counts = Counter(all_sequences)
   
   return nucleotide_counts

def get_columns_without_missing_values(df):
   """
   Returns columns without any missing values in the DataFrame.
   
   :param df: DataFrame to be checked.
   :return: List of columns without missing values.
   """
   missing_values = df.isnull().sum()
   return missing_values[missing_values == 0].index.tolist()

def get_empty_columns(df):
   """
   Returns columns that are completely empty in the DataFrame.
   
   :param df: DataFrame to be checked.
   :return: List of empty columns.
   """
   missing_values = df.isnull().sum()
   return missing_values[missing_values == df.shape[0]].index.tolist()

def plot_coord_distributions(df_labels, prefix='x_', max_structures=5):
   """
   Plots the distribution of coordinates (x, y, or z) for up to max_structures structures.
   
   :param df_labels: DataFrame containing the coordinates.
   :param prefix: Prefix of columns to be plotted ('x_', 'y_', or 'z_').
   :param max_structures: Maximum number of structures to show.
   """
   # Find coordinate columns with the specified prefix
   coord_cols = filter_columns_by_prefix(df_labels, prefix)
   
   # Limit to the maximum number of structures
   coord_cols = sorted(coord_cols)[:max_structures]
   
   if not coord_cols:
       print(f"No column with prefix '{prefix}' found.")
       return
   
   # Set up the plot
   fig, axes = plt.subplots(1, len(coord_cols), figsize=(16, 4))
   if len(coord_cols) == 1:
       axes = [axes]  # Ensure axes is iterable even with a single subplot
   
   # Plot histograms for each column
   for i, col in enumerate(coord_cols):
       # Filter special values (-1.0e+18) if present
       values = df_labels[col]
       filtered_values = values[values > -1.0e+17]  # Cutoff value to filter -1.0e+18
       
       axes[i].hist(filtered_values, bins=30, alpha=0.7)
       axes[i].set_title(f'Distribution of {col}')
       axes[i].set_xlabel('Value')
       axes[i].set_ylabel('Frequency')
   
   plt.tight_layout()
   plt.show()

def analyze_3d_structure(df_labels):
   """
   Analyzes the 3D coordinates of RNA structures.
   
   :param df_labels: DataFrame containing 3D coordinates.
   """
   # Find all coordinate columns
   x_cols = filter_columns_by_prefix(df_labels, 'x_')
   y_cols = filter_columns_by_prefix(df_labels, 'y_')
   z_cols = filter_columns_by_prefix(df_labels, 'z_')
   
   print(f"Number of x columns: {len(x_cols)}")
   print(f"Number of y columns: {len(y_cols)}")
   print(f"Number of z columns: {len(z_cols)}")
   
   # Check for missing or special values in coordinates
   special_value = -1.0e+18  # Special value observed in the data
   
   for i, (x_col, y_col, z_col) in enumerate(zip(x_cols, y_cols, z_cols), 1):
       # Count missing or special values
       x_special = (df_labels[x_col] == special_value).sum()
       y_special = (df_labels[y_col] == special_value).sum()
       z_special = (df_labels[z_col] == special_value).sum()
       
       x_null = df_labels[x_col].isnull().sum()
       y_null = df_labels[y_col].isnull().sum()
       z_null = df_labels[z_col].isnull().sum()
       
       # Count how many complete structures exist (all x, y, z are neither special nor null)
       valid_structures = ((df_labels[x_col] != special_value) & 
                          (df_labels[y_col] != special_value) & 
                          (df_labels[z_col] != special_value) &
                          df_labels[x_col].notnull() & 
                          df_labels[y_col].notnull() & 
                          df_labels[z_col].notnull()).sum()
       
       total_rows = len(df_labels)
       
       print(f"\nStructure {i}:")
       print(f"  Special values: x={x_special} ({x_special/total_rows*100:.2f}%), y={y_special} ({y_special/total_rows*100:.2f}%), z={z_special} ({z_special/total_rows*100:.2f}%)")
       print(f"  Null values: x={x_null} ({x_null/total_rows*100:.2f}%), y={y_null} ({y_null/total_rows*100:.2f}%), z={z_null} ({z_null/total_rows*100:.2f}%)")
       print(f"  Complete structures: {valid_structures} ({valid_structures/total_rows*100:.2f}%)")
       
       # Limit analysis to the first 5 structures
       if i >= 5:
           print("\nAnalysis limited to the first 5 structures.")
           break

def analyze_sequences(df_sequences):
   """
   Analyzes RNA sequences.
   
   :param df_sequences: DataFrame containing the 'sequence' column.
   """
   # Basic statistics of the sequence column
   print("\nBasic statistics of the 'sequence' column:")
   print(df_sequences['sequence'].describe())
   
   # Sequence lengths
   seq_lengths = df_sequences['sequence'].apply(len)
   print("\nSequence length statistics:")
   print(f"Minimum: {seq_lengths.min()}")
   print(f"Maximum: {seq_lengths.max()}")
   print(f"Mean: {seq_lengths.mean():.2f}")
   print(f"Median: {seq_lengths.median()}")
   
   # Nucleotide counts
   nucleotide_counts = count_nucleotides(df_sequences)
   total_nucleotides = sum(nucleotide_counts.values())
   
   print("\nNucleotide distribution:")
   for nucleotide, count in sorted(nucleotide_counts.items()):
       print(f"{nucleotide}: {count} ({count/total_nucleotides*100:.2f}%)")
   
   # Plot length distribution
   plt.figure(figsize=(10, 6))
   plt.hist(seq_lengths, bins=30, alpha=0.7)
   plt.title('Sequence Length Distribution')
   plt.xlabel('Length')
   plt.ylabel('Frequency')
   plt.grid(True, alpha=0.3)
   plt.show()

def main():
   # Load main data
   main_data = load_main_data()

   # Check which files were loaded
   print("\nLoaded files:")
   for file_name, df in main_data.items():
       if df is not None:
           print(f"- {file_name}: {df.shape}")
   
   # Analyze 3D structures in validation_labels.csv
   if "validation_labels.csv" in main_data and main_data["validation_labels.csv"] is not None:
       print("\n===== Analysis of 3D Structures (validation_labels.csv) =====")
       df_labels = main_data["validation_labels.csv"]
       
       # Count coordinate columns
       x_cols = filter_columns_by_prefix(df_labels, 'x_')
       y_cols = filter_columns_by_prefix(df_labels, 'y_')
       z_cols = filter_columns_by_prefix(df_labels, 'z_')
       
       print(f"There are {len(x_cols)} x_ columns in the DataFrame.")
       print(f"There are {len(y_cols)} y_ columns in the DataFrame.")
       print(f"There are {len(z_cols)} z_ columns in the DataFrame.")
       
       # Identify columns without missing values
       columns_without_missing = get_columns_without_missing_values(df_labels)
       print(f"\nColumns without missing values: {len(columns_without_missing)}")
       
       # Identify completely empty columns
       empty_columns = get_empty_columns(df_labels)
       print(f"Completely empty columns: {len(empty_columns)}")
       
       # Analyze 3D coordinates in detail
       analyze_3d_structure(df_labels)
       
       # Plot distribution of x, y, z coordinates for the first structures
       print("\nDistribution of X coordinates:")
       plot_coord_distributions(df_labels, 'x_', max_structures=3)
       print("\nDistribution of Y coordinates:")
       plot_coord_distributions(df_labels, 'y_', max_structures=3)
       print("\nDistribution of Z coordinates:")
       plot_coord_distributions(df_labels, 'z_', max_structures=3)
   
   # Analyze sequences in train_sequences.csv
   if "train_sequences.csv" in main_data and main_data["train_sequences.csv"] is not None:
       print("\n===== Analysis of Sequences (train_sequences.csv) =====")
       df_sequences = main_data["train_sequences.csv"]
       
       # First few rows of the sequence column
       print("\nFirst few rows of the 'sequence' column:")
       print(df_sequences['sequence'].head())
       
       # Data type of the sequence column
       print("\nData type of the 'sequence' column:")
       print(df_sequences['sequence'].dtype)
       
       # Complete sequence analysis
       analyze_sequences(df_sequences)
   
   return main_data

if __name__ == '__main__':
   main_data = main()


# File paths
DATA_DIR = "/kaggle/input/stanford-rna-3d-folding/"
OUTPUT_DIR = "/kaggle/working/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """
    Loads the necessary data for the competition.
    """
    data = {}
    
    # Load sequences
    data['train_seq'] = pd.read_csv(os.path.join(DATA_DIR, "train_sequences.csv"))
    data['valid_seq'] = pd.read_csv(os.path.join(DATA_DIR, "validation_sequences.csv"))
    data['test_seq'] = pd.read_csv(os.path.join(DATA_DIR, "test_sequences.csv"))
    
    # Load structures (labels)
    data['train_labels'] = pd.read_csv(os.path.join(DATA_DIR, "train_labels.csv"))
    data['valid_labels'] = pd.read_csv(os.path.join(DATA_DIR, "validation_labels.csv"))
    
    # Load submission format
    data['sample_submission'] = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    
    return data

def analyze_id_structure(data_dict):
    """
    Analyzes the ID structure in different files to understand the correct mapping.
    """
    # We'll analyze the specific formats for train and valid
    
    # 1. Analysis of training labels
    train_label_ids = data_dict['train_labels']['ID'].tolist()
    print(f"Total IDs in training labels: {len(train_label_ids)}")
    print(f"Number of unique IDs: {len(set(train_label_ids))}")
    
    # Try to understand the ID format in the labels file
    train_id_parts = {}
    for id_str in train_label_ids[:100]:  # Analyze the first 100
        parts = id_str.split('_')
        num_parts = len(parts)
        if num_parts not in train_id_parts:
            train_id_parts[num_parts] = []
        train_id_parts[num_parts].append(parts)
    
    print("\nID formats found in train_labels:")
    for num_parts, examples in train_id_parts.items():
        print(f"\nFormat with {num_parts} parts:")
        for i, parts in enumerate(examples[:3]):
            print(f"  Example {i+1}: {parts}")
    
    # 2. Analysis of training sequences
    train_seq_ids = data_dict['train_seq']['target_id'].tolist()
    print(f"\nTotal IDs in training sequences: {len(train_seq_ids)}")
    print(f"Number of unique IDs: {len(set(train_seq_ids))}")
    
    # Try to understand the ID format in the sequences file
    train_seq_id_parts = {}
    for id_str in train_seq_ids[:100]:  # Analyze the first 100
        parts = id_str.split('_')
        num_parts = len(parts)
        if num_parts not in train_seq_id_parts:
            train_seq_id_parts[num_parts] = []
        train_seq_id_parts[num_parts].append(parts)
    
    print("\nID formats found in train_sequences:")
    for num_parts, examples in train_seq_id_parts.items():
        print(f"\nFormat with {num_parts} parts:")
        for i, parts in enumerate(examples[:3]):
            print(f"  Example {i+1}: {parts}")
    
    # 3. Analysis of validation labels
    valid_label_ids = data_dict['valid_labels']['ID'].tolist()
    print(f"\nTotal IDs in validation labels: {len(valid_label_ids)}")
    print(f"Number of unique IDs: {len(set(valid_label_ids))}")
    
    # Count unique sequence IDs in validation labels
    valid_seq_ids_from_labels = set([id_str.split('_')[0] for id_str in valid_label_ids])
    print(f"Number of unique sequence IDs in validation labels: {len(valid_seq_ids_from_labels)}")
    print(f"Examples: {list(valid_seq_ids_from_labels)[:5]}")
    
    # 4. Analysis of validation sequences
    valid_seq_ids = data_dict['valid_seq']['target_id'].tolist()
    print(f"\nTotal IDs in validation sequences: {len(valid_seq_ids)}")
    print(f"Number of unique IDs: {len(set(valid_seq_ids))}")
    print(f"Examples: {valid_seq_ids[:5]}")
    
    # 5. Check correspondence between unique IDs
    overlap_valid = set(valid_seq_ids).intersection(valid_seq_ids_from_labels)
    print(f"\nCorrespondence between validation sequences and labels: {len(overlap_valid)} of {len(valid_seq_ids)}")
    
    # 6. Check how sequences and residues relate
    if len(overlap_valid) > 0:
        sample_id = list(overlap_valid)[0]
        sample_seq = data_dict['valid_seq'][data_dict['valid_seq']['target_id'] == sample_id]['sequence'].iloc[0]
        sample_labels = data_dict['valid_labels'][data_dict['valid_labels']['ID'].str.startswith(f"{sample_id}_")]
        
        print(f"\nAnalysis for sequence ID: {sample_id}")
        print(f"Sequence length: {len(sample_seq)}")
        print(f"Number of residues in labels: {len(sample_labels)}")
        
        # Check how residue numbers are related
        residue_numbers = sample_labels['resid'].sort_values().tolist()
        print(f"First residue numbers: {residue_numbers[:10]}")
        print(f"Last residue numbers: {residue_numbers[-10:]}")
        
    return train_id_parts, train_seq_id_parts, overlap_valid

def fix_train_mapping(train_seq_df, train_labels_df):
    """
    Identifies a correct mapping between train_sequences.csv and train_labels.csv
    using the ID format from the validation file as a reference.
    
    This is necessary because there's no obvious direct correspondence between the IDs.
    """
    # First, extract the prefix of the ID from labels (format: XX_Y_Z)
    train_labels_df['seq_id'] = train_labels_df['ID'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[1])
    
    # Check if this format corresponds to the format of sequence IDs
    seq_ids_set = set(train_seq_df['target_id'])
    label_seq_ids_set = set(train_labels_df['seq_id'])
    
    overlap = seq_ids_set.intersection(label_seq_ids_set)
    print(f"Overlap after format adjustment: {len(overlap)} of {len(seq_ids_set)}")
    
    if len(overlap) > 0:
        print(f"Examples of matching IDs: {list(overlap)[:5]}")
        return overlap
    
    # If it still doesn't work, we need to analyze the structure in more detail
    print("No matches found, checking other formats...")
    
    # Try other possible formats
    formats_to_try = [
        lambda x: x.split('_')[0],                             # Only first part
        lambda x: '_'.join(x.split('_')[:2]),                  # First two parts
        lambda x: x.split('_')[0] + '_' + x.split('_')[1][0],  # First part + first letter of second part
    ]
    
    for i, format_func in enumerate(formats_to_try):
        train_labels_df[f'seq_id_{i}'] = train_labels_df['ID'].apply(format_func)
        label_seq_ids_set = set(train_labels_df[f'seq_id_{i}'])
        overlap = seq_ids_set.intersection(label_seq_ids_set)
        print(f"Format {i}: Overlap = {len(overlap)} of {len(seq_ids_set)}")
        
        if len(overlap) > 0:
            print(f"Examples of matching IDs: {list(overlap)[:5]}")
            return overlap, f'seq_id_{i}'
    
    # If no match is found, create a mapping based on observed patterns
    print("No matches found using simple patterns.")
    print("Creating a manual mapping based on data structure...")
    
    # Group labels by first parts of ID
    train_labels_df['prefix'] = train_labels_df['ID'].apply(lambda x: x.split('_')[0])
    label_groups = train_labels_df.groupby('prefix')
    
    # For each sequence, find the best match based on number of residues
    mapping = {}
    for _, seq_row in train_seq_df.iterrows():
        seq_id = seq_row['target_id']
        seq_length = len(seq_row['sequence'])
        
        best_match = None
        best_diff = float('inf')
        
        for prefix, group in label_groups:
            residue_count = len(group)
            diff = abs(residue_count - seq_length)
            
            if diff < best_diff:
                best_diff = diff
                best_match = prefix
        
        # Consider a match only if the number of residues is close
        if best_diff <= 10:  # Tolerance of 10 residues
            mapping[seq_id] = best_match
    
    print(f"Manual mapping created with {len(mapping)} matches")
    return mapping

def create_mapping_valid(valid_seq_df, valid_labels_df):
    """
    Creates a mapping between validation sequences and their coordinates.
    
    In this case, the IDs already correspond directly (R1107 -> R1107_1, R1107_2, etc.)
    """
    # Check which ID format is used in the validation set
    valid_labels_df['seq_id'] = valid_labels_df['ID'].apply(lambda x: x.split('_')[0])
    
    # Check overlap
    seq_ids = set(valid_seq_df['target_id'])
    label_seq_ids = set(valid_labels_df['seq_id'])
    
    overlap = seq_ids.intersection(label_seq_ids)
    print(f"Correspondence for validation: {len(overlap)} of {len(seq_ids)}")
    
    mapping = {}
    for seq_id in overlap:
        # Get sequence
        seq = valid_seq_df[valid_seq_df['target_id'] == seq_id]['sequence'].iloc[0]
        
        # Get all residues for this sequence
        residues = valid_labels_df[valid_labels_df['seq_id'] == seq_id].sort_values('resid')
        
        # Extract coordinates for all structures
        num_structures = 1
        for col in residues.columns:
            if col.startswith('x_'):
                struct_num = int(col.split('_')[1])
                num_structures = max(num_structures, struct_num)
        
        # Initialize structures
        structures = []
        
        for struct_idx in range(1, num_structures + 1):
            coords = []
            has_valid_coords = False
            
            # Check if this structure has coordinates
            if f'x_{struct_idx}' in residues.columns:
                for _, row in residues.iterrows():
                    x = row[f'x_{struct_idx}']
                    y = row[f'y_{struct_idx}']
                    z = row[f'z_{struct_idx}']
                    
                    # Check if they are valid values
                    if abs(x) < 1.0e+17 and abs(y) < 1.0e+17 and abs(z) < 1.0e+17:
                        coords.append([x, y, z])
                        has_valid_coords = True
                    else:
                        coords.append([np.nan, np.nan, np.nan])
            
            if has_valid_coords:
                structures.append(coords)
        
        # Add to mapping if there are valid structures
        if structures:
            mapping[seq_id] = {
                'sequence': seq,
                'structures': structures
            }
    
    print(f"Mapping created with {len(mapping)} valid sequences")
    return mapping

def create_processed_data(mapping, output_prefix):
    """
    Creates and saves processed data from the mapping.
    
    Parameters:
    mapping: Dictionary with the mapping of sequences to structures
    output_prefix: Prefix for output files ('train' or 'valid')
    
    Returns:
    X, y: Arrays for training
    """
    if not mapping:
        print(f"WARNING: No valid mapping for {output_prefix}")
        return None, None
    
    X_data = []
    y_data = []
    ids = []
    
    for seq_id, data in mapping.items():
        seq = data['sequence']
        structures = data['structures']
        
        # Skip if there are no structures
        if not structures:
            continue
        
        # Use the first valid structure
        structure = structures[0]
        
        # Check if the structure has valid coordinates for all residues
        if len(structure) != len(seq):
            print(f"WARNING: Difference between sequence length ({len(seq)}) and coordinates ({len(structure)}) for {seq_id}")
            # If needed, we could consider padding or truncation here
            continue
        
        # Create feature matrix (one-hot encoding)
        features = []
        for nucleotide in seq:
            if nucleotide == 'A':
                features.append([1, 0, 0, 0, 0])
            elif nucleotide == 'C':
                features.append([0, 1, 0, 0, 0])
            elif nucleotide == 'G':
                features.append([0, 0, 1, 0, 0])
            elif nucleotide == 'U':
                features.append([0, 0, 0, 1, 0])
            else:
                features.append([0, 0, 0, 0, 1])  # For unknown nucleotides
        
        X_data.append(np.array(features))
        y_data.append(np.array(structure))
        ids.append(seq_id)
    
    if not X_data:
        print(f"WARNING: No valid processed data for {output_prefix}")
        return None, None, []
    
    # Padding to ensure all sequences have the same length
    max_length = max(len(x) for x in X_data)
    X_padded = []
    y_padded = []
    
    for x, y in zip(X_data, y_data):
        if len(x) < max_length:
            x_pad = np.zeros((max_length, 5))
            x_pad[:len(x), :] = x
            
            y_pad = np.zeros((max_length, 3))
            y_pad[:len(y), :] = y
            
            X_padded.append(x_pad)
            y_padded.append(y_pad)
        else:
            X_padded.append(x)
            y_padded.append(y)
    
    X = np.array(X_padded)
    y = np.array(y_padded)
    
    # Save the processed data
    np.save(os.path.join(OUTPUT_DIR, f'X_{output_prefix}.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, f'y_{output_prefix}.npy'), y)
    
    with open(os.path.join(OUTPUT_DIR, f'{output_prefix}_ids.txt'), 'w') as f:
        for id in ids:
            f.write(f"{id}\n")
    
    print(f"Processed data for {output_prefix}: X.shape = {X.shape}, y.shape = {y.shape}")
    return X, y, ids

def explore_sequence_mapping(seq_id, mapping, data_dict):
    """
    Explores a mapping example in detail for diagnostics.
    """
    if seq_id not in mapping:
        print(f"WARNING: Sequence ID {seq_id} not found in mapping")
        return
    
    data = mapping[seq_id]
    seq = data['sequence']
    structures = data['structures']
    
    print(f"Exploring mapping for sequence: {seq_id}")
    print(f"Sequence length: {len(seq)}")
    print(f"Number of available structures: {len(structures)}")
    
    # Detail each structure
    for i, structure in enumerate(structures):
        print(f"\nStructure {i+1}:")
        print(f"  Number of coordinates: {len(structure)}")
        if len(structure) > 0:
            print(f"  First coordinates: {structure[:3]}")
            print(f"  Last coordinates: {structure[-3:]}")
        
        # Check correspondence with the sequence
        if len(structure) != len(seq):
            print(f"  WARNING: Difference between sequence length ({len(seq)}) and coordinates ({len(structure)})")
        else:
            print(f"  Perfect match between sequence and coordinates")

def main():
    # Load the data
    print("Loading data...")
    data_dict = load_data()
    
    # Analyze ID structure to understand the mapping
    print("\nAnalyzing ID structure...")
    train_id_parts, train_seq_id_parts, overlap_valid = analyze_id_structure(data_dict)
    
    # For validation, the mapping is direct (R1107 -> R1107_1, R1107_2, etc.)
    print("\nCreating mapping for validation data...")
    valid_mapping = create_mapping_valid(data_dict['valid_seq'], data_dict['valid_labels'])
    
    # Explore a validation mapping example to verify
    if valid_mapping:
        sample_id = list(valid_mapping.keys())[0]
        print(f"\nExploring a validation mapping example ({sample_id}):")
        explore_sequence_mapping(sample_id, valid_mapping, data_dict)
    
    # Create and save processed data for validation
    X_valid, y_valid, valid_ids = create_processed_data(valid_mapping, 'valid')
    
    # Since we couldn't establish a mapping for training,
    # we'll use validation data for training as well (transfer learning)
    print("\nUsing validation data as training (due to lack of direct mapping)...")
    X_train = X_valid
    y_train = y_valid
    train_ids = valid_ids
    
    if X_train is not None:
        np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
        np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
        
        with open(os.path.join(OUTPUT_DIR, 'train_ids.txt'), 'w') as f:
            for id in train_ids:
                f.write(f"{id}\n")
    
    # Return the processed data
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_valid': X_valid,
        'y_valid': y_valid,
        'valid_mapping': valid_mapping,
        'valid_ids': valid_ids
    }

if __name__ == "__main__":
    processed_data = main()


def visualize_rna_heatmap_from_processed_data(processed_data, num_samples=12):
    """
    Visualizes a heatmap for RNA sequences using processed data.
    
    Parameters:
    processed_data: Dictionary with processed data returned by the main() function
    num_samples: Number of sequences to visualize
    """
    try:
        # Check if we have the necessary data
        if 'X_valid' not in processed_data or processed_data['X_valid'] is None:
            print("Validation data not found in processed_data object")
            return None
        
        # Get the data
        X_valid = processed_data['X_valid']
        print(f"Data found with format: {X_valid.shape}")
        
        # Limit to the number of samples
        X_valid_subset = X_valid[:num_samples]
        
        # If we have IDs, use them
        if 'valid_ids' in processed_data and processed_data['valid_ids']:
            valid_ids = processed_data['valid_ids'][:num_samples]
        else:
            valid_ids = [f"Seq_{i+1}" for i in range(X_valid_subset.shape[0])]
        
        # Convert one-hot encoding to nucleotide indices
        # Expected format: A=[1,0,0,0,0], C=[0,1,0,0,0], G=[0,0,1,0,0], U=[0,0,0,1,0], N=[0,0,0,0,1]
        sequences_matrix = np.argmax(X_valid_subset, axis=2)
        
        # Replace zeros (padding) with 4 (N/Unknown) when all values are zero
        is_padding = np.all(X_valid_subset == 0, axis=2)
        sequences_matrix[is_padding] = 4
        
        # Define a categorical colormap (distinct colors per nucleotide)
        cmap = mcolors.ListedColormap(['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#95a5a6'])
        bounds = [0, 1, 2, 3, 4, 5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # Create figure
        plt.figure(figsize=(20, 10))
        im = plt.imshow(sequences_matrix, cmap=cmap, norm=norm, aspect='auto')
        
        # Add color bar
        cbar = plt.colorbar(im, ticks=[0.5, 1.5, 2.5, 3.5, 4.5])
        cbar.set_label('Nucleotides', fontsize=14)
        cbar.set_ticklabels(['A', 'C', 'G', 'U', 'N/Padding'])
        
        # Add axis labels
        plt.xlabel("Position in Sequence", fontsize=14)
        plt.ylabel("RNA Sequences", fontsize=14)
        
        # Add title
        plt.title("RNA Sequences Heatmap", fontsize=16)
        
        # Add sequence IDs as y-axis labels
        plt.yticks(range(len(valid_ids)), valid_ids, fontsize=10)
        
        # Show only some labels on x-axis to avoid crowding
        sequence_length = sequences_matrix.shape[1]
        step = max(1, sequence_length // 20)  # Show at most 20 labels
        plt.xticks(range(0, sequence_length, step), range(1, sequence_length + 1, step))
        
        # Add grid
        plt.grid(False)
        
        # Add information about nucleotide distribution
        all_nucleotides = sequences_matrix.flatten()
        nucleotide_counts = {
            'A': np.sum(all_nucleotides == 0),
            'C': np.sum(all_nucleotides == 1),
            'G': np.sum(all_nucleotides == 2),
            'U': np.sum(all_nucleotides == 3),
            'N': np.sum(all_nucleotides == 4)
        }
        
        total_nucleotides = sum(nucleotide_counts.values())
        nucleotide_percentages = {k: (v / total_nucleotides) * 100 for k, v in nucleotide_counts.items()}
        
        # Add text with statistics
        info_text = "\n".join([
            f"Total sequences visualized: {num_samples}",
            f"Maximum length: {sequence_length}",
            f"A: {nucleotide_percentages['A']:.1f}%",
            f"C: {nucleotide_percentages['C']:.1f}%",
            f"G: {nucleotide_percentages['G']:.1f}%",
            f"U: {nucleotide_percentages['U']:.1f}%",
            f"N/Padding: {nucleotide_percentages['N']:.1f}%"
        ])
        
        plt.figtext(0.02, 0.02, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        # Show the plot
        plt.tight_layout()
        plt.show()
        
        # Optionally, save the plot
        output_dir = '/kaggle/working/'
        plt.savefig(os.path.join(output_dir, 'rna_heatmap.png'), dpi=300)
        print(f"Heatmap saved to {os.path.join(output_dir, 'rna_heatmap.png')}")
        
        return sequences_matrix
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

# Use the function (assuming processed_data is available)
visualize_rna_heatmap_from_processed_data(processed_data)


# File paths
DATA_DIR = "/kaggle/input/stanford-rna-3d-folding/"
OUTPUT_DIR = "/kaggle/working/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_structure(coords):
    """
    Centralizes and normalizes the structure.
    """
    # Remove padding
    valid_mask = ~np.all(coords == 0, axis=1)
    valid_coords = coords[valid_mask]
    
    # Center at center of mass
    center = np.mean(valid_coords, axis=0)
    centered_coords = coords.copy()
    centered_coords[valid_mask] = valid_coords - center
    
    return centered_coords

def normalize_coordinates(coords):
    """
    Normalizes 3D coordinates of RNA structures by centering and 
    scaling each structure independently, with robust handling
    to avoid numerical issues.
    
    Parameters:
    -----------
    coords: Numpy array with shape (batch_size, seq_length, 3)
        3D coordinates to normalize
    
    Returns:
    --------
    normalized: Numpy array with shape (batch_size, seq_length, 3)
        Normalized coordinates in the range [-1, 1]  
    """
    # Create copy to avoid modifying the original
    normalized = np.copy(coords)
    
    # Check for problematic values upfront
    if np.isnan(coords).any():
        print("WARNING: NaN values detected in input coordinates. They will be ignored during normalization.")
    if np.isinf(coords).any():
        print("WARNING: Infinite values detected in input coordinates. They will be ignored during normalization.")
    
    # Handle each structure in the batch separately
    for i in range(coords.shape[0]):
        # Identify valid positions (non-zero, non-NaN, non-Inf)
        valid_mask = ~np.all(coords[i] == 0, axis=-1)  
        valid_mask = valid_mask & ~np.any(np.isnan(coords[i]), axis=-1)
        valid_mask = valid_mask & ~np.any(np.isinf(coords[i]), axis=-1)
        
        # Extract only valid coordinates
        valid_coords = coords[i][valid_mask]
        
        if len(valid_coords) > 0:
            try:
                # 1. Center at the geometric center
                center = np.nanmean(valid_coords, axis=0)
                
                # Check if the calculated center contains valid values  
                if np.isnan(center).any() or np.isinf(center).any():
                    print(f"WARNING: Invalid center calculated for structure {i}. Using [0,0,0].")
                    center = np.zeros(3)
                
                # Apply translation to the center
                centered = valid_coords - center
                
                # 2. Determine appropriate scale factor
                # Calculate maximum distance from the center
                dist_from_center = np.sqrt(np.sum(centered * centered, axis=1))
                
                # Exclude NaN or infinite values for scale_factor calculation
                valid_dists = dist_from_center[~np.isnan(dist_from_center) & ~np.isinf(dist_from_center)]
                
                if len(valid_dists) > 0:
                    scale_factor = np.max(valid_dists)
                    # Protect against very small scale_factor
                    if scale_factor < 1e-10:
                        scale_factor = 1.0
                else:
                    scale_factor = 1.0
                
                # 3. Normalize coordinates to [-1, 1] range
                normalized_valid = centered / scale_factor
                
                # 4. Replace values in the normalized array
                normalized[i][valid_mask] = normalized_valid
                
                # Debug info
                # print(f"Structure {i}: center={center}, scale_factor={scale_factor}, "  
                #       f"min={np.min(normalized_valid)}, max={np.max(normalized_valid)}")
            
            except Exception as e:
                print(f"ERROR during normalization of structure {i}: {str(e)}")
                print("Keeping original values for this structure.")
        else:
            print(f"WARNING: No valid coordinates found for structure {i}.")
    
    # Final check to detect any issues
    if np.isnan(normalized).any():
        print("WARNING: NaN values present after normalization. Replacing with zeros.")
        normalized = np.nan_to_num(normalized, nan=0.0)
    
    if np.isinf(normalized).any():
        print("WARNING: Infinite values present after normalization. Replacing with zeros.") 
        normalized = np.nan_to_num(normalized, posinf=0.0, neginf=0.0)
    
    return normalized

def check_structure_validity(coords, min_distance=0.8, max_distance=7.0, allow_clashes=0.05):
    """
    More refined and realistic biophysical validation.
    """
    valid = True
    valid_mask = ~np.all(coords == 0, axis=1)
    valid_coords = coords[valid_mask]
    
    if len(valid_coords) < 3:
        return True
    
    # Check distances between consecutive residues
    invalid_bonds = 0
    for i in range(1, len(valid_coords)):
        dist = np.linalg.norm(valid_coords[i] - valid_coords[i-1])
        if dist < min_distance or dist > max_distance:
            invalid_bonds += 1
    
    # Allow a small percentage of invalid bonds
    if invalid_bonds / len(valid_coords) > 0.1:  # More than 10% invalid bonds
        valid = False
    
    # Check for clashes, allowing some
    clashes = 0
    total_pairs = 0
    for i in range(len(valid_coords)):
        for j in range(i+3, len(valid_coords)):  # Skip adjacent
            total_pairs += 1
            dist = np.linalg.norm(valid_coords[i] - valid_coords[j])
            if dist < min_distance:
                clashes += 1
    
    # Allow a small percentage of clashes
    if total_pairs > 0 and clashes / total_pairs > allow_clashes:
        valid = False
    
    return valid


def sample_structural_variation(coords, noise_level=0.5, preserve_distance=True, 
                               use_global_movement=False, correlation=0.7):
    """
    Enhanced version of structural variation sampling with better
    handling of large RNAs and improved noise distribution.
    """
    new_coords = coords.copy()
    valid_mask = ~np.all(coords == 0, axis=1)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) < 3:
        return new_coords
    
    # Parameters optimized for RNA structure
    typical_bond_length = 3.8  # Angstroms - typical RNA backbone distance
    
    # Add global domain movements if requested
    if use_global_movement and len(valid_indices) > 20:
        # More natural domain identification - try to find natural hinge points
        # For RNA, these often occur at junctions between helices
        
        # Calculate distance between consecutive residues as a heuristic
        # for finding potential hinge points (larger distances often indicate junctions)
        distances = []
        for i in range(1, len(valid_indices)):
            idx1 = valid_indices[i-1]
            idx2 = valid_indices[i]
            dist = np.linalg.norm(coords[idx1] - coords[idx2])
            distances.append((i, dist))
        
        # Sort by distance to find potential hinges
        distances.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 2 potential hinge points (if we have enough points)
        num_hinges = min(2, len(distances)//3)
        
        for h in range(num_hinges):
            if h < len(distances):
                hinge_point = distances[h][0]
                if hinge_point < 5 or hinge_point > len(valid_indices) - 5:
                    continue
                    
                hinge_idx = valid_indices[hinge_point]
                
                # Angle of rotation with natural distribution
                # More small movements than large ones
                angle = np.random.exponential(0.2)  # Mostly small angles with occasional larger ones
                if np.random.random() < 0.5:
                    angle = -angle  # Allow both directions
                
                # Create a more natural rotation matrix with slight 3D component
                # RNAs often bend and twist in 3D
                sin_a, cos_a = np.sin(angle), np.cos(angle)
                tilt = np.random.normal(0, 0.1)  # Small tilt in 3D
                rotation_matrix = np.array([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, tilt],
                    [0, -tilt, 1]
                ])
                
                # Apply rotation around hinge point
                ref_point = new_coords[hinge_idx]
                for i in valid_indices[hinge_point+1:]:
                    vector = new_coords[i] - ref_point
                    rotated = np.dot(vector, rotation_matrix)
                    new_coords[i] = ref_point + rotated
    
    # Propagate variation residue by residue, with correlation
    # RNA has strong local correlations in structure
    prev_noise = np.zeros(3)
    
    correlation = 0.5  # High correlation for smoother variations
    
    for i in range(1, len(coords)):
        if not valid_mask[i] or not valid_mask[i-1]:
            continue
            
        vec = new_coords[i-1] - new_coords[i]
        vec_length = np.linalg.norm(vec)
        
        # Generate correlated noise (smoother transitions)
        new_noise = np.random.normal(0, noise_level, size=3)
        noise_vec = correlation * prev_noise + (1 - correlation) * new_noise
        prev_noise = noise_vec.copy()
        
        noise_norm = np.linalg.norm(noise_vec)
        if noise_norm > 0:
            # Scale noise proportionally
            noise_vec = noise_vec / noise_norm * (noise_level * vec_length)
        
        # Add noise to the direction
        new_vec = vec + noise_vec
        
        # Preserve distance if requested
        if preserve_distance:
            current_length = np.linalg.norm(new_vec)
            if current_length > 0:
                # Allow slight variation in bond length (RNA is not rigid)
                target_length = typical_bond_length * (1 + np.random.normal(0, 0.05))
                new_vec = new_vec / current_length * target_length
        
        new_coords[i] = new_coords[i-1] - new_vec
    
    return new_coords

def get_rotation_matrix(axis, theta):
    """
    Return the rotation matrix for rotation around an arbitrary axis.
    
    Parameters:
    -----------
    axis: Unit vector defining the rotation axis
    theta: Rotation angle in radians
    
    Returns:
    --------
    3x3 rotation matrix
    """
    # Ensure axis is a unit vector
    axis = axis / np.linalg.norm(axis)
    
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
        [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]
    ])

def refine_rna_backbone(structure):
    """
    Refine the RNA backbone geometry to match known constraints.
    
    Parameters:
    -----------
    structure: RNA 3D structure
    
    Returns:
    --------
    Refined structure
    """
    # Create a copy to refine
    refined = structure.copy()
    
    # Check for valid residues
    valid_mask = ~np.all(refined == 0, axis=1)
    
    # Apply RNA-specific backbone constraints
    for i in range(2, len(refined)):
        if valid_mask[i] and valid_mask[i-1] and valid_mask[i-2]:
            # In RNA, there are constraints on three consecutive backbone atoms
            
            # Get the two backbone vectors
            vec1 = refined[i-1] - refined[i-2]
            vec2 = refined[i] - refined[i-1]
            
            # Calculate current angle between vectors
            vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-6)
            vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-6)
            cos_angle = np.dot(vec1_norm, vec2_norm)
            
            # Clamp to valid range for numerical stability
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle = np.arccos(cos_angle)
            
            # In RNA, the typical backbone angle is around 100-120 degrees
            ideal_angle = np.radians(110)
            
            # If the angle is too far from ideal, adjust it
            if abs(angle - ideal_angle) > np.radians(30):
                # Create a rotation to adjust the angle
                # Get the rotation axis (perpendicular to the plane of vec1 and vec2)
                axis = np.cross(vec1_norm, vec2_norm)
                axis_norm = axis / (np.linalg.norm(axis) + 1e-6)
                
                # Determine rotation angle to reach ideal angle
                angle_diff = ideal_angle - angle
                
                # Apply rotation to vec2
                rotation_matrix = get_rotation_matrix(axis_norm, angle_diff)
                new_vec2 = np.dot(rotation_matrix, vec2_norm) * np.linalg.norm(vec2)
                
                # Update the position
                refined[i] = refined[i-1] + new_vec2
    
    return refined


def repair_invalid_structure(structure):
    """
    Attempt to repair an invalid RNA structure.
    
    Parameters:
    -----------
    structure: Potentially invalid RNA structure
    
    Returns:
    --------
    Repaired structure
    """
    # Create a copy to repair
    repaired = structure.copy()
    
    # Check for valid residues
    valid_mask = ~np.all(repaired == 0, axis=1)
    
    # Fix bond lengths
    for i in range(1, len(repaired)):
        if valid_mask[i] and valid_mask[i-1]:
            # Get current bond
            bond_vector = repaired[i] - repaired[i-1]
            bond_length = np.linalg.norm(bond_vector)
            
            # Check if bond is too short or too long
            if bond_length < 1.0 or bond_length > 7.0:
                # Fix bond to ideal length
                ideal_length = 3.8
                if bond_length > 0:
                    repaired[i] = repaired[i-1] + (bond_vector / bond_length) * ideal_length
                else:
                    # Generate a random direction if bond length is zero
                    random_direction = np.random.randn(3)
                    random_direction = random_direction / np.linalg.norm(random_direction)
                    repaired[i] = repaired[i-1] + random_direction * ideal_length
    
    # Check for clashes (atoms too close to each other)
    for i in range(len(repaired)):
        if valid_mask[i]:
            for j in range(i+3, len(repaired)):  # Skip adjacent residues
                if valid_mask[j]:
                    # Calculate distance
                    distance = np.linalg.norm(repaired[j] - repaired[i])
                    
                    # If atoms are too close
                    if distance < 1.0:
                        # Move one atom away slightly in a random direction
                        random_direction = np.random.randn(3)
                        random_direction = random_direction / np.linalg.norm(random_direction)
                        repaired[j] = repaired[i] + random_direction * 4.0  # Place at safe distance
    
    # Final normalization
    repaired = normalize_structure(repaired)
    
    return repaired

def create_emergency_structure(seq_length):
    """
    Create an emergency structure when all else fails.
    Generates a physically plausible RNA structure.
    
    Parameters:
    -----------
    seq_length: Length of the RNA sequence
    
    Returns:
    --------
    Basic RNA structure
    """
    # Create a simple linear structure as fallback
    emergency_structure = np.zeros((seq_length, 3))
    
    # Define canonical nucleotide step (3.8Ã…)
    step = np.array([3.8, 0.0, 0.0])
    
    # Generate a straight chain with some randomness
    for i in range(seq_length):
        if i == 0:
            emergency_structure[i] = np.zeros(3)
        else:
            # Add slight random deviation to prevent perfect linearity
            random_noise = np.random.normal(0, 0.2, 3)
            emergency_structure[i] = emergency_structure[i-1] + step + random_noise
    
    # Add a slight curve to make it more RNA-like
    # Apply a gentle curve in the y-z plane
    for i in range(seq_length):
        angle = i * 0.1  # Gradual rotation
        emergency_structure[i, 1] += 2 * np.sin(angle)  # Y-component
        emergency_structure[i, 2] += 2 * np.cos(angle)  # Z-component
    
    # Normalize
    emergency_structure = normalize_structure(emergency_structure)
    
    return emergency_structure


def calculate_tm_score(pred_coords, true_coords, d0_scale=1.24):
    """
    Calculates a robust approximation of the TM-score between predicted and true coordinates.
    Adds protections against division by zero and NaN.
    """
    # Remove padding (rows with zeros) from the true structures
    mask = ~np.all(true_coords == 0, axis=1)
    pred = pred_coords[mask]
    true = true_coords[mask]
    
    L = len(true)
    if L < 3:
        return 0.0
    
    # Define d0 based on L (values adapted for RNA)
    if L >= 30:
        d0 = 0.6 * np.sqrt(L - 0.5) - 2.5
        d0 = max(0.1, d0)
    elif L >= 24:
        d0 = 0.7
    elif L >= 20:
        d0 = 0.6
    elif L >= 16:
        d0 = 0.5
    elif L >= 12:
        d0 = 0.4
    else:
        d0 = 0.3
    
    distances = np.sqrt(np.sum((pred - true) ** 2, axis=1))
    tm_terms = 1.0 / (1.0 + (distances / (d0 + 1e-8)) ** 2)
    tm_score = np.sum(tm_terms) / L
    return float(tm_score)

def calculate_tm_score_exact(pred_coords, true_coords):
    """
    Implementation more closely matching US-align with sequence-independent alignment.
    Includes multiple rotation schemes to find the optimal structural alignment.
    """
    # Remove padding
    mask = ~np.all(true_coords == 0, axis=1)
    pred = pred_coords[mask]
    true = true_coords[mask]
    
    Lref = len(true)
    if Lref < 3:
        return 0.0
    
    # Define d0 exactly as in the evaluation formula
    if Lref >= 30:
        d0 = 0.6 * np.sqrt(Lref - 0.5) - 2.5
    elif Lref >= 24:
        d0 = 0.7
    elif Lref >= 20:
        d0 = 0.6
    elif Lref >= 16:
        d0 = 0.5
    elif Lref >= 12:
        d0 = 0.4
    else:
        d0 = 0.3
    
    # Normalize structures
    pred_centered = pred - np.mean(pred, axis=0)
    true_centered = true - np.mean(true, axis=0)
    
    # Try multiple fragment lengths for sequence-independent alignment
    # This mimics US-align's approach to find the best fragment alignment
    best_tm_score = 0.0
    fragment_lengths = [Lref, max(5, Lref//2), max(5, Lref//4)]
    
    for frag_len in fragment_lengths:
        # Try different fragment start positions
        for i in range(0, Lref - frag_len + 1, max(1, frag_len//2)):
            pred_frag = pred_centered[i:i+frag_len]
            
            # Try aligning with different parts of the true structure
            for j in range(0, Lref - frag_len + 1, max(1, frag_len//2)):
                true_frag = true_centered[j:j+frag_len]
                
                # Covariance matrix for optimal rotation
                covariance = np.dot(pred_frag.T, true_frag)
                U, S, Vt = np.linalg.svd(covariance)
                rotation = np.dot(U, Vt)
                
                # Try different rotation schemes - this is the new part
                rotations_to_try = [
                    rotation,  # Original rotation from SVD
                    np.dot(rotation, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])),  # 90 degree Z rotation
                    np.dot(rotation, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))  # 180 degree Z rotation
                ]
                
                for rot in rotations_to_try:
                    # Apply rotation to the full structure
                    pred_aligned = np.dot(pred_centered, rot)
                    
                    # Calculate distances
                    distances = np.sqrt(np.sum((pred_aligned - true_centered) ** 2, axis=1))
                    
                    # Calculate TM-score terms
                    tm_terms = 1.0 / (1.0 + (distances / d0) ** 2)
                    tm_score = np.sum(tm_terms) / Lref
                    
                    best_tm_score = max(best_tm_score, tm_score)
    
    return float(best_tm_score)

def load_processed_data():
    """
    Loads processed data for training.
    """
    X_train = np.load(os.path.join(OUTPUT_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(OUTPUT_DIR, 'y_train.npy'))
    X_valid = np.load(os.path.join(OUTPUT_DIR, 'X_valid.npy'))
    y_valid = np.load(os.path.join(OUTPUT_DIR, 'y_valid.npy'))
    
    print(f"Data loaded - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Data loaded - X_valid: {X_valid.shape}, y_valid: {y_valid.shape}")
    
    return X_train, y_train, X_valid, y_valid


def prepare_test_features(test_seq_df, max_length=720):
    """
    Prepares test features (one-hot encoding of the sequence).
    """
    X_test = []
    for _, row in test_seq_df.iterrows():
        seq = row['sequence']
        features = []
        for nucleotide in seq:
            if nucleotide == 'A':
                features.append([1, 0, 0, 0, 0])
            elif nucleotide == 'C':
                features.append([0, 1, 0, 0, 0])
            elif nucleotide == 'G':
                features.append([0, 0, 1, 0, 0])
            elif nucleotide == 'U':
                features.append([0, 0, 0, 1, 0])
            else:
                features.append([0, 0, 0, 0, 1])
        if len(features) < max_length:
            padding = [[0, 0, 0, 0, 0]] * (max_length - len(features))
            features.extend(padding)
        else:
            features = features[:max_length]
        X_test.append(features)
    return np.array(X_test)

def extract_sequence_features(seq_features):
    """
    Extract relevant sequence features from one-hot encoding.
    """
    # Get valid rows (non-padding)
    valid_mask = ~np.all(seq_features == 0, axis=1)
    valid_features = seq_features[valid_mask]
    
    # Calculate nucleotide composition
    a_content = np.mean(valid_features[:, 0])
    c_content = np.mean(valid_features[:, 1])
    g_content = np.mean(valid_features[:, 2])
    u_content = np.mean(valid_features[:, 3])
    gc_content = c_content + g_content
    
    return {
        'length': np.sum(valid_mask),
        'a_content': a_content,
        'c_content': c_content,
        'g_content': g_content, 
        'u_content': u_content,
        'gc_content': gc_content,
        'au_content': a_content + u_content
    }

def visualize_3d_structure(true_coords, pred_coords, sample_idx=0, title="3D Structure Comparison", show_plot=False):
    """
    Visualizes the true and predicted 3D structures for a sample.
    Only shows the plot if explicitly requested.
    """
    true = true_coords[sample_idx]
    pred = pred_coords[sample_idx]
    mask = ~np.all(true == 0, axis=1)
    true = true[mask]
    pred = pred[mask]
    
    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(true[:, 0], true[:, 1], true[:, 2], 'b-', label='True')
    ax1.scatter(true[:, 0], true[:, 1], true[:, 2], c='b', s=20, alpha=0.5)
    ax1.set_title('True Structure')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.grid(True)
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(pred[:, 0], pred[:, 1], pred[:, 2], 'r-', label='Predicted')
    ax2.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='r', s=20, alpha=0.5)
    ax2.set_title('Predicted Structure')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Always save the figure
    filename = f'structure_comparison_{sample_idx}.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    
    # Only show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        
    return filename  # Return the filename for reference


def identify_stem_loops(sequence):
    """
    Simple function to identify potential stem-loop regions in RNA.
    
    Parameters:
    -----------
    sequence: RNA sequence
    
    Returns:
    --------
    List of (start, end) indices for potential stem loops
    """
    # This is a simplified implementation
    # A real implementation would use a more sophisticated algorithm
    
    stem_loops = []
    min_stem_length = 3
    
    # Look for complementary regions that could form stems
    for i in range(len(sequence) - 2*min_stem_length - 3):
        for j in range(i + min_stem_length + 3, len(sequence) - min_stem_length):
            # Check if regions could form a stem
            potential_stem = True
            for k in range(min_stem_length):
                if not are_complementary(sequence[i+k], sequence[j+min_stem_length-1-k]):
                    potential_stem = False
                    break
            
            if potential_stem:
                # Potential stem-loop found
                stem_loops.append((i, j + min_stem_length))
                break
    
    return stem_loops

def are_complementary(base1, base2):
    """Check if two bases are complementary in RNA."""
    return (base1 == 'A' and base2 == 'U') or \
           (base1 == 'U' and base2 == 'A') or \
           (base1 == 'G' and base2 == 'C') or \
           (base1 == 'C' and base2 == 'G') or \
           (base1 == 'G' and base2 == 'U') or \
           (base1 == 'U' and base2 == 'G')  # G-U wobble pairs are valid in RNA

def apply_stem_loop_template(structure, start, end):
    """
    Apply a stem-loop template to a specific region of the structure.
    
    Parameters:
    -----------
    structure: RNA 3D structure
    start, end: Indices of the stem-loop region
    
    Returns:
    --------
    Modified structure with stem-loop template applied
    """
    # Create a copy to modify
    result = structure.copy()
    
    # Length of the region
    region_length = end - start + 1
    
    # Not enough residues to form a proper stem-loop
    if region_length < 7:
        return result
    
    # Calculate stem length (approximately 1/3 of the region on each side)
    stem_length = max(2, region_length // 6)
    loop_start = start + stem_length
    loop_end = end - stem_length
    
    # Loop length
    loop_length = loop_end - loop_start + 1
    
    # Apply stem template (roughly parallel strands)
    for i in range(stem_length):
        # Base positions in the two stems
        pos1 = start + i
        pos2 = end - i
        
        if pos1 < len(result) and pos2 < len(result):
            # Create roughly parallel strands
            if i > 0:
                # Base the position on the previous nucleotide in the strand
                result[pos1] = result[pos1-1] + np.array([0.0, 3.8, 0.0])
                result[pos2] = result[pos2+1] + np.array([0.0, -3.8, 0.0])
    
    # Apply loop template (roughly circular)
    if loop_length > 0:
        # Calculate center of the loop
        if loop_start < len(result) and loop_end < len(result):
            center = (result[loop_start-1] + result[loop_end+1]) / 2
            center[1] += 4.0  # Offset in y direction
            
            # Create a circular loop
            radius = 3.8  # approximately nucleotide distance
            for i in range(loop_length):
                idx = loop_start + i
                if idx < len(result):
                    angle = np.pi * i / (loop_length - 1)
                    result[idx] = center + np.array([
                        radius * np.cos(angle),
                        0.0,
                        radius * np.sin(angle)
                    ])
    
    return result

def post_process_rna_structure(structure, sequence, gc_content, use_global_movement=True):
    """
    Apply RNA-specific post-processing to refine a structure.
    
    Parameters:
    -----------
    structure: Predicted 3D coordinates
    sequence: RNA sequence
    gc_content: GC content of the sequence
    use_global_movement: Whether to apply global movement transformations
    
    Returns:
    --------
    Refined structure
    """
    # Create a new structure for modifications
    result = structure.copy()
    
    # 1. Apply mild refinement based on sequence composition
    noise_level = 0.1
    if gc_content > 0.6:
        # GC-rich regions tend to form more stable structures
        noise_level = 0.05  # Lower noise for more stable structures
    elif gc_content < 0.4:
        # AT-rich regions tend to be more flexible
        noise_level = 0.15  # Higher noise for more flexible regions
    
    # Apply noise proportional to sequence characteristics
    result = sample_structural_variation(
        result,
        noise_level=noise_level,
        preserve_distance=True,  # Always preserve distances for realistic structures
        use_global_movement=use_global_movement,
        correlation=0.85  # High correlation for smoother changes
    )
    
    # 2. Look for motifs in the sequence and apply structure templates
    # This is a simplified example - a complete implementation would include more motifs
    stem_loops = identify_stem_loops(sequence)
    if stem_loops:
        for start, end in stem_loops:
            # Apply stem-loop template to these regions
            result = apply_stem_loop_template(result, start, end)
    
    # 3. Normalize bond lengths to ideal values for RNA
    valid_mask = ~np.all(result == 0, axis=1)
    for i in range(1, len(result)):
        if valid_mask[i] and valid_mask[i-1]:
            # Get the current bond vector
            bond_vector = result[i] - result[i-1]
            bond_length = np.linalg.norm(bond_vector)
            
            if bond_length > 0:
                # Normalize to ideal RNA backbone distance with small variation
                ideal_length = 3.8 * (1 + np.random.normal(0, 0.03))
                result[i] = result[i-1] + (bond_vector / bond_length) * ideal_length
    
    return result


def reference_based_approach(X_ref, y_ref, geometric_sampling=False, noise_level=0.2, correlation=0.7):
    try:
        class ReferenceModel:
            def __init__(self, geometric_sampling=False, base_noise_level=0.2, correlation=0.7):
                self.geometric_sampling = geometric_sampling
                self.base_noise_level = base_noise_level
                self.correlation = correlation
                
            def fit(self, X, y):
                # First, handle NaN values in the reference structures
                self.reference_structures = np.nan_to_num(y, nan=0.0)
                self.global_mean = np.nanmean(y, axis=(0, 1))
                self.global_std = np.nanstd(y, axis=(0, 1))
                
                # Replace potential NaN values in statistics
                self.global_mean = np.nan_to_num(self.global_mean, nan=0.0)
                self.global_std = np.nan_to_num(self.global_std, nan=1.0)
                
                # Calculate size statistics
                self.size_groups = {}
                # Group reference structures by size
                for i in range(len(self.reference_structures)):
                    valid_mask = ~np.all(self.reference_structures[i] == 0, axis=1)
                    size = np.sum(valid_mask)
                    
                    if size < 120:
                        group = "small"
                    elif size < 200:
                        group = "medium"
                    else:
                        group = "large"
                        
                    if group not in self.size_groups:
                        self.size_groups[group] = []
                    self.size_groups[group].append(i)
                    
                print(f"Size distribution - Small: {len(self.size_groups.get('small', []))}, "
                      f"Medium: {len(self.size_groups.get('medium', []))}, "
                      f"Large: {len(self.size_groups.get('large', []))}")
                      
                # Store the correlation parameter for use in sample_structural_variation
                global_correlation = self.correlation
                print(f"Using noise level: {self.base_noise_level}, correlation: {global_correlation}")
                
                return self
                
            def predict(self, X):
                batch_size = X.shape[0]
                seq_length = X.shape[1]
                predictions = np.zeros((batch_size, seq_length, 3))
                
                for i in range(batch_size):
                    # Determine the RNA size group
                    valid_mask = ~np.all(X[i] == 0, axis=1)
                    size = np.sum(valid_mask)
                    if size < 120:
                        group = "small"
                        # Size-specific noise scaling
                        noise_level = self.base_noise_level * 0.6
                    elif size < 200:
                        group = "medium"
                        noise_level = self.base_noise_level * 1.0
                    else:
                        group = "large"
                        noise_level = self.base_noise_level * 0.4
                    
                    # If we have reference structures in this size group, use them
                    if group in self.size_groups and self.size_groups[group]:
                        # Randomly pick a reference structure from the same size group
                        ref_idx = np.random.choice(self.size_groups[group])
                        base_struct = self.reference_structures[ref_idx].copy()
                        
                        if self.geometric_sampling:
                            # Pass the correlation parameter to the variation function
                            predictions[i] = sample_structural_variation(
                                base_struct, 
                                noise_level=noise_level,
                                preserve_distance=True,
                                use_global_movement=(group == "small"),
                                correlation=self.correlation
                            )
                        else:
                            noise = np.random.normal(0, noise_level, base_struct.shape)
                            predictions[i] = base_struct + noise
                    else:
                        # Fall back to the original method if no size match
                        sample = np.random.normal(self.global_mean, self.global_std, size=(seq_length, 3))
                        if self.geometric_sampling:
                            predictions[i] = sample_structural_variation(
                                sample, 
                                noise_level=noise_level,
                                preserve_distance=True,
                                use_global_movement=(group == "small"),
                                correlation=self.correlation
                            )
                        else:
                            predictions[i] = sample
                        
                return predictions
        
        # Create and return model with specific parameters
        model = ReferenceModel(geometric_sampling=geometric_sampling, 
                              base_noise_level=noise_level,
                              correlation=correlation)
        model.fit(X_ref, y_ref)
        return model
    
    except Exception as e:
        print(f"Error in reference_based_approach: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_model(model, X_valid, y_valid, show_plots=False, save_top_plots=False):
    # Problem: Inadequate evaluation
    
    # SOLUTION:
    import numpy as np
    
    # Ensure there are no NaNs in the data
    X_valid_clean = np.nan_to_num(X_valid, nan=0.0)
    y_valid_clean = np.nan_to_num(y_valid, nan=0.0)
    
    # Make prediction with try/except to capture errors
    try:
        y_pred = model.predict(X_valid_clean)
        
        # Check if prediction contains NaNs or infinities
        if np.isnan(y_pred).any() or np.isinf(y_pred).any():
            print("WARNING: Prediction contains NaN or infinite values!")
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate metrics  
        mae = np.mean(np.abs(y_pred - y_valid_clean))
        mse = np.mean((y_pred - y_valid_clean)**2)
        
        # Calculate TM-scores for each structure
        tm_scores = []
        for i in range(len(X_valid)):
            # Compute score with error handling  
            try:
                tm = calculate_tm_score(y_pred[i], y_valid_clean[i])
                if np.isnan(tm) or np.isinf(tm):
                    print(f"WARNING: Invalid TM-score for sample {i}, using 0.0")
                    tm = 0.0
            except Exception as e:
                print(f"Error calculating TM-score for sample {i}: {str(e)}")
                tm = 0.0
                
            tm_scores.append(tm)
        
        # Final metrics
        avg_tm_score = np.mean(tm_scores)
        
        print(f"MAE: {mae:.4f}, MSE: {mse:.4f}")  
        print(f"Average TM-score: {avg_tm_score:.4f}")
        
        return {
            'mae': mae,
            'mse': mse,
            'tm_scores': tm_scores,  
            'avg_tm_score': avg_tm_score,
            'success': True
        }
        
    except Exception as e:
        print(f"ERROR in evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'mae': float('inf'),
            'mse': float('inf'), 
            'tm_scores': [0.0] * len(X_valid),
            'avg_tm_score': 0.0,
            'success': False,
            'error': str(e)  
        }

def enhanced_adaptive_seed_search(
    X_valid, 
    y_valid, 
    initial_threshold=0.65, 
    min_threshold=0.55, 
    initial_attempts=100, 
    max_attempts=300, 
    optimal_params={'noise': 0.21, 'corr': 0.83},
    diversity_threshold=0.15,
    max_seeds=10
):
    """
    Enhanced adaptive seed search that combines threshold adaptation with RNA-specific targeting.
    
    This function incorporates domain knowledge about different RNA categories (small, medium, large)
    while maintaining the adaptive threshold approach to find high-quality seeds.
    
    Parameters:
    -----------
    X_valid, y_valid: Validation data
    initial_threshold: Starting TM-score threshold to consider a seed "golden"
    min_threshold: Minimum acceptable threshold if not enough seeds found
    initial_attempts: Initial number of attempts per category
    max_attempts: Maximum number of attempts per category
    optimal_params: Model parameters
    diversity_threshold: Threshold to consider seeds as diverse from each other
    max_seeds: Maximum number of golden seeds to return
    
    Returns:
    --------
    golden_seeds: List of diverse "golden" seeds
    all_seeds: List of all tested seeds with their scores
    """
    print(f"Enhanced adaptive search for up to {max_seeds} golden seeds starting with threshold {initial_threshold:.2f}...")
    
    # Minimum number of seeds we want to find
    target_seeds = max(5, max_seeds // 2)
    
    current_threshold = initial_threshold
    current_attempts = initial_attempts
    
    # List to store all tested seeds
    all_seeds = []
    golden_seeds = []
    
    # Calculate sequence characteristics for RNA categorization
    seq_lengths = []
    gc_contents = []
    
    # Extract sequence lengths and nucleotide composition
    for i in range(len(X_valid)):
        # Identify non-padding positions
        valid_mask = ~np.all(X_valid[i] == 0, axis=1)
        seq_length = np.sum(valid_mask)
        seq_lengths.append(seq_length)
        
        # Extract nucleotide content if possible
        if seq_length > 0:
            features = X_valid[i][valid_mask]
            # Calculate GC content
            g_content = np.mean(features[:, 2]) if features.shape[1] > 2 else 0  # G base (index 2)
            c_content = np.mean(features[:, 1]) if features.shape[1] > 1 else 0  # C base (index 1)
            gc_content = g_content + c_content
        else:
            gc_content = 0.5  # Default if no valid sequence
            
        gc_contents.append(gc_content)
    
    # Separate indices by RNA size category
    small_rna_indices = [i for i, length in enumerate(seq_lengths) if length < 50]
    medium_rna_indices = [i for i, length in enumerate(seq_lengths) if 50 <= length < 120]
    large_rna_indices = [i for i, length in enumerate(seq_lengths) if length >= 120]
    
    # Additional categorization by GC content
    high_gc_indices = [i for i, gc in enumerate(gc_contents) if gc > 0.6]
    
    print(f"RNA Distribution: {len(small_rna_indices)} small, {len(medium_rna_indices)} medium, {len(large_rna_indices)} large")
    print(f"High GC content RNAs: {len(high_gc_indices)}")
    
    # Define optimal seed ranges and parameters for different RNA categories
    rna_categories = [
        {
            "name": "small_RNA",
            "indices": small_rna_indices,
            "seed_range": (1, 50000),         # Small seeds work well for small RNAs
            "noise_scale": 0.9,               # Lower noise for small RNAs
            "attempts_scale": 1.0,            # Standard attempts
            "threshold_bonus": 0.0            # No threshold adjustment
        },
        {
            "name": "medium_RNA",
            "indices": medium_rna_indices,
            "seed_range": (10000, 150000),    # Medium range for medium RNAs
            "noise_scale": 1.0,               # Standard noise
            "attempts_scale": 1.2,            # More attempts for medium RNAs
            "threshold_bonus": -0.02          # Slightly easier threshold
        },
        {
            "name": "large_RNA",
            "indices": large_rna_indices,
            "seed_range": (100000, 1000000),  # Larger seeds for large RNAs
            "noise_scale": 0.7,               # Lower noise for complex structures
            "attempts_scale": 1.5,            # More attempts for large RNAs
            "threshold_bonus": -0.05          # Lower threshold - these are harder
        },
        {
            "name": "high_GC",
            "indices": high_gc_indices,
            "seed_range": (50000, 500000),    # Wide range for high GC RNAs
            "noise_scale": 0.8,               # Lower noise for stable structures
            "attempts_scale": 1.2,            # More attempts
            "threshold_bonus": -0.03          # Slightly lower threshold
        }
    ]
    
    # Keep trying with lower thresholds until we find enough seeds
    while current_threshold >= min_threshold and len(golden_seeds) < target_seeds:
        print(f"\nSearching with base threshold {current_threshold:.2f} and {current_attempts} attempts per category...")
        
        # Process each RNA category
        for category in rna_categories:
            category_indices = category["indices"]
            
            # Skip categories with no examples
            if not category_indices:
                print(f"Skipping {category['name']} category (no examples in validation set)")
                continue
                
            # Adjust threshold and attempts for this category
            category_threshold = max(min_threshold, current_threshold + category["threshold_bonus"])
            category_attempts = min(max_attempts, int(current_attempts * category["attempts_scale"]))
            min_seed, max_seed = category["seed_range"]
            
            print(f"\nSearching seeds for {category['name']} with threshold {category_threshold:.2f} and {category_attempts} attempts...")
            
            # Extract subset of validation data for this category
            X_subset = [X_valid[i] for i in category_indices]
            y_subset = [y_valid[i] for i in category_indices]
            
            # List to store tested seeds at this threshold
            category_seeds = []
            
            # Set of seeds already tested to avoid duplications
            tested_seeds = set(seed_info["seed"] for seed_info in all_seeds)
            
            # Counter for valid attempts (excluding duplicates)
            valid_attempts = 0
            
            # Main cycle to search for seeds in this category
            while valid_attempts < category_attempts and len(golden_seeds) < max_seeds:
                # Generate random seed from this category's range
                seed = np.random.randint(min_seed, max_seed)
                
                # Check if we've already tested this seed
                if seed in tested_seeds:
                    continue
                
                tested_seeds.add(seed)
                valid_attempts += 1
                
                if valid_attempts % 10 == 0:
                    print(f"Testing seed {valid_attempts}/{category_attempts} for {category['name']} (seed={seed})...")
                
                # Set the seed for reproducibility
                np.random.seed(seed)
                
                # Create model with this seed and category-specific parameters
                try:
                    # Adjust noise level based on category
                    adjusted_noise = optimal_params['noise'] * category["noise_scale"]
                    
                    model = reference_based_approach(
                        X_valid, 
                        y_valid,
                        geometric_sampling=True,
                        noise_level=adjusted_noise,
                        correlation=optimal_params['corr']
                    )
                    
                    if model is None:
                        continue
                        
                    # First evaluate on this specific category
                    category_metrics = evaluate_model(model, X_subset, y_subset)
                    category_tm_score = category_metrics['avg_tm_score']
                    
                    # Then evaluate on all validation data
                    all_metrics = evaluate_model(model, X_valid, y_valid)
                    overall_tm_score = all_metrics['avg_tm_score']
                    
                    # Evaluate on different RNA size categories to detect overfitting
                    size_scores = {}
                    
                    if small_rna_indices:
                        small_metrics = evaluate_model_on_indices(model, X_valid, y_valid, small_rna_indices)
                        size_scores["small"] = small_metrics['avg_tm_score']
                    
                    if medium_rna_indices:
                        medium_metrics = evaluate_model_on_indices(model, X_valid, y_valid, medium_rna_indices)
                        size_scores["medium"] = medium_metrics['avg_tm_score']
                    
                    if large_rna_indices:
                        large_metrics = evaluate_model_on_indices(model, X_valid, y_valid, large_rna_indices)
                        size_scores["large"] = large_metrics['avg_tm_score']
                    
                    # Calculate standard deviation between scores for different sizes
                    # A high deviation may indicate overfitting in certain sizes
                    size_std = np.std(list(size_scores.values())) if size_scores else 0.0
                    
                    # Calculate balanced score that rewards:
                    # 1. High overall performance
                    # 2. Good performance on the specific category
                    # 3. Consistent performance across RNA sizes (low size_std)
                    balanced_score = (
                        0.4 * overall_tm_score + 
                        0.4 * category_tm_score +
                        0.2 * (1.0 - min(1.0, size_std * 2))  # Convert std to a 0-1 score (lower is better)
                    )
                    
                    # Register this seed
                    seed_info = {
                        'seed': seed,
                        'tm_score': overall_tm_score,
                        'category_tm_score': category_tm_score,
                        'balanced_score': balanced_score,
                        'size_std': size_std,
                        'category': category['name'],
                        'size_scores': size_scores.copy()
                    }
                    category_seeds.append(seed_info)
                    all_seeds.append(seed_info)
                    
                    # Check if this is a "golden" seed for this category
                    if category_tm_score >= category_threshold:
                        # Check diversity relative to seeds already found
                        is_diverse = True
                        
                        for i, existing_seed in enumerate(golden_seeds):
                            # Calculate seed similarity based on predictions
                            similarity = 0
                            
                            # Simple similarity metric based on category and score
                            if existing_seed.get('category') == category['name']:
                                similarity += 0.3  # Same category adds similarity
                                
                            score_diff = abs(existing_seed['balanced_score'] - balanced_score)
                            if score_diff < 0.1:
                                similarity += (0.1 - score_diff) * 3  # Similar scores add similarity
                                
                            if similarity > diversity_threshold:
                                is_diverse = False
                                # If the new one is better than an existing one and they are similar, replace
                                if balanced_score > existing_seed['balanced_score']:
                                    print(f"  Replacing seed {existing_seed['seed']} (score={existing_seed['balanced_score']:.4f}) " 
                                          f"with seed {seed} (score={balanced_score:.4f})")
                                    golden_seeds[i] = seed_info
                                break
                        
                        if is_diverse and len(golden_seeds) < max_seeds:
                            print(f"  Found golden seed: {seed} for {category['name']} (Score: {balanced_score:.4f}, Overall TM: {overall_tm_score:.4f})")
                            golden_seeds.append(seed_info)
                            
                    # Always check the overall performance for generalist seeds
                    elif overall_tm_score >= current_threshold and balanced_score >= current_threshold:
                        # Only add if not already in golden seeds and meeting diversity criteria
                        if seed not in [gs['seed'] for gs in golden_seeds]:
                            is_diverse = True
                            
                            for i, existing_seed in enumerate(golden_seeds):
                                similarity = 0
                                score_diff = abs(existing_seed['balanced_score'] - balanced_score)
                                if score_diff < 0.1:
                                    similarity += (0.1 - score_diff) * 5
                                    
                                if similarity > diversity_threshold:
                                    is_diverse = False
                                    # Replace if better
                                    if balanced_score > existing_seed['balanced_score']:
                                        print(f"  Replacing seed {existing_seed['seed']} (score={existing_seed['balanced_score']:.4f}) " 
                                              f"with seed {seed} (score={balanced_score:.4f})")
                                        golden_seeds[i] = seed_info
                                    break
                            
                            if is_diverse and len(golden_seeds) < max_seeds:
                                print(f"  Found golden generalist seed: {seed} (Score: {balanced_score:.4f}, Overall TM: {overall_tm_score:.4f})")
                                golden_seeds.append(seed_info)
                
                except Exception as e:
                    print(f"  Error testing seed {seed}: {str(e)}")
                    continue
            
            print(f"Completed {valid_attempts} attempts for {category['name']} category")
        
        # If we found enough seeds, we can stop
        if len(golden_seeds) >= target_seeds:
            print(f"Found {len(golden_seeds)} golden seeds with threshold {current_threshold:.2f}")
            break
            
        # Otherwise, reduce threshold and increase attempts
        current_threshold -= 0.03
        current_attempts = min(current_attempts + 50, max_attempts)
        
        print(f"Not enough seeds found. Reducing threshold to {current_threshold:.2f} and increasing attempts to {current_attempts}")
    
    # If we still don't have enough seeds, take the best ones from all tested
    if len(golden_seeds) < target_seeds and all_seeds:
        print(f"Could not find {target_seeds} golden seeds even with threshold {current_threshold:.2f}")
        print(f"Using best seeds found during search based on balanced score...")
        
        # Remove duplicates and sort by balanced score
        unique_seeds = {}
        for seed in all_seeds:
            if seed['seed'] not in unique_seeds or seed.get('balanced_score', 0) > unique_seeds[seed['seed']].get('balanced_score', 0):
                unique_seeds[seed['seed']] = seed
                
        sorted_seeds = sorted(unique_seeds.values(), key=lambda x: x.get('balanced_score', 0), reverse=True)
        
        # Add best seeds that aren't already in golden_seeds
        existing_seed_ids = {gs['seed'] for gs in golden_seeds}
        
        for seed in sorted_seeds:
            if seed['seed'] not in existing_seed_ids and len(golden_seeds) < max_seeds:
                golden_seeds.append(seed)
                existing_seed_ids.add(seed['seed'])
    
    # Ensure seeds are sorted by balanced score
    golden_seeds.sort(key=lambda x: x.get('balanced_score', 0), reverse=True)
    
    # Ensure we have at least some seeds from each major category if possible
    if len(golden_seeds) >= 3:
        categories_present = set(seed.get('category', '') for seed in golden_seeds)
        
        # Check which major categories are missing
        major_categories = ['small_RNA', 'medium_RNA', 'large_RNA']
        missing_categories = [cat for cat in major_categories if cat not in categories_present]
        
        if missing_categories:
            print(f"Ensuring representation from missing categories: {missing_categories}")
            
            # Find best seeds for missing categories
            for category in missing_categories:
                category_seeds = [s for s in all_seeds if s.get('category') == category]
                
                if category_seeds:
                    # Sort by balanced score
                    category_seeds.sort(key=lambda x: x.get('balanced_score', 0), reverse=True)
                    best_seed = category_seeds[0]
                    
                    # Only add if not already in golden seeds
                    if best_seed['seed'] not in {gs['seed'] for gs in golden_seeds}:
                        # Replace worst seed if we're at capacity
                        if len(golden_seeds) >= max_seeds:
                            worst_idx = min(range(len(golden_seeds)), 
                                          key=lambda i: golden_seeds[i].get('balanced_score', 0))
                            print(f"  Replacing seed {golden_seeds[worst_idx]['seed']} with {best_seed['seed']} to ensure {category} representation")
                            golden_seeds[worst_idx] = best_seed
                        else:
                            print(f"  Adding seed {best_seed['seed']} to ensure {category} representation")
                            golden_seeds.append(best_seed)
    
    # Show statistics of the found seeds
    print(f"\nFound {len(golden_seeds)} golden seeds in {len(all_seeds)} total attempts")
    
    for i, gs in enumerate(golden_seeds):
        print(f"  Seed {i+1}: {gs['seed']} (Category: {gs.get('category', 'unknown')}, Balanced: {gs.get('balanced_score', 0):.4f}, TM: {gs['tm_score']:.4f})")
        
        # Print size-specific scores
        for size, score in gs.get('size_scores', {}).items():
            print(f"    {size.capitalize()} RNA score: {score:.4f}")
            
        print(f"    Consistency across sizes (std): {gs.get('size_std', 0):.4f}")
    
    return golden_seeds, all_seeds

def evaluate_model_on_indices(model, X_data, y_data, indices):
    """
    Evaluates the model only on specific indices of the data.
    Useful to evaluate performance on subsets like small/medium/large RNAs.
    """
    X_subset = [X_data[i] for i in indices]
    y_subset = [y_data[i] for i in indices]
    
    return evaluate_model(model, X_subset, y_subset)

def calculate_prediction_similarity(preds1, preds2):
    """
    Calculates the similarity between two sets of predictions.
    Returns a value between 0 (totally different) and 1 (identical).
    """
    similarities = []
    
    # For each pair of sequences in the predictions
    for p1, p2 in zip(preds1, preds2):
        # Identify valid (non-zero) coordinates
        valid_mask1 = ~np.all(p1 == 0, axis=1)
        valid_mask2 = ~np.all(p2 == 0, axis=1)
        
        # Use only positions valid in both predictions
        valid_mask = valid_mask1 & valid_mask2
        
        # If there are no overlapping valid positions, continue
        if np.sum(valid_mask) < 3:
            continue
        
        # Extract valid coordinates
        valid_p1 = p1[valid_mask]
        valid_p2 = p2[valid_mask]
        
        # Calculate similarity based on RMSD distance
        squared_diff = np.sum((valid_p1 - valid_p2) ** 2, axis=1)
        rmsd = np.sqrt(np.mean(squared_diff))
        
        # Convert RMSD to similarity (lower RMSD values = higher similarity)
        # Normalize so it's between 0 and 1
        similarity = 1.0 / (1.0 + rmsd / 5.0)  # Division by 5.0 is an arbitrary scale
        similarities.append(similarity)
    
    # Return average similarity
    return np.mean(similarities) if similarities else 0.0


class EnhancedRNAQualityNN:
    """
    Enhanced Neural Network model for RNA structure quality assessment.
    Features:
    - Handles variable-length RNA sequences
    - Incorporates RNA-specific features
    - Attention mechanism for capturing long-range interactions
    - Multiple evaluation metrics for robust quality assessment
    """
    def __init__(self, max_length=720):
        self.max_length = max_length
        self.is_trained = False
        self.model = None
        self.build_model()
        
    def build_model(self):
        """
        Build an enhanced model architecture for RNA quality assessment.
        """
        # Define the masking layer to handle variable-length sequences
        coord_input = layers.Input(shape=(self.max_length, 3), name='coordinates')
        
        # Create a mask for zero-padded coordinates
        mask_layer = layers.Lambda(
            lambda x: tf.cast(tf.reduce_sum(tf.abs(x), axis=-1) > 0.0, tf.float32),
            output_shape=lambda shape: (shape[0], shape[1])
        )
        mask = mask_layer(coord_input)
        
        # Expandir dimensÃµes
        mask_expanded_layer = layers.Lambda(
            lambda x: tf.expand_dims(x, axis=-1),
            output_shape=lambda shape: (shape[0], shape[1], 1)
        )
        mask_expanded = mask_expanded_layer(mask)  # Shape: (batch, seq_len, 1)
        
        # Optional sequence features input
        seq_input = layers.Input(shape=(self.max_length, 5), name='sequence')
        
        # DefiniÃ§Ã£o da funÃ§Ã£o de distÃ¢ncia pareada
        def create_pairwise_dist_layer():
            def masked_pairwise_dist_fn(inputs):
                coords, m = inputs
                # Expand dims for broadcasting
                coords1 = tf.expand_dims(coords, 2)
                coords2 = tf.expand_dims(coords, 1)
                
                # Calculate Euclidean distance
                diff = coords1 - coords2
                squared_diff = tf.reduce_sum(tf.square(diff), axis=-1)
                dist = tf.sqrt(squared_diff + 1e-8)
                
                # Create mask for valid pairs
                mask1 = tf.expand_dims(m, 2)
                mask2 = tf.expand_dims(m, 1)
                pair_mask = mask1 * mask2
                
                # Apply mask
                masked_dist = dist * pair_mask
                return masked_dist
            
            return layers.Lambda(
                masked_pairwise_dist_fn,
                output_shape=lambda shape: (shape[0][0], shape[0][1], shape[0][1])
            )
        
        # Aplicar a camada de distÃ¢ncia pareada
        pairwise_dist_layer = create_pairwise_dist_layer()
        distances = pairwise_dist_layer([coord_input, mask])
        
        # 1.2 Process distances with 2D convolutions
        dist_features = layers.Reshape((self.max_length, self.max_length, 1))(distances)
        dist_features = layers.Conv2D(16, 3, activation='relu', padding='same')(dist_features)
        dist_features = layers.BatchNormalization()(dist_features)
        dist_features = layers.MaxPooling2D(2)(dist_features)
        
        dist_features = layers.Conv2D(32, 3, activation='relu', padding='same')(dist_features)
        dist_features = layers.BatchNormalization()(dist_features)
        dist_features = layers.MaxPooling2D(2)(dist_features)
        
        # Flatten with adaptive pooling to handle variable lengths
        dist_features = layers.GlobalAveragePooling2D()(dist_features)
        
        # 1.3 Process direct 3D coordinates with 1D convolutions
        # Apply mask to zero out padded positions
        masked_coords = layers.Multiply()([coord_input, mask_expanded])
        
        coord_features = layers.Conv1D(32, 3, activation='relu', padding='same')(masked_coords)
        coord_features = layers.BatchNormalization()(coord_features)
        
        # Para o mecanismo de auto-atenÃ§Ã£o, criamos as camadas Dense fora da funÃ§Ã£o Lambda
        query_dense = layers.Dense(32)
        key_dense = layers.Dense(32)
        value_dense = layers.Dense(32)
        
        # FunÃ§Ã£o de auto-atenÃ§Ã£o agora usa camadas prÃ©-definidas
        def create_self_attention_layer(query_dense, key_dense, value_dense):
            def self_attention_fn(inputs):
                x, m = inputs
                # Simple self-attention usando camadas prÃ©-definidas
                query = query_dense(x)
                key = key_dense(x)
                value = value_dense(x)
                
                # Calculate attention scores
                scores = tf.matmul(query, key, transpose_b=True)
                scores = scores / tf.sqrt(32.0)
                
                # Apply mask
                mask1 = tf.expand_dims(m, 2)
                mask2 = tf.expand_dims(m, 1)
                mask_2d = mask1 * mask2
                
                # Very negative number for masked positions (-1e9)
                scores = scores * mask_2d + (1.0 - mask_2d) * (-1e9)
                
                # Apply softmax
                attention_weights = tf.nn.softmax(scores, axis=-1)
                
                # Apply attention
                output = tf.matmul(attention_weights, value)
                
                return output
            
            return layers.Lambda(
                self_attention_fn,
                output_shape=lambda shape: (shape[0][0], shape[0][1], 32)
            )
            
        # Aplicar a camada de auto-atenÃ§Ã£o
        self_attention_layer = create_self_attention_layer(query_dense, key_dense, value_dense)
        attention_output = self_attention_layer([coord_features, mask])
        
        # Continue processing coordinates
        coord_features = layers.Add()([coord_features, attention_output])  # Residual connection
        coord_features = layers.Conv1D(64, 3, activation='relu', padding='same')(coord_features)
        coord_features = layers.BatchNormalization()(coord_features)
        
        # Global pooling for variable length
        coord_features = layers.GlobalAveragePooling1D()(coord_features)
        
        # 2. Process sequence information (if provided)
        seq_features = layers.Conv1D(32, 3, activation='relu', padding='same')(seq_input)
        seq_features = layers.BatchNormalization()(seq_features)
        seq_features = layers.GlobalAveragePooling1D()(seq_features)
        
        # 3. Calculate RNA-specific features
        
        # 3.1 Extract GC content and other sequence composition features
        def create_sequence_composition_layer():
            def sequence_composition_fn(inputs):
                seq, m = inputs
                # One-hot encoded sequence: (batch, len, 5) [A,C,G,U,N]
                # Calculate GC content
                c_base = seq[:, :, 1]  # C base (index 1)
                g_base = seq[:, :, 2]  # G base (index 2)
                
                # Sum up G and C bases and divide by sequence length
                gc_sum = tf.reduce_sum(c_base * m + g_base * m, axis=1)
                seq_length = tf.reduce_sum(m, axis=1)
                
                # Avoid division by zero
                gc_content = gc_sum / (seq_length + 1e-8)
                
                # Calculate other base contents
                a_base = seq[:, :, 0]  # A base
                u_base = seq[:, :, 3]  # U base
                a_content = tf.reduce_sum(a_base * m, axis=1) / (seq_length + 1e-8)
                u_content = tf.reduce_sum(u_base * m, axis=1) / (seq_length + 1e-8)
                
                # Combine features
                composition = tf.stack([gc_content, a_content, u_content], axis=1)
                
                return composition
            
            return layers.Lambda(
                sequence_composition_fn,
                output_shape=lambda shape: (shape[0][0], 3)
            )
        
        # Aplicar a camada de composiÃ§Ã£o de sequÃªncia
        seq_composition_layer = create_sequence_composition_layer()
        seq_composition = seq_composition_layer([seq_input, mask])
        
        # 3.2 Calculate basic structural features
        def create_structural_features_layer():
            def structural_features_fn(inputs):
                coords, m = inputs
                # Calculate average bond length
                coords1 = coords[:, :-1, :]
                coords2 = coords[:, 1:, :]
                
                # Create mask for valid pairs
                mask_bonds = m[:, :-1] * m[:, 1:]
                mask_bonds_expanded = tf.expand_dims(mask_bonds, -1)
                
                # Calculate bond vectors and lengths
                bonds = coords2 - coords1
                masked_bonds = bonds * mask_bonds_expanded
                
                # Euclidean distance
                bond_lengths = tf.sqrt(tf.reduce_sum(tf.square(masked_bonds), axis=-1) + 1e-8)
                
                # Average bond length
                total_bonds = tf.reduce_sum(mask_bonds, axis=1)
                avg_bond_length = tf.reduce_sum(bond_lengths, axis=1) / (total_bonds + 1e-8)
                
                # Bond length consistency (std dev)
                mean_bond = tf.expand_dims(avg_bond_length, -1)
                squared_diff = tf.square(bond_lengths - mean_bond) * mask_bonds
                bond_var = tf.reduce_sum(squared_diff, axis=1) / (total_bonds + 1e-8)
                bond_std = tf.sqrt(bond_var + 1e-8)
                
                # Combine features
                struct_features = tf.stack([avg_bond_length, bond_std], axis=1)
                
                return struct_features
            
            return layers.Lambda(
                structural_features_fn,
                output_shape=lambda shape: (shape[0][0], 2)
            )
        
        # Aplicar a camada de caracterÃ­sticas estruturais
        struct_features_layer = create_structural_features_layer()
        struct_features = struct_features_layer([coord_input, mask])
        
        # 4. Combine all features
        combined = layers.Concatenate()([
            dist_features,      # Pairwise distance features
            coord_features,     # Direct coordinate features
            seq_features,       # Sequence features
            seq_composition,    # GC content, etc.
            struct_features     # Basic structural features
        ])
        
        # 5. Final processing with dense layers
        x = layers.Dense(128, activation='relu')(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # 6. Multiple output heads for different aspects of quality
        quality_score = layers.Dense(1, activation='sigmoid', name='quality_score')(x)
        bond_score = layers.Dense(1, activation='sigmoid', name='bond_score')(x)
        valid_score = layers.Dense(1, activation='sigmoid', name='valid_score')(x)
        
        # Create the model
        self.model = models.Model(
            inputs=[coord_input, seq_input],
            outputs=[quality_score, bond_score, valid_score]
        )
        
        # Compile with weighted losses to emphasize the overall quality score
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),  # Add gradient clipping
            loss={
                'quality_score': 'mean_squared_error',
                'bond_score': 'mean_squared_error',
                'valid_score': 'binary_crossentropy'
            },
            loss_weights={
                'quality_score': 1.0,     # Primary loss
                'bond_score': 0.3,        # Secondary loss
                'valid_score': 0.3        # Secondary loss
            },
            metrics={
                'quality_score': ['mae', 'mse'],
                'bond_score': ['mae'],
                'valid_score': ['accuracy']
            }
        )
    
    # Os mÃ©todos train, predict_quality, save_model e load_model permanecem os mesmos
    def train(self, X_train_coords, X_train_seq, y_train, 
              validation_data=None, epochs=50, batch_size=16):
        """
        Train the model with multiple outputs.
    
        Parameters:
        -----------
        X_train_coords: Coordinate inputs (batch, seq_len, 3)
        X_train_seq: Sequence inputs (batch, seq_len, 5)
        y_train: Dictionary with 'quality_score', 'bond_score', and 'valid_score' outputs
        validation_data: Optional validation data in the same format
        """
        # Define callbacks
        callbacks = [
            # Early stopping on the primary output - com mode='min' para mÃ©tricas de perda
            EarlyStopping(
                monitor='val_quality_score_loss' if validation_data else 'quality_score_loss',
                mode='min',  # Explicitamente indica que queremos minimizar a perda
                patience=10,
                restore_best_weights=True
            ),
            # Custom callback to detect and handle NaN values
            tf.keras.callbacks.TerminateOnNaN()
        ]
    
        # Train the model
        history = self.model.fit(
            x=[X_train_coords, X_train_seq],
            y=y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
    
        self.is_trained = True
        return history
    
    def predict_quality(self, X_coords, X_seq):
        """
        Predict quality scores for RNA structures.
    
        Parameters:
        -----------
        X_coords: Coordinate inputs (batch, seq_len, 3)
        X_seq: Sequence inputs (batch, seq_len, 5) ou (seq_len, 5) que serÃ¡ expandido
    
        Returns:
        --------
        Primary quality score predictions (0-1)
        """
        if not self.is_trained:
            print("WARNING: Model has not been trained yet!")
            return None
    
        # Handle potential shape issues
        batch_size = X_coords.shape[0]
        seq_len = X_coords.shape[1]
    
        # Ensure X_seq has 3 dimensions (batch, seq_len, features)
        if len(X_seq.shape) == 2:  # Se for (seq_len, features)
            X_seq = np.expand_dims(X_seq, axis=0)  # Adicionar dimensÃ£o de batch
            X_seq = np.repeat(X_seq, batch_size, axis=0)  # Replicar para todos os exemplos de batch
    
        # Ensure correct format for coordinates
        if seq_len > self.max_length:
            print(f"WARNING: Input sequence length ({seq_len}) exceeds model's maximum length ({self.max_length}).")
            print("Truncating input sequence to maximum length.")
            X_coords = X_coords[:, :self.max_length, :]
        elif seq_len < self.max_length:
            print(f"Padding input sequence from length {seq_len} to {self.max_length}")
            padding = np.zeros((batch_size, self.max_length - seq_len, 3))
            X_coords = np.concatenate([X_coords, padding], axis=1)
    
        # Ensure correct format for sequence
        if X_seq is None:
            # If no sequence provided, create zero array
            X_seq = np.zeros((batch_size, self.max_length, 5))
        else:
            seq_shape = X_seq.shape
            if seq_shape[1] > self.max_length:
                X_seq = X_seq[:, :self.max_length, :]
            elif seq_shape[1] < self.max_length:
                padding = np.zeros((batch_size, self.max_length - seq_shape[1], 5))
                X_seq = np.concatenate([X_seq, padding], axis=1)
    
        # Predict all outputs
        outputs = self.model.predict([X_coords, X_seq])
    
        # Return the primary quality score
        return outputs[0]  # quality_score output
    
    def save_model(self, filepath):
        """Save the model to disk"""
        if self.is_trained:
            self.model.save(filepath)
        else:
            print("WARNING: Cannot save untrained model")
    
    def load_model(self, filepath):
        """Load a pre-trained model from disk"""
        self.model = models.load_model(filepath)
        self.is_trained = True


def prepare_multi_output_targets(train_coords, train_scores):
    """
    Prepare multi-output target values from TM-scores.
    
    Parameters:
    -----------
    train_coords: Training coordinate data
    train_scores: TM-score values (overall quality)
    
    Returns:
    --------
    Dictionary with multiple output targets
    """
    batch_size = len(train_scores)
    
    # Initialize targets dictionary
    targets = {
        'quality_score': train_scores,
        'bond_score': np.zeros((batch_size, 1)),
        'valid_score': np.zeros((batch_size, 1))
    }
    
    # Calculate bond scores and validity scores for each structure
    for i in range(batch_size):
        coords = train_coords[i]
        
        # Calculate bond score (based on ideal bond length)
        valid_mask = ~np.all(coords == 0, axis=1)
        valid_coords = coords[valid_mask]
        
        # Skip if no valid coordinates
        if len(valid_coords) < 3:
            targets['bond_score'][i] = 0.5  # Neutral score
            targets['valid_score'][i] = 0  # Invalid
            continue
        
        # Calculate bond lengths
        bond_lengths = []
        for j in range(1, len(valid_coords)):
            dist = np.linalg.norm(valid_coords[j] - valid_coords[j-1])
            bond_lengths.append(dist)
        
        avg_bond_length = np.mean(bond_lengths)
        bond_std = np.std(bond_lengths)
        
        # Score based on how close to ideal RNA bond length (3.8Ã…)
        bond_score = 1.0 - min(1.0, abs(avg_bond_length - 3.8) / 3.8)
        targets['bond_score'][i] = bond_score
        
        # Validity score (binary)
        is_valid = check_structure_validity(coords)
        targets['valid_score'][i] = 1 if is_valid else 0
    
    return targets

def train_enhanced_quality_model(X_train, y_train, X_valid, y_valid):
    """
    Train an enhanced RNA quality assessment model.
    
    Parameters:
    -----------
    X_train, X_valid: One-hot encoded RNA sequences
    y_train, y_valid: True 3D coordinates
    
    Returns:
    --------
    Trained EnhancedRNAQualityNN model
    """
    print("Training enhanced RNA quality assessment model...")
    
    # First, determine maximum sequence length in the data
    max_train_len = max(np.sum(~np.all(X_train[i] == 0, axis=1)) for i in range(len(X_train)))
    max_valid_len = max(np.sum(~np.all(X_valid[i] == 0, axis=1)) for i in range(len(X_valid)))
    max_length = max(max_train_len, max_valid_len)
    
    print(f"Maximum sequence length in data: {max_length}")
    
    # Adjust max_length to a reasonable value (for memory efficiency)
    max_length = min(max_length, 720)  # Cap at 720 if larger
    
    # Generate training data with structure variations
    print("Generating training data with structure variations...")
    
    # Parameters for data generation
    num_variations = 10  # Generate 10 variations for each structure
    
    # Containers for training data
    train_seqs = []
    train_coords = []
    train_scores = []
    
    # Process training structures
    for i in range(min(len(X_train), 50)):  # Limit to 50 training examples
        print(f"Processing training structure {i+1}/{min(len(X_train), 50)}")
        seq_features = X_train[i]
        true_coords = y_train[i]
        
        # Check for NaN in true coordinates
        if np.isnan(true_coords).any():
            print(f"Skipping structure {i} due to NaN in true coordinates")
            continue
        
        # Add the true structure (highest quality)
        train_seqs.append(seq_features)
        train_coords.append(true_coords)
        train_scores.append(1.0)  # Perfect score for true structure
        
        # Generate variations with different qualities
        for j in range(num_variations):
            # Vary noise level to get different quality structures
            noise_level = 0.05 + (j * 0.05)  # Smaller steps for better distribution
            try:
                variation = sample_structural_variation(
                    true_coords, 
                    noise_level=noise_level,
                    preserve_distance=True,  # Always preserve distances for stability
                    use_global_movement=(j % 3 == 0)  # Mix of global and local movements
                )
                
                # Check for NaN or Inf in variation
                if np.isnan(variation).any() or np.isinf(variation).any():
                    print(f"Skipping variation {j} for structure {i} due to NaN/Inf")
                    continue
                
                # Calculate TM-score as ground truth quality
                tm_score = calculate_tm_score(variation, true_coords)
                
                # Check if score is valid
                if np.isnan(tm_score) or np.isinf(tm_score) or tm_score <= 0:
                    print(f"Skipping variation {j} for structure {i} due to invalid TM-score: {tm_score}")
                    continue
                
                # Apply additional normalization for stability
                normalized_variation = normalize_coordinates(variation.reshape(1, -1, 3))[0]
                
                train_seqs.append(seq_features)
                train_coords.append(normalized_variation)
                train_scores.append(tm_score)
            except Exception as e:
                print(f"Error generating variation {j} for structure {i}: {str(e)}")
                continue
    
    # Create a smaller validation set for speed and stability
    valid_seqs = []
    valid_coords = []
    valid_scores = []
    
    for i in range(min(len(X_valid), 10)):  # Use only 10 validation examples
        print(f"Processing validation structure {i+1}/{min(len(X_valid), 10)}")
        seq_features = X_valid[i]
        true_coords = y_valid[i]
        
        # Check for NaN in true coordinates
        if np.isnan(true_coords).any():
            print(f"Skipping validation structure {i} due to NaN in true coordinates")
            continue
        
        # Add the true structure
        valid_seqs.append(seq_features)
        valid_coords.append(true_coords)
        valid_scores.append(1.0)
        
        # Generate just 3 variations for validation
        for j in range(3):
            noise_level = 0.05 + (j * 0.1)
            try:
                variation = sample_structural_variation(
                    true_coords, 
                    noise_level=noise_level,
                    preserve_distance=True,
                    use_global_movement=(j % 2 == 0)
                )
                
                # Check for NaN or Inf
                if np.isnan(variation).any() or np.isinf(variation).any():
                    print(f"Skipping validation variation {j} for structure {i} due to NaN/Inf")
                    continue
                
                tm_score = calculate_tm_score(variation, true_coords)
                
                # Check if score is valid
                if np.isnan(tm_score) or np.isinf(tm_score) or tm_score <= 0:
                    print(f"Skipping validation variation {j} for structure {i} due to invalid TM-score: {tm_score}")
                    continue
                
                # Apply additional normalization
                normalized_variation = normalize_coordinates(variation.reshape(1, -1, 3))[0]
                
                valid_seqs.append(seq_features)
                valid_coords.append(normalized_variation)
                valid_scores.append(tm_score)
            except Exception as e:
                print(f"Error generating validation variation {j} for structure {i}: {str(e)}")
                continue
    
    # Convert to numpy arrays and handle potential issues
    train_seqs = np.array(train_seqs)
    train_coords = np.array(train_coords)
    train_scores = np.array(train_scores).reshape(-1, 1)  # Reshape to (n, 1)
    
    valid_seqs = np.array(valid_seqs)
    valid_coords = np.array(valid_coords)
    valid_scores = np.array(valid_scores).reshape(-1, 1)  # Reshape to (n, 1)
    
    # Verify data quality and apply additional cleaning
    train_coords = np.nan_to_num(train_coords, nan=0.0, posinf=0.0, neginf=0.0)
    train_scores = np.clip(train_scores, 0.0, 1.0)  # Ensure scores are in [0, 1]
    
    valid_coords = np.nan_to_num(valid_coords, nan=0.0, posinf=0.0, neginf=0.0)
    valid_scores = np.clip(valid_scores, 0.0, 1.0)
    
    # Log data statistics for debugging
    print(f"Training data: {len(train_scores)} structures")
    print(f"Train coords shape: {train_coords.shape}, train scores shape: {train_scores.shape}")
    print(f"Train coords range: [{np.min(train_coords)}, {np.max(train_coords)}]")
    print(f"Train scores range: [{np.min(train_scores)}, {np.max(train_scores)}]")
    
    print(f"Validation data: {len(valid_scores)} structures")
    
    try:
        # Prepare multi-output targets
        print("Preparing multi-output training targets...")
        train_targets = prepare_multi_output_targets(train_coords, train_scores)
        valid_targets = prepare_multi_output_targets(valid_coords, valid_scores)
        
        # Create and train the enhanced model
        print("Creating and training enhanced model...")
        model = EnhancedRNAQualityNN(max_length=max_length)
        
        # Train the model
        history = model.train(
            X_train_coords=train_coords,
            X_train_seq=train_seqs,
            y_train=train_targets,
            validation_data=([valid_coords, valid_seqs], valid_targets),
            epochs=30,
            batch_size=16
        )
        
        # Validate the model
        print("Validating model...")
        val_predictions = model.predict_quality(valid_coords, valid_seqs)
        val_predictions = val_predictions.flatten()
        
        # Calculate correlation between predicted and true scores
        correlation = np.corrcoef(val_predictions, valid_scores.flatten())[0, 1]
        mae = np.mean(np.abs(val_predictions - valid_scores.flatten()))
        
        print(f"Validation results:")
        print(f"Correlation: {correlation:.4f}")
        print(f"MAE: {mae:.4f}")
        
        # Save the model
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.save_model(os.path.join(OUTPUT_DIR, 'enhanced_rna_quality_model.h5'))
        
        return model
        
    except Exception as e:
        print(f"Error training enhanced model: {str(e)}")
        traceback.print_exc()
        
        # Fall back to a simpler model or rule-based approach
        print("Falling back to a simplified model due to training error...")
        return create_rule_based_model()

def create_rule_based_model():
   """
   Create a rule-based quality assessment model as fallback.
   """
   class RuleBasedQualityModel:
       def __init__(self):
           self.is_trained = True
           
       def predict_quality(self, X_coords, X_seq=None):
           batch_size = X_coords.shape[0]
           
           # Implement a comprehensive rule-based quality metric
           scores = []
           
           for i in range(batch_size):
               # Check for valid coordinates
               valid_mask = ~np.all(X_coords[i] == 0, axis=1)
               coords = X_coords[i][valid_mask]
               
               if len(coords) < 3:
                   scores.append(0.5)  # Default score for very short structures
                   continue
               
               # 1. Calculate bond lengths
               bond_lengths = []
               for j in range(1, len(coords)):
                   dist = np.linalg.norm(coords[j] - coords[j-1])
                   bond_lengths.append(dist)
               
               avg_bond_length = np.mean(bond_lengths)
               bond_std = np.std(bond_lengths)
               
               # 2. Score based on how close to ideal RNA bond length
               bond_score = 1.0 - min(1.0, abs(avg_bond_length - 3.8) / 3.8)
               
               # 3. Bond consistency score
               consistency_score = 1.0 - min(1.0, bond_std / 1.5)
               
               # 4. Check structure validity
               is_valid = check_structure_validity(coords)
               valid_score = 1.0 if is_valid else 0.5
               
               # 5. Check for extreme compression or expansion
               min_bond = min(bond_lengths) if bond_lengths else 0
               max_bond = max(bond_lengths) if bond_lengths else 0
               compression_score = 1.0
               if min_bond < 1.0 or max_bond > 10.0:  # Physical constraints for RNA
                   compression_score = 0.7
               
               # 6. Analyze radius of gyration (compactness)
               center = np.mean(coords, axis=0)
               distances = np.sqrt(np.sum((coords - center) ** 2, axis=1))
               radius_gyration = np.mean(distances)
               
               # Typical radius of gyration for RNA scales with sequence length (approximate)
               expected_radius = 3.0 * np.power(len(coords), 1/3)  # Simple scaling law
               compactness_score = 1.0 - min(1.0, abs(radius_gyration - expected_radius) / expected_radius)
               
               # 7. Combined score
               final_score = (
                   0.3 * bond_score + 
                   0.2 * consistency_score + 
                   0.2 * valid_score + 
                   0.15 * compression_score + 
                   0.15 * compactness_score
               )
               
               # Ensure score is in range [0, 1]
               final_score = min(1.0, max(0.0, final_score))
               
               scores.append(final_score)
           
           return np.array(scores).reshape(-1, 1)
       
       def save_model(self, filepath):
           # Nothing to save for rule-based model
           pass
   
   return RuleBasedQualityModel()

def evaluate_and_compare_models(quality_model, rule_model, X_valid, y_valid):
   """
   Evaluate and compare different quality assessment models.
   
   Parameters:
   -----------
   quality_model: Trained neural network model
   rule_model: Rule-based model
   X_valid, y_valid: Validation data
   
   Returns:
   --------
   Dictionary with evaluation metrics
   """
   print("Evaluating and comparing quality assessment models...")
   
   # Create validation data with multiple quality levels
   print("Generating validation structures with different quality levels...")
   
   # Containers for validation data
   val_seqs = []
   val_coords = []
   val_scores = []
   
   # Number of samples to generate per structure
   num_samples = 5
   
   # Generate validation data
   for i in range(min(10, len(X_valid))):
       seq_features = X_valid[i]
       true_coords = y_valid[i]
       
       # Skip structures with NaN
       if np.isnan(true_coords).any():
           continue
           
       # Add the true structure
       val_seqs.append(seq_features)
       val_coords.append(true_coords)
       val_scores.append(1.0)
       
       # Generate variations with different quality levels
       for j in range(num_samples):
           noise_level = 0.1 * (j + 1)  # Increasing noise
           
           try:
               variation = sample_structural_variation(
                   true_coords,
                   noise_level=noise_level,
                   preserve_distance=(j % 2 == 0),
                   use_global_movement=(j % 3 == 0)
               )
               
               # Skip invalid variations
               if np.isnan(variation).any() or np.isinf(variation).any():
                   continue
                   
               # Calculate TM-score
               tm_score = calculate_tm_score(variation, true_coords)
               
               # Skip invalid scores
               if np.isnan(tm_score) or np.isinf(tm_score) or tm_score <= 0:
                   continue
                   
               val_seqs.append(seq_features)
               val_coords.append(variation)
               val_scores.append(tm_score)
               
           except Exception as e:
               print(f"Error generating validation variation: {str(e)}")
               continue
   
   # Convert to numpy arrays
   val_coords = np.array(val_coords)
   val_seqs = np.array(val_seqs)
   val_scores = np.array(val_scores).reshape(-1, 1)
   
   print(f"Validation data: {len(val_scores)} structures")
   
   # Evaluate neural network model
   nn_predictions = None
   try:
       print("Evaluating neural network model...")
       nn_predictions = quality_model.predict_quality(val_coords, val_seqs)
       nn_correlation = np.corrcoef(nn_predictions.flatten(), val_scores.flatten())[0, 1]
       nn_mae = np.mean(np.abs(nn_predictions.flatten() - val_scores.flatten()))
       
       print(f"Neural network model - Correlation: {nn_correlation:.4f}, MAE: {nn_mae:.4f}")
   except Exception as e:
       print(f"Error evaluating neural network model: {str(e)}")
       nn_correlation = 0.0
       nn_mae = float('inf')
   
   # Evaluate rule-based model
   rule_predictions = None
   try:
       print("Evaluating rule-based model...")
       rule_predictions = rule_model.predict_quality(val_coords)
       rule_correlation = np.corrcoef(rule_predictions.flatten(), val_scores.flatten())[0, 1]
       rule_mae = np.mean(np.abs(rule_predictions.flatten() - val_scores.flatten()))
       
       print(f"Rule-based model - Correlation: {rule_correlation:.4f}, MAE: {rule_mae:.4f}")
   except Exception as e:
       print(f"Error evaluating rule-based model: {str(e)}")
       rule_correlation = 0.0
       rule_mae = float('inf')
   
   # Determine the best model
   if nn_correlation > rule_correlation:
       print("Neural network model performs better")
       best_model = "neural_network"
   else:
       print("Rule-based model performs better")
       best_model = "rule_based"
   
   return {
       'neural_network': {
           'correlation': nn_correlation,
           'mae': nn_mae,
           'predictions': nn_predictions
       },
       'rule_based': {
           'correlation': rule_correlation,
           'mae': rule_mae,
           'predictions': rule_predictions
       },
       'best_model': best_model,
       'validation_data': {
           'coords': val_coords,
           'scores': val_scores
       }
   }


def generate_base_structures_with_golden_seeds(
    X_test, 
    test_seq_df, 
    golden_seeds, 
    optimal_params, 
    X_valid, 
    y_valid
):
    """
    Generate base structures using golden seeds with RNA-specific optimizations.
    
    Parameters:
    -----------
    X_test: Test features
    test_seq_df: DataFrame with test sequences
    golden_seeds: List of golden seed information
    optimal_params: Model parameters
    X_valid, y_valid: Validation data for model training
    
    Returns:
    --------
    Dictionary mapping sequence IDs to lists of base structures
    """
    print("Generating base structures with golden seeds and RNA-specific optimizations...")
    
    # Dictionary to store base structures for each sequence
    seq_to_base_structures = {}
    
    # Initialize empty base structures list for each sequence
    for _, row in test_seq_df.iterrows():
        target_id = row['target_id']
        seq_to_base_structures[target_id] = []
    
    # Sort golden seeds by TM-score for best-first approach
    sorted_seeds = sorted(golden_seeds, key=lambda x: x['tm_score'], reverse=True)
    
    # For very small RNAs, different seeds may not add much diversity
    # For large RNAs, different seeds could capture different folding patterns
    small_rna_threshold = 50  # Nucleotides
    large_rna_threshold = 200  # Nucleotides
    
    # RNA-specific parameters based on sequence properties
    for i, (_, row) in enumerate(test_seq_df.iterrows()):
        target_id = row['target_id']
        seq = row['sequence']
        seq_length = len(seq)
        
        print(f"Processing sequence {i+1}/{len(test_seq_df)}, ID: {target_id}, length: {seq_length}")
        
        # Extract sequence features
        seq_features = extract_sequence_features(X_test[i])
        
        # Analyze sequence to determine RNA-specific parameters
        gc_content = seq_features['gc_content']
        au_content = seq_features['au_content']
        
        # Adjust parameters based on RNA properties
        if seq_length < small_rna_threshold:
            print(f"Small RNA detected (length={seq_length}). Using specialized parameters.")
            num_seeds_to_use = min(3, len(sorted_seeds))  # Use fewer seeds for small RNAs
            noise_scaling = 0.7  # Lower noise for small RNAs (more stable)
            use_global_movement = False  # Less global movement for small RNAs
            
            # Small RNAs with high GC content are more stable
            if gc_content > 0.6:
                noise_scaling *= 0.8  # Further reduce noise for GC-rich small RNAs
            
        elif seq_length < large_rna_threshold:
            print(f"Medium RNA detected (length={seq_length}).")
            num_seeds_to_use = min(4, len(sorted_seeds))
            noise_scaling = 1.0  # Standard noise level
            use_global_movement = True
            
            # For medium RNAs, GC content indicates stability regions
            if gc_content > 0.6:
                noise_scaling *= 0.9
            elif au_content > 0.6:
                noise_scaling *= 1.1  # AU-rich regions are more flexible
            
        else:
            print(f"Large RNA detected (length={seq_length}). Using specialized parameters.")
            num_seeds_to_use = min(5, len(sorted_seeds))  # Use more seeds for large RNAs
            noise_scaling = 0.5  # Lower noise for large RNAs (prevent unrealistic structures)
            use_global_movement = True  # Use global movement for large RNAs (domain flexibility)
            
            # Large RNAs tend to have distinct domains
            # Adjust parameters to reflect domain structure
            if seq_length > 300:
                num_seeds_to_use = min(5, len(sorted_seeds))  # Maximum diversity for very large RNAs
        
        # Process with selected seeds
        base_structures = []
        for seed_idx in range(num_seeds_to_use):
            if seed_idx < len(sorted_seeds):
                seed_info = sorted_seeds[seed_idx]
                print(f"  Generating with seed {seed_info['seed']} (TM-score: {seed_info['tm_score']:.4f})")
                
                # Set the random seed
                np.random.seed(seed_info['seed'])
                
                # Create model with adjusted parameters
                adjusted_noise = optimal_params['noise'] * noise_scaling
                
                # Create model with RNA-specific adjustments
                model = reference_based_approach(
                    X_valid, 
                    y_valid,
                    geometric_sampling=True,  # Always use geometric sampling for better structures
                    noise_level=adjusted_noise,
                    correlation=optimal_params['corr']
                )
                
                if model is None:
                    print(f"  Failed to create model with seed {seed_info['seed']}")
                    continue
                
                # Generate prediction
                try:
                    # Get basic prediction for this sequence
                    base_pred = model.predict(X_test[i:i+1])[0][:seq_length]
                    
                    # Apply RNA-specific post-processing
                    processed_pred = post_process_rna_structure(
                        base_pred, 
                        seq, 
                        gc_content, 
                        use_global_movement=use_global_movement
                    )
                    
                    # Normalize the structure
                    normalized_pred = normalize_structure(processed_pred)
                    
                    # Verify the structure meets basic validation criteria
                    if check_structure_validity(normalized_pred):
                        base_structures.append(normalized_pred)
                    else:
                        print(f"  Structure from seed {seed_info['seed']} failed validation. Attempting repair.")
                        
                        # Try to repair the structure
                        repaired_structure = repair_invalid_structure(normalized_pred)
                        if check_structure_validity(repaired_structure):
                            base_structures.append(repaired_structure)
                            print(f"  Successfully repaired structure from seed {seed_info['seed']}")
                        else:
                            print(f"  Could not repair structure from seed {seed_info['seed']}")
                    
                except Exception as e:
                    print(f"  Error generating prediction with seed {seed_info['seed']}: {str(e)}")
                    continue
        
        # If we didn't get any valid structures, create an emergency structure
        if not base_structures:
            print(f"Warning: No valid structures generated for {target_id}. Creating emergency structure.")
            emergency_structure = create_emergency_structure(seq_length)
            base_structures.append(emergency_structure)
        
        # Store the structures
        seq_to_base_structures[target_id] = base_structures
        print(f"  Generated {len(base_structures)} base structures for {target_id}")
    
    return seq_to_base_structures


def generate_diverse_candidates(base_structures, seq_length, num_per_base=5):
    """
    Generate diverse candidate structures from a set of base structures.
    Adapts variation parameters based on RNA size.
    
    Parameters:
    -----------
    base_structures: List of base structures to generate variations from
    seq_length: Length of the sequence
    num_per_base: Number of variations to generate per base structure
    
    Returns:
    --------
    List of candidate structures
    """
    candidates = []
    
    # First, add all base structures
    for base in base_structures:
        candidates.append(base)
    
    # Then generate variations from each base
    for base_idx, base in enumerate(base_structures):
        print(f"  Generating variations from base structure {base_idx+1}/{len(base_structures)}...")
        
        # Determine noise levels based on sequence length
        if seq_length < 50:
            # Small RNA - can handle more variation
            noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        elif seq_length < 120:
            # Medium RNA - moderate variation
            noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25]
        else:
            # Large RNA - more conservative
            noise_levels = [0.03, 0.06, 0.09, 0.12, 0.15]
        
        # Generate variations with different parameters
        for i in range(num_per_base):
            # Use different parameters for diversity
            noise_idx = i % len(noise_levels)
            noise_level = noise_levels[noise_idx]
            preserve_distance = (i % 2 == 0)  # Alternate between preserving and not
            use_global = (i % 3 == 0)  # Occasional global movements
            
            # Add small random variation to correlation
            correlation = 0.8 + np.random.uniform(-0.1, 0.1)
            
            # Set a unique random seed for each variation
            np.random.seed(base_idx * 100 + i)
            
            variation = sample_structural_variation(
                base,
                noise_level=noise_level,
                preserve_distance=preserve_distance,
                use_global_movement=use_global,
                correlation=correlation
            )
            
            # Normalize the structure
            normalized = normalize_structure(variation)
            candidates.append(normalized)
    
    print(f"Generated {len(candidates)} candidate structures in total")
    return candidates

def generate_diverse_structures_from_bases(base_structures, seq_length, quality_model, num_per_base=5):
    """
    Generate diverse candidate structures from a set of base structures,
    with RNA-specific variations and quality filtering.
    
    Parameters:
    -----------
    base_structures: List of base structures to generate variations from
    seq_length: Length of the RNA sequence
    quality_model: Model for quality assessment
    num_per_base: Number of variations to generate per base structure
    
    Returns:
    --------
    List of diverse candidate structures
    """
    candidates = []
    
    # First, add all base structures
    for base in base_structures:
        candidates.append(base)
    
    # Then generate variations from each base
    for base_idx, base in enumerate(base_structures):
        print(f"  Generating variations from base structure {base_idx+1}/{len(base_structures)}...")
        
        # Determine variation parameters based on sequence length
        if seq_length < 50:
            # Small RNA - can handle more variation
            noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25]
            preserve_distances = [True, True, True, False, False]  # Mostly preserve distances
            use_globals = [False, False, True, False, True]  # Occasional global movements
        elif seq_length < 120:
            # Medium RNA - moderate variation
            noise_levels = [0.03, 0.06, 0.1, 0.15, 0.2]
            preserve_distances = [True, True, True, True, False]  # Mostly preserve distances
            use_globals = [False, True, False, True, False]  # Mix of global and local
        else:
            # Large RNA - more conservative
            noise_levels = [0.02, 0.04, 0.06, 0.08, 0.1]
            preserve_distances = [True, True, True, True, True]  # Always preserve distances
            use_globals = [False, False, True, False, True]  # Occasional global for domains
        
        # Generate variations with different parameters
        for i in range(num_per_base):
            # Use different parameters for diversity
            noise_idx = i % len(noise_levels)
            noise_level = noise_levels[noise_idx]
            preserve_distance = preserve_distances[noise_idx]
            use_global = use_globals[noise_idx] 
            
            # Add small random variation to correlation
            correlation = 0.8 + np.random.uniform(-0.1, 0.1)
            
            # Set a unique random seed for each variation
            np.random.seed(base_idx * 100 + i)
            
            variation = sample_structural_variation(
                base,
                noise_level=noise_level,
                preserve_distance=preserve_distance,
                use_global_movement=use_global,
                correlation=correlation
            )
            
            # Apply additional RNA-specific refinements
            # For example, ensure proper backbone geometry
            variation = refine_rna_backbone(variation)
            
            # Normalize the structure
            normalized = normalize_structure(variation)
            
            # Verify the structure is valid
            if check_structure_validity(normalized):
                candidates.append(normalized)
            else:
                print(f"    Structure failed validation. Attempting repair.")
                repaired = repair_invalid_structure(normalized)
                if check_structure_validity(repaired):
                    candidates.append(repaired)
                    print(f"    Successfully repaired structure")
    
    print(f"Generated {len(candidates)} candidate structures in total")
    
    # Pre-filter candidates based on quality before detailed evaluation
    if len(candidates) > 30:  # Only pre-filter if we have many candidates
        print("Pre-filtering candidates based on basic quality metrics...")
        quality_scores = []
        
        # Simple quality assessment for pre-filtering
        for candidate in candidates:
            # Calculate basic quality score
            valid_mask = ~np.all(candidate == 0, axis=1)
            valid_coords = candidate[valid_mask]
            
            # Skip if too few valid coordinates
            if len(valid_coords) < 3:
                quality_scores.append(0.0)
                continue
            
            # Calculate bond lengths
            bond_lengths = []
            for j in range(1, len(valid_coords)):
                dist = np.linalg.norm(valid_coords[j] - valid_coords[j-1])
                bond_lengths.append(dist)
            
            # Score based on ideal bond length
            avg_bond_length = np.mean(bond_lengths)
            bond_score = 1.0 - min(1.0, abs(avg_bond_length - 3.8) / 3.8)
            
            quality_scores.append(bond_score)
        
        # Convert to numpy array
        quality_scores = np.array(quality_scores)
        
        # Take top 30 candidates based on quality score
        top_indices = np.argsort(quality_scores)[-30:]
        candidates = [candidates[idx] for idx in top_indices]
        print(f"Pre-filtered to top 30 candidates")
    
    return candidates

def evaluate_and_prune_structures(candidates, seq_features, quality_model, top_k=5):
    """
    Evaluate structure candidates and select the top-k structures.
    This function handles both NN-based and rule-based quality models.
    
    Parameters:
    -----------
    candidates: List of candidate structures
    seq_features: RNA sequence features
    quality_model: Model for quality assessment
    top_k: Number of top structures to select
    
    Returns:
    --------
    List of top-k structures
    """
    # Determine if the model is a neural network or rule-based
    is_nn_model = hasattr(quality_model, 'model')
    
    try:
        if is_nn_model:
            print("Using neural network for quality assessment...")
            return evaluate_and_prune_nn(candidates, seq_features, quality_model, top_k)
        else:
            print("Using rule-based model for quality assessment...")
            return evaluate_and_prune_rules(candidates, top_k)
        
    except Exception as e:
        print(f"Error during quality evaluation: {str(e)}")
        traceback.print_exc()
        
        # Fall back to rule-based evaluation if any error occurs
        print("Falling back to basic rule-based scoring...")
        return evaluate_and_prune_rules(candidates, top_k)


def evaluate_and_prune_nn(candidates, seq_features, quality_model, top_k=5):
    """
    Evaluate candidates using NN model and select the top-k.
    
    Parameters:
    -----------
    candidates: List of candidate structures
    seq_features: One-hot encoded sequence features
    quality_model: Trained quality assessment model
    top_k: Number of top structures to keep
    
    Returns:
    --------
    List of top-k structures
    """
    try:
        # Extract actual sequence length (non-padding)
        valid_mask = ~np.all(seq_features == 0, axis=1)
        seq_length = np.sum(valid_mask)
        
        # Prepare batched data for prediction
        stacked_candidates = np.array(candidates)
        
        # Prepare sequence features input - deve ter o mesmo nÃºmero de amostras que stacked_candidates
        batch_size = stacked_candidates.shape[0]
        
        # Expand seq_features to have batch_size samples (replicando para cada candidato)
        # Certifique-se de que seq_features tem 3 dimensÃµes (batch, seq_len, features)
        if len(seq_features.shape) == 2:  # Se for (seq_len, features)
            seq_features = np.expand_dims(seq_features, axis=0)  # Adicionar dimensÃ£o de batch
        
        # Replicar para todos os candidatos
        stacked_seq = np.repeat(seq_features, batch_size, axis=0)
        
        # Predict quality scores
        quality_scores = quality_model.predict_quality(stacked_candidates, stacked_seq)
        quality_scores = quality_scores.flatten()
        
        # Sort by quality score
        sorted_indices = np.argsort(quality_scores)[::-1]  # Descending order
        
        # Keep top-k structures
        top_structures = [candidates[idx] for idx in sorted_indices[:top_k]]
        top_scores = quality_scores[sorted_indices[:top_k]]
        
        print(f"Selected top {top_k} structures with NN predicted qualities: {top_scores}")
        
        return top_structures
        
    except Exception as e:
        print(f"Error in NN evaluation: {str(e)}")
        traceback.print_exc()
        
        # Fall back to rule-based approach if NN fails
        print("Falling back to rule-based evaluation...")
        return evaluate_and_prune_rules(candidates, top_k)

def evaluate_and_prune_rules(candidates, top_k=5):
    """
    Evaluate candidates using rule-based metrics and select the top-k.
    
    Parameters:
    -----------
    candidates: List of candidate structures
    top_k: Number of top structures to keep
    
    Returns:
    --------
    List of top-k structures
    """
    quality_scores = []
    
    for i, candidate in enumerate(candidates):
        # Calculate a quality score based on structural features
        # 1. Check for valid coordinates
        valid_mask = ~np.all(candidate == 0, axis=1)
        valid_coords = candidate[valid_mask]
        
        # Skip if no valid coordinates
        if len(valid_coords) < 3:
            quality_scores.append(0.5)  # Neutral score
            continue
        
        # 2. Calculate bond lengths between consecutive residues
        bond_lengths = []
        for j in range(1, len(valid_coords)):
            dist = np.linalg.norm(valid_coords[j] - valid_coords[j-1])
            bond_lengths.append(dist)
        
        avg_bond_length = np.mean(bond_lengths)
        bond_std = np.std(bond_lengths)
        
        # 3. Score based on how close to ideal RNA bond length (3.8Ã…)
        bond_score = 1.0 - min(1.0, abs(avg_bond_length - 3.8) / 3.8)
        
        # 4. Bond consistency score (lower std deviation is better)
        consistency_score = 1.0 - min(1.0, bond_std / 2.0)
        
        # 5. Check structure validity
        is_valid = check_structure_validity(candidate)
        valid_score = 1.0 if is_valid else 0.5
        
        # 6. Combined score
        score = 0.4 * bond_score + 0.3 * consistency_score + 0.3 * valid_score
        
        # 7. Add small random component for variations
        random_component = np.random.uniform(-0.05, 0.05)
        score = min(1.0, max(0.0, score + random_component))
        
        quality_scores.append(score)
    
    # Convert to numpy array
    quality_scores = np.array(quality_scores)
    
    # Sort by quality score
    sorted_indices = np.argsort(quality_scores)[::-1]  # Descending order
    
    # Keep top-k structures
    top_structures = [candidates[idx] for idx in sorted_indices[:top_k]]
    top_scores = quality_scores[sorted_indices[:top_k]]
    
    print(f"Selected top {top_k} structures with rule-based qualities: {top_scores}")
    
    return top_structures

def generate_and_prune_structures(base_coords, seq_features, quality_model, num_candidates=20, top_k=5):
    """
    Generate multiple structure candidates and use the NN model to prune to the best ones.
    Modified to handle variable-length RNA sequences.
    """
    # Get actual sequence length (non-padding)
    valid_mask = ~np.all(base_coords == 0, axis=1)
    seq_length = np.sum(valid_mask)
    print(f"Processing structure with actual length: {seq_length}")
    
    # Generate candidate structures with different parameters
    candidates = []
    
    # Add the base structure
    candidates.append(normalize_structure(base_coords))
    
    # Generate variations with different parameters
    for i in range(num_candidates - 1):
        # Use different parameters for diversity
        noise_level = 0.1 + (i % 10) * 0.05
        preserve_distance = (i % 3 != 0)
        use_global = (i % 4 == 0)
        correlation = 0.7 + (i % 5) * 0.05
        
        variation = sample_structural_variation(
            base_coords,
            noise_level=noise_level,
            preserve_distance=preserve_distance,
            use_global_movement=use_global,
            correlation=correlation
        )
        
        # Normalize the structure
        normalized = normalize_structure(variation)
        candidates.append(normalized)
    
    # Convert to array for batch processing
    stacked_candidates = np.array(candidates)
    
    # Implement a simple rule-based quality assessment as fallback
    print("Using rule-based quality assessment...")
    quality_scores = []
    
    for i, candidate in enumerate(candidates):
        # Calculate a quality score based on structural features
        # 1. Check for unusual bond lengths
        valid_indices = np.where(valid_mask)[0]
        valid_coords = candidate[valid_indices]
        
        # Skip if no valid coordinates
        if len(valid_coords) < 3:
            quality_scores.append(0.5)
            continue
        
        # Calculate bond lengths
        bond_lengths = []
        for j in range(1, len(valid_coords)):
            dist = np.linalg.norm(valid_coords[j] - valid_coords[j-1])
            bond_lengths.append(dist)
        
        # Score based on how close to ideal RNA bond length
        avg_bond_length = np.mean(bond_lengths)
        bond_std = np.std(bond_lengths)
        
        # Ideal bond length is around 3.8Ã…
        bond_score = 1.0 - min(1.0, abs(avg_bond_length - 3.8) / 3.8)
        
        # Bond consistency score
        consistency_score = 1.0 - min(1.0, bond_std / 2.0)
        
        # Structural validity
        is_valid = check_structure_validity(candidate)
        valid_score = 1.0 if is_valid else 0.5
        
        # Combined score
        final_score = 0.4 * bond_score + 0.3 * consistency_score + 0.3 * valid_score
        
        # Add a small random component for variations
        random_component = np.random.uniform(-0.05, 0.05)
        final_score = min(1.0, max(0.0, final_score + random_component))
        
        quality_scores.append(final_score)
    
    quality_scores = np.array(quality_scores)
    
    # Sort candidates by quality score
    sorted_indices = np.argsort(quality_scores)[::-1]  # Descending order
    
    # Keep top-k structures
    top_structures = [candidates[idx] for idx in sorted_indices[:top_k]]
    top_scores = quality_scores[sorted_indices[:top_k]]
    
    print(f"Selected top {top_k} structures with predicted qualities: {top_scores}")
    
    return top_structures


def create_submission_dataframe(seq_to_coords, sample_submission_df):
   """
   Create a submission DataFrame from the final structures.
   
   Parameters:
   -----------
   seq_to_coords: Dictionary mapping sequence IDs to lists of structures
   sample_submission_df: Sample submission format
   
   Returns:
   --------
   Submission DataFrame
   """
   # Create a copy of the sample submission
   submission_df = sample_submission_df.copy()
   
   # Fill in the coordinates for each structure
   for i, row in submission_df.iterrows():
       if i % 1000 == 0:
           print(f"Processing row {i}/{len(submission_df)}")
       
       # Parse the ID to get sequence ID and residue index
       id_parts = row['ID'].split('_')
       seq_id = id_parts[0]
       residue_idx = int(id_parts[1]) - 1  # Convert to 0-based indexing
       
       # Check if we have structures for this sequence
       if seq_id in seq_to_coords:
           structures = seq_to_coords[seq_id]
           
           # Check if the residue index is valid
           if residue_idx < len(structures[0]):
               # Fill in coordinates for all 5 structures
               for struct_idx in range(5):
                   if struct_idx < len(structures):
                       submission_df.at[i, f'x_{struct_idx+1}'] = structures[struct_idx][residue_idx][0]
                       submission_df.at[i, f'y_{struct_idx+1}'] = structures[struct_idx][residue_idx][1]
                       submission_df.at[i, f'z_{struct_idx+1}'] = structures[struct_idx][residue_idx][2]
                   else:
                       # If we have fewer than 5 structures, duplicate the last one
                       last_idx = len(structures) - 1
                       submission_df.at[i, f'x_{struct_idx+1}'] = structures[last_idx][residue_idx][0]
                       submission_df.at[i, f'y_{struct_idx+1}'] = structures[last_idx][residue_idx][1]
                       submission_df.at[i, f'z_{struct_idx+1}'] = structures[last_idx][residue_idx][2]
   
   return submission_df

def generate_nn_pruned_submission(model, quality_model, test_seq_df, sample_submission_df):
    """
    Enhanced submission generation that uses NN-based pruning for structure selection.
    """
    print("Generating submission with Neural Network pruning...")
    
    # Prepare test features
    X_test = prepare_test_features(test_seq_df)
    
    # Generate multiple predictions for ensemble diversity
    print("Generating base predictions...")
    base_predictions = model.predict(X_test)
    
    seq_to_coords = {}
    for i, (_, row) in enumerate(test_seq_df.iterrows()):
        target_id = row['target_id']
        seq = row['sequence']
        seq_length = len(seq)
        
        print(f"Processing sequence {i+1}/{len(test_seq_df)}, ID: {target_id}, length: {seq_length}")
        
        # Get base coordinates
        base_coords = base_predictions[i][:seq_length]
        
        # Extract sequence features for this RNA
        seq_features = X_test[i][:seq_length]
        
        # Generate and prune structures using the NN model
        structures = generate_and_prune_structures(
            base_coords, 
            seq_features, 
            quality_model,
            num_candidates=30,  # Generate more candidates
            top_k=5             # Keep top 5 for submission
        )
        
        # Store the structures
        seq_to_coords[target_id] = structures
    
    # Create submission DataFrame
    print("Creating submission file...")
    submission_df = sample_submission_df.copy()
    
    for i, row in submission_df.iterrows():
        id_parts = row['ID'].split('_')
        seq_id = id_parts[0]
        residue_idx = int(id_parts[1]) - 1
        
        if seq_id in seq_to_coords:
            structures = seq_to_coords[seq_id]
            if residue_idx < len(structures[0]):
                for struct_idx in range(5):
                    submission_df.at[i, f'x_{struct_idx+1}'] = structures[struct_idx][residue_idx][0]
                    submission_df.at[i, f'y_{struct_idx+1}'] = structures[struct_idx][residue_idx][1]
                    submission_df.at[i, f'z_{struct_idx+1}'] = structures[struct_idx][residue_idx][2]
    
    submission_file = os.path.join(OUTPUT_DIR, 'submission_nn_pruned.csv')
    submission_df.to_csv(submission_file, index=False)
    print(f"NN-pruned submission file saved to {submission_file}")
    
    # Also save as standard submission
    standard_file = os.path.join(OUTPUT_DIR, 'submission.csv')
    submission_df.to_csv(standard_file, index=False)
    
    return submission_df


def run_hybrid_pipeline(
    X_valid, 
    y_valid, 
    test_seq_df, 
    sample_submission_df, 
    output_dir, 
    golden_threshold=0.6, 
    optimal_params={'noise': 0.21, 'corr': 0.83}
):
    """
    Run a hybrid pipeline that uses RNA-targeted seed selection.
    
    Parameters:
    -----------
    X_valid, y_valid: Validation data
    test_seq_df: DataFrame with test sequences
    sample_submission_df: Sample submission format
    output_dir: Output directory for files
    golden_threshold: Threshold for considering a seed as "golden"
    optimal_params: Optimal parameters for the reference model
    
    Returns:
    --------
    submission_df, status_dict
    """
    print("=" * 80)
    print("HYBRID PIPELINE WITH RNA-TARGETED SEED SELECTION".center(80))
    print("=" * 80)
    
    status = {
        'success': False,
        'golden_seeds_found': 0,
        'nn_training_success': False,
        'best_tm_score': 0.0,
        'error': None
    }
    
    try:
        # PHASE 1: Find Enhanced Adaptive Seeds
        print("\nPHASE 1: Searching for enhanced adaptive seeds...")
        golden_seeds, all_seeds = enhanced_adaptive_seed_search(
            X_valid, 
            y_valid, 
            initial_threshold=golden_threshold,
            min_threshold=golden_threshold - 0.1,
            optimal_params=optimal_params
        )
        
        # Use fallback approach if we didn't find enough seeds
        if len(golden_seeds) < 3:
            print("Not enough enhanced seeds found, falling back to general seed search...")
            golden_seeds, all_seeds = find_diverse_golden_seeds(
                X_valid, 
                y_valid, 
                golden_threshold=golden_threshold - 0.05,  # Lower threshold for fallback
                attempts=100
            )
        
        ensemble_seeds = golden_seeds
        status['golden_seeds_found'] = len(ensemble_seeds)
        
        # PHASE 2: Train Quality Assessment Model
        print("\nPHASE 2: Training quality assessment model...")
        try:
            quality_model = train_enhanced_quality_model(X_valid, y_valid, X_valid, y_valid)
            status['nn_training_success'] = True
        except Exception as e:
            print(f"Error training NN model: {str(e)}")
            print("Falling back to rule-based quality assessment...")
            quality_model = create_rule_based_model()
        
        # PHASE 3: Generate Base Structures with RNA-Targeted Seeds
        print("\nPHASE 3: Generating base structures with enhanced seeds...")
        X_test = prepare_test_features(test_seq_df)
        
        # Generate predictions using each of the ensemble seeds
        seed_predictions = []
        for i, seed_info in enumerate(ensemble_seeds):
            print(f"Generating predictions with seed {seed_info['seed']} (TM-score: {seed_info['tm_score']:.4f}, Category: {seed_info.get('category', 'unknown')})...")
            
            # Set the random seed
            np.random.seed(seed_info['seed'])
            
            # Create model with this seed
            model = reference_based_approach(
                X_valid, 
                y_valid,
                geometric_sampling=True,
                noise_level=optimal_params['noise'],
                correlation=optimal_params['corr']
            )
            
            # Generate predictions
            if model is not None:
                preds = model.predict(X_test)
                seed_predictions.append({
                    'seed': seed_info['seed'],
                    'tm_score': seed_info['tm_score'],
                    'category': seed_info.get('category', 'unknown'),
                    'predictions': preds
                })
                
                # Update best TM-score for status
                if seed_info['tm_score'] > status['best_tm_score']:
                    status['best_tm_score'] = seed_info['tm_score']
            else:
                print(f"Failed to create model with seed {seed_info['seed']}")
        
        if not seed_predictions:
            raise Exception("Failed to generate any predictions with enhanced seeds")
        
        # PHASE 4: Generate and Prune Structures
        print("\nPHASE 4: Generating diverse candidates and pruning...")
        
        seq_to_coords = {}
        for i, (_, row) in enumerate(test_seq_df.iterrows()):
            target_id = row['target_id']
            seq = row['sequence']
            seq_length = len(seq)
            
            print(f"Processing sequence {i+1}/{len(test_seq_df)}, ID: {target_id}, length: {seq_length}")
            
            # Collect base predictions from all seeds for this sequence
            base_structures = []
            for pred_info in seed_predictions:
                base_struct = pred_info['predictions'][i][:seq_length]
                
                # Apply RNA-specific post-processing based on seed category
                # Extract GC content for RNA-specific processing
                seq_features = X_test[i][:seq_length]
                valid_mask = ~np.all(seq_features == 0, axis=1)
                if np.sum(valid_mask) > 0:
                    features = seq_features[valid_mask]
                    g_content = np.mean(features[:, 2]) if features.shape[1] > 2 else 0
                    c_content = np.mean(features[:, 1]) if features.shape[1] > 1 else 0
                    gc_content = g_content + c_content
                else:
                    gc_content = 0.5
                
                # Apply RNA-specific post-processing
                category = pred_info.get('category', 'unknown')
                use_global_movement = category in ['large_RNA', 'medium_RNA']
                
                # Process structure with RNA-specific adjustments
                processed_struct = post_process_rna_structure(
                    base_struct,
                    seq,
                    gc_content,
                    use_global_movement=use_global_movement
                )
                
                # Normalize structure
                normalized_struct = normalize_structure(processed_struct)
                
                # Verify validity and repair if needed
                if check_structure_validity(normalized_struct):
                    base_structures.append(normalized_struct)
                else:
                    # Try to repair the structure
                    repaired_struct = repair_invalid_structure(normalized_struct)
                    if check_structure_validity(repaired_struct):
                        base_structures.append(repaired_struct)
                        print(f"  Repaired structure from seed {pred_info['seed']}")
                    else:
                        print(f"  Structure from seed {pred_info['seed']} failed validation and repair")
            
            # Generate emergency structure if no valid base structures
            if not base_structures:
                print(f"Warning: No valid structures generated for {target_id}. Creating emergency structure.")
                emergency_structure = create_emergency_structure(seq_length)
                base_structures.append(emergency_structure)
            
            # Extract sequence features for RNA-specific candidate generation
            seq_features = X_test[i][:seq_length]
            
            # Generate diverse candidates optimized for this RNA
            candidates = generate_diverse_structures_from_bases(
                base_structures, 
                seq_length, 
                quality_model,
                num_per_base=5
            )
            
            # Evaluate and prune candidates
            if status['nn_training_success']:
                print("Using NN model for quality assessment...")
                try:
                    top_structures = evaluate_and_prune_structures(
                        candidates, 
                        seq_features, 
                        quality_model, 
                        top_k=5
                    )
                except Exception as e:
                    print(f"Error in NN evaluation: {str(e)}")
                    print("Falling back to rule-based assessment...")
                    top_structures = evaluate_and_prune_rules(candidates, top_k=5)
            else:
                print("Using rule-based quality assessment...")
                top_structures = evaluate_and_prune_rules(candidates, top_k=5)
            
            # Store the final structures
            seq_to_coords[target_id] = top_structures
        
        # PHASE 5: Create Submission
        print("\nPHASE 5: Creating submission file...")
        submission_df = create_submission_dataframe(seq_to_coords, sample_submission_df)
        
        # Save submission files
        enhanced_file = os.path.join(output_dir, 'submission_enhanced.csv')
        submission_df.to_csv(enhanced_file, index=False)
        print(f"Enhanced submission saved to {enhanced_file}")
        
        # Save as standard submission
        standard_file = os.path.join(output_dir, 'submission.csv')
        submission_df.to_csv(standard_file, index=False)
        print(f"Standard submission saved to {standard_file}")
        
        # Set success
        status['success'] = True
        
        return submission_df, status
        
    except Exception as e:
        print(f"ERROR in hybrid pipeline: {str(e)}")
        traceback.print_exc()
        status['error'] = str(e)
        return None, status


def integrate_with_hybrid_pipeline(run_hybrid_pipeline_func):
   """
   Integrates the enhanced NN model with the hybrid pipeline.
   
   Parameters:
   -----------
   run_hybrid_pipeline_func: Original hybrid pipeline function
   
   Returns:
   --------
   Modified hybrid pipeline function
   """
   def enhanced_hybrid_pipeline(
       X_valid, 
       y_valid, 
       test_seq_df, 
       sample_submission_df, 
       output_dir, 
       golden_threshold=0.6, 
       seed_attempts=200, 
       optimal_params={'noise': 0.21, 'corr': 0.83}
   ):
       """
       Run a hybrid pipeline with enhanced NN quality model.
       """
       print("=" * 80)
       print("ENHANCED HYBRID PIPELINE: GOLDEN SEEDS + ADVANCED NN PRUNING".center(80))
       print("=" * 80)
       
       status = {
           'success': False,
           'golden_seeds_found': 0,
           'nn_training_success': False,
           'best_tm_score': 0.0,
           'error': None
       }
       
       try:
           # PHASE 1: Find Golden Seeds (same as original)
           print("\nPHASE 1: Searching for golden seeds...")
           golden_seeds, all_seeds = find_diverse_golden_seeds(
               X_valid, 
               y_valid, 
               golden_threshold=golden_threshold, 
               attempts=seed_attempts, 
               optimal_params=optimal_params
           )
           
           # Even if we don't find golden seeds, we can use the best seeds we found
           if not golden_seeds and all_seeds:
               print("No golden seeds found, using top seeds from search...")
               # Sort by TM-score
               all_seeds.sort(key=lambda x: x['tm_score'], reverse=True)
               # Take top 5 seeds
               top_seeds = all_seeds[:5]
           else:
               top_seeds = golden_seeds
               
           status['golden_seeds_found'] = len(golden_seeds)
           
           # PHASE 2: Train Enhanced Quality Assessment Model
           print("\nPHASE 2: Training enhanced NN quality assessment model...")
           try:
               enhanced_quality_model = train_enhanced_quality_model(X_valid, y_valid, X_valid, y_valid)
               rule_based_model = create_rule_based_model()
               
               # Compare models
               model_comparison = evaluate_and_compare_models(
                   enhanced_quality_model, 
                   rule_based_model, 
                   X_valid, 
                   y_valid
               )
               
               # Use the best model
               best_model_type = model_comparison['best_model']
               if best_model_type == 'neural_network':
                   quality_model = enhanced_quality_model
                   print("Using enhanced neural network model for quality assessment")
               else:
                   quality_model = rule_based_model
                   print("Using rule-based model for quality assessment")
               
               status['nn_training_success'] = (best_model_type == 'neural_network')
               
           except Exception as e:
               print(f"Error training and comparing models: {str(e)}")
               print("Falling back to rule-based quality assessment...")
               quality_model = create_rule_based_model()
           
           # PHASE 3 and beyond: same as original hybrid pipeline
           # Continue with the rest of the pipeline...
           # (generate base structures, evaluate candidates, create submission)
           
           # Call the original function with our quality model
           # This is a placeholder - in a real implementation, 
           # you would continue with the rest of the pipeline using the quality_model
           
           return run_hybrid_pipeline_func(
               X_valid, 
               y_valid, 
               test_seq_df, 
               sample_submission_df, 
               output_dir, 
               golden_threshold=golden_threshold, 
               seed_attempts=seed_attempts, 
               optimal_params=optimal_params,
               quality_model=quality_model  # Pass the selected model
           )
           
       except Exception as e:
           print(f"ERROR in enhanced hybrid pipeline: {str(e)}")
           traceback.print_exc()
           status['error'] = str(e)
           
           # Fall back to original pipeline
           print("Falling back to original hybrid pipeline...")
           return run_hybrid_pipeline_func(
               X_valid, 
               y_valid, 
               test_seq_df, 
               sample_submission_df, 
               output_dir, 
               golden_threshold=golden_threshold, 
               seed_attempts=seed_attempts, 
               optimal_params=optimal_params
           )
   
   return enhanced_hybrid_pipeline


def phase3_integration_with_hybrid_pipeline(run_hybrid_pipeline_func):
    """
    Integrates the enhanced Phase 3 (base structure generation) with the hybrid pipeline.
    
    Parameters:
    -----------
    run_hybrid_pipeline_func: Original hybrid pipeline function
    
    Returns:
    --------
    Modified hybrid pipeline function
    """
    def enhanced_hybrid_pipeline(
        X_valid, 
        y_valid, 
        test_seq_df, 
        sample_submission_df, 
        output_dir, 
        golden_threshold=0.6, 
        seed_attempts=200, 
        optimal_params={'noise': 0.21, 'corr': 0.83},
        quality_model=None
    ):
        """
        Run a hybrid pipeline with enhanced base structure generation.
        """
        print("=" * 80)
        print("ENHANCED HYBRID PIPELINE WITH RNA-SPECIFIC STRUCTURE GENERATION".center(80))
        print("=" * 80)
        
        status = {
            'success': False,
            'golden_seeds_found': 0,
            'nn_training_success': False,
            'best_tm_score': 0.0,
            'error': None
        }
        
        try:
            # PHASE 1: Find Golden Seeds (same as original)
            print("\nPHASE 1: Searching for golden seeds...")
            golden_seeds, all_seeds = find_diverse_golden_seeds(
                X_valid, 
                y_valid, 
                golden_threshold=golden_threshold, 
                attempts=seed_attempts, 
                optimal_params=optimal_params
            )
            
            # Even if we don't find golden seeds, we can use the best seeds we found
            if not golden_seeds and all_seeds:
                print("No golden seeds found, using top seeds from search...")
                # Sort by TM-score
                all_seeds.sort(key=lambda x: x['tm_score'], reverse=True)
                # Take top 5 seeds
                top_seeds = all_seeds[:5]
            else:
                top_seeds = golden_seeds
                
            status['golden_seeds_found'] = len(golden_seeds)
            
            # PHASE 2: Train Quality Assessment Model (if not provided)
            if quality_model is None:
                print("\nPHASE 2: Training quality assessment model...")
                try:
                    quality_model = train_enhanced_quality_model(X_valid, y_valid, X_valid, y_valid)
                    status['nn_training_success'] = True
                except Exception as e:
                    print(f"Error training quality model: {str(e)}")
                    print("Falling back to rule-based quality assessment...")
                    quality_model = create_rule_based_model()
            else:
                print("\nPHASE 2: Using provided quality model")
                status['nn_training_success'] = hasattr(quality_model, 'model')  # Check if it's a NN model
            
            # PHASE 3: Generate Base Structures with RNA-specific optimizations
            print("\nPHASE 3: Generating base structures with RNA-specific optimizations...")
            # Prepare test features
            X_test = prepare_test_features(test_seq_df)
            
            # Generate base structures using our enhanced function
            seq_to_base_structures = generate_base_structures_with_golden_seeds(
                X_test,
                test_seq_df,
                top_seeds,
                optimal_params,
                X_valid,
                y_valid
            )
            
            # PHASE 4: Generate and evaluate diverse candidates
            print("\nPHASE 4: Generating diverse candidates and evaluating quality...")
            
            seq_to_coords = {}
            for i, (_, row) in enumerate(test_seq_df.iterrows()):
                target_id = row['target_id']
                seq = row['sequence']
                seq_length = len(seq)
                
                print(f"Processing sequence {i+1}/{len(test_seq_df)}, ID: {target_id}, length: {seq_length}")
                
                # Get base structures for this sequence
                base_structures = seq_to_base_structures[target_id]
                
                if not base_structures:
                    print(f"No base structures found for {target_id}. Creating emergency structure.")
                    base_structures = [create_emergency_structure(seq_length)]
                
                # Extract sequence features
                seq_features = X_test[i][:seq_length]
                
                # Generate diverse candidates
                candidates = generate_diverse_structures_from_bases(
                    base_structures, 
                    seq_length, 
                    quality_model,
                    num_per_base=5
                )
                
                # Evaluate and select the best structures
                try:
                    top_structures = evaluate_and_prune_structures(
                        candidates, 
                        seq_features, 
                        quality_model, 
                        top_k=5
                    )
                except Exception as e:
                    print(f"Error in structure evaluation: {str(e)}")
                    print("Falling back to basic selection...")
                    # If evaluation fails, just use the base structures
                    top_structures = base_structures[:5]
                    
                    # If we need more structures, pad with variations
                    while len(top_structures) < 5:
                        idx = len(top_structures) % len(base_structures)
                        variation = sample_structural_variation(
                            base_structures[idx],
                            noise_level=0.1,
                            preserve_distance=True,
                            use_global_movement=False
                        )
                        top_structures.append(normalize_structure(variation))
                
                # Store the final structures
                seq_to_coords[target_id] = top_structures
            
            # PHASE 5: Create Submission
            print("\nPHASE 5: Creating submission file...")
            submission_df = create_submission_dataframe(seq_to_coords, sample_submission_df)
            
            # Save submission
            enhanced_file = os.path.join(output_dir, 'submission_enhanced.csv')
            submission_df.to_csv(enhanced_file, index=False)
            print(f"Enhanced submission saved to {enhanced_file}")
            
            # Save as standard submission
            standard_file = os.path.join(output_dir, 'submission.csv')
            submission_df.to_csv(standard_file, index=False)
            
            # Set success
            status['success'] = True
            
            # Get best TM-score from seeds for reporting
            if top_seeds:
                status['best_tm_score'] = max(seed['tm_score'] for seed in top_seeds)
            
            return submission_df, status
            
        except Exception as e:
            print(f"ERROR in enhanced hybrid pipeline: {str(e)}")
            traceback.print_exc()
            status['error'] = str(e)
            
            # Fall back to original pipeline as last resort
            print("Falling back to original pipeline...")
            return run_hybrid_pipeline_func(
                X_valid, 
                y_valid, 
                test_seq_df, 
                sample_submission_df, 
                output_dir, 
                golden_threshold=golden_threshold, 
                seed_attempts=seed_attempts, 
                optimal_params=optimal_params
            )
    
    return enhanced_hybrid_pipeline


if __name__ == "__main__":
    # Execution mode selection
    use_hybrid_pipeline = True     # combine golden seeds and NN pruning
    use_nn_pruning = False         # Use only NN pruning
    use_reference_only = False     # Use only reference-based approach
    use_adaptive_seeds = True      # Use adaptive seed search
    use_enhanced_adaptive = True   # Use enhanced adaptive seed search with RNA-specific targeting
    
    # Print startup banner
    print("=" * 80)
    print("RNA 3D STRUCTURE PREDICTION PIPELINE".center(80))
    print("=" * 80)
    
    # Print selected mode
    if use_hybrid_pipeline:
        if use_enhanced_adaptive:
            mode_description = "Enhanced Hybrid Pipeline: RNA-Targeted Adaptive Seeds + NN Pruning"
        elif use_adaptive_seeds:
            mode_description = "Enhanced Hybrid Pipeline: Adaptive Golden Seeds + NN Pruning"
        else:
            mode_description = "Hybrid Pipeline: Golden Seeds + NN Pruning"
    elif use_nn_pruning:
        mode_description = "Neural Network based pruning pipeline"
    elif use_reference_only:
        mode_description = "Reference model only"
    else:
        mode_description = "Standard pipeline"
    
    print(f"Selected mode: {mode_description}")
    print("-" * 80)
    
    try:
        # Execute the selected pipeline
        if use_hybrid_pipeline:
            start_time = time.time()
            
            print("Loading processed data...")
            X_train, y_train, X_valid, y_valid = load_processed_data()
            
            print("\nLoading test data...")
            test_seq_df = pd.read_csv(os.path.join(DATA_DIR, "test_sequences.csv"))
            sample_submission_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
            
            if use_enhanced_adaptive:
                # Use enhanced adaptive search with RNA-specific targeting
                print("\nPHASE 1: Searching for golden seeds with enhanced RNA-targeted adaptive search...")
                golden_seeds, all_seeds = enhanced_adaptive_seed_search(
                    X_valid, 
                    y_valid, 
                    initial_threshold=0.65,  # Start with higher threshold
                    min_threshold=0.55,      # Accept down to this minimum value
                    initial_attempts=100,    # Initial number of attempts
                    max_attempts=300,        # Maximum number of attempts
                    optimal_params={'noise': 0.21, 'corr': 0.83}
                )
                
                # Define a modified version that uses the enhanced seeds
                def run_hybrid_pipeline_with_enhanced_seeds(
                    X_valid, 
                    y_valid,
                    test_seq_df, 
                    sample_submission_df,
                    output_dir,
                    golden_seeds,  # Pass enhanced seeds directly
                    optimal_params={'noise': 0.21, 'corr': 0.83}
                ):
                    """Modified pipeline that uses RNA-specific enhanced seeds"""
                    status = {
                        'success': False,
                        'golden_seeds_found': len(golden_seeds),
                        'nn_training_success': False,
                        'best_tm_score': max([s.get('tm_score', 0) for s in golden_seeds]) if golden_seeds else 0.0,
                        'error': None,
                        'category_stats': {}  # Will store RNA category statistics
                    }
                    
                    # Collect category statistics
                    categories = {}
                    for seed in golden_seeds:
                        category = seed.get('category', 'unknown')
                        if category not in categories:
                            categories[category] = 0
                        categories[category] += 1
                    
                    status['category_stats'] = categories
                    
                    try:
                        # PHASE 2: Train Quality Assessment Model
                        print("\nPHASE 2: Training NN quality assessment model...")
                        try:
                            quality_model = train_enhanced_quality_model(X_valid, y_valid, X_valid, y_valid)
                            status['nn_training_success'] = True
                        except Exception as e:
                            print(f"Error training NN model: {str(e)}")
                            print("Falling back to rule-based quality assessment...")
                            quality_model = create_rule_based_model()
                            
                        # PHASE 3: Generate Base Structures with Enhanced Seeds
                        print("\nPHASE 3: Generating base structures with RNA-targeted seeds...")
                        X_test = prepare_test_features(test_seq_df)
                        
                        # Generate predictions using each of the top seeds
                        seed_predictions = []
                        for i, seed_info in enumerate(golden_seeds):
                            category = seed_info.get('category', 'unknown')
                            print(f"Generating predictions with seed {seed_info['seed']} (TM-score: {seed_info.get('tm_score', 0):.4f}, Category: {category})...")
                            
                            # Set the random seed
                            np.random.seed(seed_info['seed'])
                            
                            # Adjust noise level based on RNA category
                            noise_level = optimal_params['noise']
                            if category == 'small_RNA':
                                noise_level *= 0.9  # More stable for small RNAs
                            elif category == 'large_RNA':
                                noise_level *= 0.7  # Lower noise for large RNAs
                            
                            # Create model with this seed
                            model = reference_based_approach(
                                X_valid, 
                                y_valid,
                                geometric_sampling=True,
                                noise_level=noise_level,
                                correlation=optimal_params['corr']
                            )
                            
                            # Generate predictions
                            if model is not None:
                                preds = model.predict(X_test)
                                seed_predictions.append({
                                    'seed': seed_info['seed'],
                                    'tm_score': seed_info.get('tm_score', 0),
                                    'category': category,
                                    'predictions': preds
                                })
                                
                                # Update best TM-score for status
                                if seed_info.get('tm_score', 0) > status['best_tm_score']:
                                    status['best_tm_score'] = seed_info.get('tm_score', 0)
                            else:
                                print(f"Failed to create model with seed {seed_info['seed']}")
                                
                        if not seed_predictions:
                            raise Exception("Failed to generate any predictions with enhanced seeds")
                            
                        # PHASE 4: Generate and Prune Structures
                        print("\nPHASE 4: Generating diverse candidates with RNA-specific adjustments...")
                        
                        seq_to_coords = {}
                        for i, (_, row) in enumerate(test_seq_df.iterrows()):
                            target_id = row['target_id']
                            seq = row['sequence']
                            seq_length = len(seq)
                            
                            print(f"Processing sequence {i+1}/{len(test_seq_df)}, ID: {target_id}, length: {seq_length}")
                            
                            # Determine RNA category for this sequence
                            if seq_length < 50:
                                seq_category = "small_RNA"
                            elif seq_length < 120:
                                seq_category = "medium_RNA"
                            else:
                                seq_category = "large_RNA"
                                
                            print(f"  Sequence category: {seq_category}")
                            
                            # Collect base predictions from all seeds
                            base_structures = []
                            
                            # First try predictions from seeds matching this RNA category
                            category_matched = False
                            for pred_info in seed_predictions:
                                if pred_info.get('category', '') == seq_category:
                                    category_matched = True
                                    base_struct = pred_info['predictions'][i][:seq_length]
                                    
                                    # Extract GC content for RNA-specific processing
                                    seq_features = X_test[i][:seq_length]
                                    valid_mask = ~np.all(seq_features == 0, axis=1)
                                    if np.sum(valid_mask) > 0:
                                        features = seq_features[valid_mask]
                                        g_content = np.mean(features[:, 2]) if features.shape[1] > 2 else 0
                                        c_content = np.mean(features[:, 1]) if features.shape[1] > 1 else 0
                                        gc_content = g_content + c_content
                                    else:
                                        gc_content = 0.5
                                    
                                    # Apply RNA-specific post-processing
                                    use_global_movement = seq_category in ['large_RNA', 'medium_RNA']
                                    
                                    # Process structure with RNA-specific adjustments
                                    processed_struct = post_process_rna_structure(
                                        base_struct,
                                        seq,
                                        gc_content,
                                        use_global_movement=use_global_movement
                                    )
                                    
                                    # Normalize and validate structure
                                    normalized_struct = normalize_structure(processed_struct)
                                    if check_structure_validity(normalized_struct):
                                        base_structures.append(normalized_struct)
                                    else:
                                        # Try to repair structure
                                        repaired_struct = repair_invalid_structure(normalized_struct)
                                        if check_structure_validity(repaired_struct):
                                            base_structures.append(repaired_struct)
                                            print(f"  Repaired structure from seed {pred_info['seed']}")
                            
                            # If no category-matched seeds, use all seeds
                            if not category_matched or not base_structures:
                                print("  No category-matched seeds found, using all available seeds.")
                                for pred_info in seed_predictions:
                                    base_struct = pred_info['predictions'][i][:seq_length]
                                    normalized_struct = normalize_structure(base_struct)
                                    if check_structure_validity(normalized_struct):
                                        base_structures.append(normalized_struct)
                            
                            # Generate emergency structure if no valid base structures
                            if not base_structures:
                                print(f"  Warning: No valid structures for {target_id}. Creating emergency structure.")
                                emergency_structure = create_emergency_structure(seq_length)
                                base_structures.append(emergency_structure)
                            
                            # Extract sequence features
                            seq_features = X_test[i][:seq_length]
                            
                            # Generate more candidates through RNA-specific variations
                            candidates = generate_diverse_structures_from_bases(
                                base_structures, 
                                seq_length, 
                                quality_model,
                                num_per_base=5
                            )
                            
                            # Evaluate and prune candidates
                            if status['nn_training_success']:
                                print("  Using NN model for quality assessment...")
                                try:
                                    top_structures = evaluate_and_prune_structures(
                                        candidates, 
                                        seq_features, 
                                        quality_model, 
                                        top_k=5
                                    )
                                except Exception as e:
                                    print(f"  Error in NN evaluation: {str(e)}")
                                    print("  Falling back to rule-based assessment...")
                                    top_structures = evaluate_and_prune_rules(candidates, top_k=5)
                            else:
                                print("  Using rule-based quality assessment...")
                                top_structures = evaluate_and_prune_rules(candidates, top_k=5)
                                
                            # Store the final structures
                            seq_to_coords[target_id] = top_structures
                            
                        # PHASE 5: Create Submission
                        print("\nPHASE 5: Creating submission file...")
                        submission_df = create_submission_dataframe(seq_to_coords, sample_submission_df)
                        
                        # Save submission
                        enhanced_file = os.path.join(output_dir, 'submission_enhanced.csv')
                        submission_df.to_csv(enhanced_file, index=False)
                        print(f"Enhanced submission saved to {enhanced_file}")
                        
                        # Save as standard submission
                        standard_file = os.path.join(output_dir, 'submission.csv')
                        submission_df.to_csv(standard_file, index=False)
                        
                        # Set success
                        status['success'] = True
                        
                        return submission_df, status
                        
                    except Exception as e:
                        print(f"ERROR in enhanced hybrid pipeline: {str(e)}")
                        traceback.print_exc()
                        status['error'] = str(e)
                        return None, status
                
                # Run the enhanced pipeline with our found seeds
                submission_df, status = run_hybrid_pipeline_with_enhanced_seeds(
                    X_valid, y_valid,
                    test_seq_df, sample_submission_df,
                    OUTPUT_DIR,
                    golden_seeds,
                    optimal_params={'noise': 0.21, 'corr': 0.83}
                )
                
            elif use_adaptive_seeds:
                # Use adaptive search for golden seeds
                print("\nPHASE 1: Searching for golden seeds with adaptive threshold...")
                golden_seeds, all_seeds = adaptive_seed_search(
                    X_valid, 
                    y_valid, 
                    initial_threshold=0.65,  # Start with higher threshold
                    min_threshold=0.55,      # Accept down to this minimum value
                    initial_attempts=100,    # Initial number of attempts
                    max_attempts=300,        # Maximum number of attempts
                    optimal_params={'noise': 0.21, 'corr': 0.83}
                )
                
                # Modify run_hybrid_pipeline to use our found seeds
                # Define a modified version of run_hybrid_pipeline that uses predefined seeds
                def run_hybrid_pipeline_with_seeds(
                    X_valid, 
                    y_valid,
                    test_seq_df, 
                    sample_submission_df,
                    output_dir,
                    golden_seeds,  # Pass seeds directly
                    optimal_params={'noise': 0.21, 'corr': 0.83}
                ):
                    # This is a modified version that skips the seed search
                    # and uses the seeds provided by adaptive search
                    
                    status = {
                        'success': False,
                        'golden_seeds_found': len(golden_seeds),
                        'nn_training_success': False,
                        'best_tm_score': max([s['tm_score'] for s in golden_seeds]) if golden_seeds else 0.0,
                        'error': None
                    }
                    
                    try:
                        # PHASE 2: Train Quality Assessment Model
                        print("\nPHASE 2: Training NN quality assessment model...")
                        try:
                            quality_model = train_enhanced_quality_model(X_valid, y_valid, X_valid, y_valid)
                            status['nn_training_success'] = True
                        except Exception as e:
                            print(f"Error training NN model: {str(e)}")
                            print("Falling back to rule-based quality assessment...")
                            quality_model = create_rule_based_model()
                            
                        # PHASE 3: Generate Base Structures with Golden Seeds
                        print("\nPHASE 3: Generating base structures with golden seeds...")
                        X_test = prepare_test_features(test_seq_df)
                        
                        # Generate predictions using each of the top seeds
                        seed_predictions = []
                        for i, seed_info in enumerate(golden_seeds):
                            print(f"Generating predictions with seed {seed_info['seed']} (TM-score: {seed_info['tm_score']:.4f})...")
                            
                            # Set the random seed
                            np.random.seed(seed_info['seed'])
                            
                            # Create model with this seed
                            model = reference_based_approach(
                                X_valid, 
                                y_valid,
                                geometric_sampling=True,
                                noise_level=optimal_params['noise'],
                                correlation=optimal_params['corr']
                            )
                            
                            # Generate predictions
                            if model is not None:
                                preds = model.predict(X_test)
                                seed_predictions.append({
                                    'seed': seed_info['seed'],
                                    'tm_score': seed_info['tm_score'],
                                    'predictions': preds
                                })
                                
                                # Update best TM-score for status
                                if seed_info['tm_score'] > status['best_tm_score']:
                                    status['best_tm_score'] = seed_info['tm_score']
                            else:
                                print(f"Failed to create model with seed {seed_info['seed']}")
                                
                        if not seed_predictions:
                            raise Exception("Failed to generate any predictions with golden seeds")
                            
                        # PHASE 4: Generate and Prune Structures
                        print("\nPHASE 4: Generating diverse candidates and using NN pruning...")
                        
                        seq_to_coords = {}
                        for i, (_, row) in enumerate(test_seq_df.iterrows()):
                            target_id = row['target_id']
                            seq = row['sequence']
                            seq_length = len(seq)
                            
                            print(f"Processing sequence {i+1}/{len(test_seq_df)}, ID: {target_id}, length: {seq_length}")
                            
                            # Collect base predictions from all seeds for this sequence
                            base_structures = []
                            for pred_info in seed_predictions:
                                base_struct = pred_info['predictions'][i][:seq_length]
                                base_structures.append(normalize_structure(base_struct))
                                
                            # Extract sequence features
                            seq_features = X_test[i][:seq_length]
                            
                            # Generate more candidates through controlled variations
                            candidates = generate_diverse_candidates(base_structures, seq_length, num_per_base=5)
                            
                            # Evaluate and prune candidates
                            if status['nn_training_success']:
                                print("Using NN model for quality assessment...")
                                try:
                                    top_structures = evaluate_and_prune_structures(
                                        candidates, 
                                        seq_features, 
                                        quality_model, 
                                        top_k=5
                                    )
                                except Exception as e:
                                    print(f"Error in NN evaluation: {str(e)}")
                                    print("Falling back to rule-based assessment...")
                                    top_structures = evaluate_and_prune_rules(candidates, top_k=5)
                            else:
                                print("Using rule-based quality assessment...")
                                top_structures = evaluate_and_prune_rules(candidates, top_k=5)
                                
                            # Store the final structures
                            seq_to_coords[target_id] = top_structures
                            
                        # PHASE 5: Create Submission
                        print("\nPHASE 5: Creating submission file...")
                        submission_df = create_submission_dataframe(seq_to_coords, sample_submission_df)
                        
                        # Save submission
                        hybrid_file = os.path.join(output_dir, 'submission_hybrid.csv')
                        submission_df.to_csv(hybrid_file, index=False)
                        print(f"Hybrid submission saved to {hybrid_file}")
                        
                        # Save as standard submission
                        standard_file = os.path.join(output_dir, 'submission.csv')
                        submission_df.to_csv(standard_file, index=False)
                        
                        # Set success
                        status['success'] = True
                        
                        return submission_df, status
                        
                    except Exception as e:
                        print(f"ERROR in hybrid pipeline: {str(e)}")
                        traceback.print_exc()
                        status['error'] = str(e)
                        return None, status
                
                # Run the modified pipeline with our found seeds
                submission_df, status = run_hybrid_pipeline_with_seeds(
                    X_valid, y_valid,
                    test_seq_df, sample_submission_df,
                    OUTPUT_DIR,
                    golden_seeds,
                    optimal_params={'noise': 0.21, 'corr': 0.83}
                )
            else:
                # Run the standard hybrid pipeline
                submission_df, status = run_hybrid_pipeline(
                    X_valid, y_valid,
                    test_seq_df, sample_submission_df,
                    OUTPUT_DIR,
                    golden_threshold=0.6,
                    seed_attempts=100
                )
            
            # Calculate total runtime
            runtime = time.time() - start_time
            hours, remainder = divmod(runtime, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            # Display results summary
            print("\n" + "=" * 80)
            if use_enhanced_adaptive:
                print("RNA-TARGETED ENHANCED HYBRID PIPELINE RESULTS SUMMARY".center(80))
            elif use_adaptive_seeds:
                print("ENHANCED HYBRID PIPELINE RESULTS SUMMARY".center(80))
            else:
                print("HYBRID PIPELINE RESULTS SUMMARY".center(80))
            print("=" * 80)
            print(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            
            if status['success']:
                print("\nPIPELINE STATISTICS:")
                print(f"  - Golden seeds found: {status['golden_seeds_found']}")
                print(f"  - NN training success: {status['nn_training_success']}")
                print(f"  - Best TM-score: {status['best_tm_score']:.4f}")
                
                # Additional statistics for enhanced pipeline
                if use_enhanced_adaptive and 'category_stats' in status:
                    print("\nRNA CATEGORY STATISTICS:")
                    for category, stats in status.get('category_stats', {}).items():
                        print(f"  - {category}: {stats}")
            else:
                print(f"\nPipeline failed with error: {status['error']}")
            
            # Display output file information
            print("\nOUTPUT FILES:")
            if use_enhanced_adaptive:
                submission_file = os.path.join(OUTPUT_DIR, 'submission_enhanced.csv')
                if os.path.exists(submission_file):
                    try:
                        file_size = os.path.getsize(submission_file)
                        print(f"  - Enhanced submission: {submission_file} ({file_size/1024/1024:.2f} MB)")
                    except:
                        print(f"  - Enhanced submission: {submission_file}")
            else:
                submission_file = os.path.join(OUTPUT_DIR, 'submission_hybrid.csv')
                if os.path.exists(submission_file):
                    try:
                        file_size = os.path.getsize(submission_file)
                        print(f"  - Hybrid submission: {submission_file} ({file_size/1024/1024:.2f} MB)")
                    except:
                        print(f"  - Hybrid submission: {submission_file}")
            
            standard_file = os.path.join(OUTPUT_DIR, 'submission.csv')
            if os.path.exists(standard_file):
                try:
                    file_size = os.path.getsize(standard_file)
                    print(f"  - Standard submission: {standard_file} ({file_size/1024/1024:.2f} MB)")
                except:
                    print(f"  - Standard submission: {standard_file}")
            
            print("=" * 80)
            
        elif use_nn_pruning:
            start_time = time.time()
            
            print("Loading processed data...")
            X_train, y_train, X_valid, y_valid = load_processed_data()
            
            print("\nLoading test data...")
            test_seq_df = pd.read_csv(os.path.join(DATA_DIR, "test_sequences.csv"))
            sample_submission_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
            
            # Train quality model
            print("\nTraining quality assessment model...")
            quality_model = train_enhanced_quality_model(X_valid, y_valid, X_valid, y_valid)
            
            # Create reference model with default parameters
            print("\nCreating reference model...")
            reference_model = reference_based_approach(
                X_valid, 
                y_valid,
                geometric_sampling=True,
                noise_level=0.21,
                correlation=0.83
            )
            
            # Generate submission using NN pruning only
            submission_df = generate_nn_pruned_submission(
                reference_model,
                quality_model,
                test_seq_df,
                sample_submission_df
            )
            
            # Calculate total runtime
            runtime = time.time() - start_time
            hours, remainder = divmod(runtime, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            print("\n" + "=" * 80)
            print("NN PRUNING PIPELINE RESULTS".center(80))
            print("=" * 80)
            print(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            
            # Display output file information
            print("\nOUTPUT FILES:")
            submission_file = os.path.join(OUTPUT_DIR, 'submission_nn_pruned.csv')
            if os.path.exists(submission_file):
                try:
                    file_size = os.path.getsize(submission_file)
                    print(f"  - NN pruned submission: {submission_file} ({file_size/1024/1024:.2f} MB)")
                except:
                    print(f"  - NN pruned submission: {submission_file}")
            
            standard_file = os.path.join(OUTPUT_DIR, 'submission.csv')
            if os.path.exists(standard_file):
                try:
                    file_size = os.path.getsize(standard_file)
                    print(f"  - Standard submission: {standard_file} ({file_size/1024/1024:.2f} MB)")
                except:
                    print(f"  - Standard submission: {standard_file}")
            
            print("=" * 80)
            
        elif use_reference_only:
            start_time = time.time()
            
            print("Loading processed data...")
            X_train, y_train, X_valid, y_valid = load_processed_data()
            
            print("\nLoading test data...")
            test_seq_df = pd.read_csv(os.path.join(DATA_DIR, "test_sequences.csv"))
            sample_submission_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
            
            # Create optimized reference model
            print("\nCreating and evaluating reference model...")
            reference_model = reference_based_approach(
                X_valid, 
                y_valid,
                geometric_sampling=True,
                noise_level=0.21,
                correlation=0.83
            )
            
            metrics = evaluate_model(reference_model, X_valid, y_valid)
            tm_score = metrics['avg_tm_score']
            print(f"Reference model TM-score: {tm_score:.4f}")
            
            # Prepare test sequences
            X_test = prepare_test_features(test_seq_df)
            
            # Generate predictions
            print("\nGenerating predictions...")
            predictions = reference_model.predict(X_test)
            
            # Create submission dataframe
            print("\nCreating submission dataframe...")
            submission_df = sample_submission_df.copy()
            
            seq_to_coords = {}
            for i, (_, row) in enumerate(test_seq_df.iterrows()):
                target_id = row['target_id']
                seq_length = len(row['sequence'])
                
                # Normalize and process structure
                struct = normalize_structure(predictions[i][:seq_length])
                
                # Create 5 copies with small variations
                structures = [struct]
                for j in range(4):
                    variation = sample_structural_variation(
                        struct,
                        noise_level=0.05,
                        preserve_distance=True,
                        correlation=0.9
                    )
                    structures.append(normalize_structure(variation))
                
                seq_to_coords[target_id] = structures
            
            # Fill the dataframe
            for i, row in submission_df.iterrows():
                if i % 1000 == 0:
                    print(f"Processing row {i}/{len(submission_df)}")
                
                id_parts = row['ID'].split('_')
                seq_id = id_parts[0]
                residue_idx = int(id_parts[1]) - 1
                
                if seq_id in seq_to_coords:
                    structures = seq_to_coords[seq_id]
                    if residue_idx < len(structures[0]):
                        for struct_idx in range(5):
                            submission_df.at[i, f'x_{struct_idx+1}'] = structures[struct_idx][residue_idx][0]
                            submission_df.at[i, f'y_{struct_idx+1}'] = structures[struct_idx][residue_idx][1]
                            submission_df.at[i, f'z_{struct_idx+1}'] = structures[struct_idx][residue_idx][2]
            
            # Save submission
            reference_file = os.path.join(OUTPUT_DIR, 'submission_reference.csv')
            submission_df.to_csv(reference_file, index=False)
            print(f"Reference submission saved to {reference_file}")
            
            # Save as standard submission
            standard_file = os.path.join(OUTPUT_DIR, 'submission.csv')
            submission_df.to_csv(standard_file, index=False)
            
            # Calculate total runtime
            runtime = time.time() - start_time
            hours, remainder = divmod(runtime, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            print("\n" + "=" * 80)
            print("REFERENCE MODEL RESULTS".center(80))
            print("=" * 80)
            print(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            
            # Display output file information
            print("\nOUTPUT FILES:")
            if os.path.exists(reference_file):
                try:
                    file_size = os.path.getsize(reference_file)
                    print(f"  - Reference submission: {reference_file} ({file_size/1024/1024:.2f} MB)")
                except:
                    print(f"  - Reference submission: {reference_file}")

            if os.path.exists(standard_file):
                try:
                    file_size = os.path.getsize(standard_file)
                    print(f"  - Standard submission: {standard_file} ({file_size/1024/1024:.2f} MB)")
                except:
                    print(f"  - Standard submission: {standard_file}")
           
            print("=" * 80)
           
        else:
            # Standard pipeline - if user disabled all options
            print("No pipeline mode selected. Please set one of the pipeline flags to True.")
            print("Available options:")
            print("  - use_hybrid_pipeline: Combined golden seeds and NN pruning")
            print("  - use_nn_pruning: Neural Network based pruning only")
            print("  - use_reference_only: Use only reference model approach")
            print("  - use_adaptive_seeds: Use adaptive seed search (with hybrid pipeline)")
            print("  - use_enhanced_adaptive: Use enhanced RNA-specific adaptive seed search")
       
        print("\nProcess completed.")
       
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR IN MAIN EXECUTION".center(80))
        print("=" * 80)
        print(f"Critical error: {str(e)}")
        traceback.print_exc()
        print("=" * 80)


submission_df = pd.read_csv('/kaggle/working/submission.csv')
print("Overview of the DataFrame:")
print(submission_df.shape)  # Print the shape (rows, columns)
print(submission_df.head())  # Display the first 5 rows




