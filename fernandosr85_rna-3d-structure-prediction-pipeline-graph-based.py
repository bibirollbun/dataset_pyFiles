# Standard Library Imports
import datetime
import gc
import hashlib
import json
import os
import random
import time
import traceback
import warnings
from collections import Counter

# Scientific Computing and Numerical Libraries
import numpy as np
import pandas as pd

# Graph Processing
import networkx as nx

# Machine Learning and Deep Learning
import torch
import torch.nn as nn

# Visualization Libraries
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# Machine Learning Library Import with Error Handling
try:
    # TensorFlow and Keras
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import (
        BatchNormalization, Bidirectional, Conv1D, 
        Dense, Dropout, Flatten, Input, LSTM, Reshape
    )
    from tensorflow.keras.models import Model
    
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
    if "train_labels.csv" in main_data and "validation_labels.csv" in main_data:
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

class StructureWrapper:
    """
    Wrapper for RNA structure arrays that provides both quality attributes
    and compatibility with NumPy array operations.
    """
    def __init__(self, structure, quality_score=0.5):
        self.structure = structure
        self.quality = {'quality_score': quality_score}
        # Store shape from the underlying structure for numpy compatibility
        self.shape = structure.shape if hasattr(structure, 'shape') else None
        
    def __getitem__(self, idx):
        return self.structure[idx]
        
    def __len__(self):
        return len(self.structure)
    
    # Implement arithmetic operators for numpy compatibility
    def __sub__(self, other):
        """Implement subtraction between structures"""
        if isinstance(other, StructureWrapper):
            # Subtract the underlying structures
            return StructureWrapper(self.structure - other.structure)
        else:
            # Subtract a scalar or numpy array directly
            return StructureWrapper(self.structure - other)
    
    def __add__(self, other):
        """Implement addition between structures"""
        if isinstance(other, StructureWrapper):
            return StructureWrapper(self.structure + other.structure)
        else:
            return StructureWrapper(self.structure + other)
    
    def __mul__(self, other):
        """Implement multiplication between structures"""
        if isinstance(other, StructureWrapper):
            return StructureWrapper(self.structure * other.structure)
        else:
            return StructureWrapper(self.structure * other)
    
    def __truediv__(self, other):
        """Implement division between structures"""
        if isinstance(other, StructureWrapper):
            return StructureWrapper(self.structure / other.structure)
        else:
            return StructureWrapper(self.structure / other)
    
    def __neg__(self):
        """Implement negation"""
        return StructureWrapper(-self.structure)
    
    def __abs__(self):
        """Implement absolute value"""
        return StructureWrapper(abs(self.structure))
    
    # Implement reverse operations (for scalar op structure)
    def __radd__(self, other):
        return StructureWrapper(other + self.structure)
    
    def __rsub__(self, other):
        return StructureWrapper(other - self.structure)
    
    def __rmul__(self, other):
        return StructureWrapper(other * self.structure)
    
    def __rtruediv__(self, other):
        return StructureWrapper(other / self.structure)
    
    # Implement comparison operators
    def __eq__(self, other):
        if isinstance(other, StructureWrapper):
            return self.structure == other.structure
        else:
            return self.structure == other
    
    def __lt__(self, other):
        if isinstance(other, StructureWrapper):
            return self.structure < other.structure
        else:
            return self.structure < other
            
    def __gt__(self, other):
        if isinstance(other, StructureWrapper):
            return self.structure > other.structure
        else:
            return self.structure > other
            
    def __le__(self, other):
        if isinstance(other, StructureWrapper):
            return self.structure <= other.structure
        else:
            return self.structure <= other
            
    def __ge__(self, other):
        if isinstance(other, StructureWrapper):
            return self.structure >= other.structure
        else:
            return self.structure >= other
    
    # Implement numpy compatibility methods
    def __array__(self):
        """Allow numpy to automatically convert to array when needed"""
        import numpy as np
        return np.array(self.structure)
    
    def sum(self, *args, **kwargs):
        """Implement sum method for numpy compatibility"""
        return self.structure.sum(*args, **kwargs)
    
    def mean(self, *args, **kwargs):
        """Implement mean method for numpy compatibility"""
        return self.structure.mean(*args, **kwargs)
    
    def max(self, *args, **kwargs):
        """Implement max method for numpy compatibility"""
        return self.structure.max(*args, **kwargs)
    
    def min(self, *args, **kwargs):
        """Implement min method for numpy compatibility"""
        return self.structure.min(*args, **kwargs)
    
    def reshape(self, *args, **kwargs):
        """Implement reshape method for numpy compatibility"""
        reshaped = self.structure.reshape(*args, **kwargs)
        return StructureWrapper(reshaped, self.quality.get('quality_score', 0.5))
    
    def transpose(self, *args, **kwargs):
        """Implement transpose method for numpy compatibility"""
        transposed = self.structure.transpose(*args, **kwargs)
        return StructureWrapper(transposed, self.quality.get('quality_score', 0.5))
    
    # String representation
    def __repr__(self):
        return f"StructureWrapper(shape={self.shape}, quality_score={self.quality.get('quality_score', 0.5)})"

class ParameterOptimizer:
    """
    Meta-learning system for continuous parameter optimization
    based on historical results.
    """
    
    def __init__(self, history_file=None):
        """
        Initializes the optimizer, optionally loading previous history.
        
        Parameters:
        -----------
        history_file: str, optional
            Path to a file containing the parameter and result history
        """
        self.history = []
        if history_file and os.path.exists(history_file):
            self.load_history(history_file)
            
        # Parameter bounds
        self.param_bounds = {
            'divisor_mean': (3.0, 4.5),
            'divisor_std': (0.5, 1.5),
            'noise_base': (0.01, 0.3),
            'correlation': (0.7, 0.95)
        }
    
    def load_history(self, filename):
        """Loads previous parameter and result history"""
        try:
            with open(filename, 'r') as f:
                self.history = json.load(f)
        except Exception as e:
            print(f"Error loading history: {str(e)}")
    
    def save_history(self, filename):
        """Saves the current history to a file"""
        with open(filename, 'w') as f:
            json.dump(self.history, f)
    
    def record_result(self, params, size_category, mode, quality_score):
        """
        Records a new result into the history
        
        Parameters:
        -----------
        params: dict
            Parameters used
        size_category: str
            Size category ('small', 'medium', 'large')
        mode: str
            Mode used ('adaptive' or 'fixed')
        quality_score: float
            Quality score obtained
        """
        self.history.append({
            'params': params,
            'size_category': size_category,
            'mode': mode,
            'quality_score': quality_score,
            'timestamp': datetime.datetime.now().isoformat()
        })
    
    def suggest_parameters(self, size_category, mode):
        """
        Suggests optimized parameters based on the history
        for a given size category and mode
        
        Parameters:
        -----------
        size_category: str
            Size category ('small', 'medium', 'large')
        mode: str
            Operation mode ('adaptive' or 'fixed')
            
        Returns:
        --------
        dict: Suggested parameters
        """
        # Filter history for the specified category and mode
        relevant_history = [
            entry for entry in self.history 
            if entry['size_category'] == size_category and entry['mode'] == mode
        ]
        
        if len(relevant_history) < 5:
            # Not enough history, use default values
            return self._get_default_params(size_category, mode)
        
        # Sort by quality score, from best to worst
        relevant_history.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # Extract parameters from the top N results
        top_n = min(5, len(relevant_history))
        top_params = [entry['params'] for entry in relevant_history[:top_n]]
        
        # Compute weighted average of parameters
        weights = [0.4, 0.25, 0.15, 0.1, 0.1][:top_n]  # Weights for top N results
        
        suggested_params = {}
        for param in self.param_bounds.keys():
            if all(param in p for p in top_params):
                weighted_sum = sum(w * p[param] for w, p in zip(weights, top_params))
                suggested_params[param] = weighted_sum / sum(weights[:top_n])
        
        # Ensure suggested parameters are within bounds
        for param, (min_val, max_val) in self.param_bounds.items():
            if param in suggested_params:
                suggested_params[param] = max(min_val, min(suggested_params[param], max_val))
        
        return suggested_params
    
    def _get_default_params(self, size_category, mode):
        """Returns default parameters when historical data is insufficient"""
        # Default values for different size categories and modes
        defaults = {
            'small': {
                'adaptive': {'divisor_mean': 3.6, 'divisor_std': 0.9, 'noise_base': 0.15},
                'fixed': {'noise_base': 0.12, 'correlation': 0.85}
            },
            'medium': {
                'adaptive': {'divisor_mean': 3.8, 'divisor_std': 1.0, 'noise_base': 0.1},
                'fixed': {'noise_base': 0.08, 'correlation': 0.85}
            },
            'large': {
                'adaptive': {'divisor_mean': 4.0, 'divisor_std': 1.1, 'noise_base': 0.05},
                'fixed': {'noise_base': 0.04, 'correlation': 0.9}
            }
        }
        
        return defaults.get(size_category, {}).get(mode, {})

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

def refine_with_distance_geometry(initial_coords, target_distances, weights, max_iterations=200):
    """
    Optimizes 3D coordinates to better satisfy distance constraints.

    Parameters:
    -----------
    initial_coords: Initial coordinates (array of shape (n, 3))
    target_distances: Target distance matrix (array of shape (n, n)) 
    weights: Matrix of weights for each constraint (array of shape (n, n))
    max_iterations: Maximum number of iterations

    Returns:
    --------
    Refined coordinates (array of shape (n, 3))
    """
    coords = initial_coords.copy()
    n = coords.shape[0]
    learning_rate = 0.01

    for iteration in range(max_iterations):
        # Calculate current distance matrix
        current_distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(coords[i] - coords[j])
                current_distances[i, j] = dist
                current_distances[j, i] = dist

        # Calculate gradients  
        grad = np.zeros_like(coords)
        for i in range(n):
            for j in range(i+1, n):
                if weights[i, j] > 0:
                    # Vector from i to j
                    direction = coords[j] - coords[i]
                    current_dist = np.linalg.norm(direction)
                    
                    # Avoid division by zero
                    if current_dist < 1e-10:
                        continue

                    direction = direction / current_dist
                    
                    # Difference between current and target distance
                    diff = current_distances[i, j] - target_distances[i, j]
                    
                    # Update gradients
                    grad_ij = weights[i, j] * diff * direction
                    grad[i] += grad_ij
                    grad[j] -= grad_ij
        
        # Update coordinates
        coords = coords - learning_rate * grad
        
        # Gradually reduce learning rate
        learning_rate *= 0.995

    return coords


def reference_based_approach(X_valid, y_valid, geometric_sampling=True, noise_level=0.21, correlation=0.83):
    """
    Placeholder for reference-based RNA structure prediction approach.
    
    Parameters:
    -----------
    X_valid : array-like
        Validation input features
    y_valid : array-like
        Validation target structures
    geometric_sampling : bool, optional
        Whether to use geometric sampling (default True)
    noise_level : float, optional
        Level of noise to add to the structure (default 0.21)
    correlation : float, optional
        Correlation parameter for structure generation (default 0.83)
    
    Returns:
    --------
    model : object
        A placeholder model object with a predict method
    """
    class ReferenceModel:
        def __init__(self, noise_level, correlation):
            self.noise_level = noise_level
            self.correlation = correlation
        
        def predict(self, X):
            """
            Generate placeholder predictions based on input features.
            
            Parameters:
            -----------
            X : array-like
                Input features for prediction
            
            Returns:
            --------
            predictions : numpy.ndarray
                Generated 3D coordinates
            """
            # Create placeholder predictions with some randomness
            predictions = []
            for seq_features in X:
                # Generate a simple 3D structure 
                # Assume sequence length based on input features
                seq_length = seq_features.shape[0]
                
                # Create a simple linear structure with some noise
                structure = np.zeros((seq_length, 3))
                for i in range(1, seq_length):
                    # Simple linear progression with small random variations
                    structure[i] = structure[i-1] + np.array([3.8, 0, 0]) + \
                                   np.random.normal(0, self.noise_level, 3)
                
                predictions.append(structure)
            
            return np.array(predictions)
    
    # Create and return the reference model
    return ReferenceModel(noise_level, correlation)

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

# Auxiliary function to calculate the dihedral angle (in degrees)
def calculate_dihedral(p0, p1, p2, p3):
    """
    Calculates the dihedral angle (in degrees) defined by the points p0, p1, p2, and p3.
    """
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    # Normalize b1 so its length does not influence the calculation
    b1 /= np.linalg.norm(b1) + 1e-8

    # Normal vectors to the planes formed by (p0,p1,p2) and (p1,p2,p3)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    angle = np.degrees(np.arctan2(y, x))
    return angle

# Auxiliary function to generate a rotation matrix
def get_rotation_matrix(axis, theta):
    """
    Returns the 3x3 rotation matrix for a rotation of theta radians around the given 'axis'.
    """
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([
        [aa+bb-cc-dd, 2*(bc+ad),   2*(bd-ac)],
        [2*(bc-ad),   aa+cc-bb-dd, 2*(cd+ab)],
        [2*(bd+ac),   2*(cd-ab),   aa+dd-bb-cc]
    ])

# New function to refine the RNA backbone with dihedral angle adjustment
def refine_rna_backbone_with_dihedrals(structure, ideal_dihedral=180.0):
    """
    Refines the geometry of the RNA backbone by adjusting the dihedral angles to an ideal value.

    Parameters:
      structure: np.array of shape (seq_length, 3) containing the coordinates.
      ideal_dihedral: Ideal angle (in degrees) for the backbone segments (e.g., 180Â°).

    Returns:
      np.array with the refined structure.
    """
    refined = structure.copy()
    n = len(refined)
    if n < 4:
        return refined  # There are no dihedral angles to correct

    for i in range(n - 3):
        p0, p1, p2, p3 = refined[i], refined[i+1], refined[i+2], refined[i+3]
        current_angle = calculate_dihedral(p0, p1, p2, p3)
        # Compute the difference (in radians) between the ideal angle and the current one
        angle_diff = np.radians(ideal_dihedral - current_angle)
        
        # Define the rotation axis as the direction of the segment (p3 - p2)
        axis = p3 - p2
        norm_axis = np.linalg.norm(axis)
        if norm_axis < 1e-6:
            continue
        axis /= norm_axis
        
        # Get the rotation matrix for the correction angle
        R = get_rotation_matrix(axis, angle_diff)
        
        # Apply the rotation to all points starting from p3
        for j in range(i+3, n):
            vec = refined[j] - p2
            refined[j] = p2 + np.dot(vec, R.T)
    
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


class GraphAttentionLayer(nn.Module):
    """
    Graph attention layer for processing node and edge features
    """
    def __init__(self, in_features, out_features, heads=8, dropout=0.1):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.dropout = dropout
        
        # Linear transformations for query, key and value
        self.query = nn.Linear(in_features, out_features * heads)
        self.key = nn.Linear(in_features, out_features * heads)
        self.value = nn.Linear(in_features, out_features * heads)
        
        # Transformation for edge type
        self.edge_attn = nn.Linear(in_features, heads)
        
        # Dropout and output layer
        self.dropout_layer = nn.Dropout(dropout)
        self.output_transform = nn.Linear(out_features * heads, out_features)
    
    def forward(self, x, edge_index, edge_attr=None):
        # Shape: x = [num_nodes, in_features]
        # edge_index = [2, num_edges]
        # edge_attr = [num_edges, edge_features]
        
        # Calculate queries, keys and values
        queries = self.query(x).view(-1, self.heads, self.out_features)
        keys = self.key(x).view(-1, self.heads, self.out_features)
        values = self.value(x).view(-1, self.heads, self.out_features)
        
        # Extract source and destination nodes for each edge
        src, dst = edge_index
        
        # Calculate attention scores
        q_i = queries[dst]
        k_j = keys[src]
        
        # Dot product attention
        alpha = torch.sum(q_i * k_j, dim=-1) / np.sqrt(self.out_features)
        
        # Add attention based on edge type, if available
        if edge_attr is not None:
            edge_attention = self.edge_attn(edge_attr).view(-1, self.heads)
            alpha = alpha + edge_attention
        
        # Softmax to normalize attention between neighbors
        alpha = torch.softmax(alpha, dim=0)
        alpha = self.dropout_layer(alpha)
        
        # Apply attention to values
        v_j = values[src].view(-1, self.heads, self.out_features)
        weighted_values = v_j * alpha.unsqueeze(-1)
        
        # Aggregate weighted values
        output = torch.zeros_like(queries)
        for i in range(dst.max() + 1):
            mask = (dst == i)
            if mask.any():
                output[i] = weighted_values[mask].sum(dim=0)
        
        # Concatenate/transform attention heads for final dimensionality
        output = output.reshape(-1, self.heads * self.out_features)
        output = self.output_transform(output)
        
        return output

class RNAGraphTransformer(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim=128, n_layers=6):
        super(RNAGraphTransformer, self).__init__()
        
        # Embedding layers
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.edge_embedding = nn.Linear(edge_features, hidden_dim)
        
        # Graph attention layers
        self.graph_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, heads=8, dropout=0.1)
            for _ in range(n_layers)
        ])
        
        # MLPs for processing by interaction type
        self.covalent_mlp = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.basepair_mlp = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.tertiary_mlp = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output layer for 3D coordinates
        self.coords_output = nn.Linear(hidden_dim, 3)
        
        # Output layer for distances between residues
        self.distance_output = nn.Linear(hidden_dim*2, 1)
        
    def forward(self, G):
        # Get node and edge features
        node_features = torch.stack([G.nodes[i]['features'] for i in G.nodes])
        edge_index, edge_features = self._get_edge_data(G)
        
        # Initial embedding
        h_nodes = self.node_embedding(node_features)
        
        # Pass through graph layers
        for layer in self.graph_layers:
            h_nodes = layer(h_nodes, edge_index, edge_features)
        
        # Predict coordinates for each node
        coords = self.coords_output(h_nodes)
        
        # Predict distances between connected pairs
        distances = {}
        for (i, j) in G.edges():
            # Concatenate features of the two nodes
            edge_repr = torch.cat([h_nodes[i], h_nodes[j]], dim=0)
            
            # Process with specific MLP based on edge type
            edge_type = G[i][j]['type']
            if edge_type == 'covalent':
                edge_repr = self.covalent_mlp(edge_repr)
            elif edge_type == 'basepair':
                edge_repr = self.basepair_mlp(edge_repr)
            elif edge_type == 'tertiary':
                edge_repr = self.tertiary_mlp(edge_repr)
            
            # Predict distance
            distances[(i, j)] = self.distance_output(edge_repr)
        
        return coords, distances
    
    def _get_edge_data(self, G):
        # Convert edge data to the format expected by PyTorch Geometric
        edge_indices = []
        edge_features_list = []
        
        for i, j in G.edges():
            edge_indices.append([i, j])
            
            # Convert edge attributes to feature vector
            edge_attr = G[i][j]
            edge_type = edge_attr['type']
            
            # One-hot encoding of edge type
            edge_type_vector = [0, 0, 0, 0]  # [covalent, basepair, stacking, tertiary]
            if edge_type == 'covalent':
                edge_type_vector[0] = 1
            elif edge_type == 'basepair':
                edge_type_vector[1] = 1
            elif edge_type == 'stacking':
                edge_type_vector[2] = 1
            elif edge_type == 'tertiary':
                edge_type_vector[3] = 1
            
            # Add edge weight
            weight = edge_attr.get('weight', 1.0)
            
            # Combine into a single feature vector
            combined_features = edge_type_vector + [weight]
            edge_features_list.append(combined_features)
        
        # Convert to PyTorch tensors
        edge_index = torch.tensor(edge_indices).t().contiguous()  # Transpose to shape [2, num_edges]
        edge_features = torch.tensor(edge_features_list, dtype=torch.float)
        
        return edge_index, edge_features

# ==== 2. FUNCTIONS FOR RNA GRAPH PROCESSING ====

def get_base_features(base):
    """
    Converts a nucleotide into a feature vector.
    """
    # One-hot encoding for base type
    base_encoding = {
        'A': [1, 0, 0, 0, 0],  # Adenine
        'C': [0, 1, 0, 0, 0],  # Cytosine
        'G': [0, 0, 1, 0, 0],  # Guanine
        'U': [0, 0, 0, 1, 0],  # Uracil
        'T': [0, 0, 0, 1, 0],  # Treats T as U
        'N': [0, 0, 0, 0, 1]   # Unknown base
    }
    
    # Basic base features
    encoding = base_encoding.get(base, [0, 0, 0, 0, 1])  # Default to N if not recognized
    
    # Additional properties
    is_purine = 1.0 if base in ['A', 'G'] else 0.0
    is_pyrimidine = 1.0 if base in ['C', 'U', 'T'] else 0.0
    
    # Pairing features
    can_pair_with_A = 1.0 if base in ['U', 'T'] else 0.0
    can_pair_with_C = 1.0 if base in ['G'] else 0.0
    can_pair_with_G = 1.0 if base in ['C', 'U', 'T'] else 0.0  # G-U wobble
    can_pair_with_U = 1.0 if base in ['A', 'G'] else 0.0  # G-U wobble
    
    # Combine all features
    features = encoding + [is_purine, is_pyrimidine, 
                         can_pair_with_A, can_pair_with_C, 
                         can_pair_with_G, can_pair_with_U]
    
    return torch.tensor(features, dtype=torch.float)

def predict_pair_type(base1, base2):
    """
    Determines the type of base pair between two nucleotides.
    """
    if (base1 == 'A' and base2 == 'U') or (base1 == 'U' and base2 == 'A'):
        return 'AU'
    elif (base1 == 'G' and base2 == 'C') or (base1 == 'C' and base2 == 'G'):
        return 'GC'
    elif (base1 == 'G' and base2 == 'U') or (base1 == 'U' and base2 == 'G'):
        return 'GU'  # wobble pair
    else:
        return 'noncanonical'

def predict_tertiary_interactions(sequence):
    """
    Predicts possible tertiary interactions based on the sequence.
    This is a simplified implementation and should be replaced by a trained model.
    """
    tertiary_interactions = []
    
    # Simple rules for potential tertiary interactions
    for i in range(len(sequence)):
        for j in range(i + 4, len(sequence)):  # At least 4 bases apart
            base_i = sequence[i]
            base_j = sequence[j]
            
            # Heuristic rules for tertiary interactions
            if (base_i == 'A' and base_j == 'G') or (base_i == 'G' and base_j == 'A'):
                interaction_type = 'A-minor'
                prob = 0.3
                tertiary_interactions.append((i, j, interaction_type, prob))
            elif (base_i == 'G' and base_j == 'G'):
                interaction_type = 'G-quadruplex'
                prob = 0.2
                tertiary_interactions.append((i, j, interaction_type, prob))
            
    return tertiary_interactions

def create_rna_graph(sequence, predicted_contacts=None):
    """
    Creates a graph representing the RNA molecule.
    """
    G = nx.Graph()
    
    # 1. Add nodes (nucleotides)
    for i, base in enumerate(sequence):
        G.add_node(i, base=base, position=None, features=get_base_features(base))
    
    # 2. Add covalent bond edges (backbone)
    for i in range(len(sequence)-1):
        G.add_edge(i, i+1, type='covalent', weight=1.0)
    
    # 3. Add base pairing edges (if predicted)
    if predicted_contacts is not None:
        for i, j, prob in predicted_contacts:
            if i < j-3:  # Avoid trivial contacts
                G.add_edge(i, j, type='basepair', weight=prob, 
                           pair_type=predict_pair_type(sequence[i], sequence[j]))
    
    # 4. Add possible stacking interactions
    for i in range(len(sequence)-1):
        G.add_edge(i, i+1, type='stacking', weight=0.8)  # Adjacent stacking
    
    # 5. Add possible tertiary interactions
    tertiary_interactions = predict_tertiary_interactions(sequence)
    for i, j, interaction_type, prob in tertiary_interactions:
        G.add_edge(i, j, type='tertiary', interaction=interaction_type, weight=prob)
    
    return G

# ==== 3. GEOMETRY AND REFINEMENT FUNCTIONS ====

def compute_distance_matrix(coords):
    """
    Calculates the distance matrix from 3D coordinates.
    """
    n = coords.shape[0]
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return dist_matrix

def distance_geometry_optimization(initial_coords, target_distances, weights, max_iterations=200):
    """
    Optimizes 3D coordinates to better satisfy distance constraints.
    """
    coords = initial_coords.copy()
    n = coords.shape[0]
    learning_rate = 0.01
    
    for iteration in range(max_iterations):
        # Calculate current distance matrix
        current_distances = compute_distance_matrix(coords)
        
        # Calculate gradients
        grad = np.zeros_like(coords)
        for i in range(n):
            for j in range(i+1, n):
                if weights[i, j] > 0:
                    # Unit vector from i to j
                    direction = coords[j] - coords[i]
                    current_dist = np.linalg.norm(direction)
                    
                    # Avoid division by zero
                    if current_dist < 1e-10:
                        continue
                    
                    direction = direction / current_dist
                    
                    # Difference between current and target distance
                    diff = current_distances[i, j] - target_distances[i, j]
                    
                    # Update gradients
                    grad_ij = weights[i, j] * diff * direction
                    grad[i] += grad_ij
                    grad[j] -= grad_ij
        
        # Update coordinates
        coords = coords - learning_rate * grad
        
        # Gradually reduce learning rate
        learning_rate *= 0.995
    
    return coords

def weighted_coordinate_average(coord_weight_pairs):
    """
    Calculates a weighted average of coordinate sets.
    """
    total_weight = sum(weight for _, weight in coord_weight_pairs)
    avg_coords = np.zeros_like(coord_weight_pairs[0][0])
    
    for coords, weight in coord_weight_pairs:
        avg_coords += (coords * weight / total_weight)
    
    return avg_coords

def load_fragment_library():
    """
    Loads library of RNA structural fragments.
    This is a simplified implementation and should be replaced by a real library.
    """
    # Create a simple example library
    fragment_library = {
        'stem': [np.random.randn(10, 3) for _ in range(5)],  # 5 examples of stems with 10 nucleotides
        'loop': [np.random.randn(5, 3) for _ in range(3)],   # 3 examples of loops with 5 nucleotides
        'bulge': [np.random.randn(3, 3) for _ in range(2)],  # 2 examples of bulges with 3 nucleotides
        'junction': [np.random.randn(8, 3) for _ in range(2)], # 2 examples of junctions with 8 nucleotides
    }
    
    print("Loaded simplified structural fragment library")
    return fragment_library

def predict_secondary_structure(sequence):
    """
    Predicts RNA secondary structure from sequence.
    This is a simplified implementation.
    """
    # Example implementation - in a real system, you would use
    # secondary structure prediction methods like ViennaRNA
    structure_elements = []
    
    # Simplification: treat everything as stem or loop
    i = 0
    while i < len(sequence):
        if i < len(sequence) - 10:
            # Check for possible stem
            stem_length = min(5, (len(sequence) - i) // 2)
            structure_elements.append({
                'id': len(structure_elements),
                'type': 'stem',
                'start': i,
                'end': i + 2*stem_length - 1,
                'sequence': sequence[i:i + 2*stem_length]
            })
            i += 2*stem_length
        else:
            # Remainder as loop
            structure_elements.append({
                'id': len(structure_elements),
                'type': 'loop',
                'start': i,
                'end': len(sequence) - 1,
                'sequence': sequence[i:]
            })
            i = len(sequence)
    
    return structure_elements

def find_best_fragment(sequence, fragment_library):
    """
    Finds the best fragment in the library for a given sequence.
    """
    # Simple implementation - just returns the first fragment
    # In a real system, you would do a comparison based on sequence/geometry
    if len(fragment_library) > 0:
        return fragment_library[0]
    return np.zeros((len(sequence), 3))  # Return zeros if library is empty

def assemble_fragments(selected_fragments, structure_elements):
    """
    Assembles an initial structure from selected fragments.
    """
    # Determine total sequence length
    max_pos = max([elem['end'] for elem in structure_elements]) + 1
    
    # Initialize structure
    assembled_structure = np.zeros((max_pos, 3))
    
    # Current position for assembly
    current_pos = np.zeros(3)
    
    # Assemble each structural element
    for elem in sorted(structure_elements, key=lambda x: x['start']):
        elem_id = elem['id']
        fragment = selected_fragments[elem_id]
        
        # Place fragment at current position
        length = elem['end'] - elem['start'] + 1
        fragment_resized = fragment
        
        # Resize or truncate fragment if necessary
        if len(fragment) != length:
            if len(fragment) > length:
                fragment_resized = fragment[:length]
            else:
                # Extend fragment by repeating last coordinate
                fragment_resized = np.vstack([fragment, np.tile(fragment[-1], (length - len(fragment), 1))])
        
        # Position in 3D space (simplified)
        fragment_centered = fragment_resized - fragment_resized[0] + current_pos
        
        # Add to assembled structure
        assembled_structure[elem['start']:elem['end']+1] = fragment_centered
        
        # Update current position
        current_pos = fragment_centered[-1] + np.array([3.8, 0, 0])  # Approximate bond distance
    
    return assembled_structure

def generate_model_ensemble(structure, sequence, num_models=5, perturbation_scale=0.2):
    """
    Generates an ensemble of structural models based on controlled perturbations.
    """
    ensemble = [structure]  # Include original model
    
    for i in range(1, num_models):
        # Apply perturbation with gradually increasing scale
        scale = perturbation_scale * i / num_models
        
        # Use existing structural variation function
        perturbed = sample_structural_variation(
            structure,
            noise_level=scale,
            preserve_distance=True,
            use_global_movement=(i % 2 == 0),
            correlation=0.9 - (i * 0.1 / num_models)
        )
        
        # Normalize and add to ensemble
        perturbed = normalize_structure(perturbed)
        ensemble.append(perturbed)
    
    return ensemble[:num_models]

# ==== 4. MAIN PREDICTION PIPELINE ====

def predict_rna_contacts(sequence):
    """
    Predicts contacts to form base pairs in the RNA molecule.
    Simplified rule-based implementation; in production, replace with trained model.
    """
    contacts = []
    
    # Simple implementation based on Watson-Crick pairing rules
    for i in range(len(sequence)):
        for j in range(i+4, len(sequence)):  # At least 4 bases separation
            base_i = sequence[i]
            base_j = sequence[j]
            
            # Check pairing compatibility
            if (base_i == 'A' and base_j == 'U') or (base_i == 'U' and base_j == 'A'):
                contacts.append((i, j, 0.95))  # High confidence
            elif (base_i == 'G' and base_j == 'C') or (base_i == 'C' and base_j == 'G'):
                contacts.append((i, j, 0.98))  # Highest confidence
            elif (base_i == 'G' and base_j == 'U') or (base_i == 'U' and base_j == 'G'):
                contacts.append((i, j, 0.85))  # Wobble pair, lower confidence
    
    # Filter redundant or mutually exclusive contacts
    filtered_contacts = []
    used_positions = set()
    
    for i, j, prob in sorted(contacts, key=lambda x: x[2], reverse=True):
        if i not in used_positions and j not in used_positions:
            filtered_contacts.append((i, j, prob))
            used_positions.add(i)
            used_positions.add(j)
    
    return filtered_contacts

def refine_with_constraints(structure, predicted_contacts, physical_constraints):
    """
    Refines the structure to satisfy contact and physical constraints.
    """
    # Simplified implementation
    refined = structure.copy()
    
    # Apply distance constraints for predicted contacts
    for i, j, prob in predicted_contacts:
        if i >= len(structure) or j >= len(structure):
            continue
            
        # Current coordinates
        pos_i = structure[i]
        pos_j = structure[j]
        
        # Current distance
        current_dist = np.linalg.norm(pos_i - pos_j)
        
        # Target distance for a base pair (~5-6Ã…)
        target_dist = 5.5
        
        # Move nucleotides closer or further as needed
        if current_dist > 0:
            direction = (pos_j - pos_i) / current_dist
            adjustment = (current_dist - target_dist) * prob * 0.5
            
            refined[i] = pos_i + direction * adjustment
            refined[j] = pos_j - direction * adjustment
    
    # Apply constraints for backbone bond lengths
    bond_length = physical_constraints.get('bond_lengths', {}).get('backbone', 3.8)
    
    for i in range(1, len(structure)):
        pos_prev = refined[i-1]
        pos_curr = refined[i]
        
        current_bond = np.linalg.norm(pos_curr - pos_prev)
        
        if current_bond > 0:
            direction = (pos_curr - pos_prev) / current_bond
            refined[i] = pos_prev + direction * bond_length
    
    return refined

def geometric_assembly(sequence, predicted_contacts, fragment_library):
    """
    Assembles a 3D structure using fragments and satisfying physical constraints.
    """
    # 1. Decomposition of sequence into likely structural elements
    structure_elements = predict_secondary_structure(sequence)
    
    # 2. Select appropriate fragments from the library
    selected_fragments = {}
    for element in structure_elements:
        element_type = element['type']
        element_seq = element['sequence']
        
        # Find the most compatible fragment
        if element_type in fragment_library:
            best_fragment = find_best_fragment(element_seq, fragment_library[element_type])
            selected_fragments[element['id']] = best_fragment
    
    # 3. Initial assembly with geometric superposition
    initial_structure = assemble_fragments(selected_fragments, structure_elements)
    
    # 4. Refinement to satisfy physical constraints
    refined_structure = refine_with_constraints(
        initial_structure, 
        predicted_contacts,
        physical_constraints={
            'bond_lengths': {'backbone': 3.8},
            'bond_angles': {'backbone': 110},
            'stacking_geometry': {'distance': 3.4},
            'nonbonded_distances': {'min': 3.0}
        }
    )
    
    return refined_structure

def energy_based_refinement(structure, sequence):
    """
    Simplified version that uses the existing function for refinement.
    """
    # Extract GC content to use with the post_process_rna_structure function
    gc_count = sum(1 for base in sequence if base in ['G', 'C'])
    gc_content = gc_count / len(sequence) if len(sequence) > 0 else 0.5
    
    # Use existing functions from original code
    refined = refine_rna_backbone(structure)
    refined = post_process_rna_structure(refined, sequence, gc_content, use_global_movement=True)
    
    return refined

def advanced_rna_structure_prediction(sequence, X_valid, y_valid, optimal_params):
    """
    Advanced 3D RNA structure prediction pipeline integrated with existing code.
    
    Parameters:
    -----------
    sequence: str
        RNA sequence to predict the structure for
    X_valid, y_valid: 
        Validation data from existing code (required for compatibility)
    optimal_params: dict
        Optimal parameters from existing code
        
    Returns:
    --------
    ensemble: list
        List of predicted 3D structures (ensemble of models)
    """
    print(f"Starting advanced 3D structure prediction for RNA sequence of size {len(sequence)}")
    
    # 1. Predict RNA-RNA contact map
    print("Predicting contact map...")
    contact_map = predict_rna_contacts(sequence)
    
    # 2. Build initial graph
    print("Building RNA graph...")
    rna_graph = create_rna_graph(sequence, contact_map)
    
    # 3. Create simplified GNN model (or load if trained)
    print("Initializing GNN model...")
    # In a complete implementation, you would load a pre-trained model
    # Here, we create a simple model for demonstration
    node_features = 11  # 5 for base one-hot encoding + 6 for additional features
    edge_features = 5   # 4 for edge type (one-hot) + 1 for weight
    
    # Initializing the model (note: untrained, just for example)
    gnn_model = RNAGraphTransformer(node_features, edge_features)
    
    # 4. Generate initial coordinates using existing reference model
    print("Generating initial coordinates with reference model...")
    np.random.seed(optimal_params.get('seed', 42))
    reference_model = reference_based_approach(
        X_valid, 
        y_valid,
        geometric_sampling=True,
        noise_level=optimal_params.get('noise', 0.21),
        correlation=optimal_params.get('corr', 0.83)
    )
    
    # Convert sequence to input format of reference model
    print("Preparing sequence for prediction...")
    seq_features = np.zeros((1, 720, 5))  # One-hot encoding with padding to 720
    for i, base in enumerate(sequence):
        if i >= 720:  # Limit to avoid out of bounds index
            break
        if base == 'A':
            seq_features[0, i, 0] = 1
        elif base == 'C':
            seq_features[0, i, 1] = 1
        elif base == 'G':
            seq_features[0, i, 2] = 1
        elif base == 'U' or base == 'T':
            seq_features[0, i, 3] = 1
        else:
            seq_features[0, i, 4] = 1  # Unknown base
    
    # FIX: Ensure correct dimensions for initial coordinates
    try:
        # Generate initial prediction with reference model
        predictions = reference_model.predict(seq_features)
        
        # Extract coordinates for current sequence and ensure correct dimensions
        raw_coords = predictions[0]
        
        # Ensure coordinates have exactly the length of the sequence
        initial_coords = np.zeros((len(sequence), 3))
        
        # Copy only the necessary length
        min_length = min(len(raw_coords), len(sequence))
        initial_coords[:min_length] = raw_coords[:min_length]
        
        print(f"Generated initial coordinates with shape: {initial_coords.shape}")
    except Exception as e:
        print(f"Error generating initial coordinates: {str(e)}")
        # Emergency coordinates in case of failure
        initial_coords = np.zeros((len(sequence), 3))
        # Create a simple linear structure
        for i in range(1, len(sequence)):
            initial_coords[i] = initial_coords[i-1] + np.array([3.8, 0, 0])
        print("Using emergency coordinates due to error.")
    
    # 5. Generate fragment-based structure
    print("Generating fragment-based structure...")
    fragment_library = load_fragment_library()
    try:
        fragment_based_coords = geometric_assembly(sequence, contact_map, fragment_library)
        
        # Ensure fragment-based structure has correct size
        if len(fragment_based_coords) != len(sequence):
            temp_coords = np.zeros((len(sequence), 3))
            min_length = min(len(fragment_based_coords), len(sequence))
            temp_coords[:min_length] = fragment_based_coords[:min_length]
            fragment_based_coords = temp_coords
        
        # Normalize fragment-based structure
        fragment_based_coords = normalize_structure(fragment_based_coords)
    except Exception as e:
        print(f"Error in fragment-based assembly: {str(e)}")
        # Fallback: use only reference model
        fragment_based_coords = initial_coords.copy()
    
    # 6. Combine evidence (weighted average of coordinates)
    print("Combining structures...")
    combined_coords = weighted_coordinate_average([
        (initial_coords, 0.6),  # Higher weight for reference model prediction
        (fragment_based_coords, 0.4)
    ])
    
    # 7. Refinement with geometric distance and structure
    print("Refining geometry...")
    # Convert predicted contacts to target distance matrix
    distances = {}
    for i, j, prob in contact_map:
        if i < len(sequence) and j < len(sequence) and prob > 0.5:  # Validate indices and use only high probability contacts
            # Typical base pair distance (~5-6Ã…)
            distances[(i, j)] = 5.5
            distances[(j, i)] = 5.5
    
    # Add backbone distances (consecutive bonds)
    for i in range(len(sequence) - 1):
        distances[(i, i+1)] = 3.8  # Typical RNA backbone distance
        distances[(i+1, i)] = 3.8
    
    # Refine with distance geometry
    try:
        distance_matrix = np.zeros((len(sequence), len(sequence)))
        for i in range(len(sequence)):
            for j in range(i+1, len(sequence)):
                dist = np.linalg.norm(combined_coords[i] - combined_coords[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        weights = np.zeros((len(sequence), len(sequence)))
        
        for (i, j), _ in distances.items():
            if i < len(weights) and j < len(weights):  # Validate indices
                weights[i, j] = 1.0
        
        target_distances = distance_matrix.copy()
        for (i, j), dist in distances.items():
            if i < len(target_distances) and j < len(target_distances):  # Validate indices
                target_distances[i, j] = dist
                target_distances[j, i] = dist
        
        refined_coords = refine_with_distance_geometry(
            combined_coords, target_distances, weights
        )
    except Exception as e:
        print(f"Error in geometric refinement: {str(e)}")
        refined_coords = combined_coords
    
    # 8. Final refinement with energy minimization
    print("Applying final refinement...")
    try:
        # Calculate GC content for post_process_rna_structure
        gc_count = sequence.count('G') + sequence.count('C')
        gc_content = gc_count / len(sequence) if len(sequence) > 0 else 0.5
        
        # Use post_process_rna_structure instead of energy_based_refinement
        final_structure = post_process_rna_structure(
            refined_coords, 
            sequence, 
            gc_content, 
            use_global_movement=True
        )
        
        # Also apply backbone structure refinement
        if 'refine_rna_backbone_with_dihedrals' in globals():
            final_structure = refine_rna_backbone_with_dihedrals(final_structure)
        elif 'refine_rna_backbone' in globals():
            final_structure = refine_rna_backbone(final_structure)
    except Exception as e:
        print(f"Error in energy refinement: {str(e)}")
        final_structure = refined_coords
    
    # 9. Quality assessment and ensemble generation
    print("Generating final model ensemble...")
    ensemble = []
    
    # Add the main model
    ensemble.append(normalize_structure(final_structure))
    
    # Generate variations to complete the ensemble
    for i in range(1, 5):  # Generate 4 additional variations
        try:
            # Use progressively larger perturbation scale
            perturbation = 0.05 * i
            variation = sample_structural_variation(
                final_structure,
                noise_level=perturbation,
                preserve_distance=True,
                use_global_movement=(i % 2 == 0),
                correlation=0.9 - (i * 0.05)
            )
            ensemble.append(normalize_structure(variation))
        except Exception as e:
            print(f"Error generating variation {i}: {str(e)}")
            # In case of error, duplicate the main model
            ensemble.append(normalize_structure(final_structure))
    
    print("Advanced 3D structure prediction completed.")
    return ensemble

# ==== 5. PREDICTION PIPELINE EXECUTION ğŸš€ğŸ§¬ ====

def run_graph_based_pipeline(X_valid, y_valid, test_seq_df, sample_submission_df, output_dir, optimal_params):
    """
    Pipeline that uses the graph-based approach for predicting 3D RNA structures.
    This function replaces or complements the original run_hybrid_pipeline function.
    
    Parameters:
    -----------
    X_valid, y_valid: Validation data
    test_seq_df: DataFrame with test sequences
    sample_submission_df: Example submission format
    output_dir: Output directory for files
    optimal_params: Optimized parameters for the model
    
    Returns:
    --------
    submission_df: DataFrame
        Submission file with predicted structures
    status_dict: dict
        Dictionary with pipeline execution status
    """
    print("=" * 80)
    print("ADVANCED PIPELINE: GRAPH-BASED APPROACH".center(80))
    print("=" * 80)
    
    status = {
        'success': False,
        'model_type': 'graph_based',
        'structures_generated': 0,
        'error': None
    }
    
    try:
        # Dictionary to store structures for each sequence
        seq_to_structures = {}
        
        # Process each test sequence
        for i, (_, row) in enumerate(test_seq_df.iterrows()):
            target_id = row['target_id']
            sequence = row['sequence']
            seq_length = len(sequence)
            
            print(f"Processing sequence {i+1}/{len(test_seq_df)}, ID: {target_id}, length: {seq_length}")
            
            # Generate 3D structures using advanced pipeline
            structures = advanced_rna_structure_prediction(
                sequence, 
                X_valid, 
                y_valid, 
                optimal_params
            )
            
            # Store structures
            seq_to_structures[target_id] = structures
            status['structures_generated'] += 1
        
        # Create submission DataFrame
        print("\nCreating submission file...")
        submission_df = sample_submission_df.copy()
        
        for i, row in submission_df.iterrows():
            if i % 1000 == 0:
                print(f"Processing row {i}/{len(submission_df)}")
                
            # Parse ID to get sequence ID and residue index
            id_parts = row['ID'].split('_')
            seq_id = id_parts[0]
            residue_idx = int(id_parts[1]) - 1  # Convert to zero-based indexing
            
            # Check if we have structures for this sequence
            if seq_id in seq_to_structures:
                structures = seq_to_structures[seq_id]
                
                # Check if residue index is valid
                if residue_idx < len(structures[0]):
                    # Fill coordinates for all 5 structures
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
        
        # Save submission
        graph_file = os.path.join(output_dir, 'submission_graph_based.csv')
        submission_df.to_csv(graph_file, index=False)
        print(f"Graph-based submission saved at {graph_file}")
        
        # Save as standard submission
        standard_file = os.path.join(output_dir, 'submission.csv')
        submission_df.to_csv(standard_file, index=False)
        
        # Mark as success
        status['success'] = True
        
        return submission_df, status
        
    except Exception as e:
        print(f"ERROR in graph-based pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        status['error'] = str(e)
        return None, status


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


if __name__ == "__main__":
    # Print startup banner
    print("=" * 80)
    print("RNA 3D STRUCTURE PREDICTION PIPELINE".center(80))
    print("=" * 80)
    
    # Print selected mode
    mode_description = "Advanced Graph-Based Pipeline: Modeling Long-Range Interactions"
    print(f"Selected mode: {mode_description}")
    print("-" * 80)
    
    try:
        # Start the graph-based pipeline
        start_time = time.time()
        
        print("Loading processed data...")
        X_train, y_train, X_valid, y_valid = load_processed_data()
        
        print("\nLoading test data...")
        test_seq_df = pd.read_csv(os.path.join(DATA_DIR, "test_sequences.csv"))
        sample_submission_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
        
        # Optimal parameters (can be adjusted)
        optimal_params = {
            'noise': 0.21,
            'corr': 0.83,
            'seed': 42
        }
        
        # Run the graph-based pipeline
        submission_df, status = run_graph_based_pipeline(
            X_valid, y_valid,
            test_seq_df, sample_submission_df,
            OUTPUT_DIR,
            optimal_params=optimal_params
        )
        
        # Calculate total runtime
        runtime = time.time() - start_time
        hours, remainder = divmod(runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Display results summary
        print("\n" + "=" * 80)
        print("GRAPH-BASED PIPELINE RESULTS SUMMARY".center(80))
        print("=" * 80)
        print(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        if status['success']:
            print("\nGRAPH-BASED PIPELINE STATISTICS:")
            print(f"  - Model type: {status['model_type']}")
            print(f"  - Structures generated: {status['structures_generated']}")
        else:
            print(f"\nPipeline failed with error: {status['error']}")
        
        # Display output file information
        print("\nOUTPUT FILES:")
        submission_file = os.path.join(OUTPUT_DIR, 'submission_graph_based.csv')
        if os.path.exists(submission_file):
            try:
                file_size = os.path.getsize(submission_file)
                print(f"  - Graph-based submission: {submission_file} ({file_size/1024/1024:.2f} MB)")
            except:
                print(f"  - Graph-based submission: {submission_file}")
        
        standard_file = os.path.join(OUTPUT_DIR, 'submission.csv')
        if os.path.exists(standard_file):
            try:
                file_size = os.path.getsize(standard_file)
                print(f"  - Standard submission: {standard_file} ({file_size/1024/1024:.2f} MB)")
            except:
                print(f"  - Standard submission: {standard_file}")
        
        print("=" * 80)
        
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


def normalize_for_visualization(coords):
    """
    Normaliza e centraliza coordenadas para visualizaÃ§Ã£o consistente.
    Lida com valores invÃ¡lidos e padroniza a escala.
    
    Parameters:
    -----------
    coords : numpy.ndarray
        Coordenadas 3D para normalizar
        
    Returns:
    --------
    numpy.ndarray
        Coordenadas normalizadas
    """
    import numpy as np
    
    # Cria uma cÃ³pia para evitar modificar o original
    normalized = coords.copy()
    
    # Identifica coordenadas vÃ¡lidas (nÃ£o-zero e nÃ£o-NaN)
    valid_mask = ~np.all(normalized == 0, axis=1) & ~np.any(np.isnan(normalized), axis=1)
    
    # Se nÃ£o houver coordenadas vÃ¡lidas, retornar as originais
    if not np.any(valid_mask):
        print("AVISO: Nenhuma coordenada vÃ¡lida encontrada para normalizaÃ§Ã£o")
        return normalized
    
    # Extrair apenas coordenadas vÃ¡lidas
    valid_coords = normalized[valid_mask]
    
    # 1. Centralizar na origem
    center = np.mean(valid_coords, axis=0)
    valid_coords = valid_coords - center
    
    # 2. Normalizar para escala padrÃ£o (valores mÃ¡ximos entre -50 e 50)
    max_dist = np.max(np.abs(valid_coords))
    if max_dist > 0:
        scale_factor = 50.0 / max_dist
        valid_coords = valid_coords * scale_factor
    
    # Aplicar transformaÃ§Ãµes apenas Ã s coordenadas vÃ¡lidas
    normalized[valid_mask] = valid_coords
    
    return normalized

def visualize_rna_structure_comparison(sequence_str, real_structure, X_valid, y_valid, optimal_params, title=None):
    """
    Visualize comparison between real and predicted RNA 3D structures using the graph-based approach.
    Improved with better normalization and error handling.
    
    Parameters:
    -----------
    sequence_str : str
        RNA sequence as string
    real_structure : array
        True structure coordinates  
    X_valid, y_valid : array
        Validation data for the model
    optimal_params : dict
        Parameters for the model
    title : str, optional
        Title for the visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Verificar se real_structure contÃ©m dados vÃ¡lidos
    if real_structure is None or np.all(np.isnan(real_structure)) or len(real_structure) == 0:
        print("ERRO: Estrutura real invÃ¡lida ou vazia")
        return None
    
    # Generate prediction using the graph-based approach
    try:
        structures = advanced_rna_structure_prediction(
            sequence_str, 
            X_valid, 
            y_valid, 
            optimal_params
        )
        
        # Take the best structure (first one in the ensemble)
        predicted_structure = structures[0]
    except Exception as e:
        print(f"ERRO ao gerar previsÃ£o: {str(e)}")
        return None
    
    # Verificar se predicted_structure contÃ©m dados vÃ¡lidos
    if predicted_structure is None or np.all(np.isnan(predicted_structure)) or len(predicted_structure) == 0:
        print("ERRO: Estrutura prevista invÃ¡lida ou vazia")
        return None
    
    # Ensure structures have the same length
    min_length = min(len(real_structure), len(predicted_structure))
    real_structure = real_structure[:min_length].copy()
    predicted_structure = predicted_structure[:min_length].copy()
    
    # Normalizar ambas estruturas para visualizaÃ§Ã£o consistente
    real_structure_viz = normalize_for_visualization(real_structure)
    predicted_structure_viz = normalize_for_visualization(predicted_structure)
    
    # Criar uma figura com dois subplots lado a lado
    fig = plt.figure(figsize=(16, 6))
    
    # Plot real structure
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Real RNA Structure', fontsize=12)
    
    # Verificar se hÃ¡ dados vÃ¡lidos para plotar
    valid_mask_real = ~np.all(real_structure_viz == 0, axis=1) & ~np.any(np.isnan(real_structure_viz), axis=1)
    if np.any(valid_mask_real):
        # Plot points
        ax1.scatter(real_structure_viz[valid_mask_real, 0], 
                    real_structure_viz[valid_mask_real, 1],
                    real_structure_viz[valid_mask_real, 2],
                    c=np.arange(np.sum(valid_mask_real)),
                    cmap='viridis',
                    s=50)
        
        # Connect consecutive points to show backbone
        ax1.plot(real_structure_viz[valid_mask_real, 0],
                real_structure_viz[valid_mask_real, 1], 
                real_structure_viz[valid_mask_real, 2],
                color='gray',
                alpha=0.5,
                linewidth=2)
    else:
        ax1.text(0, 0, 0, "No valid data", ha='center', va='center', fontsize=14)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Definir limites consistentes
    ax1.set_xlim([-60, 60])
    ax1.set_ylim([-60, 60])
    ax1.set_zlim([-60, 60])
    
    # Plot predicted structure
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Predicted RNA Structure (Graph-Based)', fontsize=12)
    
    # Verificar se hÃ¡ dados vÃ¡lidos para plotar
    valid_mask_pred = ~np.all(predicted_structure_viz == 0, axis=1) & ~np.any(np.isnan(predicted_structure_viz), axis=1)
    if np.any(valid_mask_pred):
        # Plot points
        ax2.scatter(predicted_structure_viz[valid_mask_pred, 0],
                    predicted_structure_viz[valid_mask_pred, 1], 
                    predicted_structure_viz[valid_mask_pred, 2],
                    c=np.arange(np.sum(valid_mask_pred)),
                    cmap='plasma',
                    s=50)
        
        # Connect consecutive points to show backbone 
        ax2.plot(predicted_structure_viz[valid_mask_pred, 0],
                predicted_structure_viz[valid_mask_pred, 1],
                predicted_structure_viz[valid_mask_pred, 2], 
                color='red',
                alpha=0.5,
                linewidth=2)
    else:
        ax2.text(0, 0, 0, "No valid data", ha='center', va='center', fontsize=14)
    
    ax2.set_xlabel('X') 
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Definir limites consistentes
    ax2.set_xlim([-60, 60])
    ax2.set_ylim([-60, 60])
    ax2.set_zlim([-60, 60])
    
    # Overall title if provided
    if title:
        fig.suptitle(title, fontsize=16)
    
    fig.tight_layout()
    plt.show()
    
    return predicted_structure  # Return for metrics calculation

def visualize_ensemble_structures(sequence_str, X_valid, y_valid, optimal_params, title=None):
    """
    Visualize all 5 structures in the ensemble generated by the graph-based approach.
    Enhanced with better normalization and error handling.
    
    Parameters:
    -----------
    sequence_str : str
        RNA sequence as string
    X_valid, y_valid : array
        Validation data for the model
    optimal_params : dict
        Parameters for the model
    title : str, optional
        Title for the visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Generate prediction using the graph-based approach
    try:
        structures = advanced_rna_structure_prediction(
            sequence_str, 
            X_valid, 
            y_valid, 
            optimal_params
        )
    except Exception as e:
        print(f"ERRO ao gerar previsÃµes: {str(e)}")
        return
    
    # Normalize all structures for visualization
    normalized_structures = []
    for structure in structures:
        if structure is not None and len(structure) > 0:
            norm_struct = normalize_for_visualization(structure)
            normalized_structures.append(norm_struct)
    
    if not normalized_structures:
        print("ERRO: Nenhuma estrutura vÃ¡lida para visualizar")
        return
    
    # Create a figure with subplots for all structures
    fig = plt.figure(figsize=(20, 10))
    
    # Plot each structure in the ensemble
    for i, structure in enumerate(normalized_structures):
        ax = fig.add_subplot(1, len(normalized_structures), i+1, projection='3d')
        ax.set_title(f'Ensemble Structure {i+1}', fontsize=12)
        
        # Verificar dados vÃ¡lidos
        valid_mask = ~np.all(structure == 0, axis=1) & ~np.any(np.isnan(structure), axis=1)
        if np.any(valid_mask):
            # Plot points
            ax.scatter(structure[valid_mask, 0], 
                       structure[valid_mask, 1],
                       structure[valid_mask, 2],
                       c=np.arange(np.sum(valid_mask)),
                       cmap='plasma',
                       s=40)
            
            # Connect consecutive points to show backbone
            ax.plot(structure[valid_mask, 0],
                    structure[valid_mask, 1], 
                    structure[valid_mask, 2],
                    color='red',
                    alpha=0.5,
                    linewidth=2)
        else:
            ax.text(0, 0, 0, "No valid data", ha='center', va='center', fontsize=14)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Definir limites consistentes
        ax.set_xlim([-60, 60])
        ax.set_ylim([-60, 60])
        ax.set_zlim([-60, 60])
    
    # Overall title if provided
    if title:
        fig.suptitle(title, fontsize=16)
    
    fig.tight_layout()
    plt.show()

def calculate_structure_metrics(real_structure, predicted_structure):
    """
    Calculate key metrics to compare real and predicted structures.
    Enhanced with better handling of invalid data.
    
    Parameters:
    -----------
    real_structure : numpy.ndarray
        Original 3D structure coordinates
    predicted_structure : numpy.ndarray
        Predicted 3D structure coordinates
    
    Returns:
    --------
    metrics : dict
        Dictionary of comparison metrics
    """
    import numpy as np
    
    # Verificar se as estruturas sÃ£o vÃ¡lidas
    if (real_structure is None or predicted_structure is None or
        len(real_structure) == 0 or len(predicted_structure) == 0):
        print("ERRO: Estruturas vazias ou invÃ¡lidas")
        return {
            'Distance MAE': 0.0,
            'Coordinate RMSE': 0.0,
            'Structural Similarity': 0.0,
            'TM-Score': 0.0
        }
    
    # Ensure structures are the same length
    min_length = min(len(real_structure), len(predicted_structure))
    real_structure = real_structure[:min_length].copy()
    predicted_structure = predicted_structure[:min_length].copy()
    
    # Identificar coordenadas vÃ¡lidas em ambas estruturas
    valid_mask = (~np.all(real_structure == 0, axis=1) & 
                 ~np.any(np.isnan(real_structure), axis=1) &
                 ~np.all(predicted_structure == 0, axis=1) &
                 ~np.any(np.isnan(predicted_structure), axis=1))
    
    # Verificar se hÃ¡ coordenadas vÃ¡lidas suficientes
    if np.sum(valid_mask) < 3:
        print("AVISO: Menos de 3 coordenadas vÃ¡lidas para comparaÃ§Ã£o")
        return {
            'Distance MAE': 0.0,
            'Coordinate RMSE': 0.0,
            'Structural Similarity': 0.0,
            'TM-Score': 0.0
        }
    
    # Usar apenas coordenadas vÃ¡lidas
    real_valid = real_structure[valid_mask]
    pred_valid = predicted_structure[valid_mask]
    
    try:
        # Calculate pairwise distances
        real_dist_matrix = np.linalg.norm(
            real_valid[:, np.newaxis] - real_valid, 
            axis=2
        )
        pred_dist_matrix = np.linalg.norm(
            pred_valid[:, np.newaxis] - pred_valid, 
            axis=2
        )
        
        # Mean absolute error of distances
        distance_mae = np.mean(np.abs(real_dist_matrix - pred_dist_matrix))
        
        # Root Mean Squared Error (RMSE) of coordinates
        rmse = np.sqrt(np.mean((real_valid - pred_valid)**2))
        
        # Structural similarity (cosine similarity of distance matrices)
        try:
            similarity = np.corrcoef(
                real_dist_matrix.ravel(), 
                pred_dist_matrix.ravel()
            )[0, 1]
        except:
            print("AVISO: Erro ao calcular similaridade estrutural")
            similarity = 0.0
        
        # Calculate TM-score
        try:
            tm_score = calculate_tm_score(pred_valid, real_valid)
        except:
            print("AVISO: Erro ao calcular TM-score")
            tm_score = 0.0
        
    except Exception as e:
        print(f"ERRO ao calcular mÃ©tricas: {str(e)}")
        return {
            'Distance MAE': 0.0,
            'Coordinate RMSE': 0.0,
            'Structural Similarity': 0.0,
            'TM-Score': 0.0
        }
    
    return {
        'Distance MAE': distance_mae,
        'Coordinate RMSE': rmse, 
        'Structural Similarity': similarity,
        'TM-Score': tm_score
    }

def plot_structure_metrics(metrics):
    """
    Visualize structure comparison metrics.
    Enhanced with better error handling.
    
    Parameters:
    -----------
    metrics : dict  
        Dictionary of comparison metrics
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Verificar se hÃ¡ mÃ©tricas vÃ¡lidas
    if not metrics:
        print("ERRO: MÃ©tricas vazias ou invÃ¡lidas")
        return
    
    # Garantir que os valores sÃ£o nÃºmeros vÃ¡lidos
    for k, v in list(metrics.items()):
        if v is None or np.isnan(v) or np.isinf(v):
            print(f"AVISO: Valor invÃ¡lido para {k}, substituindo por 0.0")
            metrics[k] = 0.0
    
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_names = list(metrics.keys())
    metrics_values = list(metrics.values())
    
    bars = ax.bar(metrics_names, metrics_values, color=['#1f77b4', '#2ca02c', '#d62728', '#9467bd'])
    ax.set_title('ğŸ§¬ RNA Structure Prediction Metrics (Graph-Based Approach) ğŸ”¬', fontsize=14)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.tick_params(axis='x', labelrotation=45)
    
    # Add value labels on top of each bar
    for bar, v in zip(bars, metrics_values):
        ax.text(bar.get_x() + bar.get_width()/2., v, 
                f'{v:.4f}', ha='center', va='bottom')
    
    # Ajustar limites do eixo y para garantir que todos os valores sejam visÃ­veis
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(min(y_min, -0.05), max(y_max, max(metrics_values) * 1.1))
    
    fig.tight_layout()
    plt.show()

def main_graph_visualization():
    """
    Main function to visualize RNA structures predicted with the graph-based approach.
    Enhanced with better error handling.
    """
    import numpy as np
    
    # Load processed data  
    try:
        X_train, y_train, X_valid, y_valid = load_processed_data()
    except Exception as e:
        print(f"ERRO ao carregar dados: {str(e)}")
        return
    
    # Verificar se os dados sÃ£o vÃ¡lidos
    if X_valid is None or y_valid is None or len(X_valid) == 0 or len(y_valid) == 0:
        print("ERRO: Dados de validaÃ§Ã£o vazios ou invÃ¡lidos")
        return
    
    # Set optimal parameters
    optimal_params = {
        'noise': 0.21,
        'corr': 0.83,
        'seed': 42
    }
    
    # Visualize multiple sequences
    num_sequences = min(3, len(X_valid))
    
    for i in range(num_sequences):
        try:
            print(f"\nVisualizando sequÃªncia {i+1}")
            
            # Converter one-hot para sequÃªncia de bases
            try:
                sequence = np.argmax(X_valid[i], axis=-1)
                base_map = {0: 'A', 1: 'C', 2: 'G', 3: 'U', 4: 'N'}
                valid_indices = np.where(X_valid[i].sum(axis=1) > 0)[0]
                
                if len(valid_indices) == 0:
                    print("AVISO: Nenhum Ã­ndice vÃ¡lido na sequÃªncia")
                    sequence_str = "N" * 10  # SequÃªncia padrÃ£o
                else:
                    sequence_str = ''.join(base_map[sequence[j]] for j in valid_indices)
                
                print(f"RNA Sequence: {sequence_str[:10]}... (length: {len(sequence_str)})")
            except Exception as e:
                print(f"ERRO ao extrair sequÃªncia: {str(e)}")
                sequence_str = "N" * 10  # SequÃªncia padrÃ£o em caso de erro
            
            # Verificar se a estrutura real Ã© vÃ¡lida
            if np.all(np.isnan(y_valid[i])):
                print("AVISO: Estrutura real contÃ©m apenas valores NaN")
                continue
            
            # Visualize comparison
            predicted_structure = visualize_rna_structure_comparison(
                sequence_str, 
                y_valid[i], 
                X_valid, 
                y_valid, 
                optimal_params,
                title=f'ğŸ§¬ RNA Structure Comparison - Sequence {i+1} (Graph-Based Method) ğŸ”�'
            )
            
            if predicted_structure is not None:
                # Calculate and plot metrics
                metrics = calculate_structure_metrics(y_valid[i], predicted_structure)
                plot_structure_metrics(metrics)
            
            # Visualize all structures in the ensemble
            visualize_ensemble_structures(
                sequence_str,
                X_valid, 
                y_valid, 
                optimal_params,
                title=f'ğŸ§¬ RNA Structure Ensemble - Sequence {i+1} ğŸ”�'
            )
            
        except Exception as e:
            print(f"ERRO ao processar sequÃªncia {i+1}: {str(e)}")
            import traceback
            traceback.print_exc()

# Para executar a visualizaÃ§Ã£o
main_graph_visualization()




