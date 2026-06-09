!pip install /kaggle/input/biopython-1-85/biopython-1.85-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


# =============================================================================
# ğŸ”§ REPRODUCIBILITY CONFIGURATION
# =============================================================================

# Master seed for entire pipeline
MASTER_SEED = 8339

# Environment variables for deterministic operations
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ['PYTHONHASHSEED'] = str(MASTER_SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['SKLEARN_SEED'] = str(MASTER_SEED)

# =============================================================================
# ğŸ“¦ STANDARD LIBRARY IMPORTS
# =============================================================================

import sys
import time
import gc
import random
import traceback
import warnings
import glob
import pickle
import hashlib
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

# Configure warnings
warnings.filterwarnings('ignore')

# =============================================================================
# ğŸ”¢ NUMERICAL COMPUTING
# =============================================================================

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize

# =============================================================================
# ğŸ¤– MACHINE LEARNING CORE
# =============================================================================

from sklearn.cluster import DBSCAN
from sklearn.utils import check_random_state
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.utils import resample

# =============================================================================
# ğŸ”¥ DEEP LEARNING (PyTorch)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# ğŸ“Š VISUALIZATION
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.gridspec import GridSpec

# =============================================================================
# ğŸ§¬ BIOINFORMATICS AND STRUCTURAL BIOLOGY
# =============================================================================

try:
    from Bio.PDB import MMCIFParser, Select
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from Bio.Seq import Seq
    from Bio.SeqUtils import seq1
    import Bio.PDB
    print("âœ… BioPython libraries loaded successfully")
except ImportError:
    print("âš ï¸� BioPython not available - some features may be limited")

# =============================================================================
# ğŸŒ� DATA STRUCTURES AND ALGORITHMS
# =============================================================================

try:
    import networkx as nx
    print("âœ… NetworkX loaded successfully")
except ImportError:
    print("âš ï¸� NetworkX not available - graph analysis features disabled")

# =============================================================================
# ğŸ�¯ REPRODUCIBILITY SETUP
# =============================================================================

def set_global_seed(seed):
    """Set global seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # CUDA reproducibility (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Apply global seed
set_global_seed(MASTER_SEED)

# =============================================================================
# ğŸ”§ CONDITIONAL IMPORTS AND CONFIGURATION
# =============================================================================

# TensorFlow Configuration (if available)
try:
    import tensorflow as tf
    
    # Set TensorFlow seed for reproducibility
    tf.random.set_seed(MASTER_SEED)
    
    # Configure threads for deterministic operations
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    # Enable determinism for ops (available in TF 2.9+)
    try:
        tf.config.experimental.enable_op_determinism()
        print("âœ… TensorFlow determinism enabled")
    except AttributeError:
        print("âš ï¸� TensorFlow experimental determinism not available in this version")
        
except ImportError:
    print("âš ï¸� TensorFlow not available - PyTorch will be used for deep learning")

# =============================================================================
# ğŸ“� DIRECTORY CONFIGURATION
# =============================================================================

# Kaggle paths configuration
DATA_DIR = "/kaggle/input/stanford-rna-3d-folding"
OUTPUT_DIR = "/kaggle/working"
WORKING_DIR = "/kaggle/working"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# âœ… FINAL REPRODUCIBILITY VERIFICATION
# =============================================================================

# Force sequential thread model for NumPy (redundant but ensures consistency)
np.random.seed(MASTER_SEED)

# Verify PyTorch determinism
torch.manual_seed(MASTER_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(MASTER_SEED)

print(f"ğŸ�¯ Master Seed: {MASTER_SEED}")
print(f"ğŸ�� Python: {sys.version.split()[0]}")
print(f"ğŸ“Š NumPy: {np.__version__}")
print(f"ğŸ�¼ Pandas: {pd.__version__}")
print(f"ğŸ”¥ PyTorch: {torch.__version__}")
print(f"ğŸ¤– Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
print(f"ğŸ“� Data Dir: {DATA_DIR}")
print(f"ğŸ’¾ Output Dir: {OUTPUT_DIR}")
print("âœ… All libraries configured with reproducibility settings")
print("=" * 80)


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


"""
â•”â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•—
â•‘     TURBINED RNA 3D - PLAN B: PHYSICS + VOTING             â•‘
â•‘     Stanford RNA 3D Folding Competition                      â•‘
â•‘                                                              â•‘
â•‘     Base: Conservative (0.38346 private)                    â•‘
â•‘     + Refinamento iterativo com fÃ­sica                      â•‘
â•‘     + Ensemble por votaÃ§Ã£o ponderada                        â•‘
â•‘                                                              â•‘
â•‘     Score esperado: 0.39-0.395 private                      â•‘
â•‘     Tempo: ~4-5 minutos                                     â•‘
â•‘     Risco: Baixo-MÃ©dio                                      â•‘
â•‘                                                              â•‘
â•‘     SE < 0.383: REVERTER para Conservative                  â•‘
â•‘     SE > 0.390: SUCESSO! ğŸ�‰                                 â•‘
â•šâ•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÃ‡ÃƒO
# ============================================================================

DATA_DIR = "/kaggle/input/stanford-rna-3d-folding/"
OUTPUT_DIR = "/kaggle/working/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("TURBINED RNA 3D - PLAN B: PHYSICS + VOTING")
print("="*70)
print("Base: Conservative + 2 melhorias cirÃºrgicas")
print("Score esperado: 0.39-0.395 private")
print("="*70)

# ============================================================================
# FUNÃ‡Ã•ES BASE (mantidas do Conservative)
# ============================================================================

def decode_onehot_to_string(sequence):
    if isinstance(sequence, str):
        return sequence
    if len(sequence.shape) == 1:
        return 'N' * len(sequence)
    bases = ['A', 'C', 'G', 'U', 'N']
    seq_string = ''
    for one_hot in sequence:
        idx = np.argmax(one_hot)
        seq_string += bases[min(idx, 4)]
    return seq_string

def string_to_onehot(sequence):
    bases = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    seq_length = len(sequence)
    one_hot = np.zeros((seq_length, 5))
    for i, base in enumerate(sequence):
        if base in bases:
            one_hot[i, bases[base]] = 1
        else:
            one_hot[i, 4] = 1
    return one_hot

def calculate_gc_content(sequence):
    seq_str = decode_onehot_to_string(sequence)
    if len(seq_str) == 0:
        return 0.5
    return (seq_str.count('G') + seq_str.count('C')) / len(seq_str)

def calculate_radius_of_gyration(coords):
    valid_mask = ~np.all(coords == 0, axis=1)
    valid_coords = coords[valid_mask]
    if len(valid_coords) == 0:
        return 0
    center = np.mean(valid_coords, axis=0)
    distances = np.linalg.norm(valid_coords - center, axis=1)
    return np.sqrt(np.mean(distances ** 2))

def calculate_tm_score_simple(coords1, coords2):
    valid_mask1 = ~np.all(coords1 == 0, axis=1)
    valid_mask2 = ~np.all(coords2 == 0, axis=1)
    valid_mask = valid_mask1 & valid_mask2
    if np.sum(valid_mask) < 3:
        return 0.0
    coords1_valid = coords1[valid_mask]
    coords2_valid = coords2[valid_mask]
    rmsd = np.sqrt(np.mean(np.sum((coords1_valid - coords2_valid)**2, axis=1)))
    L = len(coords1_valid)
    d0 = 1.24 * (L - 15)**(1/3) - 1.8 if L > 15 else 0.5
    tm_score = np.mean(1 / (1 + (rmsd / d0)**2))
    return tm_score

def predict_secondary_structure(sequence):
    seq_str = decode_onehot_to_string(sequence)
    try:
        import RNA
        structure, mfe = RNA.fold(seq_str)
        return structure
    except ImportError:
        n = len(seq_str)
        structure = ['.' for _ in range(n)]
        paired = set()
        for i in range(n):
            if i in paired:
                continue
            for j in range(i+3, min(i+30, n)):
                if j in paired:
                    continue
                if (seq_str[i] == 'G' and seq_str[j] == 'C') or \
                   (seq_str[i] == 'C' and seq_str[j] == 'G') or \
                   (seq_str[i] == 'A' and seq_str[j] == 'U') or \
                   (seq_str[i] == 'U' and seq_str[j] == 'A'):
                    structure[i] = '('
                    structure[j] = ')'
                    paired.add(i)
                    paired.add(j)
                    break
        return ''.join(structure)

def find_base_pairs(secondary_structure):
    pairs = []
    stack = []
    for i, char in enumerate(secondary_structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                pairs.append((j, i))
    return pairs

def classify_structural_regions(secondary_structure):
    classification = {}
    n = len(secondary_structure)
    pairs = find_base_pairs(secondary_structure)
    pair_dict = {i: j for i, j in pairs}
    pair_dict.update({j: i for i, j in pairs})
    
    for i in range(n):
        if secondary_structure[i] == '.':
            classification[i] = 'unpaired'
        elif i in pair_dict:
            is_helix = False
            if i > 0 and i+1 < n:
                if (i-1 in pair_dict and i+1 in pair_dict):
                    is_helix = True
            classification[i] = 'helix' if is_helix else 'junction'
        else:
            classification[i] = 'unknown'
    
    for i in range(n):
        if classification[i] == 'unpaired':
            neighbors_paired = sum(
                1 for j in range(max(0, i-3), min(n, i+4))
                if j in pair_dict
            )
            if neighbors_paired >= 4:
                classification[i] = 'loop'
    
    return classification

def calculate_sequence_properties(sequence):
    seq_str = decode_onehot_to_string(sequence)
    if len(seq_str) == 0:
        return {
            'length': 0, 'gc_content': 0.5, 'purine_ratio': 0.5,
            'a_content': 0.25, 'u_content': 0.25,
            'g_content': 0.25, 'c_content': 0.25,
            'secondary_structure': ''
        }
    props = {
        'length': len(seq_str),
        'gc_content': (seq_str.count('G') + seq_str.count('C')) / len(seq_str),
        'purine_ratio': (seq_str.count('A') + seq_str.count('G')) / len(seq_str),
        'a_content': seq_str.count('A') / len(seq_str),
        'u_content': seq_str.count('U') / len(seq_str),
        'g_content': seq_str.count('G') / len(seq_str),
        'c_content': seq_str.count('C') / len(seq_str),
    }
    props['secondary_structure'] = predict_secondary_structure(sequence)
    return props

def compare_secondary_structures(struct1, struct2):
    if len(struct1) != len(struct2):
        min_len = min(len(struct1), len(struct2))
        struct1 = struct1[:min_len]
        struct2 = struct2[:min_len]
    if len(struct1) == 0:
        return 0.0
    matches = sum(1 for a, b in zip(struct1, struct2) if a == b)
    return matches / len(struct1)

def calculate_overall_similarity(props1, props2):
    gc_sim = 1 - abs(props1['gc_content'] - props2['gc_content'])
    length_sim = min(props1['length'], props2['length']) / max(props1['length'], props2['length'])
    purine_sim = 1 - abs(props1['purine_ratio'] - props2['purine_ratio'])
    struct_sim = compare_secondary_structures(
        props1['secondary_structure'],
        props2['secondary_structure']
    )
    total = (gc_sim * 0.3 + length_sim * 0.4 + 
             purine_sim * 0.15 + struct_sim * 0.15)
    return total

class SmartReferenceCache:
    def __init__(self, max_cache_size=2000):
        self.cache = {}
        self.hit_counts = {}
        self.max_size = max_cache_size
    
    def get_key(self, sequence_props):
        gc_bin = int(sequence_props['gc_content'] * 10)
        length_bin = int(sequence_props['length'] / 50)
        return f"gc{gc_bin}_len{length_bin}"
    
    def get_references(self, target_seq, all_references):
        props = calculate_sequence_properties(target_seq)
        key = self.get_key(props)
        if key in self.cache:
            self.hit_counts[key] = self.hit_counts.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def add_to_cache(self, target_seq, references):
        props = calculate_sequence_properties(target_seq)
        key = self.get_key(props)
        if len(self.cache) >= self.max_size:
            if self.hit_counts:
                min_key = min(self.hit_counts, key=self.hit_counts.get)
                del self.cache[min_key]
                del self.hit_counts[min_key]
        self.cache[key] = references
        self.hit_counts[key] = 1
    
    def get_stats(self):
        total = sum(self.hit_counts.values())
        size = len(self.cache)
        return {
            'cache_size': size,
            'total_requests': total,
            'hit_rate': sum(c > 1 for c in self.hit_counts.values()) / max(size, 1)
        }

def simple_sequence_similarity(seq1, seq2):
    str1 = decode_onehot_to_string(seq1)
    str2 = decode_onehot_to_string(seq2)
    min_len = min(len(str1), len(str2))
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 0.0
    matches = sum(1 for i in range(min_len) if str1[i] == str2[i])
    return matches / max_len

def select_best_reference_by_sequence(target_seq, reference_pool, top_k=5):
    similarities = []
    for ref_idx, ref_data in enumerate(reference_pool):
        ref_seq = ref_data['sequence']
        similarity = simple_sequence_similarity(target_seq, ref_seq)
        similarities.append({
            'index': ref_idx,
            'similarity': similarity,
            'data': ref_data
        })
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return [s['data'] for s in similarities[:top_k]]

def select_by_physical_properties(target_seq, reference_pool, top_k=5):
    target_props = calculate_sequence_properties(target_seq)
    scores = []
    for ref_data in reference_pool:
        ref_props = calculate_sequence_properties(ref_data['sequence'])
        score = calculate_overall_similarity(target_props, ref_props)
        scores.append({'score': score, 'data': ref_data})
    scores.sort(key=lambda x: x['score'], reverse=True)
    return [s['data'] for s in scores[:top_k]]

def select_references_adaptive(target_seq, reference_pool, base_top_k=10):
    pool_size = len(reference_pool)
    
    if pool_size < 1000:
        top_k_seq = base_top_k
        top_k_props = 5
    elif pool_size < 3000:
        top_k_seq = base_top_k + 5
        top_k_props = 7
    else:
        top_k_seq = base_top_k + 10
        top_k_props = 10
    
    seq_similar = select_best_reference_by_sequence(
        target_seq, reference_pool, top_k=top_k_seq
    )
    
    best_refs = select_by_physical_properties(
        target_seq, seq_similar, top_k=top_k_props
    )
    
    return best_refs

def interpolate_multiple_references(references, target_props, method='weighted'):
    if len(references) == 0:
        return None
    
    target_length = target_props['length']
    best_ref = None
    best_score = -1
    
    for ref in references:
        if len(ref['structure']) != target_length:
            continue
        ref_props = calculate_sequence_properties(ref['sequence'])
        similarity = calculate_overall_similarity(target_props, ref_props)
        if similarity > best_score:
            best_score = similarity
            best_ref = ref
    
    if best_ref is None:
        best_ref = max(references, 
                      key=lambda r: calculate_overall_similarity(
                          calculate_sequence_properties(r['sequence']), 
                          target_props
                      ))
        base_structure = best_ref['structure']
        if len(base_structure) > target_length:
            return base_structure[:target_length]
        else:
            extended = np.zeros((target_length, 3))
            extended[:len(base_structure)] = base_structure
            return extended
    
    return best_ref['structure']

def generate_synthetic_references(original_refs, n_synthetic=100):
    if len(original_refs) < 2:
        return []
    
    size_groups = {}
    for ref in original_refs:
        size = len(ref['structure'])
        size_bin = size // 50
        if size_bin not in size_groups:
            size_groups[size_bin] = []
        size_groups[size_bin].append(ref)
    
    synthetic_refs = []
    
    for size_bin, refs_in_bin in size_groups.items():
        if len(refs_in_bin) < 2:
            continue
        
        n_for_bin = max(1, int(n_synthetic * len(refs_in_bin) / len(original_refs)))
        
        for _ in range(n_for_bin):
            n_to_combine = min(np.random.randint(2, 4), len(refs_in_bin))
            selected_indices = np.random.choice(len(refs_in_bin), n_to_combine, replace=False)
            selected_refs = [refs_in_bin[i] for i in selected_indices]
            
            sizes = [len(ref['structure']) for ref in selected_refs]
            target_size = max(set(sizes), key=sizes.count)
            exact_refs = [ref for ref in selected_refs if len(ref['structure']) == target_size]
            
            if len(exact_refs) < 2:
                continue
            
            combined_structure = np.mean(
                [ref['structure'] for ref in exact_refs],
                axis=0
            )
            
            synthetic_refs.append({
                'structure': combined_structure,
                'sequence': exact_refs[0]['sequence'],
                'is_synthetic': True
            })
    
    return synthetic_refs[:n_synthetic]

def residue_specific_noise(coords, sequence, base_noise_level=0.2):
    noise_scales = {'A': 0.8, 'G': 0.85, 'C': 1.1, 'U': 1.15, 'N': 1.0}
    new_coords = coords.copy()
    valid_mask = ~np.all(coords == 0, axis=1)
    seq_str = decode_onehot_to_string(sequence)
    for i in range(len(coords)):
        if not valid_mask[i]:
            continue
        base = seq_str[i] if i < len(seq_str) else 'N'
        scale = noise_scales.get(base, 1.0)
        noise = np.random.normal(0, base_noise_level * scale, 3)
        new_coords[i] += noise
    return new_coords

def context_aware_noise(coords, sequence, base_noise_level=0.2):
    secondary_structure = predict_secondary_structure(sequence)
    structural_context = classify_structural_regions(secondary_structure)
    noise_by_context = {
        'helix': 0.5, 'loop': 1.5, 'junction': 1.0,
        'unpaired': 1.3, 'unknown': 1.0
    }
    new_coords = coords.copy()
    valid_mask = ~np.all(coords == 0, axis=1)
    for i in range(len(coords)):
        if not valid_mask[i]:
            continue
        context = structural_context.get(i, 'unknown')
        scale = noise_by_context[context]
        noise_level = base_noise_level * scale
        noise = np.random.normal(0, noise_level, 3)
        new_coords[i] += noise
    return new_coords

def simple_noise(coords, sequence, base_noise_level=0.2):
    new_coords = coords.copy()
    valid_mask = ~np.all(coords == 0, axis=1)
    noise = np.random.normal(0, base_noise_level, coords.shape)
    new_coords[valid_mask] += noise[valid_mask]
    return new_coords

def generate_candidates_diverse(base_structure, target_seq, n_candidates=20, noise_level=0.21):
    candidates = []
    n_per_type = n_candidates // 3
    
    for i in range(n_per_type):
        noise_variation = noise_level * np.random.uniform(0.7, 1.3)
        candidate = residue_specific_noise(base_structure, target_seq, noise_variation)
        candidates.append(candidate)
    
    for i in range(n_per_type):
        noise_variation = noise_level * np.random.uniform(0.7, 1.3)
        candidate = context_aware_noise(base_structure, target_seq, noise_variation)
        candidates.append(candidate)
    
    remaining = n_candidates - len(candidates)
    for i in range(remaining):
        noise_variation = noise_level * np.random.uniform(0.7, 1.3)
        candidate = simple_noise(base_structure, target_seq, noise_variation)
        candidates.append(candidate)
    
    return candidates

def estimate_structure_quality(coords):
    valid_mask = ~np.all(coords == 0, axis=1)
    valid_coords = coords[valid_mask]
    if len(valid_coords) < 3:
        return 0
    
    quality = 0
    rg = calculate_radius_of_gyration(coords)
    quality += 1.0 / (1.0 + rg / 10)
    
    backbone_dists = []
    for i in range(len(valid_coords) - 1):
        dist = np.linalg.norm(valid_coords[i+1] - valid_coords[i])
        backbone_dists.append(dist)
    
    if backbone_dists:
        backbone_var = np.var(backbone_dists)
        quality += 1.0 / (1.0 + backbone_var)
    
    return max(quality, 0)

# ============================================================================
# MELHORIA #1: REFINAMENTO ITERATIVO COM FÃ�SICA
# ============================================================================

def regularize_backbone(coords, target_dist=5.9, alpha=0.1):
    """
    ForÃ§a distÃ¢ncias backbone prÃ³ximas de 5.9Ã… (distÃ¢ncia C1'-C1')
    """
    new_coords = coords.copy()
    valid_mask = ~np.all(coords == 0, axis=1)
    
    for i in range(len(coords) - 1):
        if not (valid_mask[i] and valid_mask[i+1]):
            continue
        
        vec = new_coords[i+1] - new_coords[i]
        dist = np.linalg.norm(vec)
        
        if dist < 1e-6:
            continue
        
        # Corrigir para target_dist
        correction = (target_dist - dist) * alpha
        direction = vec / dist
        
        new_coords[i+1] = new_coords[i+1] + correction * direction
    
    return new_coords

def remove_steric_clashes(coords, min_dist=3.0, alpha=0.05):
    """
    Remove clashes movendo Ã¡tomos que estÃ£o muito prÃ³ximos
    RNA nÃ£o pode ter distÃ¢ncias < 3.0Ã… entre resÃ­duos nÃ£o-adjacentes
    """
    new_coords = coords.copy()
    valid_mask = ~np.all(coords == 0, axis=1)
    
    for i in range(len(coords)):
        if not valid_mask[i]:
            continue
        
        for j in range(i + 3, len(coords)):  # Pular adjacentes
            if not valid_mask[j]:
                continue
            
            vec = new_coords[j] - new_coords[i]
            dist = np.linalg.norm(vec)
            
            if dist < min_dist and dist > 1e-6:
                # Empurrar para longe
                push = (min_dist - dist) * alpha
                direction = vec / dist
                new_coords[j] += push * direction
                new_coords[i] -= push * direction
    
    return new_coords

def smooth_structure(coords, window=3):
    """
    Suaviza estrutura usando mÃ©dia mÃ³vel
    Reduz jitters e irregularidades
    """
    new_coords = coords.copy()
    valid_mask = ~np.all(coords == 0, axis=1)
    
    for i in range(len(coords)):
        if not valid_mask[i]:
            continue
        
        # Coletar vizinhos
        neighbors = []
        for j in range(max(0, i - window), min(len(coords), i + window + 1)):
            if valid_mask[j]:
                neighbors.append(coords[j])
        
        if len(neighbors) > 1:
            # MÃ©dia ponderada (peso maior no centro)
            weights = np.exp(-np.arange(len(neighbors)) / 2.0)
            weights = weights / np.sum(weights)
            new_coords[i] = np.sum([w * n for w, n in zip(weights, neighbors)], axis=0)
    
    return new_coords

def apply_basepairing_constraints(coords, sequence):
    """
    Aplica constraints de base pairing
    G-C devem estar ~10.5Ã…, A-U ~11.0Ã…
    """
    new_coords = coords.copy()
    valid_mask = ~np.all(coords == 0, axis=1)
    
    # Encontrar pares de bases
    sec_struct = predict_secondary_structure(sequence)
    base_pairs = find_base_pairs(sec_struct)
    
    if not base_pairs:
        return new_coords
    
    seq_str = decode_onehot_to_string(sequence)
    alpha = 0.05  # Taxa de ajuste suave
    
    for i, j in base_pairs:
        if not (valid_mask[i] and valid_mask[j]):
            continue
        
        # Determinar distÃ¢ncia ideal
        if (seq_str[i] == 'G' and seq_str[j] == 'C') or \
           (seq_str[i] == 'C' and seq_str[j] == 'G'):
            ideal_dist = 10.5
        else:  # A-U ou outras
            ideal_dist = 11.0
        
        # DistÃ¢ncia atual
        vec = new_coords[j] - new_coords[i]
        dist = np.linalg.norm(vec)
        
        if dist < 1e-6:
            continue
        
        # Ajustar
        correction = (ideal_dist - dist) * alpha
        direction = vec / dist
        
        new_coords[j] += correction * direction * 0.5
        new_coords[i] -= correction * direction * 0.5
    
    return new_coords

def iterative_physics_refinement(coords, sequence, n_iterations=8):
    """
    Refina estrutura iterativamente com constraints fÃ­sicos
    
    Args:
        coords: Coordenadas iniciais
        sequence: SequÃªncia do RNA
        n_iterations: NÃºmero de iteraÃ§Ãµes (default: 8)
    
    Returns:
        Coordenadas refinadas
    """
    current = coords.copy()
    
    for iteration in range(n_iterations):
        # 1. Regularizar backbone
        current = regularize_backbone(current, target_dist=5.9, alpha=0.1)
        
        # 2. Aplicar base-pairing constraints
        current = apply_basepairing_constraints(current, sequence)
        
        # 3. Remover clashes
        current = remove_steric_clashes(current, min_dist=3.0, alpha=0.05)
        
        # 4. Suavizar
        if iteration % 2 == 0:  # Suavizar a cada 2 iteraÃ§Ãµes
            current = smooth_structure(current, window=3)
    
    return current

# ============================================================================
# MELHORIA #2: ENSEMBLE POR VOTAÃ‡ÃƒO PONDERADA
# ============================================================================

def weighted_voting_ensemble(candidates, target_seq, n_final=5):
    """
    Ensemble por votaÃ§Ã£o ponderada:
    - Para cada resÃ­duo, calcula centrÃ³ide de todas as posiÃ§Ãµes
    - Seleciona as n_final estruturas mais prÃ³ximas do consenso
    - Mais robusto que apenas pegar top-k por qualidade global
    """
    if len(candidates) <= n_final:
        return candidates
    
    n_residues = len(candidates[0])
    
    # Calcular qualidade de cada candidato
    qualities = np.array([estimate_structure_quality(c) for c in candidates])
    
    # Pegar top-20 por qualidade (filtro inicial)
    top_indices = np.argsort(qualities)[::-1][:min(20, len(candidates))]
    top_candidates = [candidates[i] for i in top_indices]
    top_qualities = qualities[top_indices]
    
    # Para cada candidato, calcular distÃ¢ncia ao consenso
    consensus_distances = []
    
    for candidate in top_candidates:
        total_dist = 0
        
        for res_idx in range(n_residues):
            # Coletar todas as posiÃ§Ãµes deste resÃ­duo
            positions = [c[res_idx] for c in top_candidates]
            
            # Calcular centrÃ³ide (consenso)
            centroid = np.mean(positions, axis=0)
            
            # DistÃ¢ncia deste candidato ao centrÃ³ide
            dist = np.linalg.norm(candidate[res_idx] - centroid)
            total_dist += dist
        
        # MÃ©dia da distÃ¢ncia
        avg_dist = total_dist / n_residues
        consensus_distances.append(avg_dist)
    
    # Combinar qualidade e proximidade ao consenso
    # 60% consenso, 40% qualidade
    normalized_distances = np.array(consensus_distances)
    normalized_distances = 1 - (normalized_distances / (np.max(normalized_distances) + 1e-6))
    
    normalized_qualities = top_qualities / (np.max(top_qualities) + 1e-6)
    
    combined_scores = 0.6 * normalized_distances + 0.4 * normalized_qualities
    
    # Selecionar top-n_final
    best_indices = np.argsort(combined_scores)[::-1][:n_final]
    
    selected = [top_candidates[i] for i in best_indices]
    
    return selected

# ============================================================================
# PREPARAÃ‡ÃƒO DE DADOS
# ============================================================================

def prepare_training_data_final(train_seq_df, train_labels_df, max_length=None):
    print("\n" + "="*70)
    print("PREPARANDO DADOS DE TREINO")
    print("="*70)
    
    X_train = []
    y_train = []
    metadata = []
    
    print("\nExtraindo target_id dos labels...")
    train_labels_df['target_id'] = train_labels_df['ID'].str.rsplit('_', n=1).str[0]
    
    print("Agrupando labels por target_id...")
    grouped_labels = train_labels_df.groupby('target_id')
    
    print(f"\nTotal de targets Ãºnicos nos labels: {len(grouped_labels)}")
    
    print("\nProcessando sequÃªncias...")
    skipped = 0
    
    for idx, row in tqdm(train_seq_df.iterrows(), total=len(train_seq_df), desc="Processing"):
        target_id = row['target_id']
        sequence = row['sequence']
        
        if target_id not in grouped_labels.groups:
            skipped += 1
            continue
        
        if max_length and len(sequence) > max_length:
            skipped += 1
            continue
        
        target_labels = grouped_labels.get_group(target_id)
        target_labels = target_labels.sort_values('resid')
        coords = target_labels[['x_1', 'y_1', 'z_1']].values
        
        seq_onehot = string_to_onehot(sequence)
        
        if len(seq_onehot) != len(coords):
            min_len = min(len(seq_onehot), len(coords))
            seq_onehot = seq_onehot[:min_len]
            coords = coords[:min_len]
            if min_len < 10:
                skipped += 1
                continue
        
        X_train.append(seq_onehot)
        y_train.append(coords)
        metadata.append({
            'target_id': target_id,
            'sequence': sequence[:len(coords)],
            'length': len(coords),
            'gc_content': calculate_gc_content(seq_onehot)
        })
    
    metadata_df = pd.DataFrame(metadata)
    
    print(f"\nâœ“ Preparados: {len(X_train)} exemplos de treino")
    print(f"âœ— Pulados: {skipped} exemplos")
    
    if len(X_train) > 0:
        lengths = [len(x) for x in X_train]
        print(f"\nEstatÃ­sticas de comprimento:")
        print(f"  Min: {min(lengths)}")
        print(f"  Max: {max(lengths)}")
        print(f"  MÃ©dia: {np.mean(lengths):.1f}")
        print(f"  Mediana: {np.median(lengths):.1f}")
    
    print("="*70)
    
    return X_train, y_train, metadata_df

def prepare_test_data(test_seq_df):
    print("\n" + "="*70)
    print("PREPARANDO DADOS DE TESTE")
    print("="*70)
    
    X_test = []
    test_metadata = []
    
    for idx, row in tqdm(test_seq_df.iterrows(), total=len(test_seq_df), desc="Processing"):
        target_id = row['target_id']
        sequence = row['sequence']
        seq_onehot = string_to_onehot(sequence)
        X_test.append(seq_onehot)
        test_metadata.append({
            'target_id': target_id,
            'sequence': sequence,
            'length': len(sequence),
            'gc_content': calculate_gc_content(seq_onehot)
        })
    
    test_metadata_df = pd.DataFrame(test_metadata)
    print(f"\nâœ“ Preparados: {len(X_test)} exemplos de teste")
    
    if len(X_test) > 0:
        lengths = [len(x) for x in X_test]
        print(f"\nEstatÃ­sticas de comprimento:")
        print(f"  Min: {min(lengths)}")
        print(f"  Max: {max(lengths)}")
        print(f"  MÃ©dia: {np.mean(lengths):.1f}")
    
    print("="*70)
    
    return X_test, test_metadata_df

# ============================================================================
# MODELO - PLAN B
# ============================================================================

class TurbinedReferenceModelPlanB:
    """
    Modelo Plan B:
    Base Conservative + Refinamento FÃ­sico + Ensemble por VotaÃ§Ã£o
    """
    def __init__(self, noise_level=0.21, n_candidates=20, n_final=5,
                 use_cache=True, generate_synthetic=True, n_synthetic=100,
                 use_physics_refinement=True, physics_iterations=8):
        self.noise_level = noise_level
        self.n_candidates = n_candidates
        self.n_final = n_final
        self.use_cache = use_cache
        self.generate_synthetic = generate_synthetic
        self.n_synthetic = n_synthetic
        self.use_physics_refinement = use_physics_refinement
        self.physics_iterations = physics_iterations
        if self.use_cache:
            self.cache = SmartReferenceCache(max_cache_size=2000)
        self.is_fitted = False
    
    def fit(self, X_train, y_train):
        print("\n" + "="*70)
        print("TREINANDO MODELO - PLAN B: PHYSICS + VOTING")
        print("="*70)
        
        self.reference_pool = []
        for i in range(len(X_train)):
            self.reference_pool.append({
                'sequence': X_train[i],
                'structure': y_train[i],
                'is_synthetic': False
            })
        
        print(f"âœ“ {len(self.reference_pool)} referÃªncias originais")
        
        if self.generate_synthetic and len(self.reference_pool) >= 2:
            print(f"\nGerando {self.n_synthetic} referÃªncias sintÃ©ticas...")
            synthetic_refs = generate_synthetic_references(
                self.reference_pool, n_synthetic=self.n_synthetic
            )
            self.reference_pool.extend(synthetic_refs)
            print(f"âœ“ Total: {len(self.reference_pool)} referÃªncias")
        
        print(f"\nğŸ”§ ConfiguraÃ§Ãµes:")
        print(f"  Refinamento fÃ­sico: {'Ativado' if self.use_physics_refinement else 'Desativado'}")
        if self.use_physics_refinement:
            print(f"  IteraÃ§Ãµes de fÃ­sica: {self.physics_iterations}")
        print(f"  Ensemble: VotaÃ§Ã£o ponderada (consenso)")
        
        self.is_fitted = True
        print("\nâœ“ MODELO TREINADO!")
        print("="*70)
        return self
    
    def predict(self, X_test, verbose=True):
        if not self.is_fitted:
            raise ValueError("Modelo nÃ£o foi treinado!")
        
        if verbose:
            print("\n" + "="*70)
            print("GERANDO PREDIÃ‡Ã•ES - PLAN B")
            print("="*70)
        
        all_predictions = []
        
        iterator = tqdm(enumerate(X_test), total=len(X_test), desc="Predicting") if verbose else enumerate(X_test)
        
        for idx, target_seq in iterator:
            # Cache lookup
            best_refs = None
            if self.use_cache:
                best_refs = self.cache.get_references(target_seq, self.reference_pool)
            
            if best_refs is None:
                best_refs = select_references_adaptive(
                    target_seq, self.reference_pool, base_top_k=10
                )
                if self.use_cache:
                    self.cache.add_to_cache(target_seq, best_refs)
            
            # InterpolaÃ§Ã£o
            target_props = calculate_sequence_properties(target_seq)
            base_structure = interpolate_multiple_references(
                best_refs, target_props, method='weighted'
            )
            
            if base_structure is None:
                base_structure = best_refs[0]['structure']
            
            # Gerar candidatos com diversidade
            candidates = generate_candidates_diverse(
                base_structure, target_seq, 
                n_candidates=self.n_candidates,
                noise_level=self.noise_level
            )
            
            # MELHORIA #1: Refinamento fÃ­sico
            if self.use_physics_refinement:
                refined_candidates = []
                for candidate in candidates:
                    refined = iterative_physics_refinement(
                        candidate, target_seq, n_iterations=self.physics_iterations
                    )
                    refined_candidates.append(refined)
                candidates = refined_candidates
            
            # MELHORIA #2: Ensemble por votaÃ§Ã£o
            final_structures = weighted_voting_ensemble(
                candidates, target_seq, n_final=self.n_final
            )
            
            while len(final_structures) < self.n_final:
                final_structures.append(final_structures[0])
            final_structures = final_structures[:self.n_final]
            
            all_predictions.append(np.array(final_structures))
        
        if verbose:
            print("\nâœ“ PREDIÃ‡Ã•ES CONCLUÃ�DAS!")
            if self.use_cache:
                stats = self.cache.get_stats()
                print(f"Cache: {stats['cache_size']} entries, hit rate: {stats['hit_rate']:.2%}")
            print("="*70)
        
        return all_predictions

# ============================================================================
# SUBMISSÃƒO
# ============================================================================

def fix_nans_in_submission(submission_df):
    nan_count_before = submission_df.isna().sum().sum()
    
    if nan_count_before == 0:
        return submission_df
    
    print(f"\nâš ï¸� Corrigindo {nan_count_before} NaNs...")
    
    coord_cols = []
    for i in range(1, 6):
        coord_cols.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
    
    for idx, row in submission_df.iterrows():
        if row[coord_cols].isna().any():
            for struct_idx in range(1, 6):
                x_col = f'x_{struct_idx}'
                y_col = f'y_{struct_idx}'
                z_col = f'z_{struct_idx}'
                
                if pd.isna(row[x_col]) or pd.isna(row[y_col]) or pd.isna(row[z_col]):
                    valid_coords = None
                    
                    for other_idx in range(1, 6):
                        if other_idx == struct_idx:
                            continue
                        
                        other_x = row[f'x_{other_idx}']
                        other_y = row[f'y_{other_idx}']
                        other_z = row[f'z_{other_idx}']
                        
                        if not pd.isna(other_x) and not pd.isna(other_y) and not pd.isna(other_z):
                            valid_coords = (other_x, other_y, other_z)
                            break
                    
                    if valid_coords:
                        submission_df.at[idx, x_col] = valid_coords[0]
                        submission_df.at[idx, y_col] = valid_coords[1]
                        submission_df.at[idx, z_col] = valid_coords[2]
                    else:
                        resid = row['resid']
                        submission_df.at[idx, x_col] = resid * 5.9
                        submission_df.at[idx, y_col] = 0.0
                        submission_df.at[idx, z_col] = 0.0
    
    nan_count_after = submission_df.isna().sum().sum()
    
    if nan_count_after > 0:
        submission_df[coord_cols] = submission_df[coord_cols].fillna(0.0)
    
    print(f"âœ“ Corrigidos: {nan_count_before} â†’ {nan_count_after}")
    
    return submission_df

def create_submission_correct(predictions, test_metadata, test_sequences_df, output_file="submission.csv"):
    print("\n" + "="*70)
    print("CRIANDO SUBMISSÃƒO")
    print("="*70)
    
    submission_rows = []
    
    for idx, target_id in enumerate(tqdm(test_metadata['target_id'], desc="Creating submission")):
        sequence = test_metadata.iloc[idx]['sequence']
        structures = predictions[idx]
        n_residues = len(structures[0])
        
        for res_idx in range(n_residues):
            row_id = f"{target_id}_{res_idx + 1}"
            base = sequence[res_idx] if res_idx < len(sequence) else 'N'
            resid = res_idx + 1
            
            coords_all_structures = []
            for struct_idx in range(5):
                coords = structures[struct_idx][res_idx]
                coords_all_structures.extend([coords[0], coords[1], coords[2]])
            
            row = {
                'ID': row_id,
                'resname': base,
                'resid': resid
            }
            
            for struct_idx in range(5):
                base_col_idx = struct_idx * 3
                row[f'x_{struct_idx + 1}'] = coords_all_structures[base_col_idx]
                row[f'y_{struct_idx + 1}'] = coords_all_structures[base_col_idx + 1]
                row[f'z_{struct_idx + 1}'] = coords_all_structures[base_col_idx + 2]
            
            submission_rows.append(row)
    
    submission_df = pd.DataFrame(submission_rows)
    
    coord_cols = []
    for i in range(1, 6):
        coord_cols.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
    
    columns_order = ['ID', 'resname', 'resid'] + coord_cols
    submission_df = submission_df[columns_order]
    
    submission_df = fix_nans_in_submission(submission_df)
    
    output_path = os.path.join(OUTPUT_DIR, output_file)
    submission_df.to_csv(output_path, index=False)
    
    print(f"\nâœ“ SubmissÃ£o salva: {output_path}")
    print(f"  Linhas: {len(submission_df)}")
    print(f"  Colunas: {len(submission_df.columns)}")
    print(f"  NaNs: {submission_df.isna().sum().sum()}")
    
    print("="*70)
    
    return submission_df

# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def main_pipeline_plan_b(use_v2=True, max_train_length=None, 
                         use_physics=True, physics_iterations=8):
    """
    Pipeline Plan B: Conservative + FÃ­sica + VotaÃ§Ã£o
    
    Args:
        use_v2: Usar dados v2 (default: True)
        max_train_length: Limite de comprimento (None = sem limite)
        use_physics: Ativar refinamento fÃ­sico (default: True)
        physics_iterations: NÃºmero de iteraÃ§Ãµes de fÃ­sica (default: 8)
    """
    print("""
    â•”â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•—
    â•‘     TURBINED RNA 3D - PLAN B                                â•‘
    â•‘                                                              â•‘
    â•‘     Base: Conservative 0.38346 private                      â•‘
    â•‘     + Refinamento iterativo com fÃ­sica (8 iteraÃ§Ãµes)        â•‘
    â•‘     + Ensemble por votaÃ§Ã£o ponderada (consenso)             â•‘
    â•‘                                                              â•‘
    â•‘     Score esperado: 0.39-0.395 private                      â•‘
    â•‘     Tempo estimado: 4-5 minutos                             â•‘
    â•‘                                                              â•‘
    â•‘     âš ï¸�  SE < 0.383: REVERTER para Conservative             â•‘
    â•‘     ğŸ�‰ SE > 0.390: SUCESSO!                                 â•‘
    â•šâ•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    """)
    
    # Carregar dados
    print("\nğŸ“‚ Carregando dados...")
    if use_v2:
        train_seq = pd.read_csv(os.path.join(DATA_DIR, "train_sequences.v2.csv"))
        train_labels = pd.read_csv(os.path.join(DATA_DIR, "train_labels.v2.csv"))
    else:
        train_seq = pd.read_csv(os.path.join(DATA_DIR, "train_sequences.csv"))
        train_labels = pd.read_csv(os.path.join(DATA_DIR, "train_labels.csv"))
    
    test_seq = pd.read_csv(os.path.join(DATA_DIR, "test_sequences.csv"))
    
    print(f"âœ“ Dados carregados")
    print(f"  Train sequences: {len(train_seq)}")
    print(f"  Test sequences: {len(test_seq)}")
    
    # Preparar
    X_train, y_train, train_metadata = prepare_training_data_final(
        train_seq, train_labels, max_length=max_train_length
    )
    
    if len(X_train) == 0:
        print("â�Œ ERRO: Nenhum dado de treino preparado!")
        return None, None, None
    
    X_test, test_metadata = prepare_test_data(test_seq)
    
    # Treinar modelo Plan B
    model = TurbinedReferenceModelPlanB(
        noise_level=0.21,
        n_candidates=20,
        n_final=5,
        use_cache=True,
        generate_synthetic=True,
        n_synthetic=100,
        use_physics_refinement=use_physics,
        physics_iterations=physics_iterations
    )
    
    model.fit(X_train, y_train)
    
    # Predizer
    predictions = model.predict(X_test, verbose=True)
    
    # Criar submissÃ£o
    submission = create_submission_correct(
        predictions, 
        test_metadata, 
        test_seq,
        "submission.csv"
    )
    
    print("\n" + "="*70)
    
    return model, predictions, submission

# ============================================================================
# EXECUÃ‡ÃƒO
# ============================================================================

if __name__ == "__main__":
    
    model, predictions, submission = main_pipeline_plan_b(
        use_v2=True,
        max_train_length=None,      # Processar TODAS as sequÃªncias
        use_physics=True,            # Ativar refinamento fÃ­sico
        physics_iterations=8         # 8 iteraÃ§Ãµes de fÃ­sica
    )
    
    if model is not None:
        print("\nâœ… SUCESSO! Arquivo pronto para submissÃ£o.")




