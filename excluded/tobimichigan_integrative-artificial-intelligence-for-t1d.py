import pandas as pd
import numpy as np
import gc
import psutil
import time
import os
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, validation_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                           roc_curve, auc, precision_recall_curve, f1_score, 
                           precision_score, recall_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer
from joblib import dump, load
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from pathlib import Path
import warnings
import requests
import zipfile
import shutil
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100

# --- 0. Dataset Download Integration ---
def download_file(url, filename):
    """Download a file with progress bar using tqdm."""
    print(f"Downloading {filename}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get total file size from headers
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))
    
    print(f" Downloaded: {filename}")

def extract_zip(zip_path, extract_to=None):
    """Extract zip file with progress bar using tqdm."""
    if extract_to is None:
        extract_to = os.path.dirname(zip_path)
    
    print(f"Extracting {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of files in zip
        file_list = zip_ref.namelist()
        
        # Extract with progress bar
        for file in tqdm(file_list, desc=f"Extracting {os.path.basename(zip_path)}"):
            zip_ref.extract(file, extract_to)
    
    print(f"âœ“ Extracted: {zip_path}")

def download_and_prepare_datasets():
    """Download and extract datasets with integrated progress tracking."""
    print("=" * 60)
    print("DOWNLOADING AND PREPARING DIABETES DATASETS")
    print("=" * 60)
    
    # Create working directory if it doesn't exist
    working_dir = "/kaggle/working" if "/kaggle" in os.getcwd() else "diabetes_data"
    os.makedirs(working_dir, exist_ok=True)
    original_dir = os.getcwd()
    os.chdir(working_dir)
    
    # Dataset URLs and filenames
    datasets = [
        {
            "url": "https://zenodo.org/records/15806142/files/sharpic/ManchesterCSCoordinatedDiabetesStudy-V1.0.3.zip",
            "filename": "ManchesterCSCoordinatedDiabetesStudy-V1.0.3.zip"
        },
        {
            "url": "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/gk9m674wcx-1.zip",
            "filename": "gk9m674wcx-1.zip"
        }
    ]
    
    downloaded_files = []
    
    try:
        # Download all datasets
        print("\n DOWNLOADING PHASE")
        print("-" * 30)
        
        for dataset in tqdm(datasets, desc="Downloading datasets", position=0):
            if not os.path.exists(dataset["filename"]):
                download_file(dataset["url"], dataset["filename"])
            else:
                print(f" Already exists: {dataset['filename']}")
            downloaded_files.append(dataset["filename"])
        
        # Extract all datasets
        print("\n EXTRACTION PHASE")
        print("-" * 30)
        
        for filename in tqdm(downloaded_files, desc="Extracting datasets", position=0):
            if os.path.exists(filename):
                extract_zip(filename)
                
                # Special handling for nested zip files
                if filename == "gk9m674wcx-1.zip":
                    azt1d_path = "AZT1D  A Real-World Dataset for Type 1 Diabetes/AZT1D 2025.zip"
                    if os.path.exists(azt1d_path):
                        print(f"\n Found nested zip file: {azt1d_path}")
                        extract_zip(azt1d_path)
                        downloaded_files.append(azt1d_path)
        
        # Clean up zip files
        print("\n CLEANUP PHASE")
        print("-" * 30)
        
        for filename in downloaded_files:
            if os.path.exists(filename) and filename.endswith('.zip'):
                os.remove(filename)
                print(f"âœ“ Deleted: {filename}")
        
        print("\n DOWNLOAD AND EXTRACTION COMPLETE!")
        return working_dir
        
    except Exception as e:
        print(f" Error in dataset preparation: {e}")
        os.chdir(original_dir)
        return None
    finally:
        # Return to original directory
        os.chdir(original_dir)

# --- 1. Global Configuration and Memory Management ---
MEMORY_LIMIT_GB = 6.0
SAVED_MODEL_PATH = 'best_diabetes_model.keras'
SAVED_PREPROCESSOR_PATH = 'preprocessor.joblib'
BASE_PATHS = [
     #'diabetes_data',
    '/kaggle/working'
]

RANDOM_STATE = 42

def memory_usage_mb():
    """Returns current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def aggressive_cleanup():
    """Forces garbage collection and prints memory usage."""
    gc.collect()
    time.sleep(1)
    print(f"Memory after cleanup: {memory_usage_mb():.2f} MB")

def check_memory_limit(process_name=""):
    """Checks if memory usage exceeds the defined limit."""
    current_memory = memory_usage_mb()
    if current_memory / 1024 > MEMORY_LIMIT_GB:
        raise MemoryError(f"Memory usage for {process_name} exceeds the set limit of {MEMORY_LIMIT_GB} GB. Current: {current_memory/1024:.2f} GB")

def identify_diabetes_features(df):
    """
    Identifies potential diabetes-related features based on common column names.
    """
    potential_columns = [
        'glucose', 'blood_glucose', 'cgm', 'bmi', 'insulin',
        'carbs', 'carbohydrates', 'activity', 'exercise', 'sleep',
        'heart_rate', 'blood_pressure', 'age', 'gender', 'time',
        'timestamp', 'value', 'level', 'units', 'bolus', 'basal'
    ]
    
    # Get all numeric and relevant categorical columns
    found_features = []
    for col in df.columns:
        col_lower = col.lower()
        # Check if column name contains diabetes-related keywords
        if any(keyword in col_lower for keyword in potential_columns):
            found_features.append(col)
        # Also include numeric columns that might be relevant
        elif df[col].dtype in ['int64', 'float64'] and col not in ['source_file', 'source_directory']:
            found_features.append(col)
    
    # Remove metadata columns
    metadata_cols = ['source_file', 'source_directory', 'is_outlier']
    found_features = [col for col in found_features if col not in metadata_cols]
    
    return found_features

def create_synthetic_target(df):
    """
    Creates a synthetic target variable for demonstration purposes
    based on available features in the dataset.
    """
    print("Creating synthetic 'outcome' target variable...")
    
    # Find numeric columns that could be used to create a target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        print("No numeric columns found for creating synthetic target.")
        return df
    
    # Create a simple rule-based target
    np.random.seed(RANDOM_STATE)
    
    # If we have glucose-related data, use that
    glucose_cols = [col for col in df.columns if 'glucose' in col.lower() or 'cgm' in col.lower()]
    if glucose_cols:
        glucose_col = glucose_cols[0]
        # Create target based on glucose levels (simplified)
        df['outcome'] = (df[glucose_col] > df[glucose_col].quantile(0.7)).astype(int)
        print(f"Created target based on {glucose_col} column (high glucose = 1)")
    else:
        # Create random target with some bias towards features
        target_proba = np.random.random(len(df))
        # Add some correlation with the first numeric column if available
        if numeric_cols:
            first_numeric = df[numeric_cols[0]].fillna(df[numeric_cols[0]].median())
            normalized_vals = (first_numeric - first_numeric.min()) / (first_numeric.max() - first_numeric.min())
            target_proba = 0.7 * target_proba + 0.3 * normalized_vals
        
        df['outcome'] = (target_proba > 0.5).astype(int)
        print("Created random synthetic target variable")
    
    print(f"Target distribution: {df['outcome'].value_counts().to_dict()}")
    return df

def discover_csv_files_recursive(base_paths):
    """
    Recursively discovers all CSV files from multiple base paths.
    """
    print("=== Enhanced Recursive CSV File Discovery ===")
    csv_files_info = []
    total_directories_scanned = 0
    
    for base_path_idx, base_path in enumerate(base_paths, 1):
        print(f"\n[Base Path {base_path_idx}] Scanning: {base_path}")
        
        # Check if base path exists
        if not os.path.exists(base_path):
            print(f"   Base path does not exist: {base_path}")
            continue
            
        if not os.path.isdir(base_path):
            print(f"   Base path is not a directory: {base_path}")
            continue
            
        print(f"   Base path exists and is accessible")
        
        # Recursive directory traversal
        path_csv_count = 0
        path_directories_scanned = 0
        
        for root, dirs, files in os.walk(base_path):
            path_directories_scanned += 1
            total_directories_scanned += 1
            
            # Filter CSV files (exclude hidden files)
            csv_files_in_dir = [f for f in files if f.lower().endswith('.csv') and not f.startswith('.')]
            
            if csv_files_in_dir:
                relative_path = os.path.relpath(root, base_path)
                print(f"     Directory: {relative_path if relative_path != '.' else '<root>'}")
                print(f"       Found {len(csv_files_in_dir)} CSV file(s)")
                
                for csv_file in csv_files_in_dir:
                    full_path = os.path.join(root, csv_file)
                    file_size = os.path.getsize(full_path) if os.path.exists(full_path) else 0
                    
                    csv_info = {
                        'file_path': full_path,
                        'file_name': csv_file,
                        'directory': root,
                        'relative_directory': relative_path,
                        'base_path_index': base_path_idx,
                        'base_path': base_path,
                        'file_size_bytes': file_size,
                        'file_size_mb': file_size / (1024 * 1024)
                    }
                    
                    csv_files_info.append(csv_info)
                    path_csv_count += 1
                    
                    print(f"         - {csv_file} ({file_size / (1024 * 1024):.2f} MB)")
        
        print(f"   Summary for Base Path {base_path_idx}:")
        print(f"      Directories scanned: {path_directories_scanned}")
        print(f"      CSV files found: {path_csv_count}")
    
    # Overall summary
    print(f"\n=== Discovery Summary ===")
    print(f"Total base paths processed: {len(BASE_PATHS)}")
    print(f"Total directories scanned: {total_directories_scanned}")
    print(f"Total CSV files discovered: {len(csv_files_info)}")
    
    if csv_files_info:
        total_size_mb = sum(info['file_size_mb'] for info in csv_files_info)
        print(f"Total data size: {total_size_mb:.2f} MB")
    
    return csv_files_info

def load_csv_with_fallback_encoding(file_path, file_info):
    """
    Attempts to load CSV with multiple encoding and separator combinations.
    """
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
    separators = [',', ';', '\t', '|']
    
    for encoding in encodings:
        for separator in separators:
            try:
                df = pd.read_csv(file_path, encoding=encoding, sep=separator, low_memory=False)
                
                # Validate that we have a meaningful DataFrame
                if len(df) > 0 and len(df.columns) > 1:
                    # Additional validation: check if we have reasonable data
                    non_null_ratio = df.notna().sum().sum() / (len(df) * len(df.columns))
                    if non_null_ratio > 0.1:  # At least 10% non-null data
                        return df, encoding, separator
                        
            except Exception as e:
                continue
    
    return None, None, None

# --- 2. Enhanced Data Loading and Preparation ---
def load_real_data():
    """
    Enhanced data loading with multi-path recursive CSV discovery.
    """
    print("=== Enhanced Data Loading from Multiple Diabetes Study Sources ===")
    
    # Discover all CSV files recursively
    csv_files_info = discover_csv_files_recursive(BASE_PATHS)
    
    if not csv_files_info:
        print(" No CSV files found in any of the specified base paths.")
        print("Available directories in current working directory:")
        try:
            for item in os.listdir("."):
                item_path = os.path.join(".", item)
                if os.path.isdir(item_path):
                    print(f"   {item}")
        except Exception as e:
            print(f"Error listing directories: {e}")
        return None, None, None
    
    print(f"\n=== Loading CSV Files ===")
    all_dfs = []
    dataset_info = {}
    successful_loads = 0
    failed_loads = 0
    
    # Process each discovered CSV file with tqdm
    for idx, file_info in enumerate(tqdm(csv_files_info, desc=" Loading CSV files", position=0), 1):
        file_path = file_info['file_path']
        file_name = file_info['file_name']
        
        try:
            # Memory check before loading each file
            check_memory_limit(f"loading {file_name}")
            
            # Attempt to load CSV with fallback encoding
            df, encoding, separator = load_csv_with_fallback_encoding(file_path, file_info)
            
            if df is not None:
                # Add comprehensive metadata
                df['source_file'] = file_name
                df['source_directory'] = os.path.basename(file_info['directory'])
                df['source_base_path_index'] = file_info['base_path_index']
                df['source_relative_path'] = file_info['relative_directory']
                
                # Clean column names
                df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
                
                all_dfs.append(df)
                successful_loads += 1
                
                # Store comprehensive dataset info
                dataset_info[file_path] = {
                    'file_info': file_info,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                    'column_names': list(df.columns),
                    'encoding_used': encoding,
                    'separator_used': separator,
                    'load_success': True
                }
                
            else:
                failed_loads += 1
                dataset_info[file_path] = {
                    'file_info': file_info,
                    'load_success': False,
                    'error': 'Unable to parse with any encoding/separator combination'
                }
                
        except Exception as e:
            failed_loads += 1
            dataset_info[file_path] = {
                'file_info': file_info,
                'load_success': False,
                'error': str(e)
            }
    
    print(f"\n=== Loading Results ===")
    print(f"Successfully loaded: {successful_loads} files")
    print(f"Failed to load: {failed_loads} files")
    print(f"Success rate: {(successful_loads / len(csv_files_info) * 100):.1f}%")
    
    if not all_dfs:
        print(" No valid CSV files could be loaded. Cannot proceed.")
        return None, None, None
    
    # Combine all dataframes with progress bar
    print(f"\n=== Combining DataFrames ===")
    print(f"Combining {len(all_dfs)} dataframes...")
    
    combined_df = pd.concat(all_dfs, ignore_index=True, sort=False)
    
    # Comprehensive dataset summary
    print(f"\n=== Final Dataset Summary ===")
    print(f"Total rows: {len(combined_df):,}")
    print(f"Total columns: {len(combined_df.columns)}")
    print(f"Memory usage: {combined_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Column analysis
    print(f"\n=== Column Analysis ===")
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    categorical_cols = combined_df.select_dtypes(include=['object']).columns
    datetime_cols = combined_df.select_dtypes(include=['datetime']).columns
    
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"Datetime columns: {len(datetime_cols)}")
    
    # Create synthetic target if none exists
    if 'outcome' not in combined_df.columns:
        combined_df = create_synthetic_target(combined_df)
    
    # Identify potential features
    potential_features = identify_diabetes_features(combined_df)
    print(f"\n=== Feature Identification ===")
    print(f"Identified {len(potential_features)} potential diabetes-related features")
    
    return combined_df, potential_features, dataset_info

# --- 3. Enhanced EDA and Preprocessing ---
def perform_enhanced_eda_and_preprocess(df):
    """
    Performs comprehensive EDA with enhanced visualizations and creates preprocessing pipeline.
    """
    print("\n--- Performing Enhanced EDA and Preprocessing ---")
    
    # Create EDA plots directory
    Path("eda_plots").mkdir(exist_ok=True)
    
    # Get numeric and categorical columns (excluding metadata)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove metadata columns
    metadata_cols = ['source_file', 'source_directory', 'source_base_path_index', 'source_relative_path', 'is_outlier']
    numerical_cols = [col for col in numerical_cols if col not in metadata_cols and col != 'outcome']
    categorical_cols = [col for col in categorical_cols if col not in metadata_cols]
    
    print(f" EDA Progress: Found {len(numerical_cols)} numeric and {len(categorical_cols)} categorical features")
    
    # 3.1. Target Distribution
    if 'outcome' in df.columns:
        plt.figure(figsize=(8, 6))
        outcome_counts = df['outcome'].value_counts()
        bars = plt.bar(outcome_counts.index, outcome_counts.values, color=['lightcoral', 'lightblue'])
        plt.title('Target Variable Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Outcome')
        plt.ylabel('Count')
        
        # Add value labels on bars
        for bar, count in zip(bars, outcome_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(outcome_counts.values), 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('eda_plots/outcome_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(" Saved and displayed outcome distribution plot")

    # 3.2. Missing Values Analysis
    if len(numerical_cols) > 0 or len(categorical_cols) > 0:
        all_feature_cols = numerical_cols + categorical_cols
        missing_data = df[all_feature_cols].isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            plt.figure(figsize=(12, 8))
            missing_data.head(20).plot(kind='bar')
            plt.title('Missing Values by Feature (Top 20)', fontsize=16, fontweight='bold')
            plt.xlabel('Features')
            plt.ylabel('Missing Values Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('eda_plots/missing_values.png', dpi=300, bbox_inches='tight')
            plt.show()
            print(" Saved and displayed missing values analysis")
        else:
            print(" No missing values found in feature columns")

    # 3.3. Correlation Heatmap for numeric columns
    if len(numerical_cols) > 1:
        numerical_cols_subset = numerical_cols[:15]  # Limit to avoid overcrowded plot
        
        plt.figure(figsize=(14, 12))
        correlation_data = df[numerical_cols_subset].corr()
        mask = np.triu(np.ones_like(correlation_data))
        
        sns.heatmap(correlation_data, annot=True, cmap='RdBu_r', center=0, 
                   square=True, mask=mask, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('eda_plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("âœ“ Saved and displayed correlation heatmap")

    # 3.4. Distribution plots for key numeric features
    if len(numerical_cols) > 0:
        n_features = min(6, len(numerical_cols))
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols[:n_features]):
            df[col].hist(bins=50, ax=axes[i], alpha=0.7, color='steelblue')
            axes[i].set_title(f'Distribution of {col}', fontweight='bold')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
            
        plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('eda_plots/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(" Saved and displayed feature distribution plots")

    # 3.5. Box plots for outlier detection
    if len(numerical_cols) > 0 and 'outcome' in df.columns:
        n_features = min(4, len(numerical_cols))
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols[:n_features]):
            df.boxplot(column=col, by='outcome', ax=axes[i])
            axes[i].set_title(f'{col} by Outcome')
            axes[i].set_xlabel('Outcome')
            axes[i].set_ylabel(col)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
            
        plt.suptitle('Feature Distributions by Outcome', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('eda_plots/boxplots_by_outcome.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("âœ“ Saved and displayed box plots by outcome")

    # 3.6. Enhanced Outlier Detection
    if len(numerical_cols) > 0:
        print(" Detecting outliers using Local Outlier Factor...")
        lof = LocalOutlierFactor(n_neighbors=min(20, len(df)//2), contamination='auto')
        
        # Use subset for outlier detection to manage memory
        sample_size = min(5000, len(df))
        sample_indices = np.random.choice(df.index, sample_size, replace=False)
        numerical_data = df.loc[sample_indices, numerical_cols].dropna()
        
        if len(numerical_data) > 10:
            with tqdm(total=1, desc=" Outlier detection", position=0) as pbar:
                outlier_scores = lof.fit_predict(numerical_data)
                df.loc[numerical_data.index, 'is_outlier'] = outlier_scores
                pbar.update(1)
            
            n_outliers = len(df[df['is_outlier'] == -1])
            print(f"âœ“ Detected {n_outliers} outliers ({n_outliers/len(df)*100:.2f}% of data)")
        else:
            df['is_outlier'] = 1
            print("âœ“ Insufficient data for outlier detection")

    # 3.7. Preprocessing Pipeline Creation
    numeric_features = []
    categorical_features = []
    
    for col in df.columns:
        if col in ['outcome', 'is_outlier'] + metadata_cols:
            continue
            
        if df[col].dtype in ['int64', 'float64']:
            numeric_features.append(col)
        elif df[col].dtype == 'object':
            if df[col].nunique() <= 50:  # Reasonable threshold for categorical
                categorical_features.append(col)

    print(f" Preprocessing: {len(numeric_features)} numeric, {len(categorical_features)} categorical features")

    # Create preprocessing pipelines
    transformers = []
    
    if numeric_features:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_features))

    if not transformers:
        raise ValueError("No valid features found for preprocessing")

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    
    print(" Preprocessing pipeline created successfully")
    return preprocessor, numeric_features + categorical_features

# --- 4. Enhanced Model Training with Overfitting Protection ---
def create_nn_model_with_regularization(input_shape):
    """Creates a neural network with enhanced regularization to prevent overfitting."""
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),  # Increased dropout
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),  # Increased dropout
        Dense(32, activation='relu'),  # Additional layer
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile with lower learning rate to prevent overfitting
    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Reduced learning rate
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def train_models_with_cross_validation(X, y, preprocessor, feature_names):
    """
    Trains multiple models with comprehensive cross-validation and evaluation.
    """
    print("\n=== Enhanced Model Training with Cross-Validation ===")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Target distribution in training: {np.bincount(y_train)}")
    
    # Fit preprocessor on training data
    print(" Fitting preprocessor...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Processed feature shape: {X_train_processed.shape}")
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,  # Limit depth to prevent overfitting
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            C=1.0,  # Regularization
            class_weight='balanced'
        )
    }
    
    # Cross-validation setup
    cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    
    # Train and evaluate traditional ML models
    for name, model in tqdm(models.items(), desc=" Training ML models"):
        print(f"\n Training {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_processed, y_train, 
                                  cv=cv_folds, scoring='accuracy', n_jobs=-1)
        
        # Train on full training set
        model.fit(X_train_processed, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train_processed)
        y_test_pred = model.predict(X_test_processed)
        
        # Calculate metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'test_f1': f1_score(y_test, y_test_pred),
            'test_precision': precision_score(y_test, y_test_pred),
            'test_recall': recall_score(y_test, y_test_pred),
            'y_test_pred': y_test_pred,
            'overfitting_gap': train_acc - test_acc
        }
        
        print(f"   CV Accuracy: {cv_scores.mean():.4f} (Â±{cv_scores.std()*2:.4f})")
        print(f"   Test Accuracy: {test_acc:.4f}")
        print(f"   Overfitting Gap: {train_acc - test_acc:.4f}")
    
    # Train Neural Network
    print(f"\n Training Neural Network...")
    
    # Create and compile model
    nn_model = create_nn_model_with_regularization(X_train_processed.shape[1])
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7)
    ]
    
    # Split training data for validation
    X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
        X_train_processed, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )
    
    # Train neural network
    history = nn_model.fit(
        X_train_nn, y_train_nn,
        validation_data=(X_val_nn, y_val_nn),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )
    
    # Neural network predictions
    y_train_pred_nn = (nn_model.predict(X_train_processed) > 0.5).astype(int).flatten()
    y_test_pred_nn = (nn_model.predict(X_test_processed) > 0.5).astype(int).flatten()
    
    # Neural network metrics
    train_acc_nn = accuracy_score(y_train, y_train_pred_nn)
    test_acc_nn = accuracy_score(y_test, y_test_pred_nn)
    
    results['Neural Network'] = {
        'model': nn_model,
        'history': history,
        'train_accuracy': train_acc_nn,
        'test_accuracy': test_acc_nn,
        'test_f1': f1_score(y_test, y_test_pred_nn),
        'test_precision': precision_score(y_test, y_test_pred_nn),
        'test_recall': recall_score(y_test, y_test_pred_nn),
        'y_test_pred': y_test_pred_nn,
        'overfitting_gap': train_acc_nn - test_acc_nn
    }
    
    print(f"   Test Accuracy: {test_acc_nn:.4f}")
    print(f"   Overfitting Gap: {train_acc_nn - test_acc_nn:.4f}")
    
    return results, X_test, y_test, X_test_processed

# --- 5. Enhanced Evaluation and Visualization ---
def evaluate_and_visualize_models(results, X_test, y_test, X_test_processed):
    """
    Comprehensive model evaluation with enhanced visualizations.
    """
    print("\n=== Model Evaluation and Visualization ===")
    
    # Create evaluation plots directory
    Path("model_evaluation").mkdir(exist_ok=True)
    
    # 5.1. Model Comparison Table
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Model': name,
            'Test Accuracy': f"{result['test_accuracy']:.4f}",
            'F1 Score': f"{result['test_f1']:.4f}",
            'Precision': f"{result['test_precision']:.4f}",
            'Recall': f"{result['test_recall']:.4f}",
            'Overfitting Gap': f"{result['overfitting_gap']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nğŸ“Š Model Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # 5.2. Performance Bar Chart
    plt.figure(figsize=(14, 8))
    metrics = ['test_accuracy', 'test_f1', 'test_precision', 'test_recall']
    model_names = list(results.keys())
    
    x = np.arange(len(model_names))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[name][metric] for name in model_names]
        plt.bar(x + i*width, values, width, label=metric.replace('test_', '').title())
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.xticks(x + width*1.5, model_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_evaluation/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5.3. Confusion Matrices
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
    if len(results) == 1:
        axes = [axes]
    
    for idx, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['y_test_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{name}\nConfusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('model_evaluation/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5.4. ROC Curves
    plt.figure(figsize=(10, 8))
    
    for name, result in results.items():
        if name == 'Neural Network':
            y_score = result['model'].predict(X_test_processed).flatten()
        else:
            y_score = result['model'].predict_proba(X_test_processed)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, linewidth=2, 
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_evaluation/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5.5. Neural Network Training History (if available)
    if 'Neural Network' in results and 'history' in results['Neural Network']:
        history = results['Neural Network']['history']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Neural Network Training Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Neural Network Training Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_evaluation/nn_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
    best_model = results[best_model_name]
    
    print(f"\n Best Model: {best_model_name}")
    print(f"   Test Accuracy: {best_model['test_accuracy']:.4f}")
    print(f"   F1 Score: {best_model['test_f1']:.4f}")
    print(f"   Overfitting Gap: {best_model['overfitting_gap']:.4f}")
    
    return best_model_name, best_model

# --- 6. Model Saving and Loading ---
def save_best_model(best_model_name, best_model, preprocessor):
    """
    Saves the best model and preprocessor.
    """
    print(f"\n=== Saving Best Model: {best_model_name} ===")
    
    try:
        # Save preprocessor
        dump(preprocessor, SAVED_PREPROCESSOR_PATH)
        print(f"âœ“ Preprocessor saved to {SAVED_PREPROCESSOR_PATH}")
        
        # Save model based on type
        if best_model_name == 'Neural Network':
            best_model['model'].save(SAVED_MODEL_PATH)
            print(f"âœ“ Neural Network model saved to {SAVED_MODEL_PATH}")
        else:
            # Save sklearn model
            model_path = f'best_{best_model_name.lower().replace(" ", "_")}_model.joblib'
            dump(best_model['model'], model_path)
            print(f"âœ“ {best_model_name} model saved to {model_path}")
            
    except Exception as e:
        print(f" Error saving model: {e}")

def load_saved_model():
    """
    Loads the saved model and preprocessor.
    """
    try:
        preprocessor = load(SAVED_PREPROCESSOR_PATH)
        
        if os.path.exists(SAVED_MODEL_PATH):
            model = tf.keras.models.load_model(SAVED_MODEL_PATH)
            model_type = 'Neural Network'
        else:
            # Try loading sklearn models
            sklearn_models = [f for f in os.listdir('.') if f.endswith('_model.joblib')]
            if sklearn_models:
                model_path = sklearn_models[0]
                model = load(model_path)
                model_type = model_path.replace('best_', '').replace('_model.joblib', '').replace('_', ' ').title()
            else:
                raise FileNotFoundError("No saved model found")
        
        print(f"âœ“ Loaded {model_type} model and preprocessor successfully")
        return model, preprocessor, model_type
        
    except Exception as e:
        print(f" Error loading model: {e}")
        return None, None, None

# --- 7. Prediction Function ---
def make_predictions(model, preprocessor, model_type, X_new):
    """
    Makes predictions on new data.
    """
    try:
        # Preprocess the new data
        X_processed = preprocessor.transform(X_new)
        
        # Make predictions based on model type
        if model_type == 'Neural Network':
            predictions = (model.predict(X_processed) > 0.5).astype(int).flatten()
            probabilities = model.predict(X_processed).flatten()
        else:
            predictions = model.predict(X_processed)
            probabilities = model.predict_proba(X_processed)[:, 1]
        
        return predictions, probabilities
        
    except Exception as e:
        print(f" Error making predictions: {e}")
        return None, None

def create_submission_file(predictions, test_ids, filename='submission.csv'):
    """
    Creates a submission file in the specified format.
    """
    print(f"\n=== Creating Submission File: {filename} ===")
    
    # Ensure test_ids are available
    if test_ids is None or len(test_ids) != len(predictions):
        print("Error: Test IDs are not available or do not match prediction count. Cannot create submission file.")
        return
        
    submission_df = pd.DataFrame({
        'id': test_ids,
        'prediction': predictions
    })
    
    submission_df.to_csv(filename, index=False)
    print(f"âœ“ Submission file created successfully at {filename}")

def preview_submission(filename='submission.csv'):
    """
    Reads the submission file, prints a preview, and provides a summary.
    """
    print(f"\n=== Previewing Submission File: {filename} ===")
    try:
        submission_df = pd.read_csv(filename)
        print("First 5 rows of the submission file:")
        print(submission_df.head().to_string(index=False))
        
        print("\nPrediction Summary:")
        prediction_counts = submission_df['prediction'].value_counts().to_dict()
        print(f"  Class 0 (No Diabetes Risk): {prediction_counts.get(0, 0)} samples")
        print(f"  Class 1 (High Diabetes Risk): {prediction_counts.get(1, 0)} samples")
        
        print("\nâœ“ Submission file preview complete.")
    except FileNotFoundError:
        print(f"Error: Submission file '{filename}' not found.")
    except Exception as e:
        print(f"Error reading submission file: {e}")
        

# --- 8. Main Execution Function ---
def main():
    """
    Main execution function that orchestrates the entire pipeline.
    """
    print(" COMPREHENSIVE DIABETES PREDICTION SYSTEM")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # Step 0: Download and prepare datasets (optional)
        print("\n STEP 0: Dataset Preparation")
        download_success = download_and_prepare_datasets()
        if download_success:
            print(f" Datasets prepared successfully in {download_success}")
        else:
            print("  Dataset download failed, will use any existing local data")
        
        # Step 1: Load Data
        print("\n STEP 1: Data Loading")
        df, feature_names, dataset_info = load_real_data()
        
        if df is None:
            print(" No data could be loaded. Exiting.")
            return
        
        aggressive_cleanup()
        
        # Step 2: EDA and Preprocessing
        print("\n STEP 2: EDA and Preprocessing")
        preprocessor, selected_features = perform_enhanced_eda_and_preprocess(df)
        
        # Prepare features and target
        if not selected_features:
            print(" No valid features found. Exiting.")
            return
        
        # Add an ID column for the submission file
        df['id'] = np.arange(len(df))
        
        X = df[selected_features]
        y = df['outcome']
        
        print(f" Features prepared: {X.shape}")
        print(f" Target prepared: {len(y)} samples")
        
        aggressive_cleanup()
        
        # Step 3: Model Training
        print("\n STEP 3: Model Training")
        results, X_test, y_test, X_test_processed = train_models_with_cross_validation(
            X, y, preprocessor, selected_features
        )
        
        # Find best model
        best_model_name, best_model = evaluate_and_visualize_models(
            results, X_test, y_test, X_test_processed
        )
        
        # Step 4: Create Submission File
        print("\n STEP 4: Creating Submission File")
        
        # Use predictions from the best model
        predictions = results[best_model_name]['y_test_pred']
        
        # Get the corresponding IDs for the test set
        test_ids = X_test.index.values
        
        create_submission_file(predictions, test_ids)
        
        # Step 5: Preview Submission File
        print("\n STEP 5: Previewing Submission")
        preview_submission()
        
        # Step 6: Save Best Model
        print("\n STEP 6: Model Saving")
        save_best_model(best_model_name, best_model, preprocessor)
        
        # Step 7: Demo Prediction
        print("\n STEP 7: Demo Prediction")
        sample_data = X_test.head(5)
        predictions, probabilities = make_predictions(
            best_model['model'], preprocessor, best_model_name, sample_data
        )
        
        if predictions is not None:
            print("Sample Predictions:")
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                print(f"  Sample {i+1}: Prediction = {pred}, Probability = {prob:.3f}")
        
        # Final Summary
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "=" * 80)
        print(" PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"  Total execution time: {total_time/60:.2f} minutes")
        print(f" Dataset size: {len(df):,} rows, {len(df.columns)} columns")
        print(f" Best model: {best_model_name} (Accuracy: {best_model['test_accuracy']:.4f})")
        print(f" Model saved and ready for deployment")
        print(f" Results saved in: eda_plots/, model_evaluation/")
        
    except Exception as e:
        print(f"\n Pipeline failed with error: {e}")
        print(" Check your data paths and dependencies")
        
    finally:
        aggressive_cleanup()

# --- 9. Entry Point ---
if __name__ == "__main__":
    main()




