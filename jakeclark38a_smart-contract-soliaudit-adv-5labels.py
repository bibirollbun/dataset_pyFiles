import time
import socket
import os

def is_connected(host="8.8.8.8", port=53, timeout=3):
    """Check internet connectivity by attempting to connect to a public DNS server."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False

print("Checking internet connection...")
while not is_connected():
    print("No internet connection. Retrying in 1 second...")
    time.sleep(1)

print("âœ… Internet connection is active.")

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("âœ… Environment configured successfully.")


import subprocess
import os

def install_uv():
    """Install uv package manager and configure PATH"""
    try:
        # Install uv using the official installer
        result = subprocess.run(
            ["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
            capture_output=True, text=True, check=True
        )
        
        # Add uv to PATH for current session
        uv_path = os.path.expanduser("~/.local/bin")
        if uv_path not in os.environ["PATH"]:
            os.environ["PATH"] = f"{uv_path}:{os.environ['PATH']}"
        
        # Verify installation
        version_result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if version_result.returncode == 0:
            print(f"âœ… uv installed successfully: {version_result.stdout.strip()}")
            return True
        else:
            print("â�Œ uv installation verification failed")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"â�Œ Failed to install uv: {e}")
        return False

# Install uv package manager
uv_installed = install_uv()


import os

def detect_environment():
    """Detect execution environment and configure settings accordingly"""
    if os.environ.get('KAGGLE_URL_BASE') or os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        return "Kaggle"
    elif 'google.colab' in str(get_ipython()):
        return "Colab"
    else:
        return "Local"

env_type = detect_environment()
print(f"ğŸŒ� Environment detected: {env_type}")

# Configure environment-specific settings
if env_type == "Local":
    batch_size = 20
    num_processors = 1
    train_batch_size = 1
    epochs = 3  # Smaller for local 
    maximal_text_length = 510
else:
    batch_size = 64
    num_processors = 4
    train_batch_size = 20
    epochs = 10
    maximal_text_length = None

# Set model configuration
transformers_model = "distilbert/distilbert-base-uncased"
TOKENIZE_TRUNCATE = None  # Use None for BELT, True for truncated tokenization
TRAINING_FRAMEWORK = "belt"  # Options: "belt", "pytorch", "tensorflow"

print(f"ğŸ“Š Batch size: {batch_size}")
print(f"ğŸ”§ Processors: {num_processors}")
print(f"ğŸ�ƒ Training batch size: {train_batch_size}")
print(f"ğŸ”„ Epochs: {epochs}")
print(f"ğŸ¤– Model: {transformers_model}")
print(f"ğŸš€ Training framework: {TRAINING_FRAMEWORK}")


import subprocess
import sys

def check_package_installed(package):
    """Check if a package is already installed"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def install_packages_with_uv():
    """Install packages using uv with multi-threading benefits"""
    # Core packages (always needed)
    packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "sklearn": "scikit-learn",
        "spektral": "spektral",
        "xgboost": "xgboost",
        "nltk": "nltk",
        "transformers": "transformers",
        "datasets": "datasets",
        "huggingface_hub": "huggingface_hub"
    }
    
    # Framework-specific packages based on TRAINING_FRAMEWORK
    if TRAINING_FRAMEWORK == "belt":
        packages["belt_nlp"] = "belt_nlp"
        packages["torch"] = "torch"  # BELT requires PyTorch
        print(f"ğŸ�¯ BELT framework selected - including belt_nlp and torch")
    elif TRAINING_FRAMEWORK == "pytorch":
        packages["torch"] = "torch"
        print(f"ğŸ�¯ PyTorch framework selected - including torch")
    elif TRAINING_FRAMEWORK == "tensorflow":
        packages["tensorflow"] = "tensorflow"
        print(f"ğŸ�¯ TensorFlow framework selected - including tensorflow")
    else:
        # Install both for safety if framework is unknown
        packages["torch"] = "torch"
        packages["tensorflow"] = "tensorflow"
        packages["belt_nlp"] = "belt_nlp"
        print(f"âš ï¸� Unknown framework '{TRAINING_FRAMEWORK}' - installing all ML packages")
    
    # Check which packages need installation
    packages_to_install = []
    for import_name, install_name in packages.items():
        if not check_package_installed(import_name):
            print(f"ğŸ“¦ {install_name} not found, will install...")
            packages_to_install.append(install_name)
        else:
            print(f"âœ… {install_name} already installed")
    
    if not packages_to_install:
        print("ğŸ�‰ All packages are already installed!")
        return True
    
    print(f"\nğŸš€ Installing {len(packages_to_install)} packages with uv...")
    print(f"ğŸ“¦ Packages: {', '.join(packages_to_install)}")
    
    # Install all packages at once using uv for better performance
    if uv_installed:
        try:
            cmd = ["uv", "pip", "install"] + packages_to_install + ["--quiet"]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("âœ… All packages installed successfully with uv!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"âš ï¸� uv installation failed: {e}")
            print("ğŸ”„ Falling back to pip...")
    
    # Fallback to pip if uv fails or is not available
    print("ğŸ�� Installing packages with pip...")
    for package in packages_to_install:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
            print(f"âœ… {package} installed with pip")
        except subprocess.CalledProcessError as e:
            print(f"â�Œ Failed to install {package}: {e}")
            return False
    
    print("ğŸ�‰ Package installation completed!")
    return True

# Install all required packages
install_success = install_packages_with_uv()


import nltk
from nltk.corpus import stopwords
import os

# Download NLTK stopwords
try:
    nltk.download('stopwords', quiet=True)
    print("âœ… NLTK stopwords downloaded successfully")
except Exception as e:
    print(f"âš ï¸� NLTK download warning: {e}")

# Additional configuration
os.environ["WANDB_DISABLED"] = "true"  # Disable Weights & Biases logging
print("âœ… Additional configurations applied")


import pickle
import hashlib
import os
from pathlib import Path
import json
import time

class SmartContractDatasetCache:
    """Intelligent caching system for smart contract dataset preprocessing"""
    
    def __init__(self, cache_dir="./caches"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.load_metadata()
    
    def load_metadata(self):
        """Load cache metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_file_hash(self, file_path):
        """Get hash of file for cache validation"""
        if not os.path.exists(file_path):
            return None
        
        # For large files, sample first and last chunks
        with open(file_path, 'rb') as f:
            # Read first 1MB
            first_chunk = f.read(1024 * 1024)
            f.seek(-min(1024 * 1024, f.tell()), 2)  # Last 1MB
            last_chunk = f.read()
            
        content = first_chunk + last_chunk
        return hashlib.md5(content).hexdigest()
    
    def get_cache_key(self, operation, params=None):
        """Generate cache key for operation"""
        key_data = f"{operation}_{params}" if params else operation
        return hashlib.md5(key_data.encode()).hexdigest()[:16]  # Shorter keys
    
    def is_cache_valid(self, cache_key, source_file=None, params_hash=None):
        """Check if cache is valid"""
        if cache_key not in self.metadata:
            return False
        
        cache_info = self.metadata[cache_key]
        
        # Check if cache file exists
        cache_file = Path(cache_info['cache_file'])
        if not cache_file.exists():
            return False
        
        # Check source file hash if provided
        if source_file:
            cached_hash = cache_info.get('source_hash')
            current_hash = self.get_file_hash(source_file)
            if cached_hash != current_hash:
                return False
        
        # Check parameters hash if provided
        if params_hash:
            cached_params = cache_info.get('params_hash')
            if cached_params != params_hash:
                return False
        
        return True
    
    def save_to_cache(self, cache_key, data, source_file=None, description="", params_hash=None):
        """Save data to cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Save data
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Update metadata
        self.metadata[cache_key] = {
            'source_hash': self.get_file_hash(source_file) if source_file else None,
            'cache_file': str(cache_file),
            'timestamp': time.time(),
            'description': description,
            'params_hash': params_hash,
            'size_mb': cache_file.stat().st_size / (1024 * 1024)
        }
        self.save_metadata()
        
        print(f"ğŸ’¾ Cached {description}: {cache_file.stat().st_size / (1024 * 1024):.1f}MB")
    
    def load_from_cache(self, cache_key):
        """Load data from cache"""
        if cache_key not in self.metadata:
            return None
        
        cache_file = Path(self.metadata[cache_key]['cache_file'])
        if not cache_file.exists():
            return None
        
        print(f"ğŸš€ Loading from cache: {self.metadata[cache_key]['description']}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    def clear_cache(self, pattern=None):
        """Clear cache files"""
        if pattern:
            # Clear specific pattern
            for key, info in list(self.metadata.items()):
                if pattern in info['description']:
                    cache_file = Path(info['cache_file'])
                    if cache_file.exists():
                        cache_file.unlink()
                    del self.metadata[key]
        else:
            # Clear all
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            self.metadata = {}
        
        self.save_metadata()
        print(f"ğŸ—‘ï¸� Cache cleared: {pattern if pattern else 'all'}")
    
    def get_cache_info(self):
        """Get cache information"""
        total_size = 0
        info = []
        
        for key, data in self.metadata.items():
            cache_file = Path(data['cache_file'])
            if cache_file.exists():
                size_mb = data.get('size_mb', 0)
                total_size += size_mb
                info.append({
                    'description': data['description'],
                    'size_mb': size_mb,
                    'timestamp': time.ctime(data['timestamp'])
                })
        
        print(f"ğŸ“Š Cache Summary:")
        print(f"  Total size: {total_size:.1f}MB")
        print(f"  Cache files: {len(info)}")
        
        for item in info:
            print(f"  - {item['description']}: {item['size_mb']:.1f}MB ({item['timestamp']})")
        
        return info

# Initialize cache system
cache_system = SmartContractDatasetCache()
print("âœ… Cache system initialized")
print(f"ğŸ“� Cache directory: {cache_system.cache_dir}")

# Show existing cache info
cache_system.get_cache_info()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configure pandas display options
pd.set_option('display.max_colwidth', 100)

def load_dataset_with_cache():
    """Load dataset with intelligent caching"""
    
    # Determine dataset path based on environment
    if env_type == "Kaggle":
        dataset_path = '/kaggle/input/soliaudit-va-dataset-sourcecode/SoliAudit-VA-Dataset-SourceCode.csv'
    elif env_type == "Colab":
        dataset_path = '/content/drive/MyDrive/Colab Notebooks/input/SoliAudit-VA-Dataset-SourceCode.csv'
    else:
        # Try multiple local paths
        possible_paths = [
            './datasets/SoliAudit-VA-Dataset-SourceCode.csv',
            './datasets/SoliAudit-VA-Dataset-Full-SourceCode.csv',
            './datasets/SoliAudit-VA-Dataset.csv'
        ]
        dataset_path = None
        for path in possible_paths:
            if os.path.exists(path):
                dataset_path = path
                break
        
        if not dataset_path:
            print(f"â�Œ Dataset not found in any of these locations:")
            for path in possible_paths:
                print(f"  - {path}")
            return None
    
    print(f"ğŸ“‚ Dataset path: {dataset_path}")
    
    # Check cache first
    cache_key = cache_system.get_cache_key("raw_dataset", dataset_path)
    
    if cache_system.is_cache_valid(cache_key, dataset_path):
        print("ğŸš€ Loading dataset from cache...")
        return cache_system.load_from_cache(cache_key)
    
    # Load from CSV if cache miss
    try:
        print("ğŸ“Š Loading dataset from CSV...")
        df = pd.read_csv(dataset_path)
        print(f"âœ… Dataset loaded successfully from {dataset_path}")
        
        # Cache the loaded dataset
        cache_system.save_to_cache(
            cache_key, df, dataset_path, 
            f"Raw dataset ({df.shape[0]} rows, {df.shape[1]} cols)"
        )
        
        return df
        
    except FileNotFoundError:
        print(f"â�Œ Dataset not found at {dataset_path}")
        print("ğŸ“� Please ensure the SoliAudit dataset is available in the correct location")
        return None
    except Exception as e:
        print(f"â�Œ Error loading dataset: {e}")
        return None

# Load the dataset with caching
df = load_dataset_with_cache()

if df is not None:
    print(f"\nğŸ“Š Dataset Shape: {df.shape}")
    print(f"ğŸ“‹ Columns: {list(df.columns)}")
    
    # Display first few rows
    print("\nğŸ“‹ First 5 rows:")
    display(df.head())
    
    # Basic statistics
    print(f"\nğŸ“ˆ Basic Statistics:")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\nâš ï¸� Missing values found:")
        for col, missing in missing_values[missing_values > 0].items():
            print(f"  {col}: {missing} ({missing/len(df)*100:.1f}%)")
    else:
        print("\nâœ… No missing values found")
else:
    print("â�Œ Cannot proceed without dataset")


def preprocess_dataset_with_cache(df):
    """Preprocess dataset with caching"""
    if df is None:
        return None
    
    # Create parameters hash for cache validation
    preprocessing_params = {
        'remove_columns': ['CallDepth'],
        'essential_columns': ["Overflow", "Underflow", "AssertFail", "CheckEffects", 
                             "LowlevelCalls", "BlockTimestamp", "source_code"],
        'dataset_shape': df.shape
    }
    params_hash = hashlib.md5(str(preprocessing_params).encode()).hexdigest()
    
    # Check cache
    cache_key = cache_system.get_cache_key("preprocessing", "basic")
    
    if cache_system.is_cache_valid(cache_key, params_hash=params_hash):
        print("ğŸš€ Loading preprocessed dataset from cache...")
        return cache_system.load_from_cache(cache_key)
    
    print("ğŸ”§ Preprocessing dataset...")
    df_processed = df.copy()
    
    # Ensure source_code is string type and calculate word count
    df_processed['source_code'] = df_processed['source_code'].astype(str)
    df_processed['source_code_word_count'] = df_processed['source_code'].apply(lambda x: len(x.split()))
    
    # Remove extremely imbalanced columns
    columns_to_remove = ['CallDepth'] if 'CallDepth' in df_processed.columns else []
    if columns_to_remove:
        df_processed = df_processed.drop(columns=columns_to_remove)
        print(f"ğŸ—‘ï¸� Removed columns: {columns_to_remove}")
    
    # Keep essential columns for vulnerability analysis
    essential_columns = ["Overflow", "Underflow", "AssertFail", "CheckEffects", "LowlevelCalls", "BlockTimestamp", "source_code"]
    available_columns = [col for col in essential_columns if col in df_processed.columns]
    
    if len(available_columns) == len(essential_columns):
        df_processed = df_processed[available_columns]
        print(f"âœ… Dataset filtered to essential columns: {available_columns}")
    else:
        missing_cols = set(essential_columns) - set(available_columns)
        print(f"âš ï¸� Missing columns: {missing_cols}")
        print(f"ğŸ“‹ Available columns: {available_columns}")
        df_processed = df_processed[available_columns]
    
    # Add word count column back if not in essential columns
    if 'source_code_word_count' not in df_processed.columns:
        df_processed['source_code_word_count'] = df_processed['source_code'].apply(lambda x: len(x.split()))
    
    # Cache the preprocessed dataset
    cache_system.save_to_cache(
        cache_key, df_processed, 
        description=f"Preprocessed dataset ({df_processed.shape[0]} rows, {df_processed.shape[1]} cols)",
        params_hash=params_hash
    )
    
    return df_processed

# Data preprocessing with caching
if df is not None:
    df_processed = preprocess_dataset_with_cache(df)
    
    if df_processed is not None:
        # Display source code statistics
        print(f"\nğŸ“Š Source Code Statistics:")
        print(df_processed['source_code_word_count'].describe())
        
        # Plot word count distribution
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(df_processed['source_code_word_count'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.title('Source Code Word Count Distribution')
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        df_processed['source_code_word_count'].plot(kind='box', color='lightgreen')
        plt.ylabel('Word Count')
        plt.title('Source Code Word Count Box Plot')
        
        plt.tight_layout()
        plt.show()
        
        print(f"âœ… Data preprocessing completed")
    else:
        print("â�Œ Preprocessing failed")
else:
    print("â�Œ Skipping preprocessing - no dataset loaded")


# Configuration for DASP categories
USE_6_LABELS = False  # Set to False to use only 5 labels (excluding Access_Control)

def create_dasp_labels_with_cache(df, include_access_control=True):
    """
    Create merged DASP category labels from original vulnerability columns with caching
    Based on DASP vulnerability classification mapping
    
    Args:
        df: DataFrame with original vulnerability columns
        include_access_control: Whether to include Access_Control as 6th category
    
    Returns:
        DataFrame with DASP category columns added
    """
    if df is None:
        return None
    
    # Create parameters hash for cache validation
    dasp_params = {
        'include_access_control': include_access_control,
        'dataset_shape': df.shape,
        'columns': sorted(df.columns.tolist())
    }
    params_hash = hashlib.md5(str(dasp_params).encode()).hexdigest()
    
    # Check cache
    cache_key = cache_system.get_cache_key("dasp_labels", f"ac_{include_access_control}")
    
    if cache_system.is_cache_valid(cache_key, params_hash=params_hash):
        print("ğŸš€ Loading DASP labeled dataset from cache...")
        return cache_system.load_from_cache(cache_key)
    
    print("ğŸ�·ï¸� Creating DASP category mappings...")
    df_dasp = df.copy()
    
    # 1. Arithmetic (Integer Overflow and Underflow)
    # Maps: Overflow, Underflow -> Arithmetic
    df_dasp['Arithmetic'] = ((df_dasp.get('Overflow', 0) == 1) | 
                            (df_dasp.get('Underflow', 0) == 1)).astype(int)
    
    # 2. Unchecked Low-Level Calls
    # Maps: CallDepth, InlineAssembly, LowlevelCalls -> Unchecked_Low_Level_Calls
    df_dasp['Unchecked_Low_Level_Calls'] = (
        (df_dasp.get('LowlevelCalls', 0) == 1) |
        (df_dasp.get('CallDepth', 0) == 1) |
        (df_dasp.get('InlineAssembly', 0) == 1)
    ).astype(int)
    
    # 3. Time Manipulation
    # Maps: TOD, TimeDep, BlockTimestamp, BlockHash -> Time_Manipulation
    df_dasp['Time_Manipulation'] = (
        (df_dasp.get('BlockTimestamp', 0) == 1) |
        (df_dasp.get('TOD', 0) == 1) |
        (df_dasp.get('TimeDep', 0) == 1) |
        (df_dasp.get('BlockHash', 0) == 1)
    ).astype(int)
    
    # 4. DoS (Denial of Service)
    # Maps: AssertFail, SelfDestruct, GasLimit -> DoS
    df_dasp['DoS'] = (
        (df_dasp.get('AssertFail', 0) == 1) |
        (df_dasp.get('SelfDestruct', 0) == 1) |
        (df_dasp.get('GasLimit', 0) == 1)
    ).astype(int)
    
    # 5. Reentrancy
    # Maps: Reentracy, CheckEffects -> Reentrancy
    df_dasp['Reentrancy'] = (
        (df_dasp.get('Reentracy', 0) == 1) |
        (df_dasp.get('CheckEffects', 0) == 1)
    ).astype(int)
    
    # 6. Access Control (Optional - only if include_access_control=True)
    if include_access_control:
        # Maps: TxOrigin, Multisig -> Access_Control
        df_dasp['Access_Control'] = (
            (df_dasp.get('TxOrigin', 0) == 1) |
            (df_dasp.get('Multisig', 0) == 1)
        ).astype(int)
    
    # Create list of DASP categories for later use
    dasp_categories = ['Arithmetic', 'Unchecked_Low_Level_Calls', 'Time_Manipulation', 'DoS', 'Reentrancy']
    if include_access_control:
        dasp_categories.append('Access_Control')
    
    # Cache the results
    cache_system.save_to_cache(
        cache_key, df_dasp,
        description=f"DASP labeled dataset ({len(dasp_categories)} categories)",
        params_hash=params_hash
    )
    
    # Display mapping results
    print(f"âœ… DASP category mapping completed")
    print(f"ğŸ“Š Categories created: {dasp_categories}")
    
    # Show label distribution
    print(f"\nğŸ“ˆ DASP Label Distribution:")
    for category in dasp_categories:
        positive_count = df_dasp[category].sum()
        positive_rate = positive_count / len(df_dasp) * 100
        print(f"  {category:25}: {positive_count:4d} ({positive_rate:5.1f}%)")
    
    return df_dasp

# Create DASP labels with caching
if 'df_processed' in locals() and df_processed is not None:
    df_dasp_clean = create_dasp_labels_with_cache(df_processed, include_access_control=USE_6_LABELS)
    
    if df_dasp_clean is not None:
        # Define DASP categories based on configuration
        dasp_categories = ['Arithmetic', 'Unchecked_Low_Level_Calls', 'Time_Manipulation', 'DoS', 'Reentrancy']
        if USE_6_LABELS:
            dasp_categories.append('Access_Control')
        
        print(f"\nğŸ�¯ Final configuration:")
        print(f"  Labels: {len(dasp_categories)} ({'6-label' if USE_6_LABELS else '5-label'} classification)")
        print(f"  Categories: {dasp_categories}")
        print(f"  Dataset shape: {df_dasp_clean.shape}")
        
        # Calculate multi-label statistics
        label_matrix = df_dasp_clean[dasp_categories].values
        samples_with_labels = (label_matrix.sum(axis=1) > 0).sum()
        samples_with_multiple_labels = (label_matrix.sum(axis=1) > 1).sum()
        
        print(f"\nğŸ“Š Multi-label Statistics:")
        print(f"  Samples with at least one label: {samples_with_labels} ({samples_with_labels/len(df_dasp_clean)*100:.1f}%)")
        print(f"  Samples with multiple labels: {samples_with_multiple_labels} ({samples_with_multiple_labels/len(df_dasp_clean)*100:.1f}%)")
        print(f"  Average labels per sample: {label_matrix.sum(axis=1).mean():.2f}")
        
        print("âœ… DASP label creation completed")
    else:
        print("â�Œ DASP label creation failed")
        
else:
    print("â�Œ Cannot create DASP labels - missing preprocessed dataset")


import re
from nltk.corpus import stopwords

# This function preprocesses Solidity code
def preprocess_solidity_code(source_code):
    # Step 1: Remove block comments (/* ... */)
    source_code = re.sub(r'/\*.*?\*/', '', source_code, flags=re.DOTALL)
    
    # Step 2: Remove inline comments (// ...)
    source_code = re.sub(r'//.*', '', source_code)
    
    # Step 3: Remove all pragma directives (e.g., pragma solidity ^0.4.24;)
    source_code = re.sub(r'pragma\s+solidity\s+[^\n;]*;', '', source_code, flags=re.IGNORECASE)
    
    # Step 4: Remove Solidity keywords
    solidity_keywords = [
        'abstract', 'after', 'alias', 'apply', 'auto', 'case', 'catch', 'constant', 'copyof', 'default', 'define', 'final', 'immutable',
        'implements', 'in', 'inline', 'let', 'macro', 'match', 'mutable', 'null', 'of', 'partial', 'promise', 'reference', 'relocatable',
        'sealed', 'sizeof', 'static', 'supports', 'switch', 'typedef', 'typeof', 'var'
    ]
    for keyword in solidity_keywords:
        source_code = re.sub(r'\b' + keyword + r'\b', '', source_code, flags=re.IGNORECASE)
    
    # Step 5: Remove normal strings (e.g., "string" or 'string')
    source_code = re.sub(r'".*?"', '', source_code)
    source_code = re.sub(r"'.*?'", '', source_code)
    
    # Step 6: Remove address numbers (e.g., starting with 0x)
    source_code = re.sub(r'\b0x[a-fA-F0-9]+\b', '', source_code)
    
    # Step 7: Remove trivial operations (e.g., hello; or _;)
    source_code = re.sub(r'\b\w+;\b', '', source_code)
    
    # Step 8: Remove inherited contracts (e.g., contract A is B { ... })
    source_code = re.sub(r'contract\s+\w+\s+is\s+[^{]+{', 'contract {', source_code, flags=re.IGNORECASE)
    
    # Step 9: Remove event declarations (e.g., event Transfer(address indexed from, address indexed to, uint value);)
    source_code = re.sub(r'event\s+\w+\s*\(.*?\);', '', source_code, flags=re.DOTALL)
    
    # Step 10: Remove modifier declarations (e.g., modifier onlyOwner { ... })
    source_code = re.sub(r'modifier\s+\w+\s*{.*?}', '', source_code, flags=re.DOTALL)
    
    # Step 11: Remove interface declarations (e.g., interface ERC20 { ... })
    source_code = re.sub(r'interface\s+\w+\s*{.*?}', '', source_code, flags=re.DOTALL)
    
    # Step 12: Remove struct and enum declarations
    source_code = re.sub(r'struct\s+\w+\s*{.*?};', '', source_code, flags=re.DOTALL)
    source_code = re.sub(r'enum\s+\w+\s*{.*?};', '', source_code, flags=re.DOTALL)
    
    # Step 13: Simplify function declarations (e.g., function foo() public returns (uint) { ... } to function foo() { ... })
    source_code = re.sub(r'function\s+(\w+)\s*\(.*?\)\s*(public|private|internal|external)?\s*(returns\s*\(.*?\))?\s*{', r'function \1 {', source_code, flags=re.IGNORECASE)
    
    # Step 14: Remove library declarations (e.g., library SafeMath { ... })
    source_code = re.sub(r'library\s+\w+\s*{.*?}', '', source_code, flags=re.DOTALL)
    
    # Step 15: Remove all whitespaces (newlines, tabs) and redundant spaces
    source_code = re.sub(r'\s+', ' ', source_code).strip()
    
    # Step 16: Convert to lowercase
    source_code = source_code.lower()
    
    # Step 17: Remove stopwords
    stop_words = set(stopwords.words('english'))
    source_code = ' '.join(word for word in source_code.split() if word not in stop_words)

    # Step 18: Rename variables and functions
    # Find all variable and function names
    variable_pattern = re.compile(r'\b(uint|int|string|address|bool|bytes\d*)\s+(\w+)\b')
    function_pattern = re.compile(r'\bfunction\s+(\w+)\s*\(')

    variables = variable_pattern.findall(source_code)
    functions = function_pattern.findall(source_code)

    # Create mappings for variables and functions
    variable_mapping = {var[1]: f'v{i+1}' for i, var in enumerate(variables)}
    function_mapping = {func: f'f{i+1}' for i, func in enumerate(functions)}

    # Replace variable names
    for original, new in variable_mapping.items():
        source_code = re.sub(r'\b' + original + r'\b', new, source_code)

    # Replace function names
    for original, new in function_mapping.items():
        source_code = re.sub(r'\b' + original + r'\b', new, source_code)
    
    return source_code

# Test preprocessing on sample code
sample_code = """
pragma solidity ^0.8.0;

contract SimpleStorage {
    uint256 private storedData;
    
    function set(uint256 x, uint256 y) public {
        storedData = x + y + storedData; // Set value
    }
    
    function get() public view returns (uint256) {
        return storedData;
    }
}
"""

print("ğŸ§ª Testing preprocessing on sample code:")
print("=" * 50)
print("Original:")
print(sample_code)
print("\nPreprocessed:")
preprocessed_sample = preprocess_solidity_code(sample_code)
print(preprocessed_sample)
print("=" * 50)
print("âœ… Preprocessing function defined and tested")


from datasets import Dataset
from transformers import AutoTokenizer
import gc
import torch

def clear_memory():
    """Clear GPU and CPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        current_memory = torch.cuda.memory_allocated() / 1073741824
        print(f"ğŸ§¹ GPU memory cleared. Current usage: {current_memory:.2f} GB")
    print("âœ… Memory cleared successfully")

def preprocess_and_tokenize_with_cache(df, model_name, truncate_tokenization=True):
    """Preprocess source code and tokenize with intelligent caching"""
    
    if df is None:
        return None, None
    
    # Create parameters hash for cache validation
    tokenization_params = {
        'model_name': model_name,
        'truncate': truncate_tokenization,
        'dataset_shape': df.shape,
        'preprocessing_enabled': True
    }
    params_hash = hashlib.md5(str(tokenization_params).encode()).hexdigest()
    
    # Check cache for preprocessed source code
    preprocess_cache_key = cache_system.get_cache_key("source_preprocessing", model_name)
    tokenize_cache_key = cache_system.get_cache_key("tokenization", f"{model_name}_{truncate_tokenization}")
    
    # Try to load preprocessed data from cache
    if cache_system.is_cache_valid(preprocess_cache_key, params_hash=params_hash):
        print("ğŸš€ Loading preprocessed source code from cache...")
        dataset = cache_system.load_from_cache(preprocess_cache_key)
    else:
        print("ğŸ”„ Preprocessing source code...")
        
        # Apply preprocessing to source code
        df_processed = df.copy()
        
        print("ğŸ”§ Applying source code preprocessing...")
        df_processed['source_code'] = df_processed['source_code'].apply(preprocess_solidity_code)
        
        # Update word count after preprocessing
        df_processed['source_code_word_count'] = df_processed['source_code'].apply(lambda x: len(x.split()))
        
        print("ğŸ“ˆ Word count statistics after preprocessing:")
        print(df_processed['source_code_word_count'].describe())
        
        # Create dataset object
        dataset = Dataset.from_pandas(df_processed, preserve_index=False)
        print(f"ğŸ“Š Dataset created with {len(dataset)} samples")
        
        # Cache preprocessed data
        cache_system.save_to_cache(
            preprocess_cache_key, dataset,
            description=f"Preprocessed source code ({len(dataset)} samples)",
            params_hash=params_hash
        )
    
    # Try to load tokenized data from cache
    if cache_system.is_cache_valid(tokenize_cache_key, params_hash=params_hash):
        print("ğŸš€ Loading tokenized dataset from cache...")
        tokenized_dataset = cache_system.load_from_cache(tokenize_cache_key)
        
        # We also need to load the tokenizer
        print(f"ğŸ¤– Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenized_dataset, tokenizer
    
    # Clear memory before tokenization
    clear_memory()
    
    # Setup tokenizer
    print(f"ğŸ¤– Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"ğŸ“� Tokenizer info:")
    print(f"  Padding side: {tokenizer.padding_side}")
    print(f"  Padding token ID: {tokenizer.pad_token_id}")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Model max length: {tokenizer.model_max_length}")
    print(f"  Model input names: {tokenizer.model_input_names}")
    
    # Define tokenization functions
    def tokenize_batch_truncated(batch):
        """Tokenize with truncation and padding"""
        return tokenizer(
            batch["source_code"],
            truncation=True,
            padding=True,
            max_length=tokenizer.model_max_length,
            return_tensors=None,
            add_special_tokens=True
        )
    
    def tokenize_batch_no_truncation(batch):
        """Tokenize without truncation for BELT model"""
        return tokenizer(
            batch["source_code"],
            truncation=False,
            padding=False,
            return_tensors=None,
            add_special_tokens=False
        )
    
    # Apply tokenization based on configuration
    if truncate_tokenization is True:
        print("ğŸ”§ Applying truncated tokenization...")
        tokenized_dataset = dataset.map(
            tokenize_batch_truncated,
            batched=True,
            batch_size=batch_size,
            num_proc=num_processors if env_type != "Local" else 1,
            desc="Tokenizing (truncated)"
        )
        
        # Remove unnecessary columns and set format
        columns_to_keep = tokenizer.model_input_names + ["labels"] if "labels" in tokenized_dataset.column_names else tokenizer.model_input_names
        columns_to_remove = [col for col in tokenized_dataset.column_names if col not in columns_to_keep and col not in dasp_categories]
        if columns_to_remove:
            tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
        
        # Set PyTorch format
        tokenized_dataset.set_format(type='torch', columns=[col for col in tokenized_dataset.column_names if col in tokenizer.model_input_names])
        
    elif truncate_tokenization is False:
        print("ğŸ”§ Applying non-truncated tokenization...")
        tokenized_dataset = dataset.map(
            tokenize_batch_no_truncation,
            batched=True,
            batch_size=batch_size,
            num_proc=num_processors if env_type != "Local" else 1,
            desc="Tokenizing (no truncation)"
        )
    else:
        print("â�­ï¸� Skipping tokenization (will be handled by BELT model)")
        tokenized_dataset = dataset
    
    # Cache tokenized data
    cache_system.save_to_cache(
        tokenize_cache_key, tokenized_dataset,
        description=f"Tokenized dataset ({'truncated' if truncate_tokenization else 'no-truncation'})",
        params_hash=params_hash
    )
    
    print("âœ… Tokenization completed")
    return tokenized_dataset, tokenizer

# Apply preprocessing and tokenization with caching
if 'df_dasp_clean' in locals() and df_dasp_clean is not None:
    dataset, tokenizer = preprocess_and_tokenize_with_cache(
        df_dasp_clean, 
        transformers_model, 
        TOKENIZE_TRUNCATE
    )
    
    if dataset is not None:
        print(f"âœ… Preprocessing and tokenization setup completed")
        print(f"ğŸ“Š Final dataset: {len(dataset)} samples")
        
        # Update df_preprocessed for consistency
        df_preprocessed = dataset.to_pandas() if hasattr(dataset, 'to_pandas') else df_dasp_clean
        
    else:
        print("â�Œ Preprocessing and tokenization failed")
        
else:
    print("â�Œ Cannot proceed with tokenization - missing dataset")


def show_cache_status():
    """Display comprehensive cache status"""
    print("ğŸ”� Cache Status Report")
    print("="*50)
    cache_system.get_cache_info()
    
    # Additional cache statistics
    cache_files = list(cache_system.cache_dir.glob("*.pkl"))
    total_files = len(cache_files)
    total_size_bytes = sum(f.stat().st_size for f in cache_files)
    total_size_mb = total_size_bytes / (1024 * 1024)
    
    print(f"\nğŸ“� Cache Directory: {cache_system.cache_dir}")
    print(f"ğŸ—‚ï¸� Total cache files: {total_files}")
    print(f"ğŸ’¾ Total cache size: {total_size_mb:.1f} MB")
    
    # Show available operations
    print(f"\nğŸ› ï¸� Available Cache Operations:")
    print(f"  - show_cache_status(): Show this status report")
    print(f"  - clear_cache_all(): Clear all cached data")
    print(f"  - clear_cache_pattern('pattern'): Clear specific cache pattern")
    print(f"  - rebuild_cache(): Force rebuild of all caches")

def clear_cache_all():
    """Clear all cached data"""
    response = input("âš ï¸� This will clear ALL cached data. Continue? (y/N): ")
    if response.lower() == 'y':
        cache_system.clear_cache()
        print("âœ… All cache cleared successfully")
    else:
        print("â�Œ Cache clearing cancelled")

def clear_cache_pattern(pattern):
    """Clear cache files matching a pattern"""
    print(f"ğŸ”� Clearing cache files matching: {pattern}")
    cache_system.clear_cache(pattern)

def rebuild_cache():
    """Force rebuild of all caches by clearing them"""
    response = input("âš ï¸� This will clear all caches and force rebuild on next run. Continue? (y/N): ")
    if response.lower() == 'y':
        cache_system.clear_cache()
        print("âœ… Cache cleared. Run the dataset loading cells to rebuild caches.")
    else:
        print("â�Œ Cache rebuild cancelled")

def optimize_cache():
    """Optimize cache by removing invalid entries"""
    print("ğŸ”§ Optimizing cache...")
    
    # Check each cache entry for validity
    invalid_keys = []
    for key, info in cache_system.metadata.items():
        cache_file = Path(info['cache_file'])
        if not cache_file.exists():
            invalid_keys.append(key)
    
    # Remove invalid entries
    for key in invalid_keys:
        del cache_system.metadata[key]
    
    cache_system.save_metadata()
    
    if invalid_keys:
        print(f"ğŸ—‘ï¸� Removed {len(invalid_keys)} invalid cache entries")
    else:
        print("âœ… Cache is already optimized")

# Show initial cache status
show_cache_status()

# Quick access functions
print(f"\nâš¡ Quick Commands:")
print(f"  show_cache_status()  - Show cache information")
print(f"  clear_cache_all()    - Clear all caches") 
print(f"  optimize_cache()     - Remove invalid cache entries")


def prepare_labels_and_split_with_cache(dataset, dasp_categories, test_size=0.1, val_size=0.1, seed=42):
    """
    Prepare labels and split dataset into train/validation/test sets with caching
    
    Args:
        dataset: HuggingFace Dataset object
        dasp_categories: List of DASP category names
        test_size: Proportion for test set
        val_size: Proportion for validation set
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, id2label, label2id)
    """
    
    if dataset is None:
        return None, None, None, None, None
    
    # Create parameters hash for cache validation
    split_params = {
        'dasp_categories': dasp_categories,
        'test_size': test_size,
        'val_size': val_size,
        'seed': seed,
        'dataset_length': len(dataset)
    }
    params_hash = hashlib.md5(str(split_params).encode()).hexdigest()
    
    # Check cache for split data
    split_cache_key = cache_system.get_cache_key("dataset_split", f"test{test_size}_val{val_size}_seed{seed}")
    
    if cache_system.is_cache_valid(split_cache_key, params_hash=params_hash):
        print("ğŸš€ Loading dataset splits from cache...")
        cached_data = cache_system.load_from_cache(split_cache_key)
        if cached_data:
            return cached_data['train'], cached_data['val'], cached_data['test'], cached_data['id2label'], cached_data['label2id']
    
    print("ğŸ”„ Creating dataset splits...")
    
    # Create label mappings
    id2label = {i: label for i, label in enumerate(dasp_categories)}
    label2id = {label: i for i, label in enumerate(dasp_categories)}
    
    print(f"ğŸ�·ï¸� Label mappings created:")
    print(f"  ğŸ“Š Number of labels: {len(dasp_categories)}")
    print(f"  ğŸ”¢ id2label: {id2label}")
    print(f"  ğŸ�·ï¸� label2id: {label2id}")
    
    # Add labels column
    def add_labels(batch):
        labels = []
        for i in range(len(batch[dasp_categories[0]])):  # Get batch size
            label_vector = [batch[cat][i] for cat in dasp_categories]
            labels.append(label_vector)
        return {"labels": labels}
    
    dataset_with_labels = dataset.map(
        add_labels,
        batched=True,
        desc="Adding labels"
    )
    
    # Shuffle dataset
    dataset_shuffled = dataset_with_labels.shuffle(seed=seed)
    
    # Calculate split sizes
    total_size = len(dataset_shuffled)
    test_count = int(total_size * test_size)
    val_count = int(total_size * val_size)
    train_count = total_size - test_count - val_count
    
    print(f"\nğŸ“ˆ Dataset split:")
    print(f"  ğŸš‚ Training: {train_count} samples ({train_count/total_size*100:.1f}%)")
    print(f"  âœ… Validation: {val_count} samples ({val_count/total_size*100:.1f}%)")
    print(f"  ğŸ§ª Test: {test_count} samples ({test_count/total_size*100:.1f}%)")
    
    # Split dataset
    train_dataset = dataset_shuffled.select(range(train_count))
    val_dataset = dataset_shuffled.select(range(train_count, train_count + val_count))
    test_dataset = dataset_shuffled.select(range(train_count + val_count, total_size))
    
    # Cache the split data
    split_data = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset,
        'id2label': id2label,
        'label2id': label2id
    }
    
    cache_system.save_to_cache(
        split_cache_key, split_data,
        description=f"Dataset splits (train:{train_count}, val:{val_count}, test:{test_count})",
        params_hash=params_hash
    )
    
    return train_dataset, val_dataset, test_dataset, id2label, label2id

# Prepare labels and split dataset with caching
if 'dataset' in locals() and 'dasp_categories' in locals() and dataset is not None:
    
    train_dataset, val_dataset, test_dataset, id2label, label2id = prepare_labels_and_split_with_cache(
        dataset, dasp_categories
    )
    
    if train_dataset is not None:
        # Prepare data for different frameworks
        if TOKENIZE_TRUNCATE is None:
            # For BELT model - extract source code and labels
            X_train = train_dataset['source_code']
            y_train = train_dataset['labels']
            X_val = val_dataset['source_code']
            y_val = val_dataset['labels']
            X_test = test_dataset['source_code']
            y_test = test_dataset['labels']
            
            print(f"ğŸ“Š BELT data prepared:")
            print(f"  Training: {len(X_train)} samples")
            print(f"  Validation: {len(X_val)} samples")
            print(f"  Testing: {len(X_test)} samples")
            
        else:
            # For PyTorch/TensorFlow models - use tokenized datasets
            print(f"ğŸ“Š Tokenized datasets prepared:")
            print(f"  Training: {train_dataset.shape}")
            print(f"  Validation: {val_dataset.shape}")
            print(f"  Testing: {test_dataset.shape}")
        
        # Display label distribution
        print(f"\nğŸ“Š Label distribution in training set:")
        if TOKENIZE_TRUNCATE is None:
            label_sums = np.array(y_train).sum(axis=0)
        else:
            label_sums = np.array(train_dataset['labels']).sum(axis=0)
        
        for i, (category, count) in enumerate(zip(dasp_categories, label_sums)):
            percentage = (count / len(train_dataset)) * 100
            print(f"  {category:25}: {count:4d} ({percentage:5.1f}%)")
        
        print("âœ… Dataset preparation completed")
    else:
        print("â�Œ Dataset splitting failed")
        
else:
    print("â�Œ Cannot prepare dataset - missing required components")


if TRAINING_FRAMEWORK == "belt":
    try:
        from pathlib import Path
        import json
        import torch
        import numpy as np
        from torch import argmax, Tensor
        from torch.optim import AdamW, Optimizer
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
        from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, Sigmoid, Softmax
        from belt_nlp.bert import TokenizedDataset
        from belt_nlp.bert_classifier_with_pooling import BertClassifierWithPooling
        from typing import Any, Optional, Union
        from IPython.display import clear_output, display
        import matplotlib.pyplot as plt
        
        print("ğŸ“š BELT libraries imported successfully")
        
        class CustomBertClassifierWithPooling(BertClassifierWithPooling):
            """Custom BERT classifier with enhanced multi-label support, verbose training, checkpointing, and epoch loss tracking/plotting"""
            
            def __init__(self, *args, **kwargs):
                # Extract checkpoint parameters
                self.checkpoint_dir = kwargs.pop("checkpoint_dir", "./checkpoints")
                self.num_checkpoint_save = kwargs.pop("num_checkpoint_save", 5)
                self.load_checkpoint = kwargs.pop("load_checkpoint", -1)
                self.multilabel = kwargs.pop("multilabel", False)
                
                super().__init__(*args, **kwargs)
                
                # Initialize checkpoint tracking
                self.current_epoch = 0
                self.checkpoint_dir = Path(self.checkpoint_dir)
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                # Loss tracking artifacts
                self.epoch_losses = []
                self.epoch_loss_file = self.checkpoint_dir / "epoch-loss.txt"
                self.loss_plot_path = self.checkpoint_dir / "loss.png"
                # Load previous losses if resuming
                if self.epoch_loss_file.exists():
                    try:
                        with open(self.epoch_loss_file, "r", encoding="utf-8") as f:
                            lines = [l.strip() for l in f.readlines() if l.strip()]
                        # accept formats: "epoch,loss" or just "loss"
                        parsed = []
                        for idx, line in enumerate(lines, start=1):
                            if "," in line:
                                try:
                                    ep_str, loss_str = line.split(",", 1)
                                    parsed.append(float(loss_str))
                                except Exception:
                                    continue
                            else:
                                try:
                                    parsed.append(float(line))
                                except Exception:
                                    continue
                        if parsed:
                            self.epoch_losses = parsed
                    except Exception:
                        pass
                
                # Load checkpoint if requested
                if self.load_checkpoint >= 0:
                    self._load_checkpoint()
            
            def _save_checkpoint(self, epoch: int, optimizer_state: dict = None) -> Path:
                """Save training checkpoint"""
                checkpoint_name = f"checkpoint_epoch_{epoch:03d}.pt"
                checkpoint_path = self.checkpoint_dir / checkpoint_name
                
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': self.neural_network.module.state_dict() if self.many_gpus else self.neural_network.state_dict(),
                    'params': self._params,
                    'current_epoch': epoch
                }
                
                if optimizer_state:
                    checkpoint_data['optimizer_state_dict'] = optimizer_state
                
                torch.save(checkpoint_data, checkpoint_path)
                
                # Manage checkpoint count
                self._cleanup_old_checkpoints()
                
                return checkpoint_path
            
            def _load_checkpoint(self) -> bool:
                """Load checkpoint based on load_checkpoint parameter"""
                try:
                    if self.load_checkpoint == 0:
                        # Load latest checkpoint
                        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
                        if not checkpoint_files:
                            print("ğŸ“‚ No checkpoints found, starting fresh training")
                            return False
                        
                        # Sort by epoch number and get latest
                        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
                        checkpoint_path = checkpoint_files[-1]
                        
                    else:
                        # Load specific checkpoint
                        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.load_checkpoint:03d}.pt"
                        if not checkpoint_path.exists():
                            print(f"â�Œ Checkpoint epoch {self.load_checkpoint} not found")
                            return False
                    
                    print(f"ğŸ“‚ Loading checkpoint: {checkpoint_path.name}")
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    
                    # Load model state
                    if self.many_gpus:
                        self.neural_network.module.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.neural_network.load_state_dict(checkpoint['model_state_dict'])
                    
                    # Update current epoch
                    self.current_epoch = checkpoint['epoch']
                    print(f"âœ… Checkpoint loaded successfully (epoch {self.current_epoch})")
                    
                    return True
                    
                except Exception as e:
                    print(f"â�Œ Error loading checkpoint: {e}")
                    return False
            
            def _cleanup_old_checkpoints(self) -> None:
                """Keep only the latest N checkpoints"""
                checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
                if len(checkpoint_files) > self.num_checkpoint_save:
                    # Sort by epoch number
                    checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
                    
                    # Remove oldest checkpoints
                    for old_checkpoint in checkpoint_files[:-self.num_checkpoint_save]:
                        old_checkpoint.unlink()
                        print(f"ğŸ—‘ï¸� Removed old checkpoint: {old_checkpoint.name}")
            
            def _list_checkpoints(self) -> list:
                """List available checkpoints"""
                checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
                checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
                return [int(f.stem.split('_')[-1]) for f in checkpoint_files]
            
            def save(self, model_dir: str) -> None:
                """Save model with proper type conversion"""
                model_dir = Path(model_dir)
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # Convert int64 to int for JSON serialization
                params = {
                    key: int(value) if isinstance(value, (np.int64, torch.Tensor)) else value
                    for key, value in self._params.items()
                }
                
                with open(model_dir / "params.json", "w", encoding="utf-8") as f:
                    json.dump(params, f)
                
                self.tokenizer.save_pretrained(model_dir)
                
                if self.many_gpus:
                    torch.save(self.neural_network.module, model_dir / "model.bin")
                else:
                    torch.save(self.neural_network, model_dir / "model.bin")
            
            def _record_and_plot_loss(self, epoch: int, avg_loss: float, verbose: bool = False) -> None:
                """Record epoch loss to file and update/save plot"""
                # Ensure list length matches epoch count
                if len(self.epoch_losses) < epoch:
                    self.epoch_losses.extend([None] * (epoch - len(self.epoch_losses)))
                self.epoch_losses[epoch - 1] = float(avg_loss)
                
                # Write epoch-loss.txt (epoch,loss per line)
                try:
                    with open(self.epoch_loss_file, "w", encoding="utf-8") as f:
                        for i, loss in enumerate(self.epoch_losses, start=1):
                            if loss is not None:
                                f.write(f"{i},{loss:.6f}\n")
                except Exception as e:
                    print(f"âš ï¸� Failed writing epoch-loss.txt: {e}")
                
                # Plot and save
                try:
                    plt.figure(figsize=(8, 4))
                    epochs_axis = [i for i, l in enumerate(self.epoch_losses, start=1) if l is not None]
                    losses_axis = [l for l in self.epoch_losses if l is not None]
                    plt.plot(epochs_axis, losses_axis, marker='o', color='tab:blue')
                    plt.title('Training Loss per Epoch')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.grid(True, linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    plt.savefig(self.loss_plot_path)
                    if verbose:
                        clear_output(wait=True)
                        display(plt.gcf())
                    plt.close()
                except Exception as e:
                    print(f"âš ï¸� Failed plotting/saving loss graph: {e}")
            
            def fit(self, x_train: list, y_train: list, epochs: Optional[int] = None, verbose: bool = False) -> None:
                """Enhanced training with verbose output, memory management, checkpointing, and epoch loss logging"""
                if not epochs:
                    epochs = self.epochs
                
                optimizer = AdamW(self.neural_network.parameters(), lr=self.learning_rate)
                
                # Load optimizer state if resuming from checkpoint
                if self.current_epoch > 0 and verbose:
                    print(f"ğŸ”„ Resuming training from epoch {self.current_epoch + 1}")
                
                if verbose:
                    available_checkpoints = self._list_checkpoints()
                    if available_checkpoints:
                        print(f"ğŸ“‚ Available checkpoints: {available_checkpoints}")
                    print(f"ğŸš€ Starting training for {epochs} epochs (current: {self.current_epoch})...")
                    print("ğŸ”¤ Tokenizing training data...")
                
                tokens = self._tokenize(x_train)
                dataset = TokenizedDataset(tokens, y_train)
                dataloader = DataLoader(
                    dataset, 
                    sampler=RandomSampler(dataset), 
                    batch_size=self.batch_size, 
                    collate_fn=self.collate_fn
                )
                
                # Clear memory after tokenization
                del tokens
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if verbose:
                    print(f"âœ… Tokenization complete. Starting training...")
                
                # Training loop starting from current epoch
                for epoch in range(self.current_epoch, epochs):
                    if verbose:
                        print(f"\nğŸ“ˆ Epoch {epoch + 1}/{epochs}")
                    
                    avg_loss = self._train_single_epoch(dataloader, optimizer, verbose)
                    
                    # Save checkpoint after each epoch
                    checkpoint_path = self._save_checkpoint(epoch + 1, optimizer.state_dict())
                    
                    # Update current epoch
                    self.current_epoch = epoch + 1
                    
                    # Record loss and update plot (and save epoch-loss.txt)
                    self._record_and_plot_loss(self.current_epoch, avg_loss, verbose=verbose)
                    
                    if verbose:
                        print(f"âœ… Epoch {epoch + 1} completed | Avg Loss: {avg_loss:.4f}")
                        print(f"ğŸ’¾ Checkpoint saved: {checkpoint_path.name}")
                
                if verbose:
                    print(f"ğŸ�‰ Training completed! Final checkpoint saved at epoch {self.current_epoch}")
            
            def _train_single_epoch(self, dataloader: DataLoader, optimizer: Optimizer, verbose: bool = False) -> float:
                """Single epoch training with enhanced monitoring; returns average loss"""
                import time
                
                self.neural_network.train()
                total_loss = 0.0
                num_batches = len(dataloader)
                
                # Initialize timing
                epoch_start_time = time.time()
                
                for step, batch in enumerate(dataloader):
                    if self.multilabel:
                        labels = batch[-1].float().to(self.device)
                        loss_function = BCEWithLogitsLoss()
                        logits = self._evaluate_single_batch(batch)
                        loss = loss_function(logits, labels) / self.accumulation_steps
                    elif self.num_labels > 1:
                        labels = batch[-1].long().to(self.device)
                        loss_function = CrossEntropyLoss()
                        logits = self._evaluate_single_batch(batch)
                        loss = loss_function(logits, labels) / self.accumulation_steps
                    else:
                        labels = batch[-1].float().to(self.device)
                        loss_function = MSELoss()
                        scores = torch.flatten(self._evaluate_single_batch(batch))
                        loss = loss_function(scores, labels) / self.accumulation_steps
                    
                    loss.backward()
                    total_loss += loss.item()
                    
                    if ((step + 1) % self.accumulation_steps == 0) or (step + 1 == num_batches):
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    if verbose and (step + 1) % 10 == 0:
                        current_time = time.time()
                        elapsed_time = current_time - epoch_start_time
                        avg_loss = total_loss / (step + 1)
                        memory_gb = torch.cuda.memory_allocated() / 1073741824 if torch.cuda.is_available() else 0
                        
                        # Calculate timing metrics
                        steps_per_second = (step + 1) / elapsed_time if elapsed_time > 0 else 0.0
                        seconds_per_step = elapsed_time / (step + 1) if (step + 1) > 0 else 0.0
                        
                        print(
                            f"  Step {step + 1:3d}/{num_batches} | Loss: {avg_loss:.4f} | GPU: {memory_gb:.1f}GB | {steps_per_second:.2f} steps/s ({seconds_per_step:.2f}s/step)",
                            end='\r'
                        )
                    
                    # Memory cleanup
                    del batch, labels, loss
                    if 'logits' in locals():
                        del logits
                    if self.num_labels == 1 and 'scores' in locals():
                        del scores
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                epoch_end_time = time.time()
                total_epoch_time = epoch_end_time - epoch_start_time
                avg_loss = total_loss / max(num_batches, 1)
                if verbose:
                    avg_steps_per_second = num_batches / total_epoch_time if total_epoch_time > 0 else 0.0
                    avg_seconds_per_step = total_epoch_time / max(num_batches, 1)
                    print(f"\n  ğŸ“Š Average loss: {avg_loss:.4f} | Epoch time: {total_epoch_time:.1f}s | Avg: {avg_steps_per_second:.2f} steps/s ({avg_seconds_per_step:.2f}s/step)")
                return avg_loss
            
            def predict(self, x: list, batch_size: Optional[int] = None, verbose: bool = False) -> Tensor:
                """Prediction with verbose output"""
                logits = self._predict_logits(x, batch_size, verbose)
                
                if self.multilabel:
                    sigmoid = Sigmoid()
                    probabilities = sigmoid(logits)
                    classes = (probabilities >= 0.5).int()
                else:
                    classes = argmax(logits, dim=1)
                
                return classes
            
            def _predict_logits(self, x: list, batch_size: Optional[int] = None, verbose: bool = False) -> Tensor:
                """Prediction with enhanced monitoring"""
                if not batch_size:
                    batch_size = self.batch_size
                
                if verbose:
                    print("ğŸ”¤ Tokenizing prediction data...")
                
                tokens = self._tokenize(x)
                dataset = TokenizedDataset(tokens)
                dataloader = DataLoader(
                    dataset, 
                    sampler=SequentialSampler(dataset), 
                    batch_size=batch_size, 
                    collate_fn=self.collate_fn
                )
                
                total_logits = []
                self.neural_network.eval()
                
                if verbose:
                    print("ğŸ”® Running inference...")
                
                num_batches = len(dataloader)
                for step, batch in enumerate(dataloader):
                    with torch.no_grad():
                        logits = self._evaluate_single_batch(batch)
                        total_logits.append(logits)
                    
                    if verbose and (step + 1) % 10 == 0:
                        print(f"  Batch {step + 1}/{num_batches} processed", end='\\r')
                
                if verbose:
                    print(f"\\nâœ… Inference completed")
                
                return torch.cat(total_logits)
        
        print("âœ… Custom BELT model class defined")
        
    except ImportError as e:
        print(f"âš ï¸� BELT library not available: {e}")
        print("ğŸ“¦ Install with: pip install belt-nlp")
        TRAINING_FRAMEWORK = "pytorch"  # Fallback to PyTorch
        print(f"ğŸ”„ Switching to {TRAINING_FRAMEWORK} framework")

else:
    print(f"â�­ï¸� Skipping BELT setup (selected framework: {TRAINING_FRAMEWORK})")


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


import shutil
# BELT Model Configuration and Training
if TRAINING_FRAMEWORK == "belt" and 'CustomBertClassifierWithPooling' in locals():
    
    # Clear memory before model creation
    clear_memory()
    
    # Model parameters
    MODEL_PARAMS = {
        "num_labels": len(dasp_categories),
        "batch_size": train_batch_size,
        "learning_rate": 2e-5,
        "epochs": epochs,
        "chunk_size": 510,
        "stride": 256,
        "minimal_chunk_length": 510,
        "maximal_text_length": maximal_text_length,
        "pooling_strategy": "mean",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "many_gpus": torch.cuda.device_count() > 1,
        "multilabel": True,
        "pretrained_model_name_or_path": transformers_model,
        "checkpoint_dir": "./checkpoints",
        "num_checkpoint_save": 5,  # Keep last 5 checkpoints
        "load_checkpoint": 0,  # 0 = latest, specific number to load specific checkpoint, -1 = don't load
    }
    
    print("ğŸ¤– BELT Model Configuration:")
    for key, value in MODEL_PARAMS.items():
        print(f"  {key}: {value}")
    
    # Create model
    try:
        belt_model = CustomBertClassifierWithPooling(**MODEL_PARAMS)
        print("âœ… BELT model created successfully")
        
        # Train model
        if 'X_train' in locals() and 'y_train' in locals():
            print(f"\nğŸš€ Starting BELT model training...")
            print(f"ğŸ“Š Training on {len(X_train)} samples")
            
            belt_model.fit(X_train, y_train, verbose=True)
            
            print("âœ… BELT model training completed")
            
            # Evaluate model
            print(f"\nğŸ§ª Evaluating on {len(X_test)} test samples...")
            y_pred_belt = belt_model.predict(X_test, verbose=True)
            y_pred_belt_np = y_pred_belt.cpu().numpy()
            y_test_np = np.array(y_test)
            
            # Calculate metrics
            from sklearn.metrics import classification_report, accuracy_score, f1_score, hamming_loss, precision_score, recall_score
            
            print("\nğŸ“Š BELT Model Results:")
            print("=" * 50)
            
            # Overall metrics
            # accuracy = accuracy_score(y_test_np, y_pred_belt_np)(y_test_np, y_pred_belt_np)
            print("Overall:")
            precision = precision_score(y_test_np, y_pred_belt_np)
            recall = recall_score(y_test_np, y_pred_belt_np)
            f1_micro = f1_score(y_test_np, y_pred_belt_np, average='micro')
            f1_macro = f1_score(y_test_np, y_pred_belt_np, average='macro')
            hammingScore = hamming_score(y_test_np, y_pred_belt_np)
            hammingLoss = hamming_loss(y_test_np, y_pred_belt_np)

            print(f"Precision Score: {precision:.4f}")
            print(f"Recall Score: {recall:.4f}")
            print(f"F1-Score (Micro): {f1_micro:.4f}")
            print(f"F1-Score (Macro): {f1_macro:.4f}")
            print(f"Hamming Score: {hammingScore:.4f}")
            print(f"Hamming Loss: {hammingLoss:.4f}")
            
            # Per-category results
            print("\\nPer-category Classification Report:")
            print(classification_report(
                y_test_np, y_pred_belt_np, 
                target_names=dasp_categories, 
                zero_division=0
            ))

            for i in range(len(dasp_categories)):
                print(f"Classification report for {dasp_categories[i]}: ")
                print(classification_report(y_test_np[:, i], y_pred_belt_np[:, i]))
            
            # Save model
            model_save_path = "./belt_dasp_model"
            belt_model.save(model_save_path)
            print(f"ğŸ’¾ Model saved to {model_save_path}")
            shutil.make_archive("belt_dasp_model", "zip", "belt_dasp_model")
            
        else:
            print("â�Œ Training data not available")
            
    except Exception as e:
        print(f"â�Œ BELT model training failed: {e}")
        print("ğŸ”„ Consider switching to PyTorch framework")

elif TRAINING_FRAMEWORK == "belt":
    print("â�Œ BELT framework selected but not available")
    print("ğŸ”„ Please install belt-nlp or switch framework")
else:
    print(f"â�­ï¸� Skipping BELT training (selected framework: {TRAINING_FRAMEWORK})")


if TRAINING_FRAMEWORK == "pytorch":
    try:
        from transformers import (
            AutoModelForSequenceClassification, 
            TrainingArguments, 
            Trainer, 
            DataCollatorWithPadding,
            EvalPrediction,
            TrainerCallback
        )
        from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
        import torch
        from pathlib import Path
        import matplotlib.pyplot as plt
        from IPython.display import clear_output, display
        
        print("ğŸ“š PyTorch transformers libraries imported successfully")
        
        # Clear memory before model creation
        clear_memory()
        
        # Create model
        print(f"ğŸ¤– Loading PyTorch model: {transformers_model}")
        pytorch_model = AutoModelForSequenceClassification.from_pretrained(
            transformers_model,
            num_labels=len(dasp_categories),
            problem_type="multi_label_classification",
            id2label=id2label,
            label2id=label2id
        )
        
        print(f"âœ… PyTorch model created with {len(dasp_categories)} labels")
        
        # Define metrics computation
        def compute_metrics(eval_pred: EvalPrediction):
            """Compute multi-label classification metrics"""
            predictions, labels = eval_pred
            
            # Apply sigmoid and threshold
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(torch.Tensor(predictions))
            y_pred = (probs >= 0.5).int().numpy()
            y_true = labels
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            f1_micro = f1_score(y_true, y_pred, average='micro')
            f1_macro = f1_score(y_true, y_pred, average='macro')
            
            try:
                roc_auc = roc_auc_score(y_true, y_pred, average='micro')
            except ValueError:
                roc_auc = 0.0  # Handle case where only one class is present
            
            return {
                'accuracy': accuracy,
                'f1_micro': f1_micro,
                'f1_macro': f1_macro,
                'roc_auc': roc_auc
            }
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./pytorch_dasp_model",
            num_train_epochs=epochs,
            learning_rate=2e-5,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=train_batch_size,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_micro",
            greater_is_better=True,
            overwrite_output_dir=True,
            report_to=None,
            disable_tqdm=False,
            remove_unused_columns=False,
        )
        
        print("âš™ï¸� Training arguments configured")
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        class EpochLossLogger(TrainerCallback):
            """Logs per-epoch loss to file and updates a loss plot under output_dir"""
            def __init__(self, output_dir: str):
                self.output_dir = Path(output_dir)
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self.losses = []
                self.loss_file = self.output_dir / "epoch-loss.txt"
                self.plot_path = self.output_dir / "loss.png"
                # Load previous if exist (helpful if resuming)
                if self.loss_file.exists():
                    try:
                        with open(self.loss_file, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                if "," in line:
                                    _, loss_str = line.split(",", 1)
                                    self.losses.append(float(loss_str))
                                else:
                                    self.losses.append(float(line))
                    except Exception:
                        pass
            
            def on_epoch_end(self, args, state, control, **kwargs):
                # Use training loss from state.log_history if available, else state.train_loss
                train_loss = None
                if state.log_history:
                    # last dict may contain 'loss' or 'train_loss'
                    last = state.log_history[-1]
                    train_loss = last.get('loss', last.get('train_loss', None))
                if train_loss is None:
                    train_loss = getattr(state, 'train_loss', None)
                if train_loss is None and hasattr(kwargs.get('metrics', {}), 'get'):
                    train_loss = kwargs['metrics'].get('train_loss')
                if train_loss is None:
                    # As a fallback don't record for this epoch
                    return
                self.losses.append(float(train_loss))
                # Write file fresh each time as epoch,loss
                try:
                    with open(self.loss_file, "w", encoding="utf-8") as f:
                        for i, l in enumerate(self.losses, start=1):
                            f.write(f"{i},{l:.6f}\n")
                except Exception as e:
                    print(f"âš ï¸� Failed writing epoch-loss.txt: {e}")
                # Plot
                try:
                    plt.figure(figsize=(8,4))
                    xs = list(range(1, len(self.losses)+1))
                    plt.plot(xs, self.losses, marker='o', color='tab:blue')
                    plt.title('Training Loss per Epoch')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.grid(True, linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    plt.savefig(self.plot_path)
                    clear_output(wait=True)
                    display(plt.gcf())
                    plt.close()
                except Exception as e:
                    print(f"âš ï¸� Failed plotting/saving loss graph: {e}")
        
        # Create trainer
        trainer = Trainer(
            model=pytorch_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            callbacks=[EpochLossLogger(output_dir=training_args.output_dir)],
        )
        
        print("âœ… Trainer configured successfully")
        
        # Train model
        if 'train_dataset' in locals():
            print(f"ğŸš€ Starting PyTorch model training...")
            print(f"ğŸ“Š Training on {len(train_dataset)} samples")
            print(f"ğŸ“Š Validation on {len(val_dataset)} samples")
            
            # Train
            train_result = trainer.train()
            
            print("âœ… PyTorch model training completed")
            
            # Evaluate on test set
            print(f"ğŸ§ª Evaluating on {len(test_dataset)} test samples...")
            test_results = trainer.evaluate(test_dataset)
            
            print("\nğŸ“Š PyTorch Model Test Results:")
            print("=" * 50)
            for key, value in test_results.items():
                if key.startswith('eval_'):
                    metric_name = key.replace('eval_', '').replace('_', ' ').title()
                    print(f"{metric_name}: {value:.4f}")
            
            # Save model
            trainer.save_model("./pytorch_dasp_model_final")
            tokenizer.save_pretrained("./pytorch_dasp_model_final")
            print("ğŸ’¾ Model and tokenizer saved")
            
            # Detailed prediction analysis
            test_predictions = trainer.predict(test_dataset)
            y_pred_pytorch = (torch.nn.Sigmoid()(torch.Tensor(test_predictions.predictions)) >= 0.5).int().numpy()
            y_true_pytorch = test_predictions.label_ids
            
            print("\nDetailed Classification Report:")
            from sklearn.metrics import classification_report
            print(classification_report(
                y_true_pytorch, y_pred_pytorch,
                target_names=dasp_categories,
                zero_division=0
            ))
            
        else:
            print("â�Œ Training dataset not available")
        
    except ImportError as e:
        print(f"âš ï¸� PyTorch transformers not available: {e}")
        print("ğŸ“¦ Install with: pip install transformers torch")
    except Exception as e:
        print(f"â�Œ PyTorch training failed: {e}")

else:
    print(f"â�­ï¸� Skipping PyTorch training (selected framework: {TRAINING_FRAMEWORK})")


if TRAINING_FRAMEWORK == "tensorflow":
    try:
        import tensorflow as tf
        from transformers import TFAutoModelForSequenceClassification
        from tensorflow.keras import backend as K
        from pathlib import Path
        import matplotlib.pyplot as plt
        from IPython.display import clear_output, display
        
        print("ğŸ“š TensorFlow transformers libraries imported successfully")
        
        # Clear Keras session
        K.clear_session()
        
        # Configure GPU memory growth
        if tf.config.list_physical_devices('GPU'):
            try:
                for gpu in tf.config.experimental.list_physical_devices('GPU'):
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("âœ… GPU memory growth configured")
            except RuntimeError as e:
                print(f"âš ï¸� GPU configuration warning: {e}")
        
        # Clear memory before model creation
        clear_memory()
        
        # Create TensorFlow model
        print(f"ğŸ¤– Loading TensorFlow model: {transformers_model}")
        tf_model = TFAutoModelForSequenceClassification.from_pretrained(
            transformers_model,
            num_labels=len(dasp_categories),
            problem_type="multi_label_classification"
        )
        
        print(f"âœ… TensorFlow model created with {len(dasp_categories)} labels")
        
        # Convert datasets to TensorFlow format
        if 'train_dataset' in locals() and TOKENIZE_TRUNCATE is not None:
            print("ğŸ”„ Converting datasets to TensorFlow format...")
            
            tokenizer_columns = tokenizer.model_input_names
            tf_train_dataset = train_dataset.to_tf_dataset(
                columns=tokenizer_columns,
                label_cols="labels",
                shuffle=True,
                batch_size=train_batch_size
            )
            tf_val_dataset = val_dataset.to_tf_dataset(
                columns=tokenizer_columns,
                label_cols="labels",
                shuffle=False,
                batch_size=train_batch_size
            )
            tf_test_dataset = test_dataset.to_tf_dataset(
                columns=tokenizer_columns,
                label_cols="labels",
                shuffle=False,
                batch_size=train_batch_size
            )
            
            print("âœ… Datasets converted to TensorFlow format")
            
            # Compile model
            tf_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[
                    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')
                ]
            )
            
            print("âœ… Model compiled successfully")
            
            # Custom callback for epoch loss logging and plotting
            class EpochLossLoggerTF(tf.keras.callbacks.Callback):
                def __init__(self, out_dir: str = "./tensorflow_dasp_model"):
                    super().__init__()
                    self.out_dir = Path(out_dir)
                    self.out_dir.mkdir(parents=True, exist_ok=True)
                    self.losses = []
                    self.loss_file = self.out_dir / "epoch-loss.txt"
                    self.plot_path = self.out_dir / "loss.png"
                    if self.loss_file.exists():
                        try:
                            with open(self.loss_file, "r", encoding="utf-8") as f:
                                for line in f:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    if "," in line:
                                        _, loss_str = line.split(",", 1)
                                        self.losses.append(float(loss_str))
                                    else:
                                        self.losses.append(float(line))
                        except Exception:
                            pass
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    loss = logs.get('loss')
                    if loss is None:
                        return
                    self.losses.append(float(loss))
                    # Write file
                    try:
                        with open(self.loss_file, "w", encoding="utf-8") as f:
                            for i, l in enumerate(self.losses, start=1):
                                f.write(f"{i},{l:.6f}\n")
                    except Exception as e:
                        print(f"âš ï¸� Failed writing epoch-loss.txt: {e}")
                    # Plot
                    try:
                        plt.figure(figsize=(8,4))
                        xs = list(range(1, len(self.losses)+1))
                        plt.plot(xs, self.losses, marker='o', color='tab:blue')
                        plt.title('Training Loss per Epoch')
                        plt.xlabel('Epoch')
                        plt.ylabel('Loss')
                        plt.grid(True, linestyle='--', alpha=0.5)
                        plt.tight_layout()
                        plt.savefig(self.plot_path)
                        clear_output(wait=True)
                        display(plt.gcf())
                        plt.close()
                    except Exception as e:
                        print(f"âš ï¸� Failed plotting/saving loss graph: {e}")
            
            # Define callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=3,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,
                    min_lr=1e-7
                ),
                EpochLossLoggerTF(out_dir="./tensorflow_dasp_model")
            ]
            
            # Train model
            print(f"ğŸš€ Starting TensorFlow model training...")
            print(f"ğŸ“Š Training for {epochs} epochs")
            
            history = tf_model.fit(
                tf_train_dataset,
                validation_data=tf_val_dataset,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            print("âœ… TensorFlow model training completed")
            
            # Evaluate on test set
            print(f"ğŸ§ª Evaluating on test dataset...")
            test_results = tf_model.evaluate(tf_test_dataset, verbose=1)
            
            print("\nğŸ“Š TensorFlow Model Test Results:")
            print("=" * 50)
            metric_names = tf_model.metrics_names
            for name, value in zip(metric_names, test_results):
                print(f"{name.title()}: {value:.4f}")
            
            # Save model
            model_save_path = "./tensorflow_dasp_model"
            tf_model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            print(f"ğŸ’¾ Model and tokenizer saved to {model_save_path}")
            
            # Plot training history
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 3, 2)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.subplot(1, 3, 3)
            if 'precision' in history.history:
                plt.plot(history.history['precision'], label='Training Precision')
                plt.plot(history.history['val_precision'], label='Validation Precision')
                plt.title('Model Precision')
                plt.xlabel('Epoch')
                plt.ylabel('Precision')
                plt.legend()
            
            plt.tight_layout()
            plt.show()
            
            # Detailed predictions
            print("\nğŸ”® Making detailed predictions...")
            predictions = tf_model.predict(tf_test_dataset)
            y_pred_tf = (tf.nn.sigmoid(predictions.logits) >= 0.5).numpy().astype(int)
            
            # Get true labels
            y_true_tf = np.concatenate([y for x, y in tf_test_dataset], axis=0)
            
            # Classification report
            from sklearn.metrics import classification_report, accuracy_score, f1_score
            
            accuracy = accuracy_score(y_true_tf, y_pred_tf)
            f1_micro = f1_score(y_true_tf, y_pred_tf, average='micro')
            f1_macro = f1_score(y_true_tf, y_pred_tf, average='macro')
            
            print(f"\nğŸ“Š Detailed Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1-Score (Micro): {f1_micro:.4f}")
            print(f"F1-Score (Macro): {f1_macro:.4f}")
            
            print("\nPer-category Classification Report:")
            print(classification_report(
                y_true_tf, y_pred_tf,
                target_names=dasp_categories,
                zero_division=0
            ))
            
        else:
            print("â�Œ Tokenized datasets not available for TensorFlow training")
            print("ğŸ’¡ Set TOKENIZE_TRUNCATE=True to enable TensorFlow training")
        
    except ImportError as e:
        print(f"âš ï¸� TensorFlow not available: {e}")
        print("ğŸ“¦ Install with: pip install tensorflow transformers")
    except Exception as e:
        print(f"â�Œ TensorFlow training failed: {e}")
        import traceback
        traceback.print_exc()

else:
    print(f"â�­ï¸� Skipping TensorFlow training (selected framework: {TRAINING_FRAMEWORK})")


from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    f1_score, precision_score, recall_score, hamming_loss
)
import seaborn as sns

def hamming_score(y_true, y_pred):
    """
    Compute the Hamming score (label-based accuracy) for multi-label classification
    """
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1.0
        else:
            tmp_a = len(set_true.intersection(set_pred)) / len(set_true.union(set_pred))
        acc_list.append(tmp_a)
    
    return np.mean(acc_list)

def evaluate_multilabel_model(y_true, y_pred, category_names):
    """
    Comprehensive evaluation of multi-label classification model
    """
    print("ğŸ�¯ Multi-Label Classification Evaluation")
    print("=" * 60)
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    hamming = hamming_loss(y_true, y_pred)
    hamming_acc = hamming_score(y_true, y_pred)
    
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    precision_micro = precision_score(y_true, y_pred, average='micro')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    
    recall_micro = recall_score(y_true, y_pred, average='micro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    
    print(f"ğŸ“Š Overall Metrics:")
    print(f"  Subset Accuracy (Exact Match): {accuracy:.4f}")
    print(f"  Hamming Score (Label-based):    {hamming_acc:.4f}")
    print(f"  Hamming Loss:                   {hamming:.4f}")
    print(f"  F1-Score (Micro):               {f1_micro:.4f}")
    print(f"  F1-Score (Macro):               {f1_macro:.4f}")
    print(f"  F1-Score (Weighted):            {f1_weighted:.4f}")
    print(f"  Precision (Micro):              {precision_micro:.4f}")
    print(f"  Precision (Macro):              {precision_macro:.4f}")
    print(f"  Recall (Micro):                 {recall_micro:.4f}")
    print(f"  Recall (Macro):                 {recall_macro:.4f}")
    
    # Per-category analysis
    print(f"\\nğŸ“‹ Per-Category Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=category_names,
        zero_division=0
    ))
    
    # Category-wise confusion matrices
    print(f"\\nğŸ”� Category-wise Analysis:")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, category in enumerate(category_names):
        if i < len(axes):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Safe', 'Vulnerable'],
                yticklabels=['Safe', 'Vulnerable'],
                ax=axes[i]
            )
            axes[i].set_title(f'{category}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
    
    # Hide empty subplots
    for i in range(len(category_names), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Label distribution analysis
    print(f"\\nğŸ“ˆ Label Distribution Analysis:")
    
    true_counts = y_true.sum(axis=0)
    pred_counts = y_pred.sum(axis=0)
    
    distribution_df = pd.DataFrame({
        'Category': category_names,
        'True_Positives': true_counts,
        'Predicted_Positives': pred_counts,
        'True_Rate': true_counts / len(y_true) * 100,
        'Predicted_Rate': pred_counts / len(y_pred) * 100
    })
    
    print(distribution_df.to_string(index=False, float_format='%.2f'))
    
    # Visualization of prediction distribution
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    x = np.arange(len(category_names))
    width = 0.35
    
    plt.bar(x - width/2, true_counts, width, label='True Positives', alpha=0.8, color='green')
    plt.bar(x + width/2, pred_counts, width, label='Predicted Positives', alpha=0.8, color='blue')
    
    plt.xlabel('Categories')
    plt.ylabel('Count')
    plt.title('True vs Predicted Positive Counts')
    plt.xticks(x, category_names, rotation=45, ha='right')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    error_rates = np.abs(true_counts - pred_counts) / true_counts * 100
    error_rates = np.where(np.isnan(error_rates), 0, error_rates)  # Handle division by zero
    
    colors = plt.cm.Reds(error_rates / error_rates.max()) if error_rates.max() > 0 else ['blue'] * len(category_names)
    bars = plt.bar(category_names, error_rates, color=colors, alpha=0.8)
    
    plt.xlabel('Categories')
    plt.ylabel('Prediction Error Rate (%)')
    plt.title('Prediction Error Rate by Category')
    plt.xticks(rotation=45, ha='right')
    
    # Add error rate labels on bars
    for bar, error_rate in zip(bars, error_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{error_rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'accuracy': accuracy,
        'hamming_score': hamming_acc,
        'hamming_loss': hamming,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro
    }

# Evaluate based on the trained framework
evaluation_results = None

if TRAINING_FRAMEWORK == "belt" and 'y_pred_belt_np' in locals():
    print("ğŸ”� Evaluating BELT Model Results...")
    evaluation_results = evaluate_multilabel_model(y_test_np, y_pred_belt_np, dasp_categories)

elif TRAINING_FRAMEWORK == "pytorch" and 'y_pred_pytorch' in locals():
    print("ğŸ”� Evaluating PyTorch Model Results...")
    evaluation_results = evaluate_multilabel_model(y_true_pytorch, y_pred_pytorch, dasp_categories)

elif TRAINING_FRAMEWORK == "tensorflow" and 'y_pred_tf' in locals():
    print("ğŸ”� Evaluating TensorFlow Model Results...")
    evaluation_results = evaluate_multilabel_model(y_true_tf, y_pred_tf, dasp_categories)

else:
    print(f"âš ï¸� No evaluation data available for {TRAINING_FRAMEWORK} framework")
    print("ğŸ’¡ Make sure to run the training section first")

if evaluation_results:
    print(f"\\nâœ… Evaluation completed for {TRAINING_FRAMEWORK} framework")
    print(f"ğŸ�† Best metric - F1 Micro: {evaluation_results['f1_micro']:.4f}")
else:
    print("â�Œ No evaluation performed")


def predict_vulnerabilities(contract_code, model, framework, tokenizer=None, categories=None):
    """
    Predict vulnerabilities in a smart contract
    
    Args:
        contract_code: Raw Solidity source code
        model: Trained model
        framework: Training framework used ('belt', 'pytorch', 'tensorflow')
        tokenizer: Tokenizer (for pytorch/tensorflow)
        categories: List of category names
    
    Returns:
        Dictionary with predictions and probabilities
    """
    
    # Preprocess the contract
    preprocessed_code = preprocess_solidity_code(contract_code)
    
    if framework == "belt":
        # BELT prediction
        probabilities = model.predict_scores([preprocessed_code], verbose=False)
        predictions = model.predict([preprocessed_code], verbose=False)
        
        probs = probabilities.cpu().numpy()[0]
        preds = predictions.cpu().numpy()[0]
        
    elif framework == "pytorch":
        # PyTorch prediction
        inputs = tokenizer(
            preprocessed_code,
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        probabilities = torch.nn.Sigmoid()(logits)
        predictions = (probabilities >= 0.5).int()
        
        probs = probabilities.cpu().numpy()[0]
        preds = predictions.cpu().numpy()[0]
        
    elif framework == "tensorflow":
        # TensorFlow prediction
        inputs = tokenizer(
            preprocessed_code,
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="tf"
        )
        
        logits = model(inputs).logits
        probabilities = tf.nn.sigmoid(logits)
        predictions = (probabilities >= 0.5).numpy().astype(int)
        
        probs = probabilities.numpy()[0]
        preds = predictions[0]
    
    else:
        raise ValueError(f"Unknown framework: {framework}")
    
    # Format results
    results = {
        'predictions': {},
        'probabilities': {},
        'vulnerable_categories': [],
        'safe_categories': []
    }
    
    for i, category in enumerate(categories):
        results['predictions'][category] = bool(preds[i])
        results['probabilities'][category] = float(probs[i])
        
        if preds[i]:
            results['vulnerable_categories'].append(category)
        else:
            results['safe_categories'].append(category)
    
    return results

# Test contracts
test_contracts = {
    "Simple Storage": '''
    pragma solidity ^0.8.0;
    
    contract SimpleStorage {
        uint256 public storedData;
        
        function set(uint256 x) public {
            storedData = x;
        }
        
        function get() public view returns (uint256) {
            return storedData;
        }
    }
    ''',
    
    "Overflow Vulnerable": '''
    pragma solidity ^0.4.24;
    
    contract OverflowExample {
        uint256 public balance;
        
        function addBalance(uint256 amount) public {
            balance += amount; // Potential overflow
        }
        
        function withdraw(uint256 amount) public {
            balance -= amount; // Potential underflow
        }
    }
    ''',
    
    "Reentrancy Vulnerable": '''
    pragma solidity ^0.8.0;
    
    contract ReentrancyVulnerable {
        mapping(address => uint256) public balances;
        
        function withdraw() public {
            uint256 amount = balances[msg.sender];
            (bool success, ) = msg.sender.call{value: amount}("");
            require(success, "Transfer failed");
            balances[msg.sender] = 0; // State change after external call
        }
        
        function deposit() public payable {
            balances[msg.sender] += msg.value;
        }
    }
    ''',
    
    "Timestamp Dependence": '''
    pragma solidity ^0.8.0;
    
    contract TimestampDependence {
        uint256 public lastAction;
        
        function performAction() public {
            require(block.timestamp > lastAction + 1 hours, "Too soon");
            lastAction = block.timestamp;
            // Some action that depends on timestamp
        }
    }
    '''
}

# Perform inference based on the trained framework
if TRAINING_FRAMEWORK == "belt" and 'belt_model' in locals():
    model_to_use = belt_model
    tokenizer_to_use = None
elif TRAINING_FRAMEWORK == "pytorch" and 'pytorch_model' in locals():
    model_to_use = pytorch_model
    tokenizer_to_use = tokenizer
elif TRAINING_FRAMEWORK == "tensorflow" and 'tf_model' in locals():
    model_to_use = tf_model
    tokenizer_to_use = tokenizer
else:
    model_to_use = None
    tokenizer_to_use = None

if model_to_use is not None and 'dasp_categories' in locals():
    print("ğŸ”® Testing Model on Sample Contracts")
    print("=" * 60)
    
    for contract_name, contract_code in test_contracts.items():
        print(f"\\nğŸ“‹ Contract: {contract_name}")
        print("-" * 40)
        
        try:
            results = predict_vulnerabilities(
                contract_code, 
                model_to_use, 
                TRAINING_FRAMEWORK,
                tokenizer_to_use,
                dasp_categories
            )
            
            print("ğŸ�¯ Vulnerability Predictions:")
            for category in dasp_categories:
                is_vulnerable = results['predictions'][category]
                probability = results['probabilities'][category]
                status = "ğŸ”´ VULNERABLE" if is_vulnerable else "ğŸŸ¢ SAFE"
                print(f"  {category:25}: {status} (confidence: {probability:.3f})")
            
            if results['vulnerable_categories']:
                print(f"\\nâš ï¸� Detected vulnerabilities: {', '.join(results['vulnerable_categories'])}")
            else:
                print(f"\\nâœ… No vulnerabilities detected")
                
        except Exception as e:
            print(f"â�Œ Prediction failed: {e}")
    
    # Summary of model capabilities
    print(f"\\nğŸ“Š Model Summary:")
    print("=" * 40)
    print(f"Framework: {TRAINING_FRAMEWORK}")
    print(f"Model: {transformers_model}")
    print(f"Categories: {len(dasp_categories)}")
    print(f"DASP Categories: {', '.join(dasp_categories)}")
    
    if evaluation_results:
        print(f"\\nğŸ�† Performance Metrics:")
        for metric, value in evaluation_results.items():
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")

else:
    print("â�Œ Cannot perform inference - model not available")
    print(f"ğŸ’¡ Current framework: {TRAINING_FRAMEWORK}")
    print("ğŸ”„ Please run the training section first")

# Memory cleanup
print(f"\\nğŸ§¹ Final Memory Cleanup...")
clear_memory()
print("âœ… Notebook execution completed!")

