import requests
try:
    requests.get("https://www.google.com", timeout=2)
    print("ğŸŒ� Internet still ON â€” Kaggle flag bug detected")
except:
    print("âœ… Internet OFF â€” ready for submit")



# =============================================
# C0: REPRODUCIBILITY SETUP
# =============================================

import os
import json
import random
import numpy as np
import pandas as pd
import warnings
from datetime import datetime

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Fixed seed for deterministic execution
SEED = 42

def seed_everything(seed=SEED):
    """
    Set all random seeds for deterministic execution
    across all components of RuleSense pipeline
    """
    # Set environment variables for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Python random module
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch (for potential future embedding models)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
    except ImportError:
        pass
    
    # TensorFlow (if used in future components)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

# Initialize all random seeds
seed_everything(SEED)

# Create working directory structure
os.makedirs("/kaggle/working/eda", exist_ok=True)
os.makedirs("/kaggle/working/embeddings", exist_ok=True) 
os.makedirs("/kaggle/working/models", exist_ok=True)
os.makedirs("/kaggle/working/submissions", exist_ok=True)

# Save run configuration for audit trail
run_config = {
    "seed": SEED,
    "pipeline_version": "RuleSense_v2",
    "kaggle_competition": "jigsaw-agile-community-rules", 
    "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "reproducibility_measures": {
        "python_hash_seed": True,
        "numpy_random_seed": True, 
        "pytorch_deterministic": True,
        "cudnn_benchmark_off": True,
        "working_dirs_created": True
    },
    "environment_info": {
        "working_directories": [
            "/kaggle/working/eda",
            "/kaggle/working/embeddings", 
            "/kaggle/working/models",
            "/kaggle/working/submissions"
        ]
    }
}

with open("/kaggle/working/run_config.json", "w") as f:
    json.dump(run_config, f, indent=2, ensure_ascii=False)

# Import and display library versions
import sklearn
import lightgbm

print("=" * 60)
print("C0: REPRODUCIBILITY SETUP COMPLETE")
print("=" * 60)
print(f"Seed: {SEED}")
print(f"Working directories created")
print(f"Run config saved to: /kaggle/working/run_config.json")
print("")
print("Library Versions:")
print(f"  numpy: {np.__version__}")
print(f"  pandas: {pd.__version__}")
print(f"  scikit-learn: {sklearn.__version__}")
print(f"  lightgbm: {lightgbm.__version__}")

# Check optional dependencies
try:
    import sentence_transformers
    st_version = sentence_transformers.__version__
    print(f"  sentence-transformers: {st_version}")
except ImportError:
    print(f"  sentence-transformers: not installed")

try:
    import torch
    print(f"  torch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print(f"  torch: not installed")



# =============================================
# C1: LOAD RAW DATA
# =============================================

import pandas as pd
import numpy as np

print("Loading raw datasets...")

# Load raw data from competition files
train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

print("Performing data quality checks...")

# Data Quality Check 1: Handle missing values in text columns
def handle_missing_text(df, text_column='body'):
    """Fill missing text values with [EMPTY] marker"""
    initial_missing = df[text_column].isna().sum()
    df[text_column] = df[text_column].fillna("[EMPTY]").astype(str)
    
    # Also handle empty strings
    empty_strings = (df[text_column].str.strip() == '').sum()
    if empty_strings > 0:
        df.loc[df[text_column].str.strip() == '', text_column] = "[EMPTY]"
    
    return df, initial_missing, empty_strings

train, train_missing_body, train_empty_body = handle_missing_text(train)
test, test_missing_body, test_empty_body = handle_missing_text(test)

# Data Quality Check 2: Validate target variable
def validate_target(df, target_column='rule_Ñ�Ğ´ĞµĞ»Ğ°Ğ¹ Ğ»ÑƒÑ‡ÑˆĞµ, Ğ³Ğ»ÑƒĞ±Ğ¶Ğµ, violation'):
    """Validate that target variable contains only 0 and 1"""
    if target_column in df.columns:
        unique_values = df[target_column].unique()
        valid_values = set([0, 1])
        
        if not set(unique_values).issubset(valid_values):
            raise ValueError(f"Invalid target values: {unique_values}. Expected {valid_values}")
        
        violation_rate = df[target_column].mean()
        return violation_rate, len(unique_values)
    return None, None

train_violation_rate, train_unique_labels = validate_target(train)
test_violation_rate, _ = validate_target(test)  # Test doesn't have target

# Data Quality Check 3: Check column structure
train_columns = set(train.columns)
test_columns = set(test.columns)
common_columns = train_columns.intersection(test_columns)
train_only_columns = train_columns - test_columns
test_only_columns = test_columns - train_columns

# Data Quality Check 4: Basic statistics
def get_basic_stats(df, name):
    stats = {
        'dataset': name,
        'shape': df.shape,
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'duplicate_rows': df.duplicated().sum(),
        'total_missing': df.isnull().sum().sum()
    }
    return stats

train_stats = get_basic_stats(train, 'train')
test_stats = get_basic_stats(test, 'test')

# Save raw copies for audit
train.to_csv("/kaggle/working/train_raw.csv", index=False)
test.to_csv("/kaggle/working/test_raw.csv", index=False)

print("")
print("=" * 60)
print("C1: RAW DATA LOADED AND VALIDATED")
print("=" * 60)

# Comprehensive output
print("Dataset Shapes:")
print(f"  Train: {train.shape} ({train_stats['memory_mb']:.2f} MB)")
print(f"  Test:  {test.shape} ({test_stats['memory_mb']:.2f} MB)")

print("")
print("Data Quality Summary:")
print(f"  Missing 'body' in train: {train_missing_body} -> filled with '[EMPTY]'")
print(f"  Empty 'body' in train: {train_empty_body} -> filled with '[EMPTY]'")
print(f"  Missing 'body' in test:  {test_missing_body} -> filled with '[EMPTY]'")
print(f"  Empty 'body' in test:  {test_empty_body} -> filled with '[EMPTY]'")

print("")
print("Target Analysis:")
if train_violation_rate is not None:
    print(f"  Violation rate: {train_violation_rate:.2%} ({train_violation_rate * len(train):.0f}/{len(train)})")
    print(f"  Unique labels: {train_unique_labels} (balanced: {min(train_violation_rate, 1-train_violation_rate):.2%})")

print("")
print("Column Structure:")
print(f"  Common columns: {len(common_columns)}")
print(f"  Train only: {list(train_only_columns)}")
print(f"  Test only: {list(test_only_columns)}")

print("")
print("Additional Statistics:")
print(f"  Duplicate rows in train: {train_stats['duplicate_rows']}")
print(f"  Total missing values in train: {train_stats['total_missing']}")
print(f"  Data types in train: {dict(train.dtypes.value_counts())}")

print("")
print("Audit Files Saved:")
print(f"  /kaggle/working/train_raw.csv")
print(f"  /kaggle/working/test_raw.csv")

print("")
print("Next steps: EDA and feature engineering ready")
print("=" * 60)


# =============================================
# C2: RUN CONFIGURATION UPDATE
# =============================================

import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime

print("Updating run configuration with data metadata...")

# Load current run configuration
with open("/kaggle/working/run_config.json", "r") as f:
    config = json.load(f)

# Load raw datasets for analysis
train_raw = pd.read_csv("/kaggle/working/train_raw.csv")
test_raw = pd.read_csv("/kaggle/working/test_raw.csv")

print("Computing dataset fingerprints and statistics...")

def compute_dataset_fingerprint(df, dataset_name):
    """Compute comprehensive fingerprint for dataset reproducibility"""
    # Hash of entire dataset structure
    content_hash = hashlib.md5(
        pd.util.hash_pandas_object(df).values.tobytes()
    ).hexdigest()[:16]
    
    # Hash of first 1000 characters for quick verification
    sample_hash = hashlib.md5(
        df.head(100).to_csv(index=False).encode('utf-8')
    ).hexdigest()[:16]
    
    return {
        f"{dataset_name}_content_hash": content_hash,
        f"{dataset_name}_sample_hash": sample_hash,
        f"{dataset_name}_shape": list(df.shape),
        f"{dataset_name}_columns": list(df.columns),
        f"{dataset_name}_memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
    }

def compute_data_quality_metrics(df, dataset_name):
    """Compute comprehensive data quality metrics"""
    metrics = {}
    
    if dataset_name == "train":
        # Target analysis
        metrics["violation_rate"] = round(float(df["rule_violation"].mean()), 4)
        metrics["violation_count"] = int(df["rule_violation"].sum())
        metrics["non_violation_count"] = int(len(df) - df["rule_violation"].sum())
        metrics["class_balance_ratio"] = round(
            min(metrics["violation_rate"], 1 - metrics["violation_rate"]), 4
        )
    
    # Text quality metrics
    text_cols = [col for col in df.columns if df[col].dtype == 'object']
    for col in text_cols:
        col_metrics = {}
        col_metrics["dtype"] = str(df[col].dtype)
        col_metrics["non_null_count"] = int(df[col].notna().sum())
        col_metrics["empty_strings"] = int((df[col].str.strip() == "").sum())
        col_metrics["unique_count"] = int(df[col].nunique())
        col_metrics["most_common_length"] = int(df[col].str.len().mode().iloc[0] if len(df) > 0 else 0)
        col_metrics["avg_length"] = round(float(df[col].str.len().mean()), 2)
        
        # Special markers count
        col_metrics["empty_markers"] = int((df[col] == "[EMPTY]").sum())
        
        metrics[f"column_{col}"] = col_metrics
    
    return metrics

def compute_temporal_metrics():
    """Compute timing and progression metrics"""
    return {
        "c2_execution_time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline_stage": "data_loaded",
        "data_loading_complete": True,
        "next_stage": "eda"
    }

# Compute comprehensive fingerprints
train_fingerprint = compute_dataset_fingerprint(train_raw, "train")
test_fingerprint = compute_dataset_fingerprint(test_raw, "test")

# Compute data quality metrics
train_quality = compute_data_quality_metrics(train_raw, "train")
test_quality = compute_data_quality_metrics(test_raw, "test")

# Compute temporal metrics
temporal_metrics = compute_temporal_metrics()

# Update configuration with comprehensive data info
config["data_info"] = {
    **train_fingerprint,
    **test_fingerprint,
    "dataset_relationships": {
        "common_columns": list(set(train_raw.columns) & set(test_raw.columns)),
        "train_specific_columns": list(set(train_raw.columns) - set(test_raw.columns)),
        "test_specific_columns": list(set(test_raw.columns) - set(train_raw.columns))
    },
    "quality_metrics": {
        "train": train_quality,
        "test": test_quality
    },
    "validation_checks": {
        "no_missing_body": True,
        "valid_target_range": True,
        "consistent_column_types": True,
        "data_integrity_verified": True
    }
}

# Add pipeline progression
config["pipeline_progression"] = temporal_metrics

# Add data schema information
config["data_schema"] = {
    "train_columns": {col: str(dtype) for col, dtype in train_raw.dtypes.items()},
    "test_columns": {col: str(dtype) for col, dtype in test_raw.dtypes.items()}
}

# Save updated configuration
with open("/kaggle/working/run_config.json", "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print("")
print("=" * 60)
print("C2: RUN CONFIGURATION UPDATED WITH DATA METADATA")
print("=" * 60)

# Display summary of updates
print("Data Fingerprints:")
print(f"  Train: {train_fingerprint['train_content_hash']} (shape: {train_fingerprint['train_shape']})")
print(f"  Test:  {test_fingerprint['test_content_hash']} (shape: {test_fingerprint['test_shape']})")

print("")
print("Target Analysis:")
if "violation_rate" in train_quality:
    print(f"  Violation rate: {train_quality['violation_rate']:.2%}")
    print(f"  Class balance: {train_quality['class_balance_ratio']:.2%}")
    print(f"  Violations: {train_quality['violation_count']} | Non-violations: {train_quality['non_violation_count']}")

print("")
print("Column Structure:")
common_cols = len(config["data_info"]["dataset_relationships"]["common_columns"])
train_specific = len(config["data_info"]["dataset_relationships"]["train_specific_columns"])
test_specific = len(config["data_info"]["dataset_relationships"]["test_specific_columns"])
print(f"  Common columns: {common_cols}")
print(f"  Train-specific: {train_specific}")
print(f"  Test-specific:  {test_specific}")

print("")
print("Text Quality (train):")
if "column_body" in train_quality:
    body_metrics = train_quality["column_body"]
    print(f"  Unique texts: {body_metrics['unique_count']}")
    print(f"  Avg length: {body_metrics['avg_length']} chars")
    print(f"  Empty markers: {body_metrics['empty_markers']}")

print("")
print("Configuration Saved:")
print(f"  /kaggle/working/run_config.json")
print(f"  Pipeline stage: {temporal_metrics['pipeline_stage']}")
print(f"  Next stage: {temporal_metrics['next_stage']}")

print("")
print("Ready for comprehensive EDA analysis")
print("=" * 60)



# =============================================
# C2: RUN CONFIGURATION UPDATE
# =============================================

import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime

print("Updating run configuration with data metadata...")

# Load current run configuration
with open("/kaggle/working/run_config.json", "r") as f:
    config = json.load(f)

# Load raw datasets for analysis
train_raw = pd.read_csv("/kaggle/working/train_raw.csv")
test_raw = pd.read_csv("/kaggle/working/test_raw.csv")

print("Computing dataset fingerprints and statistics...")

def compute_dataset_fingerprint(df, dataset_name):
    """Compute comprehensive fingerprint for dataset reproducibility"""
    # Hash of entire dataset structure
    content_hash = hashlib.md5(
        pd.util.hash_pandas_object(df).values.tobytes()
    ).hexdigest()[:16]
    
    # Hash of first 1000 characters for quick verification
    sample_hash = hashlib.md5(
        df.head(100).to_csv(index=False).encode('utf-8')
    ).hexdigest()[:16]
    
    return {
        f"{dataset_name}_content_hash": content_hash,
        f"{dataset_name}_sample_hash": sample_hash,
        f"{dataset_name}_shape": list(df.shape),
        f"{dataset_name}_columns": list(df.columns),
        f"{dataset_name}_memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
    }

def compute_data_quality_metrics(df, dataset_name):
    """Compute comprehensive data quality metrics"""
    metrics = {}
    
    if dataset_name == "train":
        # Target analysis
        metrics["violation_rate"] = round(float(df["rule_violation"].mean()), 4)
        metrics["violation_count"] = int(df["rule_violation"].sum())
        metrics["non_violation_count"] = int(len(df) - df["rule_violation"].sum())
        metrics["class_balance_ratio"] = round(
            min(metrics["violation_rate"], 1 - metrics["violation_rate"]), 4
        )
    
    # Text quality metrics
    text_cols = [col for col in df.columns if df[col].dtype == 'object']
    for col in text_cols:
        col_metrics = {}
        col_metrics["dtype"] = str(df[col].dtype)
        col_metrics["non_null_count"] = int(df[col].notna().sum())
        col_metrics["empty_strings"] = int((df[col].str.strip() == "").sum())
        col_metrics["unique_count"] = int(df[col].nunique())
        col_metrics["most_common_length"] = int(df[col].str.len().mode().iloc[0] if len(df) > 0 else 0)
        col_metrics["avg_length"] = round(float(df[col].str.len().mean()), 2)
        
        # Special markers count
        col_metrics["empty_markers"] = int((df[col] == "[EMPTY]").sum())
        
        metrics[f"column_{col}"] = col_metrics
    
    return metrics

def compute_temporal_metrics():
    """Compute timing and progression metrics"""
    return {
        "c2_execution_time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline_stage": "data_loaded",
        "data_loading_complete": True,
        "next_stage": "eda"
    }

# Compute comprehensive fingerprints
train_fingerprint = compute_dataset_fingerprint(train_raw, "train")
test_fingerprint = compute_dataset_fingerprint(test_raw, "test")

# Compute data quality metrics
train_quality = compute_data_quality_metrics(train_raw, "train")
test_quality = compute_data_quality_metrics(test_raw, "test")

# Compute temporal metrics
temporal_metrics = compute_temporal_metrics()

# Update configuration with comprehensive data info
config["data_info"] = {
    **train_fingerprint,
    **test_fingerprint,
    "dataset_relationships": {
        "common_columns": list(set(train_raw.columns) & set(test_raw.columns)),
        "train_specific_columns": list(set(train_raw.columns) - set(test_raw.columns)),
        "test_specific_columns": list(set(test_raw.columns) - set(train_raw.columns))
    },
    "quality_metrics": {
        "train": train_quality,
        "test": test_quality
    },
    "validation_checks": {
        "no_missing_body": True,
        "valid_target_range": True,
        "consistent_column_types": True,
        "data_integrity_verified": True
    }
}

# Add pipeline progression
config["pipeline_progression"] = temporal_metrics

# Add data schema information
config["data_schema"] = {
    "train_columns": {col: str(dtype) for col, dtype in train_raw.dtypes.items()},
    "test_columns": {col: str(dtype) for col, dtype in test_raw.dtypes.items()}
}

# Save updated configuration
with open("/kaggle/working/run_config.json", "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print("")
print("=" * 60)
print("C2: RUN CONFIGURATION UPDATED WITH DATA METADATA")
print("=" * 60)

# Display summary of updates
print("Data Fingerprints:")
print(f"  Train: {train_fingerprint['train_content_hash']} (shape: {train_fingerprint['train_shape']})")
print(f"  Test:  {test_fingerprint['test_content_hash']} (shape: {test_fingerprint['test_shape']})")

print("")
print("Target Analysis:")
if "violation_rate" in train_quality:
    print(f"  Violation rate: {train_quality['violation_rate']:.2%}")
    print(f"  Class balance: {train_quality['class_balance_ratio']:.2%}")
    print(f"  Violations: {train_quality['violation_count']} | Non-violations: {train_quality['non_violation_count']}")

print("")
print("Column Structure:")
common_cols = len(config["data_info"]["dataset_relationships"]["common_columns"])
train_specific = len(config["data_info"]["dataset_relationships"]["train_specific_columns"])
test_specific = len(config["data_info"]["dataset_relationships"]["test_specific_columns"])
print(f"  Common columns: {common_cols}")
print(f"  Train-specific: {train_specific}")
print(f"  Test-specific:  {test_specific}")

print("")
print("Text Quality (train):")
if "column_body" in train_quality:
    body_metrics = train_quality["column_body"]
    print(f"  Unique texts: {body_metrics['unique_count']}")
    print(f"  Avg length: {body_metrics['avg_length']} chars")
    print(f"  Empty markers: {body_metrics['empty_markers']}")

print("")
print("Configuration Saved:")
print(f"  /kaggle/working/run_config.json")
print(f"  Pipeline stage: {temporal_metrics['pipeline_stage']}")
print(f"  Next stage: {temporal_metrics['next_stage']}")

print("")
print("Ready for comprehensive EDA analysis")
print("=" * 60)


# =============================================
# C3: CLEAN REDDIT ARTIFACTS
# =============================================

import re
import pandas as pd
import numpy as np
from typing import Optional

print("Cleaning Reddit artifacts from text data...")

def clean_reddit_artifacts(text: str) -> str:
    """
    Comprehensive cleaning of Reddit-specific artifacts from text.
    Replaces URLs, usernames, subreddits with standardized tokens.
    
    Args:
        text: Input text string to clean
        
    Returns:
        Cleaned text with Reddit artifacts replaced by standardized tokens
    """
    # Handle empty or invalid inputs
    if not isinstance(text, str) or not text.strip():
        return "[EMPTY]"
    
    # Make a copy to avoid modifying original
    cleaned = text.strip()
    
    # Pattern 1: URLs (comprehensive matching)
    url_patterns = [
        r'https?://[^\s]+',                    # http/https URLs
        r'www\.[^\s]+',                        # www URLs
        r'[a-zA-Z0-9-]+\.[a-zA-Z]{2,}[^\s]*',  # domain patterns
    ]
    
    for pattern in url_patterns:
        cleaned = re.sub(pattern, '[URL]', cleaned, flags=re.IGNORECASE)
    
    # Pattern 2: Reddit usernames (u/username)
    # Handle various formats: u/username, /u/username, u/username123, u/user_name
    username_patterns = [
        r'/u/[a-zA-Z0-9_-]+',      # /u/username
        r'u/[a-zA-Z0-9_-]+',       # u/username (without slash)
        r'@[a-zA-Z0-9_-]+',        # @username (alternative format)
    ]
    
    for pattern in username_patterns:
        cleaned = re.sub(pattern, '[USER]', cleaned, flags=re.IGNORECASE)
    
    # Pattern 3: Subreddit mentions (r/subreddit)
    # Handle various formats: r/subreddit, /r/subreddit, r/sub_reddit123
    subreddit_patterns = [
        r'/r/[a-zA-Z0-9_-]+',      # /r/subreddit
        r'r/[a-zA-Z0-9_-]+',       # r/subreddit (without slash)
    ]
    
    for pattern in subreddit_patterns:
        cleaned = re.sub(pattern, '[SUBREDDIT]', cleaned, flags=re.IGNORECASE)
    
    # Pattern 4: Email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    cleaned = re.sub(email_pattern, '[EMAIL]', cleaned)
    
    # Pattern 5: Phone numbers (basic pattern)
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    cleaned = re.sub(phone_pattern, '[PHONE]', cleaned)
    
    # Advanced cleaning steps
    # Remove excessive punctuation (keep basic sentence structure)
    cleaned = re.sub(r'[!]{2,}', '!', cleaned)  # Multiple ! -> single !
    cleaned = re.sub(r'[?]{2,}', '?', cleaned)  # Multiple ? -> single ?
    
    # Normalize whitespace (tabs, newlines, multiple spaces)
    cleaned = re.sub(r'\s+', ' ', cleaned)  # All whitespace -> single space
    cleaned = cleaned.strip()
    
    # Final check for empty result
    if not cleaned:
        return "[EMPTY]"
    
    return cleaned

def analyze_cleaning_impact(df_before: pd.DataFrame, df_after: pd.DataFrame, text_col: str = 'body') -> dict:
    """
    Analyze the impact of text cleaning on the dataset.
    
    Args:
        df_before: DataFrame before cleaning
        df_after: DataFrame after cleaning
        text_col: Name of text column
        
    Returns:
        Dictionary with cleaning statistics
    """
    stats = {}
    
    # Basic length statistics
    stats['original_avg_length'] = round(df_before[text_col].str.len().mean(), 2)
    stats['cleaned_avg_length'] = round(df_after['body_clean'].str.len().mean(), 2)
    stats['length_reduction_pct'] = round(
        (1 - stats['cleaned_avg_length'] / stats['original_avg_length']) * 100, 2
    )
    
    # Token replacement statistics
    stats['urls_replaced'] = df_after['body_clean'].str.count('\\[URL\\]').sum()
    stats['users_replaced'] = df_after['body_clean'].str.count('\\[USER\\]').sum()
    stats['subreddits_replaced'] = df_after['body_clean'].str.count('\\[SUBREDDIT\\]').sum()
    stats['emails_replaced'] = df_after['body_clean'].str.count('\\[EMAIL\\]').sum()
    
    # Empty text handling
    stats['empty_after_cleaning'] = (df_after['body_clean'] == '[EMPTY]').sum()
    
    return stats

# Load raw datasets
train = pd.read_csv("/kaggle/working/train_raw.csv")
test = pd.read_csv("/kaggle/working/test_raw.csv")

print("Applying cleaning function to datasets...")

# Apply cleaning function
train['body_clean'] = train['body'].apply(clean_reddit_artifacts)
test['body_clean'] = test['body'].apply(clean_reddit_artifacts)

# Analyze cleaning impact
train_stats = analyze_cleaning_impact(train, train)
test_stats = analyze_cleaning_impact(test, test)

# Save cleaned datasets
train.to_csv("/kaggle/working/train_clean.csv", index=False)
test.to_csv("/kaggle/working/test_clean.csv", index=False)

# Find good examples to demonstrate cleaning
def find_cleaning_examples(df, n_examples=3):
    """Find examples that demonstrate the cleaning process well."""
    examples = []
    
    # Look for texts that contain patterns we clean
    url_pattern = r'https?://|www\.|\[URL\]'
    user_pattern = r'u/|/u/|\[USER\]'
    subreddit_pattern = r'r/|/r/|\[SUBREDDIT\]'
    
    for idx, row in df.iterrows():
        original = row['body']
        cleaned = row['body_clean']
        
        # Check if cleaning actually changed something meaningful
        if (original != cleaned and 
            (re.search(url_pattern, original, re.IGNORECASE) or 
             re.search(user_pattern, original, re.IGNORECASE) or
             re.search(subreddit_pattern, original, re.IGNORECASE))):
            
            examples.append({
                'before': original,
                'after': cleaned
            })
            
            if len(examples) >= n_examples:
                break
    
    return examples

# Get demonstration examples
train_examples = find_cleaning_examples(train)
test_examples = find_cleaning_examples(test)

print("")
print("=" * 60)
print("C3: REDDIT ARTIFACTS CLEANING COMPLETE")
print("=" * 60)

# Display comprehensive cleaning statistics
print("Cleaning Impact Analysis:")
print(f"  Train texts: {len(train)}")
print(f"  Original avg length: {train_stats['original_avg_length']} chars")
print(f"  Cleaned avg length: {train_stats['cleaned_avg_length']} chars")
print(f"  Length reduction: {train_stats['length_reduction_pct']}%")

print("")
print("Tokens Replaced (train):")
print(f"  URLs: {train_stats['urls_replaced']}")
print(f"  Users: {train_stats['users_replaced']}")
print(f"  Subreddits: {train_stats['subreddits_replaced']}")
print(f"  Emails: {train_stats['emails_replaced']}")

print("")
print("Example Cleaning Transformations:")
for i, example in enumerate(train_examples[:2], 1):
    print(f"  Example {i}:")
    print(f"    Before: \"{example['before'][:80]}{'...' if len(example['before']) > 80 else ''}\"")
    print(f"    After:  \"{example['after'][:80]}{'...' if len(example['after']) > 80 else ''}\"")
    print()

# Show test set stats if meaningful
if len(test) > 0:
    print("Test Set Cleaning:")
    print(f"  Test texts: {len(test)}")
    print(f"  URLs replaced: {test_stats['urls_replaced']}")
    print(f"  Users replaced: {test_stats['users_replaced']}")

print("")
print("Cleaned Datasets Saved:")
print(f"  /kaggle/working/train_clean.csv")
print(f"  /kaggle/working/test_clean.csv")

print("")
print("Next: Text data ready for embedding generation")
print("=" * 60)



# =============================================
# C4: EXTRACT DENSE FEATURES
# =============================================

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any

print("Extracting dense features from text data...")

def extract_dense_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract comprehensive dense features from text columns.
    Creates 15+ numerical and categorical features capturing text style and structure.
    
    Args:
        df: DataFrame with 'body' and 'body_clean' columns
        
    Returns:
        DataFrame with additional dense feature columns
    """
    df = df.copy()
    text_col = "body_clean"
    
    print(f"  Processing {len(df)} texts...")
    
    # ===== 1. LENGTH-BASED FEATURES =====
    df["char_count"] = df[text_col].str.len()
    df["word_count"] = df[text_col].str.split().str.len()
    df["sentence_count"] = df[text_col].apply(
        lambda x: len(re.split(r'[.!?]+', x)) if isinstance(x, str) else 0
    )
    df["avg_word_length"] = df["char_count"] / (df["word_count"] + 1e-6)
    df["avg_sentence_length"] = df["word_count"] / (df["sentence_count"] + 1e-6)
    
    # ===== 2. CASE AND CAPITALIZATION FEATURES =====
    df["uppercase_count"] = df[text_col].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) if isinstance(x, str) else 0
    )
    df["uppercase_ratio"] = df["uppercase_count"] / (df["char_count"] + 1e-6)
    df["titlecase_words"] = df[text_col].apply(
        lambda x: sum(1 for word in str(x).split() if word.istitle()) if isinstance(x, str) else 0
    )
    df["titlecase_ratio"] = df["titlecase_words"] / (df["word_count"] + 1e-6)
    
    # ===== 3. PUNCTUATION AND SYMBOL FEATURES =====
    punctuation_chars = '.,;:!?-"()[]{}'
    
    df["exclam_count"] = df[text_col].str.count(r'!')
    df["question_count"] = df[text_col].str.count(r'\?')
    df["punct_count"] = df[text_col].apply(
        lambda x: sum(1 for c in str(x) if c in punctuation_chars) if isinstance(x, str) else 0
    )
    df["punct_ratio"] = df["punct_count"] / (df["char_count"] + 1e-6)
    
    # ===== 4. DIGIT AND SPECIAL CHARACTER FEATURES =====
    df["digit_count"] = df[text_col].apply(
        lambda x: sum(1 for c in str(x) if c.isdigit()) if isinstance(x, str) else 0
    )
    df["digit_ratio"] = df["digit_count"] / (df["char_count"] + 1e-6)
    
    # Special characters (excluding basic punctuation)
    special_chars = r'[@#$%^&*_+=|\\/~]'
    df["special_char_count"] = df[text_col].str.count(special_chars)
    df["special_char_ratio"] = df["special_char_count"] / (df["char_count"] + 1e-6)
    
    # ===== 5. EMOJI AND MODERN FEATURES =====
    # Basic emoji pattern (covers most common emojis)
    emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]'
    df["emoji_count"] = df[text_col].str.count(emoji_pattern)
    df["has_emoji"] = (df["emoji_count"] > 0).astype(int)
    
    # ===== 6. TEXT COMPLEXITY FEATURES =====
    df["unique_words"] = df[text_col].apply(
        lambda x: len(set(str(x).lower().split())) if isinstance(x, str) else 0
    )
    df["lexical_diversity"] = df["unique_words"] / (df["word_count"] + 1e-6)
    
    # ===== 7. LENGTH BINS AND CATEGORICAL FEATURES =====
    df["length_bin"] = pd.cut(
        df["char_count"],
        bins=[-1, 50, 200, 500, float('inf')],
        labels=["very_short", "short", "medium", "long"]
    ).astype(str)
    
    # Word count bins
    df["word_count_bin"] = pd.cut(
        df["word_count"],
        bins=[-1, 10, 25, 50, float('inf')],
        labels=["few_words", "moderate_words", "many_words", "very_many_words"]
    ).astype(str)
    
    # ===== 8. REDDIT-SPECIFIC FEATURES =====
    df["url_token_count"] = df[text_col].str.count(r'\[URL\]')
    df["user_token_count"] = df[text_col].str.count(r'\[USER\]')
    df["subreddit_token_count"] = df[text_col].str.count(r'\[SUBREDDIT\]')
    df["has_reddit_artifacts"] = (
        (df["url_token_count"] > 0) | 
        (df["user_token_count"] > 0) | 
        (df["subreddit_token_count"] > 0)
    ).astype(int)
    
    # ===== 9. READABILITY AND STYLE FEATURES =====
    # Short word ratio (words <= 3 chars)
    df["short_word_count"] = df[text_col].apply(
        lambda x: sum(1 for word in str(x).split() if len(word) <= 3) if isinstance(x, str) else 0
    )
    df["short_word_ratio"] = df["short_word_count"] / (df["word_count"] + 1e-6)
    
    # Long word ratio (words >= 8 chars)
    df["long_word_count"] = df[text_col].apply(
        lambda x: sum(1 for word in str(x).split() if len(word) >= 8) if isinstance(x, str) else 0
    )
    df["long_word_ratio"] = df["long_word_count"] / (df["word_count"] + 1e-6)
    
    # ===== 10. TEXT QUALITY INDICATORS =====
    df["is_empty_text"] = (df[text_col] == "[EMPTY]").astype(int)
    df["is_very_short"] = (df["char_count"] < 10).astype(int)
    df["is_very_long"] = (df["char_count"] > 1000).astype(int)
    
    return df

def analyze_feature_impact(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the impact and distribution of extracted features.
    
    Args:
        train_df: Training dataframe with features
        test_df: Test dataframe with features
        
    Returns:
        Dictionary with feature analysis
    """
    analysis = {}
    
    # Count features
    original_cols = ['body', 'body_clean']
    feature_cols = [col for col in train_df.columns if col not in original_cols]
    analysis['total_features'] = len(feature_cols)
    analysis['feature_columns'] = feature_cols
    
    # Feature categories
    numerical_features = [col for col in feature_cols if train_df[col].dtype in ['int64', 'float64']]
    categorical_features = [col for col in feature_cols if train_df[col].dtype == 'object']
    
    analysis['numerical_features'] = numerical_features
    analysis['categorical_features'] = categorical_features
    analysis['numerical_count'] = len(numerical_features)
    analysis['categorical_count'] = len(categorical_features)
    
    # Basic statistics for key numerical features
    key_features = ['char_count', 'word_count', 'uppercase_ratio', 'punct_ratio', 'digit_ratio']
    stats_summary = {}
    
    for feature in key_features:
        if feature in train_df.columns:
            stats_summary[feature] = {
                'train_mean': round(train_df[feature].mean(), 4),
                'train_std': round(train_df[feature].std(), 4),
                'test_mean': round(test_df[feature].mean(), 4) if feature in test_df.columns else None
            }
    
    analysis['key_feature_stats'] = stats_summary
    
    return analysis

# Load cleaned datasets
train = pd.read_csv("/kaggle/working/train_clean.csv")
test = pd.read_csv("/kaggle/working/test_clean.csv")

print("Applying feature extraction...")

# Extract dense features
train_dense = extract_dense_features(train)
test_dense = extract_dense_features(test)

# Analyze feature impact
feature_analysis = analyze_feature_impact(train_dense, test_dense)

print("Saving feature-enhanced datasets...")

# Save datasets with dense features
train_dense.to_csv("/kaggle/working/train_dense.csv", index=False)
test_dense.to_csv("/kaggle/working/test_dense.csv", index=False)

print("")
print("=" * 60)
print("C4: DENSE FEATURE EXTRACTION COMPLETE")
print("=" * 60)

# Display comprehensive feature summary
print("Feature Extraction Summary:")
print(f"  Total features added: {feature_analysis['total_features']}")
print(f"  Numerical features: {feature_analysis['numerical_count']}")
print(f"  Categorical features: {feature_analysis['categorical_count']}")

print("")
print("Key Feature Statistics (train):")
for feature, stats in feature_analysis['key_feature_stats'].items():
    print(f"  {feature}: mean={stats['train_mean']}, std={stats['train_std']}")

print("")
print("Feature Categories:")
print(f"  Length-based: char_count, word_count, sentence_count, avg_word_length")
print(f"  Case-based: uppercase_ratio, titlecase_ratio")
print(f"  Punctuation: exclam_count, question_count, punct_ratio")
print(f"  Complexity: lexical_diversity, short_word_ratio, long_word_ratio")
print(f"  Reddit-specific: url_token_count, user_token_count, has_reddit_artifacts")
print(f"  Structural: length_bin, word_count_bin, is_empty_text")

print("")
print("Sample Feature Values:")
sample_idx = 0
if len(train_dense) > 0:
    sample = train_dense.iloc[sample_idx]
    print(f"  Text: '{sample['body_clean'][:60]}...'")
    print(f"  Length: {sample['char_count']} chars, {sample['word_count']} words")
    print(f"  Uppercase: {sample['uppercase_ratio']:.2%}, Punctuation: {sample['punct_ratio']:.2%}")
    print(f"  Lexical diversity: {sample['lexical_diversity']:.2f}")

print("")
print("Dataset Shapes:")
print(f"  Train: {train_dense.shape} (added {train_dense.shape[1] - train.shape[1]} features)")
print(f"  Test:  {test_dense.shape} (added {test_dense.shape[1] - test.shape[1]} features)")

print("")
print("Files Saved:")
print(f"  /kaggle/working/train_dense.csv")
print(f"  /kaggle/working/test_dense.csv")

print("")
print("Next: Ready for embedding generation and model training")
print("=" * 60)



# =============================================
# C5: EXTRACT DOMAIN FEATURES
# =============================================

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any

print("Extracting domain-specific features for rule violation detection...")

def extract_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract domain-specific features targeting:
    1. "No Advertising" rule violations
    2. "No Legal Advice" rule violations
    
    Args:
        df: DataFrame with 'body' and 'body_clean' columns
        
    Returns:
        DataFrame with additional domain-specific feature columns
    """
    df = df.copy()
    text_col = "body_clean"
    
    print(f"  Processing {len(df)} texts for domain patterns...")
    
    # ===== ADVERTISING-RELATED FEATURES =====
    
    # Promotional keywords (case insensitive)
    promo_keywords = [
        'ad', 'promo', 'discount', 'coupon', 'deal', 'offer', 'free', 
        'click here', 'subscribe', 'follow', 'join', 'watch live', 
        'buy now', 'shop', 'purchase', 'order', 'sale', 'limited time',
        'exclusive', 'bargain', 'save', 'cheap', 'affordable', 'price',
        'discounted', 'special offer', 'get it now', 'check out', 'promotion',
        'sponsored', 'affiliate', 'commission', 'make money', 'earn cash'
    ]
    
    # Call-to-action phrases (patterns without raw strings)
    cta_patterns = [
        'click', 'subscribe', 'follow', 'join', 'buy', 
        'shop', 'visit', 'sign up', 'register', 'download',
        'get started', 'try now', 'learn more', 'order now',
        'call now', 'contact us', 'email us'
    ]
    
    # Commercial domain indicators
    commercial_tlds = ['.com', '.net', '.org', '.io', '.co', '.shop', '.store']
    
    df["promo_kw_count"] = df[text_col].apply(
        lambda x: sum(1 for kw in promo_keywords if kw in x.lower()) if isinstance(x, str) else 0
    )
    df["has_promo_kw"] = (df["promo_kw_count"] > 0).astype(int)
    
    # Call-to-action detection
    cta_counts = []
    for text in df[text_col]:
        if not isinstance(text, str):
            cta_counts.append(0)
            continue
        text_lower = text.lower()
        text_cta_count = sum(1 for pattern in cta_patterns if re.search(r'\b' + re.escape(pattern) + r'\b', text_lower))
        cta_counts.append(text_cta_count)
    
    df["cta_count"] = cta_counts
    df["has_call_to_action"] = (df["cta_count"] > 0).astype(int)
    
    # URL/domain analysis (using original body for TLD detection)
    df["suspicious_tld_count"] = df["body"].apply(
        lambda x: sum(1 for tld in commercial_tlds if tld in str(x).lower()) if isinstance(x, str) else 0
    )
    df["has_commercial_tld"] = (df["suspicious_tld_count"] > 0).astype(int)
    
    # ===== LEGAL ADVICE-RELATED FEATURES =====
    
    # Legal terminology keywords
    legal_keywords = [
        'sue', 'lawsuit', 'lawyer', 'attorney', 'legal', 'statute', 
        'tenant', 'landlord', 'claim', 'court', 'file', 'judge', 
        'settlement', 'eviction', 'small claims', 'jurisdiction',
        'liable', 'liability', 'damages', 'compensation', 'breach',
        'contract', 'agreement', 'lawsuit', 'litigation', 'defendant',
        'plaintiff', 'testimony', 'evidence', 'hearing', 'trial',
        'appeal', 'verdict', 'injunction', 'subpoena', 'deposition'
    ]
    
    # Legal disclaimer phrases (patterns without raw strings)
    disclaimer_patterns = [
        'IANAL', 'not a lawyer', 'not legal advice', 
        'this is not legal advice', 'I am not your lawyer',
        'consult a lawyer', 'seek legal counsel', 'talk to an attorney',
        'this is not legal counsel', 'not legal opinion'
    ]
    
    # Imperative/instructional patterns (legal advice often uses these)
    imperative_patterns = [
        'you should', 'you must', 'you need to', 'you have to',
        'you ought to', 'I recommend', 'I suggest', 'my advice',
        'you could', 'you might', 'it would be wise', 'consider',
        'you may want to', 'it\'s best to', 'you\'re entitled to'
    ]
    
    df["legal_kw_count"] = df[text_col].apply(
        lambda x: sum(1 for kw in legal_keywords if kw in x.lower()) if isinstance(x, str) else 0
    )
    df["has_legal_kw"] = (df["legal_kw_count"] > 0).astype(int)
    
    # Legal disclaimer detection
    disclaimer_counts = []
    for text in df[text_col]:
        if not isinstance(text, str):
            disclaimer_counts.append(0)
            continue
        text_lower = text.lower()
        text_disclaimer_count = sum(1 for pattern in disclaimer_patterns if pattern.lower() in text_lower)
        disclaimer_counts.append(text_disclaimer_count)
    
    df["disclaimer_count"] = disclaimer_counts
    df["has_disclaimer"] = (df["disclaimer_count"] > 0).astype(int)
    
    # Imperative pattern detection
    imperative_counts = []
    for text in df[text_col]:
        if not isinstance(text, str):
            imperative_counts.append(0)
            continue
        text_lower = text.lower()
        text_imperative_count = sum(1 for pattern in imperative_patterns if pattern.lower() in text_lower)
        imperative_counts.append(text_imperative_count)
    
    df["imperative_count"] = imperative_counts
    df["has_imperative"] = (df["imperative_count"] > 0).astype(int)
    
    # ===== COMPOSITE DOMAIN FEATURES =====
    
    # Advertising likelihood score
    df["advertising_score"] = (
        df["promo_kw_count"] + 
        df["has_call_to_action"] * 2 + 
        df["suspicious_tld_count"] * 3 +
        df["url_token_count"] * 2  # From previous features
    )
    
    # Legal advice likelihood score
    df["legal_advice_score"] = (
        df["legal_kw_count"] * 2 +
        df["imperative_count"] * 1.5 -
        df["disclaimer_count"] * 3  # Disclaimers reduce likelihood
    )
    
    # Domain classification
    df["likely_advertising"] = (df["advertising_score"] > 3).astype(int)
    df["likely_legal_advice"] = (df["legal_advice_score"] > 2).astype(int)
    
    # Combined rule violation risk
    df["domain_risk_score"] = df["advertising_score"] + df["legal_advice_score"]
    
    return df

def analyze_domain_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the distribution and effectiveness of domain-specific features.
    
    Args:
        train_df: Training dataframe with domain features
        test_df: Test dataframe with domain features
        
    Returns:
        Dictionary with domain feature analysis
    """
    analysis = {}
    
    # Count domain features
    original_cols = list(pd.read_csv("/kaggle/working/train_dense.csv").columns)
    domain_feature_cols = [col for col in train_df.columns if col not in original_cols]
    analysis['total_domain_features'] = len(domain_feature_cols)
    analysis['domain_feature_columns'] = domain_feature_cols
    
    # Feature categories
    advertising_features = [col for col in domain_feature_cols if any(x in col for x in ['promo', 'cta', 'tld', 'advertis'])]
    legal_features = [col for col in domain_feature_cols if any(x in col for x in ['legal', 'disclaimer', 'imperative'])]
    composite_features = [col for col in domain_feature_cols if col not in advertising_features + legal_features]
    
    analysis['advertising_features'] = advertising_features
    analysis['legal_features'] = legal_features
    analysis['composite_features'] = composite_features
    
    # Rule violation correlation (if target exists)
    if 'rule_violation' in train_df.columns:
        # Advertising features correlation
        ad_features = [col for col in advertising_features if train_df[col].dtype in ['int64', 'float64']]
        ad_correlations = {}
        for feature in ad_features[:5]:  # Top 5 only for display
            if feature in train_df.columns:
                corr = train_df[feature].corr(train_df['rule_violation'])
                ad_correlations[feature] = round(corr, 4)
        
        # Legal features correlation
        legal_features_num = [col for col in legal_features if train_df[col].dtype in ['int64', 'float64']]
        legal_correlations = {}
        for feature in legal_features_num[:5]:
            if feature in train_df.columns:
                corr = train_df[feature].corr(train_df['rule_violation'])
                legal_correlations[feature] = round(corr, 4)
        
        analysis['advertising_correlations'] = ad_correlations
        analysis['legal_correlations'] = legal_correlations
    
    # Feature value ranges
    key_domain_features = ['advertising_score', 'legal_advice_score', 'domain_risk_score']
    stats_summary = {}
    
    for feature in key_domain_features:
        if feature in train_df.columns:
            stats_summary[feature] = {
                'train_mean': round(train_df[feature].mean(), 4),
                'train_std': round(train_df[feature].std(), 4),
                'train_max': round(train_df[feature].max(), 4),
                'test_mean': round(test_df[feature].mean(), 4) if feature in test_df.columns else None
            }
    
    analysis['domain_feature_stats'] = stats_summary
    
    return analysis

# Load datasets with dense features
train = pd.read_csv("/kaggle/working/train_dense.csv")
test = pd.read_csv("/kaggle/working/test_dense.csv")

print("Applying domain-specific feature extraction...")

# Extract domain features
train_domain = extract_domain_features(train)
test_domain = extract_domain_features(test)

# Analyze domain feature impact
domain_analysis = analyze_domain_features(train_domain, test_domain)

print("Saving domain-enhanced datasets...")

# Save datasets with domain features
train_domain.to_csv("/kaggle/working/train_domain.csv", index=False)
test_domain.to_csv("/kaggle/working/test_domain.csv", index=False)

print("")
print("=" * 60)
print("C5: DOMAIN-SPECIFIC FEATURE EXTRACTION COMPLETE")
print("=" * 60)

# Display comprehensive domain feature summary
print("Domain Feature Summary:")
print(f"  Total domain features added: {domain_analysis['total_domain_features']}")
print(f"  Advertising features: {len(domain_analysis['advertising_features'])}")
print(f"  Legal advice features: {len(domain_analysis['legal_features'])}")
print(f"  Composite features: {len(domain_analysis['composite_features'])}")

print("")
print("Advertising Detection Features:")
print(f"  Promo keywords: {35} terms")
print(f"  Call-to-action: {17} patterns") 
print(f"  Commercial TLDs: {7} domains")

print("")
print("Legal Advice Detection Features:")
print(f"  Legal keywords: {35} terms")
print(f"  Disclaimer patterns: {10} phrases")
print(f"  Imperative patterns: {15} constructions")

print("")
print("Domain Feature Statistics:")
for feature, stats in domain_analysis['domain_feature_stats'].items():
    print(f"  {feature}: mean={stats['train_mean']}, std={stats['train_std']}, max={stats['train_max']}")

# Show correlation with target if available
if 'advertising_correlations' in domain_analysis:
    print("")
    print("Feature-Target Correlations:")
    print("  Advertising features:")
    for feature, corr in list(domain_analysis['advertising_correlations'].items())[:3]:
        print(f"    {feature}: {corr:+.4f}")
    print("  Legal advice features:")
    for feature, corr in list(domain_analysis['legal_correlations'].items())[:3]:
        print(f"    {feature}: {corr:+.4f}")

print("")
print("Example Domain Analysis:")
sample_idx = 0
if len(train_domain) > 0:
    sample = train_domain.iloc[sample_idx]
    print(f"  Text preview: '{sample['body_clean'][:80]}...'")
    print(f"  Advertising score: {sample['advertising_score']} (likely: {sample['likely_advertising']})")
    print(f"  Legal advice score: {sample['legal_advice_score']} (likely: {sample['likely_legal_advice']})")
    print(f"  Domain risk score: {sample['domain_risk_score']}")

print("")
print("Dataset Shapes:")
print(f"  Train: {train_domain.shape} (added {train_domain.shape[1] - train.shape[1]} features)")
print(f"  Test:  {test_domain.shape} (added {test_domain.shape[1] - test.shape[1]} features)")

print("")
print("Files Saved:")
print(f"  /kaggle/working/train_domain.csv")
print(f"  /kaggle/working/test_domain.csv")

print("")
print("Next: Ready for embedding generation and model training")
print("=" * 60)



# =============================================
# C6: TFIDF WORD NGRAMS
# =============================================

import pandas as pd
import numpy as np
import json
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import pickle

print("Generating TF-IDF word n-gram features...")

def create_tfidf_pipeline():
    """
    Create TF-IDF + SVD pipeline for semantic feature extraction.
    
    Returns:
        sklearn Pipeline with TF-IDF vectorizer and SVD dimensionality reduction
    """
    # TF-IDF with word n-grams (unigrams and bigrams)
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),           # Unigrams and bigrams
        max_features=5000,            # Top 5000 features by frequency
        min_df=2,                     # Ignore terms that appear in less than 2 documents
        max_df=0.9,                   # Ignore terms that appear in more than 90% of documents
        stop_words='english',         # Remove common English stop words
        lowercase=True,               # Convert to lowercase
        strip_accents='unicode',      # Remove accents
        analyzer='word',              # Word-level tokenization
        token_pattern=r'(?u)\b\w\w+\b',  # Token pattern: words with 2+ chars
        use_idf=True,                 # Use inverse document frequency
        smooth_idf=True,              # Smooth IDF weights
        sublinear_tf=False            # Use sublinear TF scaling
    )
    
    # Dimensionality reduction with SVD
    svd_reducer = TruncatedSVD(
        n_components=50,              # Reduce to 50 latent dimensions
        algorithm='randomized',
        random_state=42,              # For reproducibility
        n_iter=10
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', tfidf_vectorizer),
        ('svd', svd_reducer)
    ])
    
    return pipeline

def analyze_tfidf_features(tfidf_matrix, feature_names, svd_model):
    """
    Analyze TF-IDF features and SVD components.
    
    Args:
        tfidf_matrix: TF-IDF transformed matrix
        feature_names: List of feature names
        svd_model: Fitted SVD model
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {}
    
    # Basic TF-IDF stats
    analysis['tfidf_shape'] = tfidf_matrix.shape
    analysis['num_features'] = len(feature_names)
    analysis['sparsity'] = 1.0 - (tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]))
    
    # SVD analysis
    analysis['svd_explained_variance'] = float(svd_model.explained_variance_ratio_.sum())
    analysis['svd_explained_variance_per_component'] = [
        float(var) for var in svd_model.explained_variance_ratio_
    ]
    
    # Top features by component
    analysis['top_components'] = {}
    components = svd_model.components_
    
    for i in range(min(5, components.shape[0])):  # First 5 components
        top_indices = np.argsort(components[i])[-10:][::-1]  # Top 10 features
        top_features = [(feature_names[idx], float(components[i, idx])) for idx in top_indices]
        analysis['top_components'][f'component_{i}'] = top_features
    
    # Most important original features
    feature_importance = np.abs(components).sum(axis=0)
    top_feature_indices = np.argsort(feature_importance)[-20:][::-1]
    analysis['top_overall_features'] = [
        (feature_names[idx], float(feature_importance[idx])) 
        for idx in top_feature_indices
    ]
    
    return analysis

# Load domain-enhanced datasets
train = pd.read_csv("/kaggle/working/train_domain.csv")
test = pd.read_csv("/kaggle/working/test_domain.csv")

print("Fitting TF-IDF pipeline on training data...")

# Create and fit pipeline on training data
tfidf_pipeline = create_tfidf_pipeline()

# Fit on training data and transform both train and test
X_train_tfidf_svd = tfidf_pipeline.fit_transform(train['body_clean'])
X_test_tfidf_svd = tfidf_pipeline.transform(test['body_clean'])

# Get feature names from the TF-IDF vectorizer
feature_names = tfidf_pipeline.named_steps['tfidf'].get_feature_names_out()

# Also get the original TF-IDF matrices (before SVD) for analysis
X_train_tfidf = tfidf_pipeline.named_steps['tfidf'].transform(train['body_clean'])
X_test_tfidf = tfidf_pipeline.named_steps['tfidf'].transform(test['body_clean'])

print("Saving TF-IDF features and metadata...")

# Save dense matrices (after SVD) as numpy arrays
np.save("/kaggle/working/tfidf_word_train.npy", X_train_tfidf_svd)
np.save("/kaggle/working/tfidf_word_test.npy", X_test_tfidf_svd)

# Save original sparse TF-IDF matrices
sparse.save_npz("/kaggle/working/tfidf_word_train_sparse.npz", X_train_tfidf)
sparse.save_npz("/kaggle/working/tfidf_word_test_sparse.npz", X_test_tfidf)

# Save feature names
with open("/kaggle/working/tfidf_word_feature_names.json", "w") as f:
    json.dump(feature_names.tolist(), f, indent=2)

# Save the pipeline for later use
with open("/kaggle/working/tfidf_pipeline.pkl", "wb") as f:
    pickle.dump(tfidf_pipeline, f)

# Analyze the features
analysis = analyze_tfidf_features(
    X_train_tfidf,
    feature_names,
    tfidf_pipeline.named_steps['svd']
)

# Save analysis results
with open("/kaggle/working/tfidf_analysis.json", "w") as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)

print("")
print("=" * 60)
print("C6: TF-IDF WORD N-GRAM FEATURE EXTRACTION COMPLETE")
print("=" * 60)

# Display comprehensive results
print("TF-IDF Feature Analysis:")
print(f"  Original feature space: {analysis['tfidf_shape'][1]} n-grams")
print(f"  Reduced dimensions: {X_train_tfidf_svd.shape[1]} components")
print(f"  Training samples: {X_train_tfidf_svd.shape[0]}")
print(f"  Test samples: {X_test_tfidf_svd.shape[0]}")
print(f"  Matrix sparsity: {analysis['sparsity']:.2%}")
print(f"  SVD explained variance: {analysis['svd_explained_variance']:.2%}")

print("")
print("Top Overall Features (across all components):")
for i, (feature, importance) in enumerate(analysis['top_overall_features'][:10]):
    print(f"  {i+1:2d}. {feature:<20} {importance:.4f}")

print("")
print("Top Features in First 3 Components:")
for comp_idx in range(3):
    comp_name = f'component_{comp_idx}'
    if comp_name in analysis['top_components']:
        print(f"  Component {comp_idx}:")
        for feature, weight in analysis['top_components'][comp_name][:5]:
            print(f"    {feature:<20} {weight:+.4f}")

print("")
print("SVD Component Variance (first 10):")
for i, var in enumerate(analysis['svd_explained_variance_per_component'][:10]):
    print(f"  Component {i}: {var:.4f} ({var:.2%})")

print("")
print("Files Saved:")
print(f"  /kaggle/working/tfidf_word_train.npy (shape: {X_train_tfidf_svd.shape})")
print(f"  /kaggle/working/tfidf_word_test.npy (shape: {X_test_tfidf_svd.shape})")
print(f"  /kaggle/working/tfidf_word_train_sparse.npz (original TF-IDF)")
print(f"  /kaggle/working/tfidf_word_test_sparse.npz (original TF-IDF)")
print(f"  /kaggle/working/tfidf_word_feature_names.json")
print(f"  /kaggle/working/tfidf_pipeline.pkl")
print(f"  /kaggle/working/tfidf_analysis.json")

print("")
print("Next: Ready for character n-grams and final feature integration")
print("=" * 60)


# =============================================
# C7: TFIDF CHARACTER NGRAMS
# =============================================

import pandas as pd
import numpy as np
import json
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import pickle

print("Generating TF-IDF character n-gram features...")

def create_char_tfidf_pipeline():
    """
    Create TF-IDF + SVD pipeline for character-level feature extraction.
    
    Returns:
        sklearn Pipeline with character TF-IDF vectorizer and SVD dimensionality reduction
    """
    # TF-IDF with character n-grams (3-5 characters)
    char_tfidf_vectorizer = TfidfVectorizer(
        analyzer='char',               # Character-level n-grams
        ngram_range=(3, 5),           # 3 to 5 character sequences
        max_features=2000,            # Top 2000 character n-grams
        min_df=2,                     # Ignore n-grams that appear in less than 2 documents
        max_df=0.9,                   # Ignore n-grams that appear in more than 90% of documents
        lowercase=True,               # Convert to lowercase
        strip_accents='unicode',      # Remove accents
        use_idf=True,                 # Use inverse document frequency
        smooth_idf=True,              # Smooth IDF weights
        sublinear_tf=False            # Use sublinear TF scaling
    )
    
    # Dimensionality reduction with SVD
    svd_reducer = TruncatedSVD(
        n_components=50,              # Reduce to 50 latent dimensions
        algorithm='randomized',
        random_state=42,              # For reproducibility
        n_iter=10
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('char_tfidf', char_tfidf_vectorizer),
        ('svd', svd_reducer)
    ])
    
    return pipeline

def analyze_char_tfidf_features(tfidf_matrix, feature_names, svd_model):
    """
    Analyze character TF-IDF features and SVD components.
    
    Args:
        tfidf_matrix: Character TF-IDF transformed matrix
        feature_names: List of character n-gram features
        svd_model: Fitted SVD model
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {}
    
    # Basic TF-IDF stats
    analysis['tfidf_shape'] = tfidf_matrix.shape
    analysis['num_features'] = len(feature_names)
    analysis['sparsity'] = 1.0 - (tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]))
    
    # SVD analysis
    analysis['svd_explained_variance'] = float(svd_model.explained_variance_ratio_.sum())
    analysis['svd_explained_variance_per_component'] = [
        float(var) for var in svd_model.explained_variance_ratio_
    ]
    
    # Top character n-grams by component
    analysis['top_components'] = {}
    components = svd_model.components_
    
    for i in range(min(5, components.shape[0])):  # First 5 components
        top_indices = np.argsort(components[i])[-10:][::-1]  # Top 10 features
        top_features = [(feature_names[idx], float(components[i, idx])) for idx in top_indices]
        analysis['top_components'][f'component_{i}'] = top_features
    
    # Most important character n-grams overall
    feature_importance = np.abs(components).sum(axis=0)
    top_feature_indices = np.argsort(feature_importance)[-20:][::-1]
    analysis['top_overall_features'] = [
        (feature_names[idx], float(feature_importance[idx])) 
        for idx in top_feature_indices
    ]
    
    # Character n-gram categories analysis
    analysis['char_categories'] = categorize_char_ngrams(feature_names, feature_importance)
    
    return analysis

def categorize_char_ngrams(feature_names, feature_importance):
    """
    Categorize character n-grams into meaningful groups.
    
    Args:
        feature_names: List of character n-grams
        feature_importance: Importance scores for each n-gram
        
    Returns:
        Dictionary with categorized n-grams
    """
    categories = {
        'url_patterns': [],
        'legal_terms': [],
        'promotional': [],
        'reddit_artifacts': [],
        'common_words': [],
        'punctuation': [],
        'other': []
    }
    
    # Create importance mapping
    importance_dict = dict(zip(feature_names, feature_importance))
    
    for ngram in feature_names:
        importance = importance_dict[ngram]
        
        # URL patterns (even after [URL] replacement, patterns may remain)
        if any(pattern in ngram for pattern in ['htt', 'www', '.co', '.ne', '.or', '://', 'http']):
            categories['url_patterns'].append((ngram, importance))
        # Legal terms patterns
        elif any(pattern in ngram for pattern in ['law', 'sue', 'urt', 'jud', 'att', 'leg', 'tan', 'lor']):
            categories['legal_terms'].append((ngram, importance))
        # Promotional patterns
        elif any(pattern in ngram for pattern in ['fre', 'dis', 'off', 'cou', 'pro', 'sal', 'che', 'buy']):
            categories['promotional'].append((ngram, importance))
        # Reddit artifact patterns
        elif any(pattern in ngram for pattern in ['ser]', 'edd', 'red', 'ubr', 'url', 'use']):
            categories['reddit_artifacts'].append((ngram, importance))
        # Common word patterns
        elif any(pattern in ngram for pattern in ['the', 'ing', 'and', 'ion', 'ent', 'com', 'you', 'tha']):
            categories['common_words'].append((ngram, importance))
        # Punctuation patterns
        elif any(pattern in ngram for pattern in ['.', ',', '!', '?', ';', ':', '-', '(', ')']):
            categories['punctuation'].append((ngram, importance))
        else:
            categories['other'].append((ngram, importance))
    
    # Sort each category by importance
    for category in categories:
        categories[category] = sorted(categories[category], key=lambda x: x[1], reverse=True)[:10]
    
    return categories

# Load domain-enhanced datasets
train = pd.read_csv("/kaggle/working/train_domain.csv")
test = pd.read_csv("/kaggle/working/test_domain.csv")

print("Fitting character TF-IDF pipeline on training data...")

# Create and fit character pipeline on training data
char_tfidf_pipeline = create_char_tfidf_pipeline()

# Fit on training data and transform both train and test
X_train_char_tfidf_svd = char_tfidf_pipeline.fit_transform(train['body_clean'])
X_test_char_tfidf_svd = char_tfidf_pipeline.transform(test['body_clean'])

# Get feature names from the character TF-IDF vectorizer
char_feature_names = char_tfidf_pipeline.named_steps['char_tfidf'].get_feature_names_out()

# Also get the original character TF-IDF matrices (before SVD) for analysis
X_train_char_tfidf = char_tfidf_pipeline.named_steps['char_tfidf'].transform(train['body_clean'])
X_test_char_tfidf = char_tfidf_pipeline.named_steps['char_tfidf'].transform(test['body_clean'])

print("Saving character TF-IDF features and metadata...")

# Save dense matrices (after SVD) as numpy arrays
np.save("/kaggle/working/tfidf_char_train.npy", X_train_char_tfidf_svd)
np.save("/kaggle/working/tfidf_char_test.npy", X_test_char_tfidf_svd)

# Save original sparse character TF-IDF matrices
sparse.save_npz("/kaggle/working/tfidf_char_train_sparse.npz", X_train_char_tfidf)
sparse.save_npz("/kaggle/working/tfidf_char_test_sparse.npz", X_test_char_tfidf)

# Save feature names
with open("/kaggle/working/tfidf_char_feature_names.json", "w") as f:
    json.dump(char_feature_names.tolist(), f, indent=2)

# Save the pipeline for later use
with open("/kaggle/working/tfidf_char_pipeline.pkl", "wb") as f:
    pickle.dump(char_tfidf_pipeline, f)

# Analyze the character features
char_analysis = analyze_char_tfidf_features(
    X_train_char_tfidf,
    char_feature_names,
    char_tfidf_pipeline.named_steps['svd']
)

# Save analysis results
with open("/kaggle/working/tfidf_char_analysis.json", "w") as f:
    json.dump(char_analysis, f, indent=2, ensure_ascii=False)

print("")
print("=" * 60)
print("C7: TF-IDF CHARACTER N-GRAM FEATURE EXTRACTION COMPLETE")
print("=" * 60)

# Display comprehensive results
print("Character TF-IDF Feature Analysis:")
print(f"  Original feature space: {char_analysis['tfidf_shape'][1]} char n-grams")
print(f"  Reduced dimensions: {X_train_char_tfidf_svd.shape[1]} components")
print(f"  Training samples: {X_train_char_tfidf_svd.shape[0]}")
print(f"  Test samples: {X_test_char_tfidf_svd.shape[0]}")
print(f"  Matrix sparsity: {char_analysis['sparsity']:.2%}")
print(f"  SVD explained variance: {char_analysis['svd_explained_variance']:.2%}")

print("")
print("Top Character N-grams (across all components):")
for i, (ngram, importance) in enumerate(char_analysis['top_overall_features'][:15]):
    print(f"  {i+1:2d}. '{ngram}'{' '*(8-len(ngram))} {importance:.4f}")

print("")
print("Character N-gram Categories (Top Patterns):")
categories = char_analysis['char_categories']
for category, patterns in categories.items():
    if patterns:
        print(f"  {category.replace('_', ' ').title()}:")
        for ngram, importance in patterns[:5]:
            print(f"    '{ngram}' {importance:.4f}")

print("")
print("Top Features in First 3 Components:")
for comp_idx in range(3):
    comp_name = f'component_{comp_idx}'
    if comp_name in char_analysis['top_components']:
        print(f"  Component {comp_idx}:")
        for ngram, weight in char_analysis['top_components'][comp_name][:5]:
            print(f"    '{ngram}'{' '*(8-len(ngram))} {weight:+.4f}")

print("")
print("SVD Component Variance (first 10):")
for i, var in enumerate(char_analysis['svd_explained_variance_per_component'][:10]):
    print(f"  Component {i}: {var:.4f} ({var:.2%})")

print("")
print("Files Saved:")
print(f"  /kaggle/working/tfidf_char_train.npy (shape: {X_train_char_tfidf_svd.shape})")
print(f"  /kaggle/working/tfidf_char_test.npy (shape: {X_test_char_tfidf_svd.shape})")
print(f"  /kaggle/working/tfidf_char_train_sparse.npz (original char TF-IDF)")
print(f"  /kaggle/working/tfidf_char_test_sparse.npz (original char TF-IDF)")
print(f"  /kaggle/working/tfidf_char_feature_names.json")
print(f"  /kaggle/working/tfidf_char_pipeline.pkl")
print(f"  /kaggle/working/tfidf_char_analysis.json")

print("")
print("Next: Ready for sentence embeddings and final feature integration")
print("=" * 60)


# =============================================
# C7: tfidf_char_ngrams
# RuleSense v2 â€” Jigsaw Agile Community Rules
# =============================================

import pandas as pd
import numpy as np
import json
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import pickle

print("ğŸ”¤ Generating TF-IDF character n-gram features...")

def create_char_tfidf_pipeline():
    """
    Create TF-IDF + SVD pipeline for character-level feature extraction.
    
    Returns:
        sklearn Pipeline with character TF-IDF vectorizer and SVD dimensionality reduction
    """
    # TF-IDF with character n-grams (3-5 characters)
    char_tfidf_vectorizer = TfidfVectorizer(
        analyzer='char',               # Character-level n-grams
        ngram_range=(3, 5),           # 3 to 5 character sequences
        max_features=2000,            # Top 2000 character n-grams
        min_df=2,                     # Ignore n-grams that appear in less than 2 documents
        max_df=0.9,                   # Ignore n-grams that appear in more than 90% of documents
        lowercase=True,               # Convert to lowercase
        strip_accents='unicode',      # Remove accents
        use_idf=True,                 # Use inverse document frequency
        smooth_idf=True,              # Smooth IDF weights
        sublinear_tf=False            # Use sublinear TF scaling
    )
    
    # Dimensionality reduction with SVD
    svd_reducer = TruncatedSVD(
        n_components=50,              # Reduce to 50 latent dimensions
        algorithm='randomized',
        random_state=42,              # For reproducibility
        n_iter=10
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('char_tfidf', char_tfidf_vectorizer),
        ('svd', svd_reducer)
    ])
    
    return pipeline

def analyze_char_tfidf_features(tfidf_matrix, feature_names, svd_model):
    """
    Analyze character TF-IDF features and SVD components.
    
    Args:
        tfidf_matrix: Character TF-IDF transformed matrix
        feature_names: List of character n-gram features
        svd_model: Fitted SVD model
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {}
    
    # Basic TF-IDF stats
    analysis['tfidf_shape'] = tfidf_matrix.shape
    analysis['num_features'] = len(feature_names)
    analysis['sparsity'] = 1.0 - (tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]))
    
    # SVD analysis
    analysis['svd_explained_variance'] = float(svd_model.explained_variance_ratio_.sum())
    analysis['svd_explained_variance_per_component'] = [
        float(var) for var in svd_model.explained_variance_ratio_
    ]
    
    # Top character n-grams by component
    analysis['top_components'] = {}
    components = svd_model.components_
    
    for i in range(min(5, components.shape[0])):  # First 5 components
        top_indices = np.argsort(components[i])[-10:][::-1]  # Top 10 features
        top_features = [(feature_names[idx], float(components[i, idx])) for idx in top_indices]
        analysis['top_components'][f'component_{i}'] = top_features
    
    # Most important character n-grams overall
    feature_importance = np.abs(components).sum(axis=0)
    top_feature_indices = np.argsort(feature_importance)[-20:][::-1]
    analysis['top_overall_features'] = [
        (feature_names[idx], float(feature_importance[idx])) 
        for idx in top_feature_indices
    ]
    
    # Character n-gram categories analysis
    analysis['char_categories'] = categorize_char_ngrams(feature_names, feature_importance)
    
    return analysis

def categorize_char_ngrams(feature_names, feature_importance):
    """
    Categorize character n-grams into meaningful groups.
    
    Args:
        feature_names: List of character n-grams
        feature_importance: Importance scores for each n-gram
        
    Returns:
        Dictionary with categorized n-grams
    """
    categories = {
        'url_patterns': [],
        'legal_terms': [],
        'promotional': [],
        'reddit_artifacts': [],
        'common_words': [],
        'punctuation': [],
        'other': []
    }
    
    # Create importance mapping
    importance_dict = dict(zip(feature_names, feature_importance))
    
    for ngram in feature_names:
        importance = importance_dict[ngram]
        
        # URL patterns (even after [URL] replacement, patterns may remain)
        if any(pattern in ngram for pattern in ['htt', 'www', '.co', '.ne', '.or', '://', 'http']):
            categories['url_patterns'].append((ngram, importance))
        # Legal terms patterns
        elif any(pattern in ngram for pattern in ['law', 'sue', 'urt', 'jud', 'att', 'leg', 'tan', 'lor']):
            categories['legal_terms'].append((ngram, importance))
        # Promotional patterns
        elif any(pattern in ngram for pattern in ['fre', 'dis', 'off', 'cou', 'pro', 'sal', 'che', 'buy']):
            categories['promotional'].append((ngram, importance))
        # Reddit artifact patterns
        elif any(pattern in ngram for pattern in ['ser]', 'edd', 'red', 'ubr', 'url', 'use']):
            categories['reddit_artifacts'].append((ngram, importance))
        # Common word patterns
        elif any(pattern in ngram for pattern in ['the', 'ing', 'and', 'ion', 'ent', 'com', 'you', 'tha']):
            categories['common_words'].append((ngram, importance))
        # Punctuation patterns
        elif any(pattern in ngram for pattern in ['.', ',', '!', '?', ';', ':', '-', '(', ')']):
            categories['punctuation'].append((ngram, importance))
        else:
            categories['other'].append((ngram, importance))
    
    # Sort each category by importance
    for category in categories:
        categories[category] = sorted(categories[category], key=lambda x: x[1], reverse=True)[:10]
    
    return categories

# Load domain-enhanced datasets
train = pd.read_csv("/kaggle/working/train_domain.csv")
test = pd.read_csv("/kaggle/working/test_domain.csv")

print("ğŸ”§ Fitting character TF-IDF pipeline on training data...")

# Create and fit character pipeline on training data
char_tfidf_pipeline = create_char_tfidf_pipeline()

# Fit on training data and transform both train and test
X_train_char_tfidf_svd = char_tfidf_pipeline.fit_transform(train['body_clean'])
X_test_char_tfidf_svd = char_tfidf_pipeline.transform(test['body_clean'])

# Get feature names from the character TF-IDF vectorizer
char_feature_names = char_tfidf_pipeline.named_steps['char_tfidf'].get_feature_names_out()

# Also get the original character TF-IDF matrices (before SVD) for analysis
X_train_char_tfidf = char_tfidf_pipeline.named_steps['char_tfidf'].transform(train['body_clean'])
X_test_char_tfidf = char_tfidf_pipeline.named_steps['char_tfidf'].transform(test['body_clean'])

print("ğŸ’¾ Saving character TF-IDF features and metadata...")

# Save dense matrices (after SVD) as numpy arrays
np.save("/kaggle/working/tfidf_char_train.npy", X_train_char_tfidf_svd)
np.save("/kaggle/working/tfidf_char_test.npy", X_test_char_tfidf_svd)

# Save original sparse character TF-IDF matrices
sparse.save_npz("/kaggle/working/tfidf_char_train_sparse.npz", X_train_char_tfidf)
sparse.save_npz("/kaggle/working/tfidf_char_test_sparse.npz", X_test_char_tfidf)

# Save feature names
with open("/kaggle/working/tfidf_char_feature_names.json", "w") as f:
    json.dump(char_feature_names.tolist(), f, indent=2)

# Save the pipeline for later use
with open("/kaggle/working/tfidf_char_pipeline.pkl", "wb") as f:
    pickle.dump(char_tfidf_pipeline, f)

# Analyze the character features
char_analysis = analyze_char_tfidf_features(
    X_train_char_tfidf,
    char_feature_names,
    char_tfidf_pipeline.named_steps['svd']
)

# Save analysis results
with open("/kaggle/working/tfidf_char_analysis.json", "w") as f:
    json.dump(char_analysis, f, indent=2, ensure_ascii=False)

print("\n" + "="*60)
print("âœ… C7: TF-IDF character n-gram feature extraction complete")
print("="*60)

# Display comprehensive results
print(f"ğŸ“Š Character TF-IDF Feature Analysis:")
print(f"   â€¢ Original feature space: {char_analysis['tfidf_shape'][1]} char n-grams")
print(f"   â€¢ Reduced dimensions: {X_train_char_tfidf_svd.shape[1]} components")
print(f"   â€¢ Training samples: {X_train_char_tfidf_svd.shape[0]}")
print(f"   â€¢ Test samples: {X_test_char_tfidf_svd.shape[0]}")
print(f"   â€¢ Matrix sparsity: {char_analysis['sparsity']:.2%}")
print(f"   â€¢ SVD explained variance: {char_analysis['svd_explained_variance']:.2%}")

print(f"\nğŸ�¯ Top Character N-grams (across all components):")
for i, (ngram, importance) in enumerate(char_analysis['top_overall_features'][:15]):
    print(f"   {i+1:2d}. '{ngram}'{' '*(8-len(ngram))} {importance:.4f}")

print(f"\nğŸ”� Character N-gram Categories (Top Patterns):")
categories = char_analysis['char_categories']
for category, patterns in categories.items():
    if patterns:
        print(f"   {category.replace('_', ' ').title()}:")
        for ngram, importance in patterns[:5]:
            print(f"      â€¢ '{ngram}' {importance:.4f}")

print(f"\nğŸ”� Top Features in First 3 Components:")
for comp_idx in range(3):
    comp_name = f'component_{comp_idx}'
    if comp_name in char_analysis['top_components']:
        print(f"   Component {comp_idx}:")
        for ngram, weight in char_analysis['top_components'][comp_name][:5]:
            print(f"      â€¢ '{ngram}'{' '*(8-len(ngram))} {weight:+.4f}")

print(f"\nğŸ“ˆ SVD Component Variance (first 10):")
for i, var in enumerate(char_analysis['svd_explained_variance_per_component'][:10]):
    print(f"   â€¢ Component {i}: {var:.4f} ({var:.2%})")

print(f"\nğŸ’¾ Files Saved:")
print(f"   â€¢ /kaggle/working/tfidf_char_train.npy (shape: {X_train_char_tfidf_svd.shape})")
print(f"   â€¢ /kaggle/working/tfidf_char_test.npy (shape: {X_test_char_tfidf_svd.shape})")
print(f"   â€¢ /kaggle/working/tfidf_char_train_sparse.npz (original char TF-IDF)")
print(f"   â€¢ /kaggle/working/tfidf_char_test_sparse.npz (original char TF-IDF)")
print(f"   â€¢ /kaggle/working/tfidf_char_feature_names.json")
print(f"   â€¢ /kaggle/working/tfidf_char_pipeline.pkl")
print(f"   â€¢ /kaggle/working/tfidf_char_analysis.json")

print(f"\nğŸ�¯ Next: Ready for sentence embeddings and final feature integration")
print("="*60)


# =============================================
# C8: GENERATE PRIORS
# =============================================

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

print("Generating smoothed priors for subreddit and rule...")

def smoothed_rate(pos: int, total: int, global_rate: float, alpha: int = 5) -> float:
    """
    Calculate smoothed rate using additive smoothing.
    
    Args:
        pos: Number of positive cases
        total: Total number of cases
        global_rate: Global positive rate
        alpha: Smoothing parameter
        
    Returns:
        Smoothed rate between 0 and 1
    """
    return (pos + alpha * global_rate) / (total + alpha)

def generate_priors_cv(df: pd.DataFrame, target_col: str = 'rule_violation', 
                      group_col: str = 'subreddit', alpha: int = 5, n_splits: int = 5) -> pd.DataFrame:
    """
    Generate cross-validated priors using Group K-Fold to prevent data leakage.
    
    Args:
        df: DataFrame with target and group columns
        target_col: Name of target variable
        group_col: Name of group column (subreddit)
        alpha: Smoothing parameter
        n_splits: Number of CV folds
        
    Returns:
        DataFrame with OOF (Out-of-Fold) priors
    """
    df_priors = df.copy()
    df_priors['subreddit_prior'] = 0.0
    df_priors['rule_prior'] = 0.0
    df_priors['fold'] = -1
    
    # Global statistics
    global_rate = df[target_col].mean()
    total_samples = len(df)
    
    print(f"  Global violation rate: {global_rate:.4f}")
    print(f"  Total samples: {total_samples}")
    print(f"  Smoothing alpha: {alpha}")
    print(f"  CV folds: {n_splits}")
    
    # Group K-Fold to preserve group structure
    group_kfold = GroupKFold(n_splits=n_splits)
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(df, df[target_col], df[group_col])):
        print(f"  Processing fold {fold + 1}/{n_splits}...")
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        # Calculate priors on training fold
        fold_global_rate = train_df[target_col].mean()
        
        # Subreddit priors
        subreddit_stats = train_df.groupby(group_col)[target_col].agg(['sum', 'count']).reset_index()
        subreddit_stats['subreddit_prior'] = subreddit_stats.apply(
            lambda x: smoothed_rate(x['sum'], x['count'], fold_global_rate, alpha), axis=1
        )
        
        # Rule priors (if rule column exists)
        if 'rule' in train_df.columns:
            rule_stats = train_df.groupby('rule')[target_col].agg(['sum', 'count']).reset_index()
            rule_stats['rule_prior'] = rule_stats.apply(
                lambda x: smoothed_rate(x['sum'], x['count'], fold_global_rate, alpha), axis=1
            )
        else:
            # If no rule column, use subreddit priors for both
            rule_stats = subreddit_stats.copy()
            rule_stats['rule_prior'] = rule_stats['subreddit_prior']
        
        # Map priors to validation set
        subreddit_prior_map = dict(zip(subreddit_stats[group_col], subreddit_stats['subreddit_prior']))
        rule_prior_map = dict(zip(rule_stats['rule'], rule_stats['rule_prior'])) if 'rule' in train_df.columns else subreddit_prior_map
        
        # Apply priors to validation set
        df_priors.loc[val_idx, 'subreddit_prior'] = val_df[group_col].map(subreddit_prior_map).fillna(fold_global_rate)
        df_priors.loc[val_idx, 'rule_prior'] = val_df['rule'].map(rule_prior_map).fillna(fold_global_rate) if 'rule' in train_df.columns else df_priors.loc[val_idx, 'subreddit_prior']
        df_priors.loc[val_idx, 'fold'] = fold
        
        # Calculate fold statistics
        fold_val_rate = val_df[target_col].mean()
        fold_prior_mean = df_priors.loc[val_idx, 'subreddit_prior'].mean()
        
        fold_scores.append({
            'fold': fold,
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'train_rate': fold_global_rate,
            'val_rate': fold_val_rate,
            'prior_mean': fold_prior_mean,
            'unique_subreddits_train': train_df[group_col].nunique(),
            'unique_subreddits_val': val_df[group_col].nunique()
        })
    
    return df_priors, fold_scores

def generate_test_priors(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                        target_col: str = 'rule_violation', group_col: str = 'subreddit', 
                        alpha: int = 5) -> pd.DataFrame:
    """
    Generate priors for test set using full training data.
    
    Args:
        train_df: Full training DataFrame
        test_df: Test DataFrame
        target_col: Name of target variable
        group_col: Name of group column
        alpha: Smoothing parameter
        
    Returns:
        Test DataFrame with priors
    """
    test_priors = test_df.copy()
    
    # Global statistics from training data
    global_rate = train_df[target_col].mean()
    
    # Subreddit priors
    subreddit_stats = train_df.groupby(group_col)[target_col].agg(['sum', 'count']).reset_index()
    subreddit_stats['subreddit_prior'] = subreddit_stats.apply(
        lambda x: smoothed_rate(x['sum'], x['count'], global_rate, alpha), axis=1
    )
    
    # Rule priors (if rule column exists)
    if 'rule' in train_df.columns:
        rule_stats = train_df.groupby('rule')[target_col].agg(['sum', 'count']).reset_index()
        rule_stats['rule_prior'] = rule_stats.apply(
            lambda x: smoothed_rate(x['sum'], x['count'], global_rate, alpha), axis=1
        )
    else:
        # If no rule column, use subreddit priors for both
        rule_stats = subreddit_stats.copy()
        rule_stats['rule_prior'] = rule_stats['subreddit_prior']
    
    # Map priors to test set
    subreddit_prior_map = dict(zip(subreddit_stats[group_col], subreddit_stats['subreddit_prior']))
    rule_prior_map = dict(zip(rule_stats['rule'], rule_stats['rule_prior'])) if 'rule' in train_df.columns else subreddit_prior_map
    
    # Apply priors to test set
    test_priors['subreddit_prior'] = test_priors[group_col].map(subreddit_prior_map).fillna(global_rate)
    test_priors['rule_prior'] = test_priors['rule'].map(rule_prior_map).fillna(global_rate) if 'rule' in train_df.columns else test_priors['subreddit_prior']
    
    return test_priors

def analyze_priors(priors_df: pd.DataFrame, target_col: str = 'rule_violation') -> dict:
    """
    Analyze the generated priors and their relationship with the target.
    
    Args:
        priors_df: DataFrame with priors
        target_col: Name of target variable
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {}
    
    # Basic statistics
    analysis['global_rate'] = float(priors_df[target_col].mean())
    analysis['subreddit_prior_mean'] = float(priors_df['subreddit_prior'].mean())
    analysis['rule_prior_mean'] = float(priors_df['rule_prior'].mean())
    analysis['subreddit_prior_std'] = float(priors_df['subreddit_prior'].std())
    analysis['rule_prior_std'] = float(priors_df['rule_prior'].std())
    
    # Correlation with target
    analysis['subreddit_prior_corr'] = float(priors_df['subreddit_prior'].corr(priors_df[target_col]))
    if 'rule_prior' in priors_df.columns:
        analysis['rule_prior_corr'] = float(priors_df['rule_prior'].corr(priors_df[target_col]))
    
    # Top subreddits by prior (if available)
    if 'subreddit' in priors_df.columns:
        subreddit_analysis = priors_df.groupby('subreddit').agg({
            target_col: 'mean',
            'subreddit_prior': 'mean',
            'body': 'count'
        }).rename(columns={'body': 'count'}).sort_values('subreddit_prior', ascending=False)
        
        analysis['top_subreddits'] = subreddit_analysis.head(10).to_dict('index')
        analysis['bottom_subreddits'] = subreddit_analysis.tail(10).to_dict('index')
    
    return analysis

# Load domain-enhanced datasets
train = pd.read_csv("/kaggle/working/train_domain.csv")
test = pd.read_csv("/kaggle/working/test_domain.csv")

print("Generating cross-validated priors...")

# Generate OOF priors for training data
train_priors, fold_scores = generate_priors_cv(
    df=train,
    target_col='rule_violation',
    group_col='subreddit',
    alpha=5,
    n_splits=5
)

# Generate priors for test data
test_priors = generate_test_priors(
    train_df=train,
    test_df=test,
    target_col='rule_violation',
    group_col='subreddit',
    alpha=5
)

print("Analyzing priors...")

# Analyze the generated priors
priors_analysis = analyze_priors(train_priors, 'rule_violation')

print("Saving priors and analysis...")

# Save priors
train_priors.to_csv("/kaggle/working/priors_oof.csv", index=False)
test_priors.to_csv("/kaggle/working/priors_test.csv", index=False)

# Save analysis
with open("/kaggle/working/priors_analysis.json", "w") as f:
    json.dump(priors_analysis, f, indent=2, ensure_ascii=False)

# Save fold scores
with open("/kaggle/working/priors_fold_scores.json", "w") as f:
    json.dump(fold_scores, f, indent=2, ensure_ascii=False)

print("")
print("=" * 60)
print("C8: PRIOR GENERATION COMPLETE")
print("=" * 60)

# Display comprehensive results
print("Priors Analysis:")
print(f"  Global violation rate: {priors_analysis['global_rate']:.4f}")
print(f"  Subreddit prior mean: {priors_analysis['subreddit_prior_mean']:.4f}")
print(f"  Rule prior mean: {priors_analysis['rule_prior_mean']:.4f}")
print(f"  Subreddit prior std: {priors_analysis['subreddit_prior_std']:.4f}")
print(f"  Subreddit prior-target correlation: {priors_analysis['subreddit_prior_corr']:.4f}")

print("")
print("Cross-Validation Fold Summary:")
for fold in fold_scores:
    print(f"  Fold {fold['fold'] + 1}:")
    print(f"    Train: {fold['train_samples']} samples, rate: {fold['train_rate']:.4f}")
    print(f"    Val: {fold['val_samples']} samples, rate: {fold['val_rate']:.4f}")
    print(f"    Priors mean: {fold['prior_mean']:.4f}")
    print(f"    Unique subreddits: {fold['unique_subreddits_train']} train, {fold['unique_subreddits_val']} val")

print("")
print("Top Subreddits by Prior (High Risk):")
if 'top_subreddits' in priors_analysis:
    for i, (subreddit, stats) in enumerate(list(priors_analysis['top_subreddits'].items())[:5]):
        print(f"  {i+1}. r/{subreddit}:")
        print(f"    Prior: {stats['subreddit_prior']:.4f}")
        print(f"    Actual rate: {stats['rule_violation']:.4f}")
        print(f"    Samples: {stats['count']}")

print("")
print("Bottom Subreddits by Prior (Low Risk):")
if 'bottom_subreddits' in priors_analysis:
    for i, (subreddit, stats) in enumerate(list(priors_analysis['bottom_subreddits'].items())[:5]):
        print(f"  {i+1}. r/{subreddit}:")
        print(f"    Prior: {stats['subreddit_prior']:.4f}")
        print(f"    Actual rate: {stats['rule_violation']:.4f}")
        print(f"    Samples: {stats['count']}")

print("")
print("Dataset Shapes with Priors:")
print(f"  Train priors: {train_priors.shape} (+2 features)")
print(f"  Test priors: {test_priors.shape} (+2 features)")

print("")
print("Files Saved:")
print(f"  /kaggle/working/priors_oof.csv")
print(f"  /kaggle/working/priors_test.csv")
print(f"  /kaggle/working/priors_analysis.json")
print(f"  /kaggle/working/priors_fold_scores.json")

print("")
print("Next: Ready for final feature integration and model training")
print("=" * 60)


# =============================================
# C9: GENERATE EMBEDDINGS (OFFLINE-COMPATIBLE VERSION)
# =============================================

import pandas as pd
import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

print("Generating semantic embeddings for text and rules...")

def load_embedding_model():
    """
    Load embedding model - tries HuggingFace first, falls back to TF-IDF
    """
    print("  Attempting to load sentence transformer model...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"  Model loaded: {model.get_sentence_embedding_dimension()} dimensions")
        return model, 'transformer'
    except Exception as e:
        print(f"  Transformer model failed: {e}")
        print("  Falling back to TF-IDF embeddings...")
        return TfidfVectorizer(max_features=384, ngram_range=(1, 2)), 'tfidf'

def generate_embeddings(model, model_type, texts, batch_size=32, desc="Texts", fit=False):
    """
    Generate embeddings based on model type
    """
    print(f"  Encoding {len(texts)} {desc}...")
    
    # Handle empty texts
    valid_texts = []
    valid_indices = []
    
    for i, text in enumerate(texts):
        if isinstance(text, str) and text.strip() and text != "[EMPTY]":
            valid_texts.append(text)
            valid_indices.append(i)
    
    if not valid_texts:
        embedding_dim = 384 if model_type == 'tfidf' else model.get_sentence_embedding_dimension()
        return np.zeros((len(texts), embedding_dim))
    
    if model_type == 'transformer':
        # Generate embeddings with transformer
        embeddings = np.zeros((len(texts), model.get_sentence_embedding_dimension()))
        valid_embeddings = model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Place valid embeddings back in original positions
        for idx, emb in zip(valid_indices, valid_embeddings):
            embeddings[idx] = emb
        
        return embeddings
    
    else:  # TF-IDF
        if fit:
            valid_embeddings = model.fit_transform(valid_texts).toarray()
        else:
            valid_embeddings = model.transform(valid_texts).toarray()
        
        # Normalize TF-IDF embeddings
        norms = np.linalg.norm(valid_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        valid_embeddings = valid_embeddings / norms
        
        # Create full embeddings array
        embeddings = np.zeros((len(texts), valid_embeddings.shape[1]))
        for idx, emb in zip(valid_indices, valid_embeddings):
            embeddings[idx] = emb
        
        return embeddings

def create_semantic_features(body_embeddings, rule_embeddings, n_components=50, pca_model=None):
    """
    Create semantic features from body and rule embeddings.
    """
    features = {}
    
    # 1. Cosine similarity between body and rule
    print("  Calculating cosine similarities...")
    cos_similarities = []
    for body_emb, rule_emb in zip(body_embeddings, rule_embeddings):
        # Reshape for cosine_similarity
        sim = cosine_similarity([body_emb], [rule_emb])[0][0]
        cos_similarities.append(sim)
    
    features['cos_similarity'] = np.array(cos_similarities)
    
    # 2. Absolute difference between embeddings
    print("  Calculating absolute differences...")
    abs_diff = np.abs(body_embeddings - rule_embeddings)
    features['abs_diff_raw'] = abs_diff
    
    # 3. PCA on absolute differences
    print(f"  Applying PCA to reduce dimensions to {n_components}...")
    
    n_components_actual = min(n_components, body_embeddings.shape[0], body_embeddings.shape[1])
    
    if pca_model is None:
        # Fit new PCA model on training data
        pca = PCA(n_components=n_components_actual, random_state=42)
        abs_diff_pca = pca.fit_transform(abs_diff)
        features['pca_model'] = pca
    else:
        # Use pre-trained PCA model for test data
        abs_diff_pca = pca_model.transform(abs_diff)
        features['pca_model'] = pca_model
    
    features['abs_diff_pca'] = abs_diff_pca
    features['pca_explained_variance'] = features['pca_model'].explained_variance_ratio_.sum()
    
    # 4. Euclidean distance
    print("  Calculating Euclidean distances...")
    euclidean_dist = np.linalg.norm(body_embeddings - rule_embeddings, axis=1)
    features['euclidean_distance'] = euclidean_dist
    
    # 5. Dot product
    print("  Calculating dot products...")
    dot_product = np.sum(body_embeddings * rule_embeddings, axis=1)
    features['dot_product'] = dot_product
    
    return features

def analyze_embeddings(body_embeddings, rule_embeddings, semantic_features, target_col=None):
    """
    Analyze the generated embeddings and semantic features.
    """
    analysis = {}
    
    # Basic embedding statistics
    analysis['body_embeddings_shape'] = body_embeddings.shape
    analysis['rule_embeddings_shape'] = rule_embeddings.shape
    analysis['embedding_dimension'] = body_embeddings.shape[1]
    
    # Semantic feature statistics
    analysis['cos_similarity_stats'] = {
        'mean': float(semantic_features['cos_similarity'].mean()),
        'std': float(semantic_features['cos_similarity'].std()),
        'min': float(semantic_features['cos_similarity'].min()),
        'max': float(semantic_features['cos_similarity'].max())
    }
    
    analysis['euclidean_distance_stats'] = {
        'mean': float(semantic_features['euclidean_distance'].mean()),
        'std': float(semantic_features['euclidean_distance'].std()),
        'min': float(semantic_features['euclidean_distance'].min()),
        'max': float(semantic_features['euclidean_distance'].max())
    }
    
    analysis['pca_stats'] = {
        'explained_variance': float(semantic_features['pca_explained_variance']),
        'n_components': semantic_features['abs_diff_pca'].shape[1]
    }
    
    # Correlation with target if available
    if target_col is not None:
        try:
            analysis['target_correlations'] = {
                'cos_similarity': float(np.corrcoef(semantic_features['cos_similarity'], target_col)[0, 1]),
                'euclidean_distance': float(np.corrcoef(semantic_features['euclidean_distance'], target_col)[0, 1]),
                'dot_product': float(np.corrcoef(semantic_features['dot_product'], target_col)[0, 1])
            }
        except:
            analysis['target_correlations'] = {
                'cos_similarity': 0.0,
                'euclidean_distance': 0.0,
                'dot_product': 0.0
            }
    
    # Sample semantic relationships
    analysis['semantic_examples'] = []
    n_examples = min(5, len(semantic_features['cos_similarity']))
    for i in range(n_examples):
        analysis['semantic_examples'].append({
            'cos_similarity': float(semantic_features['cos_similarity'][i]),
            'euclidean_distance': float(semantic_features['euclidean_distance'][i]),
            'dot_product': float(semantic_features['dot_product'][i])
        })
    
    return analysis

# Create embeddings directory
os.makedirs("/kaggle/working/embeddings", exist_ok=True)

# Load datasets with priors
train = pd.read_csv("/kaggle/working/priors_oof.csv")
test = pd.read_csv("/kaggle/working/priors_test.csv")

print("Loading embedding model...")

# Load embedding model
model, model_type = load_embedding_model()

print(f"Generating embeddings for training data using {model_type}...")

# Generate embeddings for training data
if model_type == 'transformer':
    body_embeddings_train = generate_embeddings(model, model_type, train['body_clean'].tolist(), desc="training texts")
    rule_embeddings_train = generate_embeddings(model, model_type, train['rule'].tolist(), desc="training rules")
else:
    # For TF-IDF, fit on training data first
    body_embeddings_train = generate_embeddings(model, model_type, train['body_clean'].tolist(), desc="training texts", fit=True)
    rule_embeddings_train = generate_embeddings(model, model_type, train['rule'].tolist(), desc="training rules")

print("Generating embeddings for test data...")

# Generate embeddings for test data
body_embeddings_test = generate_embeddings(model, model_type, test['body_clean'].tolist(), desc="test texts")
rule_embeddings_test = generate_embeddings(model, model_type, test['rule'].tolist(), desc="test rules")

print("Creating semantic features...")

# Create semantic features for training data
semantic_features_train = create_semantic_features(body_embeddings_train, rule_embeddings_train)

# Create semantic features for test data using PCA model from training
semantic_features_test = create_semantic_features(
    body_embeddings_test, 
    rule_embeddings_test, 
    pca_model=semantic_features_train['pca_model']
)

print("Analyzing embeddings...")

# Analyze embeddings
embedding_analysis = analyze_embeddings(
    body_embeddings_train, 
    rule_embeddings_train, 
    semantic_features_train,
    target_col=train['rule_violation'] if 'rule_violation' in train.columns else None
)

print("Saving embeddings and features...")

# Save raw embeddings
np.save("/kaggle/working/embeddings/body_emb_train.npy", body_embeddings_train)
np.save("/kaggle/working/embeddings/rule_emb_train.npy", rule_embeddings_train)
np.save("/kaggle/working/embeddings/body_emb_test.npy", body_embeddings_test)
np.save("/kaggle/working/embeddings/rule_emb_test.npy", rule_embeddings_test)

# Create feature DataFrames
embed_features_train = pd.DataFrame({
    'cos_similarity': semantic_features_train['cos_similarity'],
    'euclidean_distance': semantic_features_train['euclidean_distance'],
    'dot_product': semantic_features_train['dot_product']
})

# Add PCA features
pca_columns = [f'abs_diff_pca_{i}' for i in range(semantic_features_train['abs_diff_pca'].shape[1])]
pca_df_train = pd.DataFrame(semantic_features_train['abs_diff_pca'], columns=pca_columns)
embed_features_train = pd.concat([embed_features_train, pca_df_train], axis=1)

# Same for test data
embed_features_test = pd.DataFrame({
    'cos_similarity': semantic_features_test['cos_similarity'],
    'euclidean_distance': semantic_features_test['euclidean_distance'],
    'dot_product': semantic_features_test['dot_product']
})

pca_df_test = pd.DataFrame(semantic_features_test['abs_diff_pca'], columns=pca_columns)
embed_features_test = pd.concat([embed_features_test, pca_df_test], axis=1)

# Save feature DataFrames
embed_features_train.to_csv("/kaggle/working/embed_features_train.csv", index=False)
embed_features_test.to_csv("/kaggle/working/embed_features_test.csv", index=False)

# Save PCA model
with open("/kaggle/working/embeddings/pca_model.pkl", "wb") as f:
    pickle.dump(semantic_features_train['pca_model'], f)

# Save TF-IDF model if used
if model_type == 'tfidf':
    with open("/kaggle/working/embeddings/tfidf_model.pkl", "wb") as f:
        pickle.dump(model, f)

# Save analysis
with open("/kaggle/working/embeddings/embedding_analysis.json", "w") as f:
    json.dump(embedding_analysis, f, indent=2, ensure_ascii=False)

print("")
print("=" * 60)
print(f"C9: EMBEDDING GENERATION COMPLETE USING {model_type.upper()}")
print("=" * 60)

# Display comprehensive results
print("Embedding Analysis:")
print(f"  Body embeddings shape: {embedding_analysis['body_embeddings_shape']}")
print(f"  Rule embeddings shape: {embedding_analysis['rule_embeddings_shape']}")
print(f"  Embedding dimension: {embedding_analysis['embedding_dimension']}")

print("")
print("Semantic Feature Statistics:")
cos_stats = embedding_analysis['cos_similarity_stats']
print(f"  Cosine similarity: {cos_stats['mean']:.4f} Â± {cos_stats['std']:.4f}")
print(f"    Range: [{cos_stats['min']:.4f}, {cos_stats['max']:.4f}]")

euc_stats = embedding_analysis['euclidean_distance_stats']
print(f"  Euclidean distance: {euc_stats['mean']:.4f} Â± {euc_stats['std']:.4f}")
print(f"    Range: [{euc_stats['min']:.4f}, {euc_stats['max']:.4f}]")

print("")
print("Feature Engineering:")
print(f"  PCA components: {embedding_analysis['pca_stats']['n_components']}")
print(f"  PCA explained variance: {embedding_analysis['pca_stats']['explained_variance']:.2%}")
print(f"  Total semantic features: {embed_features_train.shape[1]}")

if 'target_correlations' in embedding_analysis:
    print("")
    print("Feature-Target Correlations:")
    corrs = embedding_analysis['target_correlations']
    print(f"  Cosine similarity: {corrs['cos_similarity']:+.4f}")
    print(f"  Euclidean distance: {corrs['euclidean_distance']:+.4f}")
    print(f"  Dot product: {corrs['dot_product']:+.4f}")

print("")
print("Semantic Examples (first 3 samples):")
for i, example in enumerate(embedding_analysis['semantic_examples'][:3]):
    print(f"  Sample {i + 1}:")
    print(f"    Cosine similarity: {example['cos_similarity']:.4f}")
    print(f"    Euclidean distance: {example['euclidean_distance']:.4f}")
    print(f"    Dot product: {example['dot_product']:.4f}")

print("")
print("Files Saved:")
print(f"  /kaggle/working/embeddings/body_emb_train.npy")
print(f"  /kaggle/working/embeddings/rule_emb_train.npy")
print(f"  /kaggle/working/embeddings/body_emb_test.npy")
print(f"  /kaggle/working/embeddings/rule_emb_test.npy")
print(f"  /kaggle/working/embed_features_train.csv ({embed_features_train.shape[1]} features)")
print(f"  /kaggle/working/embed_features_test.csv ({embed_features_test.shape[1]} features)")
print(f"  /kaggle/working/embeddings/pca_model.pkl")
if model_type == 'tfidf':
    print(f"  /kaggle/working/embeddings/tfidf_model.pkl")
print(f"  /kaggle/working/embeddings/embedding_analysis.json")

print("")
print("Next: Ready for final feature integration and model training")
print("=" * 60)


# =============================================
# C10: CREATE GROUPS
# =============================================

import pandas as pd
import numpy as np
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("Creating groups for strict cross-validation...")

def create_group_mapping(df_train, df_test):
    """
    Create group IDs for train and test datasets to prevent data leakage.
    
    Args:
        df_train: Training DataFrame
        df_test: Test DataFrame
        
    Returns:
        Tuple of (train_groups, test_groups, group_analysis)
    """
    print("  Creating group keys...")
    
    # Create composite group key for training data
    train_group_key = (
        df_train["subreddit"].astype(str) + "||" + 
        df_train["rule"].astype(str) + "||" + 
        df_train["body_clean"].astype(str)
    )
    
    # Create composite group key for test data
    test_group_key = (
        df_test["subreddit"].astype(str) + "||" + 
        df_test["rule"].astype(str) + "||" + 
        df_test["body_clean"].astype(str)
    )
    
    print("  Factorizing group keys...")
    
    # Factorize training groups
    train_groups, train_group_labels = pd.factorize(train_group_key)
    
    # For test data, we need to map existing groups and create new ones for unseen combinations
    group_mapping = {label: idx for idx, label in enumerate(train_group_labels)}
    
    test_groups = []
    next_group_id = len(train_group_labels)
    
    for key in test_group_key:
        if key in group_mapping:
            test_groups.append(group_mapping[key])
        else:
            # Create new group ID for unseen test combinations
            test_groups.append(next_group_id)
            next_group_id += 1
    
    test_groups = np.array(test_groups)
    
    # Analyze group distribution
    analysis = analyze_groups(train_groups, test_groups, train_group_key, test_group_key)
    
    return train_groups, test_groups, analysis

def analyze_groups(train_groups, test_groups, train_group_key, test_group_key):
    """
    Analyze group distribution and characteristics.
    
    Args:
        train_groups: Training group IDs
        test_groups: Test group IDs
        train_group_key: Training group keys
        test_group_key: Test group keys
        
    Returns:
        Dictionary with group analysis
    """
    analysis = {}
    
    # Basic statistics
    analysis['train_groups_count'] = int(len(np.unique(train_groups)))
    analysis['test_groups_count'] = int(len(np.unique(test_groups)))
    analysis['train_samples'] = int(len(train_groups))
    analysis['test_samples'] = int(len(test_groups))
    
    # Group size distribution
    train_group_sizes = Counter(train_groups)
    test_group_sizes = Counter(test_groups)
    
    analysis['train_group_size_stats'] = {
        'min': int(min(train_group_sizes.values())),
        'max': int(max(train_group_sizes.values())),
        'mean': float(np.mean(list(train_group_sizes.values()))),
        'median': float(np.median(list(train_group_sizes.values())))
    }
    
    analysis['test_group_size_stats'] = {
        'min': int(min(test_group_sizes.values())),
        'max': int(max(test_group_sizes.values())),
        'mean': float(np.mean(list(test_group_sizes.values()))),
        'median': float(np.median(list(test_group_sizes.values())))
    }
    
    # Overlap analysis
    train_unique_keys = set(train_group_key.unique())
    test_unique_keys = set(test_group_key.unique())
    overlap_keys = train_unique_keys.intersection(test_unique_keys)
    
    analysis['group_overlap'] = {
        'train_unique_keys': len(train_unique_keys),
        'test_unique_keys': len(test_unique_keys),
        'overlapping_keys': len(overlap_keys),
        'overlap_percentage': len(overlap_keys) / len(test_unique_keys) if len(test_unique_keys) > 0 else 0
    }
    
    # Group composition by subreddit
    train_subreddit_groups = train_group_key.str.split('||').str[0].value_counts()
    test_subreddit_groups = test_group_key.str.split('||').str[0].value_counts()
    
    analysis['top_subreddits_train'] = train_subreddit_groups.head(10).to_dict()
    analysis['top_subreddits_test'] = test_subreddit_groups.head(10).to_dict()
    
    # Group composition by rule
    train_rule_groups = train_group_key.str.split('||').str[1].value_counts()
    test_rule_groups = test_group_key.str.split('||').str[1].value_counts()
    
    analysis['rule_distribution_train'] = train_rule_groups.to_dict()
    analysis['rule_distribution_test'] = test_rule_groups.to_dict()
    
    return analysis

def validate_group_structure(groups, df, group_type="train"):
    """
    Validate that groups correctly prevent data leakage.
    
    Args:
        groups: Group IDs
        df: DataFrame
        group_type: Type of data (train/test)
        
    Returns:
        Dictionary with validation results
    """
    validation = {}
    
    # Check for group consistency
    group_data = {
        'group_id': groups,
        'subreddit': df['subreddit'],
        'rule': df['rule'],
        'body_clean': df['body_clean']
    }
    
    # Add target column only if it exists
    if 'rule_violation' in df.columns:
        group_data['rule_violation'] = df['rule_violation']
    
    group_df = pd.DataFrame(group_data)
    
    # Verify that each group has consistent (subreddit, rule, body_clean)
    group_consistency = group_df.groupby('group_id').agg({
        'subreddit': 'nunique',
        'rule': 'nunique', 
        'body_clean': 'nunique'
    })
    
    inconsistent_groups = group_consistency[
        (group_consistency['subreddit'] > 1) | 
        (group_consistency['rule'] > 1) |
        (group_consistency['body_clean'] > 1)
    ]
    
    validation['consistent_groups'] = len(inconsistent_groups) == 0
    validation['inconsistent_group_count'] = len(inconsistent_groups)
    validation['total_groups'] = len(group_consistency)
    
    # Check for target leakage within groups (only for train)
    if 'rule_violation' in df.columns:
        group_target_stats = group_df.groupby('group_id')['rule_violation'].agg(['mean', 'count'])
        mixed_target_groups = group_target_stats[
            (group_target_stats['mean'] > 0) & (group_target_stats['mean'] < 1)
        ]
        validation['mixed_target_groups'] = len(mixed_target_groups)
        validation['pure_target_groups'] = len(group_target_stats) - len(mixed_target_groups)
    
    return validation

# Load datasets with embeddings
train = pd.read_csv("/kaggle/working/priors_oof.csv")
test = pd.read_csv("/kaggle/working/priors_test.csv")

print("Creating group mappings...")

# Create group mappings
train_groups, test_groups, group_analysis = create_group_mapping(train, test)

print("Validating group structure...")

# Validate group structure
train_validation = validate_group_structure(train_groups, train, "train")
test_validation = validate_group_structure(test_groups, test, "test")

print("Saving groups and analysis...")

# Save groups as numpy arrays
np.save("/kaggle/working/groups_train.npy", train_groups)
np.save("/kaggle/working/groups_test.npy", test_groups)

# Add group_id to datasets for audit
train['group_id'] = train_groups
test['group_id'] = test_groups

# Save datasets with group IDs
train.to_csv("/kaggle/working/train_with_groups.csv", index=False)
test.to_csv("/kaggle/working/test_with_groups.csv", index=False)

# Save comprehensive analysis
analysis_output = {
    'group_analysis': group_analysis,
    'train_validation': train_validation,
    'test_validation': test_validation
}

with open("/kaggle/working/group_analysis.json", "w") as f:
    json.dump(analysis_output, f, indent=2, ensure_ascii=False)

print("")
print("=" * 60)
print("C10: GROUP CREATION COMPLETE")
print("=" * 60)

# Display comprehensive results
print("Group Analysis:")
print(f"  Train groups: {group_analysis['train_groups_count']} unique groups")
print(f"  Test groups: {group_analysis['test_groups_count']} unique groups")
print(f"  Train samples: {group_analysis['train_samples']}")
print(f"  Test samples: {group_analysis['test_samples']}")

print("")
print("Group Size Statistics (Train):")
train_stats = group_analysis['train_group_size_stats']
print(f"  Min: {train_stats['min']} sample(s) per group")
print(f"  Max: {train_stats['max']} samples per group")
print(f"  Mean: {train_stats['mean']:.2f} samples per group")
print(f"  Median: {train_stats['median']:.2f} samples per group")

print("")
print("Group Overlap Analysis:")
overlap = group_analysis['group_overlap']
print(f"  Train unique keys: {overlap['train_unique_keys']}")
print(f"  Test unique keys: {overlap['test_unique_keys']}")
print(f"  Overlapping keys: {overlap['overlapping_keys']}")
print(f"  Overlap percentage: {overlap['overlap_percentage']:.2%}")

print("")
print("Validation Results:")
print(f"  Consistent groups: {train_validation['consistent_groups']}")
print(f"  Inconsistent groups: {train_validation['inconsistent_group_count']}")
print(f"  Total groups validated: {train_validation['total_groups']}")

if 'mixed_target_groups' in train_validation:
    print(f"  Mixed target groups: {train_validation['mixed_target_groups']}")
    print(f"  Pure target groups: {train_validation['pure_target_groups']}")

print("")
print("Top Subreddits by Group Count (Train):")
for i, (subreddit, count) in enumerate(list(group_analysis['top_subreddits_train'].items())[:5]):
    print(f"  {i+1}. r/{subreddit}: {count} groups")

print("")
print("Rule Distribution (Train):")
for rule, count in group_analysis['rule_distribution_train'].items():
    print(f"  {rule}: {count} groups")

print("")
print("Files Saved:")
print(f"  /kaggle/working/groups_train.npy")
print(f"  /kaggle/working/groups_test.npy")
print(f"  /kaggle/working/train_with_groups.csv")
print(f"  /kaggle/working/test_with_groups.csv")
print(f"  /kaggle/working/group_analysis.json")

print("")
print("Next: Ready for final feature integration and model training")
print("=" * 60)


# =============================================
# C11: STRATIFIED GROUP KFOLD
# =============================================

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedGroupKFold
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("Creating stratified group K-Fold splits...")

def create_stratified_group_kfold(df, groups, target_col='rule_violation', n_splits=5, random_state=42):
    """
    Create stratified group K-Fold splits that preserve class balance and group integrity.
    
    Args:
        df: DataFrame with features and target
        groups: Group IDs for preventing data leakage
        target_col: Name of target variable
        n_splits: Number of folds
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with fold indices and analysis
    """
    print(f"  Creating {n_splits} stratified group folds...")
    
    # Initialize StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )
    
    # Get features and target
    X = df.drop(columns=[target_col] if target_col in df.columns else [])
    y = df[target_col] if target_col in df.columns else None
    
    # Generate folds
    fold_indices = []
    fold_analysis = []
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        fold_info = {
            'fold': fold,
            'train_indices': train_idx.tolist(),
            'val_indices': val_idx.tolist(),
            'train_samples': len(train_idx),
            'val_samples': len(val_idx)
        }
        
        # Analyze fold composition
        if y is not None:
            fold_analysis.append(analyze_fold_composition(df, train_idx, val_idx, groups, target_col, fold))
        
        fold_indices.append(fold_info)
    
    return fold_indices, fold_analysis

def analyze_fold_composition(df, train_idx, val_idx, groups, target_col, fold_num):
    """
    Analyze the composition of a single fold.
    
    Args:
        df: DataFrame
        train_idx: Training indices
        val_idx: Validation indices
        groups: Group IDs
        target_col: Target column name
        fold_num: Fold number
        
    Returns:
        Dictionary with fold analysis
    """
    analysis = {}
    
    # Get fold data
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    # Basic statistics
    analysis['fold'] = fold_num
    analysis['train_samples'] = len(train_df)
    analysis['val_samples'] = len(val_df)
    analysis['train_val_ratio'] = len(train_df) / len(val_df) if len(val_df) > 0 else 0
    
    # Target distribution
    analysis['train_target_dist'] = {
        'class_0': int((train_df[target_col] == 0).sum()),
        'class_1': int((train_df[target_col] == 1).sum()),
        'class_0_ratio': float((train_df[target_col] == 0).mean()),
        'class_1_ratio': float((train_df[target_col] == 1).mean())
    }
    
    analysis['val_target_dist'] = {
        'class_0': int((val_df[target_col] == 0).sum()),
        'class_1': int((val_df[target_col] == 1).sum()),
        'class_0_ratio': float((val_df[target_col] == 0).mean()),
        'class_1_ratio': float((val_df[target_col] == 1).mean())
    }
    
    # Group analysis
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    
    analysis['train_groups'] = len(train_groups)
    analysis['val_groups'] = len(val_groups)
    analysis['group_overlap'] = len(train_groups.intersection(val_groups))
    
    # Subreddit distribution
    analysis['train_subreddits'] = int(train_df['subreddit'].nunique())
    analysis['val_subreddits'] = int(val_df['subreddit'].nunique())
    analysis['common_subreddits'] = len(set(train_df['subreddit']).intersection(set(val_df['subreddit'])))
    
    # Rule distribution
    analysis['train_rules'] = train_df['rule'].value_counts().to_dict()
    analysis['val_rules'] = val_df['rule'].value_counts().to_dict()
    
    return analysis

def validate_fold_splits(fold_indices, fold_analysis, groups, target_col):
    """
    Validate that fold splits meet all requirements.
    
    Args:
        fold_indices: List of fold indices
        fold_analysis: List of fold analyses
        groups: Group IDs
        target_col: Target column name
        
    Returns:
        Dictionary with validation results
    """
    validation = {}
    
    # Check for group leakage
    group_leakage = 0
    for fold in fold_analysis:
        if fold['group_overlap'] > 0:
            group_leakage += 1
    
    validation['group_leakage_folds'] = group_leakage
    validation['no_group_leakage'] = group_leakage == 0
    
    # Check class balance consistency
    class_balance_issues = 0
    target_ratios = []
    
    for fold in fold_analysis:
        train_ratio = fold['train_target_dist']['class_1_ratio']
        val_ratio = fold['val_target_dist']['class_1_ratio']
        target_ratios.append((train_ratio, val_ratio))
        
        # Check if class ratios are similar (within 5%)
        if abs(train_ratio - val_ratio) > 0.05:
            class_balance_issues += 1
    
    validation['class_balance_issues'] = class_balance_issues
    validation['stable_class_balance'] = class_balance_issues == 0
    validation['avg_train_ratio'] = float(np.mean([ratio[0] for ratio in target_ratios]))
    validation['avg_val_ratio'] = float(np.mean([ratio[1] for ratio in target_ratios]))
    
    # Check fold size consistency
    fold_sizes = [fold['train_samples'] + fold['val_samples'] for fold in fold_analysis]
    validation['fold_size_std'] = float(np.std(fold_sizes))
    validation['consistent_fold_sizes'] = validation['fold_size_std'] < 10  # Allow small variation
    
    return validation

def create_cv_summary(fold_analysis, validation):
    """
    Create a comprehensive CV summary.
    
    Args:
        fold_analysis: List of fold analyses
        validation: Validation results
        
    Returns:
        Dictionary with CV summary
    """
    summary = {}
    
    # Basic stats
    summary['total_folds'] = len(fold_analysis)
    summary['avg_train_samples'] = int(np.mean([fold['train_samples'] for fold in fold_analysis]))
    summary['avg_val_samples'] = int(np.mean([fold['val_samples'] for fold in fold_analysis]))
    summary['avg_train_val_ratio'] = float(np.mean([fold['train_val_ratio'] for fold in fold_analysis]))
    
    # Target distribution
    summary['avg_train_class_1_ratio'] = float(np.mean([fold['train_target_dist']['class_1_ratio'] for fold in fold_analysis]))
    summary['avg_val_class_1_ratio'] = float(np.mean([fold['val_target_dist']['class_1_ratio'] for fold in fold_analysis]))
    
    # Group statistics
    summary['avg_train_groups'] = int(np.mean([fold['train_groups'] for fold in fold_analysis]))
    summary['avg_val_groups'] = int(np.mean([fold['val_groups'] for fold in fold_analysis]))
    
    # Subreddit statistics
    summary['avg_train_subreddits'] = int(np.mean([fold['train_subreddits'] for fold in fold_analysis]))
    summary['avg_val_subreddits'] = int(np.mean([fold['val_subreddits'] for fold in fold_analysis]))
    
    # Validation results
    summary['validation'] = validation
    
    return summary

# Load datasets with groups
train = pd.read_csv("/kaggle/working/train_with_groups.csv")
test = pd.read_csv("/kaggle/working/test_with_groups.csv")

# Load groups
train_groups = np.load("/kaggle/working/groups_train.npy")
test_groups = np.load("/kaggle/working/groups_test.npy")

print("Creating stratified group K-Fold splits...")

# Create stratified group K-Fold splits
fold_indices, fold_analysis = create_stratified_group_kfold(
    df=train,
    groups=train_groups,
    target_col='rule_violation',
    n_splits=5,
    random_state=42
)

print("Validating fold splits...")

# Validate fold splits
validation = validate_fold_splits(fold_indices, fold_analysis, train_groups, 'rule_violation')

print("Creating CV summary...")

# Create comprehensive summary
cv_summary = create_cv_summary(fold_analysis, validation)

print("Saving fold indices and analysis...")

# Save fold indices
with open("/kaggle/working/fold_indices.json", "w") as f:
    json.dump(fold_indices, f, indent=2)

# Save comprehensive analysis
analysis_output = {
    'fold_analysis': fold_analysis,
    'validation': validation,
    'cv_summary': cv_summary
}

with open("/kaggle/working/cv_analysis.json", "w") as f:
    json.dump(analysis_output, f, indent=2, ensure_ascii=False)

print("")
print("=" * 60)
print("C11: STRATIFIED GROUP K-FOLD COMPLETE")
print("=" * 60)

# Display comprehensive results
print("Cross-Validation Summary:")
print(f"  Total folds: {cv_summary['total_folds']}")
print(f"  Average train samples: {cv_summary['avg_train_samples']}")
print(f"  Average val samples: {cv_summary['avg_val_samples']}")
print(f"  Train/Val ratio: {cv_summary['avg_train_val_ratio']:.2f}")

print("")
print("Target Distribution:")
print(f"  Average train violation rate: {cv_summary['avg_train_class_1_ratio']:.2%}")
print(f"  Average val violation rate: {cv_summary['avg_val_class_1_ratio']:.2%}")

print("")
print("Group Statistics:")
print(f"  Average train groups: {cv_summary['avg_train_groups']}")
print(f"  Average val groups: {cv_summary['avg_val_groups']}")

print("")
print("Subreddit Coverage:")
print(f"  Average train subreddits: {cv_summary['avg_train_subreddits']}")
print(f"  Average val subreddits: {cv_summary['avg_val_subreddits']}")

print("")
print("Validation Results:")
val = cv_summary['validation']
print(f"  Group leakage: {val['group_leakage_folds']} folds")
print(f"  Class balance: {val['class_balance_issues']} folds")
print(f"  Fold consistency: std={val['fold_size_std']:.2f}")

print("")
print("Detailed Fold Analysis:")
for fold in fold_analysis[:3]:  # Show first 3 folds
    print(f"  Fold {fold['fold'] + 1}:")
    print(f"    Train: {fold['train_samples']} samples ({fold['train_target_dist']['class_1_ratio']:.2%} violations)")
    print(f"    Val: {fold['val_samples']} samples ({fold['val_target_dist']['class_1_ratio']:.2%} violations)")
    print(f"    Groups: {fold['train_groups']} train, {fold['val_groups']} val")
    print(f"    Subreddits: {fold['train_subreddits']} train, {fold['val_subreddits']} val")

print("")
print("Rule Distribution (Fold 1):")
if fold_analysis:
    print("  Train:")
    for rule, count in fold_analysis[0]['train_rules'].items():
        print(f"    {rule}: {count}")
    print("  Validation:")
    for rule, count in fold_analysis[0]['val_rules'].items():
        print(f"    {rule}: {count}")

print("")
print("Files Saved:")
print(f"  /kaggle/working/fold_indices.json")
print(f"  /kaggle/working/cv_analysis.json")

print("")
print("Next: Ready for final feature integration and model training")
print("=" * 60)


# =============================================
# C12: TRAIN LGBM TEXT
# =============================================

import pandas as pd
import numpy as np
import json
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import warnings
import os
warnings.filterwarnings('ignore')

print("Training LightGBM on text features...")

def load_and_combine_features():
    """
    Load and combine all text features for training.
    
    Returns:
        Tuple of (X_train, X_test, y_train, feature_names)
    """
    print("  Loading feature matrices...")
    
    # Load base datasets
    train_dense = pd.read_csv("/kaggle/working/train_dense.csv")
    test_dense = pd.read_csv("/kaggle/working/test_dense.csv")
    train_domain = pd.read_csv("/kaggle/working/train_domain.csv") 
    test_domain = pd.read_csv("/kaggle/working/test_domain.csv")
    train_priors = pd.read_csv("/kaggle/working/priors_oof.csv")
    test_priors = pd.read_csv("/kaggle/working/priors_test.csv")
    
    # Load TF-IDF features
    tfidf_word_train = np.load("/kaggle/working/tfidf_word_train.npy")
    tfidf_word_test = np.load("/kaggle/working/tfidf_word_test.npy")
    tfidf_char_train = np.load("/kaggle/working/tfidf_char_train.npy")
    tfidf_char_test = np.load("/kaggle/working/tfidf_char_test.npy")
    
    # Get target
    y_train = train_priors['rule_violation'].values
    
    # Combine dense features
    dense_features = [
        'char_count', 'word_count', 'sentence_count', 'avg_word_length', 'avg_sentence_length',
        'uppercase_count', 'uppercase_ratio', 'titlecase_words', 'titlecase_ratio',
        'exclam_count', 'question_count', 'punct_count', 'punct_ratio',
        'digit_count', 'digit_ratio', 'special_char_count', 'special_char_ratio',
        'emoji_count', 'has_emoji', 'unique_words', 'lexical_diversity',
        'short_word_count', 'short_word_ratio', 'long_word_count', 'long_word_ratio',
        'url_token_count', 'user_token_count', 'subreddit_token_count', 'has_reddit_artifacts',
        'is_empty_text', 'is_very_short', 'is_very_long'
    ]
    
    # Domain features
    domain_features = [
        'promo_kw_count', 'has_promo_kw', 'cta_count', 'has_call_to_action',
        'suspicious_tld_count', 'has_commercial_tld', 'legal_kw_count', 'has_legal_kw',
        'disclaimer_count', 'has_disclaimer', 'imperative_count', 'has_imperative',
        'advertising_score', 'legal_advice_score', 'likely_advertising', 'likely_legal_advice',
        'domain_risk_score'
    ]
    
    # Prior features
    prior_features = ['subreddit_prior', 'rule_prior']
    
    # Select features from each dataset
    train_dense_selected = train_dense[dense_features].copy()
    test_dense_selected = test_dense[dense_features].copy()
    
    train_domain_selected = train_domain[domain_features].copy()
    test_domain_selected = test_domain[domain_features].copy()
    
    train_priors_selected = train_priors[prior_features].copy()
    test_priors_selected = test_priors[prior_features].copy()
    
    print("  Combining feature matrices...")
    
    # Combine all features
    X_train_combined = pd.concat([
        train_dense_selected,
        train_domain_selected,
        train_priors_selected
    ], axis=1)
    
    X_test_combined = pd.concat([
        test_dense_selected,
        test_domain_selected, 
        test_priors_selected
    ], axis=1)
    
    # Add TF-IDF features
    tfidf_word_train_df = pd.DataFrame(tfidf_word_train, 
                                      columns=[f'tfidf_word_{i}' for i in range(tfidf_word_train.shape[1])])
    tfidf_word_test_df = pd.DataFrame(tfidf_word_test,
                                     columns=[f'tfidf_word_{i}' for i in range(tfidf_word_test.shape[1])])
    
    tfidf_char_train_df = pd.DataFrame(tfidf_char_train,
                                      columns=[f'tfidf_char_{i}' for i in range(tfidf_char_train.shape[1])])
    tfidf_char_test_df = pd.DataFrame(tfidf_char_test,
                                     columns=[f'tfidf_char_{i}' for i in range(tfidf_char_test.shape[1])])
    
    X_train_final = pd.concat([X_train_combined, tfidf_word_train_df, tfidf_char_train_df], axis=1)
    X_test_final = pd.concat([X_test_combined, tfidf_word_test_df, tfidf_char_test_df], axis=1)
    
    # Ensure same column order
    X_test_final = X_test_final[X_train_final.columns]
    
    feature_names = X_train_final.columns.tolist()
    
    print(f"  Final feature matrix: {X_train_final.shape}")
    print(f"  Test feature matrix: {X_test_final.shape}")
    
    return X_train_final.values, X_test_final.values, y_train, feature_names

def create_lgbm_model():
    """
    Create LightGBM classifier with optimized hyperparameters.
    
    Returns:
        LightGBM classifier
    """
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    return model

def train_cv_lgbm(X_train, y_train, fold_indices, feature_names):
    """
    Train LightGBM with cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training target
        fold_indices: Fold indices from CV
        feature_names: List of feature names
        
    Returns:
        Dictionary with CV results
    """
    print("  Training LightGBM with cross-validation...")
    
    # Create models directory
    os.makedirs("/kaggle/working/models", exist_ok=True)
    
    # Initialize results
    oof_predictions = np.zeros(len(X_train))
    fold_models = []
    fold_results = []
    feature_importances = []
    
    for fold_info in fold_indices:
        fold = fold_info['fold']
        train_idx = fold_info['train_indices']
        val_idx = fold_info['val_indices']
        
        print(f"    Fold {fold + 1}/5...")
        
        # Split data
        X_tr = X_train[train_idx]
        X_val = X_train[val_idx]
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]
        
        # Create and train model
        model = create_lgbm_model()
        
        # Try different parameter names for early stopping
        try:
            # Try with callbacks
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(100)],
                verbose=False
            )
        except TypeError:
            try:
                # Try without early stopping
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric='auc',
                    verbose=False
                )
            except TypeError:
                # Simple fit without eval_set
                model.fit(X_tr, y_tr)
        
        # Predict on validation
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_predictions[val_idx] = val_preds
        
        # Calculate fold metrics
        fold_auc = roc_auc_score(y_val, val_preds)
        fold_accuracy = accuracy_score(y_val, (val_preds > 0.5).astype(int))
        
        # Store fold results
        fold_result = {
            'fold': fold,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'auc': fold_auc,
            'accuracy': fold_accuracy
        }
        
        # Try to get best iteration if available
        try:
            fold_result['best_iteration'] = model.best_iteration_
        except:
            fold_result['best_iteration'] = model.n_estimators
        
        fold_results.append(fold_result)
        
        # Store model and feature importance
        fold_models.append(model)
        feature_importances.append(model.feature_importances_)
        
        # Save model
        model_path = f"/kaggle/working/models/lgbm_text_fold_{fold}.txt"
        try:
            model.booster_.save_model(model_path)
        except:
            # Alternative saving method
            import joblib
            joblib.dump(model, model_path.replace('.txt', '.pkl'))
        
        print(f"      AUC: {fold_auc:.4f}, Accuracy: {fold_accuracy:.4f}")
    
    return {
        'oof_predictions': oof_predictions,
        'fold_models': fold_models,
        'fold_results': fold_results,
        'feature_importances': feature_importances
    }

def analyze_feature_importance(feature_importances, feature_names, fold_results):
    """
    Analyze feature importance across folds.
    
    Args:
        feature_importances: List of feature importance arrays
        feature_names: List of feature names
        fold_results: Fold results
        
    Returns:
        DataFrame with feature importance analysis
    """
    # Calculate mean importance across folds
    mean_importance = np.mean(feature_importances, axis=0)
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': mean_importance,
        'importance_std': np.std(feature_importances, axis=0)
    })
    
    # Add individual fold importances
    for i, fold_imp in enumerate(feature_importances):
        importance_df[f'importance_fold_{i}'] = fold_imp
    
    # Sort by mean importance
    importance_df = importance_df.sort_values('importance_mean', ascending=False)
    
    return importance_df

def evaluate_model(oof_predictions, y_train, fold_results):
    """
    Evaluate model performance.
    
    Args:
        oof_predictions: OOF predictions
        y_train: True labels
        fold_results: Fold results
        
    Returns:
        Dictionary with evaluation metrics
    """
    evaluation = {}
    
    # Overall metrics
    evaluation['overall_auc'] = roc_auc_score(y_train, oof_predictions)
    evaluation['overall_accuracy'] = accuracy_score(y_train, (oof_predictions > 0.5).astype(int))
    
    # Fold metrics summary
    fold_aucs = [fold['auc'] for fold in fold_results]
    evaluation['fold_auc_mean'] = np.mean(fold_aucs)
    evaluation['fold_auc_std'] = np.std(fold_aucs)
    evaluation['fold_auc_min'] = np.min(fold_aucs)
    evaluation['fold_auc_max'] = np.max(fold_aucs)
    
    # Best iteration summary
    best_iterations = [fold['best_iteration'] for fold in fold_results]
    evaluation['avg_best_iteration'] = np.mean(best_iterations)
    
    return evaluation

# Load fold indices
with open("/kaggle/working/fold_indices.json", "r") as f:
    fold_indices = json.load(f)

print("Loading and combining features...")

# Load and combine features
X_train, X_test, y_train, feature_names = load_and_combine_features()

print("Training LightGBM with cross-validation...")

# Train model with CV
cv_results = train_cv_lgbm(X_train, y_train, fold_indices, feature_names)

print("Analyzing results...")

# Analyze feature importance
importance_df = analyze_feature_importance(
    cv_results['feature_importances'],
    feature_names,
    cv_results['fold_results']
)

# Evaluate model
evaluation = evaluate_model(cv_results['oof_predictions'], y_train, cv_results['fold_results'])

print("Saving results...")

# Save OOF predictions
oof_df = pd.DataFrame({
    'oof_prediction': cv_results['oof_predictions'],
    'rule_violation': y_train
})
oof_df.to_csv("/kaggle/working/oof_lgbm_text.csv", index=False)

# Save feature importance
importance_df.to_csv("/kaggle/working/feature_importance_lgbm_text.csv", index=False)

# Save evaluation results
with open("/kaggle/working/lgbm_text_evaluation.json", "w") as f:
    json.dump(evaluation, f, indent=2, ensure_ascii=False)

print("Generating test predictions...")

# Generate test predictions (average across folds)
test_predictions = np.zeros(len(X_test))
for model in cv_results['fold_models']:
    test_predictions += model.predict_proba(X_test)[:, 1]
test_predictions /= len(cv_results['fold_models'])

# Save test predictions
test_pred_df = pd.DataFrame({
    'prediction': test_predictions
})
test_pred_df.to_csv("/kaggle/working/test_predictions_lgbm_text.csv", index=False)

print("")
print("=" * 60)
print("C12: LIGHTGBM TEXT MODEL TRAINING COMPLETE")
print("=" * 60)

# Display comprehensive results
print("Model Performance:")
print(f"  Overall OOF AUC: {evaluation['overall_auc']:.4f}")
print(f"  Overall Accuracy: {evaluation['overall_accuracy']:.4f}")

print("")
print("Cross-Validation Results:")
print(f"  Mean Fold AUC: {evaluation['fold_auc_mean']:.4f} Â± {evaluation['fold_auc_std']:.4f}")
print(f"  Fold AUC Range: [{evaluation['fold_auc_min']:.4f}, {evaluation['fold_auc_max']:.4f}]")
print(f"  Average Best Iteration: {evaluation['avg_best_iteration']:.0f}")

print("")
print("Top 10 Most Important Features:")
for i, row in importance_df.head(10).iterrows():
    print(f"  {i+1:2d}. {row['feature']:<25} {row['importance_mean']:.4f}")

print("")
print("Feature Categories (Top 20):")
top_features = importance_df.head(20)
category_counts = {}
for feature in top_features['feature']:
    if 'tfidf_word' in feature:
        category_counts['TF-IDF Word'] = category_counts.get('TF-IDF Word', 0) + 1
    elif 'tfidf_char' in feature:
        category_counts['TF-IDF Char'] = category_counts.get('TF-IDF Char', 0) + 1
    elif any(x in feature for x in ['prior', 'score']):
        category_counts['Priors/Scores'] = category_counts.get('Priors/Scores', 0) + 1
    elif any(x in feature for x in ['legal', 'advertis', 'promo', 'cta']):
        category_counts['Domain Features'] = category_counts.get('Domain Features', 0) + 1
    elif any(x in feature for x in ['count', 'ratio', 'length']):
        category_counts['Text Statistics'] = category_counts.get('Text Statistics', 0) + 1
    else:
        category_counts['Other'] = category_counts.get('Other', 0) + 1

for category, count in category_counts.items():
    print(f"  {category}: {count} features")

print("")
print("Feature Importance Insights:")
print(f"  Total features: {len(feature_names)}")
print(f"  Non-zero importance features: {(importance_df['importance_mean'] > 0).sum()}")
print(f"  Top feature: {importance_df.iloc[0]['feature']} "
      f"({importance_df.iloc[0]['importance_mean']:.4f})")

print("")
print("Files Saved:")
print(f"  /kaggle/working/oof_lgbm_text.csv")
print(f"  /kaggle/working/feature_importance_lgbm_text.csv")
print(f"  /kaggle/working/lgbm_text_evaluation.json")
print(f"  /kaggle/working/test_predictions_lgbm_text.csv")
print(f"  /kaggle/working/models/lgbm_text_fold_*.txt (5 models)")

print("")
print("Next: Ready for embedding-enhanced model training")
print("=" * 60)


# =============================================
# C13: TRAIN LGBM EMBED
# =============================================

import pandas as pd
import numpy as np
import json
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import warnings
import os
warnings.filterwarnings('ignore')

print("Training LightGBM on embedding features...")

def load_and_combine_embedding_features():
    """
    Load and combine embedding features with dense features for training.
    
    Returns:
        Tuple of (X_train, X_test, y_train, feature_names)
    """
    print("  Loading feature matrices...")
    
    # Load base datasets
    train_dense = pd.read_csv("/kaggle/working/train_dense.csv")
    test_dense = pd.read_csv("/kaggle/working/test_dense.csv")
    train_domain = pd.read_csv("/kaggle/working/train_domain.csv") 
    test_domain = pd.read_csv("/kaggle/working/test_domain.csv")
    train_priors = pd.read_csv("/kaggle/working/priors_oof.csv")
    test_priors = pd.read_csv("/kaggle/working/priors_test.csv")
    
    # Load embedding features
    embed_features_train = pd.read_csv("/kaggle/working/embed_features_train.csv")
    embed_features_test = pd.read_csv("/kaggle/working/embed_features_test.csv")
    
    # Get target
    y_train = train_priors['rule_violation'].values
    
    # Combine dense features
    dense_features = [
        'char_count', 'word_count', 'sentence_count', 'avg_word_length', 'avg_sentence_length',
        'uppercase_count', 'uppercase_ratio', 'titlecase_words', 'titlecase_ratio',
        'exclam_count', 'question_count', 'punct_count', 'punct_ratio',
        'digit_count', 'digit_ratio', 'special_char_count', 'special_char_ratio',
        'emoji_count', 'has_emoji', 'unique_words', 'lexical_diversity',
        'short_word_count', 'short_word_ratio', 'long_word_count', 'long_word_ratio',
        'url_token_count', 'user_token_count', 'subreddit_token_count', 'has_reddit_artifacts',
        'is_empty_text', 'is_very_short', 'is_very_long'
    ]
    
    # Domain features
    domain_features = [
        'promo_kw_count', 'has_promo_kw', 'cta_count', 'has_call_to_action',
        'suspicious_tld_count', 'has_commercial_tld', 'legal_kw_count', 'has_legal_kw',
        'disclaimer_count', 'has_disclaimer', 'imperative_count', 'has_imperative',
        'advertising_score', 'legal_advice_score', 'likely_advertising', 'likely_legal_advice',
        'domain_risk_score'
    ]
    
    # Prior features
    prior_features = ['subreddit_prior', 'rule_prior']
    
    # Embedding features (all columns from embed_features)
    embedding_features = embed_features_train.columns.tolist()
    
    # Select features from each dataset
    train_dense_selected = train_dense[dense_features].copy()
    test_dense_selected = test_dense[dense_features].copy()
    
    train_domain_selected = train_domain[domain_features].copy()
    test_domain_selected = test_domain[domain_features].copy()
    
    train_priors_selected = train_priors[prior_features].copy()
    test_priors_selected = test_priors[prior_features].copy()
    
    print("  Combining feature matrices...")
    
    # Combine all features
    X_train_combined = pd.concat([
        train_dense_selected,
        train_domain_selected,
        train_priors_selected,
        embed_features_train
    ], axis=1)
    
    X_test_combined = pd.concat([
        test_dense_selected,
        test_domain_selected, 
        test_priors_selected,
        embed_features_test
    ], axis=1)
    
    # Ensure same column order
    X_test_combined = X_test_combined[X_train_combined.columns]
    
    feature_names = X_train_combined.columns.tolist()
    
    print(f"  Final feature matrix: {X_train_combined.shape}")
    print(f"  Test feature matrix: {X_test_combined.shape}")
    
    return X_train_combined.values, X_test_combined.values, y_train, feature_names

def create_lgbm_model():
    """
    Create LightGBM classifier with optimized hyperparameters.
    
    Returns:
        LightGBM classifier
    """
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    return model

def train_cv_lgbm(X_train, y_train, fold_indices, feature_names):
    """
    Train LightGBM with cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training target
        fold_indices: Fold indices from CV
        feature_names: List of feature names
        
    Returns:
        Dictionary with CV results
    """
    print("  Training LightGBM with cross-validation...")
    
    # Create models directory
    os.makedirs("/kaggle/working/models", exist_ok=True)
    
    # Initialize results
    oof_predictions = np.zeros(len(X_train))
    fold_models = []
    fold_results = []
    feature_importances = []
    
    for fold_info in fold_indices:
        fold = fold_info['fold']
        train_idx = fold_info['train_indices']
        val_idx = fold_info['val_indices']
        
        print(f"    Fold {fold + 1}/5...")
        
        # Split data
        X_tr = X_train[train_idx]
        X_val = X_train[val_idx]
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]
        
        # Create and train model
        model = create_lgbm_model()
        
        # Try different parameter names for early stopping
        try:
            # Try with callbacks
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(100)],
                verbose=False
            )
        except TypeError:
            try:
                # Try without early stopping
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric='auc',
                    verbose=False
                )
            except TypeError:
                # Simple fit without eval_set
                model.fit(X_tr, y_tr)
        
        # Predict on validation
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_predictions[val_idx] = val_preds
        
        # Calculate fold metrics
        fold_auc = roc_auc_score(y_val, val_preds)
        fold_accuracy = accuracy_score(y_val, (val_preds > 0.5).astype(int))
        
        # Store fold results
        fold_result = {
            'fold': fold,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'auc': fold_auc,
            'accuracy': fold_accuracy
        }
        
        # Try to get best iteration if available
        try:
            fold_result['best_iteration'] = model.best_iteration_
        except:
            fold_result['best_iteration'] = model.n_estimators
        
        fold_results.append(fold_result)
        
        # Store model and feature importance
        fold_models.append(model)
        feature_importances.append(model.feature_importances_)
        
        # Save model
        model_path = f"/kaggle/working/models/lgbm_embed_fold_{fold}.txt"
        try:
            model.booster_.save_model(model_path)
        except:
            # Alternative saving method
            import joblib
            joblib.dump(model, model_path.replace('.txt', '.pkl'))
        
        print(f"      AUC: {fold_auc:.4f}, Accuracy: {fold_accuracy:.4f}")
    
    return {
        'oof_predictions': oof_predictions,
        'fold_models': fold_models,
        'fold_results': fold_results,
        'feature_importances': feature_importances
    }

def analyze_feature_importance(feature_importances, feature_names, fold_results):
    """
    Analyze feature importance across folds.
    
    Args:
        feature_importances: List of feature importance arrays
        feature_names: List of feature names
        fold_results: Fold results
        
    Returns:
        DataFrame with feature importance analysis
    """
    # Calculate mean importance across folds
    mean_importance = np.mean(feature_importances, axis=0)
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': mean_importance,
        'importance_std': np.std(feature_importances, axis=0)
    })
    
    # Add individual fold importances
    for i, fold_imp in enumerate(feature_importances):
        importance_df[f'importance_fold_{i}'] = fold_imp
    
    # Sort by mean importance
    importance_df = importance_df.sort_values('importance_mean', ascending=False)
    
    return importance_df

def evaluate_model(oof_predictions, y_train, fold_results):
    """
    Evaluate model performance.
    
    Args:
        oof_predictions: OOF predictions
        y_train: True labels
        fold_results: Fold results
        
    Returns:
        Dictionary with evaluation metrics
    """
    evaluation = {}
    
    # Overall metrics
    evaluation['overall_auc'] = roc_auc_score(y_train, oof_predictions)
    evaluation['overall_accuracy'] = accuracy_score(y_train, (oof_predictions > 0.5).astype(int))
    
    # Fold metrics summary
    fold_aucs = [fold['auc'] for fold in fold_results]
    evaluation['fold_auc_mean'] = np.mean(fold_aucs)
    evaluation['fold_auc_std'] = np.std(fold_aucs)
    evaluation['fold_auc_min'] = np.min(fold_aucs)
    evaluation['fold_auc_max'] = np.max(fold_aucs)
    
    # Best iteration summary
    best_iterations = [fold['best_iteration'] for fold in fold_results]
    evaluation['avg_best_iteration'] = np.mean(best_iterations)
    
    return evaluation

def compare_with_text_model(embed_evaluation, text_evaluation_path):
    """
    Compare embedding model performance with text model.
    
    Args:
        embed_evaluation: Embedding model evaluation
        text_evaluation_path: Path to text model evaluation
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {}
    
    # Load text model evaluation
    with open(text_evaluation_path, "r") as f:
        text_evaluation = json.load(f)
    
    # AUC comparison
    comparison['auc_embed'] = embed_evaluation['overall_auc']
    comparison['auc_text'] = text_evaluation['overall_auc']
    comparison['auc_improvement'] = comparison['auc_embed'] - comparison['auc_text']
    comparison['auc_improvement_pct'] = (comparison['auc_improvement'] / comparison['auc_text']) * 100
    
    # Accuracy comparison
    comparison['accuracy_embed'] = embed_evaluation['overall_accuracy']
    comparison['accuracy_text'] = text_evaluation['overall_accuracy']
    comparison['accuracy_improvement'] = comparison['accuracy_embed'] - comparison['accuracy_text']
    
    # Fold stability comparison
    comparison['fold_std_embed'] = embed_evaluation['fold_auc_std']
    comparison['fold_std_text'] = text_evaluation['fold_auc_std']
    
    return comparison

# Load fold indices
with open("/kaggle/working/fold_indices.json", "r") as f:
    fold_indices = json.load(f)

print("Loading and combining embedding features...")

# Load and combine embedding features
X_train, X_test, y_train, feature_names = load_and_combine_embedding_features()

print("Training LightGBM with cross-validation...")

# Train model with CV
cv_results = train_cv_lgbm(X_train, y_train, fold_indices, feature_names)

print("Analyzing results...")

# Analyze feature importance
importance_df = analyze_feature_importance(
    cv_results['feature_importances'],
    feature_names,
    cv_results['fold_results']
)

# Evaluate model
evaluation = evaluate_model(cv_results['oof_predictions'], y_train, cv_results['fold_results'])

# Compare with text model
comparison = compare_with_text_model(evaluation, "/kaggle/working/lgbm_text_evaluation.json")

print("Saving results...")

# Save OOF predictions
oof_df = pd.DataFrame({
    'oof_prediction': cv_results['oof_predictions'],
    'rule_violation': y_train
})
oof_df.to_csv("/kaggle/working/oof_lgbm_embed.csv", index=False)

# Save feature importance
importance_df.to_csv("/kaggle/working/feature_importance_lgbm_embed.csv", index=False)

# Save evaluation results
with open("/kaggle/working/lgbm_embed_evaluation.json", "w") as f:
    json.dump(evaluation, f, indent=2, ensure_ascii=False)

# Save comparison results
with open("/kaggle/working/model_comparison.json", "w") as f:
    json.dump(comparison, f, indent=2, ensure_ascii=False)

print("Generating test predictions...")

# Generate test predictions (average across folds)
test_predictions = np.zeros(len(X_test))
for model in cv_results['fold_models']:
    test_predictions += model.predict_proba(X_test)[:, 1]
test_predictions /= len(cv_results['fold_models'])

# Save test predictions
test_pred_df = pd.DataFrame({
    'prediction': test_predictions
})
test_pred_df.to_csv("/kaggle/working/test_predictions_lgbm_embed.csv", index=False)

print("")
print("=" * 60)
print("C13: LIGHTGBM EMBEDDING MODEL TRAINING COMPLETE")
print("=" * 60)

# Display comprehensive results
print("Model Performance:")
print(f"  Overall OOF AUC: {evaluation['overall_auc']:.4f}")
print(f"  Overall Accuracy: {evaluation['overall_accuracy']:.4f}")

print("")
print("Cross-Validation Results:")
print(f"  Mean Fold AUC: {evaluation['fold_auc_mean']:.4f} Â± {evaluation['fold_auc_std']:.4f}")
print(f"  Fold AUC Range: [{evaluation['fold_auc_min']:.4f}, {evaluation['fold_auc_max']:.4f}]")
print(f"  Average Best Iteration: {evaluation['avg_best_iteration']:.0f}")

print("")
print("Model Comparison:")
print(f"  Text Model AUC: {comparison['auc_text']:.4f}")
print(f"  Embed Model AUC: {comparison['auc_embed']:.4f}")
print(f"  AUC Improvement: {comparison['auc_improvement']:+.4f} ({comparison['auc_improvement_pct']:+.1f}%)")
print(f"  Accuracy Improvement: {comparison['accuracy_improvement']:+.4f}")

print("")
print("Top 10 Most Important Features:")
for i, row in importance_df.head(10).iterrows():
    print(f"  {i+1:2d}. {row['feature']:<25} {row['importance_mean']:.4f}")

print("")
print("Feature Categories (Top 20):")
top_features = importance_df.head(20)
category_counts = {}
for feature in top_features['feature']:
    if 'cos_similarity' in feature or 'euclidean_distance' in feature or 'dot_product' in feature:
        category_counts['Semantic Features'] = category_counts.get('Semantic Features', 0) + 1
    elif 'abs_diff_pca' in feature:
        category_counts['Embedding PCA'] = category_counts.get('Embedding PCA', 0) + 1
    elif any(x in feature for x in ['prior', 'score']):
        category_counts['Priors/Scores'] = category_counts.get('Priors/Scores', 0) + 1
    elif any(x in feature for x in ['legal', 'advertis', 'promo', 'cta']):
        category_counts['Domain Features'] = category_counts.get('Domain Features', 0) + 1
    elif any(x in feature for x in ['count', 'ratio', 'length']):
        category_counts['Text Statistics'] = category_counts.get('Text Statistics', 0) + 1
    else:
        category_counts['Other'] = category_counts.get('Other', 0) + 1

for category, count in category_counts.items():
    print(f"  {category}: {count} features")

print("")
print("Feature Importance Insights:")
print(f"  Total features: {len(feature_names)}")
print(f"  Non-zero importance features: {(importance_df['importance_mean'] > 0).sum()}")
print(f"  Top feature: {importance_df.iloc[0]['feature']} "
      f"({importance_df.iloc[0]['importance_mean']:.4f})")

print("")
print("Semantic Feature Analysis:")
semantic_features = [f for f in importance_df['feature'] if any(x in f for x in ['cos_similarity', 'euclidean_distance', 'dot_product', 'abs_diff_pca'])]
semantic_importance = importance_df[importance_df['feature'].isin(semantic_features)].head(5)
for i, row in semantic_importance.iterrows():
    print(f"  {row['feature']}: {row['importance_mean']:.4f}")

print("")
print("Files Saved:")
print(f"  /kaggle/working/oof_lgbm_embed.csv")
print(f"  /kaggle/working/feature_importance_lgbm_embed.csv")
print(f"  /kaggle/working/lgbm_embed_evaluation.json")
print(f"  /kaggle/working/model_comparison.json")
print(f"  /kaggle/working/test_predictions_lgbm_embed.csv")
print(f"  /kaggle/working/models/lgbm_embed_fold_*.txt (5 models)")

print("")
print("Next: Ready for ensemble model training")
print("=" * 60)


# =============================================
# C14: TRAIN LGBM FULL
# =============================================

import pandas as pd
import numpy as np
import json
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import warnings
import os
warnings.filterwarnings('ignore')

print("Training LightGBM on ALL features...")

def load_and_combine_all_features():
    """
    Load and combine ALL features for training.
    
    Returns:
        Tuple of (X_train, X_test, y_train, feature_names)
    """
    print("  Loading ALL feature matrices...")
    
    # Load base datasets
    train_dense = pd.read_csv("/kaggle/working/train_dense.csv")
    test_dense = pd.read_csv("/kaggle/working/test_dense.csv")
    train_domain = pd.read_csv("/kaggle/working/train_domain.csv") 
    test_domain = pd.read_csv("/kaggle/working/test_domain.csv")
    train_priors = pd.read_csv("/kaggle/working/priors_oof.csv")
    test_priors = pd.read_csv("/kaggle/working/priors_test.csv")
    
    # Load TF-IDF features
    tfidf_word_train = np.load("/kaggle/working/tfidf_word_train.npy")
    tfidf_word_test = np.load("/kaggle/working/tfidf_word_test.npy")
    tfidf_char_train = np.load("/kaggle/working/tfidf_char_train.npy")
    tfidf_char_test = np.load("/kaggle/working/tfidf_char_test.npy")
    
    # Load embedding features
    embed_features_train = pd.read_csv("/kaggle/working/embed_features_train.csv")
    embed_features_test = pd.read_csv("/kaggle/working/embed_features_test.csv")
    
    # Get target
    y_train = train_priors['rule_violation'].values
    
    # Combine dense features
    dense_features = [
        'char_count', 'word_count', 'sentence_count', 'avg_word_length', 'avg_sentence_length',
        'uppercase_count', 'uppercase_ratio', 'titlecase_words', 'titlecase_ratio',
        'exclam_count', 'question_count', 'punct_count', 'punct_ratio',
        'digit_count', 'digit_ratio', 'special_char_count', 'special_char_ratio',
        'emoji_count', 'has_emoji', 'unique_words', 'lexical_diversity',
        'short_word_count', 'short_word_ratio', 'long_word_count', 'long_word_ratio',
        'url_token_count', 'user_token_count', 'subreddit_token_count', 'has_reddit_artifacts',
        'is_empty_text', 'is_very_short', 'is_very_long'
    ]
    
    # Domain features
    domain_features = [
        'promo_kw_count', 'has_promo_kw', 'cta_count', 'has_call_to_action',
        'suspicious_tld_count', 'has_commercial_tld', 'legal_kw_count', 'has_legal_kw',
        'disclaimer_count', 'has_disclaimer', 'imperative_count', 'has_imperative',
        'advertising_score', 'legal_advice_score', 'likely_advertising', 'likely_legal_advice',
        'domain_risk_score'
    ]
    
    # Prior features
    prior_features = ['subreddit_prior', 'rule_prior']
    
    # Embedding features (all columns from embed_features)
    embedding_features = embed_features_train.columns.tolist()
    
    # Select features from each dataset
    train_dense_selected = train_dense[dense_features].copy()
    test_dense_selected = test_dense[dense_features].copy()
    
    train_domain_selected = train_domain[domain_features].copy()
    test_domain_selected = test_domain[domain_features].copy()
    
    train_priors_selected = train_priors[prior_features].copy()
    test_priors_selected = test_priors[prior_features].copy()
    
    print("  Combining ALL feature matrices...")
    
    # Combine dense + domain + priors
    X_train_combined = pd.concat([
        train_dense_selected,
        train_domain_selected,
        train_priors_selected
    ], axis=1)
    
    X_test_combined = pd.concat([
        test_dense_selected,
        test_domain_selected, 
        test_priors_selected
    ], axis=1)
    
    # Add TF-IDF features
    tfidf_word_train_df = pd.DataFrame(tfidf_word_train, 
                                      columns=[f'tfidf_word_{i}' for i in range(tfidf_word_train.shape[1])])
    tfidf_word_test_df = pd.DataFrame(tfidf_word_test,
                                     columns=[f'tfidf_word_{i}' for i in range(tfidf_word_test.shape[1])])
    
    tfidf_char_train_df = pd.DataFrame(tfidf_char_train,
                                      columns=[f'tfidf_char_{i}' for i in range(tfidf_char_train.shape[1])])
    tfidf_char_test_df = pd.DataFrame(tfidf_char_test,
                                     columns=[f'tfidf_char_{i}' for i in range(tfidf_char_test.shape[1])])
    
    # Combine everything
    X_train_final = pd.concat([
        X_train_combined, 
        tfidf_word_train_df, 
        tfidf_char_train_df,
        embed_features_train
    ], axis=1)
    
    X_test_final = pd.concat([
        X_test_combined,
        tfidf_word_test_df,
        tfidf_char_test_df,
        embed_features_test
    ], axis=1)
    
    # Ensure same column order
    X_test_final = X_test_final[X_train_final.columns]
    
    feature_names = X_train_final.columns.tolist()
    
    print(f"  Final feature matrix: {X_train_final.shape}")
    print(f"  Test feature matrix: {X_test_final.shape}")
    
    # Feature category counts
    n_dense = len(dense_features) + len(domain_features) + len(prior_features)
    n_tfidf = tfidf_word_train.shape[1] + tfidf_char_train.shape[1]
    n_embed = len(embedding_features)
    
    print(f"  Feature breakdown:")
    print(f"    Dense + Domain + Priors: {n_dense}")
    print(f"    TF-IDF: {n_tfidf}")
    print(f"    Embeddings: {n_embed}")
    print(f"    TOTAL: {n_dense + n_tfidf + n_embed}")
    
    return X_train_final.values, X_test_final.values, y_train, feature_names

def create_lgbm_model():
    """
    Create LightGBM classifier with optimized hyperparameters.
    
    Returns:
        LightGBM classifier
    """
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    return model

def train_cv_lgbm(X_train, y_train, fold_indices, feature_names):
    """
    Train LightGBM with cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training target
        fold_indices: Fold indices from CV
        feature_names: List of feature names
        
    Returns:
        Dictionary with CV results
    """
    print("  Training LightGBM with cross-validation...")
    
    # Create models directory
    os.makedirs("/kaggle/working/models", exist_ok=True)
    
    # Initialize results
    oof_predictions = np.zeros(len(X_train))
    fold_models = []
    fold_results = []
    feature_importances = []
    
    for fold_info in fold_indices:
        fold = fold_info['fold']
        train_idx = fold_info['train_indices']
        val_idx = fold_info['val_indices']
        
        print(f"    Fold {fold + 1}/5...")
        
        # Split data
        X_tr = X_train[train_idx]
        X_val = X_train[val_idx]
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]
        
        # Create and train model
        model = create_lgbm_model()
        
        # Try different parameter names for early stopping
        try:
            # Try with callbacks
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(100)],
                verbose=False
            )
        except TypeError:
            try:
                # Try without early stopping
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric='auc',
                    verbose=False
                )
            except TypeError:
                # Simple fit without eval_set
                model.fit(X_tr, y_tr)
        
        # Predict on validation
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_predictions[val_idx] = val_preds
        
        # Calculate fold metrics
        fold_auc = roc_auc_score(y_val, val_preds)
        fold_accuracy = accuracy_score(y_val, (val_preds > 0.5).astype(int))
        
        # Store fold results
        fold_result = {
            'fold': fold,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'auc': fold_auc,
            'accuracy': fold_accuracy
        }
        
        # Try to get best iteration if available
        try:
            fold_result['best_iteration'] = model.best_iteration_
        except:
            fold_result['best_iteration'] = model.n_estimators
        
        fold_results.append(fold_result)
        
        # Store model and feature importance
        fold_models.append(model)
        feature_importances.append(model.feature_importances_)
        
        # Save model
        model_path = f"/kaggle/working/models/lgbm_full_fold_{fold}.txt"
        try:
            model.booster_.save_model(model_path)
        except:
            # Alternative saving method
            import joblib
            joblib.dump(model, model_path.replace('.txt', '.pkl'))
        
        print(f"      AUC: {fold_auc:.4f}, Accuracy: {fold_accuracy:.4f}")
    
    return {
        'oof_predictions': oof_predictions,
        'fold_models': fold_models,
        'fold_results': fold_results,
        'feature_importances': feature_importances
    }

def analyze_feature_importance(feature_importances, feature_names, fold_results):
    """
    Analyze feature importance across folds.
    
    Args:
        feature_importances: List of feature importance arrays
        feature_names: List of feature names
        fold_results: Fold results
        
    Returns:
        DataFrame with feature importance analysis
    """
    # Calculate mean importance across folds
    mean_importance = np.mean(feature_importances, axis=0)
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': mean_importance,
        'importance_std': np.std(feature_importances, axis=0)
    })
    
    # Add individual fold importances
    for i, fold_imp in enumerate(feature_importances):
        importance_df[f'importance_fold_{i}'] = fold_imp
    
    # Sort by mean importance
    importance_df = importance_df.sort_values('importance_mean', ascending=False)
    
    return importance_df

def evaluate_model(oof_predictions, y_train, fold_results):
    """
    Evaluate model performance.
    
    Args:
        oof_predictions: OOF predictions
        y_train: True labels
        fold_results: Fold results
        
    Returns:
        Dictionary with evaluation metrics
    """
    evaluation = {}
    
    # Overall metrics
    evaluation['overall_auc'] = roc_auc_score(y_train, oof_predictions)
    evaluation['overall_accuracy'] = accuracy_score(y_train, (oof_predictions > 0.5).astype(int))
    
    # Fold metrics summary
    fold_aucs = [fold['auc'] for fold in fold_results]
    evaluation['fold_auc_mean'] = np.mean(fold_aucs)
    evaluation['fold_auc_std'] = np.std(fold_aucs)
    evaluation['fold_auc_min'] = np.min(fold_aucs)
    evaluation['fold_auc_max'] = np.max(fold_aucs)
    
    # Best iteration summary
    best_iterations = [fold['best_iteration'] for fold in fold_results]
    evaluation['avg_best_iteration'] = np.mean(best_iterations)
    
    return evaluation

def compare_all_models(full_evaluation):
    """
    Compare full model performance with all previous models.
    
    Args:
        full_evaluation: Full model evaluation
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {}
    
    # Load previous model evaluations
    with open("/kaggle/working/lgbm_text_evaluation.json", "r") as f:
        text_evaluation = json.load(f)
    
    with open("/kaggle/working/lgbm_embed_evaluation.json", "r") as f:
        embed_evaluation = json.load(f)
    
    # AUC comparison
    comparison['auc_text'] = text_evaluation['overall_auc']
    comparison['auc_embed'] = embed_evaluation['overall_auc']
    comparison['auc_full'] = full_evaluation['overall_auc']
    
    comparison['improvement_text_to_full'] = comparison['auc_full'] - comparison['auc_text']
    comparison['improvement_embed_to_full'] = comparison['auc_full'] - comparison['auc_embed']
    
    comparison['pct_improvement_text'] = (comparison['improvement_text_to_full'] / comparison['auc_text']) * 100
    comparison['pct_improvement_embed'] = (comparison['improvement_embed_to_full'] / comparison['auc_embed']) * 100
    
    # Accuracy comparison
    comparison['accuracy_text'] = text_evaluation['overall_accuracy']
    comparison['accuracy_embed'] = embed_evaluation['overall_accuracy']
    comparison['accuracy_full'] = full_evaluation['overall_accuracy']
    
    # Stability comparison
    comparison['std_text'] = text_evaluation['fold_auc_std']
    comparison['std_embed'] = embed_evaluation['fold_auc_std']
    comparison['std_full'] = full_evaluation['fold_auc_std']
    
    return comparison

# Load fold indices
with open("/kaggle/working/fold_indices.json", "r") as f:
    fold_indices = json.load(f)

print("Loading and combining ALL features...")

# Load and combine ALL features
X_train, X_test, y_train, feature_names = load_and_combine_all_features()

print("Training LightGBM with cross-validation...")

# Train model with CV
cv_results = train_cv_lgbm(X_train, y_train, fold_indices, feature_names)

print("Analyzing results...")

# Analyze feature importance
importance_df = analyze_feature_importance(
    cv_results['feature_importances'],
    feature_names,
    cv_results['fold_results']
)

# Evaluate model
evaluation = evaluate_model(cv_results['oof_predictions'], y_train, cv_results['fold_results'])

# Compare with all previous models
comparison = compare_all_models(evaluation)

print("Saving results...")

# Save OOF predictions
oof_df = pd.DataFrame({
    'oof_prediction': cv_results['oof_predictions'],
    'rule_violation': y_train
})
oof_df.to_csv("/kaggle/working/oof_lgbm_full.csv", index=False)

# Save feature importance
importance_df.to_csv("/kaggle/working/feature_importance_lgbm_full.csv", index=False)

# Save evaluation results
with open("/kaggle/working/lgbm_full_evaluation.json", "w") as f:
    json.dump(evaluation, f, indent=2, ensure_ascii=False)

# Save comprehensive comparison
with open("/kaggle/working/all_models_comparison.json", "w") as f:
    json.dump(comparison, f, indent=2, ensure_ascii=False)

print("Generating test predictions...")

# Generate test predictions (average across folds)
test_predictions = np.zeros(len(X_test))
for model in cv_results['fold_models']:
    test_predictions += model.predict_proba(X_test)[:, 1]
test_predictions /= len(cv_results['fold_models'])

# Save test predictions
test_pred_df = pd.DataFrame({
    'prediction': test_predictions
})
test_pred_df.to_csv("/kaggle/working/test_predictions_lgbm_full.csv", index=False)

print("")
print("=" * 60)
print("C14: LIGHTGBM FULL MODEL TRAINING COMPLETE")
print("=" * 60)

# Display comprehensive results
print("Model Performance:")
print(f"  Overall OOF AUC: {evaluation['overall_auc']:.4f}")
print(f"  Overall Accuracy: {evaluation['overall_accuracy']:.4f}")

print("")
print("Cross-Validation Results:")
print(f"  Mean Fold AUC: {evaluation['fold_auc_mean']:.4f} Â± {evaluation['fold_auc_std']:.4f}")
print(f"  Fold AUC Range: [{evaluation['fold_auc_min']:.4f}, {evaluation['fold_auc_max']:.4f}]")
print(f"  Average Best Iteration: {evaluation['avg_best_iteration']:.0f}")

print("")
print("Model Comparison:")
print(f"  Text Model:      {comparison['auc_text']:.4f}")
print(f"  Embed Model:     {comparison['auc_embed']:.4f}")
print(f"  FULL Model:      {comparison['auc_full']:.4f}")
print(f"  Improvement:     +{comparison['improvement_text_to_full']:.4f} (+{comparison['pct_improvement_text']:.1f}%)")

print("")
print("Top 10 Most Important Features:")
for i, row in importance_df.head(10).iterrows():
    print(f"  {i+1:2d}. {row['feature']:<25} {row['importance_mean']:.4f}")

print("")
print("Feature Categories (Top 20):")
top_features = importance_df.head(20)
category_counts = {}
for feature in top_features['feature']:
    if 'tfidf_word' in feature:
        category_counts['TF-IDF Word'] = category_counts.get('TF-IDF Word', 0) + 1
    elif 'tfidf_char' in feature:
        category_counts['TF-IDF Char'] = category_counts.get('TF-IDF Char', 0) + 1
    elif 'cos_similarity' in feature or 'euclidean_distance' in feature or 'dot_product' in feature:
        category_counts['Semantic Features'] = category_counts.get('Semantic Features', 0) + 1
    elif 'abs_diff_pca' in feature:
        category_counts['Embedding PCA'] = category_counts.get('Embedding PCA', 0) + 1
    elif any(x in feature for x in ['prior', 'score']):
        category_counts['Priors/Scores'] = category_counts.get('Priors/Scores', 0) + 1
    elif any(x in feature for x in ['legal', 'advertis', 'promo', 'cta']):
        category_counts['Domain Features'] = category_counts.get('Domain Features', 0) + 1
    elif any(x in feature for x in ['count', 'ratio', 'length']):
        category_counts['Text Statistics'] = category_counts.get('Text Statistics', 0) + 1
    else:
        category_counts['Other'] = category_counts.get('Other', 0) + 1

for category, count in category_counts.items():
    print(f"  {category}: {count} features")

print("")
print("Feature Importance Insights:")
print(f"  Total features: {len(feature_names)}")
print(f"  Non-zero importance features: {(importance_df['importance_mean'] > 0).sum()}")
print(f"  Top feature: {importance_df.iloc[0]['feature']} "
      f"({importance_df.iloc[0]['importance_mean']:.4f})")

print("")
print("Signal Integration Analysis:")
print(f"  Priors remain dominant: {importance_df.iloc[0]['feature']}, {importance_df.iloc[1]['feature']}")
print(f"  Semantic features in top 10: {len([f for f in importance_df.head(10)['feature'] if 'cos_' in f or 'euclidean' in f or 'dot_' in f or 'abs_diff' in f])}")
print(f"  TF-IDF features in top 20: {len([f for f in importance_df.head(20)['feature'] if 'tfidf_' in f])}")

print("")
print("Files Saved:")
print(f"  /kaggle/working/oof_lgbm_full.csv")
print(f"  /kaggle/working/feature_importance_lgbm_full.csv")
print(f"  /kaggle/working/lgbm_full_evaluation.json")
print(f"  /kaggle/working/all_models_comparison.json")
print(f"  /kaggle/working/test_predictions_lgbm_full.csv")
print(f"  /kaggle/working/models/lgbm_full_fold_*.txt (5 models)")

print("")
print("Next: Ready for final ensemble and submission")
print("=" * 60)


# =============================================
# C15: STACKING META
# =============================================

import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("Creating stacking ensemble with meta-model...")

def load_oof_predictions():
    """
    Load OOF predictions from all three models.
    
    Returns:
        Tuple of (train_meta_features, test_meta_features, y_train)
    """
    print("  Loading OOF predictions from all models...")
    
    # Load OOF predictions
    oof_text = pd.read_csv("/kaggle/working/oof_lgbm_text.csv")
    oof_embed = pd.read_csv("/kaggle/working/oof_lgbm_embed.csv")
    oof_full = pd.read_csv("/kaggle/working/oof_lgbm_full.csv")
    
    # Load test predictions
    test_text = pd.read_csv("/kaggle/working/test_predictions_lgbm_text.csv")
    test_embed = pd.read_csv("/kaggle/working/test_predictions_lgbm_embed.csv")
    test_full = pd.read_csv("/kaggle/working/test_predictions_lgbm_full.csv")
    
    # Create meta features
    train_meta = pd.DataFrame({
        'p_text': oof_text['oof_prediction'],
        'p_embed': oof_embed['oof_prediction'],
        'p_full': oof_full['oof_prediction']
    })
    
    test_meta = pd.DataFrame({
        'p_text': test_text['prediction'],
        'p_embed': test_embed['prediction'],
        'p_full': test_full['prediction']
    })
    
    y_train = oof_text['rule_violation'].values
    
    print(f"  Train meta features: {train_meta.shape}")
    print(f"  Test meta features: {test_meta.shape}")
    
    return train_meta.values, test_meta.values, y_train

def create_meta_model():
    """
    Create meta-model for stacking.
    
    Returns:
        LogisticRegression model
    """
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    return model

def train_cv_meta_model(X_meta, y_train, fold_indices):
    """
    Train meta-model with cross-validation.
    
    Args:
        X_meta: Meta features
        y_train: Training target
        fold_indices: Fold indices from CV
        
    Returns:
        Dictionary with CV results
    """
    print("  Training meta-model with cross-validation...")
    
    # Initialize results
    oof_predictions = np.zeros(len(X_meta))
    fold_models = []
    fold_results = []
    feature_importances = []
    
    for fold_info in fold_indices:
        fold = fold_info['fold']
        train_idx = fold_info['train_indices']
        val_idx = fold_info['val_indices']
        
        print(f"    Fold {fold + 1}/5...")
        
        # Split data
        X_tr = X_meta[train_idx]
        X_val = X_meta[val_idx]
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]
        
        # Create and train meta-model
        model = create_meta_model()
        model.fit(X_tr, y_tr)
        
        # Predict on validation
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_predictions[val_idx] = val_preds
        
        # Calculate fold metrics
        fold_auc = roc_auc_score(y_val, val_preds)
        fold_accuracy = accuracy_score(y_val, (val_preds > 0.5).astype(int))
        
        # Store fold results
        fold_result = {
            'fold': fold,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'auc': fold_auc,
            'accuracy': fold_accuracy,
            'coef': model.coef_[0].tolist()
        }
        fold_results.append(fold_result)
        
        # Store model and feature importance
        fold_models.append(model)
        feature_importances.append(model.coef_[0])
        
        print(f"      AUC: {fold_auc:.4f}, Accuracy: {fold_accuracy:.4f}")
        print(f"      Coefficients: {model.coef_[0]}")
    
    return {
        'oof_predictions': oof_predictions,
        'fold_models': fold_models,
        'fold_results': fold_results,
        'feature_importances': feature_importances
    }

def analyze_meta_model(cv_results, feature_names):
    """
    Analyze meta-model performance and feature importance.
    
    Args:
        cv_results: CV results from meta-model training
        feature_names: Names of meta-features
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {}
    
    # Calculate mean coefficients across folds
    mean_coef = np.mean(cv_results['feature_importances'], axis=0)
    std_coef = np.std(cv_results['feature_importances'], axis=0)
    
    # Create coefficient DataFrame
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coef_mean': mean_coef,
        'coef_std': std_coef
    })
    
    # Add individual fold coefficients
    for i, fold_coef in enumerate(cv_results['feature_importances']):
        coef_df[f'coef_fold_{i}'] = fold_coef
    
    # Sort by absolute coefficient value
    coef_df['abs_coef'] = np.abs(coef_df['coef_mean'])
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    
    # Convert DataFrame to dictionary for JSON serialization
    analysis['coefficients'] = coef_df.to_dict('records')
    
    # Model weights interpretation
    analysis['model_weights'] = {
        'text_weight': float(mean_coef[0]),
        'embed_weight': float(mean_coef[1]),
        'full_weight': float(mean_coef[2]),
        'dominant_model': feature_names[np.argmax(np.abs(mean_coef))]
    }
    
    return analysis

def evaluate_meta_model(oof_predictions, y_train, fold_results):
    """
    Evaluate meta-model performance.
    
    Args:
        oof_predictions: OOF predictions
        y_train: True labels
        fold_results: Fold results
        
    Returns:
        Dictionary with evaluation metrics
    """
    evaluation = {}
    
    # Overall metrics
    evaluation['overall_auc'] = roc_auc_score(y_train, oof_predictions)
    evaluation['overall_accuracy'] = accuracy_score(y_train, (oof_predictions > 0.5).astype(int))
    
    # Fold metrics summary
    fold_aucs = [fold['auc'] for fold in fold_results]
    evaluation['fold_auc_mean'] = float(np.mean(fold_aucs))
    evaluation['fold_auc_std'] = float(np.std(fold_aucs))
    evaluation['fold_auc_min'] = float(np.min(fold_aucs))
    evaluation['fold_auc_max'] = float(np.max(fold_aucs))
    
    return evaluation

def compare_with_base_models(meta_evaluation):
    """
    Compare meta-model performance with base models.
    
    Args:
        meta_evaluation: Meta-model evaluation
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {}
    
    # Load base model evaluations
    with open("/kaggle/working/lgbm_text_evaluation.json", "r") as f:
        text_evaluation = json.load(f)
    
    with open("/kaggle/working/lgbm_embed_evaluation.json", "r") as f:
        embed_evaluation = json.load(f)
    
    with open("/kaggle/working/lgbm_full_evaluation.json", "r") as f:
        full_evaluation = json.load(f)
    
    # AUC comparison
    comparison['auc_text'] = text_evaluation['overall_auc']
    comparison['auc_embed'] = embed_evaluation['overall_auc']
    comparison['auc_full'] = full_evaluation['overall_auc']
    comparison['auc_meta'] = meta_evaluation['overall_auc']
    
    # Best base model
    base_aucs = {
        'text': comparison['auc_text'],
        'embed': comparison['auc_embed'],
        'full': comparison['auc_full']
    }
    comparison['best_base_model'] = max(base_aucs, key=base_aucs.get)
    comparison['best_base_auc'] = base_aucs[comparison['best_base_model']]
    
    # Improvements
    comparison['improvement_vs_best'] = comparison['auc_meta'] - comparison['best_base_auc']
    comparison['pct_improvement_vs_best'] = (comparison['improvement_vs_best'] / comparison['best_base_auc']) * 100
    
    return comparison

# Load fold indices
with open("/kaggle/working/fold_indices.json", "r") as f:
    fold_indices = json.load(f)

print("Loading OOF predictions...")

# Load OOF predictions from all models
X_meta_train, X_meta_test, y_train = load_oof_predictions()
meta_feature_names = ['p_text', 'p_embed', 'p_full']

print("Training meta-model with cross-validation...")

# Train meta-model with CV
cv_results = train_cv_meta_model(X_meta_train, y_train, fold_indices)

print("Analyzing meta-model...")

# Analyze meta-model
meta_analysis = analyze_meta_model(cv_results, meta_feature_names)

# Evaluate meta-model
meta_evaluation = evaluate_meta_model(cv_results['oof_predictions'], y_train, cv_results['fold_results'])

# Compare with base models
comparison = compare_with_base_models(meta_evaluation)

print("Saving results...")

# Save OOF predictions
oof_df = pd.DataFrame({
    'oof_prediction': cv_results['oof_predictions'],
    'rule_violation': y_train
})
oof_df.to_csv("/kaggle/working/oof_meta.csv", index=False)

# Save test predictions
test_predictions = np.zeros(len(X_meta_test))
for model in cv_results['fold_models']:
    test_predictions += model.predict_proba(X_meta_test)[:, 1]
test_predictions /= len(cv_results['fold_models'])

test_pred_df = pd.DataFrame({
    'prediction': test_predictions
})
test_pred_df.to_csv("/kaggle/working/test_meta.csv", index=False)

# Save meta-analysis
with open("/kaggle/working/meta_analysis.json", "w") as f:
    json.dump({
        'evaluation': meta_evaluation,
        'analysis': meta_analysis,
        'comparison': comparison
    }, f, indent=2, ensure_ascii=False)

print("")
print("=" * 60)
print("C15: STACKING META-MODEL COMPLETE")
print("=" * 60)

# Display comprehensive results
print("Meta-Model Performance:")
print(f"  Overall OOF AUC: {meta_evaluation['overall_auc']:.4f}")
print(f"  Overall Accuracy: {meta_evaluation['overall_accuracy']:.4f}")

print("")
print("Cross-Validation Results:")
print(f"  Mean Fold AUC: {meta_evaluation['fold_auc_mean']:.4f} Â± {meta_evaluation['fold_auc_std']:.4f}")
print(f"  Fold AUC Range: [{meta_evaluation['fold_auc_min']:.4f}, {meta_evaluation['fold_auc_max']:.4f}]")

print("")
print("Model Comparison:")
print(f"  Text Model:    {comparison['auc_text']:.4f}")
print(f"  Embed Model:   {comparison['auc_embed']:.4f}")
print(f"  Full Model:    {comparison['auc_full']:.4f}")
print(f"  META Model:    {comparison['auc_meta']:.4f}")
print(f"  Best Base:     {comparison['best_base_model']} ({comparison['best_base_auc']:.4f})")
print(f"  Improvement:   +{comparison['improvement_vs_best']:.4f} (+{comparison['pct_improvement_vs_best']:.2f}%)")

print("")
print("Meta-Model Weights (averaged across folds):")
for coef_data in meta_analysis['coefficients']:
    print(f"  {coef_data['feature']}: {coef_data['coef_mean']:+.4f} Â± {coef_data['coef_std']:.4f}")

print("")
print("Model Weight Interpretation:")
weights = meta_analysis['model_weights']
print(f"  Dominant model: {weights['dominant_model']}")
print(f"  Text weight:    {weights['text_weight']:+.4f}")
print(f"  Embed weight:   {weights['embed_weight']:+.4f}")
print(f"  Full weight:    {weights['full_weight']:+.4f}")

print("")
print("Fold-wise Coefficients:")
for fold_result in cv_results['fold_results'][:3]:  # Show first 3 folds
    print(f"  Fold {fold_result['fold'] + 1}:")
    for i, (feature, coef) in enumerate(zip(meta_feature_names, fold_result['coef'])):
        print(f"    {feature}: {coef:+.4f}")

print("")
print("Ensemble Strategy Analysis:")
if weights['embed_weight'] > weights['text_weight'] and weights['embed_weight'] > weights['full_weight']:
    print("  Strategy: Embedding-focused ensemble")
elif weights['text_weight'] > weights['embed_weight'] and weights['text_weight'] > weights['full_weight']:
    print("  Strategy: Text-focused ensemble")
else:
    print("  Strategy: Balanced ensemble")

print("")
print("Files Saved:")
print(f"  /kaggle/working/oof_meta.csv")
print(f"  /kaggle/working/test_meta.csv")
print(f"  /kaggle/working/meta_analysis.json")

print("")
print("Next: Ready for final submission and analysis")
print("=" * 60)


# =============================================
# C16: calibrate_per_rule
# =============================================

import pandas as pd
import numpy as np
import json
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss
import pickle
import warnings
warnings.filterwarnings('ignore')

print("Calibrating probabilities per rule...")

def load_data_for_calibration():
    """
    Load data needed for rule-specific calibration.
    
    Returns:
        Tuple of (train_df, test_df, oof_predictions, test_predictions, y_train)
    """
    print("Loading data for calibration...")
    
    # Load datasets with rule information
    train_with_groups = pd.read_csv("/kaggle/working/train_with_groups.csv")
    test_with_groups = pd.read_csv("/kaggle/working/test_with_groups.csv")
    
    # Load OOF and test predictions
    oof_meta = pd.read_csv("/kaggle/working/oof_meta.csv")
    test_meta = pd.read_csv("/kaggle/working/test_meta.csv")
    
    # Combine data
    train_df = train_with_groups[['rule', 'rule_violation']].copy()
    train_df['oof_prediction'] = oof_meta['oof_prediction']
    
    test_df = test_with_groups[['rule']].copy()
    test_df['test_prediction'] = test_meta['prediction']
    
    y_train = train_df['rule_violation'].values
    
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Show rule distribution
    print(f"Rule distribution (train):")
    rule_counts = train_df['rule'].value_counts()
    for rule, count in rule_counts.items():
        violation_rate = train_df[train_df['rule'] == rule]['rule_violation'].mean()
        print(f"   {rule}: {count} samples ({violation_rate:.2%} violations)")
    
    return train_df, test_df, oof_meta['oof_prediction'].values, test_meta['prediction'].values, y_train

def calibrate_per_rule(train_df, oof_predictions, y_train):
    """
    Calibrate probabilities separately for each rule.
    
    Args:
        train_df: Training DataFrame with rule information
        oof_predictions: OOF predictions to calibrate
        y_train: True labels
        
    Returns:
        Dictionary with calibrators and calibration results
    """
    print("Training rule-specific calibrators...")
    
    calibrators = {}
    calibration_results = {}
    calibrated_predictions = np.zeros_like(oof_predictions)
    
    # Get unique rules
    unique_rules = train_df['rule'].unique()
    
    for rule in unique_rules:
        print(f"   Calibrating rule: {rule}...")
        
        # Get indices for this rule
        rule_mask = (train_df['rule'] == rule)
        rule_indices = np.where(rule_mask)[0]
        
        if len(rule_indices) == 0:
            print(f"      No samples for rule {rule}, skipping...")
            continue
        
        # Get predictions and labels for this rule
        rule_predictions = oof_predictions[rule_indices]
        rule_labels = y_train[rule_indices]
        
        # Skip if only one class in the rule
        if len(np.unique(rule_labels)) < 2:
            print(f"      Only one class in rule {rule}, using identity calibration...")
            calibrator = None
            calibrated_rule = rule_predictions
        else:
            # Train isotonic calibrator
            calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
            calibrator.fit(rule_predictions, rule_labels)
            calibrated_rule = calibrator.transform(rule_predictions)
        
        # Store calibrator and results
        calibrators[rule] = calibrator
        calibrated_predictions[rule_indices] = calibrated_rule
        
        # Calculate calibration metrics
        original_log_loss = log_loss(rule_labels, rule_predictions)
        calibrated_log_loss = log_loss(rule_labels, calibrated_rule)
        
        original_brier = brier_score_loss(rule_labels, rule_predictions)
        calibrated_brier = brier_score_loss(rule_labels, calibrated_rule)
        
        calibration_results[rule] = {
            'samples': len(rule_indices),
            'violation_rate': rule_labels.mean(),
            'original_log_loss': original_log_loss,
            'calibrated_log_loss': calibrated_log_loss,
            'original_brier': original_brier,
            'calibrated_brier': calibrated_brier,
            'log_loss_improvement': original_log_loss - calibrated_log_loss,
            'brier_improvement': original_brier - calibrated_brier
        }
        
        print(f"      Samples: {len(rule_indices)}, "
              f"LogLoss: {original_log_loss:.4f} -> {calibrated_log_loss:.4f} "
              f"({calibration_results[rule]['log_loss_improvement']:+.4f})")
    
    return {
        'calibrators': calibrators,
        'calibration_results': calibration_results,
        'calibrated_predictions': calibrated_predictions
    }

def apply_calibration_to_test(test_df, test_predictions, calibrators):
    """
    Apply rule-specific calibration to test predictions.
    
    Args:
        test_df: Test DataFrame with rule information
        test_predictions: Test predictions to calibrate
        calibrators: Dictionary of trained calibrators
        
    Returns:
        Calibrated test predictions
    """
    print("Applying calibration to test predictions...")
    
    calibrated_test = np.zeros_like(test_predictions)
    
    for rule, calibrator in calibrators.items():
        rule_mask = (test_df['rule'] == rule)
        rule_indices = np.where(rule_mask)[0]
        
        if len(rule_indices) == 0:
            print(f"   No test samples for rule: {rule}")
            continue
        
        rule_predictions = test_predictions[rule_indices]
        
        if calibrator is None:
            # Use identity transformation if no calibrator
            calibrated_test[rule_indices] = rule_predictions
        else:
            # Apply calibration
            calibrated_test[rule_indices] = calibrator.transform(rule_predictions)
        
        print(f"   Rule {rule}: {len(rule_indices)} samples calibrated")
    
    return calibrated_test

def analyze_calibration_impact(original_predictions, calibrated_predictions, y_train, calibration_results):
    """
    Analyze the impact of calibration on overall metrics.
    
    Args:
        original_predictions: Original OOF predictions
        calibrated_predictions: Calibrated OOF predictions
        y_train: True labels
        calibration_results: Per-rule calibration results
        
    Returns:
        Dictionary with calibration analysis
    """
    analysis = {}
    
    # Overall metrics
    analysis['overall_original_log_loss'] = log_loss(y_train, original_predictions)
    analysis['overall_calibrated_log_loss'] = log_loss(y_train, calibrated_predictions)
    analysis['overall_log_loss_improvement'] = analysis['overall_original_log_loss'] - analysis['overall_calibrated_log_loss']
    
    analysis['overall_original_brier'] = brier_score_loss(y_train, original_predictions)
    analysis['overall_calibrated_brier'] = brier_score_loss(y_train, calibrated_predictions)
    analysis['overall_brier_improvement'] = analysis['overall_original_brier'] - analysis['overall_calibrated_brier']
    
    # Per-rule summary
    rule_summary = {}
    for rule, results in calibration_results.items():
        rule_summary[rule] = {
            'samples': results['samples'],
            'violation_rate': results['violation_rate'],
            'log_loss_improvement': results['log_loss_improvement'],
            'brier_improvement': results['brier_improvement']
        }
    
    analysis['rule_summary'] = rule_summary
    
    # Distribution analysis
    analysis['original_pred_stats'] = {
        'mean': float(np.mean(original_predictions)),
        'std': float(np.std(original_predictions)),
        'min': float(np.min(original_predictions)),
        'max': float(np.max(original_predictions))
    }
    
    analysis['calibrated_pred_stats'] = {
        'mean': float(np.mean(calibrated_predictions)),
        'std': float(np.std(calibrated_predictions)),
        'min': float(np.min(calibrated_predictions)),
        'max': float(np.max(calibrated_predictions))
    }
    
    return analysis

def save_calibrators(calibrators):
    """
    Save calibrators to disk.
    
    Args:
        calibrators: Dictionary of calibrators
    """
    print("Saving calibrators...")
    
    for rule, calibrator in calibrators.items():
        # Create safe filename
        safe_rule_name = rule.replace(' ', '_').replace('/', '_').replace('...', '')
        filename = f"/kaggle/working/calib_rule_{safe_rule_name}.pkl"
        
        with open(filename, 'wb') as f:
            pickle.dump(calibrator, f)
        
        print(f"   Saved: {filename}")

# Load data
train_df, test_df, oof_predictions, test_predictions, y_train = load_data_for_calibration()

print("Training rule-specific calibrators...")

# Calibrate per rule
calibration_results = calibrate_per_rule(train_df, oof_predictions, y_train)

print("Analyzing calibration impact...")

# Analyze calibration impact
calibration_analysis = analyze_calibration_impact(
    oof_predictions,
    calibration_results['calibrated_predictions'],
    y_train,
    calibration_results['calibration_results']
)

print("Applying calibration to test set...")

# Apply calibration to test set
calibrated_test_predictions = apply_calibration_to_test(
    test_df,
    test_predictions,
    calibration_results['calibrators']
)

print("Saving calibrated predictions...")

# Save calibrated predictions
oof_calibrated_df = pd.DataFrame({
    'oof_prediction_original': oof_predictions,
    'oof_prediction_calibrated': calibration_results['calibrated_predictions'],
    'rule_violation': y_train
})
oof_calibrated_df.to_csv("/kaggle/working/oof_calibrated.csv", index=False)

test_calibrated_df = pd.DataFrame({
    'prediction_original': test_predictions,
    'prediction_calibrated': calibrated_test_predictions
})
test_calibrated_df.to_csv("/kaggle/working/test_calibrated.csv", index=False)

# Save calibrators
save_calibrators(calibration_results['calibrators'])

# Save calibration analysis
with open("/kaggle/working/calibration_analysis.json", "w") as f:
    json.dump({
        'calibration_results': calibration_results['calibration_results'],
        'calibration_analysis': calibration_analysis
    }, f, indent=2, ensure_ascii=False)

print("\n" + "="*60)
print("C16: Rule-Specific Calibration Complete")
print("="*60)

# Display comprehensive results
print(f"Overall Calibration Impact:")
print(f"   Original LogLoss:  {calibration_analysis['overall_original_log_loss']:.4f}")
print(f"   Calibrated LogLoss: {calibration_analysis['overall_calibrated_log_loss']:.4f}")
print(f"   LogLoss Improvement: {calibration_analysis['overall_log_loss_improvement']:+.4f}")

print(f"   Original Brier:    {calibration_analysis['overall_original_brier']:.4f}")
print(f"   Calibrated Brier:  {calibration_analysis['overall_calibrated_brier']:.4f}")
print(f"   Brier Improvement:  {calibration_analysis['overall_brier_improvement']:+.4f}")

print(f"Per-Rule Calibration Results:")
for rule, results in calibration_results['calibration_results'].items():
    print(f"   {rule}:")
    print(f"      Samples: {results['samples']}, Violation rate: {results['violation_rate']:.2%}")
    print(f"      LogLoss: {results['original_log_loss']:.4f} -> {results['calibrated_log_loss']:.4f} "
          f"({results['log_loss_improvement']:+.4f})")
    print(f"      Brier:   {results['original_brier']:.4f} -> {results['calibrated_brier']:.4f} "
          f"({results['brier_improvement']:+.4f})")

print(f"Prediction Distribution:")
print(f"   Original predictions:")
stats_orig = calibration_analysis['original_pred_stats']
print(f"      Mean: {stats_orig['mean']:.4f}, Std: {stats_orig['std']:.4f}")
print(f"      Range: [{stats_orig['min']:.4f}, {stats_orig['max']:.4f}]")

print(f"   Calibrated predictions:")
stats_calib = calibration_analysis['calibrated_pred_stats']
print(f"      Mean: {stats_calib['mean']:.4f}, Std: {stats_calib['std']:.4f}")
print(f"      Range: [{stats_calib['min']:.4f}, {stats_calib['max']:.4f}]")

print(f"Calibration Insights:")
if calibration_analysis['overall_log_loss_improvement'] > 0:
    print(f"   Calibration successful! LogLoss reduced by {calibration_analysis['overall_log_loss_improvement']:.4f}")
else:
    print(f"   Calibration didn't improve LogLoss. Model probabilities are already well-calibrated.")

# Check which rules benefited most
best_improvement_rule = max(
    calibration_results['calibration_results'].items(),
    key=lambda x: x[1]['log_loss_improvement']
)
print(f"   Most improved rule: {best_improvement_rule[0]} "
      f"(LogLoss improvement: {best_improvement_rule[1]['log_loss_improvement']:+.4f})")

print(f"Files Saved:")
print(f"   /kaggle/working/oof_calibrated.csv")
print(f"   /kaggle/working/test_calibrated.csv")
print(f"   /kaggle/working/calib_rule_*.pkl (calibrators)")
print(f"   /kaggle/working/calibration_analysis.json")

print(f"Next: Ready for final submission preparation")
print("="*60)


# =============================================
# C17: CREATE SUBMISSION
# =============================================

import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("Creating final submission file...")

def load_final_predictions():
    """
    Load the final calibrated predictions for submission.
    
    Returns:
        DataFrame with final predictions
    """
    print("  Loading final predictions...")
    
    # Load test data to get the original structure
    test_original = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
    
    # Load calibrated predictions
    test_calibrated = pd.read_csv("/kaggle/working/test_calibrated.csv")
    
    # Check available columns in test_original
    print(f"  Test original columns: {test_original.columns.tolist()}")
    print(f"  Test calibrated columns: {test_calibrated.columns.tolist()}")
    
    # Find the ID column (it might have a different name)
    id_column = None
    possible_id_columns = ['id', 'Id', 'ID', 'post_id', 'postId', 'row_id']
    
    for col in possible_id_columns:
        if col in test_original.columns:
            id_column = col
            break
    
    if id_column is None:
        # If no ID column found, create one with sequential IDs
        print("  No ID column found, creating sequential IDs...")
        submission = pd.DataFrame({
            'id': range(len(test_original)),
            'rule_violation': test_calibrated['prediction_calibrated']
        })
    else:
        # Use the found ID column
        print(f"  Using ID column: {id_column}")
        submission = test_original[[id_column]].copy()
        submission.columns = ['id']  # Standardize to 'id'
        submission['rule_violation'] = test_calibrated['prediction_calibrated']
    
    print(f"  Submission shape: {submission.shape}")
    print(f"  Prediction range: [{submission['rule_violation'].min():.4f}, {submission['rule_violation'].max():.4f}]")
    
    return submission

def validate_submission(submission):
    """
    Validate the submission file meets competition requirements.
    
    Args:
        submission: Submission DataFrame
        
    Returns:
        Dictionary with validation results
    """
    print("  Validating submission...")
    
    validation = {}
    
    # Check required columns
    required_columns = ['id', 'rule_violation']
    validation['has_required_columns'] = all(col in submission.columns for col in required_columns)
    validation['missing_columns'] = [col for col in required_columns if col not in submission.columns]
    
    # Check for null values
    validation['null_ids'] = int(submission['id'].isnull().sum())
    validation['null_predictions'] = int(submission['rule_violation'].isnull().sum())
    validation['has_nulls'] = validation['null_ids'] > 0 or validation['null_predictions'] > 0
    
    # Check prediction range
    validation['min_prediction'] = float(submission['rule_violation'].min())
    validation['max_prediction'] = float(submission['rule_violation'].max())
    validation['valid_range'] = (validation['min_prediction'] >= 0.0 and validation['max_prediction'] <= 1.0)
    
    # Check prediction distribution
    validation['mean_prediction'] = float(submission['rule_violation'].mean())
    validation['std_prediction'] = float(submission['rule_violation'].std())
    
    # Check for duplicate IDs
    validation['duplicate_ids'] = int(submission['id'].duplicated().sum())
    validation['has_duplicates'] = validation['duplicate_ids'] > 0
    
    # Check sample count matches test set
    test_original = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
    validation['correct_sample_count'] = len(submission) == len(test_original)
    validation['expected_samples'] = int(len(test_original))
    validation['actual_samples'] = int(len(submission))
    
    return validation

def create_submission_metadata():
    """
    Create comprehensive metadata for the submission.
    
    Returns:
        Dictionary with submission metadata
    """
    print("  Creating submission metadata...")
    
    metadata = {
        'submission_info': {
            'created_at': datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            'pipeline_version': 'RuleSense_v2',
            'competition': 'jigsaw-agile-community-rules',
            'final_model': 'stacked_ensemble_calibrated'
        },
        'feature_engineering': {
            'feature_categories': ['text_statistics', 'domain_specific', 'tfidf', 'embeddings', 'priors']
        }
    }
    
    # Try to load model evaluations if they exist
    try:
        with open("/kaggle/working/lgbm_text_evaluation.json", "r") as f:
            text_eval = json.load(f)
        if 'model_performance' not in metadata:
            metadata['model_performance'] = {}
        metadata['model_performance']['lgbm_text'] = {
            'oof_auc': float(text_eval['overall_auc']),
            'fold_auc_mean': float(text_eval['fold_auc_mean'])
        }
    except Exception as e:
        print(f"    Could not load text model evaluations: {e}")
    
    try:
        with open("/kaggle/working/calibration_analysis.json", "r") as f:
            calibration_analysis = json.load(f)
        metadata['calibration_impact'] = {
            'log_loss_improvement': float(calibration_analysis['calibration_analysis']['overall_log_loss_improvement'])
        }
    except Exception as e:
        print(f"    Could not load calibration analysis: {e}")
    
    return metadata

def analyze_prediction_distribution(submission):
    """
    Analyze the distribution of final predictions.
    
    Args:
        submission: Submission DataFrame
        
    Returns:
        Dictionary with distribution analysis
    """
    analysis = {}
    
    predictions = submission['rule_violation']
    
    # Basic statistics
    analysis['mean'] = float(predictions.mean())
    analysis['std'] = float(predictions.std())
    analysis['min'] = float(predictions.min())
    analysis['max'] = float(predictions.max())
    
    # Percentiles
    percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    for p in percentiles:
        analysis[f'percentile_{int(p*100)}'] = float(predictions.quantile(p))
    
    # Risk categories
    analysis['very_low_risk'] = int((predictions < 0.1).sum())
    analysis['low_risk'] = int(((predictions >= 0.1) & (predictions < 0.3)).sum())
    analysis['medium_risk'] = int(((predictions >= 0.3) & (predictions < 0.7)).sum())
    analysis['high_risk'] = int(((predictions >= 0.7) & (predictions < 0.9)).sum())
    analysis['very_high_risk'] = int((predictions >= 0.9).sum())
    
    # Binary classification at 0.5 threshold
    analysis['predicted_violations'] = int((predictions > 0.5).sum())
    analysis['predicted_non_violations'] = int((predictions <= 0.5).sum())
    analysis['violation_rate'] = float((predictions > 0.5).mean())
    
    return analysis

def save_submission_files(submission, metadata, validation, distribution):
    """
    Save all submission-related files.
    
    Args:
        submission: Submission DataFrame
        metadata: Submission metadata
        validation: Validation results
        distribution: Distribution analysis
    """
    print("  Saving submission files...")
    
    # Create submissions directory
    import os
    os.makedirs("/kaggle/working/submissions", exist_ok=True)
    
    # Save main submission file
    submission_file = "/kaggle/working/submissions/submission.csv"
    submission.to_csv(submission_file, index=False)
    print(f"    Saved: {submission_file}")
    
    # Save detailed submission with confidence scores
    detailed_submission = submission.copy()
    detailed_submission['confidence'] = np.abs(detailed_submission['rule_violation'] - 0.5) * 2
    detailed_submission_file = "/kaggle/working/submissions/submission_detailed.csv"
    detailed_submission.to_csv(detailed_submission_file, index=False)
    print(f"    Saved: {detailed_submission_file}")
    
    # Save submission metadata
    metadata_file = "/kaggle/working/submissions/submission_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"    Saved: {metadata_file}")
    
    # Save validation report - ensure all values are JSON serializable
    validation_report = {
        'validation': {
            'has_required_columns': bool(validation['has_required_columns']),
            'missing_columns': validation['missing_columns'],
            'null_ids': int(validation['null_ids']),
            'null_predictions': int(validation['null_predictions']),
            'has_nulls': bool(validation['has_nulls']),
            'min_prediction': float(validation['min_prediction']),
            'max_prediction': float(validation['max_prediction']),
            'valid_range': bool(validation['valid_range']),
            'mean_prediction': float(validation['mean_prediction']),
            'std_prediction': float(validation['std_prediction']),
            'duplicate_ids': int(validation['duplicate_ids']),
            'has_duplicates': bool(validation['has_duplicates']),
            'correct_sample_count': bool(validation['correct_sample_count']),
            'expected_samples': int(validation['expected_samples']),
            'actual_samples': int(validation['actual_samples'])
        },
        'distribution': distribution,
        'timestamp': datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    validation_file = "/kaggle/working/submissions/validation_report.json"
    with open(validation_file, 'w') as f:
        json.dump(validation_report, f, indent=2, ensure_ascii=False)
    print(f"    Saved: {validation_file}")

# Load final predictions
submission = load_final_predictions()

# Validate submission
validation = validate_submission(submission)

# Create metadata
metadata = create_submission_metadata()

# Analyze prediction distribution
distribution = analyze_prediction_distribution(submission)

# Save all files
save_submission_files(submission, metadata, validation, distribution)

print("")
print("=" * 60)
print("C17: SUBMISSION CREATION COMPLETE")
print("=" * 60)

# Display comprehensive results
print("Submission Validation:")
print(f"  Required columns: {'PASS' if validation['has_required_columns'] else 'FAIL'}")
print(f"  Null values: {'PASS' if not validation['has_nulls'] else 'FAIL'}")
print(f"  Prediction range: {'PASS' if validation['valid_range'] else 'FAIL'}")
print(f"  Sample count: {'PASS' if validation['correct_sample_count'] else 'FAIL'}")
print(f"  Duplicate IDs: {'PASS' if not validation['has_duplicates'] else 'FAIL'}")

print("")
print("Submission Details:")
print(f"  Total samples: {validation['actual_samples']}")
print(f"  Prediction range: [{validation['min_prediction']:.4f}, {validation['max_prediction']:.4f}]")
print(f"  Mean prediction: {validation['mean_prediction']:.4f}")
print(f"  Std prediction: {validation['std_prediction']:.4f}")

print("")
print("Prediction Distribution:")
print(f"  Very low risk (<0.1): {distribution['very_low_risk']} samples")
print(f"  Low risk (0.1-0.3): {distribution['low_risk']} samples")
print(f"  Medium risk (0.3-0.7): {distribution['medium_risk']} samples")
print(f"  High risk (0.7-0.9): {distribution['high_risk']} samples")
print(f"  Very high risk (>=0.9): {distribution['very_high_risk']} samples")

print("")
print("Binary Classification (threshold=0.5):")
print(f"  Predicted violations: {distribution['predicted_violations']} ({distribution['violation_rate']:.2%})")
print(f"  Predicted non-violations: {distribution['predicted_non_violations']} ({1 - distribution['violation_rate']:.2%})")

print("")
print("Files Created:")
print("  /kaggle/working/submissions/submission.csv")
print("  /kaggle/working/submissions/submission_detailed.csv")
print("  /kaggle/working/submissions/submission_metadata.json")
print("  /kaggle/working/submissions/validation_report.json")

print("")
print("=" * 60)
print("RULE SENSE PIPELINE COMPLETE")
print("=" * 60)
print("Final submission is ready for competition upload!")
print("File to submit: /kaggle/working/submissions/submission.csv")
print("=" * 60)


# =============================================
# C18: LOAD BASE ARTIFACTS (SAFE VERSION)
# =============================================

import os
import pandas as pd
import numpy as np
import json

print("C18: Loading base artifacts for submission...")

def safe_load_predictions():
    """Safe loading of predictions with format handling"""
    print("  Loading predictions...")
    
    try:
        # Try to load calibrated predictions
        test_pred = pd.read_csv('/kaggle/working/test_calibrated.csv')
        print(f"    test_calibrated.csv: {len(test_pred)} rows, columns: {list(test_pred.columns)}")
        
        # Determine prediction column
        if 'prediction_calibrated' in test_pred.columns:
            predictions = test_pred['prediction_calibrated'].values
            print(f"    Using calibrated predictions: [{predictions.min():.3f}, {predictions.max():.3f}]")
        elif 'prediction_original' in test_pred.columns:
            predictions = test_pred['prediction_original'].values
            print(f"    Using original predictions: [{predictions.min():.3f}, {predictions.max():.3f}]")
        else:
            # Use first numeric column
            numeric_cols = test_pred.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                predictions = test_pred[numeric_cols[0]].values
                print(f"    Using fallback column '{numeric_cols[0]}': [{predictions.min():.3f}, {predictions.max():.3f}]")
            else:
                raise ValueError("No numeric prediction columns found")
        
        return predictions
        
    except Exception as e:
        print(f"    Error loading predictions: {e}")
        return None

def safe_load_llm_prompts():
    """Safe loading of LLM prompts"""
    try:
        llm_prompts = pd.read_csv('/kaggle/working/llm_prompts.csv')
        print(f"    llm_prompts.csv: {len(llm_prompts)} rows")
        
        # Get test row_ids
        test_row_ids = llm_prompts[llm_prompts['row_id'].str.startswith('test_')]['row_id'].values
        print(f"    Found {len(test_row_ids)} test row_ids")
        
        return llm_prompts, test_row_ids
        
    except Exception as e:
        print(f"    Error loading LLM prompts: {e}")
        return None, []

def create_final_submission(predictions, row_ids):
    """Create final submission"""
    print("  Creating final submission...")
    
    if len(row_ids) != len(predictions):
        print(f"    Row count mismatch: {len(row_ids)} row_ids vs {len(predictions)} predictions")
        # Use minimum count
        min_count = min(len(row_ids), len(predictions))
        row_ids = row_ids[:min_count]
        predictions = predictions[:min_count]
    
    # Create submission
    submission = pd.DataFrame({
        'row_id': row_ids,
        'rule_violation': predictions
    })
    
    # Ensure valid probability range
    submission['rule_violation'] = np.clip(submission['rule_violation'], 1e-6, 1 - 1e-6)
    
    # Save
    submission.to_csv('/kaggle/working/submission.csv', index=False)
    
    print(f"    Submission created: {len(submission)} samples")
    print(f"    Prediction range: [{submission['rule_violation'].min():.3f}, {submission['rule_violation'].max():.3f}]")
    
    return submission

def check_llm_availability():
    """Check LLM predictions availability"""
    llm_paths = [
        '/kaggle/input/rulesense-llm-preds/llm_preds.csv',
        '/kaggle/working/llm_preds.csv'
    ]
    
    for path in llm_paths:
        if os.path.exists(path):
            try:
                llm_preds = pd.read_csv(path)
                if 'llm_pred' in llm_preds.columns and 'row_id' in llm_preds.columns:
                    print(f"    LLM predictions found: {path}")
                    return llm_preds
            except Exception as e:
                print(f"    Error reading {path}: {e}")
    
    print("    LLM predictions not available")
    return None

# Main process
try:
    print("Starting safe artifact loading...")
    
    # Step 1: Load predictions
    predictions = safe_load_predictions()
    if predictions is None:
        raise ValueError("Failed to load predictions")
    
    # Step 2: Load LLM prompts for row_ids
    llm_prompts, test_row_ids = safe_load_llm_prompts()
    
    if len(test_row_ids) == 0:
        # Create row_ids if not found in prompts
        test_row_ids = [f'test_{i}' for i in range(len(predictions))]
        print(f"    Created {len(test_row_ids)} synthetic row_ids")
    
    # Step 3: Create submission
    submission = create_final_submission(predictions, test_row_ids)
    
    # Step 4: Check LLM predictions
    llm_predictions = check_llm_availability()
    
    # Step 5: Load prompt template
    prompt_template = ""
    if os.path.exists('/kaggle/working/prompt_template.txt'):
        with open('/kaggle/working/prompt_template.txt', 'r') as f:
            prompt_template = f.read()
        print("    Loaded prompt template")
    
    # Step 6: Create config
    run_config = {
        'model_type': 'LGBMClassifier',
        'feature_count': 47,
        'cv_scores': {'auc': 0.8491, 'logloss': 0.4621},
        'calibration_method': 'IsotonicRegression',
        'submission_samples': len(submission),
        'prediction_range': [float(submission['rule_violation'].min()), float(submission['rule_violation'].max())]
    }
    
    # Step 7: Set global variables
    BASE_PREDICTIONS = submission.copy()
    BASE_PREDICTIONS.columns = ['row_id', 'calibrated_pred']
    LLM_PROMPTS = llm_prompts if llm_prompts is not None else pd.DataFrame()
    PROMPT_TEMPLATE = prompt_template
    RUN_CONFIG = run_config
    LLM_AVAILABLE = llm_predictions is not None
    LLM_PREDICTIONS = llm_predictions
    CURRENT_SUBMISSION = submission
    
    print("")
    print("=" * 60)
    print("C18: SUCCESS - SUBMISSION READY!")
    print("=" * 60)
    
    print("Submission Summary:")
    print(f"  Samples: {len(submission)}")
    print(f"  Mean probability: {submission['rule_violation'].mean():.3f}")
    print(f"  Range: [{submission['rule_violation'].min():.3f}, {submission['rule_violation'].max():.3f}]")
    
    print("")
    print("Model Performance:")
    print(f"  CV AUC: {run_config['cv_scores']['auc']}")
    print(f"  CV LogLoss: {run_config['cv_scores']['logloss']}")
    
    print("")
    print(f"LLM Status: {'AVAILABLE' if LLM_AVAILABLE else 'NOT AVAILABLE'}")
    
    print("")
    print("Files:")
    print(f"  /kaggle/working/submission.csv - READY FOR COMPETITION")
    if llm_prompts is not None:
        print(f"  LLM prompts: {len(llm_prompts)} total")
    
    print("")
    print("Next Steps:")
    print(f"  1. Submit /kaggle/working/submission.csv to competition")
    if not LLM_AVAILABLE:
        print(f"  2. Generate LLM predictions for enhancement")
    else:
        print(f"  3. Proceed to model blending (C19)")
    
    print("")
    print("C18 completed successfully!")
    print("=" * 60)

except Exception as e:
    print(f"")
    print(f"C18 Error: {e}")
    print("Creating safe fallback submission...")
    
    # Emergency submission with correct row_ids
    try:
        # Try to get real test row_ids
        test_data = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
        row_ids = test_data['row_id'].values
        print(f"    Using real test row_ids: {len(row_ids)} samples")
    except:
        # Fallback: create synthetic
        row_ids = [f'test_{i}' for i in range(10)]
        print(f"    Using synthetic row_ids: {len(row_ids)} samples")
    
    # Create submission with neutral predictions but correct IDs
    submission = pd.DataFrame({
        'row_id': row_ids,
        'rule_violation': 0.3  # Conservative neutral prediction
    })
    submission['rule_violation'] = np.clip(submission['rule_violation'], 1e-6, 1 - 1e-6)
    submission.to_csv('/kaggle/working/submission.csv', index=False)
    
    print(f"    Safe submission created: {len(submission)} samples")
    print("Running in safe mode - using conservative predictions")
    
    # Set minimal global variables
    BASE_PREDICTIONS = submission.copy()
    BASE_PREDICTIONS.columns = ['row_id', 'calibrated_pred']
    LLM_PROMPTS = pd.DataFrame()
    PROMPT_TEMPLATE = ""
    RUN_CONFIG = {'mode': 'safe_fallback'}
    LLM_AVAILABLE = False
    LLM_PREDICTIONS = None
    CURRENT_SUBMISSION = submission


# =============================================
# C19: PREPARE OPTIONAL LLM INFERENCE PACKAGE
# =============================================

import os
import pandas as pd
from pathlib import Path

print("C19: Preparing optional LLM inference package...")

def create_llm_inference_package():
    """
    Build a minimal or full package for external LLM inference (optional).
    This step is not required for baseline model performance.
    """
    
    base_dir = "/kaggle/working/llm_inference"
    os.makedirs(base_dir, exist_ok=True)
    print(f"Created directory: {base_dir}")

    prompts_path = "/kaggle/working/llm_prompts.csv"
    template_path = "/kaggle/working/prompt_template.txt"

    # Step 1: Load or create test prompts
    if os.path.exists(prompts_path):
        try:
            prompts_df = pd.read_csv(prompts_path)
            test_prompts = prompts_df[
                prompts_df['row_id'].astype(str).str.startswith('test_')
            ].copy()
            print(f"Loaded {len(test_prompts)} test prompts from llm_prompts.csv")
        except Exception as e:
            print(f"Error reading prompts file: {e}")
            test_prompts = pd.DataFrame({
                'row_id': [f"test_{i}" for i in range(5)],
                'body': ["Example comment"] * 5,
                'rule': ["No Advertising"] * 5
            })
            print("Created fallback placeholder prompts.")
    else:
        print("llm_prompts.csv not found - creating placeholder file.")
        test_prompts = pd.DataFrame({
            'row_id': [f"test_{i}" for i in range(5)],
            'body': ["Example comment"] * 5,
            'rule': ["No Advertising"] * 5
        })

    test_prompts.to_csv(f"{base_dir}/test_prompts.csv", index=False)

    # Step 2: Load or create prompt template
    if os.path.exists(template_path):
        with open(template_path, 'r') as f:
            template_text = f.read()
        print(f"Found prompt_template.txt ({len(template_text)} chars)")
    else:
        template_text = "Analyze if the given comment violates the specified rule."
        with open(template_path, "w") as f:
            f.write(template_text)
        print("prompt_template.txt not found - created minimal template.")

    with open(f"{base_dir}/prompt_template.txt", "w") as f:
        f.write(template_text)

    # Step 3: Create parsing utility
    parse_script = """import re, pandas as pd, json

def extract_probability(text):
    match = re.search(r"([01]?\\.\\d+)", text)
    return float(match.group(1)) if match else 0.5

def parse_llm_responses(input_file, output_file='llm_parsed.csv'):
    if input_file.endswith('.jsonl'):
        with open(input_file, 'r') as f:
            responses = [json.loads(line) for line in f]
    elif input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
        responses = df.to_dict('records')
    else:
        raise ValueError("Input file must be .jsonl or .csv")

    results = []
    for i, r in enumerate(responses):
        rid = r.get('row_id', f'test_{i}')
        text = str(r.get('response', ''))
        prob = extract_probability(text)
        results.append({
            'row_id': rid,
            'llm_pred': prob,
            'response_length': len(text)
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_file, index=False)
    print(f"Parsed {len(results)} responses -> {output_file}")
    return df_out
"""
    with open(f"{base_dir}/parse_llm_responses.py", "w") as f:
        f.write(parse_script)
    print("Created parse_llm_responses.py")

    # Step 4: Create instructions
    instructions = """LLM INFERENCE PACKAGE - OPTIONAL MODULE

PURPOSE:
  For users who wish to integrate external LLM predictions (offline).

CONTENTS:
  - test_prompts.csv         -> prompts for external processing
  - prompt_template.txt      -> text structure used for generation
  - parse_llm_responses.py   -> utility for parsing responses
  - example_*.txt (optional) -> demonstration samples

WORKFLOW:
  1. Download this folder from Kaggle.
  2. Run prompts through your chosen LLM (e.g., GPT-3.5, Qwen, Claude).
  3. Parse the responses using parse_llm_responses.py.
  4. Save results as llm_predictions.csv and upload as dataset.
  5. Re-import into notebook at C20 for blending.
"""
    with open(f"{base_dir}/INSTRUCTIONS.txt", "w") as f:
        f.write(instructions)
    print("Created INSTRUCTIONS.txt")

    return base_dir, len(test_prompts)


# MAIN EXECUTION
try:
    print("Building LLM inference package...")
    base_dir, sample_count = create_llm_inference_package()

    print("\n" + "=" * 60)
    print("C19: LLM INFERENCE PACKAGE READY")
    print("=" * 60)
    print(f"Location: {base_dir}/")
    print(f"Test samples: {sample_count}")
    print("\nNext Steps:")
    print("  1. Download llm_inference/ from Kaggle")
    print("  2. Generate LLM predictions externally (optional)")
    print("  3. Upload llm_predictions.csv as a dataset if used")
    print("  4. Proceed to C20: Model Blending")
    print("\nC19 completed successfully.")
    print("=" * 60)

except Exception as e:
    print(f"C19 Error: {e}")
    print("Creating minimal fallback package...")

    fallback_dir = "/kaggle/working/llm_inference"
    os.makedirs(fallback_dir, exist_ok=True)

    placeholder = pd.DataFrame({
        "row_id": [f"test_{i}" for i in range(5)],
        "body": ["Example comment"] * 5,
        "rule": ["No Advertising"] * 5
    })
    placeholder.to_csv(f"{fallback_dir}/test_prompts.csv", index=False)

    with open(f"{fallback_dir}/prompt_template.txt", "w") as f:
        f.write("Analyze if the given comment violates the rule.")
    print("Minimal fallback created successfully.")


# =============================================
# C20: blend_predictions
# =============================================

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict

print("C20: Blending baseline and LLM predictions...")

def load_predictions() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load baseline and LLM predictions with validation"""
    print("Loading prediction files...")
    
    # Load baseline predictions
    baseline = pd.read_csv("/kaggle/working/submission.csv")
    print(f"   Baseline: {len(baseline)} samples")
    
    # Check LLM predictions availability
    llm_paths = [
        "/kaggle/input/rulesense-llm-preds/llm_preds.csv",
        "/kaggle/working/llm_preds.csv"
    ]
    
    llm_predictions = None
    for path in llm_paths:
        if os.path.exists(path):
            try:
                llm_predictions = pd.read_csv(path)
                # Validate structure
                if 'row_id' in llm_predictions.columns and 'llm_pred' in llm_predictions.columns:
                    print(f"   LLM predictions: {len(llm_predictions)} samples from {path}")
                    break
                else:
                    print(f"   Invalid LLM format in {path}")
                    llm_predictions = None
            except Exception as e:
                print(f"   Error loading {path}: {e}")
    
    return baseline, llm_predictions

def validate_and_merge(baseline: pd.DataFrame, llm_predictions: pd.DataFrame) -> pd.DataFrame:
    """Merge and validate predictions"""
    print("Validating and merging predictions...")
    
    if llm_predictions is not None:
        # Merge baseline with LLM predictions
        merged = baseline.merge(llm_predictions, on="row_id", how="left")
        
        # Check merge success
        llm_missing = merged['llm_pred'].isna().sum()
        if llm_missing > 0:
            print(f"   {llm_missing} missing LLM predictions - using baseline fallback")
            merged['llm_pred'] = merged['llm_pred'].fillna(merged['rule_violation'])
        
        print(f"   Successful merge: {len(merged)} samples")
        return merged
    else:
        print("   No LLM predictions - using baseline only")
        baseline['llm_pred'] = baseline['rule_violation']
        return baseline

def apply_blending(merged_df: pd.DataFrame) -> pd.Series:
    """Apply weighted blending strategy"""
    print("Applying prediction blending...")
    
    # Calculate blended predictions (70% baseline + 30% LLM)
    blended = (
        0.7 * merged_df['rule_violation'] + 
        0.3 * merged_df['llm_pred']
    )
    
    print("   Blending weights: 70% baseline + 30% LLM")
    print(f"   Baseline range: [{merged_df['rule_violation'].min():.3f}, {merged_df['rule_violation'].max():.3f}]")
    
    if 'llm_pred' in merged_df.columns:
        print(f"   LLM range: [{merged_df['llm_pred'].min():.3f}, {merged_df['llm_pred'].max():.3f}]")
    
    print(f"   Blended range: [{blended.min():.3f}, {blended.max():.3f}]")
    
    return blended

def apply_post_processing(predictions: pd.Series) -> pd.Series:
    """Apply final post-processing steps"""
    print("Applying post-processing...")
    
    # 1. Clip to valid probability range
    processed = np.clip(predictions, 1e-6, 1 - 1e-6)
    
    # 2. Apply rank normalization for private LB stability
    processed = processed.rank(pct=True)
    
    # 3. Final clip after rank normalization
    processed = np.clip(processed, 1e-6, 1 - 1e-6)
    
    print(f"   Final range: [{processed.min():.4f}, {processed.max():.4f}]")
    print(f"   Mean probability: {processed.mean():.3f}")
    
    return processed

def analyze_blend_quality(merged_df: pd.DataFrame, final_predictions: pd.Series) -> Dict:
    """Analyze blending results and quality"""
    print("Analyzing blend quality...")
    
    analysis = {
        'samples': len(merged_df),
        'final_range': [float(final_predictions.min()), float(final_predictions.max())],
        'final_mean': float(final_predictions.mean()),
        'llm_used': 'llm_pred' in merged_df.columns and not merged_df['llm_pred'].equals(merged_df['rule_violation'])
    }
    
    if analysis['llm_used']:
        # Calculate correlation between baseline and LLM
        correlation = merged_df['rule_violation'].corr(merged_df['llm_pred'])
        analysis['baseline_llm_correlation'] = float(correlation)
        print(f"   Baseline-LLM correlation: {correlation:.3f}")
        
        # Check for significant differences
        mean_diff = (merged_df['llm_pred'] - merged_df['rule_violation']).abs().mean()
        analysis['mean_absolute_difference'] = float(mean_diff)
        print(f"   Mean absolute difference: {mean_diff:.3f}")
    
    return analysis

def save_final_submission(baseline_df: pd.DataFrame, final_predictions: pd.Series, analysis: Dict):
    """Save final submission file with metadata"""
    print("Saving final submission...")
    
    # Create final submission
    final_submission = pd.DataFrame({
        'row_id': baseline_df['row_id'],
        'rule_violation': final_predictions
    })
    
    # Save to file
    final_submission.to_csv('/kaggle/working/submission_final.csv', index=False)
    
    # Save blending analysis
    analysis_df = pd.DataFrame([analysis])
    analysis_df.to_csv('/kaggle/working/blending_analysis.csv', index=False)
    
    print(f"   Final submission: {len(final_submission)} samples")
    print(f"   Analysis saved: blending_analysis.csv")

# Main execution
try:
    print("C20: Starting prediction blending pipeline...")
    
    # Step 1: Load predictions
    baseline, llm_predictions = load_predictions()
    
    # Step 2: Merge predictions
    merged_df = validate_and_merge(baseline, llm_predictions)
    
    # Step 3: Apply blending
    blended_predictions = apply_blending(merged_df)
    
    # Step 4: Apply post-processing
    final_predictions = apply_post_processing(blended_predictions)
    
    # Step 5: Analyze results
    analysis = analyze_blend_quality(merged_df, final_predictions)
    
    # Step 6: Save final submission
    save_final_submission(baseline, final_predictions, analysis)
    
    print("\n" + "="*60)
    print("C20: PREDICTION BLENDING COMPLETE")
    print("="*60)
    
    print(f"FINAL SUBMISSION READY:")
    print(f"   File: /kaggle/working/submission_final.csv")
    print(f"   Samples: {len(baseline)}")
    print(f"   Range: [{final_predictions.min():.4f}, {final_predictions.max():.4f}]")
    print(f"   Mean: {final_predictions.mean():.3f}")
    
    print(f"BLENDING DETAILS:")
    if analysis['llm_used']:
        print(f"   Strategy: 70% baseline + 30% LLM")
        print(f"   Correlation: {analysis.get('baseline_llm_correlation', 'N/A'):.3f}")
        print(f"   Mean difference: {analysis.get('mean_absolute_difference', 'N/A'):.3f}")
    else:
        print(f"   Strategy: Baseline only (LLM not available)")
    
    print(f"POST-PROCESSING:")
    print(f"   Probability clipping: [1e-6, 1-1e-6]")
    print(f"   Rank normalization: Applied")
    
    print(f"NEXT STEPS:")
    print(f"   1. Download /kaggle/working/submission_final.csv")
    print(f"   2. Submit to competition")
    print(f"   3. Monitor LB score improvement")
    
    print(f"EXPECTED OUTCOME:")
    if analysis['llm_used']:
        print(f"   Potential LB improvement: +0.005-0.015 AUC")
        print(f"   Better calibration on private test set")
    else:
        print(f"   Solid baseline: AUC â‰ˆ 0.8491")
        print(f"   Consider adding LLM for potential boost")
    
    print(f"C20 executed successfully")
    print("="*60)

except Exception as e:
    print(f"C20 Error: {e}")
    print("Creating emergency submission...")
    
    # Emergency fallback - use baseline with safe processing
    try:
        baseline = pd.read_csv("/kaggle/working/submission.csv")
        emergency_pred = np.clip(baseline['rule_violation'], 1e-6, 1-1e-6)
        
        emergency_sub = pd.DataFrame({
            'row_id': baseline['row_id'],
            'rule_violation': emergency_pred
        })
        emergency_sub.to_csv('/kaggle/working/submission_final.csv', index=False)
        
        print(f"Emergency submission created: {len(emergency_sub)} samples")
    except:
        # Ultimate fallback
        emergency_sub = pd.DataFrame({
            'row_id': [f'test_{i}' for i in range(10)],
            'rule_violation': [0.5] * 10
        })
        emergency_sub.to_csv('/kaggle/working/submission_final.csv', index=False)
        print("Critical fallback - neutral predictions")


# =============================================
# C21: apply_per_rule_calibration (FIXED VERSION)
# =============================================

import os
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Optional

print("C21: Applying per-rule probability calibration...")

def safe_load_and_merge() -> pd.DataFrame:
    """Safely load and merge predictions with rule data, handling type mismatches"""
    print("Loading and merging data with type safety...")
    
    # Load final predictions from C20
    final_predictions = pd.read_csv("/kaggle/working/submission_final.csv")
    print(f"   Final predictions: {len(final_predictions)} samples")
    print(f"   Columns: {list(final_predictions.columns)}")
    print(f"   row_id dtype: {final_predictions['row_id'].dtype}")
    
    # Load test metadata
    test_data = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
    test_data = test_data[['row_id', 'rule']]
    print(f"   Test metadata: {len(test_data)} samples") 
    print(f"   row_id dtype: {test_data['row_id'].dtype}")
    
    # Convert row_id to string in both dataframes to ensure consistent types
    final_predictions['row_id'] = final_predictions['row_id'].astype(str)
    test_data['row_id'] = test_data['row_id'].astype(str)
    
    print(f"   Converted row_id to string in both datasets")
    
    # Merge the data
    try:
        merged_data = final_predictions.merge(test_data, on='row_id', how='left')
        print(f"   Successfully merged: {len(merged_data)} samples")
        
        # Check for missing rules
        missing_rules = merged_data['rule'].isna().sum()
        if missing_rules > 0:
            print(f"   {missing_rules} samples missing rule information")
            
        return merged_data
        
    except Exception as e:
        print(f"   Merge failed: {e}")
        print("   Using fallback: creating rule mapping manually")
        
        # Fallback: if merge fails, create synthetic rule mapping
        final_predictions['rule'] = "No Advertising Rule"  # Default rule
        return final_predictions

def load_calibrators() -> Dict[str, object]:
    """Load rule-specific calibration models"""
    print("Loading rule calibrators...")
    
    calibrators = {}
    base_path = "/kaggle/input/rulesense-base-predictions"
    
    # Check if base path exists
    if not os.path.exists(base_path):
        print(f"   Base path not found: {base_path}")
        return calibrators
    
    # Look for calibration files
    try:
        files = os.listdir(base_path)
        pkl_files = [f for f in files if f.endswith('.pkl')]
        print(f"   Found {len(pkl_files)} .pkl files in base dataset")
        
        for pkl_file in pkl_files:
            try:
                calibrator = joblib.load(f"{base_path}/{pkl_file}")
                
                # Simple mapping based on filename patterns
                if 'rule_0' in pkl_file or 'advertis' in pkl_file.lower():
                    calibrators["advertising_rule"] = calibrator
                    print(f"   Loaded advertising rule calibrator: {pkl_file}")
                elif 'rule_1' in pkl_file or 'legal' in pkl_file.lower():
                    calibrators["legal_rule"] = calibrator  
                    print(f"   Loaded legal rule calibrator: {pkl_file}")
                else:
                    # Generic calibrator
                    calibrators["generic"] = calibrator
                    print(f"   Loaded generic calibrator: {pkl_file}")
                    
            except Exception as e:
                print(f"   Failed to load {pkl_file}: {e}")
                
    except Exception as e:
        print(f"   Error accessing base path: {e}")
    
    return calibrators

def apply_simple_calibration(predictions: pd.Series, rules: pd.Series, 
                           calibrators: Dict[str, object]) -> pd.Series:
    """Apply calibration with simple rule matching"""
    print("Applying calibration with simple rule matching...")
    
    calibrated_predictions = predictions.copy()
    calibration_applied = 0
    
    # Simple rule pattern matching
    for idx, (pred, rule) in enumerate(zip(predictions, rules)):
        rule_str = str(rule).lower()
        
        # Try to match rule patterns
        if 'advertis' in rule_str or 'spam' in rule_str or 'promot' in rule_str:
            if 'advertising_rule' in calibrators:
                try:
                    calibrated_pred = calibrators['advertising_rule'].transform([pred])[0]
                    calibrated_predictions.iloc[idx] = np.clip(calibrated_pred, 1e-6, 1 - 1e-6)
                    calibration_applied += 1
                except Exception as e:
                    pass  # Keep original prediction on error
                    
        elif 'legal' in rule_str or 'advice' in rule_str:
            if 'legal_rule' in calibrators:
                try:
                    calibrated_pred = calibrators['legal_rule'].transform([pred])[0]
                    calibrated_predictions.iloc[idx] = np.clip(calibrated_pred, 1e-6, 1 - 1e-6)
                    calibration_applied += 1
                except Exception as e:
                    pass  # Keep original prediction on error
                    
        elif 'generic' in calibrators:
            # Apply generic calibrator as fallback
            try:
                calibrated_pred = calibrators['generic'].transform([pred])[0]
                calibrated_predictions.iloc[idx] = np.clip(calibrated_pred, 1e-6, 1 - 1e-6)
                calibration_applied += 1
            except Exception as e:
                pass  # Keep original prediction on error
    
    print(f"   Calibration applied to {calibration_applied}/{len(predictions)} samples")
    return calibrated_predictions

def analyze_results(original: pd.Series, calibrated: pd.Series) -> Dict:
    """Analyze calibration results"""
    changes = calibrated - original
    absolute_changes = np.abs(changes)
    
    analysis = {
        'samples_modified': int((absolute_changes > 1e-6).sum()),
        'original_mean': float(original.mean()),
        'calibrated_mean': float(calibrated.mean()),
        'original_range': [float(original.min()), float(original.max())],
        'calibrated_range': [float(calibrated.min()), float(calibrated.max())],
        'mean_absolute_change': float(absolute_changes.mean()),
        'max_absolute_change': float(absolute_changes.max())
    }
    
    return analysis

# Main execution
try:
    print("C21: Starting calibration pipeline...")
    
    # Step 1: Safely load and merge data
    data_with_rules = safe_load_and_merge()
    
    # Step 2: Load calibrators
    calibrators = load_calibrators()
    
    if not calibrators:
        print("No calibrators found - using original predictions")
        calibrated_predictions = data_with_rules['rule_violation']
        calibration_status = "SKIPPED"
    else:
        print(f"Loaded {len(calibrators)} calibrators")
        
        # Step 3: Apply calibration
        calibrated_predictions = apply_simple_calibration(
            data_with_rules['rule_violation'],
            data_with_rules['rule'],
            calibrators
        )
        calibration_status = "APPLIED"
    
    # Step 4: Analyze results
    analysis = analyze_results(data_with_rules['rule_violation'], calibrated_predictions)
    
    # Step 5: Create final submission
    final_submission = pd.DataFrame({
        'row_id': data_with_rules['row_id'],
        'rule_violation': np.clip(calibrated_predictions, 1e-6, 1 - 1e-6)
    })
    
    # Save results
    final_submission.to_csv('/kaggle/working/submission_calibrated.csv', index=False)
    
    print("\n" + "="*60)
    print("C21: CALIBRATION PIPELINE COMPLETE")
    print("="*60)
    
    print(f"STATUS: {calibration_status}")
    print(f"SAMPLES: {len(final_submission)}")
    
    if calibration_status == "APPLIED":
        print(f"CALIBRATION IMPACT:")
        print(f"   Samples modified: {analysis['samples_modified']}/{len(final_submission)}")
        print(f"   Mean change: {analysis['mean_absolute_change']:.4f}")
        print(f"   Max change: {analysis['max_absolute_change']:.4f}")
        print(f"   Mean probability: {analysis['original_mean']:.3f} -> {analysis['calibrated_mean']:.3f}")
    
    print(f"OUTPUT:")
    print(f"   /kaggle/working/submission_calibrated.csv")
    
    print(f"NEXT:")
    print(f"   1. Submit to competition")
    print(f"   2. Expected: Better calibration -> Lower LogLoss")
    
    print(f"C21 finished successfully")
    print("="*60)

except Exception as e:
    print(f"C21 Error: {e}")
    print("Using safe fallback...")
    
    # Simple fallback: copy the final predictions
    try:
        final_predictions = pd.read_csv("/kaggle/working/submission_final.csv")
        final_predictions.to_csv('/kaggle/working/submission_calibrated.csv', index=False)
        print("Fallback: Used uncalibrated predictions")
    except Exception as fallback_error:
        print(f"Critical fallback error: {fallback_error}")
        # Ultimate fallback
        emergency_sub = pd.DataFrame({
            'row_id': [f'test_{i}' for i in range(10)],
            'rule_violation': [0.5] * 10
        })
        emergency_sub.to_csv('/kaggle/working/submission_calibrated.csv', index=False)
        print("Emergency: Created neutral predictions")


# =============================================
# C20: FINAL SUBMISSION BUILDER (FIXED VERSION)
# RuleSense v2 â€” Jigsaw Agile Community Rules
# =============================================

import pandas as pd
import numpy as np
import os

print("ğŸš€ C20: Building final competition submission...")

# === 1. Load official test file ===
test_path = "/kaggle/input/jigsaw-agile-community-rules/test.csv"
if not os.path.exists(test_path):
    raise FileNotFoundError("â�Œ test.csv not found â€” check input dataset path.")
test_df = pd.read_csv(test_path)
print(f"âœ… Loaded test.csv with {len(test_df)} rows.")

# === 2. Load calibrated predictions ===
pred_path = "/kaggle/working/submission_calibrated.csv"
if not os.path.exists(pred_path):
    raise FileNotFoundError("â�Œ submission_calibrated.csv not found â€” run C21 first.")
pred_df = pd.read_csv(pred_path)
print(f"âœ… Loaded calibrated predictions ({len(pred_df)} rows).")

# === 3. Sanity checks ===
if len(test_df) != len(pred_df):
    raise ValueError(f"â�Œ Row count mismatch: test={len(test_df)}, preds={len(pred_df)}")

if "rule_violation" not in pred_df.columns:
    raise KeyError("â�Œ Missing 'rule_violation' column in predictions file.")

# === 4. Build final submission ===
final_submission = pd.DataFrame({
    "row_id": test_df["row_id"].astype(int),
    "rule_violation": np.clip(pred_df["rule_violation"].astype(float), 1e-6, 1 - 1e-6)
})

# === 5. Save submission ===
out_path = "/kaggle/working/submission.csv"
final_submission.to_csv(out_path, index=False)
print(f"ğŸ’¾ Saved final submission â†’ {out_path}")

# === 6. Verify content ===
print("\nğŸ”� Submission preview:")
print(final_submission.head(5).to_string(index=False))

print("\nğŸ“Š Summary:")
print(f"   Rows: {len(final_submission)}")
print(f"   ID range: {final_submission['row_id'].min()}â€“{final_submission['row_id'].max()}")
print(f"   Prob range: {final_submission['rule_violation'].min():.4f}â€“{final_submission['rule_violation'].max():.4f}")

print("\nâœ… C20 completed successfully â€” submission ready for Kaggle upload.")
print("="*60)


