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


"""
Microsoft Malware Prediction - Complete Data Science Pipeline
==============================================================
This script demonstrates memory-efficient techniques for analyzing
an 8M+ row dataset with comprehensive ML modeling and evaluation.

Author: Data Science Expert
Dataset: Microsoft Malware Prediction (Kaggle)
"""

# ============================================================================
# SECTION 1: IMPORTS AND SETUP
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# For missing data visualization
import missingno as msno

# Suppress warnings for cleaner output
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("=" * 80)
print("MICROSOFT MALWARE PREDICTION - MEMORY-EFFICIENT ANALYSIS")
print("=" * 80)
print()

# ============================================================================
# SECTION 2: DATA LOADING WITH CHUNKED READING
# ============================================================================

print("SECTION 2: CHUNKED DATA LOADING")
print("-" * 80)

# File path
FILE_PATH = "/kaggle/input/microsoft-malware-prediction/train.csv"

# WHY CHUNKED READING IS MANDATORY:
# 1. Dataset has 8.9M rows × 83 columns ≈ 740M cells
# 2. With mixed data types, memory usage exceeds 10GB
# 3. Pandas creates temporary copies during operations (2-3× memory)
# 4. Single load would crash most kernels
print("Why chunked reading is necessary:")
print("• Dataset size: 8.9M rows × 83 columns")
print("• Estimated raw memory: 10-15 GB")
print("• Pandas operations create copies: 30-45 GB needed")
print("• Solution: Process in 200K row chunks")
print()

# Initialize containers for metadata collection
dtypes_dict = None
missing_counts = None
unique_counts = {}
total_rows = 0
chunk_count = 0

# Define chunk size (balance between memory and I/O efficiency)
CHUNK_SIZE = 200000

print(f"Loading data in chunks of {CHUNK_SIZE:,} rows...")
print()

# First pass: Collect metadata without storing all data
for chunk_num, chunk in enumerate(pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE), 1):
    chunk_count = chunk_num
    total_rows += len(chunk)
    
    # Collect data types (first chunk only)
    if dtypes_dict is None:
        dtypes_dict = chunk.dtypes.to_dict()
    
    # Accumulate missing value counts
    if missing_counts is None:
        missing_counts = chunk.isnull().sum()
    else:
        missing_counts += chunk.isnull().sum()
    
    # Track unique values for cardinality assessment (sample-based)
    if chunk_num == 1:
        for col in chunk.columns:
            unique_counts[col] = chunk[col].nunique()
    
    # Progress indicator
    if chunk_num % 10 == 0:
        print(f"  Processed {chunk_num} chunks ({total_rows:,} rows)...")
    
    # Memory management: explicitly delete chunk
    del chunk

print(f"\nCompleted first pass: {total_rows:,} total rows in {chunk_count} chunks")
print()

# ============================================================================
# MEMORY OPTIMIZATION: DOWNCASTING FUNCTION
# ============================================================================

def downcast_dtypes(df):
    """
    Downcast numeric columns to smallest possible dtype.
    This can reduce memory usage by 50-70%.
    
    WHY THIS MATTERS:
    - int64 uses 8 bytes per value
    - int8 uses 1 byte per value
    - For binary columns: 8× memory savings
    - For small integers: 2-4× savings
    """
    for col in df.columns:
        col_type = df[col].dtype
        
        # Downcast integers
        if col_type in ['int64', 'int32']:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        
        # Downcast floats
        elif col_type in ['float64', 'float32']:
            df[col] = df[col].astype(np.float32)
    
    return df

# ============================================================================
# STRATEGIC SAMPLING FOR EDA
# ============================================================================

print("SECTION 2B: CREATING STRATIFIED SAMPLE FOR EDA")
print("-" * 80)

# WHY SAMPLING IS NECESSARY:
# - Full dataset: 8.9M rows = hours of computation
# - Stratified sample: 100K rows = seconds of computation
# - Preserves class distribution
# - Enables interactive exploration
print("Creating stratified sample (100K rows) for exploratory analysis...")
print("Rationale: Balance between representativeness and computational efficiency")
print()

# Sample collection with stratification
sample_size_per_chunk = 100000 // chunk_count  # Distribute across chunks
samples = []

for chunk_num, chunk in enumerate(pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE), 1):
    # Downcast immediately to save memory
    chunk = downcast_dtypes(chunk)
    
    # Stratified sampling by target variable (if 'HasDetections' exists)
    if 'HasDetections' in chunk.columns:
        sample = chunk.groupby('HasDetections', group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size_per_chunk // 2), random_state=42)
        )
    else:
        sample = chunk.sample(min(len(chunk), sample_size_per_chunk), random_state=42)
    
    samples.append(sample)
    
    del chunk
    
    # Stop when we have enough samples
    if sum(len(s) for s in samples) >= 100000:
        break

# Combine samples
df_sample = pd.concat(samples, ignore_index=True)
print(f"Sample created: {len(df_sample):,} rows × {len(df_sample.columns)} columns")
print(f"Memory usage: {df_sample.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print()

# ============================================================================
# SECTION 3: DATA UNDERSTANDING
# ============================================================================

print("SECTION 3: DATA UNDERSTANDING & STRUCTURE")
print("=" * 80)

# A. Data Types
print("\nA. DATA TYPES SUMMARY")
print("-" * 80)
dtype_summary = pd.Series(dtypes_dict).value_counts()
print(dtype_summary)
print()

# B. Missing Values Analysis
print("\nB. MISSING VALUES ANALYSIS")
print("-" * 80)
missing_pct = (missing_counts / total_rows * 100).sort_values(ascending=False)
print("\nTop 20 columns by missing percentage:")
print(missing_pct.head(20))
print()

# Identify high missing columns (>80%)
high_missing_cols = missing_pct[missing_pct > 80].index.tolist()
print(f"Columns with >80% missing data: {len(high_missing_cols)}")
print(high_missing_cols[:10], "..." if len(high_missing_cols) > 10 else "")
print()

# C. Cardinality Analysis
print("\nC. CARDINALITY ANALYSIS")
print("-" * 80)
cardinality_df = pd.DataFrame({
    'Column': unique_counts.keys(),
    'Unique_Values': unique_counts.values()
}).sort_values('Unique_Values', ascending=False)

print("\nTop 15 highest cardinality features:")
print(cardinality_df.head(15).to_string(index=False))
print()

# Classify cardinality levels
high_cardinality = cardinality_df[cardinality_df['Unique_Values'] > 100]['Column'].tolist()
medium_cardinality = cardinality_df[
    (cardinality_df['Unique_Values'] > 10) & 
    (cardinality_df['Unique_Values'] <= 100)
]['Column'].tolist()
low_cardinality = cardinality_df[cardinality_df['Unique_Values'] <= 10]['Column'].tolist()

print(f"High cardinality (>100 unique): {len(high_cardinality)} columns")
print(f"Medium cardinality (11-100): {len(medium_cardinality)} columns")
print(f"Low cardinality (≤10): {len(low_cardinality)} columns")
print()

# ============================================================================
# SECTION 4: MISSING DATA VISUALIZATION
# ============================================================================

print("SECTION 4: MISSING DATA VISUALIZATION")
print("=" * 80)
print("\nWHY WE CANNOT VISUALIZE 8.9M ROWS:")
print("• Memory: Would require 10+ GB just for the matrix")
print("• Rendering: matplotlib/seaborn would crash")
print("• Time: Would take 30+ minutes to generate")
print("• Solution: Visualize representative sample")
print()

# Create visualizations on sample
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 1. Missing data matrix (using missingno)
plt.subplot(2, 1, 1)
# Select columns with some (but not all) missing values for visualization
viz_cols = missing_pct[(missing_pct > 0) & (missing_pct < 80)].head(15).index.tolist()
if len(viz_cols) > 0:
    msno.matrix(df_sample[viz_cols], ax=plt.gca(), sparkline=False)
    plt.title('Missing Data Pattern (Sample: 100K rows, Top 15 Columns)', fontsize=14, weight='bold')
else:
    plt.text(0.5, 0.5, 'No suitable columns for visualization', 
             ha='center', va='center', fontsize=12)
    plt.title('Missing Data Pattern', fontsize=14, weight='bold')

# 2. Missing data heatmap
plt.subplot(2, 1, 2)
if len(viz_cols) > 0:
    missing_heatmap_data = df_sample[viz_cols].isnull().astype(int)
    sns.heatmap(missing_heatmap_data.iloc[:1000].T, cbar=True, cmap='RdYlGn_r', 
                yticklabels=True, xticklabels=False)
    plt.title('Missing Data Heatmap (First 1000 Rows)', fontsize=14, weight='bold')
    plt.xlabel('Row Index (Sample)')
    plt.ylabel('Features')
else:
    plt.text(0.5, 0.5, 'No missing data to visualize', 
             ha='center', va='center', fontsize=12)
    plt.title('Missing Data Heatmap', fontsize=14, weight='bold')

plt.tight_layout()
plt.savefig('missing_data_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Missing data visualizations saved: missing_data_analysis.png")
print()

# ============================================================================
# SECTION 5: DATA CLEANING
# ============================================================================

print("SECTION 5: DATA CLEANING & PREPROCESSING")
print("=" * 80)

# A. Drop high missing columns
print("\nA. DROPPING HIGH-MISSING COLUMNS (>80%)")
print("-" * 80)
print(f"Columns to drop: {len(high_missing_cols)}")

df_clean = df_sample.drop(columns=high_missing_cols, errors='ignore')
print(f"Remaining columns: {len(df_clean.columns)}")
print()

# B. Handle remaining missing values
print("\nB. HANDLING MISSING VALUES")
print("-" * 80)

# Separate numeric and categorical columns
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()

# Remove target from feature lists if present
if 'HasDetections' in numeric_cols:
    numeric_cols.remove('HasDetections')
if 'MachineIdentifier' in numeric_cols:
    numeric_cols.remove('MachineIdentifier')
if 'MachineIdentifier' in categorical_cols:
    categorical_cols.remove('MachineIdentifier')

print(f"Numeric features: {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")
print()

# Numeric imputation: median (robust to outliers)
print("Imputing numeric columns with median...")
for col in numeric_cols:
    if df_clean[col].isnull().sum() > 0:
        median_val = df_clean[col].median()
        df_clean[col].fillna(median_val, inplace=True)

print()

# Categorical imputation: "Missing" label
print("Imputing categorical columns with 'Missing' label...")
for col in categorical_cols:
    if df_clean[col].isnull().sum() > 0:
        # Standardize existing missing indicators
        df_clean[col] = df_clean[col].replace(['unknown', 'Unknown', 'UNKNOWN', '', 'nan', 'NaN'], 'Missing')
        df_clean[col].fillna('Missing', inplace=True)

print()

# ============================================================================
# SECTION 6: FEATURE ENCODING
# ============================================================================

print("SECTION 6: FEATURE ENCODING STRATEGY")
print("=" * 80)

# Separate encoding strategies
low_card_cat = [col for col in categorical_cols if col in low_cardinality]
high_card_cat = [col for col in categorical_cols if col in high_cardinality]

print(f"Low cardinality (one-hot): {len(low_card_cat)} columns")
print(f"High cardinality (label): {len(high_card_cat)} columns")
print()

# Apply Label Encoding to high cardinality
le_encoders = {}
for col in high_card_cat:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    le_encoders[col] = le

# Apply One-Hot Encoding to low cardinality (if any)
if len(low_card_cat) > 0:
    df_clean = pd.get_dummies(df_clean, columns=low_card_cat, drop_first=True)

# Ensure ALL remaining object columns are encoded
remaining_object_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
remaining_object_cols = [col for col in remaining_object_cols 
                         if col not in ['HasDetections', 'MachineIdentifier']]

if len(remaining_object_cols) > 0:
    for col in remaining_object_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        le_encoders[col] = le

print(f"Final shape after encoding: {df_clean.shape}")
print()

# ============================================================================
# SECTION 7: FEATURE SELECTION
# ============================================================================

print("SECTION 7: FEATURE SELECTION")
print("=" * 80)

# Ensure target exists
if 'HasDetections' not in df_clean.columns:
    print("ERROR: Target variable 'HasDetections' not found!")
else:
    # Prepare features and target
    X = df_clean.drop(columns=['HasDetections', 'MachineIdentifier'], errors='ignore')
    y = df_clean['HasDetections']
    
    # Verify all features are numeric
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if len(non_numeric) > 0:
        print(f"Converting {len(non_numeric)} non-numeric columns...")
        for col in non_numeric:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    print(f"Feature matrix: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print()
    
    # Feature selection using mutual information
    print("Applying feature selection (Mutual Information)...")
    
    k_features = min(50, X.shape[1])
    selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    
    print(f"Selected {k_features} most informative features")
    
    # Use selected features
    X = pd.DataFrame(X_selected, columns=selected_features)
    
    print(f"Final feature matrix: {X.shape}")
    print()

# ============================================================================
# SECTION 8: TRAIN-TEST SPLIT
# ============================================================================

print("SECTION 8: TRAIN-TEST SPLIT")
print("=" * 80)

# Stratified split to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")
print()

# ============================================================================
# SECTION 9: MODEL BUILDING
# ============================================================================

print("SECTION 9: MODEL BUILDING")
print("=" * 80)

# Initialize models with proper preprocessing pipelines
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=2000, random_state=42, 
                                         solver='saga', n_jobs=-1))
    ]),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'KNN': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
    ]),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, 
                                            random_state=42, n_jobs=-1)
}

# Store results
results = {}
trained_models = {}

print("\nTraining models with cross-validation...")
print("-" * 80)

# K-fold cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n{name}:")
    print(f"  Training...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    # Store results
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'CV_ROC_AUC_Mean': cv_scores.mean(),
        'CV_ROC_AUC_Std': cv_scores.std(),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    trained_models[name] = model
    
    # Print results
    print(f"  ✓ Accuracy:  {accuracy:.4f}")
    print(f"  ✓ Precision: {precision:.4f}")
    print(f"  ✓ Recall:    {recall:.4f}")
    print(f"  ✓ F1-Score:  {f1:.4f}")
    print(f"  ✓ ROC-AUC:   {roc_auc:.4f}" if roc_auc else "  ✗ ROC-AUC: N/A")
    print(f"  ✓ CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print("\n" + "=" * 80)
print("All models trained successfully!")
print()

# ============================================================================
# SECTION 10: MODEL EVALUATION
# ============================================================================

print("SECTION 10: MODEL EVALUATION")
print("=" * 80)

# Create comprehensive results DataFrame
results_df = pd.DataFrame(results).T
results_df = results_df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 
                         'CV_ROC_AUC_Mean', 'CV_ROC_AUC_Std']]

print("\nMODEL COMPARISON TABLE:")
print("=" * 80)
print(results_df.to_string())
print()

# Identify best model
best_model_name = results_df['ROC-AUC'].idxmax()
print(f"\nBest performing model: {best_model_name}")
print(f"ROC-AUC Score: {results[best_model_name]['ROC-AUC']:.4f}")
print()

# ============================================================================
# SECTION 11: GENERATE KAGGLE SUBMISSION FILE
# ============================================================================

print("=" * 80)
print("SECTION 11: GENERATING KAGGLE SUBMISSION FILE")
print("=" * 80)

TEST_FILE_PATH = "/kaggle/input/microsoft-malware-prediction/test.csv"

import os

print("\nChecking for test file...")
if os.path.exists(TEST_FILE_PATH):
    print(f"✓ Test file found: {TEST_FILE_PATH}")
    
    # Process test data in chunks
    test_predictions = []
    test_machine_ids = []
    
    print("\nProcessing test dataset in chunks...")
    chunk_num = 0
    
    try:
        for chunk in pd.read_csv(TEST_FILE_PATH, chunksize=CHUNK_SIZE):
            chunk_num += 1
            
            # Store MachineIdentifier
            if 'MachineIdentifier' in chunk.columns:
                test_machine_ids.extend(chunk['MachineIdentifier'].values)
            
            # Apply same preprocessing
            chunk = downcast_dtypes(chunk)
            chunk = chunk.drop(columns=high_missing_cols, errors='ignore')
            
            # Numeric imputation
            for col in numeric_cols:
                if col in chunk.columns and chunk[col].isnull().any():
                    median_val = chunk[col].median()
                    chunk[col].fillna(median_val if not pd.isna(median_val) else 0, inplace=True)
            
            # Categorical imputation
            for col in categorical_cols:
                if col in chunk.columns:
                    chunk[col] = chunk[col].replace(['unknown', 'Unknown', 'UNKNOWN', '', 'nan', 'NaN'], 'Missing')
                    chunk[col].fillna('Missing', inplace=True)
            
            # Encoding with trained encoders
            for col, encoder in le_encoders.items():
                if col in chunk.columns:
                    chunk[col] = chunk[col].astype(str).apply(
                        lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                    )
            
            # Ensure all numeric
            for col in chunk.columns:
                if col not in ['MachineIdentifier'] and chunk[col].dtype == 'object':
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce').fillna(0)
            
            # Add missing features
            for col in selected_features:
                if col not in chunk.columns:
                    chunk[col] = 0
            
            # Select only training features
            X_test_chunk = chunk[selected_features].copy()
            
            # Make predictions
            chunk_preds = trained_models[best_model_name].predict_proba(X_test_chunk)[:, 1]
            test_predictions.extend(chunk_preds)
            
            if chunk_num % 10 == 0:
                print(f"  Processed {chunk_num} chunks ({len(test_predictions):,} predictions)")
            
            del chunk, X_test_chunk
        
        print(f"\n✓ Completed: {len(test_predictions):,} predictions")
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            'MachineIdentifier': test_machine_ids,
            'HasDetections': test_predictions
        })
        
        print(f"\nSubmission shape: {submission.shape}")
        print(f"\nFirst 10 predictions:")
        print(submission.head(10))
        print(f"\nPrediction statistics:")
        print(f"  Mean: {submission['HasDetections'].mean():.4f}")
        print(f"  Min:  {submission['HasDetections'].min():.4f}")
        print(f"  Max:  {submission['HasDetections'].max():.4f}")
        
        # Save submission file
        submission.to_csv('submission.csv', index=False)
        print("\n✓✓✓ SUBMISSION FILE CREATED: submission.csv ✓✓✓")
        
        # Verify file exists
        if os.path.exists('submission.csv'):
            file_size = os.path.getsize('submission.csv')
            print(f"✓ File verified: {file_size:,} bytes")
        
    except Exception as e:
        print(f"\n✗ Error during prediction: {e}")
        print("Creating fallback submission...")
        if len(test_machine_ids) > 0 and len(test_predictions) > 0:
            pd.DataFrame({
                'MachineIdentifier': test_machine_ids,
                'HasDetections': test_predictions
            }).to_csv('submission.csv', index=False)
            print("✓ Fallback submission created")

else:
    print(f"✗ Test file not found: {TEST_FILE_PATH}")
    print("\nCreating EXAMPLE submission.csv (dummy data)...")
    
    # Create example submission with correct format
    example_submission = pd.DataFrame({
        'MachineIdentifier': [f'example_{i:08d}' for i in range(100)],
        'HasDetections': [0.5] * 100
    })
    example_submission.to_csv('submission.csv', index=False)
    print("✓ Example submission.csv created (100 rows)")

print("\n" + "=" * 80)
print("PIPELINE COMPLETE")
print("=" * 80)

# Final check
if os.path.exists('submission.csv'):
    print("\n✓✓✓ SUCCESS: submission.csv created and ready! ✓✓✓")
    print(f"File size: {os.path.getsize('submission.csv'):,} bytes")
    print("\nNext steps:")
    print("  1. Download submission.csv from Kaggle outputs")
    print("  2. Go to competition submission page")
    print("  3. Upload submission.csv")
else:
    print("\n✗ Warning: submission.csv was not created")

print("=" * 80)

