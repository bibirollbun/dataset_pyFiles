!pip install lightgbm


!pip install xgboost


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import tarfile
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Install required packages if needed
!pip install pyarrow

# Load the parquet files
train_df = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet')
sample_submission = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/sample_submission.parquet')

print("Train shape:", train_df.shape)
print("Sample submission shape:", sample_submission.shape)
print("\nTrain columns:", train_df.columns.tolist())
print("Sample submission columns:", sample_submission.columns.tolist())


print(sample_submission.head(10))





import tarfile
import json
import os
import gc

# Don't extract all files at once - process them one by one
def process_json_tar_efficiently(tar_path):
    """
    Process JSON files from tar archive without extracting all to disk
    """
    all_json_data = []
    
    with tarfile.open(tar_path, 'r') as tar:
        # Get list of JSON files in the archive
        json_members = [member for member in tar.getmembers() if member.name.endswith('.json')]
        print(f"Found {len(json_members)} JSON files in archive")
        
        # Process files in batches to manage memory
        batch_size = 10  # Process 10 files at a time
        
        for i in range(0, len(json_members), batch_size):
            batch = json_members[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(json_members)-1)//batch_size + 1}")
            
            for member in batch:
                try:
                    # Extract file to memory, not disk
                    f = tar.extractfile(member)
                    if f is not None:
                        content = f.read().decode('utf-8')
                        data = json.loads(content)
                        
                        # Process the JSON data
                        processed_data = process_single_json(data, member.name)
                        if processed_data:
                            all_json_data.append(processed_data)
                        
                        # Clean up
                        f.close()
                        del content, data
                        
                except Exception as e:
                    print(f"Error processing {member.name}: {e}")
                    continue
            
            # Force garbage collection after each batch
            gc.collect()
            
            # Stop after processing enough files for feature extraction
            if len(all_json_data) >= 1000:  # Adjust this number based on your needs
                print(f"Processed {len(all_json_data)} files, stopping to save memory")
                break
    
    return all_json_data

def process_single_json(data, filename):
    """
    Extract features from a single JSON object
    Customize this based on your JSON structure
    """
    try:
        processed = {
            'filename': filename,
        }
        
        # Add feature extraction based on your JSON structure
        if isinstance(data, dict):
            # Example feature extractions - adjust based on your data
            processed['num_keys'] = len(data.keys())
            
            # Common RecSys features
            if 'user_id' in data:
                processed['user_id'] = data['user_id']
            if 'item_id' in data:
                processed['item_id'] = data['item_id']
            if 'rating' in data:
                processed['rating'] = data['rating']
            if 'timestamp' in data:
                processed['timestamp'] = data['timestamp']
            if 'interactions' in data:
                processed['num_interactions'] = len(data['interactions']) if isinstance(data['interactions'], list) else 1
            
        return processed
        
    except Exception as e:
        print(f"Error processing JSON {filename}: {e}")
        return None

# Process the JSON files efficiently
json_data = process_json_tar_efficiently('/kaggle/input/aeroclub-recsys-2025/jsons_raw.tar.kaggle')
json_df = pd.DataFrame(json_data)

print(f"Extracted features from {len(json_df)} JSON files")
if not json_df.empty:
    print("JSON DataFrame shape:", json_df.shape)
    print("JSON DataFrame columns:", json_df.columns.tolist())
    print(json_df.head())



# Data Exploration and Understanding
print("=== TRAIN DATA EXPLORATION ===")
print(train_df.head())
print("\nTrain data info:")
print(train_df.info())
print("\nTrain data description:")
print(train_df.describe())

print("\n=== SAMPLE SUBMISSION EXPLORATION ===")
print(sample_submission.head())
print("\nSample submission info:")
print(sample_submission.info())

# Check for missing values
print("\n=== MISSING VALUES ===")
print("Train missing values:")
print(train_df.isnull().sum())
print("\nSample submission missing values:")
print(sample_submission.isnull().sum())

# Initialize extracted_files as empty list since extraction failed
extracted_files = []

# Try to explore JSON structure without extracting files
print("\n=== JSON STRUCTURE EXPLORATION ===")
try:
    # Read the JSON structure documentation
    with open('/kaggle/input/aeroclub-recsys-2025/jsons_structure.md', 'r') as f:
        json_structure = f.read()
    print("JSON Structure Documentation:")
    print(json_structure)
except Exception as e:
    print(f"Could not read JSON structure file: {e}")

# Try to peek at one JSON file without extracting all
print("\n=== SAMPLE JSON PEEK ===")
try:
    import tarfile
    import json
    
    with tarfile.open('/kaggle/input/aeroclub-recsys-2025/jsons_raw.tar.kaggle', 'r') as tar:
        # Get first JSON file
        json_members = [member for member in tar.getmembers() if member.name.endswith('.json')]
        if json_members:
            first_json = json_members[0]
            f = tar.extractfile(first_json)
            if f is not None:
                content = f.read().decode('utf-8')
                sample_json = json.loads(content)
                print(f"Sample JSON file: {first_json.name}")
                print(f"JSON keys: {list(sample_json.keys()) if isinstance(sample_json, dict) else 'Not a dictionary'}")
                print(f"JSON structure preview: {str(sample_json)[:500]}...")
                f.close()
        else:
            print("No JSON files found in archive")
            
except Exception as e:
    print(f"Could not peek at JSON files: {e}")



def create_recommendation_features(train_df):
    """
    Create features for recommendation system using only parquet data
    """
    df_features = train_df.copy()
    
    # Identify potential user/item columns
    print("Available columns:", df_features.columns.tolist())
    
    # Basic feature engineering based on common RecSys patterns
    # Check if we have user_id column (or similar)
    user_cols = [col for col in df_features.columns if 'user' in col.lower()]
    item_cols = [col for col in df_features.columns if any(x in col.lower() for x in ['item', 'product', 'movie', 'book', 'song'])]
    rating_cols = [col for col in df_features.columns if any(x in col.lower() for x in ['rating', 'score', 'target'])]
    
    print(f"Detected user columns: {user_cols}")
    print(f"Detected item columns: {item_cols}")
    print(f"Detected rating columns: {rating_cols}")
    
    # User-based features
    if user_cols:
        user_col = user_cols[0]  # Fix: Take first element, not the list
        target_col = rating_cols[0] if rating_cols else df_features.columns[-1]  # Fix: Take first element
        
        user_stats = df_features.groupby(user_col).agg({
            target_col: ['count', 'mean', 'std', 'min', 'max']
        }).reset_index()
        user_stats.columns = [user_col] + [f'user_{stat}' for stat in ['count', 'mean', 'std', 'min', 'max']]
        df_features = df_features.merge(user_stats, on=user_col, how='left')
        print(f"Added user-based features for column: {user_col}")
    
    # Item-based features
    if item_cols:
        item_col = item_cols[0]  # Fix: Take first element, not the list
        target_col = rating_cols[0] if rating_cols else df_features.columns[-1]  # Fix: Take first element
        
        item_stats = df_features.groupby(item_col).agg({
            target_col: ['count', 'mean', 'std', 'min', 'max']
        }).reset_index()
        item_stats.columns = [item_col] + [f'item_{stat}' for stat in ['count', 'mean', 'std', 'min', 'max']]
        df_features = df_features.merge(item_stats, on=item_col, how='left')
        print(f"Added item-based features for column: {item_col}")
    
    # Additional time-based features if timestamp exists
    timestamp_cols = [col for col in df_features.columns if any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
    if timestamp_cols:
        timestamp_col = timestamp_cols[0]  # Fix: Take first element, not the list
        try:
            df_features[timestamp_col] = pd.to_datetime(df_features[timestamp_col])
            df_features['hour'] = df_features[timestamp_col].dt.hour
            df_features['day_of_week'] = df_features[timestamp_col].dt.dayofweek
            df_features['month'] = df_features[timestamp_col].dt.month
            print(f"Added time-based features for column: {timestamp_col}")
        except Exception as e:
            print(f"Could not convert {timestamp_col} to datetime: {e}")
    
    return df_features

# Apply feature engineering
train_features = create_recommendation_features(train_df)
print(f"Enhanced train features shape: {train_features.shape}")
print("New columns added:", [col for col in train_features.columns if col not in train_df.columns])



def preprocess_features(df):
    """
    Handle missing values and encode categorical variables
    """
    df_processed = df.copy()
    
    # Fill missing values
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    
    print(f"Numeric columns: {len(numeric_columns)}")
    print(f"Categorical columns: {len(categorical_columns)}")
    
    for col in numeric_columns:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
            print(f"Filled {col} missing values with median")
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_columns:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 'unknown', inplace=True)
        
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded categorical column: {col}")
    
    return df_processed, label_encoders

# Apply preprocessing
train_processed, encoders = preprocess_features(train_features)
print(f"Processed train shape: {train_processed.shape}")
print("Preprocessing completed!")



# Prepare features and target
# Identify target column based on sample_submission
target_col = sample_submission.columns[-1]  # Usually the last column is target
if target_col not in train_processed.columns:
    # If target column name doesn't match, use the last column of train data
    target_col = train_processed.columns[-1]

print(f"Target column: {target_col}")

# Prepare feature columns
feature_cols = [col for col in train_processed.columns if col != target_col]
print(f"Number of features: {len(feature_cols)}")

# Check if we have valid features and target
if len(feature_cols) == 0:
    print("No feature columns found! Using all columns except the last one as features")
    feature_cols = train_processed.columns[:-1].tolist()
    target_col = train_processed.columns[-1]

X = train_processed[feature_cols]
y = train_processed[target_col]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target statistics: {y.describe()}")

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")



# Check data types and identify problematic columns
print("Checking data types:")
print(X_train.dtypes)
print("\nProblematic columns:")
datetime_cols = X_train.select_dtypes(include=['datetime64']).columns
print("Datetime columns:", datetime_cols.tolist())

# Fix datetime columns by converting to numeric
def fix_datetime_columns(df):
    """Convert datetime columns to numeric format"""
    df_fixed = df.copy()
    
    # Convert datetime columns to timestamp (numeric)
    datetime_cols = df_fixed.select_dtypes(include=['datetime64']).columns
    for col in datetime_cols:
        print(f"Converting datetime column: {col}")
        # Convert to timestamp (seconds since epoch)
        df_fixed[col] = pd.to_datetime(df_fixed[col]).astype('int64') // 10**9
        
    # Ensure all columns are numeric
    for col in df_fixed.columns:
        if df_fixed[col].dtype == 'object':
            print(f"Converting object column to numeric: {col}")
            df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')
            df_fixed[col].fillna(0, inplace=True)
    
    return df_fixed

# Apply fixes to training and validation data
X_train_fixed = fix_datetime_columns(X_train)
X_val_fixed = fix_datetime_columns(X_val)

print("Fixed data types:")
print(X_train_fixed.dtypes)



# import lightgbm as lgb
# from sklearn.metrics import mean_squared_error

# # LightGBM parameters
# lgb_params = {
#     'objective': 'regression',
#     'metric': 'rmse',
#     'boosting_type': 'gbdt',
#     'num_leaves': 31,
#     'learning_rate': 0.05,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'verbose': -1,
#     'random_state': 42
# }

# # Create datasets with fixed data
# train_data = lgb.Dataset(X_train_fixed, label=y_train)
# val_data = lgb.Dataset(X_val_fixed, label=y_val, reference=train_data)

# # Train model
# print("Training LightGBM model...")
# lgb_model = lgb.train(
#     lgb_params,
#     train_data,
#     valid_sets=[train_data, val_data],
#     num_boost_round=1000,
#     callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
# )

# # Validation predictions
# val_pred = lgb_model.predict(X_val_fixed, num_iteration=lgb_model.best_iteration)
# val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
# print(f"Validation RMSE: {val_rmse:.4f}")



import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np

# XGBoost parameters
xgb_params = {
    'objective': 'reg:squarederror',  # XGBoost 的回归目标
    'max_depth': 8,                   # 类似于 LightGBM 的 num_leaves，通常需要调整
    'learning_rate': 0.05,
    'colsample_bytree': 0.9,          # 类似于 LightGBM 的 feature_fraction
    'subsample': 0.8,                 # 类似于 LightGBM 的 bagging_fraction
    'random_state': 42,
    'n_estimators': 3000,             # 初始树的数量
    'early_stopping_rounds': 100,      # 早停的轮数
    'verbosity': 0                    # 类似于 LightGBM 的 verbose=-1
}

# 创建数据集
dtrain = xgb.DMatrix(X_train_fixed, label=y_train)
dval = xgb.DMatrix(X_val_fixed, label=y_val)

# 训练模型
print("Training XGBoost model...")
xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=2000,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=50,
    verbose_eval=100
)

# 验证预测
val_pred = xgb_model.predict(dval, ntree_limit=xgb_model.best_iteration)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {val_rmse:.4f}")








# Create test features based on sample_submission structure
print("Preparing test data...")

# Check sample_submission structure
print("Sample submission columns:", sample_submission.columns.tolist())
print("Sample submission shape:", sample_submission.shape)

# Method 1: If sample_submission contains features
if len(sample_submission.columns) > 1:
    # Extract features from sample_submission (excluding target column)
    test_features = sample_submission.drop(columns=[sample_submission.columns[-1]])
    
    # Apply same feature engineering as training data
    print("Applying feature engineering to test data...")
    test_enhanced = create_recommendation_features(test_features)
    
    # Apply same preprocessing
    test_processed = test_enhanced.copy()
    
    # Handle categorical encoding for test data
    categorical_columns = test_processed.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col in encoders:
            test_processed[col] = test_processed[col].astype(str)
            # Handle unseen categories
            unseen_mask = ~test_processed[col].isin(encoders[col].classes_)
            if unseen_mask.any():
                test_processed.loc[unseen_mask, col] = encoders[col].classes_[0]
            test_processed[col] = encoders[col].transform(test_processed[col])
    
    # Fill missing values
    numeric_columns = test_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if test_processed[col].isnull().sum() > 0:
            test_processed[col].fillna(test_processed[col].median(), inplace=True)
    
    # Ensure test data has same features as training data
    missing_cols = set(feature_cols) - set(test_processed.columns)
    for col in missing_cols:
        test_processed[col] = 0
    
    # Select and reorder columns to match training features
    test_processed = test_processed[feature_cols]
    
    print(f"Test data shape after preprocessing: {test_processed.shape}")
    
    # Generate predictions
    test_pred = xgb_model.predict(test_processed)#, num_iteration=lgb_model.best_iteration)
    
else:
    # Method 2: If sample_submission only has ID and target columns
    print("Sample submission appears to only have ID and target columns")
    print("You may need to create test features based on your competition requirements")
    
    # Placeholder - replace with actual test data preparation logic
    test_pred = np.random.random(len(sample_submission))
    print("Using placeholder predictions - replace with actual test data logic")

print(f"Generated {len(test_pred)} test predictions")
print(f"Prediction statistics: min={test_pred.min():.4f}, max={test_pred.max():.4f}, mean={test_pred.mean():.4f}")



# Create submission file
submission = sample_submission.copy()
submission[submission.columns[-1]] = test_pred

# Apply post-processing if needed
# For example, if predictions should be within a certain range:
if 'rating' in submission.columns[-1].lower():
    submission[submission.columns[-1]] = np.clip(submission[submission.columns[-1]], 1, 5)
    print("Clipped predictions to rating range [1, 5]")

print("Submission preview:")
print(submission.head(10))
print(f"\nSubmission shape: {submission.shape}")
print(f"Final prediction statistics:")
print(submission[submission.columns[-1]].describe())

# Save submission files
submission.to_parquet('submission.parquet', index=False)
submission.to_csv('submission.csv', index=False)

print("Submission saved as 'submission.parquet' and 'submission.csv'")
print("Ready for upload to Kaggle!")





