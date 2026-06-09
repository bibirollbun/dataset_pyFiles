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


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')



def load_and_explore_data():
    """Load và khám phá dữ liệu training"""
    print("Loading training data...")
    
    # Load các file training
    delay_4_6 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv')
    delay_7_9 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv')
    not_delay_4_6 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/not_delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv')
    not_delay_7_9 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/not_delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv')
    
    print(f"Data shapes:")
    print(f" - Delay 4-6: {delay_4_6.shape}")
    print(f" - Delay 7-9: {delay_7_9.shape}")
    print(f" - Not delay 4-6: {not_delay_4_6.shape}")
    print(f" - Not delay 7-9: {not_delay_7_9.shape}")
    
    # Thêm label
    delay_4_6['label'] = 1  # delay
    delay_7_9['label'] = 1  # delay
    not_delay_4_6['label'] = 0  # not delay
    not_delay_7_9['label'] = 0  # not delay
    
    # Gộp tất cả dữ liệu training
    train_data = pd.concat([delay_4_6, delay_7_9, not_delay_4_6, not_delay_7_9], 
                          ignore_index=True)
    
    print(f"\n Combined training data shape: {train_data.shape}")
    print(f" Label distribution:")
    label_counts = train_data['label'].value_counts()
    print(f" - Not delay (0): {label_counts[0]:,} ({label_counts[0]/len(train_data)*100:.1f}%)")
    print(f" - Delay (1): {label_counts[1]:,} ({label_counts[1]/len(train_data)*100:.1f}%)")
    
    return train_data

# Load data
train_df = load_and_explore_data()


print(f"\nColumns ({len(train_df.columns)}):")
for i, col in enumerate(train_df.columns, 1):
    print(f"  {i:2d}. {col}")


print(f"\nData Types:")
print(train_df.dtypes.value_counts())


missing_data = train_df.isnull().sum()
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
if len(missing_data) > 0:
    print(missing_data.head(10))
else:
    print("  No missing values found!")


# Load test data
def load_test_data():
    """Load dữ liệu test"""
    test_data = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/PILOT_10.csv')
    print(f"Test data shape: {test_data.shape}")
    return test_data

test_df = load_test_data()


print(f"\nTest data columns ({len(test_df.columns)}):")
for i, col in enumerate(test_df.columns, 1):
    print(f"  {i:2d}. {col}")


def preprocess_data(train_df, test_df):
    """Tiền xử lý dữ liệu"""
    print("Preprocessing data...")
    
    # Chuẩn hóa tên cột - xử lý vấn đề SPECIAL DIV vs SPECIAL_DIV
    print("Standardizing column names...")
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Rename columns để tránh conflict
    if 'SPECIAL DIV' in train_df.columns and 'SPECIAL_DIV' in train_df.columns:
        print("Found both 'SPECIAL DIV' and 'SPECIAL_DIV' - renaming...")
        train_df = train_df.rename(columns={'SPECIAL DIV': 'SPECIAL_DIV_SPACE'})
    
    if 'SPECIAL DIV' in test_df.columns and 'SPECIAL_DIV' in test_df.columns:
        test_df = test_df.rename(columns={'SPECIAL DIV': 'SPECIAL_DIV_SPACE'})
    
    # Lấy các cột chung giữa train và test (trừ label)
    train_cols = set(train_df.columns) - {'label'}
    test_cols = set(test_df.columns) - {'ID'}
    common_cols = list(train_cols.intersection(test_cols))
    
    print(f"Common features: {len(common_cols)}")
    
    # Chỉ giữ lại các cột chung
    X_train = train_df[common_cols].copy()
    y_train = train_df['label'].copy()
    X_test = test_df[common_cols].copy()
    
    # Kiểm tra duplicate columns
    print("Checking for duplicate columns...")
    duplicate_cols = X_train.columns[X_train.columns.duplicated()].tolist()
    if duplicate_cols:
        print(f"Found duplicate columns: {duplicate_cols}")
        X_train = X_train.loc[:, ~X_train.columns.duplicated()]
        X_test = X_test.loc[:, ~X_test.columns.duplicated()]
        print(f"After removing duplicates - Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, y_train, X_test

# Preprocess data
X_train_raw, y_train, X_test_raw = preprocess_data(train_df, test_df)


def feature_engineering(X_train, X_test):
    """Feature engineering"""
    print("Feature engineering...")
    
    # Xử lý datetime columns
    datetime_cols = []
    for col in X_train.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            datetime_cols.append(col)
    
    print(f"Datetime columns: {datetime_cols}")
    
    # Convert datetime và extract features
    for col in datetime_cols:
        if col in X_train.columns:
            try:
                # Convert to datetime
                X_train[col] = pd.to_datetime(X_train[col], errors='coerce')
                X_test[col] = pd.to_datetime(X_test[col], errors='coerce')
                
                # Extract features
                X_train[f'{col}_year'] = X_train[col].dt.year
                X_train[f'{col}_month'] = X_train[col].dt.month
                X_train[f'{col}_day'] = X_train[col].dt.day
                X_train[f'{col}_dayofweek'] = X_train[col].dt.dayofweek
                
                X_test[f'{col}_year'] = X_test[col].dt.year
                X_test[f'{col}_month'] = X_test[col].dt.month
                X_test[f'{col}_day'] = X_test[col].dt.day
                X_test[f'{col}_dayofweek'] = X_test[col].dt.dayofweek
                
                # Drop original datetime column
                X_train = X_train.drop(col, axis=1)
                X_test = X_test.drop(col, axis=1)
                
            except Exception as e:
                print(f"Error processing {col}: {e}")
    
    # Xử lý categorical columns
    categorical_cols = []
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            categorical_cols.append(col)
    
    print(f"Categorical columns: {len(categorical_cols)}")
    
    # Label encoding cho categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        
        # Combine train and test for consistent encoding
        combined_values = pd.concat([X_train[col].astype(str), X_test[col].astype(str)])
        le.fit(combined_values.fillna('missing'))
        
        X_train[col] = le.transform(X_train[col].astype(str).fillna('missing'))
        X_test[col] = le.transform(X_test[col].astype(str).fillna('missing'))
        
        label_encoders[col] = le
    
    # Fill missing values cho numeric columns
    for col in X_train.columns:
        if X_train[col].dtype in ['int64', 'float64']:
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
    
    print(f"Final feature shape - Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test, label_encoders

# Apply feature engineering
X_train, X_test, label_encoders = feature_engineering(X_train_raw, X_test_raw)


# Visualize label distribution
plt.figure(figsize=(10, 6))

# Subplot 1: Count plot
plt.subplot(1, 2, 1)
label_counts = y_train.value_counts()
plt.bar(['Not Delay (0)', 'Delay (1)'], label_counts.values, color=['lightblue', 'salmon'])
plt.title('Label Distribution (Count)')
plt.ylabel('Count')
for i, v in enumerate(label_counts.values):
    plt.text(i, v + 10000, f'{v:,}', ha='center', va='bottom')

# Subplot 2: Percentage plot
plt.subplot(1, 2, 2)
plt.pie(label_counts.values, labels=['Not Delay (0)', 'Delay (1)'], 
        autopct='%1.1f%%', colors=['lightblue', 'salmon'])
plt.title('Label Distribution (Percentage)')

plt.tight_layout()
plt.show()

print(f"Class Imbalance Ratio: {label_counts[0]/label_counts[1]:.1f}:1")


# Feature correlation analysis
print("Feature Correlation Analysis:")
numeric_features = X_train.select_dtypes(include=[np.number]).columns
print(f"Numeric features: {len(numeric_features)}")

if len(numeric_features) > 1:
    # Calculate correlation with target
    correlations = []
    for col in numeric_features:
        corr = np.corrcoef(X_train[col], y_train)[0, 1]
        if not np.isnan(corr):
            correlations.append((col, abs(corr)))
    
    # Sort by correlation strength
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 features by correlation with target:")
    for i, (feature, corr) in enumerate(correlations[:10], 1):
        print(f"  {i:2d}. {feature}: {corr:.4f}")


def train_lightgbm_model(X_train, y_train):
    """Training LightGBM model với cross-validation"""
    print("Training LightGBM model...")
    
    # LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    # Cross-validation với Macro F1-Score
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_scores = []
    models = []
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\nTraining fold {fold + 1}/5...")
        
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
        val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        # Predict và tính F1-score
        val_pred = model.predict(X_fold_val)
        val_pred_binary = (val_pred > 0.5).astype(int)
        
        f1_macro = f1_score(y_fold_val, val_pred_binary, average='macro')
        cv_scores.append(f1_macro)
        models.append(model)
        
        # Store fold results
        fold_results.append({
            'fold': fold + 1,
            'f1_score': f1_macro,
            'best_iteration': model.best_iteration
        })
        
        print(f"Fold {fold + 1} - F1-Score: {f1_macro:.4f}, Best iteration: {model.best_iteration}")
    
    print(f"\nCross-Validation Results:")
    print(f"  - Average CV Macro F1-Score: {np.mean(cv_scores):.4f} (±{np.std(cv_scores)*2:.4f})")
    print(f"  - Best fold: {np.argmax(cv_scores)+1} (F1: {np.max(cv_scores):.4f})")
    print(f"  - Worst fold: {np.argmin(cv_scores)+1} (F1: {np.min(cv_scores):.4f})")
    
    return models, cv_scores, fold_results

# Train model
models, cv_scores, fold_results = train_lightgbm_model(X_train, y_train)


# Visualize CV results
plt.figure(figsize=(12, 5))

# Subplot 1: F1-scores by fold
plt.subplot(1, 2, 1)
folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
plt.bar(folds, cv_scores, color='skyblue', alpha=0.7)
plt.axhline(y=np.mean(cv_scores), color='red', linestyle='--', 
           label=f'Mean: {np.mean(cv_scores):.4f}')
plt.title('F1-Score by Fold')
plt.ylabel('Macro F1-Score')
plt.legend()
plt.xticks(rotation=45)

# Add value labels on bars
for i, v in enumerate(cv_scores):
    plt.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')

# Subplot 2: Best iterations
plt.subplot(1, 2, 2)
iterations = [result['best_iteration'] for result in fold_results]
plt.bar(folds, iterations, color='lightcoral', alpha=0.7)
plt.axhline(y=np.mean(iterations), color='blue', linestyle='--', 
           label=f'Mean: {np.mean(iterations):.0f}')
plt.title('Best Iteration by Fold')
plt.ylabel('Best Iteration')
plt.legend()
plt.xticks(rotation=45)

# Add value labels on bars
for i, v in enumerate(iterations):
    plt.text(i, v + 5, f'{v}', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# Feature importance analysis
print("Feature Importance Analysis:")

# Calculate average feature importance across all folds
feature_importance = np.zeros(len(X_train.columns))
for model in models:
    feature_importance += model.feature_importance(importance_type='gain')

feature_importance /= len(models)

# Create feature importance dataframe
feature_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\nTop 15 Most Important Features:")
for i, (_, row) in enumerate(feature_importance_df.head(15).iterrows(), 1):
    print(f"  {i:2d}. {row['feature']}: {row['importance']:.2f}")

# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance_df.head(20)
plt.barh(range(len(top_features)), top_features['importance'], color='lightgreen', alpha=0.7)
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance (Gain)')
plt.title('Top 20 Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


def make_predictions(models, X_test, test_df):
    """Tạo predictions và file submission"""
    print("Making predictions...")
    
    # Average predictions từ tất cả models
    predictions = np.zeros(len(X_test))
    
    for i, model in enumerate(models):
        pred = model.predict(X_test)
        predictions += pred
        print(f"Model {i+1}/5 predictions completed")
    
    predictions /= len(models)
    
    # Convert to binary predictions
    binary_predictions = (predictions > 0.5).astype(int)
    
    print(f"\nPrediction Statistics:")
    print(f"- Probability mean: {predictions.mean():.4f}")
    print(f"- Probability std: {predictions.std():.4f}")
    print(f"- Probability min: {predictions.min():.4f}")
    print(f"- Probability max: {predictions.max():.4f}")
    
    # Tạo submission file
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'label': binary_predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    print(f"\nSubmission file saved as 'submission.csv'")
    print(f"File shape: {submission.shape}")
    
    pred_counts = pd.Series(binary_predictions).value_counts()
    print(f"\nFinal Prediction Distribution:")
    print(f"- Not delay (0): {pred_counts[0]:,} ({pred_counts[0]/len(binary_predictions)*100:.1f}%)")
    print(f"- Delay (1): {pred_counts[1]:,} ({pred_counts[1]/len(binary_predictions)*100:.1f}%)")
    
    return submission, predictions

# Make predictions
submission, prediction_probabilities = make_predictions(models, X_test, test_df)


# Visualize prediction results
plt.figure(figsize=(15, 5))

# Subplot 1: Prediction distribution
plt.subplot(1, 3, 1)
pred_counts = submission['label'].value_counts()
plt.bar(['Not Delay (0)', 'Delay (1)'], pred_counts.values, color=['lightblue', 'salmon'])
plt.title('Prediction Distribution')
plt.ylabel('Count')
for i, v in enumerate(pred_counts.values):
    plt.text(i, v + 1000, f'{v:,}', ha='center', va='bottom')

# Subplot 2: Prediction probabilities histogram
plt.subplot(1, 3, 2)
plt.hist(prediction_probabilities, bins=50, alpha=0.7, color='green')
plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
plt.xlabel('Prediction Probability')
plt.ylabel('Frequency')
plt.title('Prediction Probabilities Distribution')
plt.legend()

# Subplot 3: CV scores
plt.subplot(1, 3, 3)
folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
plt.plot(folds, cv_scores, 'o-', color='purple', linewidth=2, markersize=8)
plt.axhline(y=np.mean(cv_scores), color='red', linestyle='--', 
           label=f'Mean: {np.mean(cv_scores):.4f}')
plt.title('Cross-Validation F1-Scores')
plt.ylabel('Macro F1-Score')
plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

