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


train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")


train_df.info()


train_df.isnull().sum()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


def describe_and_plot_dataset(df, figsize=(15, 10)):
    """
    Generate comprehensive descriptive statistics and distribution plots for a numerical dataframe
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing only numerical columns
    figsize (tuple): Figure size for the plots
    """
    
    # Basic info about the dataset
    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing values
    print("\n" + "=" * 50)
    print("MISSING VALUES")
    print("=" * 50)
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_summary = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    })
    print(missing_summary[missing_summary['Missing Count'] > 0])
    
    if missing_summary['Missing Count'].sum() == 0:
        print("No missing values found!")
    
    # Descriptive statistics
    print("\n" + "=" * 50)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 50)
    desc_stats = df.describe()
    print(desc_stats)
    
    # Additional statistics
    print("\n" + "=" * 50)
    print("ADDITIONAL STATISTICS")
    print("=" * 50)
    additional_stats = pd.DataFrame(index=df.columns)
    additional_stats['Skewness'] = df.skew()
    additional_stats['Kurtosis'] = df.kurtosis()
    additional_stats['CV (%)'] = (df.std() / df.mean()) * 100  # Coefficient of Variation
    print(additional_stats)
    
    # Correlation matrix
    print("\n" + "=" * 50)
    print("CORRELATION MATRIX")
    print("=" * 50)
    corr_matrix = df.corr()
    print(corr_matrix)
    
    # Create distribution plots
    n_cols = len(df.columns)
    n_rows = (n_cols + 2) // 3  # 3 columns per row
    
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    fig.suptitle('Distribution Plots for All Numerical Variables', fontsize=16, y=1.02)
    
    # Flatten axes array for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    for i, column in enumerate(df.columns):
        ax = axes_flat[i]
        
        # Create histogram with KDE overlay
        sns.histplot(data=df, x=column, kde=True, ax=ax, alpha=0.7)
        ax.set_title(f'{column}\nMean: {df[column].mean():.2f}, Std: {df[column].std():.2f}')
        ax.grid(True, alpha=0.3)
        
        # Add vertical line for mean
        ax.axvline(df[column].mean(), color='red', linestyle='--', alpha=0.8, label='Mean')
        ax.axvline(df[column].median(), color='green', linestyle='--', alpha=0.8, label='Median')
        ax.legend()
    
    # Hide empty subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Box plots for outlier detection
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    fig.suptitle('Box Plots for Outlier Detection', fontsize=16, y=1.02)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    for i, column in enumerate(df.columns):
        ax = axes_flat[i]
        sns.boxplot(data=df, y=column, ax=ax)
        ax.set_title(f'{column}')
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                mask=mask, square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()
    
    # Identify potential outliers using IQR method
    print("\n" + "=" * 50)
    print("OUTLIER DETECTION (IQR Method)")
    print("=" * 50)
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        print(f"{column}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")


describe_and_plot_dataset(train_df.drop(columns=['id']))


test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")


# Check if test data has target column (some competitions include it, some don't)
has_target_in_test = 'BeatsPerMinute' in test_df.columns
if has_target_in_test:
    print("⚠️  Target column found in test data - removing it")
    test_df = test_df.drop('BeatsPerMinute', axis=1)


def preprocess_data(df, is_training=True):
    """
    Preprocess the dataframe for CatBoost training
    """
    df_processed = df.copy()
    
    # Remove ID column if exists
    if 'id' in df_processed.columns:
        print(f"  Removing ID column")
        df_processed = df_processed.drop('id', axis=1)
    
    # Convert object columns to categorical
    object_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    # Don't convert target variable
    if is_training and 'BeatsPerMinute' in object_columns:
        object_columns.remove('BeatsPerMinute')
    
    print(f"  Converting to categorical: {object_columns}")
    
    for col in object_columns:
        df_processed[col] = df_processed[col].astype('category')
        print(f"    {col}: {df_processed[col].nunique()} categories")
    
    return df_processed


# Preprocess training and test data
train_processed = preprocess_data(train_df, is_training=True)
test_processed = preprocess_data(test_df, is_training=False)

print(f"\nProcessed training shape: {train_processed.shape}")
print(f"Processed test shape: {test_processed.shape}")


# Prepare feature lists
target_col = 'BeatsPerMinute'
id_cols = ['id'] if 'id' in train_df.columns else []
exclude_cols = [target_col] + id_cols


# Get feature columns
feature_columns = [col for col in train_processed.columns if col not in exclude_cols]
print(f"Selected features ({len(feature_columns)}): {feature_columns}")


# Prepare X and y
X_train = train_processed[feature_columns]
y_train = train_processed[target_col]
X_test = test_processed[feature_columns]

# Check for any missing columns in test set
missing_in_test = set(feature_columns) - set(X_test.columns)
if missing_in_test:
    print(f"⚠️  WARNING: Features missing in test set: {missing_in_test}")

print(f"\nFinal training features shape: {X_train.shape}")
print(f"Final test features shape: {X_test.shape}")


# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
from xgboost import XGBRegressor


# Train final model on full training data
# final_model = LGBMRegressor(n_estimators=1000)
# final_model = CatBoostRegressor(n_estimators=1000, loss_function='RMSE', verbose=100, random_seed=42)
final_model = XGBRegressor(n_estimators=1000, random_state=42)


print("Training final model on full dataset...")
final_model.fit(
    X_train, y_train,
)


# Show feature importance
# feature_importance = final_model.feature_importances_
feature_importance = final_model.feature_importances_
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df.head(10).to_string(index=False))


# Generate predictions
print("Generating predictions on test set...")
test_predictions = final_model.predict(X_test)

print(f"✅ Predictions generated!")
print(f"Prediction distribution:")
print(f"  - Prediction range: [{test_predictions.min():.6f}, {test_predictions.max():.6f}]")



# Create submission dataframe
submission = pd.DataFrame()

# Add ID column (adjust based on your competition format)
if 'id' in test_df.columns:
    submission['id'] = test_df['id']
elif 'Id' in test_df.columns:
    submission['Id'] = test_df['Id']
else:
    # If no ID column, create index-based IDs
    submission['id'] = range(len(test_df))
    print("⚠️  No ID column found, using index as ID")

# Add predictions (adjust column name based on competition requirements)
# Common formats: 'y', 'target', 'prediction', 'Survived', etc.
SUBMISSION_TARGET_COLUMN = 'BeatsPerMinute'  # Change this to match your competition

if SUBMISSION_TARGET_COLUMN == 'BeatsPerMinute':
    # For binary classification, some competitions want probabilities, others want binary
    # Check your competition requirements!
    
    # Option 1: Prediction of regression (more common)
    submission[SUBMISSION_TARGET_COLUMN] = test_predictions


print(f"Submission format:")
print(f"  Columns: {list(submission.columns)}")
print(f"  Shape: {submission.shape}")
print(f"  Sample:")
print(submission.head())

# Save submission file
submission_filename = 'submission.csv'
submission.to_csv(submission_filename, index=False)

print(f"\n✅ SUBMISSION SAVED: {submission_filename}")




