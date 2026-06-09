# install dependencies
!pip install -q watermark


#------------------------------------------------------------------------------
#   Project: Calorie Expenditure Prediction
#
#   Description:    An in-depth feature analaysis
# 
#   Author:         Dr. Saad Laouadi
#   
#   Created:        May 13, 2025
#   Last Modified:  May 13, 2025
#   Version:        1.0.0
#------------------------------------------------------------------------------


# *************************************************
#                Environment Setup
# *************************************************

# Standard libraries
import os
import sys
import time
import pathlib
import warnings
from datetime import datetime

# Data processing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# Visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.pyplot import figure


# Configure warnings
warnings.filterwarnings('ignore')

# Pandas display options
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 3)
pd.set_option('display.float_format', '{:.3f}'.format)
pd.set_option('display.width', 1000)

# Matplotlib and seaborn configuration
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['figure.figsize'] = (12, 8)
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
sns.set_context("notebook", font_scale=1.2)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Display notebook information
%reload_ext watermark
%watermark -iv -v -m -p pandas,numpy,matplotlib,seaborn -ud -a "Dr. Saad Laouadi"

print("\nEnvironment setup completed successfully.")


# CONFIGURATION: File paths
INPUT_DIR = pathlib.Path("/kaggle/input/playground-series-s5e5").resolve()
TRAIN_PATH = INPUT_DIR.joinpath('train.csv')
TEST_PATH = INPUT_DIR.joinpath('test.csv')
SUBMISSION_PATH = INPUT_DIR.joinpath('sample_submission.csv')


# Verify data files exist and display status
def validate_paths():
    """Validate that all required data files exist and print status."""
    required_files = {
        "Training data": TRAIN_PATH,
        "Test data": TEST_PATH,
        "Submission template": SUBMISSION_PATH
    }
    
    all_exist = True
    for name, path in required_files.items():
        exists = path.exists()
        all_exist = all_exist and exists
        status = "✓ Found" if exists else "✗ Missing"
        print(f"{name:<20}: {status:<10} ({path})")
    
    return all_exist

# Check data files
print("Calorie Expenditure Prediction Project - Configuration")
print("=" * 50)
data_valid = validate_paths()
if not data_valid:
    print("\nWarning: Some required files are missing. Please check the paths above.")


# *************************************************
#             Dataset Exploration
# *************************************************

def explore_dataset(path, name="Dataset"):
    """
    Simple exploration of a dataset using pandas built-in methods.
    
    Parameters:
    -----------
    path : pathlib.Path
        Path to the dataset CSV file
    name : str
        Name of the dataset for display purposes
    """
    print(f"\n{'=' * 50}")
    print(f"{name} Exploration")
    print(f"{'=' * 50}")
    
    # Check if file exists
    if not path.exists():
        print(f"ERROR: File not found at {path}")
        return None
    
    # Load the dataset
    print(f"\nLoading {name} from {path}...")
    df = pd.read_csv(path)
    print(f"Dataset loaded successfully.\n")
    
    # Basic information
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    # Display data types and missing values
    print("Data Information:")
    df.info()
    
    # Display sample data
    print("\nSample Data (First 5 rows):")
    display(df.head())
    
    # Statistical summary
    print("\nStatistical Summary:")
    display(df.describe())
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\nMissing Values:")
        display(missing[missing > 0])
    else:
        print("\nNo missing values found.")
    
    return df


# Explore the datasets
train_df = explore_dataset(TRAIN_PATH, "Training Dataset")
test_df = explore_dataset(TEST_PATH, "Testing Dataset")
submission_df = explore_dataset(SUBMISSION_PATH, "Submission Template")

# Basic comparison
if train_df is not None and test_df is not None:
    print(f"\n{'=' * 50}")
    print("Train/Test Comparison")
    print(f"{'=' * 50}")
    
    # Compare columns
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    
    print(f"\nColumns in train but not in test: {sorted(train_cols - test_cols)}")
    print(f"Columns in test but not in train: {sorted(test_cols - train_cols)}")
    print(f"Common columns: {len(train_cols.intersection(test_cols))}")


# *************************************************
#         In-Depth Feature Analysis
# *************************************************

# Focus only on the training dataset from now on
print("In-Depth Feature Analysis of Training Dataset")
print("=" * 50)

# 1. Feature Type Assessment
print("\n1. Feature Type Assessment")
print("-" * 30)

# Check data types
print("Current data types:")
print(train_df.dtypes)

# 2. Univariate Analysis & Target Variable Analysis
print("\n2. Univariate Analysis")
print("-" * 30)

# Numerical features - create histograms
numerical_features = train_df.select_dtypes(include=['number']).columns

# Exclude the id column from the num_features
numerical_features = [col for col in numerical_features if col !='id']

print(f"Number of numerical features: {len(numerical_features)}")

# Set up the figure for histograms
plt.figure(figsize=(15, 10))

# Create a subplot grid - adjust rows based on feature count
n_cols = 3
n_rows = (len(numerical_features) + n_cols - 1) // n_cols

for i, feature in enumerate(numerical_features):
    plt.subplot(n_rows, n_cols, i+1)
    sns.histplot(train_df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    
plt.show()


# Special focus on the target variable
plt.figure(figsize=(10, 6))
sns.histplot(train_df['Calories'], kde=True, color='green')
plt.title('Distribution of Target Variable: Calories')
plt.xlabel('Calories')
plt.ylabel('Frequency')
plt.show()


print("\nTarget Variable Statistics:")
print(train_df['Calories'].describe())


# 3. Correlation Analysis
print("\n3. Correlation Analysis")
print("-" * 30)

cor_cols = [col for col in train_cols if col not in ['Sex', 'id']]
# Calculate correlation matrix
correlation_matrix = train_df[cor_cols].corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# Sort correlations with target variable
target_correlations = correlation_matrix['Calories'].sort_values(ascending=False)
print("\nFeature Correlations with Target (Calories):")
print(target_correlations)


# 4. Scatter plots for top correlated features
print("\n4. Scatter Plots for Top Correlated Features")
print("-" * 52)

# Get top correlated features (excluding the target itself)
top_correlated = target_correlations[1:6].index.tolist()  # Top 5 correlated features
print(f"Top 5 correlated features with calories: {top_correlated}")


# Create scatter plots
plt.figure(figsize=(15, 10))
for i, feature in enumerate(top_correlated):
    plt.subplot(2, 3, i+1)
    sns.scatterplot(x=train_df[feature], y=train_df['Calories'])
    plt.title(f'{feature} vs. calories (corr: {correlation_matrix.loc[feature, "Calories"]:.2f})')
    plt.tight_layout()

plt.show()


# Summary of findings
print("\nSummary of Feature Analysis:")
print("-" * 30)
print(f"1. Total features: {train_df.shape[1]}")
print(f"2. Numerical features: {len(numerical_features)}")
print(
    f"3. Target variable 'calories' distribution:"
    f" Mean={train_df['Calories'].mean():.2f},"
    f" Std={train_df['Calories'].std():.2f}"
)

print(
    f"4. Most correlated feature with calories:"
    f" {target_correlations.index[1]} (r={target_correlations.iloc[1]:.2f})"
)




