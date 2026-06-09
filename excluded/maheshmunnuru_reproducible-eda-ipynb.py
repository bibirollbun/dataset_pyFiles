# ==============================================================================
# CELL 1: CONFIGURATION (CORRECTED FOR REGRESSION)
# ==============================================================================
import pandas as pd
import numpy as np
import yaml
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import *
from sklearn.metrics import * 
from lightgbm import * 

# --- Notebook Settings ---
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class Cfg:
    """
    Configuration class for all notebook parameters.
    """
    # --- Core Settings ---
    DEBUG = True
    
    # --- Data Paths & Target ---
    TRAIN_PATH = '/kaggle/input/playground-series-s6e1/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e1/test.csv'
    TARGET_COL = 'exam_score'
    
    # --- EDA Settings ---
    EDA_SAMPLE_FRACTION = 0.2
    
    # --- Output Settings ---
    OUTPUT_YAML_PATH = 'features.yaml'
    
    # --- Adversarial Validation Settings ---
    N_SPLITS_ADV = 5
    RANDOM_STATE = 42

# --- Instantiate the Configuration ---
cfg = Cfg()

if cfg.DEBUG:
    print("=" * 30)
    print("--- RUNNING IN DEBUG MODE ---")
    print("=" * 30)

print("✓ Cell 1/8: Configuration loaded successfully (Corrected for Regression).")



# ==============================================================================
# CELL 2: DATA LOADING & INITIAL SCHEMA
# ==============================================================================
print("\n" + "="*80)
print("Running Cell 2/8: Loading Data and Defining Initial Schema")
print("="*80)

try:
    train_df = pd.read_csv(cfg.TRAIN_PATH)
    test_df = pd.read_csv(cfg.TEST_PATH)
    print("✓ Data loaded successfully.")
    
    # In DEBUG mode, use a sample of the data to speed up the process
    if cfg.DEBUG:
        print(f"...DEBUG MODE: Sampling data down to {cfg.EDA_SAMPLE_FRACTION*100:.0f}% of original size.")
        train_df = train_df.sample(frac=cfg.EDA_SAMPLE_FRACTION, random_state=cfg.RANDOM_STATE).reset_index(drop=True)

except FileNotFoundError as e:
    print(f"❌ ERROR: Data files not found. Please check the paths in the Cfg class.\n{e}")
    # Stop execution if data isn't loaded
    raise

# --- Programmatic Schema Definition ---
TARGET = cfg.TARGET_COL
BASE_FEATURES = [col for col in train_df.columns if col not in ['id', TARGET]]

# Automatically identify initial categorical and numerical features
CATS = train_df[BASE_FEATURES].select_dtypes(include='object').columns.tolist()
NUMS = train_df[BASE_FEATURES].select_dtypes(include=['number', 'float', 'int']).columns.tolist()

print(f"\n--- Initial Data Overview ---")
print(f"Train shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")
print(f"Target column: '{TARGET}'")
print(f"\nFound {len(CATS)} initial categorical features:\n{CATS}")
print(f"\nFound {len(NUMS)} initial numerical features:\n{NUMS}")

print("\n--- First 5 rows of training data: ---")
display(train_df.head())



train_df


# ==============================================================================
# CELL 3: TARGET VARIABLE ANALYSIS
# ==============================================================================
print("\n" + "="*80)
print("Running Cell 3/8: Analyzing the Target Variable Distribution")
print("="*80)

plt.figure(figsize=(14, 6))
sns.histplot(train_df[TARGET], kde=True, bins=50, color='blue')
plt.title(f'Distribution of Target Variable: {TARGET}', fontsize=16)
plt.xlabel(TARGET, fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.axvline(train_df[TARGET].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {train_df[TARGET].mean():.2f}')
plt.axvline(train_df[TARGET].median(), color='green', linestyle='-', linewidth=2, label=f'Median: {train_df[TARGET].median():.2f}')
plt.legend()
plt.show()

print("\n--- Target Variable Statistics ---")
display(train_df[TARGET].describe())



train_df.info()


train_df.describe()


# ==============================================================================
# CELL 4: NUMERICAL FEATURE ANALYSIS
# ==============================================================================
print("\n" + "="*80)
print("Running Cell 4/8: Analyzing Numerical Features")
print("="*80)

if NUMS: # Only run this cell if there are numerical features
    # --- 1. Numerical Feature Distributions ---
    print("--- Distributions of Numerical Features ---")
    
    num_features = len(NUMS)
    num_cols_grid = 3
    num_rows_grid = (num_features + num_cols_grid - 1) // num_cols_grid

    fig, axes = plt.subplots(num_rows_grid, num_cols_grid, figsize=(16, 5 * num_rows_grid))
    axes = axes.flatten()

    for i, col in enumerate(NUMS):
        sns.histplot(train_df[col], ax=axes[i], kde=True, bins=50)
        axes[i].set_title(f'Distribution of {col}', fontsize=12)
        axes[i].set_xlabel('')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(pad=2.0)
    plt.show()

    # --- 2. Correlation Heatmap ---
    print("\n" + "=" * 30)
    print("--- Correlation Matrix of Numerical Features ---")
    print("=" * 30)
    
    # We need to include the target variable in the correlation matrix to see feature-target correlations
    corr_df = train_df[NUMS + [TARGET]]
    corr_matrix = corr_df.corr()

    plt.figure(figsize=(12, 9))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Numerical Features and Target', fontsize=14)
    plt.show()
else:
    print("No numerical features found to analyze.")



skew_values = train_df[NUMS].skew().sort_values(ascending=False)
print(skew_values)


# ==============================================================================
# CELL 5: CATEGORICAL FEATURE ANALYSIS
# ==============================================================================
print("\n" + "="*80)
print("Running Cell 5/8: Analyzing Categorical Features")
print("="*80)

if CATS: # Only run this cell if there are categorical features
    num_features = len(CATS)
    num_cols_grid = 2
    num_rows_grid = (num_features + num_cols_grid - 1) // num_cols_grid

    fig, axes = plt.subplots(num_rows_grid, num_cols_grid, figsize=(16, 6 * num_rows_grid))
    axes = axes.flatten()

    for i, col in enumerate(CATS):
        ax = axes[i]
        
        # A boxplot is ideal for showing the distribution of a numerical target across different categories
        sns.pointplot(x=train_df[col], y=train_df[TARGET], ax=ax)
        
        ax.set_title(f'Distribution of {TARGET} by {col}', fontsize=12)
        ax.set_xlabel('') # Keep the plot clean
        ax.set_ylabel(TARGET)
        
        # If a categorical feature has many unique values, rotate the x-axis labels
        if train_df[col].nunique() > 5:
            ax.tick_params(axis='x', rotation=45)

    # Hide any empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(pad=3.0)
    plt.show()
else:
    print("No categorical features found to analyze.")



# ==============================================================================
# CELL 5: CATEGORICAL FEATURE ANALYSIS (with Point Plots)
# ==============================================================================
print("\n" + "="*80)
print("Running Cell 5/8: Analyzing Categorical Features")
print("="*80)

if CATS: # Only run this cell if there are categorical features
    num_features = len(CATS)
    num_cols_grid = 2
    num_rows_grid = (num_features + num_cols_grid - 1) // num_cols_grid

    fig, axes = plt.subplots(num_rows_grid, num_cols_grid, figsize=(16, 6 * num_rows_grid))
    axes = axes.flatten()

    for i, col in enumerate(CATS):
        ax = axes[i]
        
        # A pointplot is excellent for comparing the mean of the target across different categories.
        # The vertical lines represent the confidence interval around the mean.
        sns.pointplot(x=train_df[col], y=train_df[TARGET], ax=ax, errorbar='ci')
        
        ax.set_title(f'Mean {TARGET} by {col}', fontsize=12)
        ax.set_xlabel('') # Keep the plot clean
        ax.set_ylabel(f'Mean {TARGET}')
        
        # If a categorical feature has many unique values, rotate the x-axis labels
        if train_df[col].nunique() > 5:
            ax.tick_params(axis='x', rotation=45)

    # Hide any empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(pad=3.0)
    plt.show()
else:
    print("No categorical features found to analyze.")

















