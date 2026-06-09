# SIIM-ISIC Melanoma Classification - EDA Setup
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
sns.set_style('whitegrid')

# Define paths
base_path = Path('/kaggle/input/siim-isic-melanoma-classification')

print("Available files in the dataset:")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Load metadata CSV
train_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')

print("\n=== Dataset Overview ===")
print(f"Dataset shape: {train_df.shape}")
print(f"Number of rows: {train_df.shape[0]}")
print(f"Number of columns: {train_df.shape[1]}")
print("\nColumn names and types:")
print(train_df.dtypes)
print("\nFirst few rows:")
print(train_df.head())


# Class Distribution Analysis
print("\n=== Class Distribution ===")
class_counts = train_df['target'].value_counts()
print(f"\nClass counts:")
print(class_counts)
print(f"\nClass percentages:")
print(train_df['target'].value_counts(normalize=True) * 100)

# Calculate class imbalance ratio
imbalance_ratio = class_counts[0] / class_counts[1]
print(f"\nClass imbalance ratio (negative/positive): {imbalance_ratio:.2f}:1")
print(f"Positive class (melanoma): {class_counts[1]} ({(class_counts[1]/len(train_df)*100):.2f}%)")
print(f"Negative class (benign): {class_counts[0]} ({(class_counts[0]/len(train_df)*100):.2f}%)")


# Data Quality Checks
print("\n=== Data Quality Assessment ===")

# Check for missing values
print("\nMissing values per column:")
missing_vals = train_df.isnull().sum()
print(missing_vals[missing_vals > 0])
print(f"\nTotal missing values: {train_df.isnull().sum().sum()}")
print(f"Percentage of missing data: {(train_df.isnull().sum().sum() / (train_df.shape[0] * train_df.shape[1]) * 100):.2f}%")

# Check for duplicates
print(f"\nDuplicate rows: {train_df.duplicated().sum()}")
print(f"Duplicate image_names: {train_df['image_name'].duplicated().sum()}")
if train_df['image_name'].duplicated().sum() > 0:
    print("\nSample duplicate image_names:")
    print(train_df[train_df['image_name'].duplicated(keep=False)].sort_values('image_name').head(10))


# Visualization Setup - Class Distribution and Demographics
print("\n=== Preparing Visualizations ===")

# Setup for visualizations
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Class count bar chart
ax1 = axes[0]
class_counts.plot(kind='bar', ax=ax1, color=['skyblue', 'salmon'])
ax1.set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Target Class', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_xticklabels(['Benign (0)', 'Melanoma (1)'], rotation=0)
ax1.grid(axis='y', alpha=0.3)

# Add count labels on bars
for i, v in enumerate(class_counts):
    ax1.text(i, v + 1000, str(v), ha='center', va='bottom', fontweight='bold')

# Plot 2: Demographics analysis (age if available)
ax2 = axes[1]
if 'age_approx' in train_df.columns:
    train_df['age_approx'].hist(bins=30, ax=ax2, color='steelblue', edgecolor='black')
    ax2.set_title('Age Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Age (Approximate)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
else:
    ax2.text(0.5, 0.5, 'Age data not available', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Demographics', fontsize=14, fontweight='bold')

plt.tight_layout()
print("\nVisualization setup complete. Ready for EDA execution.")

