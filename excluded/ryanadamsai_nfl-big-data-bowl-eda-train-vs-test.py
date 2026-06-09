"""
NFL Big Data Bowl 2026 - Data Exploration
Let's understand what we're actually predicting!
"""

import pandas as pd
import numpy as np
import os

print("=" * 80)
print("ğŸ�ˆ NFL BIG DATA BOWL 2026 - DATA EXPLORATION ğŸ�ˆ")
print("=" * 80)

# Base path
base_path = '/kaggle/input/nfl-big-data-bowl-2026-prediction'

print("\nğŸ“‚ DATASET STRUCTURE:")
print(f"Base path: {base_path}")

# Check what files exist
if os.path.exists(base_path):
    print("\nâœ… Dataset found!")
    
    # List top-level files
    print("\nğŸ“� Top-level files:")
    top_files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
    for f in sorted(top_files):
        size = os.path.getsize(os.path.join(base_path, f))
        print(f"  - {f} ({size:,} bytes)")
    
    # Check train folder
    train_path = os.path.join(base_path, 'train')
    if os.path.exists(train_path):
        print("\nğŸ“� Train folder files:")
        train_files = os.listdir(train_path)
        input_files = [f for f in train_files if f.startswith('input_')]
        output_files = [f for f in train_files if f.startswith('output_')]
        print(f"  - Input files: {len(input_files)} weeks")
        print(f"  - Output files: {len(output_files)} weeks")

print("\n" + "=" * 80)
print("ğŸ“Š EXAMINING SAMPLE FILES")
print("=" * 80)

# Load sample submission to understand the task
print("\n1ï¸�âƒ£ SAMPLE SUBMISSION:")
sample_sub = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))
print(f"  Shape: {sample_sub.shape}")
print(f"  Columns: {list(sample_sub.columns)}")
print(f"\n  First few rows:")
print(sample_sub.head())
print(f"\n  Sample values:")
print(sample_sub.describe())

# Load test input to see what we're predicting from
print("\n2ï¸�âƒ£ TEST INPUT:")
test_input = pd.read_csv(os.path.join(base_path, 'test_input.csv'))
print(f"  Shape: {test_input.shape}")
print(f"  Columns: {list(test_input.columns)}")
print(f"\n  First few rows:")
print(test_input.head())
print(f"\n  Data types:")
print(test_input.dtypes)

# Load one training input file
print("\n3ï¸�âƒ£ TRAINING INPUT (Week 1):")
train_input = pd.read_csv(os.path.join(base_path, 'train', 'input_2023_w01.csv'))
print(f"  Shape: {train_input.shape}")
print(f"  Columns: {list(train_input.columns)}")
print(f"\n  First few rows:")
print(train_input.head())
print(f"\n  Summary stats:")
print(train_input.describe())

# Load corresponding training output file
print("\n4ï¸�âƒ£ TRAINING OUTPUT (Week 1):")
train_output = pd.read_csv(os.path.join(base_path, 'train', 'output_2023_w01.csv'))
print(f"  Shape: {train_output.shape}")
print(f"  Columns: {list(train_output.columns)}")
print(f"\n  First few rows:")
print(train_output.head())
print(f"\n  Summary stats:")
print(train_output.describe())

# Check if there are any ID columns to link input and output
print("\n5ï¸�âƒ£ LINKING INPUT TO OUTPUT:")
input_cols = set(train_input.columns)
output_cols = set(train_output.columns)
common_cols = input_cols & output_cols
print(f"  Common columns: {common_cols}")

# Check shapes
print(f"\n  Input rows: {len(train_input):,}")
print(f"  Output rows: {len(train_output):,}")
print(f"  Rows match: {len(train_input) == len(train_output)}")

print("\n" + "=" * 80)
print("ğŸ�¯ UNDERSTANDING THE TASK")
print("=" * 80)

# Try to understand what we're predicting
print("\nBased on the data structure:")
print(f"  - We have {len(input_files)} weeks of training data")
print(f"  - Input files contain: {len(train_input.columns)} features")
print(f"  - Output files contain: {len(train_output.columns)} targets")
print(f"  - Sample submission has: {len(sample_sub)} predictions to make")

# Identify prediction targets
non_id_output_cols = [col for col in train_output.columns if col not in ['id', 'gameId', 'playId', 'nflId', 'frameId']]
print(f"\n  Prediction targets: {non_id_output_cols}")

print("\n" + "=" * 80)
print("âœ… DATA EXPLORATION COMPLETE!")
print("=" * 80)

