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

# ============================================
# CONFIGURATION SECTION - EASY TO MODIFY
# ============================================
# Add files and their weights here
# Weights will be automatically normalized to sum to 1.0
ensemble_files = [
    {
        'path': '/kaggle/input/v3-custom-activation-functions/submission_variance_optimized.csv',
        'weight': 10,
        'name': 'variance_optimized'
    },
    {
        'path': '/kaggle/input/ps-s5e9-predicting-the-beats-h-blend-3/submission.csv',
        'weight': 250,  # INCREASED from 90 to 120
        'name': 'beats_h_blend'
    },
    {
        'path': '/kaggle/input/custom-activation-functions-2-0/submission_unified_fixed.csv',
        'weight': 30,
        'name': 'unified_fixed'
    },
    {
        'path': '/kaggle/input/ps-s5e9-blending/submission_final_blend.csv',
        'weight': 50,  # NEW FILE - adjust weight as needed
        'name': 'final_blend'
    },
    # Add more files here as needed:
    # {
    #     'path': '/path/to/another/file.csv',
    #     'weight': 50,
    #     'name': 'another_model'
    # },
]

# Output file path
output_path = 'ensemble_submission.csv'

# ============================================
# ENSEMBLE BLENDING CODE
# ============================================

def blend_submissions(ensemble_files, output_path):
    """
    Blend multiple submission files with specified weights.
    Weights are automatically normalized to sum to 1.0.
    """
    
    # Extract weights and normalize them
    weights = np.array([f['weight'] for f in ensemble_files])
    normalized_weights = weights / weights.sum()
    
    print("="*60)
    print("ENSEMBLE CONFIGURATION")
    print("="*60)
    print(f"Number of files: {len(ensemble_files)}")
    print(f"Raw weights: {weights}")
    print(f"Normalized weights: {normalized_weights}")
    print(f"Sum of normalized weights: {normalized_weights.sum():.10f}")
    print()
    
    # Read all files
    dataframes = []
    for i, file_info in enumerate(ensemble_files):
        print(f"Reading file {i+1}/{len(ensemble_files)}: {file_info['name']}")
        df = pd.read_csv(file_info['path'])
        dataframes.append(df)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Weight (normalized): {normalized_weights[i]:.4f} ({normalized_weights[i]*100:.2f}%)")
        print()
    
    # Validate all dataframes have the same structure
    print("="*60)
    print("VALIDATION")
    print("="*60)
    
    reference_shape = dataframes[0].shape
    reference_columns = list(dataframes[0].columns)
    
    for i, df in enumerate(dataframes[1:], 1):
        if df.shape != reference_shape:
            print(f"WARNING: Shape mismatch in file {i+1}: {df.shape} vs {reference_shape}")
        if list(df.columns) != reference_columns:
            print(f"WARNING: Column mismatch in file {i+1}")
    
    # Identify ID and prediction columns
    id_col = dataframes[0].columns[0]
    pred_cols = [col for col in dataframes[0].columns if col != id_col]
    
    print(f"ID column: {id_col}")
    print(f"Prediction columns: {pred_cols}")
    print()
    
    # Check if all IDs match
    ids_match = True
    reference_ids = dataframes[0][id_col]
    for i, df in enumerate(dataframes[1:], 1):
        if not reference_ids.equals(df[id_col]):
            ids_match = False
            print(f"WARNING: IDs don't match perfectly between file 1 and file {i+1}")
    
    if ids_match:
        print("✓ All ID columns match perfectly")
    print()
    
    # Create result dataframe
    result_df = dataframes[0][[id_col]].copy()
    
    # Perform weighted averaging
    print("="*60)
    print("BLENDING")
    print("="*60)
    
    for col in pred_cols:
        print(f"Blending column: {col}")
        
        # Initialize with zeros
        result_df[col] = 0.0
        
        # Add weighted contributions from each file
        for i, (df, weight) in enumerate(zip(dataframes, normalized_weights)):
            result_df[col] += weight * df[col]
            
            # Print statistics for first prediction column only
            if col == pred_cols[0]:
                print(f"  File {i+1} ({ensemble_files[i]['name']}): "
                      f"weight={weight:.4f}, "
                      f"mean={df[col].mean():.6f}, "
                      f"std={df[col].std():.6f}")
    
    print()
    
    # Save results
    result_df.to_csv(output_path, index=False)
    
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"✓ Ensemble saved to: {output_path}")
    print(f"Shape: {result_df.shape}")
    print()
    print("Summary statistics of blended predictions:")
    print(result_df[pred_cols].describe())
    print()
    print("First 5 rows of blended submission:")
    print(result_df.head())
    
    # Print contribution summary
    print()
    print("="*60)
    print("CONTRIBUTION SUMMARY")
    print("="*60)
    for i, file_info in enumerate(ensemble_files):
        print(f"{file_info['name']:30s}: {normalized_weights[i]*100:6.2f}%")
    print(f"{'TOTAL':30s}: {normalized_weights.sum()*100:6.2f}%")
    
    return result_df

# Run the ensemble blending
blended_df = blend_submissions(ensemble_files, output_path)

