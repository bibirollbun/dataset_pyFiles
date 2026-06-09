# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd

path = '/kaggle/input/stanford-rna-3d-folding/'
# Load the training sequences dataset
train_sequences_df = pd.read_csv(path + 'train_sequences.csv')

# Load the training labels dataset
train_labels_df = pd.read_csv(path + 'train_labels.csv')

# Load the test sequences dataset
test_sequences_df = pd.read_csv(path + 'test_sequences.csv')

# Load the validation sequences dataset
validation_sequences_df = pd.read_csv(path + 'validation_sequences.csv')

# Load the validation labels dataset
validation_labels_df = pd.read_csv(path + 'validation_labels.csv')

# Load the sample submission dataset
sample_submission_df = pd.read_csv(path + 'sample_submission.csv')


train_sequences_df.head(5)


train_labels_df.head(5)


test_sequences_df.head(5)


validation_sequences_df.head(5)


validation_labels_df.head(5)


sample_submission_df.head(5)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D plot of coordinates
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the coordinates
ax.scatter(train_labels_df['x_1'], train_labels_df['y_1'], train_labels_df['z_1'], c='b', marker='o')

# Set labels
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('3D Coordinates of Residues')

plt.show()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D plot of coordinates for the first structure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the coordinates
ax.scatter(validation_labels_df['x_1'], validation_labels_df['y_1'], validation_labels_df['z_1'], c='b', marker='o')

# Set labels
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('3D Coordinates of Residues (First Structure)')

plt.show()


def summarize_sequences(df):
    """Summarize the train_sequences.csv data."""
    summary = {
        'total_sequences': len(df),
        'sequence_lengths': df['sequence'].apply(len).describe(),
        'temporal_cutoffs': df['temporal_cutoff'].describe(),
        'descriptions': df['description'].unique()
    }
    return summary

def summarize_labels(df):
    """Summarize the train_labels.csv data."""
    summary = {
        'total_residues': len(df),
        'residue_names': df['resname'].value_counts(),
        'coordinate_ranges': {
            'x_1': (df['x_1'].min(), df['x_1'].max()),
            'y_1': (df['y_1'].min(), df['y_1'].max()),
            'z_1': (df['z_1'].min(), df['z_1'].max())
        }
    }
    return summary

def summarize_sample_submission(df):
    """Summarize the sample_submission.csv data."""
    summary = {
        'total_residues': len(df),
        'residue_names': df['resname'].value_counts(),
        'coordinate_ranges': {
            'x_1': (df['x_1'].min(), df['x_1'].max()),
            'y_1': (df['y_1'].min(), df['y_1'].max()),
            'z_1': (df['z_1'].min(), df['z_1'].max())
        }
    }
    return summary


import numpy as np

def calculate_d0(L_ref):
    """Calculate the distance scaling factor d0."""
    if L_ref >= 30:
        d0 = 0.6 * (L_ref - 0.5) ** 0.5 - 2.5
    else:
        if L_ref < 12:
            d0 = 0.3
        elif 12 <= L_ref < 15:
            d0 = 0.4
        elif 15 <= L_ref < 19:
            d0 = 0.5
        elif 19 <= L_ref < 23:
            d0 = 0.6
        else:
            d0 = 0.7
    return d0

def calculate_tm_score(L_ref, L_align, distances):
    """Calculate the TM-score."""
    d0 = calculate_d0(L_ref)
    tm_score = np.max(1 / L_ref * np.sum(1 / (1 + (distances / d0) ** 2)))
    return tm_score


# Read the data
train_sequences = train_sequences_df
train_labels = train_labels_df
sample_submission = sample_submission_df

# Summarize the data
sequences_summary = summarize_sequences(train_sequences)
labels_summary = summarize_labels(train_labels)
sample_submission_summary = summarize_sample_submission(sample_submission)

# Print summaries
print("Train Sequences Summary:")
print(sequences_summary)

print("\nTrain Labels Summary:")
print(labels_summary)

print("\nSample Submission Summary:")
print(sample_submission_summary)

# Example TM-score calculation
# Assume we have predictions for the first residue of the first sequence
L_ref = 27  # Length of the reference sequence
L_align = 27  # Number of aligned residues
distances = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7])

# Calculate TM-score
tm_score = calculate_tm_score(L_ref, L_align, distances)
print(f"\nTM-score: {tm_score}")


# copy datasets to new name to match
test_seq = test_sequences_df
sample_sub = sample_submission_df


def generate_positions(sequence_length, shape='linear'):
    """
    Generate a basic RNA shape with the specified number of residues
    """
    if shape == 'linear':
        # Create a simple straight line with evenly spaced residues
        positions = np.zeros((sequence_length, 3))
        for i in range(sequence_length):
            positions[i] = [i * 5.0, 0.0, 0.0]  # 5Å spacing
     
    elif shape == 'circle':
        # Create a circle
        positions = np.zeros((sequence_length, 3))
        radius = sequence_length / (2 * np.pi)  # Adjust radius based on sequence length
        for i in range(sequence_length):
            angle = 2 * np.pi * i / sequence_length
            positions[i] = [radius * np.cos(angle), radius * np.sin(angle), 0.0]
     
    elif shape == 'helix':
        # Create a helix (like A-form RNA)
        positions = np.zeros((sequence_length, 3))
        radius = 10.0  # Radius of helix
        rise_per_residue = 2.8  # Å rise per residue
        residues_per_turn = 11  # ~11 residues per turn for A-form RNA
        
        for i in range(sequence_length):
            angle = 2 * np.pi * i / residues_per_turn
            positions[i] = [
                radius * np.cos(angle), 
                radius * np.sin(angle), 
                i * rise_per_residue
            ]
     
    return positions


def generate_diverse_shapes(sequence, n_conformations=5):
    """Generate diverse RNA shapes for the 5 required conformations"""
    sequence_length = len(sequence)
    conformations = []
    
    # Basic shapes to use
    shapes = ['linear', 'circle', 'helix', 'helix', 'circle']
    
    for i in range(n_conformations):
        shape = shapes[i % len(shapes)]
        
        # Generate basic shape
        coords = generate_positions(sequence_length, shape)
        
        # Apply transformations for additional diversity
        if i > 0:
            # Add some rotation
            angle = np.radians(i * 72)  # 72 degrees = 360/5
            c, s = np.cos(angle), np.sin(angle)
            
            # Rotation matrix
            if i % 3 == 1:
                R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])  # Y-axis
            elif i % 3 == 2:
                R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])  # X-axis
            else:
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])  # Z-axis
            
            # Center, rotate, and translate back
            center = np.mean(coords, axis=0)
            coords = coords - center
            coords = np.dot(coords, R)
            coords = coords + center
            
            # Add some translation
            coords = coords + np.random.normal(0, 5, 3)
        
        conformations.append(coords)
    
    return conformations


print("\n=== Generating Predictions for Submission ===")
submission = sample_sub.copy()

for idx, row in test_seq.iterrows():
    target_id = row['target_id']
    sequence = row['sequence']
    
    print(f"Processing {target_id} (length: {len(sequence)})")
    
    # Generate 5 diverse conformations
    conformations = generate_diverse_shapes(sequence)
    
    # Fill submission with predictions
    for i, conformation in enumerate(conformations):
        # Get rows for this RNA
        mask = submission['ID'].str.startswith(target_id)
        
        # Get sorted indices by residue ID
        sorted_indices = submission.loc[mask].sort_values('resid').index
        
        # Fill coordinates for each residue
        for j, idx in enumerate(sorted_indices):
            if j < len(conformation):
                submission.loc[idx, f'x_{i+1}'] = float(conformation[j][0])
                submission.loc[idx, f'y_{i+1}'] = float(conformation[j][1])
                submission.loc[idx, f'z_{i+1}'] = float(conformation[j][2])
            else:
                # Just in case we have a mismatch - shouldn't happen
                submission.loc[idx, f'x_{i+1}'] = float(j * 5.0)
                submission.loc[idx, f'y_{i+1}'] = 0.0
                submission.loc[idx, f'z_{i+1}'] = 0.0

# Check for any NaN values and fill them
if submission.isna().any().any():
    print("Warning: NaN values detected in submission. Filling with zeros.")
    submission = submission.fillna(0.0)

# Save submission file
submission.to_csv('submission.csv', index=False)
print("\nSaved submission file: submission.csv")

# Display sample of submission
print("\nSubmission preview:")
print(submission.head())


print("\n=== Generating Predictions for Submission ===")
submission = sample_sub.copy()


for idx, row in test_seq.iterrows():
    target_id = row['target_id']
    sequence = row['sequence']
    
    print(f"Processing {target_id} (length: {len(sequence)})")
    
    # Generate 5 diverse conformations
    conformations = generate_diverse_shapes(sequence)


for i, conformation in enumerate(conformations):
    # Get rows for this RNA
    mask = submission['ID'].str.startswith(target_id)
    
    # Get sorted indices by residue ID
    sorted_indices = submission.loc[mask].sort_values('resid').index
    
    # Fill coordinates for each residue
    for j, idx in enumerate(sorted_indices):
        if j < len(conformation):
            submission.loc[idx, f'x_{i+1}'] = float(conformation[j][0])
            submission.loc[idx, f'y_{i+1}'] = float(conformation[j][1])
            submission.loc[idx, f'z_{i+1}'] = float(conformation[j][2])
        else:
            # Just in case we have a mismatch - shouldn't happen
            submission.loc[idx, f'x_{i+1}'] = float(j * 5.0)
            submission.loc[idx, f'y_{i+1}'] = 0.0
            submission.loc[idx, f'z_{i+1}'] = 0.0


if submission.isna().any().any():
    print("Warning: NaN values detected in submission. Filling with zeros.")
    submission = submission.fillna(0.0)


submission.to_csv('submission.csv', index=False)
print("\nSaved submission file: submission.csv")


# Display a sample of the submission
print("\nSubmission preview:")
print(submission.head())

