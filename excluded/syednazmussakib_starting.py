!pip install Bio


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.PDB import PDBParser, PDBIO, Select
import seaborn as sns
from collections import defaultdict
from sklearn.decomposition import PCA
from tqdm import tqdm


# Load data
train_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')
train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
validation_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv')
validation_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_labels.csv')
test_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')

# Basic data exploration
print(f"Train sequences: {train_sequences.shape}")
print(f"Train labels: {train_labels.shape}")
print(f"Validation sequences: {validation_sequences.shape}")
print(f"Validation labels: {validation_labels.shape}")
print(f"Test sequences: {test_sequences.shape}")


# Check for sequence lengths
train_sequences['seq_length'] = train_sequences['sequence'].str.len()
validation_sequences['seq_length'] = validation_sequences['sequence'].str.len()
test_sequences['seq_length'] = test_sequences['sequence'].str.len()

plt.figure(figsize=(10, 6))
sns.histplot(train_sequences['seq_length'], kde=True, label='Train')
sns.histplot(validation_sequences['seq_length'], kde=True, label='Validation')
sns.histplot(test_sequences['seq_length'], kde=True, label='Test')
plt.title('RNA Sequence Length Distribution')
plt.xlabel('Sequence Length')
plt.ylabel('Count')
plt.legend()
plt.savefig('sequence_length_distribution.png')


# Check for nucleotide distribution
def count_nucleotides(sequence):
    return {
        'A': sequence.count('A'),
        'C': sequence.count('C'),
        'G': sequence.count('G'),
        'U': sequence.count('U'),
        'Other': len(sequence) - sequence.count('A') - sequence.count('C') - 
                 sequence.count('G') - sequence.count('U')
    }

train_nucleotides = train_sequences['sequence'].apply(count_nucleotides).apply(pd.Series)
train_nucleotides_norm = train_nucleotides.div(train_sequences['seq_length'], axis=0)

plt.figure(figsize=(10, 6))
sns.boxplot(data=train_nucleotides_norm[['A', 'C', 'G', 'U']])
plt.title('Nucleotide Distribution in Training Data')
plt.ylabel('Proportion')
plt.savefig('nucleotide_distribution.png')



# Check for multiple conformations in validation data
val_conformation_counts = {}
for col in validation_labels.columns:
    if col.startswith('x_') and col != 'x_1':
        val_conformation_counts[col] = (~validation_labels[col].isna()).sum()

plt.figure(figsize=(10, 6))
plt.bar(val_conformation_counts.keys(), val_conformation_counts.values())
plt.title('Number of Residues with Multiple Conformations in Validation Data')
plt.ylabel('Count')
plt.savefig('multiple_conformations.png')


# Temporal distribution analysis
train_sequences['temporal_cutoff'] = pd.to_datetime(train_sequences['temporal_cutoff'])
plt.figure(figsize=(12, 6))
sns.histplot(train_sequences['temporal_cutoff'], kde=True, bins=50)
plt.title('Temporal Distribution of Training Data')
plt.xlabel('Publication Date')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('temporal_distribution.png')


# Create a proper temporal split
casp15_cutoff = pd.to_datetime('2022-05-27')
train_filtered = train_sequences[train_sequences['temporal_cutoff'] < casp15_cutoff]
print(f"Training data before CASP15 cutoff: {train_filtered.shape}")

# Function to extract MSA information
def process_msa(target_id, msa_dir='/kaggle/input/stanford-rna-3d-folding/MSA/'):
    """Process MSA file for a given target_id and return basic statistics."""
    try:
        msa_file = f"{msa_dir}{target_id}.MSA.fasta"
        if not os.path.exists(msa_file):
            return {'seq_count': 0, 'coverage': 0, 'exists': False}
        
        sequences = list(SeqIO.parse(msa_file, "fasta"))
        if not sequences:
            return {'seq_count': 0, 'coverage': 0, 'exists': True}
        
        seq_count = len(sequences)
        
        # Calculate coverage (proportion of non-gap positions)
        alignments = [str(seq.seq) for seq in sequences]
        alignment_length = len(alignments[0])
        coverage = sum(1 for i in range(alignment_length) 
                      if any(ali[i] != '-' for ali in alignments)) / alignment_length
        
        return {'seq_count': seq_count, 'coverage': coverage, 'exists': True}
    except Exception as e:
        print(f"Error processing MSA for {target_id}: {e}")
        return {'seq_count': 0, 'coverage': 0, 'exists': False, 'error': str(e)}

# Sample MSA processing on a few training examples
sample_targets = train_sequences['target_id'].sample(min(10, len(train_sequences))).tolist()
msa_stats = {target: process_msa(target) for target in sample_targets}


print("\nMSA Statistics for Sample Targets:")
for target, stats in msa_stats.items():
    print(f"{target}: {stats}")


# Process coordinates to create a standard format
def extract_coordinates(df, num_structures=1):
    """Extract coordinates from the dataframe and return as a numpy array."""
    coordinates = []
    for i in range(1, num_structures + 1):
        if f'x_{i}' in df.columns:
            struct_coords = df[[f'x_{i}', f'y_{i}', f'z_{i}']].values
            if not np.isnan(struct_coords).any():
                coordinates.append(struct_coords)
    return np.array(coordinates)

# Process a sample target to demonstrate coordinate extraction
sample_target_id = train_sequences['target_id'].iloc[0]
sample_target_residues = train_labels[train_labels['ID'].str.startswith(f"{sample_target_id}_")]
sample_coords = extract_coordinates(sample_target_residues)


# Visualize 3D structure of a sample target
if len(sample_coords) > 0:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sample_coords[0][:, 0], sample_coords[0][:, 1], sample_coords[0][:, 2], 'o-')
    ax.set_title(f'3D Structure of {sample_target_id}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig('sample_structure_3d.png')

# Deduplicate sequences and merge structures
sequence_to_structures = defaultdict(list)
for idx, row in train_sequences.iterrows():
    target_id = row['target_id']
    sequence = row['sequence']
    residues = train_labels[train_labels['ID'].str.startswith(f"{target_id}_")]
    coords = extract_coordinates(residues)
    if len(coords) > 0:
        sequence_to_structures[sequence].append((target_id, coords))

# Show duplicates
duplicate_sequences = [seq for seq, structs in sequence_to_structures.items() if len(structs) > 1]
print(f"\nNumber of duplicate sequences: {len(duplicate_sequences)}")
if duplicate_sequences:
    print(f"Example duplicate: {duplicate_sequences[0]}")
    print(f"Number of structures: {len(sequence_to_structures[duplicate_sequences[0]])}")

# Function to prepare data for model input
def prepare_model_input(sequences_df, labels_df, msa_dir='/kaggle/input/stanford-rna-3d-folding/MSA/'):
    """Prepare features and labels for model training."""
    data = []
    
    for idx, row in tqdm(sequences_df.iterrows(), total=len(sequences_df)):
        target_id = row['target_id']
        sequence = row['sequence']
        
        # Skip sequences with non-standard nucleotides
        if any(c not in 'ACGU' for c in sequence):
            continue
        
        # One-hot encode sequence
        seq_onehot = []
        for c in sequence:
            if c == 'A':
                seq_onehot.append([1, 0, 0, 0])
            elif c == 'C':
                seq_onehot.append([0, 1, 0, 0])
            elif c == 'G':
                seq_onehot.append([0, 0, 1, 0])
            elif c == 'U':
                seq_onehot.append([0, 0, 0, 1])
            else:
                seq_onehot.append([0.25, 0.25, 0.25, 0.25])
        
        # Extract coordinates
        residues = labels_df[labels_df['ID'].str.startswith(f"{target_id}_")]
        coords = extract_coordinates(residues)
        
        # Process MSA (basic features)
        msa_stats = process_msa(target_id, msa_dir)
        
        # Store data
        data.append({
            'target_id': target_id,
            'sequence': sequence,
            'sequence_onehot': np.array(seq_onehot),
            'coordinates': coords,
            'msa_stats': msa_stats
        })
    
    return data

# Prepare a small sample of data
sample_size = min(10, len(train_sequences))
sample_data = prepare_model_input(
    train_sequences.head(sample_size),
    train_labels
)

print(f"\nSample data prepared: {len(sample_data)} entries")
if sample_data:
    print(f"Example entry: target_id={sample_data[0]['target_id']}, " +
          f"sequence length={len(sample_data[0]['sequence'])}, " +
          f"coordinates shape={sample_data[0]['coordinates'].shape if len(sample_data[0]['coordinates']) > 0 else 'None'}")




