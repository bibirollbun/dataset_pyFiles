# Standard Library Imports
import datetime
import gc
import hashlib
import json
import os
import random
import time
import traceback
import warnings
from collections import Counter

# Scientific Computing and Numerical Libraries
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.svm import SVR

# Visualization Libraries
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# File paths
DATA_DIR = "/kaggle/input/stanford-rna-3d-folding/"
OUTPUT_DIR = "/kaggle/working/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("All libraries imported successfully!")


def load_data():
    """
    Loads the necessary data for the competition.
    """
    data = {}
    
    # Load sequences
    data['train_seq'] = pd.read_csv(os.path.join(DATA_DIR, "train_sequences.csv"))
    data['valid_seq'] = pd.read_csv(os.path.join(DATA_DIR, "validation_sequences.csv"))
    data['test_seq'] = pd.read_csv(os.path.join(DATA_DIR, "test_sequences.csv"))
    
    # Load structures (labels)
    data['train_labels'] = pd.read_csv(os.path.join(DATA_DIR, "train_labels.csv"))
    data['valid_labels'] = pd.read_csv(os.path.join(DATA_DIR, "validation_labels.csv"))
    
    # Load submission format
    data['sample_submission'] = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    
    return data

def explore_data(data_dict):
    """
    Provides a basic exploration of the dataset.
    """
    print("Data Exploration Summary:")
    print("-" * 50)
    
    # Explore sequences data
    for key in ['train_seq', 'valid_seq', 'test_seq']:
        if key in data_dict:
            print(f"{key}: {data_dict[key].shape[0]} sequences")
            if data_dict[key].shape[0] > 0:
                print(f"  - Example target_id: {data_dict[key]['target_id'].iloc[0]}")
                print(f"  - Example sequence: {data_dict[key]['sequence'].iloc[0][:20]}...")
                print(f"  - Sequence length range: {data_dict[key]['sequence'].apply(len).min()} to {data_dict[key]['sequence'].apply(len).max()}")
    
    # Explore label data
    for key in ['train_labels', 'valid_labels']:
        if key in data_dict:
            print(f"{key}: {data_dict[key].shape[0]} positions")
            if data_dict[key].shape[0] > 0:
                print(f"  - Example ID: {data_dict[key]['ID'].iloc[0]}")
                print(f"  - Example residue: {data_dict[key]['resname'].iloc[0]}")
                # Check for coordinate columns
                coord_columns = [col for col in data_dict[key].columns if col.startswith('x_') or col.startswith('y_') or col.startswith('z_')]
                print(f"  - Number of coordinate sets: {len(coord_columns) // 3}")
    
    # Explore submission format
    if 'sample_submission' in data_dict:
        print(f"sample_submission: {data_dict['sample_submission'].shape[0]} rows")
        print(f"  - Columns: {', '.join(data_dict['sample_submission'].columns)}")
    
    return

# Load the data
print("Loading data...")
data_dict = load_data()
explore_data(data_dict)


def visualize_sequence_distribution(data_dict):
    """
    Visualizes the distribution of sequence lengths and nucleotide compositions.
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Colors for nucleotides
    colors = {'A': '#3498db', 'C': '#2ecc71', 'G': '#e74c3c', 'U': '#9b59b6', 'N': '#95a5a6'}
    
    # Datasets to analyze
    datasets = ['train_seq', 'valid_seq', 'test_seq']
    
    # Plot sequence length distributions
    sequence_lengths = {}
    for i, dataset_name in enumerate(datasets):
        if dataset_name in data_dict:
            df = data_dict[dataset_name]
            sequence_lengths[dataset_name] = df['sequence'].apply(len)
            
            # Plot length distribution
            sns.histplot(sequence_lengths[dataset_name], ax=axes[0], alpha=0.3, label=dataset_name)
    
    axes[0].set_title('Distribution of RNA Sequence Lengths', fontsize=12)
    axes[0].set_xlabel('Sequence Length (nucleotides)', fontsize=10)
    axes[0].set_ylabel('Count', fontsize=10)
    axes[0].legend()
    
    # Calculate nucleotide composition across all datasets
    all_nucleotides = []
    for dataset_name in datasets:
        if dataset_name in data_dict:
            all_nucleotides.extend(''.join(data_dict[dataset_name]['sequence'].tolist()))
    
    nucleotide_counts = Counter(all_nucleotides)
    
    # Plot nucleotide composition
    nucleotides = ['A', 'C', 'G', 'U', 'N']
    counts = [nucleotide_counts.get(n, 0) for n in nucleotides]
    axes[1].bar(nucleotides, counts, color=[colors.get(n, '#cccccc') for n in nucleotides])
    axes[1].set_title('Nucleotide Composition', fontsize=12)
    axes[1].set_xlabel('Nucleotide', fontsize=10)
    axes[1].set_ylabel('Count', fontsize=10)
    
    # GC content distribution
    gc_content = {}
    for dataset_name in datasets:
        if dataset_name in data_dict:
            df = data_dict[dataset_name]
            gc_content[dataset_name] = df['sequence'].apply(lambda s: (s.count('G') + s.count('C')) / len(s) if len(s) > 0 else 0)
            
            # Plot GC content distribution
            sns.kdeplot(gc_content[dataset_name], ax=axes[2], label=dataset_name)
    
    axes[2].set_title('GC Content Distribution', fontsize=12)
    axes[2].set_xlabel('GC Content', fontsize=10)
    axes[2].set_ylabel('Density', fontsize=10)
    axes[2].legend()
    
    # Plot average nucleotide composition per position (for first 50 positions)
    position_data = []
    max_length = 50
    
    for dataset_name in datasets:
        if dataset_name in data_dict:
            df = data_dict[dataset_name]
            
            for i, seq in enumerate(df['sequence']):
                seq = seq[:max_length]  # Limit to first 50 positions
                for pos, nucleotide in enumerate(seq):
                    position_data.append({
                        'Dataset': dataset_name,
                        'Position': pos + 1,
                        'Nucleotide': nucleotide
                    })
    
    position_df = pd.DataFrame(position_data)
    
    # Count by position and nucleotide
    position_counts = position_df.groupby(['Position', 'Nucleotide']).size().unstack(fill_value=0)
    
    # Calculate percentages
    position_percentages = position_counts.div(position_counts.sum(axis=1), axis=0) * 100
    
    # Plot stacked bar chart
    position_percentages.plot(kind='bar', stacked=True, ax=axes[3], 
                             color=[colors.get(n, '#cccccc') for n in position_percentages.columns])
    
    axes[3].set_title('Nucleotide Composition by Position (first 50)', fontsize=12)
    axes[3].set_xlabel('Position', fontsize=10)
    axes[3].set_ylabel('Percentage (%)', fontsize=10)
    axes[3].legend(title='Nucleotide')
    
    # Show only some ticks to avoid overcrowding
    if len(position_percentages) > 10:
        show_ticks = list(range(0, len(position_percentages), 5))
        axes[3].set_xticks(show_ticks)
        axes[3].set_xticklabels([str(i+1) for i in show_ticks])
    
    plt.tight_layout()
    plt.show()
    
    return

# Visualize sequence distribution
visualize_sequence_distribution(data_dict)


def visualize_coordinates(data_dict, sequence_id=None, structure_idx=1):
    """
    Visualizes 3D coordinates for a specific RNA structure.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing the datasets
    sequence_id : str, optional
        ID of the sequence to visualize. If None, the first sequence is used.
    structure_idx : int, optional
        Index of the structure to visualize (1-5 for valid_labels)
    """
    # Use validation data for visualization
    if 'valid_labels' not in data_dict or 'valid_seq' not in data_dict:
        print("Validation data not available for visualization.")
        return
    
    valid_labels = data_dict['valid_labels']
    valid_seq = data_dict['valid_seq']
    
    # Get unique sequence IDs from validation labels
    seq_ids = set([id_str.split('_')[0] for id_str in valid_labels['ID']])
    
    if sequence_id is None:
        # Use the first sequence ID if none is specified
        sequence_id = list(seq_ids)[0]
    elif sequence_id not in seq_ids:
        print(f"Sequence ID {sequence_id} not found in validation data.")
        return
    
    # Filter labels for the specified sequence
    seq_labels = valid_labels[valid_labels['ID'].str.startswith(f"{sequence_id}_")]
    
    # Get the sequence
    seq_row = valid_seq[valid_seq['target_id'] == sequence_id]
    if len(seq_row) == 0:
        print(f"Sequence {sequence_id} not found in validation sequences.")
        return
    
    sequence = seq_row['sequence'].iloc[0]
    print(f"Visualizing structure for sequence {sequence_id}")
    print(f"Sequence: {sequence[:20]}... (length: {len(sequence)})")
    
    # Check if the structure index is valid
    coord_cols = [f'x_{structure_idx}', f'y_{structure_idx}', f'z_{structure_idx}']
    if not all(col in seq_labels.columns for col in coord_cols):
        print(f"Structure {structure_idx} not available for sequence {sequence_id}.")
        return
    
    # Extract coordinates for the specified structure
    coords = seq_labels[coord_cols].values
    
    # Check if coordinates are valid (not NaN or extremely large values)
    valid_coords = ~np.any(np.abs(coords) > 1e10, axis=1) & ~np.any(np.isnan(coords), axis=1)
    
    if np.sum(valid_coords) == 0:
        print(f"No valid coordinates found for structure {structure_idx} of sequence {sequence_id}.")
        return
    
    coords = coords[valid_coords]
    
    # Get residue names for coloring
    residues = seq_labels['resname'].values[valid_coords]
    
    # Create a color map based on residue type
    color_map = {'A': 'blue', 'C': 'green', 'G': 'red', 'U': 'purple'}
    colors = [color_map.get(res, 'gray') for res in residues]
    
    # Create 3D visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot structure as line to show the backbone
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 'gray', alpha=0.7, linewidth=1)
    
    # Plot nucleotides as colored points
    scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                         c=[i for i in range(len(coords))], 
                         cmap='viridis', 
                         s=50, alpha=0.8)
    
    # Add colorbar to show sequence position
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Sequence Position')
    
    # Set labels and title
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Z coordinate')
    ax.set_title(f'3D Structure of RNA Sequence {sequence_id} (Structure {structure_idx})')
    
    # Show stats about the structure
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    range_coords = max_coords - min_coords
    
    print(f"Coordinate ranges: X: {range_coords[0]:.2f}, Y: {range_coords[1]:.2f}, Z: {range_coords[2]:.2f}")
    print(f"Number of valid coordinates: {len(coords)} out of {len(seq_labels)}")
    
    plt.tight_layout()
    plt.show()
    
    return coords, sequence

# Visualize 3D coordinates for a sample structure
visualize_coordinates(data_dict)


def analyze_id_structure(data_dict):
    """
    Analyzes the ID structure in different files to understand the correct mapping.
    """
    # Analysis of training labels
    train_label_ids = data_dict['train_labels']['ID'].tolist() if 'train_labels' in data_dict else []
    print(f"Total IDs in training labels: {len(train_label_ids)}")
    print(f"Number of unique IDs: {len(set(train_label_ids))}")
    
    # Try to understand the ID format in the labels file
    train_id_parts = {}
    for id_str in train_label_ids[:100]:  # Analyze the first 100
        parts = id_str.split('_')
        num_parts = len(parts)
        if num_parts not in train_id_parts:
            train_id_parts[num_parts] = []
        train_id_parts[num_parts].append(parts)
    
    print("\nID formats found in train_labels:")
    for num_parts, examples in train_id_parts.items():
        print(f"\nFormat with {num_parts} parts:")
        for i, parts in enumerate(examples[:3]):
            print(f"  Example {i+1}: {parts}")
    
    # Analysis of training sequences
    train_seq_ids = data_dict['train_seq']['target_id'].tolist() if 'train_seq' in data_dict else []
    print(f"\nTotal IDs in training sequences: {len(train_seq_ids)}")
    print(f"Number of unique IDs: {len(set(train_seq_ids))}")
    
    # Try to understand the ID format in the sequences file
    train_seq_id_parts = {}
    for id_str in train_seq_ids[:100]:  # Analyze the first 100
        parts = id_str.split('_')
        num_parts = len(parts)
        if num_parts not in train_seq_id_parts:
            train_seq_id_parts[num_parts] = []
        train_seq_id_parts[num_parts].append(parts)
    
    print("\nID formats found in train_sequences:")
    for num_parts, examples in train_seq_id_parts.items():
        print(f"\nFormat with {num_parts} parts:")
        for i, parts in enumerate(examples[:3]):
            print(f"  Example {i+1}: {parts}")
    
    # Analysis of validation labels
    valid_label_ids = data_dict['valid_labels']['ID'].tolist() if 'valid_labels' in data_dict else []
    print(f"\nTotal IDs in validation labels: {len(valid_label_ids)}")
    print(f"Number of unique IDs: {len(set(valid_label_ids))}")
    
    # Count unique sequence IDs in validation labels
    valid_seq_ids_from_labels = set([id_str.split('_')[0] for id_str in valid_label_ids])
    print(f"Number of unique sequence IDs in validation labels: {len(valid_seq_ids_from_labels)}")
    print(f"Examples: {list(valid_seq_ids_from_labels)[:5]}")
    
    # Analysis of validation sequences
    valid_seq_ids = data_dict['valid_seq']['target_id'].tolist() if 'valid_seq' in data_dict else []
    print(f"\nTotal IDs in validation sequences: {len(valid_seq_ids)}")
    print(f"Number of unique IDs: {len(set(valid_seq_ids))}")
    print(f"Examples: {valid_seq_ids[:5]}")
    
    # Check correspondence between unique IDs
    overlap_valid = set(valid_seq_ids).intersection(valid_seq_ids_from_labels)
    print(f"\nCorrespondence between validation sequences and labels: {len(overlap_valid)} of {len(valid_seq_ids)}")
    
    # Check how sequences and residues relate
    if len(overlap_valid) > 0:
        sample_id = list(overlap_valid)[0]
        sample_seq = data_dict['valid_seq'][data_dict['valid_seq']['target_id'] == sample_id]['sequence'].iloc[0]
        sample_labels = data_dict['valid_labels'][data_dict['valid_labels']['ID'].str.startswith(f"{sample_id}_")]
        
        print(f"\nAnalysis for sequence ID: {sample_id}")
        print(f"Sequence length: {len(sample_seq)}")
        print(f"Number of residues in labels: {len(sample_labels)}")
        
        # Check how residue numbers are related
        residue_numbers = sample_labels['resid'].sort_values().tolist()
        print(f"First residue numbers: {residue_numbers[:10]}")
        print(f"Last residue numbers: {residue_numbers[-10:]}")
    
    return train_id_parts, train_seq_id_parts, overlap_valid

# Analyze ID structure
train_id_parts, train_seq_id_parts, overlap_valid = analyze_id_structure(data_dict)


def create_mapping_valid(valid_seq_df, valid_labels_df):
    """
    Creates a mapping between validation sequences and their coordinates.
    In this case, the IDs already correspond directly (e.g., R1107 -> R1107_1, R1107_2, etc.)
    """
    # Check which ID format is used in the validation set
    valid_labels_df['seq_id'] = valid_labels_df['ID'].apply(lambda x: x.split('_')[0])
    
    # Check overlap
    seq_ids = set(valid_seq_df['target_id'])
    label_seq_ids = set(valid_labels_df['seq_id'])
    
    overlap = seq_ids.intersection(label_seq_ids)
    print(f"Correspondence for validation: {len(overlap)} of {len(seq_ids)}")
    
    mapping = {}
    for seq_id in overlap:
        # Get sequence
        seq = valid_seq_df[valid_seq_df['target_id'] == seq_id]['sequence'].iloc[0]
        
        # Get all residues for this sequence
        residues = valid_labels_df[valid_labels_df['seq_id'] == seq_id].sort_values('resid')
        
        # Extract coordinates for all structures
        num_structures = 1
        for col in residues.columns:
            if col.startswith('x_'):
                struct_num = int(col.split('_')[1])
                num_structures = max(num_structures, struct_num)
        
        # Initialize structures
        structures = []
        for struct_idx in range(1, num_structures + 1):
            coords = []
            has_valid_coords = False
            
            # Check if this structure has coordinates
            if f'x_{struct_idx}' in residues.columns:
                for _, row in residues.iterrows():
                    x = row[f'x_{struct_idx}']
                    y = row[f'y_{struct_idx}']
                    z = row[f'z_{struct_idx}']
                    
                    # Check if they are valid values
                    if not (np.isnan(x) or np.isnan(y) or np.isnan(z) or 
                            abs(x) > 1e10 or abs(y) > 1e10 or abs(z) > 1e10):
                        coords.append([x, y, z])
                        has_valid_coords = True
                    else:
                        coords.append([np.nan, np.nan, np.nan])
            
            if has_valid_coords:
                structures.append(np.array(coords))
        
        # Add to mapping if there are valid structures
        if structures:
            mapping[seq_id] = {
                'sequence': seq,
                'structures': structures
            }
    
    print(f"Mapping created with {len(mapping)} valid sequences")
    return mapping

# Create mapping for validation set
valid_mapping = create_mapping_valid(data_dict['valid_seq'], data_dict['valid_labels'])


def explore_sequence_mapping(seq_id, mapping, data_dict):
    """
    Explores a mapping example in detail for diagnostics.
    """
    if seq_id not in mapping:
        print(f"WARNING: Sequence ID {seq_id} not found in mapping")
        return
    
    data = mapping[seq_id]
    seq = data['sequence']
    structures = data['structures']
    
    print(f"Exploring mapping for sequence: {seq_id}")
    print(f"Sequence length: {len(seq)}")
    print(f"Number of available structures: {len(structures)}")
    
    # Detail each structure
    for i, structure in enumerate(structures):
        print(f"\nStructure {i+1}:")
        print(f"  Number of coordinates: {len(structure)}")
        if len(structure) > 0:
            print(f"  First coordinates: {structure[:3]}")
            print(f"  Last coordinates: {structure[-3:]}")
        
        # Check correspondence with the sequence
        if len(structure) != len(seq):
            print(f"  WARNING: Difference between sequence length ({len(seq)}) and coordinates ({len(structure)})")
        else:
            print(f"  Perfect match between sequence and coordinates")

# Explore a mapping example
if valid_mapping:
    sample_id = list(valid_mapping.keys())[0]
    explore_sequence_mapping(sample_id, valid_mapping, data_dict)


def encode_sequence(sequence):
    """
    One-hot encodes an RNA sequence.
    
    Parameters:
    -----------
    sequence : str
        RNA sequence
        
    Returns:
    --------
    numpy.ndarray
        One-hot encoded sequence with shape (len(sequence), 5)
    """
    # Create a mapping for nucleotides
    nucleotide_map = {
        'A': [1, 0, 0, 0, 0],
        'C': [0, 1, 0, 0, 0],
        'G': [0, 0, 1, 0, 0],
        'U': [0, 0, 0, 1, 0],
        'T': [0, 0, 0, 1, 0],  # Treat T as U
        'N': [0, 0, 0, 0, 1]   # Unknown nucleotide
    }
    
    # Encode each nucleotide
    encoded = []
    for nucleotide in sequence:
        encoded.append(nucleotide_map.get(nucleotide, [0, 0, 0, 0, 1]))  # Default to N if not recognized
    
    return np.array(encoded)


def extract_sequence_features(sequence):
    """
    Extracts various features from an RNA sequence.
    
    Parameters:
    -----------
    sequence : str
        RNA sequence
        
    Returns:
    --------
    dict
        Dictionary of features
    """
    # Basic composition features
    total_length = len(sequence)
    a_count = sequence.count('A')
    c_count = sequence.count('C')
    g_count = sequence.count('G')
    u_count = sequence.count('U')
    n_count = total_length - (a_count + c_count + g_count + u_count)
    
    # Calculate percentages
    a_percent = a_count / total_length if total_length > 0 else 0
    c_percent = c_count / total_length if total_length > 0 else 0
    g_percent = g_count / total_length if total_length > 0 else 0
    u_percent = u_count / total_length if total_length > 0 else 0
    n_percent = n_count / total_length if total_length > 0 else 0
    
    # Calculate other ratios
    gc_content = (g_count + c_count) / total_length if total_length > 0 else 0
    au_content = (a_count + u_count) / total_length if total_length > 0 else 0
    ga_content = (g_count + a_count) / total_length if total_length > 0 else 0
    cu_content = (c_count + u_count) / total_length if total_length > 0 else 0
    
    # Dinucleotide composition
    dinucleotides = {}
    for i in range(len(sequence) - 1):
        dinuc = sequence[i:i+2]
        dinucleotides[dinuc] = dinucleotides.get(dinuc, 0) + 1
    
    # Normalize by sequence length
    for dinuc in dinucleotides:
        dinucleotides[dinuc] = dinucleotides[dinuc] / (total_length - 1) if (total_length - 1) > 0 else 0
    
    # Potential base pairing regions
    # Simple heuristic: look for reverse complementary regions
    complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    potential_pairs = 0
    
    # Look for possible stem-loop structures (simple heuristic)
    for i in range(len(sequence)):
        for j in range(i + 4, len(sequence)):  # minimum 4 nucleotides apart
            if j - i <= 30:  # maximum distance of 30 nucleotides
                if sequence[i] in complement and sequence[j] == complement[sequence[i]]:
                    potential_pairs += 1
    
    pairing_density = potential_pairs / total_length if total_length > 0 else 0
    
    # Return as a dictionary
    features = {
        'length': total_length,
        'a_count': a_count,
        'c_count': c_count,
        'g_count': g_count,
        'u_count': u_count,
        'n_count': n_count,
        'a_percent': a_percent,
        'c_percent': c_percent,
        'g_percent': g_percent,
        'u_percent': u_percent,
        'n_percent': n_percent,
        'gc_content': gc_content,
        'au_content': au_content,
        'ga_content': ga_content,
        'cu_content': cu_content,
        'pairing_density': pairing_density
    }
    
    # Add dinucleotide features
    for dinuc, value in dinucleotides.items():
        features[f'dinuc_{dinuc}'] = value
    
    return features

def create_processed_data(mapping, output_prefix):
    """
    Creates and saves processed data from the mapping.
    
    Parameters:
    -----------
    mapping: Dictionary with the mapping of sequences to structures
    output_prefix: Prefix for output files ('train' or 'valid')
    
    Returns:
    --------
    X, y, sequence_data: Arrays and metadata for training
    """
    if not mapping:
        print(f"WARNING: No valid mapping for {output_prefix}")
        return None, None, {}
    
    X_data = []
    y_data = []
    ids = []
    sequence_data = {}
    
    for seq_id, data in mapping.items():
        seq = data['sequence']
        structures = data['structures']
        
        # Skip if there are no structures
        if not structures:
            continue
        
        # Use the first valid structure
        structure = structures[0]
        
        # Check if the structure has valid coordinates for all residues
        if len(structure) != len(seq):
            print(f"WARNING: Difference between sequence length ({len(seq)}) and coordinates ({len(structure)}) for {seq_id}")
            continue
        
        # Extract sequence features
        seq_features = extract_sequence_features(seq)
        
        # One-hot encode the sequence
        encoded_seq = encode_sequence(seq)
        
        # Store in arrays
        X_data.append(encoded_seq)
        y_data.append(structure)
        ids.append(seq_id)
        
        # Store additional data
        sequence_data[seq_id] = {
            'sequence': seq,
            'features': seq_features,
            'structure': structure
        }
    
    if not X_data:
        print(f"WARNING: No valid processed data for {output_prefix}")
        return None, None, {}
    
    # Save the processed data
    X = np.array(X_data, dtype=object)
    y = np.array(y_data, dtype=object)
    
    # Save to files
    np.save(os.path.join(OUTPUT_DIR, f'X_{output_prefix}.npy'), X, allow_pickle=True)
    np.save(os.path.join(OUTPUT_DIR, f'y_{output_prefix}.npy'), y, allow_pickle=True)
    with open(os.path.join(OUTPUT_DIR, f'{output_prefix}_ids.txt'), 'w') as f:
        for id in ids:
            f.write(f"{id}\n")
            
    # Save sequence data for future reference
    with open(os.path.join(OUTPUT_DIR, f'{output_prefix}_sequence_data.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        for seq_id, data in sequence_data.items():
            json_data[seq_id] = {
                'sequence': data['sequence'],
                'features': data['features']
                # Skip structure as it's already saved in y
            }
        json.dump(json_data, f)
    
    print(f"Processed data for {output_prefix}: {len(X)} sequences")
    return X, y, sequence_data

# Create processed validation data
X_valid, y_valid, valid_sequence_data = create_processed_data(valid_mapping, 'valid')

# Since we're using the validation set for both training and testing (due to data structure)
X_train, y_train, train_sequence_data = X_valid, y_valid, valid_sequence_data


def create_unified_features(sequence_data):
    """
    Creates a unified feature matrix from sequence_data dictionary.
    
    Parameters:
    -----------
    sequence_data: Dictionary with sequence features
    
    Returns:
    --------
    X_features: Feature matrix (n_samples, n_features)
    feature_names: Names of the features
    """
    feature_records = []
    for seq_id, data in sequence_data.items():
        feature_records.append(data['features'])
    
    # Convert to DataFrame for easier handling
    feature_df = pd.DataFrame(feature_records)
    
    # Fill any missing values
    feature_df = feature_df.fillna(0)
    
    # Return as numpy array
    X_features = feature_df.values
    feature_names = feature_df.columns.tolist()
    
    return X_features, feature_names

def create_coordinate_targets(y_data, coordinate_idx=0):
    """
    Creates target arrays for each coordinate dimension.
    
    Parameters:
    -----------
    y_data: List of structure arrays
    coordinate_idx: Index of coordinate to predict (0=x, 1=y, 2=z)
    
    Returns:
    --------
    y_coord: Array of coordinates for the specified dimension
    """
    y_coord = []
    for structure in y_data:
        y_coord.append(structure[:, coordinate_idx])
    
    # Convert to ragged array
    y_coord = np.array(y_coord, dtype=object)
    
    return y_coord

# Create unified feature matrix
X_features, feature_names = create_unified_features(train_sequence_data)

# Create target arrays for each coordinate dimension
y_coord_x = create_coordinate_targets(y_train, 0)
y_coord_y = create_coordinate_targets(y_train, 1)
y_coord_z = create_coordinate_targets(y_train, 2)

print(f"Feature matrix shape: {X_features.shape}")
print(f"Number of features: {len(feature_names)}")
print(f"Example features: {feature_names[:10]}")


def visualize_feature_correlations(X_features, feature_names):
    """
    Visualizes correlations between sequence features.
    """
    # Convert to DataFrame for easier handling
    feature_df = pd.DataFrame(X_features, columns=feature_names)
    
    # Select a subset of features to visualize (to avoid overcrowding)
    # Focus on important features like composition percentages and ratios
    selected_features = [
        'length', 'a_percent', 'c_percent', 'g_percent', 'u_percent',
        'gc_content', 'au_content', 'pairing_density'
    ]
    
    # Add some dinucleotide features if available
    for feature in feature_names:
        if feature.startswith('dinuc_') and len(selected_features) < 15:
            selected_features.append(feature)
    
    # Calculate correlation matrix
    correlation = feature_df[selected_features].corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    return correlation

# Visualize feature correlations
correlation = visualize_feature_correlations(X_features, feature_names)


def visualize_pca_projection(X_features, feature_names, sequence_data):
    """
    Visualizes the PCA projection of the feature space.
    """
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_features)
    
    # Get lengths for coloring
    lengths = [data['features']['length'] for data in sequence_data.values()]
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=lengths, cmap='viridis', alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Sequence Length')
    
    # Add labels and title
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA Projection of RNA Sequence Features')
    
    # Add feature loadings
    loadings = pca.components_.T
    
    for i, feature in enumerate(feature_names):
        if feature in ['length', 'gc_content', 'au_content', 'pairing_density']:
            plt.arrow(0, 0, loadings[i, 0] * 5, loadings[i, 1] * 5, 
                     color='red', alpha=0.5, head_width=0.05)
            plt.text(loadings[i, 0] * 5.2, loadings[i, 1] * 5.2, feature, 
                    color='red', ha='center', va='center')
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Return PCA components and explained variance
    return pca.components_, pca.explained_variance_ratio_

# Visualize PCA projection
pca_components, explained_variance = visualize_pca_projection(X_features, feature_names, train_sequence_data)


def pad_sequences(X, max_length, padding_value=0):
    """
    Pads sequences to the same length.
    
    Parameters:
    -----------
    X : list of arrays
        List of sequences with varying lengths
    max_length : int
        Length to pad sequences to
    padding_value : int or float
        Value to use for padding
        
    Returns:
    --------
    numpy.ndarray
        Padded sequences with shape (n_samples, max_length, n_features)
    """
    n_samples = len(X)
    n_features = X[0].shape[1] if len(X[0].shape) > 1 else 1
    
    # Initialize padded array
    padded = np.full((n_samples, max_length, n_features), padding_value)
    
    # Fill in with actual values
    for i, seq in enumerate(X):
        seq_len = len(seq)
        if seq_len > max_length:
            # Truncate
            padded[i] = seq[:max_length]
        else:
            # Pad
            padded[i, :seq_len] = seq
    
    return padded

def build_position_predictor(X_features, y_coordinate, feature_names, model_type='rf'):
    """
    Builds a position prediction model based on sequence features.
    Parameters:
    -----------
    X_features : numpy.ndarray
        Feature matrix (n_samples, n_features)
    y_coordinate : list of arrays
        List of coordinate values for each position in each sequence
    feature_names : list
        Names of the features
    model_type : str
        Type of model to use ('rf', 'gbdt', 'ridge', 'svr', 'knn')
    Returns:
    --------
    model : dict
        Dictionary containing the trained model, scaler, and metadata
    """
    # Create position-indexed training data
    X_train_pos = []
    y_train_pos = []
    for i, (features, coords) in enumerate(zip(X_features, y_coordinate)):
        n_positions = len(coords)
        # Create a feature vector for each position
        for pos in range(n_positions):
            # Base features
            pos_features = features.copy()
            # Add position-specific features
            rel_position = pos / (n_positions - 1) if n_positions > 1 else 0.5
            abs_position = pos
            # Combine
            pos_features = np.append(pos_features, [rel_position, abs_position])
            # Add to training data if the coordinate is valid (not NaN)
            if not np.isnan(coords[pos]):  # Skip NaN values
                X_train_pos.append(pos_features)
                y_train_pos.append(coords[pos])

    # Convert to numpy arrays
    X_train_pos = np.array(X_train_pos)
    y_train_pos = np.array(y_train_pos)
    
    # Check if we have any valid data points
    if len(X_train_pos) == 0 or len(y_train_pos) == 0:
        print(f"Warning: No valid training data for {model_type} model")
        return None

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pos)

    # Initialize the model
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'gbdt':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0, random_state=42)
    elif model_type == 'svr':
        model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    elif model_type == 'knn':
        model = KNeighborsRegressor(n_neighbors=5, weights='distance')
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train the model
    model.fit(X_train_scaled, y_train_pos)

    # Calculate training error
    y_pred = model.predict(X_train_scaled)
    mse = mean_squared_error(y_train_pos, y_pred)
    mae = mean_absolute_error(y_train_pos, y_pred)

    print(f"{model_type.upper()} model trained. MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Create extended feature names
    extended_feature_names = feature_names + ['rel_position', 'abs_position']

    return {
        'model': model,
        'scaler': scaler,
        'feature_names': extended_feature_names,
        'metrics': {
            'mse': mse,
            'mae': mae
        }
    }

# Train models for each coordinate dimension
coordinate_models = {}

for coordinate, name in zip([y_coord_x, y_coord_y, y_coord_z], ['x', 'y', 'z']):
    print(f"\nTraining models for {name} coordinate:")
    
    models = {}
    for model_type in ['rf', 'gbdt', 'ridge', 'svr', 'knn']:
        print(f"Training {model_type} model...")
        model_result = build_position_predictor(
            X_features, coordinate, feature_names, model_type
        )
        # Only add the model to our dictionary if it was successfully trained
        if model_result is not None:
            models[model_type] = model_result
    
    # Only add to coordinate_models if we have any valid models
    if models:
        coordinate_models[name] = models
    else:
        print(f"Warning: No valid models could be trained for {name} coordinate")


def evaluate_models_cross_validation(X_features, y_coordinate, feature_names, model_types=None, cv=3):
    """
    Evaluates models using cross-validation.
    Parameters:
    -----------
    X_features : numpy.ndarray
        Feature matrix (n_samples, n_features)
    y_coordinate : list of arrays
        List of coordinate values for each position in each sequence
    feature_names : list
        Names of the features
    model_types : list, optional
        Types of models to evaluate (default: ['rf', 'gbdt', 'ridge', 'svr', 'knn'])
    cv : int, optional
        Number of cross-validation folds (default: 3)
    Returns:
    --------
    results : dict
        Cross-validation results
    """
    if model_types is None:
        model_types = ['rf', 'gbdt', 'ridge', 'svr', 'knn']
    
    # Create position-indexed data
    X_pos = []
    y_pos = []
    seq_indices = [] # Keep track of which sequence each position belongs to
    
    # Filter out positions with NaN coordinates
    for i, (features, coords) in enumerate(zip(X_features, y_coordinate)):
        n_positions = len(coords)
        for pos in range(n_positions):
            # Only include positions with valid (non-NaN) coordinates
            if not np.isnan(coords[pos]):
                # Base features
                pos_features = features.copy()
                # Add position-specific features
                rel_position = pos / (n_positions - 1) if n_positions > 1 else 0.5
                abs_position = pos
                # Combine
                pos_features = np.append(pos_features, [rel_position, abs_position])
                # Add to data
                X_pos.append(pos_features)
                y_pos.append(coords[pos])
                seq_indices.append(i)
    
    # Convert to numpy arrays
    X_pos = np.array(X_pos)
    y_pos = np.array(y_pos)
    seq_indices = np.array(seq_indices)
    
    # Check if we have enough data to proceed
    if len(X_pos) == 0 or len(y_pos) == 0:
        print("Error: No valid data points for cross-validation")
        return {}
    
    # Initialize results
    results = {}
    
    # Define model factories
    model_factories = {
        'rf': lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'gbdt': lambda: GradientBoostingRegressor(n_estimators=100, random_state=42),
        'ridge': lambda: Ridge(alpha=1.0, random_state=42),
        'svr': lambda: SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'knn': lambda: KNeighborsRegressor(n_neighbors=5, weights='distance')
    }
    
    # Function to create a custom cross-validation split
    # This ensures that all positions from the same sequence stay in the same fold
    def sequence_based_cv(n_splits, seq_indices):
        unique_indices = np.unique(seq_indices)
        # Check if we have enough unique sequences for the requested number of folds
        if len(unique_indices) < n_splits:
            print(f"Warning: Only {len(unique_indices)} unique sequences available, reducing folds to {max(2, len(unique_indices))}")
            n_splits = max(2, len(unique_indices))
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(unique_indices):
            train_seqs = unique_indices[train_idx]
            test_seqs = unique_indices[test_idx]
            train_mask = np.isin(seq_indices, train_seqs)
            test_mask = np.isin(seq_indices, test_seqs)
            yield np.where(train_mask)[0], np.where(test_mask)[0]
    
    # Evaluate each model type
    for model_type in model_types:
        if model_type not in model_factories:
            print(f"Warning: Unknown model type '{model_type}', skipping")
            continue
            
        print(f"Evaluating {model_type} model with {cv}-fold cross-validation...")
        model_factory = model_factories[model_type]
        mse_scores = []
        mae_scores = []
        
        try:
            for train_idx, test_idx in sequence_based_cv(cv, seq_indices):
                # Get train/test split
                X_train, X_test = X_pos[train_idx], X_pos[test_idx]
                y_train, y_test = y_pos[train_idx], y_pos[test_idx]
                
                # Skip this fold if we have no test data
                if len(X_test) == 0 or len(y_test) == 0:
                    print(f"Warning: Empty test set in a fold, skipping")
                    continue
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = model_factory()
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse_scores.append(mse)
                mae_scores.append(mae)
            
            # Only proceed if we have scores
            if mse_scores and mae_scores:
                # Calculate average scores
                avg_mse = np.mean(mse_scores)
                avg_mae = np.mean(mae_scores)
                std_mse = np.std(mse_scores)
                std_mae = np.std(mae_scores)
                
                print(f" Average MSE: {avg_mse:.4f} (Â±{std_mse:.4f})")
                print(f" Average MAE: {avg_mae:.4f} (Â±{std_mae:.4f})")
                
                # Store results
                results[model_type] = {
                    'mse': {
                        'mean': avg_mse,
                        'std': std_mse,
                        'scores': mse_scores
                    },
                    'mae': {
                        'mean': avg_mae,
                        'std': std_mae,
                        'scores': mae_scores
                    }
                }
            else:
                print(f" Warning: No valid cross-validation results for {model_type}")
        except Exception as e:
            print(f" Error evaluating {model_type} model: {str(e)}")
    
    return results
    
print("Evaluating models using cross-validation...")
cv_results = {}
for coordinate, name in zip([y_coord_x, y_coord_y, y_coord_z], ['x', 'y', 'z']):
    print(f"\nEvaluating models for {name} coordinate:")
    cv_results[name] = evaluate_models_cross_validation(
        X_features, coordinate, feature_names, cv=3
    )


def visualize_model_comparison(cv_results):
    """
    Visualizes the comparison of different models based on cross-validation results.
    """
    # Create a figure with subplots for each coordinate dimension
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Coordinates and metrics to plot
    coordinates = ['x', 'y', 'z']
    metrics = ['mae', 'mse']
    colors = {'mae': 'blue', 'mse': 'red'}
    
    # Width of bars
    width = 0.35
    
    for i, coord in enumerate(coordinates):
        if coord not in cv_results:
            continue
        
        ax = axes[i]
        
        # Get model types
        model_types = list(cv_results[coord].keys())
        x = np.arange(len(model_types))
        
        # Plot bars for each metric
        for j, metric in enumerate(metrics):
            means = [cv_results[coord][model][metric]['mean'] for model in model_types]
            errors = [cv_results[coord][model][metric]['std'] for model in model_types]
            
            ax.bar(x + width/2 - j*width, means, width, label=metric.upper(), 
                  color=colors[metric], alpha=0.7, yerr=errors, capsize=5)
        
        # Customize the plot
        ax.set_title(f'{coord.upper()} Coordinate')
        ax.set_xticks(x)
        ax.set_xticklabels(model_types)
        ax.set_ylabel('Error')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle('Model Comparison Across Coordinates', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Create a summary table of the best models
    best_models = {}
    
    print("\nBest model for each coordinate dimension:")
    for coord in coordinates:
        if coord not in cv_results:
            continue
        
        # Find model with lowest MAE
        mae_scores = {model: results['mae']['mean'] for model, results in cv_results[coord].items()}
        best_model = min(mae_scores, key=mae_scores.get)
        best_mae = mae_scores[best_model]
        
        # Find model with lowest MSE
        mse_scores = {model: results['mse']['mean'] for model, results in cv_results[coord].items()}
        best_model_mse = min(mse_scores, key=mse_scores.get)
        best_mse = mse_scores[best_model_mse]
        
        print(f"{coord.upper()}: {best_model} (MAE: {best_mae:.4f}), {best_model_mse} (MSE: {best_mse:.4f})")
        
        best_models[coord] = {
            'mae': {'model': best_model, 'score': best_mae},
            'mse': {'model': best_model_mse, 'score': best_mse}
        }
    
    return best_models

# Visualize model comparison
best_models = visualize_model_comparison(cv_results)


def analyze_feature_importance(coordinate_models, feature_names, plot=True):
    """
    Analyzes feature importance across different models.
    
    Parameters:
    -----------
    coordinate_models : dict
        Dictionary of models for each coordinate
    feature_names : list
        Names of the features
    plot : bool, optional
        Whether to plot the feature importance (default: True)
        
    Returns:
    --------
    importance_data : dict
        Feature importance data
    """
    importance_data = {}
    
    # Extended feature names
    extended_feature_names = feature_names + ['rel_position', 'abs_position']
    
    for coord in coordinate_models:
        importance_data[coord] = {}
        
        for model_type, model_data in coordinate_models[coord].items():
            # Check if model has feature_importances_ attribute
            model = model_data['model']
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                importance_data[coord][model_type] = importances
                
                if plot:
                    # Sort features by importance
                    indices = np.argsort(importances)[::-1]
                    
                    # Plot top 15 features
                    plt.figure(figsize=(12, 6))
                    plt.title(f'Feature Importance for {coord.upper()} Coordinate ({model_type.upper()})')
                    plt.bar(range(min(15, len(extended_feature_names))), 
                           importances[indices[:15]], alpha=0.7)
                    plt.xticks(range(min(15, len(extended_feature_names))), 
                              [extended_feature_names[i] for i in indices[:15]], rotation=45, ha='right')
                    plt.tight_layout()
                    plt.show()
            
            elif hasattr(model, 'coef_'):
                # For linear models
                importances = np.abs(model.coef_)
                importance_data[coord][model_type] = importances
                
                if plot:
                    # Sort features by importance
                    indices = np.argsort(importances)[::-1]
                    
                    # Plot top 15 features
                    plt.figure(figsize=(12, 6))
                    plt.title(f'Feature Importance for {coord.upper()} Coordinate ({model_type.upper()})')
                    plt.bar(range(min(15, len(extended_feature_names))), 
                           importances[indices[:15]], alpha=0.7)
                    plt.xticks(range(min(15, len(extended_feature_names))), 
                              [extended_feature_names[i] for i in indices[:15]], rotation=45, ha='right')
                    plt.tight_layout()
                    plt.show()
    
    # Aggregate feature importance across all models that support it
    aggregated_importance = np.zeros(len(extended_feature_names))
    count = 0
    
    for coord in importance_data:
        for model_type, importances in importance_data[coord].items():
            # Normalize importances
            normalized = importances / np.sum(importances)
            aggregated_importance += normalized
            count += 1
    
    if count > 0:
        # Average importances
        aggregated_importance /= count
        
        # Sort features by importance
        indices = np.argsort(aggregated_importance)[::-1]
        
        if plot:
            plt.figure(figsize=(14, 7))
            plt.title('Aggregated Feature Importance Across All Models and Coordinates')
            plt.bar(range(min(20, len(extended_feature_names))), 
                   aggregated_importance[indices[:20]], alpha=0.7)
            plt.xticks(range(min(20, len(extended_feature_names))), 
                      [extended_feature_names[i] for i in indices[:20]], rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
            
            # Print top features
            print("Top 10 most important features:")
            for i in range(min(10, len(extended_feature_names))):
                idx = indices[i]
                print(f"{i+1}. {extended_feature_names[idx]}: {aggregated_importance[idx]:.4f}")
    
    return importance_data, aggregated_importance, extended_feature_names

# Analyze feature importance
importance_data, aggregated_importance, extended_feature_names = analyze_feature_importance(coordinate_models, feature_names)


class RNAEnsembleRegressor:
    """
    Ensemble regressor for RNA structure prediction that combines multiple base models.
    
    This model uses a weighted ensemble of base models, potentially applying different
    weights for different sequence/position characteristics.
    """
    
    def __init__(self, base_models, weights=None):
        """
        Initialize the ensemble regressor.
        
        Parameters:
        -----------
        base_models : dict
            Dictionary of base models, each with 'model' and 'scaler' keys
        weights : dict or None
            Optional weights for each model. If None, equal weights are used.
        """
        self.base_models = base_models
        self.weights = weights if weights is not None else {model: 1.0 for model in base_models}
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {model: w / total_weight for model, w in self.weights.items()}
    
    def predict(self, X, sequence_lengths=None):
        """
        Make predictions using the ensemble of models.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Features for prediction
        sequence_lengths : list or None
            List of sequence lengths. If None, all sequences are assumed to have the same length.
            
        Returns:
        --------
        numpy.ndarray
            Predicted coordinates
        """
        predictions = {}
        
        # Get predictions from each base model
        for model_name, model_data in self.base_models.items():
            # Scale features
            X_scaled = model_data['scaler'].transform(X)
            
            # Make prediction
            pred = model_data['model'].predict(X_scaled)
            predictions[model_name] = pred
        
        # Combine predictions with weights
        weighted_pred = np.zeros_like(list(predictions.values())[0])
        for model_name, pred in predictions.items():
            weighted_pred += pred * self.weights[model_name]
        
        return weighted_pred

class RNACoordinatePredictor:
    """
    Predicts 3D coordinates for RNA sequences using ensemble models for each coordinate dimension.
    """
    
    def __init__(self, x_ensemble, y_ensemble, z_ensemble):
        """
        Initialize the predictor with three ensemble models.
        
        Parameters:
        -----------
        x_ensemble : RNAEnsembleRegressor
            Ensemble model for x-coordinate
        y_ensemble : RNAEnsembleRegressor
            Ensemble model for y-coordinate
        z_ensemble : RNAEnsembleRegressor
            Ensemble model for z-coordinate
        """
        self.x_ensemble = x_ensemble
        self.y_ensemble = y_ensemble
        self.z_ensemble = z_ensemble
    
    def predict(self, features, sequences):
        """
        Predict 3D coordinates for a list of RNA sequences.
        
        Parameters:
        -----------
        features : numpy.ndarray
            Feature matrix (n_samples, n_features)
        sequences : list
            List of RNA sequences
            
        Returns:
        --------
        list
            List of predicted 3D structures
        """
        # Create position-indexed features
        X_pos = []
        seq_indices = []
        positions = []
        
        for i, (seq_features, seq) in enumerate(zip(features, sequences)):
            n_positions = len(seq)
            
            for pos in range(n_positions):
                # Base features
                pos_features = seq_features.copy()
                
                # Add position-specific features
                rel_position = pos / (n_positions - 1) if n_positions > 1 else 0.5
                abs_position = pos
                
                # Combine
                pos_features = np.append(pos_features, [rel_position, abs_position])
                
                # Add to data
                X_pos.append(pos_features)
                seq_indices.append(i)
                positions.append(pos)
        
        # Convert to numpy array
        X_pos = np.array(X_pos)
        
        # Predict each coordinate
        x_coords = self.x_ensemble.predict(X_pos)
        y_coords = self.y_ensemble.predict(X_pos)
        z_coords = self.z_ensemble.predict(X_pos)
        
        # Group by sequence
        predicted_structures = []
        for i in range(len(sequences)):
            # Get positions for this sequence
            seq_mask = np.array(seq_indices) == i
            seq_positions = np.array(positions)[seq_mask]
            
            # Initialize structure
            coords = np.zeros((len(sequences[i]), 3))
            
            # Fill in coordinates
            coords[seq_positions, 0] = x_coords[seq_mask]
            coords[seq_positions, 1] = y_coords[seq_mask]
            coords[seq_positions, 2] = z_coords[seq_mask]
            
            predicted_structures.append(coords)
        
        return predicted_structures

def build_ensemble_models(coordinate_models, best_models):
    """
    Builds ensemble models based on the best individual models.
    
    Parameters:
    -----------
    coordinate_models : dict
        Dictionary of models for each coordinate
    best_models : dict
        Dictionary of best models for each coordinate
        
    Returns:
    --------
    predictor : RNACoordinatePredictor
        Coordinate predictor using ensemble models
    """
    ensembles = {}
    
    for coord in ['x', 'y', 'z']:
        # Get best models for this coordinate
        best_mae_model = best_models[coord]['mae']['model']
        best_mse_model = best_models[coord]['mse']['model']
        
        # Collect models to include in the ensemble
        ensemble_models = {}
        ensemble_weights = {}
        
        # Always include the best models
        ensemble_models[best_mae_model] = coordinate_models[coord][best_mae_model]
        ensemble_weights[best_mae_model] = 0.5
        
        if best_mse_model != best_mae_model:
            ensemble_models[best_mse_model] = coordinate_models[coord][best_mse_model]
            ensemble_weights[best_mse_model] = 0.3
        
        # Add a third model for diversity
        for model_type in ['rf', 'gbdt', 'ridge']:
            if model_type not in ensemble_models and model_type in coordinate_models[coord]:
                ensemble_models[model_type] = coordinate_models[coord][model_type]
                ensemble_weights[model_type] = 0.2
                break
        
        # Create ensemble
        ensembles[coord] = RNAEnsembleRegressor(ensemble_models, ensemble_weights)
    
    # Create coordinate predictor
    predictor = RNACoordinatePredictor(
        ensembles['x'],
        ensembles['y'],
        ensembles['z']
    )
    
    return predictor

# Build ensemble models
ensemble_predictor = build_ensemble_models(coordinate_models, best_models)
print("Ensemble predictor built successfully.")


def check_structure_validity(coords):
    """
    Check if an RNA structure is physically valid.
    
    Parameters:
    -----------
    coords : numpy.ndarray
        Array of 3D coordinates
        
    Returns:
    --------
    bool
        True if the structure is valid, False otherwise
    """
    # Filter invalid coordinates
    valid_mask = ~np.any(np.isnan(coords), axis=1) & ~np.any(np.isinf(coords), axis=1)
    if np.sum(valid_mask) < 3:
        return False
    
    valid_coords = coords[valid_mask]
    
    # Check distances between consecutive residues
    for i in range(1, len(valid_coords)):
        dist = np.linalg.norm(valid_coords[i] - valid_coords[i-1])
        # RNA nucleotides should be about 3.4-4.0 Ã… apart in the backbone
        if dist < 2.0 or dist > 6.0:
            return False
    
    # Check for unrealistic clustering (atoms too close to each other)
    for i in range(len(valid_coords)):
        for j in range(i+3, len(valid_coords)):  # Skip adjacent nucleotides
            dist = np.linalg.norm(valid_coords[i] - valid_coords[j])
            # Non-adjacent nucleotides should not be too close
            if dist < 3.0:
                return False
    
    return True

def optimize_structure(coords, sequence):
    """
    Optimize an RNA structure based on physical and chemical constraints.
    
    Parameters:
    -----------
    coords : numpy.ndarray
        Initial 3D coordinates
    sequence : str
        RNA sequence
        
    Returns:
    --------
    numpy.ndarray
        Optimized 3D coordinates
    """
    # Remove invalid coordinates before optimization
    valid_mask = ~np.any(np.isnan(coords), axis=1) & ~np.any(np.isinf(coords), axis=1)
    if np.sum(valid_mask) < 3:
        print("Warning: Too few valid coordinates for optimization")
        return coords
    
    valid_coords = coords[valid_mask].copy()
    
    # Define the objective function to minimize
    def objective(x_flat):
        # Reshape flattened coordinates
        x_reshaped = x_flat.reshape(-1, 3)
        
        # Calculate backbone bond length errors
        bond_length_error = 0
        target_bond_length = 3.8  # Ã…, typical RNA backbone distance
        for i in range(1, len(x_reshaped)):
            dist = np.linalg.norm(x_reshaped[i] - x_reshaped[i-1])
            bond_length_error += (dist - target_bond_length) ** 2
        
        # Calculate clash penalties (non-adjacent nucleotides too close)
        clash_penalty = 0
        for i in range(len(x_reshaped)):
            for j in range(i+3, len(x_reshaped)):
                dist = np.linalg.norm(x_reshaped[i] - x_reshaped[j])
                if dist < 4.0:
                    clash_penalty += (4.0 - dist) ** 2
        
        # Calculate base pairing energy (simplified)
        base_pairing_energy = 0
        # Define complementary bases
        complements = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
        
        for i in range(len(sequence)):
            if i >= len(x_reshaped):
                continue
            for j in range(i+4, len(sequence)):  # Minimum loop size of 3
                if j >= len(x_reshaped):
                    continue
                # Check if bases are complementary
                if sequence[i] in complements and sequence[j] == complements[sequence[i]]:
                    dist = np.linalg.norm(x_reshaped[i] - x_reshaped[j])
                    # Optimal base pair distance is around 5-6 Ã…
                    base_pairing_energy += min((dist - 5.5) ** 2, 10.0)
        
        # Combine all terms with appropriate weights
        total_energy = (
            10.0 * bond_length_error +  # Higher weight for bond lengths
            5.0 * clash_penalty +       # Medium weight for clashes
            1.0 * base_pairing_energy   # Lower weight for base pairing
        )
        
        return total_energy
    
    # Flatten the coordinates for optimization
    x0 = valid_coords.flatten()
    
    # Perform optimization
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 10, 'disp': True}
    )
    
    # Reshape the optimized coordinates
    optimized_coords = result.x.reshape(-1, 3)
    
    # Copy back to original array
    result_coords = coords.copy()
    result_coords[valid_mask] = optimized_coords
    
    return result_coords

def generate_structure_ensemble(coords, sequence, n_models=5):
    """
    Generate an ensemble of structures by applying perturbations.
    
    Parameters:
    -----------
    coords : numpy.ndarray
        Initial 3D coordinates
    sequence : str
        RNA sequence
    n_models : int, optional
        Number of models to generate (default: 5)
        
    Returns:
    --------
    list
        List of generated structures
    """
    # Make sure initial structure is valid
    if not check_structure_validity(coords):
        print("Warning: Initial structure is not valid, attempting to optimize")
        coords = optimize_structure(coords, sequence)
    
    # Generate ensemble
    ensemble = [coords]
    
    for i in range(1, n_models):
        # Apply increasingly larger perturbations
        noise_scale = 0.5 * (i / n_models)
        
        # Create perturbed structure
        perturbed = coords.copy()
        
        # Apply correlated noise for more realistic perturbations
        for j in range(1, len(perturbed)):
            if np.any(np.isnan(perturbed[j-1])) or np.any(np.isnan(perturbed[j])):
                continue
                
            # Generate random perturbation direction
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            
            # Apply perturbation
            perturbed[j] += direction * noise_scale
            
            # Correct bond length to keep it physically plausible
            bond_vector = perturbed[j] - perturbed[j-1]
            bond_length = np.linalg.norm(bond_vector)
            
            if bond_length > 0:
                # Target RNA backbone distance with small variation
                target_length = 3.8 * (1 + np.random.normal(0, 0.05))
                perturbed[j] = perturbed[j-1] + (bond_vector / bond_length) * target_length
        
        # Optimize the perturbed structure
        optimized = optimize_structure(perturbed, sequence)
        
        # Add to ensemble if valid
        if check_structure_validity(optimized):
            ensemble.append(optimized)
        else:
            # If not valid, try again with less perturbation
            noise_scale = 0.2 * (i / n_models)
            perturbed = coords.copy()
            for j in range(1, len(perturbed)):
                if np.any(np.isnan(perturbed[j-1])) or np.any(np.isnan(perturbed[j])):
                    continue
                    
                direction = np.random.randn(3)
                direction = direction / np.linalg.norm(direction)
                perturbed[j] += direction * noise_scale
                
                bond_vector = perturbed[j] - perturbed[j-1]
                bond_length = np.linalg.norm(bond_vector)
                
                if bond_length > 0:
                    target_length = 3.8 * (1 + np.random.normal(0, 0.02))
                    perturbed[j] = perturbed[j-1] + (bond_vector / bond_length) * target_length
            
            optimized = optimize_structure(perturbed, sequence)
            ensemble.append(optimized)
    
    # Ensure we have exactly n_models structures
    while len(ensemble) < n_models:
        # If we couldn't generate enough valid structures, duplicate the last one
        ensemble.append(ensemble[-1])
    
    return ensemble[:n_models]

def calculate_tm_score(pred_coords, true_coords):
    """
    Calculate TM-score between predicted and true coordinates.
    
    Parameters:
    -----------
    pred_coords : numpy.ndarray
        Predicted 3D coordinates
    true_coords : numpy.ndarray
        True 3D coordinates
        
    Returns:
    --------
    float
        TM-score (0.0 to 1.0)
    """
    # Remove invalid coordinates
    valid_mask = (~np.any(np.isnan(pred_coords), axis=1) & 
                 ~np.any(np.isnan(true_coords), axis=1) &
                 ~np.any(np.isinf(pred_coords), axis=1) &
                 ~np.any(np.isinf(true_coords), axis=1))
    
    if np.sum(valid_mask) < 3:
        print("Warning: Too few valid coordinates for TM-score calculation")
        return 0.0
    
    pred = pred_coords[valid_mask]
    true = true_coords[valid_mask]
    L = len(true)
    
    # Define d0 based on L (values adapted for RNA)
    if L >= 30:
        d0 = 0.6 * np.sqrt(L - 0.5) - 2.5
        d0 = max(0.1, d0)
    elif L >= 24:
        d0 = 0.7
    elif L >= 20:
        d0 = 0.6
    elif L >= 16:
        d0 = 0.5
    elif L >= 12:
        d0 = 0.4
    else:
        d0 = 0.3
    
    # Center structures at their centroids
    pred_centroid = np.mean(pred, axis=0)
    true_centroid = np.mean(true, axis=0)
    
    pred_centered = pred - pred_centroid
    true_centered = true - true_centroid
    
    # Find optimal rotation using Kabsch algorithm
    # Calculate covariance matrix
    covariance = np.dot(pred_centered.T, true_centered)
    V, S, W = np.linalg.svd(covariance)
    
    # Ensure proper rotation (no reflection)
    d = np.sign(np.linalg.det(np.dot(V, W)))
    U = np.dot(V, np.diag([1, 1, d]), W)
    
    # Rotate predicted structure
    pred_aligned = np.dot(pred_centered, U)
    
    # Calculate distances
    distances = np.sqrt(np.sum((pred_aligned - true_centered) ** 2, axis=1))
    
    # Calculate TM-score terms
    tm_terms = 1.0 / (1.0 + (distances / d0) ** 2)
    tm_score = np.sum(tm_terms) / L
    
    return float(tm_score)

def evaluate_structure(pred_coords, true_coords):
    """
    Evaluate a predicted structure against the ground truth.
    
    Parameters:
    -----------
    pred_coords : numpy.ndarray
        Predicted 3D coordinates
    true_coords : numpy.ndarray
        True 3D coordinates
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Calculate TM-score
    tm_score = calculate_tm_score(pred_coords, true_coords)
    
    # Calculate RMSD
    valid_mask = (~np.any(np.isnan(pred_coords), axis=1) & 
                 ~np.any(np.isnan(true_coords), axis=1) &
                 ~np.any(np.isinf(pred_coords), axis=1) &
                 ~np.any(np.isinf(true_coords), axis=1))
    
    if np.sum(valid_mask) < 3:
        print("Warning: Too few valid coordinates for RMSD calculation")
        rmsd = float('inf')
    else:
        pred = pred_coords[valid_mask]
        true = true_coords[valid_mask]
        
        # Center structures
        pred_centroid = np.mean(pred, axis=0)
        true_centroid = np.mean(true, axis=0)
        
        pred_centered = pred - pred_centroid
        true_centered = true - true_centroid
        
        # Find optimal rotation
        covariance = np.dot(pred_centered.T, true_centered)
        V, S, W = np.linalg.svd(covariance)
        
        # Ensure proper rotation
        d = np.sign(np.linalg.det(np.dot(V, W)))
        U = np.dot(V, np.diag([1, 1, d]), W)
        
        # Rotate predicted structure
        pred_aligned = np.dot(pred_centered, U)
        
        # Calculate RMSD
        rmsd = np.sqrt(np.mean(np.sum((pred_aligned - true_centered) ** 2, axis=1)))
    
    # Check physical validity
    validity = check_structure_validity(pred_coords)
    
    return {
        'TM-score': tm_score,
        'RMSD': rmsd,
        'Validity': validity
    }

def test_optimization_on_validation(predictor, valid_mapping):
    """
    Test structure optimization on the validation set.
    Parameters:
    -----------
    predictor : RNACoordinatePredictor
        Coordinate predictor
    valid_mapping : dict
        Mapping of validation sequences to structures
    Returns:
    --------
    dict
        Results of the evaluation
    """
    results = {}
    # Extract sequences and features
    sequences = []
    features = []
    for seq_id, data in valid_mapping.items():
        sequences.append(data['sequence'])
        features.append(extract_sequence_features(data['sequence']))
    
    # Convert features to array - ensuring all features are present in all dictionaries
    # First find all possible keys
    all_keys = set()
    for f in features:
        all_keys.update(f.keys())
    
    # Create a list of lists where each inner list has the same length and order
    feature_arrays = []
    for f in features:
        # Create a list with all features, using 0 for missing features
        feature_array = [f.get(key, 0) for key in sorted(all_keys)]
        feature_arrays.append(feature_array)
    
    # Now convert to numpy array
    X_features = np.array(feature_arrays)
    
    # Predict structures
    print("Predicting structures...")
    predicted_structures = predictor.predict(X_features, sequences)
    
    # Optimize structures
    print("Optimizing structures...")
    optimized_structures = []
    for pred, seq in zip(predicted_structures, sequences):
        optimized = optimize_structure(pred, seq)
        optimized_structures.append(optimized)
    
    # Generate ensembles
    print("Generating ensembles...")
    ensembles = []
    for opt, seq in zip(optimized_structures, sequences):
        ensemble = generate_structure_ensemble(opt, seq, n_models=5)
        ensembles.append(ensemble)
    
    # Evaluate all structures
    print("Evaluating structures...")
    for i, seq_id in enumerate(valid_mapping.keys()):
        true_structure = valid_mapping[seq_id]['structures'][0]
        
        # Evaluate original prediction
        pred_metrics = evaluate_structure(predicted_structures[i], true_structure)
        
        # Evaluate optimized structure
        opt_metrics = evaluate_structure(optimized_structures[i], true_structure)
        
        # Evaluate best ensemble structure (highest TM-score)
        ensemble_tm_scores = []
        for j in range(len(ensembles[i])):
            tm_score = calculate_tm_score(ensembles[i][j], true_structure)
            ensemble_tm_scores.append(tm_score)
        best_idx = np.argmax(ensemble_tm_scores)
        best_ensemble = ensembles[i][best_idx]
        ens_metrics = evaluate_structure(best_ensemble, true_structure)
        
        results[seq_id] = {
            'Original': pred_metrics,
            'Optimized': opt_metrics,
            'Best Ensemble': ens_metrics,
            'Ensemble TM-scores': ensemble_tm_scores
        }
    
    # Print summary
    print("\nOptimization Results Summary:")
    orig_tm_scores = [results[seq_id]['Original']['TM-score'] for seq_id in results]
    opt_tm_scores = [results[seq_id]['Optimized']['TM-score'] for seq_id in results]
    ens_tm_scores = [results[seq_id]['Best Ensemble']['TM-score'] for seq_id in results]
    
    print(f"Original - Avg TM-score: {np.mean(orig_tm_scores):.4f} (Â±{np.std(orig_tm_scores):.4f})")
    print(f"Optimized - Avg TM-score: {np.mean(opt_tm_scores):.4f} (Â±{np.std(opt_tm_scores):.4f})")
    print(f"Best Ensemble - Avg TM-score: {np.mean(ens_tm_scores):.4f} (Â±{np.std(ens_tm_scores):.4f})")
    
    return results

# Test optimization on validation set
#optimization_results = test_optimization_on_validation(ensemble_predictor, valid_mapping)

def optimize_structure_simple(coords, sequence):
    """
    ç®€åŒ–çš„ç»“æ�„ä¼˜åŒ–å‡½æ•° - ä»…å�šæœ€å°�ç¨‹åº¦çš„ä¿®æ­£
    """
    # å¤�åˆ¶å��æ ‡ä»¥é�¿å…�ä¿®æ”¹å�Ÿå§‹æ•°æ�®
    result_coords = coords.copy()
    
    # ç§»é™¤æ— æ•ˆå��æ ‡ï¼ˆNaNå’ŒInfï¼‰
    valid_mask = ~np.any(np.isnan(coords), axis=1) & ~np.any(np.isinf(coords), axis=1)
    
    # å¦‚æ�œæœ‰æ•ˆå��æ ‡å¤ªå°‘ï¼Œç›´æ�¥è¿”å›�
    if np.sum(valid_mask) < 3:
        return result_coords
    
    # å�ªå�šä¸€äº›ç®€å�•çš„ä¿®æ­£
    for i in range(1, len(coords)):
        # è·³è¿‡æ— æ•ˆå��æ ‡
        if not valid_mask[i] or not valid_mask[i-1]:
            continue
        
        # ç®€å�•åœ°è°ƒæ•´ç›¸é‚»æ®‹åŸºä¹‹é—´çš„è·�ç¦»ä¸ºå�ˆç�†èŒƒå›´
        dist = np.linalg.norm(result_coords[i] - result_coords[i-1])
        if dist < 2.0 or dist > 6.0:
            # è®¡ç®—å�•ä½�å�‘é‡�
            direction = result_coords[i] - result_coords[i-1]
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                # å°†è·�ç¦»è®¾ç½®ä¸ºç�†æƒ³å€¼3.8Ã…
                result_coords[i] = result_coords[i-1] + direction * 3.8
    
    return result_coords

def generate_structure_ensemble_simple(coords, sequence, n_models=2):
    """
    ç®€åŒ–çš„ç»“æ�„ç”Ÿæˆ�å‡½æ•° - å�ªç”Ÿæˆ�å°‘é‡�ç»“æ�„å¹¶ä½¿ç”¨ç®€å�•çš„æ‰°åŠ¨
    """
    # ç¡®ä¿�åˆ�å§‹ç»“æ�„æ­£å¸¸
    ensemble = [coords]
    
    # å�ªç”Ÿæˆ�ä¸€ä¸ªé¢�å¤–ç»“æ�„
    perturbed = coords.copy()
    # æ·»åŠ å°�çš„éš�æœºæ‰°åŠ¨
    noise = np.random.normal(0, 0.5, perturbed.shape)
    valid_mask = ~np.any(np.isnan(perturbed), axis=1) & ~np.any(np.isinf(perturbed), axis=1)
    perturbed[valid_mask] += noise[valid_mask]
    
    # å�šç®€å�•ä¼˜åŒ–
    optimized = optimize_structure_simple(perturbed, sequence)
    ensemble.append(optimized)
    
    return ensemble[:n_models]

def test_optimization_on_validation_simple(predictor, valid_mapping, max_samples=10):
    """
    ç®€åŒ–ç‰ˆçš„éªŒè¯�å‡½æ•° - å�ªå¤„ç�†å°‘é‡�æ ·æœ¬
    """
    results = {}
    
    # å�ªå�–å‰�å‡ ä¸ªæ ·æœ¬è¿›è¡Œæµ‹è¯•
    sample_ids = list(valid_mapping.keys())[:max_samples]
    
    # æ��å�–åº�åˆ—å’Œç‰¹å¾�
    sequences = []
    features = []
    for seq_id in sample_ids:
        data = valid_mapping[seq_id]
        sequences.append(data['sequence'])
        features.append(extract_sequence_features(data['sequence']))
    
    # ç»Ÿä¸€ç‰¹å¾�æ ¼å¼�
    all_keys = set()
    for f in features:
        all_keys.update(f.keys())
    
    feature_arrays = []
    for f in features:
        feature_array = [f.get(key, 0) for key in sorted(all_keys)]
        feature_arrays.append(feature_array)
    
    X_features = np.array(feature_arrays)
    
    print("Predicting structures...")
    predicted_structures = predictor.predict(X_features, sequences)
    
    print("Applying simple optimization...")
    optimized_structures = []
    for pred, seq in zip(predicted_structures, sequences):
        optimized = optimize_structure_simple(pred, seq)
        optimized_structures.append(optimized)
    
    print("Generating simple ensembles...")
    ensembles = []
    for opt, seq in zip(optimized_structures, sequences):
        ensemble = generate_structure_ensemble_simple(opt, seq, n_models=5)
        ensembles.append(ensemble)
    
    print("Evaluating structures...")
    for i, seq_id in enumerate(sample_ids):
        true_structure = valid_mapping[seq_id]['structures'][0]
        
        # è¯„ä¼°å�Ÿå§‹é¢„æµ‹
        pred_metrics = evaluate_structure(predicted_structures[i], true_structure)
        
        # è¯„ä¼°ä¼˜åŒ–å��çš„ç»“æ�„
        opt_metrics = evaluate_structure(optimized_structures[i], true_structure)
        
        # è¯„ä¼°æœ€ä½³é›†å�ˆç»“æ�„
        ensemble_tm_scores = []
        for j in range(len(ensembles[i])):
            tm_score = calculate_tm_score(ensembles[i][j], true_structure)
            ensemble_tm_scores.append(tm_score)
        best_idx = np.argmax(ensemble_tm_scores)
        best_ensemble = ensembles[i][best_idx]
        ens_metrics = evaluate_structure(best_ensemble, true_structure)
        
        results[seq_id] = {
            'Original': pred_metrics,
            'Optimized': opt_metrics,
            'Best Ensemble': ens_metrics,
            'Ensemble TM-scores': ensemble_tm_scores
        }
    
    # æ‰“å�°æ‘˜è¦�
    print("\nOptimization Results Summary:")
    if results:
        orig_tm_scores = [results[seq_id]['Original']['TM-score'] for seq_id in results]
        opt_tm_scores = [results[seq_id]['Optimized']['TM-score'] for seq_id in results]
        ens_tm_scores = [results[seq_id]['Best Ensemble']['TM-score'] for seq_id in results]
        
        print(f"Original - Avg TM-score: {np.mean(orig_tm_scores):.4f} (Â±{np.std(orig_tm_scores):.4f})")
        print(f"Optimized - Avg TM-score: {np.mean(opt_tm_scores):.4f} (Â±{np.std(opt_tm_scores):.4f})")
        print(f"Best Ensemble - Avg TM-score: {np.mean(ens_tm_scores):.4f} (Â±{np.std(ens_tm_scores):.4f})")
    
    return results

# ä½¿ç”¨æ–¹æ³•
optimization_results = test_optimization_on_validation_simple(ensemble_predictor, valid_mapping, max_samples=10)


def visualize_optimization_results(optimization_results):
    """
    Visualize the improvement from structure optimization.
    
    Parameters:
    -----------
    optimization_results : dict
        Results of the optimization evaluation
    """
    # Extract TM-scores
    seq_ids = list(optimization_results.keys())
    original_scores = [optimization_results[seq_id]['Original']['TM-score'] for seq_id in seq_ids]
    optimized_scores = [optimization_results[seq_id]['Optimized']['TM-score'] for seq_id in seq_ids]
    ensemble_scores = [optimization_results[seq_id]['Best Ensemble']['TM-score'] for seq_id in seq_ids]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot TM-score comparison
    bar_width = 0.25
    index = np.arange(len(seq_ids))
    
    axes[0].bar(index, original_scores, bar_width, label='Original', alpha=0.7)
    axes[0].bar(index + bar_width, optimized_scores, bar_width, label='Optimized', alpha=0.7)
    axes[0].bar(index + 2*bar_width, ensemble_scores, bar_width, label='Best Ensemble', alpha=0.7)
    
    axes[0].set_xlabel('Sequence ID')
    axes[0].set_ylabel('TM-score')
    axes[0].set_title('TM-score Improvement from Optimization')
    axes[0].set_xticks(index + bar_width)
    axes[0].set_xticklabels(seq_ids, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Add average improvement
    improvement_opt = [opt - orig for orig, opt in zip(original_scores, optimized_scores)]
    improvement_ens = [ens - orig for orig, ens in zip(original_scores, ensemble_scores)]
    
    # Plot improvement distribution
    axes[1].boxplot([improvement_opt, improvement_ens], labels=['Optimization', 'Ensemble'])
    axes[1].set_ylabel('TM-score Improvement')
    axes[1].set_title('Distribution of TM-score Improvement')
    axes[1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    axes[1].grid(alpha=0.3)
    
    # Calculate and display statistics
    avg_improvement_opt = np.mean(improvement_opt)
    avg_improvement_ens = np.mean(improvement_ens)
    
    print(f"Average TM-score improvement from optimization: {avg_improvement_opt:.4f}")
    print(f"Average TM-score improvement from ensemble: {avg_improvement_ens:.4f}")
    
    # Show ratio of improved sequences
    improved_opt = sum(1 for imp in improvement_opt if imp > 0)
    improved_ens = sum(1 for imp in improvement_ens if imp > 0)
    
    print(f"Optimization improved {improved_opt}/{len(seq_ids)} sequences ({improved_opt/len(seq_ids)*100:.2f}%)")
    print(f"Ensemble improved {improved_ens}/{len(seq_ids)} sequences ({improved_ens/len(seq_ids)*100:.2f}%)")
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Visualize optimization results
optimization_fig = visualize_optimization_results(optimization_results)


def create_submission(test_predictions, sample_submission_df):
    """
    Creates a submission file from test predictions.
    
    Parameters:
    -----------
    test_predictions : dict
        Dictionary mapping sequence IDs to lists of structure ensembles
    sample_submission_df : pandas.DataFrame
        Sample submission DataFrame to use as a template
        
    Returns:
    --------
    pandas.DataFrame
        Submission DataFrame
    """
    # Create a copy of the sample submission
    submission_df = sample_submission_df.copy()
    
    # Process each row in the submission
    print(f"Processing row 0/{len(submission_df)}")
    for i, row in submission_df.iterrows():
        # Print progress every 1000 rows
        if i > 0 and i % 1000 == 0:
            print(f"Processing row {i}/{len(submission_df)}")
            
        # Get the ID and extract sequence ID and position
        id_parts = row['ID'].split('_')
        seq_id = id_parts[0]
        position = int(id_parts[1]) - 1  # Convert to 0-indexed
        
        # Check if we have predictions for this sequence
        if seq_id in test_predictions:
            ensemble = test_predictions[seq_id]
            
            # Fill in coordinates for each structure in the ensemble
            for struct_idx in range(1, 6):
                # Check if we have this structure in the ensemble
                if struct_idx <= len(ensemble):
                    # Check if the position is valid
                    if position < len(ensemble[struct_idx-1]):
                        coords = ensemble[struct_idx-1][position]
                        
                        # Fill in coordinates if they are valid
                        if not (np.isnan(coords).any() or np.isinf(coords).any()):
                            submission_df.at[i, f'x_{struct_idx}'] = coords[0]
                            submission_df.at[i, f'y_{struct_idx}'] = coords[1]
                            submission_df.at[i, f'z_{struct_idx}'] = coords[2]
    
    return submission_df

def prepare_test_features(test_seq_df):
    """
    Prepares test features for prediction.
    Parameters:
    -----------
    test_seq_df : pandas.DataFrame
        DataFrame with test sequences
    Returns:
    --------
    features : numpy.ndarray
        Array of features for each test sequence
    sequences : list
        List of sequence strings
    """
    features_dict = []
    sequences = []
    
    for _, row in test_seq_df.iterrows():
        seq = row['sequence']
        seq_features = extract_sequence_features(seq)
        features_dict.append(seq_features)
        sequences.append(seq)
    
    # ç»Ÿä¸€ç‰¹å¾�æ ¼å¼�ï¼Œç¡®ä¿�æ‰€æœ‰ç‰¹å¾�å­—å…¸æœ‰ç›¸å�Œçš„é”®
    all_keys = set()
    for f in features_dict:
        all_keys.update(f.keys())
    
    # ä¸ºæ¯�ä¸ªåº�åˆ—åˆ›å»ºæ ‡å‡†åŒ–çš„ç‰¹å¾�æ•°ç»„
    feature_arrays = []
    for f in features_dict:
        # å¯¹æ¯�ä¸ªç‰¹å¾�åˆ›å»ºæœ‰åº�çš„å€¼åˆ—è¡¨ï¼Œä½¿ç”¨0å¡«å……ç¼ºå¤±å€¼
        feature_array = [f.get(key, 0) for key in sorted(all_keys)]
        feature_arrays.append(feature_array)
    
    # è½¬æ�¢ä¸ºNumPyæ•°ç»„
    features = np.array(feature_arrays)
    
    print(f"Prepared test features with shape: {features.shape}")
    return features, sequences


def predict_test_structures(predictor, test_features, test_sequences, test_seq_ids):
    """
    Predicts structures for test sequences.
    Parameters:
    -----------
    predictor : RNACoordinatePredictor
        Coordinate predictor
    test_features : numpy.ndarray
        Features for test sequences
    test_sequences : list
        List of test sequences
    test_seq_ids : list
        List of sequence IDs corresponding to test_sequences
    Returns:
    --------
    predictions : dict
        Dictionary mapping sequence IDs to lists of structure ensembles
    """
    # Predict initial structures
    print("Predicting test structures...")
    initial_structures = predictor.predict(test_features, test_sequences)
    
    # Optimize and generate ensembles
    predictions = {}
    for i, (seq_id, seq) in enumerate(zip(test_seq_ids, test_sequences)):
        print(f"Processing sequence {i+1}/{len(test_sequences)}: {seq_id}")
        # Optimize structure
        optimized = optimize_structure_simple(initial_structures[i], seq)
        # Generate ensemble
        ensemble = generate_structure_ensemble_simple(optimized, seq, n_models=5)
        # Store predictions
        predictions[seq_id] = ensemble
    
    return predictions

# ç„¶å��åœ¨ä½¿ç”¨æ—¶:
# 1. å‡†å¤‡æµ‹è¯•ç‰¹å¾�å’Œåº�åˆ—
test_features, test_sequences = prepare_test_features(data_dict['test_seq'])

# 2. è�·å�–æµ‹è¯•åº�åˆ—ID
test_seq_ids = data_dict['test_seq']['target_id'].tolist()

# 3. é¢„æµ‹æµ‹è¯•ç»“æ�„
test_predictions = predict_test_structures(ensemble_predictor, test_features, test_sequences, test_seq_ids)

# 4. åˆ›å»ºæ��äº¤
submission_df = create_submission(test_predictions, data_dict['sample_submission'])


def validate_submission(submission_df, sample_submission_df):
    """
    Validates the submission file format.
    
    Parameters:
    -----------
    submission_df : pandas.DataFrame
        Submission DataFrame
    sample_submission_df : pandas.DataFrame
        Sample submission format
        
    Returns:
    --------
    bool
        True if the submission is valid, False otherwise
    """
    # Check that all columns are present
    if not all(col in submission_df.columns for col in sample_submission_df.columns):
        print("Error: Missing columns in submission")
        return False
    
    # Check that all IDs are present
    if not all(id in submission_df['ID'].values for id in sample_submission_df['ID'].values):
        print("Error: Missing IDs in submission")
        return False
    
    # Check for NaN values
    for col in submission_df.columns:
        if col.startswith('x_') or col.startswith('y_') or col.startswith('z_'):
            if submission_df[col].isna().any():
                print(f"Warning: NaN values found in column {col}")
    
    # Check for extreme values
    for col in submission_df.columns:
        if col.startswith('x_') or col.startswith('y_') or col.startswith('z_'):
            if (submission_df[col].abs() > 1e6).any():
                print(f"Warning: Extreme values found in column {col}")
    
    print("Submission validation passed!")
    return True

# Validate submission
is_valid = validate_submission(submission_df, data_dict['sample_submission'])

# Show submission statistics
def show_submission_stats(submission_df):
    """
    Shows statistics about the submission.
    """
    print("\nSubmission Statistics:")
    print(f"Number of rows: {len(submission_df)}")
    
    # Count unique sequence IDs
    seq_ids = set(id.split('_')[0] for id in submission_df['ID'])
    print(f"Number of unique sequences: {len(seq_ids)}")
    
    # Check coordinate statistics
    for struct_idx in range(1, 6):
        # Calculate coordinate statistics
        x_col = f'x_{struct_idx}'
        y_col = f'y_{struct_idx}'
        z_col = f'z_{struct_idx}'
        
        x_mean = submission_df[x_col].mean()
        y_mean = submission_df[y_col].mean()
        z_mean = submission_df[z_col].mean()
        
        x_std = submission_df[x_col].std()
        y_std = submission_df[y_col].std()
        z_std = submission_df[z_col].std()
        
        print(f"\nStructure {struct_idx} statistics:")
        print(f"  X: mean={x_mean:.2f}, std={x_std:.2f}")
        print(f"  Y: mean={y_mean:.2f}, std={y_std:.2f}")
        print(f"  Z: mean={z_mean:.2f}, std={z_std:.2f}")

# Show submission statistics
show_submission_stats(submission_df)


def save_models(coordinate_models, ensemble_predictor, output_dir=OUTPUT_DIR):
    """
    Saves trained models to disk.
    
    Parameters:
    -----------
    coordinate_models : dict
        Dictionary of models for each coordinate
    ensemble_predictor : RNACoordinatePredictor
        Ensemble predictor
    output_dir : str, optional
        Output directory (default: OUTPUT_DIR)
        
    Returns:
    --------
    None
    """
    import pickle
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save individual models
    for coord in coordinate_models:
        for model_type, model_data in coordinate_models[coord].items():
            # Create filename
            filename = os.path.join(models_dir, f"{coord}_{model_type}_model.pkl")
            
            # Save model
            with open(filename, "wb") as f:
                pickle.dump(model_data, f)
    
    # Save ensemble predictor
    ensemble_file = os.path.join(models_dir, "ensemble_predictor.pkl")
    with open(ensemble_file, "wb") as f:
        pickle.dump(ensemble_predictor, f)
    
    print(f"Models saved to {models_dir}")
    return

def load_models(models_dir=os.path.join(OUTPUT_DIR, "models")):
    """
    Loads trained models from disk.
    
    Parameters:
    -----------
    models_dir : str, optional
        Directory containing saved models
        
    Returns:
    --------
    tuple
        Tuple of (coordinate_models, ensemble_predictor)
    """
    import pickle
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        print(f"Error: Models directory {models_dir} does not exist")
        return None, None
    
    # Load coordinate models
    coordinate_models = {'x': {}, 'y': {}, 'z': {}}
    
    # Find all model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith("_model.pkl")]
    
    for filename in model_files:
        # Parse filename to get coordinate and model type
        if "_model.pkl" in filename:
            parts = filename.replace("_model.pkl", "").split("_")
            if len(parts) >= 2:
                coord = parts[0]
                model_type = parts[1]
                
                if coord in coordinate_models:
                    # Load model
                    with open(os.path.join(models_dir, filename), "rb") as f:
                        model_data = pickle.load(f)
                    
                    coordinate_models[coord][model_type] = model_data
    
    # Load ensemble predictor
    ensemble_file = os.path.join(models_dir, "ensemble_predictor.pkl")
    if os.path.exists(ensemble_file):
        with open(ensemble_file, "rb") as f:
            ensemble_predictor = pickle.load(f)
    else:
        ensemble_predictor = None
    
    print(f"Models loaded from {models_dir}")
    return coordinate_models, ensemble_predictor

# Save models
save_models(coordinate_models, ensemble_predictor)

# Load models (to verify)
loaded_models, loaded_predictor = load_models()


def create_submission(test_predictions, sample_submission_df):
    """
    Creates a submission file from test predictions.
    
    Parameters:
    -----------
    test_predictions : dict
        Dictionary mapping sequence IDs to lists of structure ensembles
    sample_submission_df : pandas.DataFrame
        Sample submission DataFrame to use as a template
        
    Returns:
    --------
    pandas.DataFrame
        Submission DataFrame
    """
    # Create a copy of the sample submission
    submission_df = sample_submission_df.copy()
    
    # Process each row in the submission
    print(f"Processing row 0/{len(submission_df)}")
    for i, row in submission_df.iterrows():
        # Print progress every 1000 rows
        if i > 0 and i % 1000 == 0:
            print(f"Processing row {i}/{len(submission_df)}")
            
        # Get the ID and extract sequence ID and position
        id_parts = row['ID'].split('_')
        seq_id = id_parts[0]
        position = int(id_parts[1]) - 1  # Convert to 0-indexed
        
        # Check if we have predictions for this sequence
        if seq_id in test_predictions:
            ensemble = test_predictions[seq_id]
            
            # Fill in coordinates for each structure in the ensemble
            for struct_idx in range(1, 6):
                # Check if we have this structure in the ensemble
                if struct_idx <= len(ensemble):
                    # Check if the position is valid
                    if position < len(ensemble[struct_idx-1]):
                        coords = ensemble[struct_idx-1][position]
                        
                        # Fill in coordinates if they are valid
                        if not (np.isnan(coords).any() or np.isinf(coords).any()):
                            submission_df.at[i, f'x_{struct_idx}'] = coords[0]
                            submission_df.at[i, f'y_{struct_idx}'] = coords[1]
                            submission_df.at[i, f'z_{struct_idx}'] = coords[2]
    
    return submission_df

def prepare_test_features(test_seq_df):
    """
    Prepares features for test sequences.
    
    Parameters:
    -----------
    test_seq_df : pandas.DataFrame
        DataFrame containing test sequences
        
    Returns:
    --------
    tuple
        Tuple of (features, sequences)
    """
    # Extract sequences
    sequences = test_seq_df['sequence'].tolist()
    
    # Extract features
    features = []
    for seq in sequences:
        features.append(extract_sequence_features(seq))
    
    # Convert features to array
    all_keys = set()
    for f in features:
        all_keys.update(f.keys())
        
    feature_arrays = []
    for f in features:
        feature_array = [f.get(key, 0) for key in sorted(all_keys)]
        feature_arrays.append(feature_array)
        
    X_features = np.array(feature_arrays)
    
    print(f"Prepared test features with shape: {X_features.shape}")
    
    return X_features, sequences

# Create final submission file
def generate_final_submission():
    # Prepare test features if not already done
    test_features, test_sequences = prepare_test_features(data_dict['test_seq'])
    
    # Get test sequence IDs
    test_seq_ids = data_dict['test_seq']['target_id'].tolist()
    
    # Load the ensemble predictor or use the existing one
    if 'ensemble_predictor' not in globals():
        print("Loading ensemble predictor from disk...")
        _, loaded_predictor = load_models()
        predictor = loaded_predictor
    else:
        predictor = ensemble_predictor
    
    # Predict test structures
    print("Predicting test structures...")
    test_predictions = predict_test_structures(predictor, test_features, test_sequences, test_seq_ids)
    
    # Create submission DataFrame
    print("Creating submission DataFrame...")
    submission_df = create_submission(test_predictions, data_dict['sample_submission'])
    
    # Save submission to CSV
    submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    
    # Validate submission
    print("Validating submission...")
    is_valid = validate_submission(submission_df, data_dict['sample_submission'])
    
    if is_valid:
        print("Submission is valid and ready for Kaggle!")
    else:
        print("Warning: Submission validation failed.")
    
    return submission_df

# Run the submission generation function
final_submission = generate_final_submission()


def visualize_3d_structure_comparison(true_coords, pred_coords, title=None):
    """
    Visualizes a comparison between true and predicted 3D RNA structures.
    
    Parameters:
    -----------
    true_coords : numpy.ndarray
        True 3D coordinates
    pred_coords : numpy.ndarray
        Predicted 3D coordinates
    title : str, optional
        Plot title
    """
    # Create a figure with two subplots
    fig = plt.figure(figsize=(15, 7))
    
    # Add 3D subplots
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Filter out invalid coordinates
    true_valid = ~np.any(np.isnan(true_coords), axis=1) & ~np.any(np.isinf(true_coords), axis=1)
    pred_valid = ~np.any(np.isnan(pred_coords), axis=1) & ~np.any(np.isinf(pred_coords), axis=1)
    
    # Plot true structure if there are valid coordinates
    if np.sum(true_valid) > 2:
        true_filtered = true_coords[true_valid]
        
        # Plot backbone as a line
        ax1.plot(true_filtered[:, 0], true_filtered[:, 1], true_filtered[:, 2], 'b-', alpha=0.7)
        
        # Plot residues as points
        scatter1 = ax1.scatter(
            true_filtered[:, 0], 
            true_filtered[:, 1], 
            true_filtered[:, 2],
            c=range(len(true_filtered)),
            cmap='viridis',
            s=50,
            alpha=0.8
        )
        
        # Add title and labels
        ax1.set_title('True Structure')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Add colorbar to show sequence position
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Sequence Position')
    else:
        ax1.text(0, 0, 0, "No valid coordinates", ha='center', fontsize=14)
    
    # Plot predicted structure if there are valid coordinates
    if np.sum(pred_valid) > 2:
        pred_filtered = pred_coords[pred_valid]
        
        # Plot backbone as a line
        ax2.plot(pred_filtered[:, 0], pred_filtered[:, 1], pred_filtered[:, 2], 'r-', alpha=0.7)
        
        # Plot residues as points
        scatter2 = ax2.scatter(
            pred_filtered[:, 0], 
            pred_filtered[:, 1], 
            pred_filtered[:, 2],
            c=range(len(pred_filtered)),
            cmap='plasma',
            s=50,
            alpha=0.8
        )
        
        # Add title and labels
        ax2.set_title('Predicted Structure')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Add colorbar to show sequence position
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Sequence Position')
    else:
        ax2.text(0, 0, 0, "No valid coordinates", ha='center', fontsize=14)
    
    # Set equal aspect ratios
    ax1.set_box_aspect([1, 1, 1])
    ax2.set_box_aspect([1, 1, 1])
    
    # Set overall title if provided
    if title:
        plt.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Calculate TM-score
    tm_score = calculate_tm_score(pred_coords, true_coords)
    plt.figtext(0.5, 0.01, f'TM-score: {tm_score:.4f}', ha='center', fontsize=12)
    
    plt.show()
    
    return fig

def visualize_ensemble(ensemble, title=None):
    """
    Visualizes an ensemble of predicted structures.
    
    Parameters:
    -----------
    ensemble : list
        List of structure arrays
    title : str, optional
        Plot title
    """
    n_structures = len(ensemble)
    
    # Create a figure with subplots for each structure
    fig = plt.figure(figsize=(15, 3 * n_structures))
    
    for i, structure in enumerate(ensemble):
        # Add 3D subplot
        ax = fig.add_subplot(n_structures, 1, i+1, projection='3d')
        
        # Filter out invalid coordinates
        valid = ~np.any(np.isnan(structure), axis=1) & ~np.any(np.isinf(structure), axis=1)
        
        if np.sum(valid) > 2:
            filtered = structure[valid]
            
            # Plot backbone as a line
            ax.plot(filtered[:, 0], filtered[:, 1], filtered[:, 2], '-', alpha=0.7)
            
            # Plot residues as points
            scatter = ax.scatter(
                filtered[:, 0], 
                filtered[:, 1], 
                filtered[:, 2],
                c=range(len(filtered)),
                cmap='plasma',
                s=50,
                alpha=0.8
            )
            
            # Add title and labels
            ax.set_title(f'Structure {i+1}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        else:
            ax.text(0, 0, 0, "No valid coordinates", ha='center', fontsize=14)
    
    # Set overall title if provided
    if title:
        plt.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Example usage (with some sample data)
def demo_visualization():
    # Get a sample structure from the validation set
    if len(valid_mapping) > 0:
        sample_id = list(valid_mapping.keys())[0]
        true_structure = valid_mapping[sample_id]['structures'][0]
        sequence = valid_mapping[sample_id]['sequence']
        
        # Extract features
        sample_features = np.array([list(extract_sequence_features(sequence).values())])
        
        # Predict structure
        predicted_structures = ensemble_predictor.predict(sample_features, [sequence])
        predicted_structure = predicted_structures[0]
        
        # Optimize structure
        optimized = optimize_structure(predicted_structure, sequence)
        
        # Generate ensemble
        ensemble = generate_structure_ensemble(optimized, sequence, n_models=5)
        
        # Visualize comparison
        print(f"Visualizing structure for {sample_id}")
        visualize_3d_structure_comparison(true_structure, optimized, 
                                          title=f"RNA Structure Comparison for {sample_id}")
        
        # Visualize ensemble
        visualize_ensemble(ensemble, title=f"Structure Ensemble for {sample_id}")
    else:
        print("No validation structures available for visualization")

# Run the visualization demo
demo_visualization()  # Uncomment to run




