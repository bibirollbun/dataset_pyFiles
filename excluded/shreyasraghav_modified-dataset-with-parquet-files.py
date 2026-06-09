#!pip install Bio





%%writefile model.py

import numpy as np
import pandas as pd
import os
import re
from Bio import SeqIO
from Bio.Seq import Seq
from scipy.spatial.transform import Rotation
from scipy.spatial import distance
from scipy.optimize import minimize

# Constants for RNA structure
NUCLEOTIDE_PAIRS = {
    'A': 'U',
    'U': 'A',
    'G': 'C',
    'C': 'G'
}

# Distance between adjacent nucleotides (C1' atoms) in Angstroms
ADJACENT_NUCLEOTIDE_DISTANCE = 6.0  

# Distance between paired nucleotides (C1' atoms) in Angstroms 
BASE_PAIR_DISTANCE = 18.0

# Enhanced structural variation with RNA-specific constraints
def enhanced_structural_variation(base_structure, noise_level=0.5, preserve_distance=True, 
                               use_global_movement=False, correlation=0.8, sequence=None):
    """
    Generate structural variations with RNA-specific constraints.
    
    Parameters:
    - base_structure: Base coordinates to vary
    - noise_level: Level of random noise (higher = more variation)
    - preserve_distance: Maintain distances between adjacent residues
    - use_global_movement: Apply global movements (useful for larger RNAs)
    - correlation: Correlation between movements of adjacent residues
    - sequence: RNA sequence for sequence-aware variations
    
    Returns:
    - Varied structure coordinates
    """
    if base_structure is None or len(base_structure) == 0:
        return base_structure
    
    # Copy the base structure
    result = base_structure.copy()
    seq_length = len(result)
    
    # RNA-specific local structural patterns
    if use_global_movement:
        # Apply bending or twisting to simulate larger conformational changes
        # Get a random rotation axis
        rotation_axis = np.random.normal(0, 1, 3)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        
        # Center the structure
        center = np.mean(result, axis=0)
        centered = result - center
        
        for i in range(seq_length):
            # Calculate gradual rotation based on position
            rotation_factor = i / seq_length * np.pi * noise_level
            
            # Create rotation matrix
            rot = Rotation.from_rotvec(rotation_axis * rotation_factor)
            rotation_matrix = rot.as_matrix()
            
            # Apply rotation
            result[i] = np.dot(centered[i], rotation_matrix) + center
            
            # Add small noise based on position
            position_noise = np.random.normal(0, noise_level * 0.5, 3)
            result[i] += position_noise
    
    # Apply sequence-specific noise
    if sequence is not None:
        for i in range(seq_length):
            if i < len(sequence):
                # Different nucleotides have different flexibility patterns
                if sequence[i] == 'A' or sequence[i] == 'U':
                    # A and U tend to be more flexible
                    local_noise = noise_level * 1.2
                elif sequence[i] == 'G' or sequence[i] == 'C':
                    # G and C tend to be more rigid due to stronger base pairing
                    local_noise = noise_level * 0.8
                else:
                    local_noise = noise_level
                
                # Add nucleotide-specific noise
                result[i] += np.random.normal(0, local_noise, 3)
    
    # Add correlated noise to simulate connected movement
    correlated_noise = np.zeros_like(result)
    
    # Generate base noise
    base_noise = np.random.normal(0, noise_level, (seq_length, 3))
    
    for i in range(seq_length):
        if i == 0:
            correlated_noise[i] = base_noise[i]
        else:
            # Correlation with previous residue
            correlated_noise[i] = correlation * correlated_noise[i-1] + (1-correlation) * base_noise[i]
    
    # Apply the correlated noise
    result += correlated_noise
    
    # Preserve distances between adjacent residues if required
    if preserve_distance and seq_length > 1:
        for i in range(seq_length - 1):
            # Get the current vector between adjacent residues
            vec = result[i+1] - result[i]
            current_dist = np.linalg.norm(vec)
            
            # Target distance should be the normal C1'-C1' distance
            if current_dist > 0:  # Avoid division by zero
                # Scale the vector to maintain proper distance
                vec_scaled = vec * (ADJACENT_NUCLEOTIDE_DISTANCE / current_dist)
                
                # Update the position of the next residue
                result[i+1] = result[i] + vec_scaled
    
    return result

def apply_base_pairing_constraints(coords, sequence, min_loop_size=3):
    """
    Apply base pairing constraints to the structure by predicting
    potential base pairs from the sequence and adjusting coordinates.
    
    Parameters:
    - coords: 3D coordinates array
    - sequence: RNA sequence
    - min_loop_size: Minimum number of nucleotides in a loop
    
    Returns:
    - Adjusted coordinates that better respect RNA base pairing
    """
    if len(coords) != len(sequence):
        return coords  # Cannot apply constraints if lengths don't match
    
    result = coords.copy()
    n = len(sequence)
    
    # Simple base pair prediction using complementary bases
    pairs = []
    for i in range(n):
        for j in range(i + min_loop_size + 1, n):
            if are_complementary(sequence[i], sequence[j]):
                # The further apart, the more likely a base pair (weighted by GC content)
                if sequence[i] in 'GC':
                    weight = 1.2  # GC pairs are stronger
                else:
                    weight = 0.8
                    
                pairs.append((i, j, weight))
    
    # Sort pairs by weight (stronger pairs first)
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Track which residues are already paired
    paired = set()
    final_pairs = []
    
    # Greedy algorithm to select base pairs
    for i, j, weight in pairs:
        if i not in paired and j not in paired:
            paired.add(i)
            paired.add(j)
            final_pairs.append((i, j))
    
    # Adjust coordinates based on selected base pairs
    for i, j in final_pairs:
        # Current distance
        dist = np.linalg.norm(result[i] - result[j])
        
        # If the distance isn't close to the expected base pair distance
        if abs(dist - BASE_PAIR_DISTANCE) > 3.0:
            # Calculate midpoint
            midpoint = (result[i] + result[j]) / 2
            
            # Get direction vector
            direction = result[j] - result[i]
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            else:
                # If residues are at the same position, use a random direction
                direction = np.random.normal(0, 1, 3)
                direction = direction / np.linalg.norm(direction)
            
            # Move residues to be BASE_PAIR_DISTANCE apart
            result[i] = midpoint - direction * (BASE_PAIR_DISTANCE / 2)
            result[j] = midpoint + direction * (BASE_PAIR_DISTANCE / 2)
    
    return result

def are_complementary(n1, n2):
    """Check if nucleotides form a valid base pair"""
    return (n1 == 'A' and n2 == 'U') or \
           (n1 == 'U' and n2 == 'A') or \
           (n1 == 'G' and n2 == 'C') or \
           (n1 == 'C' and n2 == 'G')

def analyze_msa_for_covariation(msa_file):
    """
    Analyze a Multiple Sequence Alignment file to identify covarying positions
    which likely represent base pairs.
    
    Parameters:
    - msa_file: Path to MSA file in FASTA format
    
    Returns:
    - List of tuples (i,j) indicating positions that likely form base pairs
    """
    try:
        # Parse MSA file
        sequences = []
        with open(msa_file, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                sequences.append(str(record.seq).upper())
        
        if not sequences:
            return []
        
        n = len(sequences[0])
        covariation_matrix = np.zeros((n, n))
        
        # Simple mutual information calculation for covariation
        for i in range(n):
            for j in range(i + 3, n):  # Min loop size of 3
                # Count nucleotide frequencies
                i_counts = {'A': 0, 'C': 0, 'G': 0, 'U': 0, 'T': 0, '-': 0, 'N': 0}
                j_counts = {'A': 0, 'C': 0, 'G': 0, 'U': 0, 'T': 0, '-': 0, 'N': 0}
                pair_counts = {}
                
                valid_seqs = 0
                for seq in sequences:
                    if i < len(seq) and j < len(seq) and seq[i] != '-' and seq[j] != '-' and seq[i] != 'N' and seq[j] != 'N':
                        valid_seqs += 1
                        
                        # Convert T to U for RNA
                        i_nuc = 'U' if seq[i] == 'T' else seq[i]
                        j_nuc = 'U' if seq[j] == 'T' else seq[j]
                        
                        i_counts[i_nuc] += 1
                        j_counts[j_nuc] += 1
                        
                        pair = (i_nuc, j_nuc)
                        pair_counts[pair] = pair_counts.get(pair, 0) + 1
                
                # Skip if not enough valid sequences
                if valid_seqs < 5:
                    continue
                
                # Check for complementary base pair conservation
                complementary_pairs = 0
                for pair, count in pair_counts.items():
                    if are_complementary(pair[0], pair[1]):
                        complementary_pairs += count
                
                covariation_matrix[i, j] = complementary_pairs / valid_seqs
                covariation_matrix[j, i] = covariation_matrix[i, j]
        
        # Extract likely base pairs
        likely_pairs = []
        for i in range(n):
            for j in range(i + 3, n):
                if covariation_matrix[i, j] > 0.6:  # Threshold for base pair prediction
                    likely_pairs.append((i, j))
        
        return likely_pairs
    
    except Exception as e:
        print(f"Error analyzing MSA file: {e}")
        return []

def apply_msa_constraints(coords, sequence, msa_file):
    """
    Apply constraints from MSA analysis to the structure
    
    Parameters:
    - coords: 3D coordinates array
    - sequence: RNA sequence
    - msa_file: Path to MSA file
    
    Returns:
    - Adjusted coordinates based on MSA analysis
    """
    if msa_file is None or not os.path.exists(msa_file):
        return coords
    
    result = coords.copy()
    
    try:
        # Get likely base pairs from MSA
        likely_pairs = analyze_msa_for_covariation(msa_file)
        
        # Apply base pair constraints
        for i, j in likely_pairs:
            if i < len(coords) and j < len(coords):
                # Current distance
                dist = np.linalg.norm(result[i] - result[j])
                
                # If the distance isn't close to the expected base pair distance
                if abs(dist - BASE_PAIR_DISTANCE) > 3.0:
                    # Calculate midpoint
                    midpoint = (result[i] + result[j]) / 2
                    
                    # Get direction vector
                    direction = result[j] - result[i]
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                    else:
                        # If residues are at the same position, use a random direction
                        direction = np.random.normal(0, 1, 3)
                        direction = direction / np.linalg.norm(direction)
                    
                    # Move residues to be BASE_PAIR_DISTANCE apart
                    result[i] = midpoint - direction * (BASE_PAIR_DISTANCE / 2)
                    result[j] = midpoint + direction * (BASE_PAIR_DISTANCE / 2)
    
    except Exception as e:
        print(f"Error applying MSA constraints: {e}")
    
    return result

def normalize_structure(structure):
    """
    Normalizes the structure by centering and scaling.
    
    Parameters:
    - structure: 3D coordinates array
    
    Returns:
    - Normalized structure
    """
    if structure is None or len(structure) == 0:
        return structure
    
    result = structure.copy()
    
    # Center the structure
    center = np.mean(result, axis=0)
    result = result - center
    
    # Scale to a consistent size
    max_dist = 0
    for coord in result:
        dist = np.linalg.norm(coord)
        if dist > max_dist:
            max_dist = dist
    
    if max_dist > 0:
        # Scale the structure to a consistent radius
        result = result * (50.0 / max_dist)
    
    return result

def generate_diverse_ensemble(base_structure, sequence, msa_file=None, num_structures=5):
    """
    Generate a diverse ensemble of structures for a given RNA sequence
    
    Parameters:
    - base_structure: Base coordinates to start from
    - sequence: RNA sequence
    - msa_file: Optional path to MSA file for that sequence
    - num_structures: Number of structures to generate (default: 5)
    
    Returns:
    - List of structures (each with 3D coordinates)
    """
    structures = []
    seq_length = len(sequence)
    
    # First structure: Apply base pairing and MSA constraints
    base_with_constraints = apply_base_pairing_constraints(base_structure, sequence)
    if msa_file:
        base_with_constraints = apply_msa_constraints(base_with_constraints, sequence, msa_file)
    structures.append(normalize_structure(base_with_constraints))
    
    # Second structure: Higher noise with global movement
    structure2 = enhanced_structural_variation(
        base_structure, 
        noise_level=0.8, 
        preserve_distance=True,
        use_global_movement=True,
        sequence=sequence
    )
    structure2 = apply_base_pairing_constraints(structure2, sequence)
    structures.append(normalize_structure(structure2))
    
    # Third structure: Medium noise, no global movement
    structure3 = enhanced_structural_variation(
        base_structure, 
        noise_level=0.5, 
        preserve_distance=True,
        use_global_movement=False,
        sequence=sequence
    )
    structure3 = apply_base_pairing_constraints(structure3, sequence)
    structures.append(normalize_structure(structure3))
    
    # Fourth structure: Low noise but different rotation
    structure4 = base_structure.copy()
    
    # Apply random rotation
    center = np.mean(structure4, axis=0)
    centered = structure4 - center
    
    # Random rotation matrix
    rotation = Rotation.random()
    rotation_matrix = rotation.as_matrix()
    
    # Apply rotation and re-center
    for i in range(len(structure4)):
        structure4[i] = np.dot(centered[i], rotation_matrix) + center
    
    structure4 = enhanced_structural_variation(
        structure4, 
        noise_level=0.3, 
        preserve_distance=True,
        use_global_movement=False,
        sequence=sequence
    )
    structure4 = apply_base_pairing_constraints(structure4, sequence)
    structures.append(normalize_structure(structure4))
    
    # Fifth structure: Significantly different conformation
    # Create a different starting point with a more open conformation
    structure5 = np.zeros_like(base_structure)
    
    # Create an elongated starting structure
    for i in range(seq_length):
        structure5[i] = np.array([i * ADJACENT_NUCLEOTIDE_DISTANCE * 0.8, 0, 0])
    
    # Apply high noise and global movement
    structure5 = enhanced_structural_variation(
        structure5, 
        noise_level=1.0, 
        preserve_distance=True,
        use_global_movement=True,
        sequence=sequence
    )
    structure5 = apply_base_pairing_constraints(structure5, sequence)
    structures.append(normalize_structure(structure5))
    
    # Ensure we have exactly 5 structures
    while len(structures) < 5:
        # Create additional structures if needed
        noise = 0.5 + len(structures) * 0.2
        new_structure = enhanced_structural_variation(
            base_structure, 
            noise_level=noise, 
            preserve_distance=True,
            use_global_movement=(len(structures) % 2 == 0),
            sequence=sequence
        )
        new_structure = apply_base_pairing_constraints(new_structure, sequence)
        structures.append(normalize_structure(new_structure))
    
    return structures[:5]  # Return exactly 5 structures

class ImprovedRNAModel:
    """
    Improved RNA 3D structure prediction model that incorporates:
    - Reference-based modeling
    - Sequence-specific structural features
    - MSA-based constraints
    - RNA-specific geometry constraints
    - Ensemble prediction
    """
    
    def __init__(self, geometric_sampling=True, base_noise_level=0.5, correlation=0.8, msa_dir=None):
        """
        Initialize the RNA 3D structure model
        
        Parameters:
        - geometric_sampling: Whether to use geometric sampling for structure variation
        - base_noise_level: Base level of noise for variations
        - correlation: Correlation between adjacent residues' movements
        - msa_dir: Directory containing MSA files
        """
        self.geometric_sampling = geometric_sampling
        self.base_noise_level = base_noise_level
        self.correlation = correlation
        self.msa_dir = msa_dir
        
        # Will be populated during fit
        self.reference_structures = []
        self.reference_sequences = []
        self.size_groups = {'small': [], 'medium': [], 'large': []}
        self.global_mean = None
        self.global_std = None
        
    def fit(self, X, y):
        """
        Fit the model using reference structures
        
        Parameters:
        - X: Features (one-hot encoded RNA sequences)
        - y: 3D coordinates of reference structures
        """
        n_samples = len(X)
        self.reference_structures = []
        self.reference_sequences = []
        
        # Extract sequences from one-hot encoding
        nucleotides = ['A', 'C', 'G', 'U', 'N']
        
        all_coords = []
        
        for i in range(n_samples):
            # Get the sequence length by finding the first all-zero row
            seq_length = X[i].shape[0]
            for j in range(X[i].shape[0]):
                if np.all(X[i][j] == 0):
                    seq_length = j
                    break
            
            # Extract the sequence
            sequence = ''
            for j in range(seq_length):
                idx = np.argmax(X[i][j])
                if idx < len(nucleotides):
                    sequence += nucleotides[idx]
                else:
                    sequence += 'N'
            
            # Get coordinates for this sequence
            coords = y[i][:seq_length]
            
            # Store valid reference structures
            if not np.isnan(coords).any() and not np.all(coords == 0):
                self.reference_structures.append(coords)
                self.reference_sequences.append(sequence)
                
                # Categorize by size
                if seq_length < 60:
                    self.size_groups['small'].append(len(self.reference_structures) - 1)
                elif seq_length < 200:
                    self.size_groups['medium'].append(len(self.reference_structures) - 1)
                else:
                    self.size_groups['large'].append(len(self.reference_structures) - 1)
                
                all_coords.append(coords)
        
        # Calculate global statistics
        if all_coords:
            all_coords_array = np.vstack(all_coords)
            self.global_mean = np.mean(all_coords_array, axis=0)
            self.global_std = np.std(all_coords_array, axis=0)
        else:
            self.global_mean = np.zeros(3)
            self.global_std = np.ones(3) * 10  # Default standard deviation
            
        print(f"Model fit with {len(self.reference_structures)} reference structures")
        print(f"Size groups: Small={len(self.size_groups['small'])}, "
              f"Medium={len(self.size_groups['medium'])}, "
              f"Large={len(self.size_groups['large'])}")
    
    def predict(self, X):
        """
        Predict 3D structures for RNA sequences
        
        Parameters:
        - X: Features (one-hot encoded RNA sequences)
        
        Returns:
        - Predicted 3D coordinates
        """
        n_samples = len(X)
        predictions = []
        
        nucleotides = ['A', 'C', 'G', 'U', 'N']
        
        for i in range(n_samples):
            # Get the sequence length by finding the first all-zero row
            seq_length = X[i].shape[0]
            for j in range(X[i].shape[0]):
                if np.all(X[i][j] == 0):
                    seq_length = j
                    break
            
            # Extract the sequence
            sequence = ''
            for j in range(seq_length):
                if j < X[i].shape[0]:
                    idx = np.argmax(X[i][j])
                    if idx < len(nucleotides):
                        sequence += nucleotides[idx]
                    else:
                        sequence += 'N'
                else:
                    break
            
            # Find MSA file if available
            msa_file = None
            if self.msa_dir is not None:
                # Extract sequence ID from one-hot encoded representation
                # This requires knowledge of how the sequence IDs are stored
                target_id = f"seq_{i+1}"  # Default fallback
                
                # Look for MSA file
                potential_msa_file = os.path.join(self.msa_dir, f"{target_id}.MSA.fasta")
                if os.path.exists(potential_msa_file):
                    msa_file = potential_msa_file
            
            # Adjust noise level based on sequence length
            if seq_length < 60:
                group = "small"
                noise_level = self.base_noise_level * 1.5
            elif seq_length < 200:
                group = "medium"
                noise_level = self.base_noise_level * 1.0
            else:
                group = "large"
                noise_level = self.base_noise_level * 0.6
            
            # Create a prediction for this sequence
            base_struct = None
            
            # If we have reference structures in this size group, use them
            if group in self.size_groups and self.size_groups[group]:
                # Find the most similar sequence in terms of length
                best_idx = -1
                best_length_diff = float('inf')
                
                for idx in self.size_groups[group]:
                    ref_seq = self.reference_sequences[idx]
                    length_diff = abs(len(ref_seq) - seq_length)
                    
                    if length_diff < best_length_diff:
                        best_length_diff = length_diff
                        best_idx = idx
                
                if best_idx >= 0 and best_length_diff <= seq_length * 0.5:  # Only use if reasonably close
                    base_struct = self.reference_structures[best_idx].copy()
                    
                    # If reference is shorter, extend it
                    if len(base_struct) < seq_length:
                        extension = np.zeros((seq_length - len(base_struct), 3))
                        # Extrapolate the last few positions to extend
                        if len(base_struct) > 2:
                            direction = base_struct[-1] - base_struct[-2]
                            for j in range(seq_length - len(base_struct)):
                                extension[j] = base_struct[-1] + direction * (j + 1)
                        base_struct = np.vstack([base_struct, extension])
                    
                    # If reference is longer, truncate it
                    if len(base_struct) > seq_length:
                        base_struct = base_struct[:seq_length]
                    
                    # Apply variation
                    if self.geometric_sampling:
                        pred = enhanced_structural_variation(
                            base_struct, 
                            noise_level=noise_level,
                            preserve_distance=True,
                            use_global_movement=(group == "small"),
                            correlation=self.correlation,
                            sequence=sequence
                        )
                    else:
                        noise = np.random.normal(0, noise_level, base_struct.shape)
                        pred = base_struct + noise
            
            # Fall back to a generic structure if no suitable reference found
            if base_struct is None:
                # Create a simple extended chain as starting structure
                pred = np.zeros((seq_length, 3))
                for j in range(seq_length):
                    pred[j] = np.array([j * ADJACENT_NUCLEOTIDE_DISTANCE, 0, 0])
                
                # Apply variation to make it more realistic
                if self.geometric_sampling:
                    pred = enhanced_structural_variation(
                        pred, 
                        noise_level=noise_level * 2,  # Higher noise for generic structures
                        preserve_distance=True,
                        use_global_movement=True,
                        correlation=self.correlation,
                        sequence=sequence
                    )
                else:
                    noise = np.random.normal(0, noise_level * 2, pred.shape)
                    pred = pred + noise
            
            # Apply base pairing constraints
            pred = apply_base_pairing_constraints(pred, sequence)
            
            # Apply MSA constraints if available
            if msa_file:
                pred = apply_msa_constraints(pred, sequence, msa_file)
            
            # Pad with zeros to match maximum length
            if len(pred) < X[i].shape[0]:
                padded_pred = np.zeros((X[i].shape[0], 3))
                padded_pred[:len(pred)] = pred
                pred = padded_pred
            
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_ensemble(self, X, test_seq_df=None):
        """
        Generate an ensemble of predictions for each sequence
        
        Parameters:
        - X: Features (one-hot encoded RNA sequences)
        - test_seq_df: DataFrame with sequence information (for MSA lookup)
        
        Returns:
        - Dictionary mapping sequence IDs to lists of structures
        """
        n_samples = len(X)
        ensemble_predictions = {}
        
        nucleotides = ['A', 'C', 'G', 'U', 'N']
        
        for i in range(n_samples):
            # Extract target_id if available
            target_id = f"seq_{i+1}"  # Default
            if test_seq_df is not None and i < len(test_seq_df):
                target_id = test_seq_df.iloc[i]['target_id']
            
            # Get the sequence length
            seq_length = X[i].shape[0]
            for j in range(X[i].shape[0]):
                if np.all(X[i][j] == 0):
                    seq_length = j
                    break
            
            # Extract the sequence
            sequence = ''
            for j in range(seq_length):
                if j < X[i].shape[0]:
                    idx = np.argmax(X[i][j])
                    if idx < len(nucleotides):
                        sequence += nucleotides[idx]
                    else:
                        sequence += 'N'
                else:
                    break
            
            # Find MSA file if available
            msa_file = None
            if self.msa_dir is not None:
                potential_msa_files = [
                    os.path.join(self.msa_dir, f"{target_id}.MSA.fasta"),
                    os.path.join(self.msa_dir, f"{target_id.split('_')[0]}.MSA.fasta")
                ]
                
                for potential_file in potential_msa_files:
                    if os.path.exists(potential_file):
                        msa_file = potential_file
                        break
            
            # Get base prediction
            base_prediction = self.predict([X[i]])[0][:seq_length]
            
            # Generate diverse ensemble
            structures = generate_diverse_ensemble(
                base_prediction, 
                sequence, 
                msa_file=msa_file
            )
            
            # Store all structures for this sequence
            ensemble_predictions[target_id] = structures
        
        return ensemble_predictions 


import os
import time
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import traceback
from tqdm import tqdm
from Bio import PDB
from Bio.PDB import PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import urllib.request
import gzip
import shutil
import requests
from io import StringIO
import json
import pickle
import sys



# Import our improved model
from model import (
    ImprovedRNAModel, 
    enhanced_structural_variation,
    apply_base_pairing_constraints,
    apply_msa_constraints,
    normalize_structure,
    generate_diverse_ensemble,
    ADJACENT_NUCLEOTIDE_DISTANCE,
    BASE_PAIR_DISTANCE
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


# Directories and files
DATA_DIR = os.getenv('DATA_DIR', '/kaggle/input/stanford-rna-3d-folding/')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', '/kaggle/working/')
MSA_DIR = os.path.join(DATA_DIR, "MSA")
EXTERNAL_DATA_DIR = os.path.join(OUTPUT_DIR, "external_data")
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")

# Add path to pre-downloaded RNA data
RNA_DATA_DIR = "rna_data"
RNA_STRUCTURES_DIR = os.path.join(RNA_DATA_DIR, "structures")
RNA_CHAINS_CACHE = os.path.join(RNA_DATA_DIR, "rna_chains.pkl")

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXTERNAL_DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Local cache for RNA structures
RNA_STRUCTURES_CACHE = os.path.join(CACHE_DIR, "rna_structures.pkl")

# Hardcoded RNA PDB IDs as fallback
FALLBACK_PDB_IDS = [
    "6th6", "6adr", "6qir", "6vwl", "6zu0", "7dk6", "7eag",
    "7egp", "4u3l", "4u3k", "1bgz", "1eh4", "1eht", "1equ",
    "1exd", "1f7y", "1jid", "1kd1", "1kxk", "1l8v", "1o0c",
    "1r3o", "1s72", "1x8w", "1y27", "2oe5", "2zni", "3egz",
    "3g78", "3gx5", "3hxm", "3j0o", "3j0p", "3j0q", "3j0r"
]



# Hard-coded RNA structures as fallback (pre-parsed chains)
HARDCODED_RNA_CHAINS = [
    # A simplified representation of RNA chains with sequence and coordinates
    # These will be used if no internet connection is available
    # Format: {'chain_id': 'A', 'sequence': 'ACGU...', 'coordinates': numpy array of shape (seq_len, 3)}
]

# Local RNA structure files (add your own PDB files here)
LOCAL_RNA_STRUCTURES = [
    # Add paths to local RNA structure files if available
]

# Add paths to external Ribonanza data files
RIBONANZA_SEQ_FILE = "/kaggle/input/parquet-files-for-stanford/ext_ribonanza_labels.parquet"
RIBONANZA_LABELS_FILE = "/kaggle/input/parquet-files-for-stanford/ext_ribonanza_sequences.parquet"

# Set offline mode as default
os.environ['OFFLINE_MODE'] = 'True'


# ===== DEPENDENCY MANAGEMENT =====
def check_and_install_dependencies():
    """
    Check if required packages are installed and install them locally if needed
    """
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'tqdm': 'tqdm',
        'biopython': 'Bio',
        'requests': 'requests'
    }
    
    missing_packages = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name} is installed")
        except ImportError:
            missing_packages.append(package_name)
            print(f"✗ {package_name} is missing")
    
    if missing_packages:
        print("\nMissing packages detected. Please install them before running:")
        for package in missing_packages:
            print(f"pip install {package}")
        print("\nFor offline submission, download packages with:")
        print("pip download -d ./packages numpy pandas matplotlib tqdm biopython requests")
        print("Then install offline with:")
        print("pip install --no-index --find-links=./packages -r requirements.txt")
        
        # Create requirements.txt file
        with open("requirements.txt", "w") as f:
            for package in required_packages.keys():
                f.write(f"{package}\n")
        
        print("\nrequirements.txt has been created.")
        return False
    
    return True


# ===== EXTERNAL DATA LOADING =====
def download_pdb_structure(pdb_id, output_dir=EXTERNAL_DATA_DIR):
    """
    Download PDB structure from RCSB PDB database - DISABLED FOR OFFLINE USE
    """
    print(f"Offline mode: Cannot download {pdb_id} from internet")
    return None


def download_rna_structures_from_rna3dhub(output_dir=EXTERNAL_DATA_DIR, max_structures=100):
    """
    Download RNA structures from RNA 3D Hub - DISABLED FOR OFFLINE USE
    """
    print("Offline mode: Cannot download RNA structures from internet")
    return []


def parse_pdb_structure(structure_file):
    """
    Parse a PDB/mmCIF file and extract RNA chain information
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = None
        
        # Determine file type and parse accordingly
        if structure_file.endswith('.pdb'):
            structure = parser.get_structure('RNA', structure_file)
        elif structure_file.endswith('.cif'):
            # Use MMCIF parser for cif files
            from Bio.PDB.MMCIFParser import MMCIFParser
            mmcif_parser = MMCIFParser(QUIET=True)
            structure = mmcif_parser.get_structure('RNA', structure_file)
        else:
            print(f"Unsupported file format for {structure_file}")
            return []
        
        if structure is None:
            return []
        
        # Extract RNA chains
        rna_chains = []
        
        for model in structure:
            for chain in model:
                # Check if this is an RNA chain
                is_rna = False
                sequence = ""
                coordinates = []
                
                for residue in chain:
                    # Check if it's a nucleotide (contains C1' atom)
                    if 'C1\'' in residue:
                        is_rna = True
                        # Extract the nucleotide identity
                        if residue.get_resname() in ['A', 'C', 'G', 'U']:
                            nucleotide = residue.get_resname()
                        elif residue.get_resname() in ['DA', 'DC', 'DG', 'DT']:
                            # Convert DNA to RNA
                            dna_to_rna = {'DA': 'A', 'DC': 'C', 'DG': 'G', 'DT': 'U'}
                            nucleotide = dna_to_rna.get(residue.get_resname(), 'N')
                        else:
                            # Handle other nucleotide naming conventions
                            if 'A' in residue.get_resname():
                                nucleotide = 'A'
                            elif 'C' in residue.get_resname():
                                nucleotide = 'C'
                            elif 'G' in residue.get_resname():
                                nucleotide = 'G'
                            elif 'U' in residue.get_resname() or 'T' in residue.get_resname():
                                nucleotide = 'U'
                            else:
                                nucleotide = 'N'
                        
                        sequence += nucleotide
                        
                        # Get C1' atom coordinates
                        try:
                            c1_atom = residue['C1\'']
                            coordinates.append([c1_atom.get_coord()[0], 
                                              c1_atom.get_coord()[1], 
                                              c1_atom.get_coord()[2]])
                        except KeyError:
                            # If C1' is not available, try to use another atom
                            for atom in residue:
                                if atom.get_name() in ['P', 'C4\'']:
                                    coordinates.append([atom.get_coord()[0], 
                                                      atom.get_coord()[1], 
                                                      atom.get_coord()[2]])
                                    break
                            else:
                                # If no suitable atom found, use average of all atoms
                                all_coords = np.array([atom.get_coord() for atom in residue])
                                avg_coord = np.mean(all_coords, axis=0)
                                coordinates.append([avg_coord[0], avg_coord[1], avg_coord[2]])
                
                if is_rna and len(sequence) >= 10:  # Only consider chains with at least 10 nucleotides
                    rna_chains.append({
                        'chain_id': chain.get_id(),
                        'sequence': sequence,
                        'coordinates': np.array(coordinates)
                    })
        
        return rna_chains
    
    except Exception as e:
        print(f"Error parsing structure file {structure_file}: {str(e)}")
        return []



def load_external_rna_structures(max_structures=100):
    """
    Load external RNA structures from local files or generate synthetic data
    """
    print("Loading external RNA structures...")
    
    # First check if we have pre-downloaded data from prepare_rna_data.py
    if os.path.exists(RNA_CHAINS_CACHE):
        try:
            print(f"Loading pre-downloaded RNA chains from: {RNA_CHAINS_CACHE}")
            with open(RNA_CHAINS_CACHE, 'rb') as f:
                rna_chains = pickle.load(f)
            print(f"Loaded {len(rna_chains)} pre-downloaded RNA chains")
            return rna_chains
        except Exception as e:
            print(f"Error loading pre-downloaded RNA chains: {str(e)}")
    # Check if we have cached structures from previous runs
    if os.path.exists(RNA_STRUCTURES_CACHE):
        try:
            print(f"Loading RNA structures from cache: {RNA_STRUCTURES_CACHE}")
            with open(RNA_STRUCTURES_CACHE, 'rb') as f:
                rna_chains = pickle.load(f)
            print(f"Loaded {len(rna_chains)} RNA chains from cache")
            return rna_chains
        except Exception as e:
            print(f"Error loading cached structures: {str(e)}")
    
    # Check for pre-downloaded structure files
    pre_downloaded_files = []
    if os.path.exists(RNA_STRUCTURES_DIR):
        pre_downloaded_files = [
            os.path.join(RNA_STRUCTURES_DIR, f) 
            for f in os.listdir(RNA_STRUCTURES_DIR) 
            if f.endswith(('.pdb', '.cif'))
        ]
        if pre_downloaded_files:
            print(f"Found {len(pre_downloaded_files)} pre-downloaded structure files")
    
    # Try to load pre-downloaded structure files
    if pre_downloaded_files:
        print(f"Loading {len(pre_downloaded_files)} pre-downloaded structure files")
        rna_chains = []
        for structure_file in tqdm(pre_downloaded_files, desc="Parsing pre-downloaded structures"):
            chains = parse_pdb_structure(structure_file)
            rna_chains.extend(chains)
        
        print(f"Loaded {len(rna_chains)} RNA chains from pre-downloaded structures")
        
        # Cache the results
        if len(rna_chains) > 0:
            try:
                with open(RNA_STRUCTURES_CACHE, 'wb') as f:
                    pickle.dump(rna_chains, f)
                print(f"Cached {len(rna_chains)} RNA chains for future use")
            except Exception as e:
                print(f"Error caching RNA chains: {str(e)}")
            
        return rna_chains
    elif HARDCODED_RNA_CHAINS:
        print("Using hardcoded RNA structures")
        return HARDCODED_RNA_CHAINS
    else:
        # Generate synthetic RNA structures as fallback
        print("Generating synthetic RNA structures as fallback")
        return generate_synthetic_rna_structures(20)



def generate_synthetic_rna_structures(num_structures=20):
    """
    Generate synthetic RNA structures as fallback
    """
    rna_chains = []
    
    for i in range(num_structures):
        # Generate random RNA sequence of length 30-100
        length = np.random.randint(30, 101)
        nucleotides = ['A', 'C', 'G', 'U']
        sequence = ''.join(np.random.choice(nucleotides) for _ in range(length))
        
        # Generate random 3D coordinates in a realistic range
        # Start with a linear chain
        coords = np.zeros((length, 3))
        for j in range(length):
            # Add some randomness to the linear structure
            if j == 0:
                coords[j] = np.random.normal(0, 1, 3)
            else:
                # Average distance between adjacent nucleotides ~6Å
                direction = np.random.normal(0, 1, 3)
                direction = direction / np.linalg.norm(direction) * 6.0
                coords[j] = coords[j-1] + direction
        
        rna_chains.append({
            'chain_id': f'synthetic_{i}',
            'sequence': sequence,
            'coordinates': coords
        })
    
    return rna_chains



def convert_external_data_to_features(rna_chains, max_length=720):
    """
    Convert external RNA structures to feature format compatible with our model
    """
    if not rna_chains:
        print("No external RNA chains available, skipping conversion")
        return np.array([]), np.array([])
        
    X_external = []
    y_external = []
    
    for chain in rna_chains:
        sequence = chain['sequence']
        coordinates = chain['coordinates']
        
        # Skip if sequence is too long or coordinates don't match
        if len(sequence) > max_length or len(sequence) != len(coordinates):
            continue
        
        # Convert sequence to one-hot encoded features
        features = []
        for nucleotide in sequence:
            if nucleotide == 'A':
                features.append([1, 0, 0, 0, 0])
            elif nucleotide == 'C':
                features.append([0, 1, 0, 0, 0])
            elif nucleotide == 'G':
                features.append([0, 0, 1, 0, 0])
            elif nucleotide == 'U':
                features.append([0, 0, 0, 1, 0])
            else:
                features.append([0, 0, 0, 0, 1])
        
        # Pad or truncate features
        if len(features) < max_length:
            padding = [[0, 0, 0, 0, 0]] * (max_length - len(features))
            features.extend(padding)
        else:
            features = features[:max_length]
        
        # Pad or truncate coordinates
        coords = np.zeros((max_length, 3))
        coords[:len(coordinates)] = coordinates
        
        X_external.append(features)
        y_external.append(coords)
    
    return np.array(X_external), np.array(y_external)



# ===== COMPETITION DATA LOADING =====
def load_competition_data():
    """
    Load main data files for the competition.
    """
    print("Loading competition data...")
    main_files = {
        "train_sequences": os.path.join(DATA_DIR, "train_sequences.csv"),
        "train_labels": os.path.join(DATA_DIR, "train_labels.csv"),
        "validation_sequences": os.path.join(DATA_DIR, "validation_sequences.csv"),
        "validation_labels": os.path.join(DATA_DIR, "validation_labels.csv"),
        "test_sequences": os.path.join(DATA_DIR, "test_sequences.csv"),
        "sample_submission": os.path.join(DATA_DIR, "sample_submission.csv")
    }
    
    data = {}
    for name, file_path in main_files.items():
        if os.path.exists(file_path):
            data[name] = pd.read_csv(file_path)
            print(f"Loaded {name}: {data[name].shape}")
        else:
            print(f"Warning: File {file_path} not found")
            data[name] = None
    
    return data


def prepare_features(seq_df, max_length=720):
    """
    Prepare one-hot encoded features from RNA sequences.
    """
    print(f"Preparing features for {len(seq_df)} sequences...")
    X = []
    for _, row in seq_df.iterrows():
        seq = row['sequence']
        features = []
        for nucleotide in seq:
            if nucleotide == 'A':
                features.append([1, 0, 0, 0, 0])
            elif nucleotide == 'C':
                features.append([0, 1, 0, 0, 0])
            elif nucleotide == 'G':
                features.append([0, 0, 1, 0, 0])
            elif nucleotide == 'U':
                features.append([0, 0, 0, 1, 0])
            else:
                features.append([0, 0, 0, 0, 1])
        
        # Pad or truncate to max_length
        if len(features) < max_length:
            padding = [[0, 0, 0, 0, 0]] * (max_length - len(features))
            features.extend(padding)
        else:
            features = features[:max_length]
        
        X.append(features)
    
    return np.array(X)


def prepare_labels(labels_df, seq_df, max_length=720):
    """
    Prepare 3D coordinate labels from label dataframe.
    """
    print(f"Preparing labels for {len(seq_df)} sequences...")
    y = []
    
    # Group labels by sequence
    for i, row in seq_df.iterrows():
        target_id = row['target_id']
        seq_length = len(row['sequence'])
        
        # Extract all residues for this sequence
        seq_labels = labels_df[labels_df['ID'].str.startswith(f"{target_id}_")]
        
        # Create coordinates array
        coords = np.zeros((max_length, 3))
        
        if not seq_labels.empty:
            for _, label_row in seq_labels.iterrows():
                resid = int(label_row['resid'])
                if resid <= max_length:
                    coords[resid-1, 0] = label_row['x_1']
                    coords[resid-1, 1] = label_row['y_1']
                    coords[resid-1, 2] = label_row['z_1']
        
        y.append(coords)
    
    return np.array(y)



# ===== ADDITIONAL EXTERNAL DATA LOADING =====
def load_ribonanza_data(max_length=720):
    """
    Load external Ribonanza data from parquet files
    """
    print("Loading Ribonanza external data...")
    
    # Check if files exist
    if not os.path.exists(RIBONANZA_SEQ_FILE) or not os.path.exists(RIBONANZA_LABELS_FILE):
        print("Ribonanza parquet files not found, skipping this data source")
        return np.array([]), np.array([])
    
    try:
        # Load the data
        sequences_df = pd.read_parquet(RIBONANZA_SEQ_FILE)
        labels_df = pd.read_parquet(RIBONANZA_LABELS_FILE)
        
        print(f"Loaded Ribonanza data: {len(sequences_df)} sequences")
        
        # Prepare features and labels
        X_ribo = []
        y_ribo = []
        
        for _, row in tqdm(sequences_df.iterrows(), total=len(sequences_df), desc="Processing Ribonanza sequences"):
            seq_id = row.get('sequence_id', row.get('ID', None))
            if seq_id is None:
                continue
                
            sequence = row.get('sequence', '')
            if not sequence:
                continue
            
            # Get coordinates for this sequence
            coords = labels_df[labels_df['sequence_id'] == seq_id]
            if coords.empty:
                continue
            
            # Convert sequence to one-hot encoded features
            features = []
            for nucleotide in sequence:
                if nucleotide == 'A':
                    features.append([1, 0, 0, 0, 0])
                elif nucleotide == 'C':
                    features.append([0, 1, 0, 0, 0])
                elif nucleotide == 'G':
                    features.append([0, 0, 1, 0, 0])
                elif nucleotide == 'U':
                    features.append([0, 0, 0, 1, 0])
                else:
                    features.append([0, 0, 0, 0, 1])
            
            # Check if sequence is too long
            if len(features) > max_length:
                print(f"Sequence {seq_id} is too long ({len(features)} > {max_length}), truncating")
                features = features[:max_length]
            
            # Pad if needed
            if len(features) < max_length:
                padding = [[0, 0, 0, 0, 0]] * (max_length - len(features))
                features.extend(padding)
            
            # Process coordinates
            coordinates = np.zeros((max_length, 3))
            
            # Check which columns contain 3D coordinates
            coord_columns = []
            for col in coords.columns:
                if col.startswith('x_') or col.startswith('y_') or col.startswith('z_'):
                    coord_columns.append(col)
            
            if not coord_columns:
                # Try to look for other coordinate column patterns
                x_cols = [col for col in coords.columns if 'x' in col.lower()]
                y_cols = [col for col in coords.columns if 'y' in col.lower()]
                z_cols = [col for col in coords.columns if 'z' in col.lower()]
                
                if x_cols and y_cols and z_cols:
                    # Use first set of coordinate columns found
                    for i, row in coords.iterrows():
                        pos = min(int(i), max_length-1)
                        coordinates[pos, 0] = row[x_cols[0]]
                        coordinates[pos, 1] = row[y_cols[0]]
                        coordinates[pos, 2] = row[z_cols[0]]
            else:
                # Use standard coordinate columns
                for i, row in coords.iterrows():
                    position = min(int(i), max_length-1)
                    x_col = next((col for col in coord_columns if col.startswith('x_')), None)
                    y_col = next((col for col in coord_columns if col.startswith('y_')), None)
                    z_col = next((col for col in coord_columns if col.startswith('z_')), None)
                    
                    if x_col and y_col and z_col:
                        coordinates[position, 0] = row[x_col]
                        coordinates[position, 1] = row[y_col]
                        coordinates[position, 2] = row[z_col]
            
            # Skip if all coordinates are zero (no actual data)
            if np.all(coordinates == 0):
                continue
                
            X_ribo.append(features)
            y_ribo.append(coordinates)
        
        print(f"Prepared {len(X_ribo)} Ribonanza sequences with 3D coordinates")
        return np.array(X_ribo), np.array(y_ribo)
        
    except Exception as e:
        print(f"Error loading Ribonanza data: {str(e)}")
        traceback.print_exc()
        return np.array([]), np.array([])



def load_all_data(max_length=720, use_external=True, max_external_structures=100):
    """
    Load and process all data including external structures if specified.
    """
    # Load competition data
    competition_data = load_competition_data()
    
    # Prepare competition training data
    X_train = prepare_features(competition_data["train_sequences"], max_length)
    y_train = prepare_labels(competition_data["train_labels"], competition_data["train_sequences"], max_length)
    
    # Prepare competition validation data
    X_valid = prepare_features(competition_data["validation_sequences"], max_length)
    y_valid = prepare_labels(competition_data["validation_labels"], competition_data["validation_sequences"], max_length)
    
    # Load external data if specified
    if use_external:
        print("\nLoading external RNA structure data...")
        print("Running in OFFLINE MODE - only local data will be used")
        
        # Load RNA 3D Hub and PDB data from local files only
        rna_chains = load_external_rna_structures(max_structures=max_external_structures)
        X_external, y_external = convert_external_data_to_features(rna_chains, max_length)
        
        print(f"External RNA 3D Hub data: {len(X_external)} structures")
        
        # Load Ribonanza data
        X_ribo, y_ribo = load_ribonanza_data(max_length)
        print(f"External Ribonanza data: {len(X_ribo)} structures")
        
        # Combine all external data sources
        all_external_data = []
        
        if len(X_external) > 0:
            all_external_data.append((X_external, y_external))
            
        if len(X_ribo) > 0:
            all_external_data.append((X_ribo, y_ribo))
        
        # Combine competition and external data if we have external data
        if all_external_data:
            X_train_combined = X_train
            y_train_combined = y_train
            
            for X_ext, y_ext in all_external_data:
                X_train_combined = np.vstack([X_train_combined, X_ext])
                y_train_combined = np.vstack([y_train_combined, y_ext])
            
            print(f"Combined training data: {X_train_combined.shape}")
            
            return X_train_combined, y_train_combined, X_valid, y_valid, competition_data
    
    # Fall back to just competition data if no external data or not requested
    print(f"Using only competition data: {X_train.shape}")
    return X_train, y_train, X_valid, y_valid, competition_data



# ===== MODEL TRAINING & EVALUATION =====
def calculate_tm_score_approx(pred, true):
    """
    Calculate an approximation of TM-score.
    
    TM-score measures the similarity of two protein structures.
    This is a simplified version that doesn't perform optimal alignment.
    """
    # Filter out padding (zeros)
    mask = ~np.all(true == 0, axis=1)
    if not np.any(mask):
        return 0.0
    
    true_filtered = true[mask]
    pred_filtered = pred[mask]
    
    Lref = len(true_filtered)
    
    # Calculate d0 scaling factor
    if Lref >= 30:
        d0 = 1.24 * np.power(Lref - 15, 1/3) - 1.8
    else:
        d0 = 0.5
    
    # Calculate distances
    distances = np.sqrt(np.sum((true_filtered - pred_filtered) ** 2, axis=1))
    
    # Calculate TM-score
    tm_score = np.mean(1.0 / (1.0 + (distances / d0) ** 2))
    
    return tm_score



def train_model(X_train, y_train, X_valid, y_valid, params=None):
    """
    Train the improved RNA 3D model.
    """
    if params is None:
        params = {
            'geometric_sampling': True,
            'base_noise_level': 0.25,  # Reduced noise level for better control
            'correlation': 0.85
        }
    
    print(f"Training model with parameters: {params}")
    
    # Initialize and train the model
    model = ImprovedRNAModel(
        geometric_sampling=params['geometric_sampling'],
        base_noise_level=params['base_noise_level'],
        correlation=params['correlation'],
        msa_dir=MSA_DIR
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Validate the model
    print("Evaluating model on validation data...")
    y_pred = model.predict(X_valid)
    
    # Calculate TM-scores
    tm_scores = []
    for i in range(len(X_valid)):
        tm = calculate_tm_score_approx(y_pred[i], y_valid[i])
        tm_scores.append(tm)
    
    avg_tm_score = np.mean(tm_scores)
    print(f"Average TM-score on validation: {avg_tm_score:.4f}")
    
    return model, avg_tm_score, tm_scores


def run_multi_model_ensemble(X_train, y_train, X_valid, y_valid, test_seq_df, sample_submission_df, num_models=5):
    """
    Train multiple models with different parameters and create an ensemble.
    """
    print(f"Running ensemble with {num_models} models...")
    
    # Optimal parameters based on RNA 3D structure characteristics
    param_sets = [
        {'geometric_sampling': True, 'base_noise_level': 0.2, 'correlation': 0.9},  # More stable, higher correlation
        {'geometric_sampling': True, 'base_noise_level': 0.25, 'correlation': 0.85},
        {'geometric_sampling': True, 'base_noise_level': 0.3, 'correlation': 0.8},
        {'geometric_sampling': False, 'base_noise_level': 0.2, 'correlation': 0.9},  # Non-geometric variation
        {'geometric_sampling': True, 'base_noise_level': 0.15, 'correlation': 0.95},  # Very stable, very high correlation
    ]
    
    # Ensure we have enough parameter sets
    while len(param_sets) < num_models:
        param_sets.append({
            'geometric_sampling': np.random.choice([True, False], p=[0.8, 0.2]),  # Favor geometric sampling
            'base_noise_level': np.random.uniform(0.15, 0.3),  # Lower noise range
            'correlation': np.random.uniform(0.8, 0.95)  # Higher correlation range
        })
    
    # Train models
    models = []
    scores = []
    
    for i, params in enumerate(param_sets[:num_models]):
        print(f"\nTraining model {i+1}/{num_models}")
        model, score, _ = train_model(X_train, y_train, X_valid, y_valid, params)
        models.append(model)
        scores.append(score)
    
    # Prepare test features
    X_test = prepare_features(test_seq_df)
    
    # Generate predictions for each model
    print("\nGenerating predictions from all models...")
    model_predictions = []
    for i, model in enumerate(models):
        print(f"Model {i+1}/{len(models)} (score: {scores[i]:.4f})")
        pred = model.predict(X_test)
        model_predictions.append(pred)
    
    # Create submission from ensemble
    print("\nCreating submission from ensemble...")
    return create_ensemble_submission(model_predictions, test_seq_df, sample_submission_df, scores)



def create_ensemble_submission(model_predictions, test_seq_df, sample_submission_df, model_scores=None):
    """
    Create a submission file from ensemble predictions with improved weighting.
    """
    submission_df = sample_submission_df.copy()
    
    # Use model scores for weighted averaging if available
    if model_scores is not None:
        # Normalize scores to sum to 1
        weights = np.array(model_scores) / sum(model_scores)
        print(f"Using weighted ensemble with weights: {weights}")
    else:
        weights = np.ones(len(model_predictions)) / len(model_predictions)
    
    # Dictionary to store structures for each sequence
    seq_to_structures = {}
    
    # Process each test sequence
    for i, row in tqdm(test_seq_df.iterrows(), total=len(test_seq_df), desc="Processing sequences"):
        target_id = row['target_id']
        seq_length = len(row['sequence'])
        sequence = row['sequence']
        
        # Get base predictions from all models for this sequence
        sequence_preds = [pred[i][:seq_length] for pred in model_predictions]
        
        # Apply weights to generate a weighted average prediction
        weighted_avg = np.zeros_like(sequence_preds[0])
        for j, pred in enumerate(sequence_preds):
            weighted_avg += weights[j] * pred
        
        # Check if MSA file exists
        msa_file = None
        if os.path.exists(os.path.join(MSA_DIR, f"{target_id}.MSA.fasta")):
            msa_file = os.path.join(MSA_DIR, f"{target_id}.MSA.fasta")
        
        # Generate ensemble of 5 diverse structures using our improved method
        structures = generate_diverse_ensemble(
            weighted_avg,
            sequence,
            msa_file=msa_file
        )
        
        # Store the 5 structures for this sequence
        seq_to_structures[target_id] = structures
    
    # Fill submission dataframe with predictions
    print("Filling submission dataframe...")
    for i, row in tqdm(submission_df.iterrows(), total=len(submission_df), desc="Creating submission"):
        id_parts = row['ID'].split('_')
        seq_id = id_parts[0]
        residue_idx = int(id_parts[1]) - 1
        
        if seq_id in seq_to_structures and residue_idx < len(seq_to_structures[seq_id][0]):
            for struct_idx in range(5):
                submission_df.at[i, f'x_{struct_idx+1}'] = seq_to_structures[seq_id][struct_idx][residue_idx][0]
                submission_df.at[i, f'y_{struct_idx+1}'] = seq_to_structures[seq_id][struct_idx][residue_idx][1]
                submission_df.at[i, f'z_{struct_idx+1}'] = seq_to_structures[seq_id][struct_idx][residue_idx][2]
    
    # Save submission file
    submission_file = os.path.join(OUTPUT_DIR, 'submission.csv')
    submission_df.to_csv(submission_file, index=False)
    
    print(f"Submission saved to {submission_file}")
    print(f"File size: {os.path.getsize(submission_file) / (1024 * 1024):.2f} MB")
    
    return submission_df


# ===== MAIN EXECUTION =====
def main(use_external=True, max_external_structures=50, num_models=5, offline_mode=True):
    """
    Main execution function.
    """
    try:
        # Set offline mode if requested
        if offline_mode:
            os.environ['OFFLINE_MODE'] = 'True'
            print("Running in OFFLINE MODE - no internet access will be used")
        
        # Check dependencies
        if not check_and_install_dependencies():
            print("Please install required dependencies before running")
            return None
            
        print("Starting RNA 3D structure prediction pipeline...")
        start_time = time.time()
        
        # Load and process data including external structures
        X_train, y_train, X_valid, y_valid, data = load_all_data(
            use_external=use_external,
            max_external_structures=max_external_structures
        )
        
        # Run multi-model ensemble
        submission_df = run_multi_model_ensemble(
            X_train, y_train, X_valid, y_valid,
            data["test_sequences"], data["sample_submission"],
            num_models=num_models
        )
        
        print(f"Pipeline completed in {(time.time() - start_time) / 60:.2f} minutes")
        return submission_df
    
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Parse command line arguments
    main()








