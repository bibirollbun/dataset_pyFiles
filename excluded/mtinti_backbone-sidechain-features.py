!pip install rdkit


"""
Simplified Polymer Backbone Feature Example
Shows how to calculate a single backbone/sidechain feature
"""

import pandas as pd
from rdkit import Chem
import networkx as nx
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import multiprocessing


def process_polymer_smiles(smiles):
    """Process polymer SMILES to remove [*] markers and find connection points."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, []
        
        # Find atoms connected to stars
        star_neighbors = []
        editable_mol = Chem.RWMol(mol)
        atoms_to_remove = []
        
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:  # Star atom
                for neighbor in atom.GetNeighbors():
                    star_neighbors.append(neighbor.GetIdx())
                atoms_to_remove.append(atom.GetIdx())
        
        # Remove star atoms
        for idx in sorted(atoms_to_remove, reverse=True):
            editable_mol.RemoveAtom(idx)
        
        # Adjust indices
        adjusted_neighbors = []
        for orig_idx in star_neighbors:
            adjustment = sum(1 for removed_idx in atoms_to_remove if removed_idx < orig_idx)
            adjusted_neighbors.append(orig_idx - adjustment)
        
        return editable_mol.GetMol(), list(set(adjusted_neighbors))
    except:
        return None, []


def identify_backbone_atoms(mol, star_indices):
    """Find backbone atoms as shortest path between connection points."""
    if len(star_indices) < 2:
        return set(range(mol.GetNumAtoms()))
    
    try:
        G = nx.from_numpy_array(Chem.GetAdjacencyMatrix(mol))
        path = nx.shortest_path(G, star_indices[0], star_indices[-1])
        return set(path)
    except:
        return set(range(mol.GetNumAtoms()))


def calculate_backbone_aromatic_fraction(smiles):
    """
    Calculate the fraction of aromatic atoms in the backbone.
    This single feature often correlates strongly with Tg and rigidity.
    
    Returns:
        Dictionary with SMILES and backbone_aromatic_fraction
    """
    features = {'SMILES': smiles, 'backbone_aromatic_fraction': 0.0}
    
    try:
        # Process molecule
        mol, star_indices = process_polymer_smiles(smiles)
        if mol is None:
            return features
        
        # Find backbone atoms
        backbone_atoms = identify_backbone_atoms(mol, star_indices)
        
        # Count aromatic atoms in backbone
        aromatic_count = 0
        backbone_heavy_count = 0
        
        for idx in backbone_atoms:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() > 1:  # Heavy atom
                backbone_heavy_count += 1
                if atom.GetIsAromatic():
                    aromatic_count += 1
        
        # Calculate fraction
        if backbone_heavy_count > 0:
            features['backbone_aromatic_fraction'] = aromatic_count / backbone_heavy_count
            
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
    
    return features


def analyze_polymer_backbones(smiles_list, n_jobs=-1):
    """
    Analyze backbone aromatic fraction for a list of polymers.
    
    Args:
        smiles_list: List of SMILES strings with [*] markers
        n_jobs: Number of parallel jobs (-1 uses all CPUs)
        
    Returns:
        DataFrame with SMILES and backbone_aromatic_fraction
    """
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    print(f"Calculating backbone aromatic fraction for {len(smiles_list)} polymers...")
    
    # Parallel processing
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(calculate_backbone_aromatic_fraction)(smiles) 
        for smiles in tqdm(smiles_list, desc="Processing")
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    df.set_index('SMILES', inplace=True)
    
    print(f"\nComplete! Average backbone aromatic fraction: {df['backbone_aromatic_fraction'].mean():.3f}")
    
    return df


# Example usage
if __name__ == '__main__':
    # Example polymers
    test_polymers = [
        "[*]CC[*]",                      # Polyethylene - no aromatics
        "[*]CC([*])c1ccccc1",           # Polystyrene - aromatic sidechain only
        "[*]c1ccc(cc1)c2ccc(cc2)[*]",   # Rigid aromatic backbone
    ]
    
    # Calculate the feature
    results = analyze_polymer_backbones(test_polymers, n_jobs=1)
    print("\nResults:")
    print(results)




