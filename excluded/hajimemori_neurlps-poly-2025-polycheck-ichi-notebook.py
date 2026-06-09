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





!pip install --no-index --no-deps /kaggle/input/rdkit2025new/wheelhouse/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

def create_polymer(monomer_smiles, n_units):
    """
    Generates a head-to-tail polymer from a given monomer SMILES and number of units.
    Returns None on error, allowing the caller to handle error messages.
    """
    if n_units < 1:
        return None

    monomer_mol = Chem.MolFromSmiles(monomer_smiles)
    if monomer_mol is None:
        return None

    attachment_points = []
    for atom in monomer_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:  # Dummy atom '*'
            if len(atom.GetNeighbors()) == 1:
                neighbor = atom.GetNeighbors()[0]
                attachment_points.append({'dummy_idx': atom.GetIdx(), 'neighbor_idx': neighbor.GetIdx()})
    
    if len(attachment_points) < 2:
        return None
        
    # Sort attachment points to consistently define head and tail
    attachment_points.sort(key=lambda p: p['dummy_idx'])
    head_point = attachment_points[0]
    tail_point = attachment_points[1]

    if n_units == 1:
        return monomer_mol

    polymer_chain = Chem.Mol(monomer_mol)

    for _ in range(n_units - 1):
        next_monomer = Chem.Mol(monomer_mol)
        
        current_tail_dummy_idx = -1
        current_tail_neighbor = None
        # Find the tail dummy atom in the current polymer chain
        for atom in polymer_chain.GetAtoms():
            if atom.GetAtomicNum() == 0:
                if atom.GetIdx() > current_tail_dummy_idx:
                    current_tail_dummy_idx = atom.GetIdx()
        
        if current_tail_dummy_idx != -1:
            current_tail_neighbor = polymer_chain.GetAtomWithIdx(current_tail_dummy_idx).GetNeighbors()[0]
        else:
            return None # Should not happen in a valid chain
        
        combo = Chem.RWMol(polymer_chain)
        chain_num_atoms = polymer_chain.GetNumAtoms()
        
        # Add atoms and bonds from the next monomer to the growing chain
        for atom in next_monomer.GetAtoms():
            combo.AddAtom(atom)
        for bond in next_monomer.GetBonds():
            combo.AddBond(
                bond.GetBeginAtomIdx() + chain_num_atoms,
                bond.GetEndAtomIdx() + chain_num_atoms,
                bond.GetBondType()
            )
            
        # Form a new bond between the tail of the chain and the head of the new monomer
        monomer_head_neighbor_idx = head_point['neighbor_idx'] + chain_num_atoms
        combo.AddBond(current_tail_neighbor.GetIdx(), monomer_head_neighbor_idx, Chem.BondType.SINGLE)
        
        # Remove the dummy atoms that have been used to form the new bond
        monomer_head_dummy_idx = head_point['dummy_idx'] + chain_num_atoms
        indices_to_remove = sorted([current_tail_dummy_idx, monomer_head_dummy_idx], reverse=True)
        for idx in indices_to_remove:
            combo.RemoveAtom(idx)
            
        polymer_chain = combo.GetMol()
        if polymer_chain is None:
            return None

    try:
        Chem.SanitizeMol(polymer_chain)
        return polymer_chain
    except Exception:
        return None

# --- Main execution block ---
if __name__ == "__main__":

    # Load monomer structures
    try:
        df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
    except FileNotFoundError:
        print("Error: 'train.csv' not found.")
        print("Creating dummy data for testing purposes and continuing.")
        # Dummy data for testing
        data = {'SMILES': [
            '*CC(*)=O',          # Normal case
            'C1=CC=CC=C1',       # Error case: no dummy atoms
            '*C1=CC=CS1',        # Error case: only one dummy atom
            'InvalidSMILES',     # Error case: invalid SMILES
            '*c1c(C#N)c(F)c(F)c(C#N)c1*', # Normal case
        ]}
        df = pd.DataFrame(data)

    monomer_smi_list = df['SMILES']
    polymer_smi_list = []
    
    desired_units = 2

    print(f"--- Starting generation of {desired_units}-mers ---")
    
    # Use enumerate to get both the index (i) and the SMILES string (monomer_smi)
    for i, monomer_smi in enumerate(monomer_smi_list):
        
        # Call the polymer generation function
        polymer_mol = create_polymer(monomer_smi, desired_units)

        if polymer_mol:
            # [Success] Add the polymer SMILES to the list
            polymer_smiles = Chem.MolToSmiles(polymer_mol)
            polymer_smi_list.append(polymer_smiles)
        else:
            # [Failure] Print error information and add NA to the list
            print(f"Failed to generate polymer for compound at index {i}.")
            print(f"  - SMILES: {monomer_smi}")
            
            # Check the number of dummy atoms to provide a more specific reason
            num_dummy_atoms = monomer_smi.count('*') if isinstance(monomer_smi, str) else 0
            if Chem.MolFromSmiles(monomer_smi) is None:
                 print("  - Reason: Invalid SMILES.")
            elif num_dummy_atoms < 2:
                print(f"  - Reason: Did not find the 2 dummy atoms (*) required for polymerization. (Found: {num_dummy_atoms})")
            else:
                print("  - Reason: Other error (e.g., molecular structure issue).")

            # Add NA (Not a Number) to maintain the list length
            polymer_smi_list.append(np.nan)
            print("-" * 20) # Separator for better readability

    print("\n--- Processing Complete ---")
    
    # Verify that the length of the list matches the number of rows in the DataFrame
    if len(df) == len(polymer_smi_list):
        print(f"Original data count: {len(df)}, Generated list count: {len(polymer_smi_list)}")
        print("List length matches. Adding column to DataFrame and saving.")
        
        # Add the list as a new column to the DataFrame
        df['polymer_smi'] = polymer_smi_list
        
        # Save to CSV file (index=False prevents saving the DataFrame index)
        output_filename = f'train_add_poly{desired_units}.csv'
        df.to_csv(output_filename, index=False)
        print(f"Data saved to '{output_filename}'.")
    else:
        # This error should not occur, but it's kept as a safeguard
        print(f"Fatal Error: Original data count ({len(df)}) does not match generated list count ({len(polymer_smi_list)}).")


from rdkit.Chem import Draw
from IPython.display import Image # Use Image for better display in notebooks

# --- Code to Visualize Problematic Structures ---

# Define the indices of the monomers that caused errors during polymerization
error_indices = [576, 4783, 4836, 7129]

# Check if all indices are valid before proceeding
valid_indices = [idx for idx in error_indices if idx in df.index]
if len(valid_indices) != len(error_indices):
    print(f"Warning: Some indices are out of bounds for the loaded DataFrame.")
    # Proceeding with only the valid indices
    error_indices = valid_indices

if error_indices:
    # Extract the SMILES strings for the specified indices
    smiles_to_visualize = df.loc[error_indices, 'SMILES'].tolist()

    # Convert SMILES strings to RDKit molecule objects
    # Handle potential None results from invalid SMILES
    mol_list = [Chem.MolFromSmiles(smi) for smi in smiles_to_visualize]

    # Create legends for each molecule, including its index and SMILES
    legends = [f"Index: {idx}\n{smi}" for idx, smi in zip(error_indices, smiles_to_visualize)]

    # Generate a grid image of the molecules
    # molsPerRow specifies the number of molecules per row
    # subImgSize sets the size of each individual molecule's image
    grid_image = Draw.MolsToGridImage(
        mols=mol_list,
        molsPerRow=2,
        subImgSize=(350, 350),
        legends=legends
    )

    # Display the generated image in the notebook
    print("--- Displaying Structures of Monomers that Failed Polymerization ---")
    display(grid_image)
else:
    print("No valid indices were provided to visualize.")

