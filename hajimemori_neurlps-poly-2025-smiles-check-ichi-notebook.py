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


from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
import pandas as pd
import numpy as np

# --- Main execution ---
if __name__ == "__main__":
    # Load the training data
    df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
    smiles_list = df['SMILES']

    canonical_smi_list = []

    print("Starting SMILES validation and canonicalization...")
    for i, smi in enumerate(smiles_list):
        try:
            # Convert to molecule and back to SMILES to get the canonical version
            checked_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
            canonical_smi_list.append(checked_smiles)
        
        except:
            # --- If it fails ---
            # Print error information and append NA to the list
            print(f"Failed to process SMILES at index {i}.")
            print(f"  - SMILES: {smi}")
            
            if Chem.MolFromSmiles(smi) is None:
                print("  - Reason: Invalid SMILES string.")
            else:
                print("  - Reason: Other error (e.g., molecular structure issue).")

            # Append np.nan to maintain list length
            canonical_smi_list.append(np.nan)
            print("-" * 20) # Separator for clarity

    print("\n--- Processing Complete ---")
    
    # Confirm that the list length matches the DataFrame length
    if len(df) == len(canonical_smi_list):
        print(f"Original data count: {len(df)}, Generated list count: {len(canonical_smi_list)}")
        print("List length matches. Adding column to DataFrame and saving.")
        
        # Add the new list as a column to the DataFrame
        df['checked_canonical_smi'] = canonical_smi_list
        
        # Save to a new CSV file (index=False prevents saving the DataFrame index as a column)
        output_filename = 'train_checked.csv'
        df.to_csv(output_filename, index=False)
        print(f"Data saved to '{output_filename}'.")




