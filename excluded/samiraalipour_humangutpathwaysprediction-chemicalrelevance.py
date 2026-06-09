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


# Install RDKit 
!pip install rdkit-pypi


import pandas as pd
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
import numpy as np
from tqdm import tqdm

# Enable tqdm integration with pandas
# Explicitly set pandas integration for usability.
tqdm.pandas()


# Load data
train = pd.read_csv('/kaggle/input/prediction-human-gut-biotransformation-pathways/archive/train.csv')
test = pd.read_csv('/kaggle/input/prediction-human-gut-biotransformation-pathways/test.csv')

# Preprocess training data
train[['Reactant', 'Product']] = train['SMILE'].str.split('>>', expand=True)
def canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None

train['Reactant'] = train['Reactant'].apply(canonical_smiles)
train['Product'] = train['Product'].apply(canonical_smiles)
train = train.dropna().drop_duplicates(subset=['Reactant', 'Product'])

# Canonicalize test compounds
# Ensure test SMILES match the canonical form in fp_dict to avoid KeyError.
test['Compound'] = test['Compound'].apply(canonical_smiles)
test = test.dropna(subset=['Compound'])


G = nx.DiGraph()
for _, row in train.iterrows():
    G.add_edge(row['Reactant'], row['Product'])



nodes = list(G.nodes()) + list(test['Compound'].unique())
fp_dict = {}
for n in nodes:
    mol = Chem.MolFromSmiles(n)
    if mol:
        fp_dict[n] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def predict_pathways(start_compound, steps, n_paths=5):
    start_mol = Chem.MolFromSmiles(start_compound)
    if not start_mol:
        return [''] * n_paths
    start_smiles = canonical_smiles(start_compound)
    if not start_smiles:
        return [''] * n_paths
    # Compute fingerprint for start compound if not in fp_dict
    start_fp = fp_dict.get(start_smiles)
    if not start_fp:
        start_fp = AllChem.GetMorganFingerprintAsBitVect(start_mol, 2, nBits=2048)
        fp_dict[start_smiles] = start_fp
    
    pathways = []
    seen_paths = set()
    
    for _ in range(n_paths * 3):  # Attempt more paths for diversity
        path = [start_smiles]
        current = start_smiles
        for _ in range(steps):
            current_fp = fp_dict.get(current)
            if not current_fp:
                mol = Chem.MolFromSmiles(current)
                if mol:
                    current_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fp_dict[current] = current_fp
                else:
                    break
            
            if current in G:
                successors = [s for s in G.successors(current) if s not in path]
                if successors:
                    # Randomly select from top successors by Tanimoto similarity
                    scores = {s: TanimotoSimilarity(fp_dict[s], current_fp) for s in successors}
                    top_successors = sorted(scores, key=scores.get, reverse=True)[:3]
                    next_compound = np.random.choice(top_successors) if top_successors else None
                else:
                    # Fallback to similarity
                    similarities = {n: TanimotoSimilarity(fp_dict[n], current_fp) for n in nodes if n != current and n not in path and n in fp_dict}
                    top_similar = sorted(similarities, key=similarities.get, reverse=True)[:3]
                    next_compound = np.random.choice(top_similar) if top_similar else None
            else:
                # Fallback to similarity for compounds not in graph
                similarities = {n: TanimotoSimilarity(fp_dict[n], current_fp) for n in nodes if n != current and n not in path and n in fp_dict}
                top_similar = sorted(similarities, key=similarities.get, reverse=True)[:3]
                next_compound = np.random.choice(top_similar) if top_similar else None
            
            if not next_compound:
                break
            path.append(next_compound)
            current = next_compound
        
        path_str = '=>'.join(path[1:])  # Exclude start compound
        if path_str and path_str not in seen_paths and len(path) - 1 == steps:
            pathways.append(path_str)
            seen_paths.add(path_str)
        if len(pathways) >= n_paths:
            break
    
    # Fallback: Generate random pathways if needed
    while len(pathways) < n_paths:
        path = [start_smiles]
        current = start_smiles
        current_fp = fp_dict[start_smiles]
        for _ in range(steps):
            similarities = {n: TanimotoSimilarity(fp_dict[n], current_fp) for n in nodes if n != current and n not in path and n in fp_dict}
            top_similar = sorted(similarities, key=similarities.get, reverse=True)[:5]
            next_compound = np.random.choice(top_similar) if top_similar else None
            if not next_compound:
                break
            path.append(next_compound)
            current = next_compound
            current_fp = fp_dict.get(current)
            if not current_fp:
                mol = Chem.MolFromSmiles(current)
                if mol:
                    current_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fp_dict[current] = current_fp
                else:
                    break
        path_str = '=>'.join(path[1:])
        if path_str and path_str not in seen_paths and len(path) - 1 == steps:
            pathways.append(path_str)
            seen_paths.add(path_str)
    
    # Fill with empty strings if needed
    while len(pathways) < n_paths:
        pathways.append('')
    return pathways[:n_paths]



def inference(row):
    start_compound = row['Compound']
    steps = row['Steps']
    pathways = predict_pathways(start_compound, steps)
    return pd.Series([row['PWY_ID']] + pathways, index=['PWY_ID', 'Pathway_rank1', 'Pathway_rank2', 'Pathway_rank3', 'Pathway_rank4', 'Pathway_rank5'])

# Generate submission
submission = test.progress_apply(inference, axis=1)
submission.to_csv('submission_chemical.csv', index=False)
print("Submission file 'submission_chemical.csv' created.")

submission.head()

