import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


!pip install rdkit-pypi


import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import missingno as msno
from scipy.stats import skew, kurtosis
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, Draw


train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
sample_submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')


train.head(3)


from rdkit.Chem import AllChem, Draw, rdMolTransforms

smi = 'CCCC'
mol = Chem.MolFromSmiles(smi)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)
AllChem.UFFOptimizeMolecule(mol)
mol_60 = Chem.Mol(mol)
conf_60 = mol_60.GetConformer()
rdMolTransforms.SetDihedralDeg(conf_60, 1, 2, 3, 4, 60)

mol_180 = Chem.Mol(mol)
conf_180 = mol_180.GetConformer()
rdMolTransforms.SetDihedralDeg(conf_180, 1, 2, 3, 4, 180)

mol_60 = Chem.RemoveHs(mol_60)
mol_180 = Chem.RemoveHs(mol_180)

img_rotation_fixed = Draw.MolsToGridImage(
    [mol_60, mol_180],
    molsPerRow=2,
    subImgSize=(400, 300),
    legends=["Rotation: 60째", "Rotation: 180째"]
)
print('Molecular orientation can be changed by rotation around the bonds')
display(img_rotation_fixed)


def compute_molwt(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.MolWt(mol)

train['MolWt'] = train['SMILES'].apply(compute_molwt)
target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
plt.figure(figsize=(18, 10))

for i, target in enumerate(target_columns, 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(data=train, x='MolWt', y=target)
    plt.title(f'Molecular Weight vs {target}')
    plt.xlabel('Molecular Weight (MolWt)')
    plt.ylabel(target)

plt.tight_layout()
plt.suptitle("Molecular Weight vs Target Properties", fontsize=20, y=1.02)
plt.show()


from rdkit.Chem import Descriptors, Draw
from collections import Counter

def extract_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [None, None]
        return [
            Descriptors.MolWt(mol),
            Descriptors.NumRotatableBonds(mol)
        ]
    except:
        return [None, None]

train[['MolWt', 'NumRotatableBonds']] = train['SMILES'].apply(extract_descriptors).apply(pd.Series)

sample_df = train.dropna(subset=['MolWt', 'NumRotatableBonds']).copy()
example_rigid = sample_df[sample_df['NumRotatableBonds'] <= 2].iloc[0]
example_flexible = sample_df[sample_df['NumRotatableBonds'] >= 15].iloc[0]
mol_rigid = Chem.MolFromSmiles(example_rigid['SMILES'])
mol_flexible = Chem.MolFromSmiles(example_flexible['SMILES'])

img_crystalline_vs_amorphous = Draw.MolsToGridImage(
    [mol_rigid, mol_flexible],
    molsPerRow=2,
    subImgSize=(400, 300),
    legends=[
        f"Rigid Structure\nMolWt: {example_rigid['MolWt']:.1f}, RotBonds: {example_rigid['NumRotatableBonds']}",
        f"Flexible Structure\nMolWt: {example_flexible['MolWt']:.1f}, RotBonds: {example_flexible['NumRotatableBonds']}"
    ]
)

img_crystalline_vs_amorphous


polyethylene_smiles = "CCCCCCC" 
polyethylene_mol = Chem.MolFromSmiles(polyethylene_smiles)
img_polyethylene = Draw.MolToImage(polyethylene_mol, size=(400, 300), legend="Polyethylene Fragment")
display(img_polyethylene)

polypropylene_smiles = "CC(C)CC(C)CC(C)C"
polypropylene_mol = Chem.MolFromSmiles(polypropylene_smiles)
img_polypropylene = Draw.MolToImage(polypropylene_mol, size=(400, 300), legend="Polypropylene Fragment")
display(img_polypropylene)

pet_smiles = "O=C(C1=CC=CC=C1C(=O)OCC)OCC"  
pet_mol = Chem.MolFromSmiles(pet_smiles)
img_pet = Draw.MolToImage(pet_mol, size=(450, 300), legend="Polyethylene Terephthalate (PET) Fragment")


numeric_df = train.drop(columns=['id', 'SMILES'])

plt.figure()
numeric_df.hist(bins=30, edgecolor='black', grid=False)
plt.tight_layout()
plt.suptitle("Histograms of Numerical Columns", fontsize=16, y=1.02)
plt.show()


all_smiles_string = ''.join(train['SMILES'].dropna())
char_counts = Counter(all_smiles_string)
unique_char_freq_df = pd.DataFrame(char_counts.items(), columns=['Character', 'Frequency'])
unique_char_freq_df = unique_char_freq_df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
unique_char_freq_df


smiles_series = train['SMILES']

smiles_lengths = smiles_series.apply(len)

all_chars = ''.join(smiles_series)
char_counts = Counter(all_chars)
char_freq_df = pd.DataFrame(char_counts.items(), columns=['Character', 'Frequency']).sort_values(by='Frequency', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=char_freq_df.head(20), x='Character', y='Frequency')
plt.title("Top 20 Most Frequent Characters in SMILES")
plt.xlabel("Character")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


atom_types = ['C', 'c', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I']
atom_counts = {atom: 0 for atom in atom_types}
for smi in train['SMILES']:
    for atom in atom_types:
        atom_counts[atom] += smi.count(atom)
atom_counts_df = pd.DataFrame(list(atom_counts.items()), columns=['Atom', 'Count']).sort_values(by='Count', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(data=atom_counts_df, x='Atom', y='Count')
plt.title("Frequency of Specific Atom Types in SMILES")
plt.xlabel("Atom Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
sns.histplot(smiles_lengths, bins=50, kde=True)
plt.title("Distribution of SMILES String Lengths")
plt.xlabel("Length of SMILES")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


from rdkit.Chem import rdmolops
import networkx as nx

sample_smiles = train['SMILES'].dropna().sample(5, random_state=42)
molecular_graphs = []

for smi in sample_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        G = rdmolops.GetAdjacencyMatrix(mol)
        graph = nx.from_numpy_array(G)
        molecular_graphs.append((smi, graph))

plt.figure(figsize=(15, 10))
for i, (smi, graph) in enumerate(molecular_graphs, 1):
    plt.subplot(2, 3, i)
    nx.draw_networkx(graph, node_size=300, with_labels=False)
    plt.title(f"SMILES: {smi[:20]}...", fontsize=8)
    plt.axis('off')

plt.suptitle("Sample Molecular Graphs from SMILES", fontsize=16)
plt.tight_layout()
plt.show()


atom_types = ['C', 'c', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I']

for atom in atom_types:
    train[f'has_{atom}'] = train['SMILES'].apply(lambda x: atom in x)

train['C_and_O'] = (train['has_C'] & train['has_O']).map({True: 'With C & O', False: 'Without C & O'})
from scipy.stats import ttest_ind
t_test_results = []

for target in target_columns:
    group_with = train[train['C_and_O'] == 'With C & O'][target].dropna()
    group_without = train[train['C_and_O'] == 'Without C & O'][target].dropna()
    
    if len(group_with) > 1 and len(group_without) > 1:
        t_stat, p_value = ttest_ind(group_with, group_without, equal_var=False)
    else:
        t_stat, p_value = None, None
    
    t_test_results.append({
        'Target': target,
        'T-statistic': t_stat,
        'P-value': p_value,
        'Significant (p < 0.05)': p_value < 0.05 if p_value is not None else None
    })

t_test_df = pd.DataFrame(t_test_results)
t_test_df


def count_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumAtoms() if mol else None

train['NumAtoms'] = train['SMILES'].apply(count_atoms)

def count_rings(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetRingInfo().NumRings() if mol else None

train['NumRings'] = train['SMILES'].apply(count_rings)

train['SMILES_Length'] = train['SMILES'].apply(len)
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
sns.histplot(train['NumAtoms'].dropna(), bins=40, kde=True)
plt.title("Distribution of Atom Counts")
plt.xlabel("Number of Atoms")

plt.subplot(1, 3, 2)
sns.histplot(train['NumRings'].dropna(), bins=30, kde=True)
plt.title("Distribution of Ring Counts")
plt.xlabel("Number of Rings")

plt.subplot(1, 3, 3)
sns.histplot(train['SMILES_Length'], bins=50, kde=True)
plt.title("Distribution of SMILES Lengths")
plt.xlabel("Length of SMILES")

plt.tight_layout()
plt.show()


smiles_features = ['NumAtoms', 'NumRings', 'SMILES_Length']
correlation_matrix = train[smiles_features + target_columns].corr()
correlation_subset = correlation_matrix.loc[smiles_features, target_columns]

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_subset, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation: SMILES-Based Features vs Target Properties", fontsize=14)
plt.tight_layout()
plt.show()


plt.figure(figsize=(20, 15))
plot_number = 1

for target in target_columns:
    for feature in smiles_features:
        plt.subplot(len(target_columns), len(smiles_features), plot_number)
        sns.scatterplot(data=train, x=feature, y=target, alpha=0.6)
        plt.xlabel(feature)
        plt.ylabel(target)
        plot_number += 1

plt.tight_layout()
plt.suptitle("Pairwise Scatter Plots: SMILES-Based Features vs Target Properties", fontsize=20, y=1.02)
plt.show()




