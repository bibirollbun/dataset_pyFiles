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


test


sample_submission


print("SMILES of train:")
for i in range(9):
    smiles = train.iloc[i]['SMILES']
    print(f"{i+1:2d}. {smiles}")


print('Train columns: ', train.columns)
print('Duplicate values: ',train.duplicated().sum())
print(f'\nTrain data has {train.shape[0]} rows and {train.shape[1]} columns.')
print(f'Test data has {test.shape[0]} rows and {test.shape[1]} columns.')
print(f"The data has {train['Tg'].nunique()} distinct 'Tg'")
print(f"The data has {train['FFV'].nunique()} distinct 'FFV'")
print(f"The data has {train['Tc'].nunique()} distinct 'Tc'")
print(f"The data has {train['Density'].nunique()} distinct 'Density'")
print(f"The data has {train['Rg'].nunique()} distinct 'Rg'")


print('NaN values: \n')
null_train = train.isnull().sum()
for col, i in null_train.items():
    if i > 0:
        print(f"{col}:\t {i} ({i / len(train) * 100:.2f}%)")


msno.matrix(train)


no_id = train.drop(columns=['id'], errors='ignore')
no_id.hist(figsize=(11,7), bins=30, grid=False)
plt.tight_layout()
plt.show()


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


def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.Series([None]*6, index=[
            'MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds'
        ])
    return pd.Series([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol)
    ], index=[
        'MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds'
    ])

descriptor_df = train['SMILES'].apply(compute_descriptors)
train_descriptors = pd.concat([train, descriptor_df], axis=1)
target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
features = ['MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds']

plt.figure(figsize=(20, 18))
plot_number = 1
for target in target_columns:
    for feature in features:
        plt.subplot(len(target_columns), len(features), plot_number)
        sns.scatterplot(data=train_descriptors, x=feature, y=target)
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.tight_layout()
        plot_number += 1

plt.suptitle("SMILES content Features and Target Properties", fontsize=20, y=1.02)
plt.show()


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


sample_df = train_descriptors.dropna(subset=['MolWt', 'NumRotatableBonds']).copy()
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
print('Structure of Polymers')
display(img_crystalline_vs_amorphous)


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
display(img_pet)


sample_smiles = train["SMILES"].head(5).tolist()
molecules = []
for i in sample_smiles:
    mol = Chem.MolFromSmiles(i)
    if mol:
        molecules.append(mol)
Draw.MolsToGridImage(
    molecules,
    molsPerRow=5,
    subImgSize=(300, 200),
    legends=[f"Mol {i+1}" for i in range(len(molecules))])


print('Molecules with chemical property (Molecular Weight)')
def get_annotated_legend(smi, index):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    molwt = Descriptors.MolWt(mol)
    legend = f"Mol {index+1}\nMolWt: {molwt:.1f}"
    return mol, legend
annotated_mols = []
annotated_legends = []
for i, smi in enumerate(train["SMILES"].iloc[12:18]):
    mol, legend = get_annotated_legend(smi, i + 12)
    if mol:
        annotated_mols.append(mol)
        annotated_legends.append(legend)
img_annotated = Draw.MolsToGridImage(
    annotated_mols,
    molsPerRow=6,
    subImgSize=(300, 300),
    useSVG=False,
    legends=annotated_legends)
img_annotated


def get_fixed_extended_annotated_legend(smi, index):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    molwt = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    ring_count = Descriptors.RingCount(mol)
    heavy_atoms = Descriptors.HeavyAtomCount(mol)
    aliphatic_cycles = Descriptors.NumAliphaticCarbocycles(mol)
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    
    legend = (
        f"Mol {index+1}\nMolWt: {molwt:.1f}\nLogP: {logp:.2f}\nTPSA: {tpsa:.1f}"
        f"\nH-Donors: {h_donors}\nH-Acceptors: {h_acceptors}\nRotBonds: {rot_bonds}"
        f"\nRings: {ring_count}\nHeavy Atoms: {heavy_atoms}"
        f"\nAromatic Atoms: {aromatic_atoms}\nAliphatic Cycles: {aliphatic_cycles}"
    )
    return mol, legend

fixed_extended_mols = []
fixed_extended_legends = []

for i, smi in enumerate(train["SMILES"].iloc[24:30]):
    mol, legend = get_fixed_extended_annotated_legend(smi, i + 24)
    if mol:
        fixed_extended_mols.append(mol)
        fixed_extended_legends.append(legend)

img_fixed_extended_annotated = Draw.MolsToGridImage(
    fixed_extended_mols,
    molsPerRow=3,
    subImgSize=(350, 700),
    useSVG=False,
    legends=fixed_extended_legends
)

display(img_fixed_extended_annotated)


def is_fully_linear(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        if atom.GetDegree() > 2:
            return False
    return True

def is_branched(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        if atom.GetDegree() > 2:
            return True
    return False

def is_crosslinked(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        if atom.GetDegree() >= 4:
            return True
    return False

def is_networked_corrected(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    high_degree_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetDegree() >= 4)
    ring_count = mol.GetRingInfo().NumRings()
    branch_count = sum(1 for atom in mol.GetAtoms() if atom.GetDegree() > 2)
    return high_degree_atoms >= 2 and ring_count >= 2 and branch_count >= 4

def is_random_copolymer(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    unique_atoms = set(atom_symbols)
    diverse_atoms = len(unique_atoms) >= 4
    branch_points = sum(1 for atom in mol.GetAtoms() if atom.GetDegree() > 2)
    return diverse_atoms and branch_points >= 3

def is_alternating_copolymer(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 6:
        return False
    atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    symbols = [atom.GetSymbol() for atom in atoms]
    if len(symbols) < 6:
        return False
    pattern = [symbols[i] for i in range(4)]
    if pattern[0] != pattern[1] and pattern[0] == pattern[2] and pattern[1] == pattern[3]:
        return True
    return False

def is_block_copolymer(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 10:
        return False
    atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    symbols = [atom.GetSymbol() for atom in atoms]
    n = len(symbols)
    block1 = symbols[:n // 3]
    block2 = symbols[n // 3:2 * n // 3]
    block3 = symbols[2 * n // 3:]
    def dominant_element(block):
        return max(set(block), key=block.count)
    dom1 = dominant_element(block1)
    dom2 = dominant_element(block2)
    dom3 = dominant_element(block3)
    return (dom1 != dom2) or (dom2 != dom3)

def is_graft_copolymer(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    side_chains = sum(1 for atom in mol.GetAtoms() if atom.GetDegree() == 1 and not atom.IsInRing())
    branch_points = sum(1 for atom in mol.GetAtoms() if atom.GetDegree() > 2 and not atom.IsInRing())
    return side_chains >= 3 and branch_points >= 2

sample_structures = []
categories = {
    "Linear": [smi for smi in train['SMILES'].head(500) if is_fully_linear(smi)],
    "Branched": [smi for smi in train['SMILES'].head(500) if is_branched(smi)],
    "Crosslinked": [smi for smi in train['SMILES'].head(500) if is_crosslinked(smi)],
    "Networked": [smi for smi in train['SMILES'].head(500) if is_networked_corrected(smi)],
    "Random": [smi for smi in train['SMILES'].head(500) if is_random_copolymer(smi)],
    "Alternating": [smi for smi in train['SMILES'].head(500) if is_alternating_copolymer(smi)],
    "Block": [smi for smi in train['SMILES'].head(500) if is_block_copolymer(smi)],
    "Graft": [smi for smi in train['SMILES'].head(500) if is_graft_copolymer(smi)],
}

for label, smiles_list in categories.items():
    for smi in smiles_list[:3]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            sample_structures.append((mol, label))

mols, labels = zip(*sample_structures)
polymer_structures_img = Draw.MolsToGridImage(
    mols, molsPerRow=4, subImgSize=(300, 300), legends=labels
)
polymer_structures_img


def compute_properties_with_type(smiles, label):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'Type': label,
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'HDonors': Descriptors.NumHDonors(mol),
        'HAcceptors': Descriptors.NumHAcceptors(mol)
    }

polymer_data = []
for label, smiles_list in categories.items():
    for smi in smiles_list[:10]:
        props = compute_properties_with_type(smi, label)
        if props:
            polymer_data.append(props)

polymer_props_df = pd.DataFrame(polymer_data)
plt.figure(figsize=(12, 6))
sns.boxplot(data=polymer_props_df, x='Type', y='MolWt')
plt.title('Molecular Weight by Polymer Type')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=polymer_props_df, x='Type', y='LogP')
plt.title('LogP by Polymer Type')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=polymer_props_df, x='Type', y='TPSA')
plt.title('TPSA by Polymer Type')
plt.show()


def is_copolymer(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() < 2:
        return False
    unique_rings = {tuple(sorted(ring)) for ring in ring_info.AtomRings()}
    return len(unique_rings) >= 2
copolymer_smiles = [smi for smi in train['SMILES'].head(500) if is_copolymer(smi)]
copolymer_mols = [Chem.MolFromSmiles(smi) for smi in copolymer_smiles]
copolymer_img = Draw.MolsToGridImage(copolymer_mols[:10], molsPerRow=5, subImgSize=(300, 300),
                                     legends=[f"Copolymer {i+1}" for i in range(len(copolymer_mols[:10]))])
copolymer_img


def is_random_copolymer(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    unique_atoms = set(atom_symbols)      
    diverse_atoms = len(unique_atoms) >= 4
    branch_points = sum(1 for atom in mol.GetAtoms() if atom.GetDegree() > 2)    
    return diverse_atoms and branch_points >= 3

random_copolymer_smiles = [smi for smi in train['SMILES'].head(500) if is_random_copolymer(smi)]
random_copolymer_mols = [Chem.MolFromSmiles(smi) for smi in random_copolymer_smiles]
random_copolymer_img = Draw.MolsToGridImage(random_copolymer_mols[:10], molsPerRow=5, subImgSize=(300, 300),
                                            legends=[f"Random Co {i+1}" for i in range(len(random_copolymer_mols[:10]))])
random_copolymer_img


def is_alternating_copolymer(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 6:
        return False
    atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    symbols = [atom.GetSymbol() for atom in atoms]
    if len(symbols) < 6:
        return False
    
    pattern = [symbols[i] for i in range(4)]
    if pattern[0] != pattern[1] and pattern[0] == pattern[2] and pattern[1] == pattern[3]:
        return True
    return False

alternating_copolymer_smiles = [smi for smi in train['SMILES'].head(500) if is_alternating_copolymer(smi)]
alternating_copolymer_mols = [Chem.MolFromSmiles(smi) for smi in alternating_copolymer_smiles]
alternating_img = Draw.MolsToGridImage(alternating_copolymer_mols[:5], molsPerRow=5, subImgSize=(300, 300),
                                       legends=[f"Alternating Co {i+1}" for i in range(len(alternating_copolymer_mols[:5]))])
alternating_img


def is_block_copolymer(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 10:
        return False

    atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    symbols = [atom.GetSymbol() for atom in atoms]

    n = len(symbols)
    block1 = symbols[:n // 3]
    block2 = symbols[n // 3:2 * n // 3]
    block3 = symbols[2 * n // 3:]

    def dominant_element(block):
        return max(set(block), key=block.count)

    dom1 = dominant_element(block1)
    dom2 = dominant_element(block2)
    dom3 = dominant_element(block3)

    return (dom1 != dom2) or (dom2 != dom3)

block_copolymer_smiles = [smi for smi in train['SMILES'].head(500) if is_block_copolymer(smi)]
block_copolymer_mols = [Chem.MolFromSmiles(smi) for smi in block_copolymer_smiles]
block_img = Draw.MolsToGridImage(block_copolymer_mols[:10], molsPerRow=5, subImgSize=(300, 300),
                                 legends=[f"Block Co {i+1}" for i in range(len(block_copolymer_mols[:10]))])
block_img


def is_graft_copolymer(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    side_chains = sum(1 for atom in mol.GetAtoms() if atom.GetDegree() == 1 and not atom.IsInRing())

    branch_points = sum(1 for atom in mol.GetAtoms() if atom.GetDegree() > 2 and not atom.IsInRing())

    return side_chains >= 3 and branch_points >= 2

graft_copolymer_smiles = [smi for smi in train['SMILES'].head(500) if is_graft_copolymer(smi)]
graft_copolymer_mols = [Chem.MolFromSmiles(smi) for smi in graft_copolymer_smiles]
graft_img = Draw.MolsToGridImage(graft_copolymer_mols[:10], molsPerRow=5, subImgSize=(300, 300),
                                 legends=[f"Graft Co {i+1}" for i in range(len(graft_copolymer_mols[:10]))])
graft_img


sampled_smiles = train["SMILES"].head(100)
def get_smiles_features(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HDonors": Descriptors.NumHDonors(mol),
        "HAcceptors": Descriptors.NumHAcceptors(mol),
        "RotatableBonds": Descriptors.NumRotatableBonds(mol),
        "RingCount": Chem.rdMolDescriptors.CalcNumRings(mol)
    }

features = [get_smiles_features(smi) for smi in sampled_smiles]
features_df = pd.DataFrame([f for f in features if f is not None])
features_df.dropna(inplace=True)
sns.pairplot(features_df, diag_kind="hist")
plt.suptitle("Pairplot of Molecular Descriptors (First 100 SMILES)", y=1.02)
plt.show()




