pip install rdkit-pypi



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import numpy as np
import warnings
warnings.filterwarnings("ignore")

train=pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
test=pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
sub=pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv")



train.head()



train.info()


test.head()


targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

plt.figure(figsize=(15, 10))
for i, target in enumerate(targets, 1):
    plt.subplot(2, 3, i)
    sns.histplot(train[target], kde=True)
    plt.title(f'Distribution of {target}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
corr = train[targets].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Between Target Variables')
plt.show()



train['smiles_length'] = train['SMILES'].apply(len)
test['smiles_length'] = test['SMILES'].apply(len)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(train['smiles_length'], bins=30)
plt.title('Train SMILES Length Distribution')

plt.subplot(1, 2, 2)
sns.histplot(test['smiles_length'], bins=30)
plt.title('Test SMILES Length Distribution')
plt.show()

plt.figure(figsize=(15, 10))
for i, target in enumerate(targets, 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(x=train['smiles_length'], y=train[target])
    plt.title(f'{target} vs SMILES Length')
plt.tight_layout()
plt.show()


from rdkit import Chem
from collections import defaultdict

def get_common_fragments(smiles_list, n=5):
    fragment_counts = defaultdict(int)

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            # Get all atom-index based paths of length n
            paths = Chem.FindAllPathsOfLengthN(mol, n, useBonds=False)
            for path in paths:
                frag = Chem.MolFragmentToSmiles(mol, atomsToUse=path, canonical=True)
                fragment_counts[frag] += 1

    return sorted(fragment_counts.items(), key=lambda x: x[1], reverse=True)

sampled_smiles = train['SMILES'].sample(1000, random_state=42).tolist()
top_fragments = get_common_fragments(sampled_smiles, 5)[:10]

print("Top 10 common fragments:")
for frag, count in top_fragments:
    print(f"{frag}: {count} occurrences")




def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {
            'mol_weight': Descriptors.MolWt(mol),
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': mol.GetNumBonds(),
            'num_rings': Descriptors.RingCount(mol),
            'tpsa': Descriptors.TPSA(mol),
            'logp': Descriptors.MolLogP(mol)
        }
    return None

train_descriptors = train['SMILES'].apply(calculate_descriptors)
train_descriptors = pd.json_normalize(train_descriptors)

train = pd.concat([train, train_descriptors], axis=1)

descriptors = ['mol_weight', 'num_atoms', 'num_bonds', 'num_rings', 'tpsa', 'logp']

plt.figure(figsize=(20, 15))
for i, target in enumerate(targets, 1):
    for j, desc in enumerate(descriptors, 1):
        plt.subplot(len(targets), len(descriptors), (i-1)*len(descriptors) + j)
        sns.scatterplot(x=train[desc], y=train[target])
        plt.title(f'{target} vs {desc}')
plt.tight_layout()
plt.show()



missing = train.isnull().sum()
print("Missing values:\n", missing[missing > 0])

if missing.sum() > 0:
    sns.heatmap(train.isnull(), cbar=False)
    plt.title('Missing Value Patterns')
    plt.show()



test_descriptors = test['SMILES'].apply(calculate_descriptors)
test_descriptors = pd.json_normalize(test_descriptors)

plt.figure(figsize=(15, 10))
for i, desc in enumerate(descriptors, 1):
    plt.subplot(2, 3, i)
    sns.kdeplot(train[desc], label='Train')
    sns.kdeplot(test_descriptors[desc], label='Test')
    plt.title(f'{desc} Distribution')
    plt.legend()
plt.tight_layout()
plt.show()



def plot_extreme_molecules(property_name, n=3):
    top = train.nlargest(n, property_name)
    bottom = train.nsmallest(n, property_name)
    
    mols = []
    legends = []
    
    for _, row in top.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol:
            mols.append(mol)
            legends.append(f"High {property_name}: {row[property_name]:.2f}")
    
    for _, row in bottom.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol:
            mols.append(mol)
            legends.append(f"Low {property_name}: {row[property_name]:.2f}")
    
    img = Draw.MolsToGridImage(mols, legends=legends, molsPerRow=3)
    return img

plot_extreme_molecules('Tg')


plot_extreme_molecules('FFV')


plot_extreme_molecules('Tc')


plot_extreme_molecules('Rg')


plot_extreme_molecules('Density')


from collections import Counter

all_smiles = ''.join(train['SMILES'].tolist() + test['SMILES'].tolist())

char_counts = Counter(all_smiles)

plt.figure(figsize=(12, 6))
sns.barplot(x=list(char_counts.keys()), y=list(char_counts.values()))
plt.title("SMILES Character Frequency Distribution")
plt.xlabel("SMILES Characters")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()


from itertools import combinations

def get_n_grams(smiles, n=2):
    return [smiles[i:i+n] for i in range(len(smiles)-n+1)]

train_smiles = train['SMILES'].sample(1000).tolist()  
all_2grams = []
all_3grams = []
for smi in train_smiles:
    all_2grams.extend(get_n_grams(smi, 2))
    all_3grams.extend(get_n_grams(smi, 3))

# Get top 10 most frequent n-grams
top_2grams = Counter(all_2grams).most_common(10)
top_3grams = Counter(all_3grams).most_common(10)

print("Top 2-grams:", top_2grams)
print("Top 3-grams:", top_3grams)



train['mol_weight_bin'] = pd.cut(train['mol_weight'], bins=5)

plt.figure(figsize=(12, 6))
sns.boxplot(x='mol_weight_bin', y='Tg', data=train)
plt.title("Tg Distribution Across Molecular Weight Bins")
plt.xticks(rotation=45)
plt.show()


from scipy.stats import ks_2samp

for desc in descriptors:
    train_desc = train[desc].dropna()
    test_desc = test_descriptors[desc].dropna()
    ks_stat, p_value = ks_2samp(train_desc, test_desc)
    print(f"{desc}: KS stat = {ks_stat:.3f}, p-value = {p_value:.3f}")

