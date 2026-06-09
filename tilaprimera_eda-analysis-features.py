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


import pandas as pd
# /kaggle/input/stanford-rna-3d-folding/sample_submission.csv
# /kaggle/input/stanford-rna-3d-folding/validation_sequences.csv
# /kaggle/input/stanford-rna-3d-folding/test_sequences.csv
# /kaggle/input/stanford-rna-3d-folding/validation_labels.csv
# /kaggle/input/stanford-rna-3d-folding/train_labels.csv
# /kaggle/input/stanford-rna-3d-folding/train_sequences.csv
train_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
# print(train_sequences.head(10))

train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
# print(train_labels.head(10))


# Kaggle Cell 1 - Setup
!apt-get install -y vienna-rna

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess



#Predict Structure (RNAfold CLI)
def rnafold(sequence):
    process = subprocess.run(['RNAfold'], input=sequence.encode(), capture_output=True)
    output = process.stdout.decode()
    structure = output.strip().split('\n')[1].split(' ')[0]
    mfe = float(output.strip().split('\n')[1].split('(')[-1].strip(')'))
    return structure, mfe


# Simulate SHAPE Reactivity
def generate_mock_shape_data(sequence, seed=42):
    np.random.seed(seed)
    return np.random.uniform(0, 1, len(sequence))


# Plot SHAPE Data
def plot_shape(shape_data):
    plt.figure(figsize=(10,3))
    plt.bar(range(1, len(shape_data)+1), shape_data)
    plt.xlabel("Nucleotide Position")
    plt.ylabel("Simulated SHAPE Reactivity")
    plt.title("Mock SHAPE Reactivity Profile")
    plt.show()


# Full Example
sequence = "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAA"

# Step 1: Fold without SHAPE
structure, mfe = rnafold(sequence)

# Step 2: Generate SHAPE
shape_data = generate_mock_shape_data(sequence)

# Step 3: Display
print(f"Sequence: {sequence}")
print(f"Predicted Structure: {structure}")
print(f"MFE: {mfe} kcal/mol")

# Step 4: Visualize SHAPE
plot_shape(shape_data)


len(train_sequences.index)


import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# 1. Sequence Length Distribution
train_sequences['sequence_length'] = train_sequences['sequence'].apply(len)

plt.figure(figsize=(8, 5))
sns.histplot(train_sequences['sequence_length'], bins=30, kde=True)
plt.title("Sequence Length Distribution")
plt.xlabel("Sequence Length")
plt.ylabel("Frequency")
plt.show()

# 2. Base Composition
bases = ['A', 'C', 'G', 'U']
base_counts = {base: train_sequences['sequence'].str.count(base).sum() for base in bases}

plt.figure(figsize=(6, 4))
sns.barplot(x=list(base_counts.keys()), y=list(base_counts.values()))
plt.title("Base Composition Across All Sequences")
plt.xlabel("Base")
plt.ylabel("Count")
plt.show()

# 3. Residue Positioning (resid vs target_id)
plt.figure(figsize=(10, 6))
sns.histplot(train_labels['resid'], bins=50, kde=True)
plt.title("Residue Index Distribution")
plt.xlabel("Residue Index (resid)")
plt.ylabel("Frequency")
plt.show()

# 4. 3D Scatter Plot of Coordinates
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train_labels['x_1'], train_labels['y_1'], train_labels['z_1'], s=1, alpha=0.5)
ax.set_title("3D Scatter of Nucleotide Positions")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()



!pip install biopython
!pip install torch_geometric
# etc...



# RNA Structure Predictor Notebook (BioEmu-inspired)

# ================
# Step 1 - Load CSVs 
from Bio import SeqIO

# ================
# Step 2 - Visualize a sample

def plot_sample_structure(target_id):
    df = train_labels[train_labels['ID'].str.startswith(target_id)]
    print(f"has len of target_id df {len(df.index)}")
    plt.scatter(df['x_1'], df['y_1'], s=1)
    plt.title(f"{target_id} 2D projection")
    plt.xlabel("x_1")
    plt.ylabel("y_1")
    plt.axis('equal')
    plt.show()

first_train_seq = train_sequences['sequence'].iloc[0]
print(f" length of the sequence is: {len(first_train_seq)}")
print(f" the 2D plot of the sequence: {train_sequences['sequence'].iloc[0]}")
plot_sample_structure(train_sequences['target_id'].iloc[0])



def plot_sample_structure(target_id, sequence):
    df = train_labels[train_labels['ID'].str.startswith(target_id)]
    print(f"Target {target_id} has {len(df.index)} coordinates")
    
    # 3D Scatter Plot
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['x_1'], df['y_1'], df['z_1'], s=10)
    # Annotate each point with its nucleotide
    for i, base in enumerate(sequence):
        if i < len(df):
            ax.text(df.loc[i, 'x_1'], df.loc[i, 'y_1'], df.loc[i, 'z_1'], base, fontsize=8)
    ax.set_title(f"{target_id} - 3D Structure")
    ax.set_xlabel('x_1')
    ax.set_ylabel('y_1')
    ax.set_zlabel('z_1')
    plt.show()
    
    # 2D Projections
    
    # Projection 1: X vs Y
    # figxy = plt.figure(figsize=(5,5))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 3))
    ax1.plot(df['x_1'], df['y_1'])
    for i, base in enumerate(sequence):
        if i < len(df):
            ax1.text(df.loc[i, 'x_1'], df.loc[i, 'y_1'], base, fontsize=8)
    ax1.set_title(f"(X vs Y)")
    ax1.set_xlabel('x_1')
    ax1.set_ylabel('y_1')
    # plt.show()

    # Projection 2: X vs Z
    ax2.plot(df['x_1'], df['z_1'])
    for i, base in enumerate(sequence):
        if i < len(df):
            ax2.text(df.loc[i, 'x_1'], df.loc[i, 'z_1'], base, fontsize=8)
    ax2.set_title(f"(X vs Z)")
    ax2.set_xlabel('x_1')
    ax2.set_ylabel('z_1')
    # plt.show()
    
    # # Projection 3: Y vs Z
    ax3.plot(df['y_1'], df['z_1'])
    for i, base in enumerate(sequence):
        if i < len(df):
            ax3.text(df.loc[i, 'y_1'], df.loc[i, 'z_1'], base, fontsize=8)
    ax3.set_title(f"(Y vs Z)")
    ax3.set_xlabel('y_1')
    ax3.set_ylabel('z_1')
    plt.show()



first_train_seq = train_sequences['sequence'].iloc[0]
print(f" length of the sequence is: {len(first_train_seq)}")
print(f" the 2D plot of the sequence: {train_sequences['sequence'].iloc[0]}")
plot_sample_structure(train_sequences['target_id'].iloc[0], first_train_seq)


# Load a sample FASTA file corresponding to the target_id if available
# Let's assume the FASTA file is named as "{target_id}.fasta" in the data folder
target_id = train_sequences['target_id'].iloc[0]
fasta_file_path = f"/kaggle/input/stanford-rna-3d-folding/MSA/{target_id}.MSA.fasta"

# Parse the FASTA file using Biopython
fasta_sequences = list(SeqIO.parse(fasta_file_path, "fasta"))

# Checking if we can infer motifs, ligands, or other information from description
fasta_data = [{
    "id": fasta.id,
    "description": fasta.description,
    "sequence": str(fasta.seq)
} for fasta in fasta_sequences]

# Display as dataframe for better readability
fasta_df = pd.DataFrame(fasta_data)
from IPython.display import display
display(fasta_df)



import re

def extract_fasta_features(fasta_df):
    motif_keywords = ['hairpin', 'pseudoknot', 'quadruplex', 'stem-loop', 'kissing loop']
    ligand_keywords = ['ligand', 'bound', 'binding', 'small molecule', 'antibiotic']
    ion_keywords = ['Mg2+', 'K+', 'Na+', 'metal', 'ion']
    partner_keywords = ['protein', 'DNA', 'RNA', 'partner', 'interacting']
    experimental_keywords = ['in vitro', 'in vivo', 'crystal', 'NMR', 'EM']

    features = []

    for idx, row in fasta_df.iterrows():
        desc = row.get('description', '').lower()

        feature = {
            'target_id': row.get('target_id', ''),
            'has_motif': int(any(k in desc for k in motif_keywords)),
            'has_ligand': int(any(k in desc for k in ligand_keywords)),
            'has_ion_binding': int(any(k in desc for k in ion_keywords)),
            'has_partner': int(any(k in desc for k in partner_keywords)),
            'experimental_tag': int(any(k in desc for k in experimental_keywords)),
            'num_motifs_mentioned': sum(k in desc for k in motif_keywords),
            'description_length': len(desc.split())
        }

        features.append(feature)

    features_df = pd.DataFrame(features)
    return features_df



features_df = extract_fasta_features(train_sequences)
print(features_df.head())


import re

# Feature extraction function
def extract_features_from_metadata(df):
    features = []

    for idx, row in df.iterrows():
        desc = str(row.get('description', '')).lower()
        all_seq = str(row.get('all_sequences', ''))

        feature = {
            'target_id': row['target_id'],
            'has_partner_chain': int(all_seq.count('>') > 1),
            'num_chains': all_seq.count('>'),
            'ligand_present': int(bool(re.search(r'ligand|small molecule|antibiotic|drug|inhibitor', desc))),
            'experiment_type_nmr': int('nmr' in desc),
            'experiment_type_cryoem': int('cryo-em' in desc or 'cryo em' in desc),
            'experiment_type_xray': int('x-ray' in desc or 'xray' in desc),
            'has_protein_interaction': int('protein' in desc),
            'has_viral_origin': int('viral' in desc or 'virus' in desc),
            'has_pseudoknot': int('pseudoknot' in desc),
            'has_gquadruplex': int('g-quadruplex' in desc or 'quadruplex' in desc),
            'mentions_binding_site': int('binding' in desc or 'site' in desc or 'stem' in desc or 'loop' in desc or 'bulge' in desc)
        }

        features.append(feature)

    feature_df = pd.DataFrame(features)
    return feature_df

# Apply feature extraction
metadata_features = extract_features_from_metadata(train_sequences)

# Display extracted features
display(metadata_features)





