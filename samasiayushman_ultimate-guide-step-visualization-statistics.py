# Optional if want to Merge

import pandas as pd

# Load the CSV file
data_label1 = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')  
data_label2=pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.v2.csv")

data_label=pd.concat([data_label1,data_label2],ignore_index=True)
print(data_label.head())
print("Merge Successfull with Shape")
data_label.describe()


!pip install biopython numpy pandas torch torch-geometric


from Bio import SeqIO

def load_fasta(file_path):
    return {record.id: str(record.seq) for record in SeqIO.parse(file_path, "fasta")}

train_seqs = load_fasta("/kaggle/input/stanford-rna-3d-folding/MSA/17RA_A.MSA.fasta")
print(f"First RNA: {list(train_seqs.items())[0]}")


import pandas as pd

# Load sequence and structure data
train_seqs = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")

print("Sequences:\n", train_seqs.head())
print("\nLabels:\n", train_labels.head())


import numpy as np

def one_hot_encode(sequence):
    mapping = {'A': [1,0,0,0], 'U': [0,1,0,0], 
               'C': [0,0,1,0], 'G': [0,0,0,1]}
    return np.array([mapping.get(base, [0,0,0,0]) for base in sequence])

# Example: One-hot encode the first sequence
seq = train_seqs.iloc[0]['sequence']
encoded_seq = one_hot_encode(seq)
print(f"Encoded shape: {encoded_seq.shape}")  



structures = train_labels.groupby(['ID', 'resid'])[['x_1', 'y_1', 'z_1']].mean().reset_index()
print(structures.head())


import pandas as pd


#  submission = pd.DataFrame({
#   'atom_id': range(len(pred_coords)),
#    'resname': ['A'] * len(pred_coords),  # Placeholder
#    'resid': range(1, len(pred_coords)+1),
#    'x1': pred_coords[:, 0], 'y1': pred_coords[:, 1], 'z1': pred_coords[:, 2],
              # ... Repeat for x2,y2,z2 to x5,y5,z5 
#  })

# submission.to_csv("submission.csv", index=False)
print(" Remove # above to Submit your Contest Files")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_seqs = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")
train_seqs['seq_length'] = train_seqs['sequence'].apply(len)

# Plot histogram
plt.figure(figsize=(10, 5))
sns.histplot(train_seqs['seq_length'], bins=30, kde=True)
plt.title("Distribution of RNA Sequence Lengths")
plt.xlabel("Length")
plt.ylabel("Count")
plt.show()


from collections import Counter

# Count nucleotides in all sequences
all_nucleotides = ''.join(train_seqs['sequence'])
nucleotide_counts = Counter(all_nucleotides)

# Plot
plt.figure(figsize=(8, 4))
sns.barplot(x=list(nucleotide_counts.keys()), y=list(nucleotide_counts.values()))
plt.title("Nucleotide Frequency in Training Data")
plt.xlabel("Nucleotide")
plt.ylabel("Count")
plt.show()


import plotly.express as px

train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
# Extract first RNA's coordinates
first_rna = train_labels[train_labels['ID'] == train_labels['ID'].iloc[0]]  # Change index as needed

# 3D scatter plot
fig = px.scatter_3d(first_rna, x='x_1', y='y_1', z='z_1', color='resid', 
                    title="3D RNA Structure (Plotly)")
fig.show()


!pip install ViennaRNA


import RNA  

# Predict and plot secondary structure
sequence = train_seqs['sequence'].iloc[0]  # First sequence
structure, _ = RNA.fold(sequence)

# Draw structure
print(f"Sequence: {sequence}")
print(f"Predicted Structure: {structure}")
RNA.svg_rna_plot(sequence, structure, "secondary_structure.svg")

# Display in notebook
from IPython.display import SVG
SVG("secondary_structure.svg")


from scipy.spatial.distance import pdist, squareform

# Calculate pairwise distances between residues
coords = first_rna[['x_1', 'y_1', 'z_1']].values
dist_matrix = squareform(pdist(coords))

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(dist_matrix, cmap="viridis")
plt.title("Residue-Residue Distance Matrix")
plt.xlabel("Residue ID")
plt.ylabel("Residue ID")
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. Generate sample data (10 residues, 3D)
coords = np.random.rand(10, 3)  # 10 residues, (x,y,z) coordinates
residue_ids = np.arange(10)     # Corresponding residue IDs (0-9)

# 2. Verify data shape
print("Coordinates shape:", coords.shape)  # Should be (10, 3)
print("Residue IDs shape:", residue_ids.shape)  # Should be (10,)

# 3. Perform PCA
pca = PCA(n_components=2)
coords_2d = pca.fit_transform(coords)
print("PCA result shape:", coords_2d.shape)  # Should be (10, 2)

# 4. Plot with consistent dimensions
plt.figure(figsize=(8, 6))
scatter = plt.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                     c=residue_ids,        # Use matching residue IDs
                     cmap="tab20",
                     s=100)               # Adjust point size
plt.colorbar(scatter, label="Residue ID")
plt.title("PCA of RNA 3D Coordinates")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()


from collections import Counter
import numpy as np

import pandas as pd

counter =0

data_seq = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')  
s = data_seq["sequence"]
for seq in s:
    print(seq)
    counter+=1 
    if(counter>8): # Only Print upto 8 Strand 
        break
    nuc_counts = Counter(seq)
    gc_content = (nuc_counts['G'] + nuc_counts['C']) / len(seq) * 100
    print(f"Nucleotide counts: {nuc_counts}")
    print(f"GC content: {gc_content:.2f}%")



from scipy.spatial.distance import pdist, squareform

# coords = Nx3 array of atomic positions
dist_matrix = squareform(pdist(coords))
mean_distance = np.mean(dist_matrix)
std_distance = np.std(dist_matrix)

print(f"Mean inter-atomic distance: {mean_distance:.2f} Å")
print(f"Standard deviation: {std_distance:.2f} Å")


def radius_of_gyration(coords):
    centroid = np.mean(coords, axis=0)
    return np.sqrt(np.mean(np.sum((coords - centroid)**2, axis=1)))

rgyr = radius_of_gyration(coords)
print(f"Radius of gyration: {rgyr:.2f} Å")


# Using ViennaRNA (install with !pip install ViennaRNA)
import RNA

counter =0
data = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
target_col = data['sequence']

for s in target_col:
    counter+=1
    if(counter >8):  # Counter to Count Data 
        break
    structure, _ = RNA.fold(s)
    paired_bases = structure.count('(') + structure.count(')')
    pairing_percentage = paired_bases / len(structure) * 100
    
    print(f"Predicted structure: {structure}")
    print(f"Paired bases: {pairing_percentage:.2f}%")


