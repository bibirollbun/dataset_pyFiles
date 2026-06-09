from IPython.display import YouTubeVideo
YouTubeVideo('2XTi9LG9NnU', width=800, height=300)


# General
import glob
import os
import pandas as pd
import numpy as np
# import pandas_profiling as pp

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import missingno as msno

# Options
# pd.set_option('display.max_columns', 1000)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

# Load datasets
train_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')
validation_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv')
test_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')
train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
validation_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_labels.csv')


# Basic Info
def dataset_overview(df, name):
    print(f"{name} Dataset Overview")
    print(df.info())
    print("Missing Values:")
    print(df.isnull().sum())
    print("\n")

dataset_overview(train_sequences, "Train Sequences")
dataset_overview(train_labels, "Train Labels")


# Sequence Length Distribution
train_sequences['seq_length'] = train_sequences['sequence'].apply(len)
plt.figure(figsize=(10, 5))
sns.histplot(train_sequences['seq_length'], bins=30, kde=True)
plt.xlabel("Sequence Length")
plt.ylabel("Count")
plt.title("RNA Sequence Length Distribution")
plt.show()


# Nucleotide Composition
def nucleotide_composition(sequences):
    nucleotides = ['A', 'C', 'G', 'U']
    counts = {nuc: 0 for nuc in nucleotides}
    
    for seq in sequences:
        for char in seq:
            if char in counts:
                counts[char] += 1
    
    return counts

comp = nucleotide_composition(train_sequences['sequence'])
sns.barplot(x=list(comp.keys()), y=list(comp.values()))
plt.title("Nucleotide Composition in Training Set")
plt.show()


# Temporal Cutoff Distribution
train_sequences['temporal_cutoff'] = pd.to_datetime(train_sequences['temporal_cutoff'], errors='coerce')
plt.figure(figsize=(10, 5))
sns.histplot(train_sequences['temporal_cutoff'].dropna(), bins=30, kde=True)
plt.xlabel("Temporal Cutoff Year")
plt.ylabel("Count")
plt.title("Temporal Cutoff Distribution")
plt.show()



# Checking Structural Data (Train Labels)
def structural_summary(df):
    print("Unique Targets:", df['ID'].apply(lambda x: x.split('_')[0]).nunique())
    print("Unique Residues:", df['ID'].apply(lambda x: x.split('_')[1]).nunique())
    print("Nucleotide Types:", df['resname'].unique())
    print("Coordinate Columns:", df.columns[3:].tolist())

structural_summary(train_labels)


# PCA for Structural Data
coordinate_cols = [col for col in train_labels.columns if 'x_' in col or 'y_' in col or 'z_' in col]
pca = PCA(n_components=2)
pca_result = pca.fit_transform(train_labels[coordinate_cols].dropna())
plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection of RNA Structures")
plt.show()


# t-SNE Visualization
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(train_labels[coordinate_cols].dropna())
plt.figure(figsize=(8,6))
sns.scatterplot(x=tsne_result[:,0], y=tsne_result[:,1], alpha=0.5)
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.title("t-SNE Visualization of RNA Structures")
plt.show()


# 3D Visualization of RNA Structures
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train_labels['x_1'], train_labels['y_1'], train_labels['z_1'], alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Visualization of RNA Structures')
plt.show()

print("EDA Completed! ğŸš€")





