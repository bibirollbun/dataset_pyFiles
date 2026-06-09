import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
import scipy.stats as stats
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from collections import Counter
import subprocess
import re
import networkx as nx
import torch


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Decide between local or kaggle cloud storage         
KAGGLE_ENV = 'kaggle' in os.listdir('/')
data_path = '/kaggle/input' if KAGGLE_ENV else '../kaggle/input'

if KAGGLE_ENV:
    !pip install torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
    !apt-get update && apt-get install -y vienna-rna

from torch_geometric.data import Data
    
for dirname, _, filenames in os.walk(data_path):
    for filename in filenames:
        print(os.path.join(dirname, filename)) 


file_paths = {
    "df_sample_submission_sunny": data_path + "/standford-rna-3d-folding-sunny/sample_submission.csv",
    "df_test_sequences_sunny": data_path + "/standford-rna-3d-folding-sunny/test_sequences.csv",
    "df_train_labels_sunny": data_path + "/standford-rna-3d-folding-sunny/train_labels.csv",
    "df_train_sequences_sunny": data_path + "/standford-rna-3d-folding-sunny/train_sequences.csv",
    "df_validation_labels": data_path + "/stanford-rna-3d-folding/validation_labels.csv",
    "df_sample_submission": data_path + "/stanford-rna-3d-folding/sample_submission.csv",
    "df_test_sequences": data_path + "/stanford-rna-3d-folding/test_sequences.csv",
    "df_validation_sequences": data_path + "/stanford-rna-3d-folding/validation_sequences.csv",
    "df_train_labels": data_path + "/stanford-rna-3d-folding/train_labels.csv",
    "df_train_sequences": data_path + "/stanford-rna-3d-folding/train_sequences.csv"
}

for var_name, path in file_paths.items():
    try:
        globals()[var_name] = pd.read_csv(path)
        print(f"{var_name} load, {globals()[var_name].shape[0]} rows, {globals()[var_name].shape[1]} columns.")
    except FileNotFoundError:
        print(f"file not found: {path}")
    except Exception as e:
        print(f"error .. {path}: {e}")


display('df_train_sequences')
display(df_train_sequences.head())


display('df_train_labels')
display(df_train_labels.head(50))


df_train_sequences["length"] = df_train_sequences["sequence"].str.len()


median_value = df_train_sequences["length"].median()
q1 = df_train_sequences["length"].quantile(0.25)  
q3 = df_train_sequences["length"].quantile(0.75)  
iqr = q3 - q1 

lower_whisker = max(df_train_sequences["length"].min(), q1 - 1.5 * iqr)
upper_whisker = min(df_train_sequences["length"].max(), q3 + 1.5 * iqr)

min_value = df_train_sequences["length"].min()
max_value = df_train_sequences["length"].max()

stats_df = pd.DataFrame({
    "Statistic": ["Min", "Lower Whisker", "Q1 (25%)", "Median (50%)", "Q3 (75%)", "Upper Whisker", "Max"],
    "Value": [min_value, lower_whisker, q1, median_value, q3, upper_whisker, max_value]
})

print(stats_df)
plt.figure(figsize=(10, 5))
sns.boxplot(x=df_train_sequences["length"])
plt.title("Distribution of the Length of RNA Sequences")
plt.xlabel("Sequence Length")
plt.show()


# If this not work at kaggle, please pull it to your local machine and run it there
df_1SCL = df_train_labels[df_train_labels["ID"].str.startswith("1SCL_A_")].iloc[:29]

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=df_1SCL["x_1"], 
    y=df_1SCL["y_1"], 
    z=df_1SCL["z_1"],
    mode='markers',
    marker=dict(size=6, color='blue', opacity=0.8),
    text=df_1SCL["resname"],  # show by hovering over point the niklotide type
    name="C1' Carbon Atoms"
))

fig.update_layout(
    title="Interactive 3D RNA Structure - 1SCL",
    scene=dict(
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        zaxis_title="Z Coordinate"
    )
)
fig.show()


# Alternative 3D-Scatterplot for 1SCL RNA-Structure
df_1SCL = df_train_labels[df_train_labels["ID"].str.startswith("1SCL_A_")].iloc[:29]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df_1SCL["x_1"], df_1SCL["y_1"], df_1SCL["z_1"], c="blue", marker="o", label="C1' Carbon Atoms")

ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Z Coordinate")
ax.set_title("3D RNA Structure - 1SCL (C1' Carbon Atoms)")

plt.legend()
plt.show()


# only in X-Y plane
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_train_labels["x_1"], y=df_train_labels["y_1"])
plt.title("Scatterplot of RNA C1' Atoms (X-Y Plane)")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.show()



# Visualization of RNA C1' Atoms in 3D
if {"x_1", "y_1", "z_1"}.issubset(df_train_labels.columns):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df_train_labels["x_1"], df_train_labels["y_1"], df_train_labels["z_1"], alpha=0.6)

    ax.set_title("3D Scatterplot of RNA C1' Atoms")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")

    plt.show()

else:
    print("Missing columns: The columns 'x_1', 'y_1', and 'z_1' were not found.")


# Visualize RNA C1' Atoms in 3D with Plotly (interactive), if it is not working, please run it on your local machine
fig = px.scatter_3d(df_train_labels, x="x_1", y="y_1", z="z_1", opacity=0.7)
fig.show()


def find_atoms_in_region(df, x_target, y_target, z_target, tolerance=100):
    """
    Filter all RNA atoms that are within a given tolerance range around a target coordinate.
    
    Parameters:
    df (DataFrame): DataFrame containing atomic coordinates (must have 'x', 'y', 'z' columns).
    x_target (float): X-coordinate of the region of interest.
    y_target (float): Y-coordinate of the region of interest.
    z_target (float): Z-coordinate of the region of interest.
    tolerance (float): Range around the target coordinates to search for atoms.
    
    Returns:
    DataFrame: Filtered DataFrame with atoms in the specified region.
    """
    filtered_atoms = df[
        (df["x_1"].between(x_target - tolerance, x_target + tolerance)) &
        (df["y_1"].between(y_target - tolerance, y_target + tolerance)) &
        (df["z_1"].between(z_target - tolerance, z_target + tolerance))
    ]
    return filtered_atoms

region_atoms = find_atoms_in_region(df_train_labels, -730, -100, 400)
print(region_atoms) # there you can see how i found the 4V4G. Am I right?


# K-mer- Function, k=4
def get_kmers(sequence, k=3):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

kmer_counts = Counter()
for seq in df_train_sequences["sequence"]:
    kmer_counts.update(get_kmers(seq, k=4))
print(kmer_counts.most_common(10))


# k=8
kmer_counts = Counter()
for seq in df_train_sequences["sequence"]:
    kmer_counts.update(get_kmers(seq, k=8))

print(kmer_counts.most_common(10))


# k=12
kmer_counts = Counter()
for seq in df_train_sequences["sequence"]:
    kmer_counts.update(get_kmers(seq, k=12))  

print(kmer_counts.most_common(10))


# Calculate GC content
df_train_sequences["GC_content"] = df_train_sequences["sequence"].apply(
    lambda seq: (seq.count("G") + seq.count("C")) / len(seq)
)

plt.figure(figsize=(10, 5))
sns.histplot(df_train_sequences["GC_content"], bins=30, kde=True)
plt.title("GC Content Distribution")
plt.xlabel("GC Content")
plt.ylabel("Frequency")
plt.show()

print(df_train_sequences["GC_content"].describe())


# Merge sequence-based GC content with structural labels
df_combined = df_train_sequences.copy()
df_combined["ID"] = df_train_labels["ID"]  # Assuming IDs match
df_combined["resid"] = df_train_labels["resid"]
df_combined["x_1"] = df_train_labels["x_1"]
df_combined["y_1"] = df_train_labels["y_1"]
df_combined["z_1"] = df_train_labels["z_1"]

# Compute distance of each nucleotide from the origin (as a rough spatial feature)
df_combined["distance_from_origin"] = np.sqrt(
    df_combined["x_1"]**2 + df_combined["y_1"]**2 + df_combined["z_1"]**2
)

# Scatter plot of GC content vs. 3D structure distance
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df_combined["GC_content"], y=df_combined["distance_from_origin"])
plt.xlabel("GC Content")
plt.ylabel("Distance from Origin (3D Space)")
plt.title("GC Content vs. 3D RNA Spatial Structure")
plt.show()

# Compute correlation
correlation = df_combined[["GC_content", "distance_from_origin"]].corr()
print("Correlation between GC content and 3D structure distance:\n", correlation)


def predict_rna_structure(sequence):
    """Predicts RNA secondary structure and free energy using RNAfold."""
    process = subprocess.run(
        ["RNAfold"], input=sequence, capture_output=True, text=True
    )
    output = process.stdout.strip().split("\n")
    if len(output) < 2:
        return None, None

    structure = output[1].split(" ")[0]  # Extract secondary structure
    energy_match = re.search(r"-?\d+\.\d+", output[1])  # Extract energy
    energy = float(energy_match.group()) if energy_match else None

    return structure, energy

def rna_to_graph(sequence, structure):
    """Converts RNA sequence and secondary structure into a graph representation."""
    G = nx.Graph()
    for i, nucleotide in enumerate(sequence):
        G.add_node(i, nucleotide=nucleotide)

    # Sequential edges
    for i in range(len(sequence) - 1):
        G.add_edge(i, i + 1)

    # Base-pairing edges
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                G.add_edge(i, j)

    return G

# Get RNA sequence (1SCL example)
sequence = df_train_sequences["sequence"].iloc[0]

# Predict secondary structure & energy
structure, energy = predict_rna_structure(sequence)

if structure:
    print(f"RNA Structure: {structure} | Energy: {energy} kcal/mol")

    # Convert to graph and PyTorch Geometric format
    rna_graph = rna_to_graph(sequence, structure)
    edge_index = torch.tensor(list(rna_graph.edges), dtype=torch.long).t().contiguous()
    x = torch.tensor([[ord(n)] for n in sequence], dtype=torch.float)  

    data = Data(x=x, edge_index=edge_index, energy=torch.tensor([energy], dtype=torch.float))
    print(data)
else:
    print("Error: Could not generate RNA secondary structure.")



print(f"Total Nodes: {data.num_nodes}")
print(f"Total Edges: {data.num_edges}")
print("Edge List:")
print(data.edge_index.t().tolist())  # Print all edges to inspect them



def data_overview(data, target):
    # Overview
    display(Markdown("## Data Overview"))
    
    display(Markdown("### General Information"))
    display(Markdown(f"- Number of rows and columns: {data.shape[0]} x {data.shape[1]}"))
    display(Markdown("- Column names:"))
    display(list(data.columns))

    display(Markdown("### Data Types & Missing Values"))
    missing = data.isnull().sum()
    dtypes = pd.DataFrame(data.dtypes, columns=["Data Type"])
    missing_df = pd.DataFrame(missing, columns=["Missing Values"])
    overview_df = dtypes.join(missing_df)
    display(overview_df.style.background_gradient(cmap="coolwarm"))

    display(Markdown("### Classic head of Data"))
    display(data.head().style.set_properties(**{"background-color": "#f5f5f5"}))

    display(Markdown("### Statistical Summary (describe)"))
    display(data.describe().T.style.background_gradient(cmap="viridis"))

    # Target variable analysis
    if target is not None:
        display(Markdown(f"## Target Variable: `{target}`"))
        sns.set_style("whitegrid")  
        sns.set_palette("viridis")   

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        #   Absolute frequency barplot
        sns.barplot(x=data[target].value_counts().index, 
                y=data[target].value_counts(), 
                ax=ax[0])  

        ax[0].set_title("Absolute Frequency", fontsize=12, fontweight="bold")
        ax[0].set_ylabel("Count")
        ax[0].set_xlabel(target)
        ax[0].grid(axis="y", linestyle="--", alpha=0.5)  

        # Percentage distribution barplot
        sns.barplot(x=data[target].value_counts().index, 
                    y=data[target].value_counts(normalize=True), 
                    ax=ax[1])  

        ax[1].set_title("Percentage Distribution", fontsize=12, fontweight="bold")
        ax[1].set_ylabel("Percentage")
        ax[1].set_xlabel(target)
        ax[1].grid(axis="y", linestyle="--", alpha=0.5)

    

        for spine in ["top", "right"]:
            ax[0].spines[spine].set_visible(False)
            ax[1].spines[spine].set_visible(False)

    plt.tight_layout()
    plt.show()


data_overview(df_train_sequences, target=None)


data_overview(df_train_sequences_sunny, target=None)


data_overview(df_train_labels, target=None)


data_overview(df_train_labels_sunny, target=None)

