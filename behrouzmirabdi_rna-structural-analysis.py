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
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


# Load training labels

train_labels=pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")

train_labels.shape



train_labels.head() 


train_labels.describe()


# Define pdb_id and chain_id
# Split the 'ID' column into 'pdb_id' and 'chain_id'
train_labels['pdb_id'] = train_labels['ID'].apply(lambda x: x.split('_')[0])

train_labels['chain'] = train_labels['ID'].apply(lambda x: x.split('_')[1])


train_labels.head()


# Count missing values in each column
missing_per_column = train_labels.isna().sum()

print(missing_per_column)


perc_missing_per_column = (missing_per_column/len(train_labels))* 100
print(f"Percentage of the data is missing in each column:\n{perc_missing_per_column.round(2)}")


missing_pdb_ids = train_labels[train_labels[['x_1', 'y_1', 'z_1']].isna().any(axis=1)]['pdb_id'].unique()
print(f"Missing pdb_ids: {missing_pdb_ids}")
print(f"Number of missing pdb_ids: {len(missing_pdb_ids)}")


import pprint

missing_resids = train_labels[train_labels[['x_1', 'y_1', 'z_1']].isna().any(axis=1)][['pdb_id', 'resid']]
pprint.pprint(missing_resids)


missing_dict = missing_resids.groupby('pdb_id')['resid'].apply(list).to_dict()
for pdb_id, resids in missing_dict.items():
    pprint.pprint({pdb_id: resids})
    break 


# Group by resid and count the number of missing values
missing_resid_counts = missing_resids.groupby('resid').size().reset_index(name='count')

# Filter out residues with more than 2 missing values to focus on significant ones
filter_missing_resid_counts = missing_resid_counts[missing_resid_counts['count']>2]


# Plot the accumulation of missing values

plt.figure(figsize=(12, 6))

ax = sns.barplot(
    data=filter_missing_resid_counts,
    x='resid',
    y='count',
    color='blue'
)

# Add title and labels
plt.title('Accumulation of Missing Values by Residue', fontsize=16)
plt.xlabel('Residues')
plt.ylabel('Number of Missing Values')

# Rotate x-axis labels for better readability
plt.xticks(
    ticks=range(0,len(filter_missing_resid_counts), 50),
    rotation=45
)

# Adjust layout to prevent cutting off labels
plt.tight_layout()

plt.show()


# Drop missing values
train_labels = train_labels.dropna(subset=['x_1', 'y_1', 'z_1'])

# Check if there are any missing values left
missing_values_after = train_labels.isna().sum()
print(f"Missing values after dropping: {missing_values_after}")


train_labels.shape


# Group by target_id and chain_id

structures_by_target = defaultdict(list)
for _, row in train_labels.iterrows():
    full_id = row["ID"]
    target_chain_id = "_".join(full_id.split("_")[0:2])
    structures_by_target[target_chain_id].append(row)

print(f"Number of unique structures: {len(structures_by_target)}")


# Display the first target ID and its associated data

target_ids = list(structures_by_target.keys())

first_target_id = target_ids[0]
first_structures = structures_by_target[first_target_id]

print(f"First target ID: {first_target_id}")
print(f"Number of structures for this target: {len(first_structures)}")
print("First structure data:")
print(first_structures[0])


structural_stats = {}

for target_id, residues in structures_by_target.items():
    # Sort residues by resid
    residues.sort(key=lambda x: int(x["resid"]))

    # Calculate the distance between consecutive C1' atoms
    distances = []
    for i in range(1, len(residues)):
        prev = residues[i-1]
        curr = residues[i]

        dx = float(curr["x_1"]) - float(prev["x_1"])
        dy = float(curr["y_1"]) - float(prev["y_1"])
        dz = float(curr["z_1"]) - float(prev["z_1"])

        # Calculates the Euclidean distance between consecutive C1' atoms using the Pythagorean formula
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        distances.append(distance)

    # Calculate statistics 
    if distances:
        avg_distance = np.mean(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
    else:
        avg_distance = min_distance = max_distance = 0

    # Calculate radius of gyration (rough measure of compactness)
    coords = np.array([[float(r['x_1']), float(r['y_1']), float(r['z_1'])] for r in residues])
    center = np.mean(coords, axis=0)

    # Calculate radius of gyration
    rg = np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1)))

    structural_stats[target_id] = {
        'num_residues' : len(residues),
        'avg_consecutive_distance' : avg_distance,
        'min_consecutive_distance' : min_distance,
        'max_consecutive_distance' : max_distance,
        'radius_of_gyration' : rg
    }



# Print statistics for a few examples
print('\nStructural statistics for sample targets: ')
for target_id, stats in list(structural_stats.items())[:5]:
    print(f"\nTarget ID: {target_id}")
    print(f"Number of residues: {stats['num_residues']}")
    print(f"Average consecutive C1' distance: {stats['avg_consecutive_distance']:.2f} Ã…")
    print(f"Min/Max consecutive distances: {stats['min_consecutive_distance']:.2f}/{stats['max_consecutive_distance']:.2f} Ã…")
    print(f"Radius of gyration: {stats['radius_of_gyration']:.2f} Ã…")


# Calculate overall statistics
all_radii = [s['radius_of_gyration'] for s in structural_stats.values()]
avg_radius = np.mean(all_radii)
min_radius = np.min(all_radii)
max_radius = np.max(all_radii)

print("\nOverall structural statistics: ")
print(f"Radius of gyration (compactness measure): ")
print(f"- Average: {avg_radius:.2f} Ã…")
print(f"- Minimum: {min_radius:.2f} Ã…")
print(f"- Maximum: {max_radius:.2f} Ã…")


# Prepare data for the scatter plot
sequence_lengths = [stats['num_residues'] for stats in structural_stats.values()]
radii_of_gyration = [stats['radius_of_gyration'] for stats in structural_stats.values()]

# Create the scatter plot
plt.figure(figsize=(10, 6))
#plt.style.use('dark_background')

sns.scatterplot(
    x=sequence_lengths,
    y=radii_of_gyration,
    alpha=0.6, 
    s=80,      
    color='#1f77b4'  
)

# Add title and labels
plt.title('Radius of Gyration vs. Sequence Length', fontsize=16)
plt.xlabel('Sequence Length (number of residues)')
plt.ylabel('Radius of Gyration (Ã…)')

# Improve layout
plt.tight_layout()

plt.show()


import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit

# Define power law function
def power_law(x, a, b):
    return a * x ** b
# Fit the power law to the data
params, _ = curve_fit(power_law, sequence_lengths, radii_of_gyration, maxfev=10000)
a, b = params

# Generate trendline
x_fit = np.linspace(min(sequence_lengths), max(sequence_lengths), 500)
y_fit = power_law(x_fit, a, b)

# Create the scatter plot with trendline
plt.figure(figsize=(10, 6))
#plt.style.use('dark_background')

# Create the scatter plot with Seaborn
sns.scatterplot(
    x=sequence_lengths,
    y=radii_of_gyration,
    alpha=0.6,  
    s=80,       
    color='#1f77b4', 
    label='RNA Structures'
)

# Add the power law trendline
plt.plot(x_fit, y_fit, color='red', linewidth=2, 
         label=f'Power-Law Fit:\n Rg = {a:.2f} * L^{b:.2f}')

plt.title('Radius of Gyration vs. Sequence Length with Power Law Fit', fontsize=16)
plt.xlabel('Sequence Length (number of residues)')
plt.ylabel('Radius of Gyration (Ã…)')

plt.legend()

plt.tight_layout()

plt.show()



# Prepare data for the histogram
all_distances = [
    stats['avg_consecutive_distance']
    for stats in structural_stats.values()
    if stats['num_residues'] > 1  # Only include structures with at least 2 residues
]

plt.figure(figsize=(10, 6))
#plt.style.use('dark_background')

# Create the histogram
sns.histplot(
    all_distances,
    bins=50,
    color='cornflowerblue',
    kde=False
)

plt.title("Distribution of Average Consecutive C1' Distances", fontsize=16)
plt.xlabel("Distance (Ã…)")
plt.ylabel("Count")

plt.tight_layout()

plt.show()


! pip install py3Dmol
! pip install Bio


import py3Dmol
from Bio import PDB

def fetch_and_visualize_rna_py3dmol(pdb_id):
    """
    Fetches an RNA 3D structure from the RCSB PDB and visualizes it using py3Dmol.
    
    Parameters:
    pdb_id (str): The PDB ID of the RNA structure to fetch (e.g, '1JWC').
    
    returns:
    py3Dmol.view: An interactive 3D viewer for the RNA structure.
    """
    pdb_parser = PDB.PDBList()
    filename = pdb_parser.retrieve_pdb_file(pdb_id, file_format='pdb', pdir='.', overwrite=True)
    
    # Read the PDB file
    with open(filename, 'r') as f:
        pdb_data = f.read()
    
    # Create a py3Dmol view and add the structure
    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_data, 'pdb')
    
    # Style the view
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.addStyle({'hetflag': True}, {'stick': {}})
    
    # Configure the view
    view.zoomTo()
    view.setBackgroundColor('white')
    
    return view


view = fetch_and_visualize_rna_py3dmol('1JWC')
view.show()


train_labels.to_csv("/kaggle/working/train_label_pdb.csv", index=False)


train_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")


# Extract the relevant data into separate mapping dictionaries
avg_dist_map = {k: v['avg_consecutive_distance'] for k, v in structural_stats.items()}
rg_map = {k: v['radius_of_gyration'] for k, v in structural_stats.items()}

# Add new columns to train_sequences
train_sequences['avg_consecutive_distance'] = train_sequences['target_id'].map(avg_dist_map)
train_sequences['radius_of_gyration'] = train_sequences['target_id'].map(rg_map)


train_sequences.head()


train_sequences.to_csv("/kaggle/working/train_sequences_dist_rg.csv", index=False)




