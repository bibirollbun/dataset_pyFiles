!pip install viennarna


import pandas as pd
import numpy as np

PREFIX = '/kaggle/input/stanford-rna-3d-folding'
# Edit this
data_path = f"{PREFIX}/test_sequences.csv"
train_data = pd.read_csv(data_path)

# Remove inf values
train_data.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
train_data.head()

summary_stats = train_data.describe()
summary_stats


# Check if x, y, and z coordinates are present in the training data
if {'x', 'y', 'z'}.issubset(train_data.columns):
    # Plot boxplots for x, y, and z coordinates
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=train_data[['x', 'y', 'z']])
    plt.title('Boxplot of X, Y, and Z Coordinates in Training Data')
    plt.xlabel('Coordinate')
    plt.ylabel('Value')
    plt.show()
else:
    print("X, Y, and Z coordinates are not available in the training data.")


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the length of each sequence
train_data['sequence_length'] = train_data['sequence'].apply(len)

# Generate summary statistics
summary_stats = train_data.describe()

# Plot the distribution of sequence lengths using a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x=train_data['sequence_length'])
plt.title('Boxplot of Sequence Lengths in Training Data')
plt.xlabel('Sequence Length')
plt.show()

summary_stats


# Check for duplicate sequences in the training data
duplicate_sequences = train_data[train_data.duplicated('sequence', keep=False)]

print(len(duplicate_sequences))
duplicate_sequences


test_data_path = f"{PREFIX}/test_sequences.csv"
test_data = pd.read_csv(test_data_path)

train_data['temporal_cutoff'] = pd.to_datetime(train_data['temporal_cutoff'])
test_data['temporal_cutoff'] = pd.to_datetime(test_data['temporal_cutoff'])

# Verify temporal cutoff compliance
temporal_compliance = train_data['temporal_cutoff'].max() < test_data['temporal_cutoff'].min()

temporal_compliance


train_labels_path = f"{PREFIX}/train_labels.csv"
train_labels = pd.read_csv(train_labels_path)
# Remove nas
train_labels = train_labels.dropna()
print(train_labels)

plt.figure(figsize=(12, 6))
sns.boxplot(data=train_labels[['x_1', 'y_1', 'z_1']])
plt.title('Boxplot of X_1, Y_1, and Z_1 Coordinates in Training Labels')
plt.xlabel('Coordinate')
plt.ylabel('Value')
plt.show()


sample_target = train_labels
x = sample_target['x_1'].values
y = sample_target['y_1'].values
z = sample_target['z_1'].values

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(222, projection='3d')
ax.scatter(x, y, z, c='green', marker='o')
ax.set_title('3D RNA Structure')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()



from collections import Counter

def count_kmers(sequence, k=3):
    return Counter([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

kmer_frequencies = train_data['sequence'].apply(lambda seq: count_kmers(seq, k=3))

total_kmer_counts = sum(kmer_frequencies, Counter())

total_kmer_counts.most_common(10)


def gc_content(sequence):
    return (sequence.count('G') + sequence.count('C')) / len(sequence)

train_data['gc_content'] = train_data['sequence'].apply(gc_content)

plt.figure(figsize=(10, 6))
sns.histplot(train_data['gc_content'], bins=30, kde=True)
plt.title('Distribution of GC Content in Training Data')
plt.xlabel('GC Content')
plt.ylabel('Frequency')
plt.show()


import re

def find_g_quadruplex(sequence):
    pattern = r'(G{3,}\w{1,7}){3,}G{3,}'
    return bool(re.search(pattern, sequence))

train_data['g_quadruplex'] = train_data['sequence'].apply(find_g_quadruplex)

g_quadruplex_count = train_data['g_quadruplex'].sum()

# Print the count and strand IDs
g_quadruplex_strands = train_data[train_data['g_quadruplex']]['target_id'].tolist()
g_quadruplex_count, g_quadruplex_strands


# Filter PDB IDs with more than one occurrence
pdb_id_counts = train_data['target_id'].str.split('_').str[0].value_counts()
print(pdb_id_counts)

filtered_pdb_id_counts = pdb_id_counts[pdb_id_counts > 1]
print(len(filtered_pdb_id_counts))

# Looks like all pdb_id's are unique


train_data['chain_count'] = train_data['all_sequences'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)

plt.figure(figsize=(10, 6))
sns.histplot(train_data['chain_count'], bins=20, kde=True)
plt.title('Distribution of Chain Multiplicity in Training Data')
plt.xlabel('Number of Chains')
plt.ylabel('Frequency')
plt.show()

train_data['experimental_method'] = train_data['description'].apply(lambda x: 'X-ray' if isinstance(x, str) and 'X-ray' in x else ('NMR' if isinstance(x, str) and 'NMR' in x else 'Other'))

plt.figure(figsize=(8, 5))
sns.countplot(y='experimental_method', data=train_data)
plt.title('Distribution of Experimental Methods')
plt.xlabel('Count')
plt.ylabel('Experimental Method')
plt.show()

train_data['has_ligand'] = train_data['description'].apply(lambda x: 'ligand' in x.lower() if isinstance(x, str) else False)

# Count sequences with ligands
ligand_count = train_data['has_ligand'].sum()
ligand_count


import RNA

def get_ensemble_diversity(seq):
    fc = RNA.fold_compound(seq)
    (ss, mfe) = fc.mfe()
    fc.pf()
    ensemble_diversity = fc.mean_bp_distance()  # Measures structural variability
    return ensemble_diversity

# Uncomment to run on full dataset
# train_data['ensemble_diversity'] = train_data['sequence'].apply(get_ensemble_diversity)


sample_data = train_data.sample(10, random_state=1)
sample_data['ensemble_diversity'] = sample_data['sequence'].apply(get_ensemble_diversity)

# Display the results
sample_data[['target_id', 'ensemble_diversity']]


test_data_path = f"{PREFIX}/test_sequences.csv"
test_data = pd.read_csv(test_data_path)

test_data['temporal_cutoff'] = pd.to_datetime(test_data['temporal_cutoff'])

temporal_compliance = train_data['temporal_cutoff'].max() < test_data['temporal_cutoff'].min()

temporal_compliance


# Load the validation sequences data
validation_data_path = f"{PREFIX}/validation_sequences.csv"
validation_data = pd.read_csv(validation_data_path)

# Convert temporal_cutoff to datetime
validation_data['temporal_cutoff'] = pd.to_datetime(validation_data['temporal_cutoff'])

# Verify temporal cutoff compliance for validation set
validation_compliance = train_data['temporal_cutoff'].max() < validation_data['temporal_cutoff'].min()

# Plot timeline of temporal_cutoff dates for validation data
plt.figure(figsize=(12, 6))
validation_data['temporal_cutoff'].hist(bins=30, color='lightgreen')
plt.title('Timeline of Temporal Cutoff Dates in Validation Data')
plt.xlabel('Date')
plt.ylabel('Frequency')
plt.show()

validation_compliance


import matplotlib.pyplot as plt

# Convert temporal_cutoff to datetime if not already
train_data['temporal_cutoff'] = pd.to_datetime(train_data['temporal_cutoff'])

plt.figure(figsize=(12, 6))
train_data['temporal_cutoff'].hist(bins=30, color='skyblue')
plt.title('Timeline of Temporal Cutoff Dates')
plt.xlabel('Date')
plt.ylabel('Frequency')
plt.show()

