import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the datasets
train_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')
train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')


# Load the datasets
validation_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv')
validation_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_labels.csv')


train_sequences.head(2)


train_labels.head(2)


train_sequences.shape


train_labels.shape


validation_sequences.shape


validation_labels.shape


# Display basic info about the datasets
print("===== Train Sequences Info =====")
print(train_sequences.info())


print("\n===== Train Labels Info =====")
print(train_labels.info())


# Basic statistics for train_sequences
print("\n===== Train Sequences Description =====")
train_sequences.describe(include='all')


# Check for missing values
print("\nMissing values in train_sequences:")
print(train_sequences.isnull().sum())


# Check for missing values
print("\nMissing values in train_labels:")
print(train_labels.isnull().sum())


# Check unique values in each column
print("\nUnique values in each column:")
print(train_sequences.nunique())


# Check for non-unique values in the 'sequence' column
non_unique_sequences = train_sequences['sequence'].value_counts()
print("Non-unique sequences and their counts:")
print(non_unique_sequences[non_unique_sequences > 1])


# Basic statistics for train_labels
print("\n===== Train Labels Description =====")
train_labels.describe(include='all')


# Analyze the 'sequence' column in train_sequences
print("\n===== Sequence Length Analysis =====")
train_sequences['sequence_length'] = train_sequences['sequence'].apply(len)
train_sequences['sequence_length'].describe()


# Plot the distribution of sequence lengths
plt.figure(figsize=(10, 6))
sns.histplot(train_sequences['sequence_length'], bins=50, kde=True)
plt.title('Distribution of Sequence Lengths')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.show()


from collections import Counter

# Count nucleotide frequencies
nucleotide_counts = Counter("".join(train_sequences["sequence"]))
nucleotide_counts


# Plot nucleotide composition
plt.figure(figsize=(8, 6))
sns.barplot(x=list(nucleotide_counts.keys()), y=list(nucleotide_counts.values()))
plt.title("Nucleotide Composition")
plt.xlabel("Nucleotide")
plt.ylabel("Frequency")
plt.show()


# Analyze the 'temporal_cutoff' column in train_sequences
print("\n===== Temporal Cutoff Analysis =====")
train_sequences['temporal_cutoff'] = pd.to_datetime(train_sequences['temporal_cutoff'])
train_sequences['temporal_cutoff'].describe()


# Find the starting and ending dates
start_date = train_sequences['temporal_cutoff'].min()
end_date = train_sequences['temporal_cutoff'].max()

# Print the results
print("Starting Date:", start_date)
print("Ending Date:", end_date)


# Plot the distribution of temporal cutoff dates
plt.figure(figsize=(10, 6))
sns.histplot(train_sequences['temporal_cutoff'], bins=50, kde=True)
plt.title('Distribution of Temporal Cutoff Dates')
plt.xlabel('Temporal Cutoff')
plt.ylabel('Frequency')
plt.show()


# Analyze the 'resid' column in train_labels
print("\n===== Residue ID Analysis =====")
train_labels['resid'].describe()


# Plot the distribution of residue IDs
plt.figure(figsize=(10, 6))
sns.histplot(train_labels['resid'], bins=50, kde=True)
plt.title('Distribution of Residue IDs')
plt.xlabel('Residue ID')
plt.ylabel('Frequency')
plt.show()


# Analyze the 3D coordinates in train_labels
print("\n===== 3D Coordinates Analysis =====")
coordinates = train_labels[['x_1', 'y_1', 'z_1']].describe()
print(coordinates)


# Plot the distribution of x, y, z coordinates
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(train_labels['x_1'], bins=50, kde=True)
plt.title('Distribution of x-coordinates')

plt.subplot(1, 3, 2)
sns.histplot(train_labels['y_1'], bins=50, kde=True)
plt.title('Distribution of y-coordinates')

plt.subplot(1, 3, 3)
sns.histplot(train_labels['z_1'], bins=50, kde=True)
plt.title('Distribution of z-coordinates')

plt.tight_layout()
plt.show()


# Compute correlations
correlation_matrix = train_labels[["x_1", "y_1", "z_1"]].corr()

# Plot correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix for x, y, z Coordinates")
plt.show()


# Group by sequence length and analyze coordinates
sequence_length_groups = train_sequences.groupby("sequence_length").size().reset_index(name="count")

# Plot sequence length vs. coordinate statistics
plt.figure(figsize=(10, 6))
sns.scatterplot(data=sequence_length_groups, x="sequence_length", y="count")
plt.title("Sequence Length vs. Number of Sequences")
plt.xlabel("Sequence Length")
plt.ylabel("Number of Sequences")
plt.show()


# Boxplot for x, y, z coordinates
plt.figure(figsize=(10, 6))
sns.boxplot(data=train_labels[["x_1", "y_1", "z_1"]])
plt.title("Boxplot of x, y, z Coordinates")
plt.show()


import pandas as pd
import plotly.graph_objects as go


def plot_structure(df: pd.DataFrame, sequence_id: str) -> None:
    sequence_df = df[df["sequence_id"] == sequence_id]
    sequence_points = sequence_df[["x_1", "y_1", "z_1", "resname"]]
    
    colors = {"A": "red", "G": "blue", "C": "green", "U": "yellow"}
    fig = go.Figure()
    
    for resname, color in colors.items():
        subset = sequence_df[sequence_df["resname"] == resname]
        fig.add_trace(go.Scatter3d(
            x=subset["x_1"], y=subset["y_1"], z=subset["z_1"],
            mode='markers',
            marker=dict(size=5, color=color),
            name=resname,
        ))
    
    fig.add_trace(go.Scatter3d(
        x=sequence_df["x_1"], y=sequence_df["y_1"], z=sequence_df["z_1"],
        mode='lines',
        line=dict(color='gray', width=2),
        name='RNA Backbone'
    ))
    
    fig.update_layout(
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
            title=f'3D RNA Structure of sequence {sequence_id}',
        )
            
    fig.show(renderer="iframe")


train_labels["sequence_id"] = train_labels["ID"].apply(lambda x: "_".join(x.split("_")[:-1]))


train_labels.head(20)


plot_structure(train_labels, "1SCL_A")


# Select only rows where Group == '1SCL_A'
filtered_df = train_labels[train_labels["sequence_id"] == "1SCL_A"]
# Count occurrences of each nucleotide
nucleotide_counts = filtered_df["resname"].value_counts()
# Print the counts
print(nucleotide_counts)


# Filter nucleotides where Group is "1SCL_A"
nucleotide_sequence = "".join(train_labels[train_labels["sequence_id"] == "1SCL_A"]["resname"])
# Print the nucleotide sequence in one line
print(nucleotide_sequence)


plot_structure(train_labels, "1RHT_A")


# only in X-Y plane
plt.figure(figsize=(8, 6))
sns.scatterplot(x=train_labels["x_1"], y=train_labels["y_1"])
plt.title("Scatterplot of RNA C1' Atoms (X-Y Plane)")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.show()


# K-mer- Function, k=4
def get_kmers(sequence, k=3):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

kmer_counts = Counter()
for seq in train_sequences["sequence"]:
    kmer_counts.update(get_kmers(seq, k=4))
print(kmer_counts.most_common(20))


import matplotlib.pyplot as plt
import seaborn as sns

# Get the top 10 most common k-mers
top_kmers = kmer_counts.most_common(10)
kmer_names = [kmer for kmer, count in top_kmers]
kmer_counts = [count for kmer, count in top_kmers]

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=kmer_names, y=kmer_counts)
plt.title("Top 10 Most Frequent 4-mers")
plt.xlabel("4-mer")
plt.ylabel("Frequency")
plt.show()


def generate_kmer_counts(sequences, k):
    kmer_counts = Counter()
    for seq in sequences:
        kmer_counts.update(get_kmers(seq, k=k))
    return kmer_counts

# Generate k-mer counts for k=2, 3, 4
k2_counts = generate_kmer_counts(train_sequences["sequence"], k=2)
k3_counts = generate_kmer_counts(train_sequences["sequence"], k=3)
k4_counts = generate_kmer_counts(train_sequences["sequence"], k=4)

# Print top 5 for each k
print("Top 5 2-mers:", k2_counts.most_common(5))
print("Top 5 3-mers:", k3_counts.most_common(5))
print("Top 5 4-mers:", k4_counts.most_common(5))


def find_kmer_positions(sequences, kmer):
    positions = []
    for seq in sequences:
        start_indices = [i for i in range(len(seq) - len(kmer) + 1) if seq[i:i+len(kmer)] == kmer]
        positions.extend(start_indices)
    return positions

# Example: Find positions of the k-mer "AUGG"
kmer = "AUGG"
positions = find_kmer_positions(train_sequences["sequence"], kmer)
print(f"Positions of '{kmer}': {positions}")


from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Function to generate k-mers
def get_kmers(sequence, k=3):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

# Initialize a Counter to store k-mer frequencies
kmer_counts = Counter()

# Loop through each sequence in the dataset
for seq in train_sequences["sequence"]:
    kmer_counts.update(get_kmers(seq, k=4))

kmer_dict = dict(kmer_counts.most_common(30))

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(kmer_dict)

# Plot
plt.figure(figsize=(5, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("4-mer Word Cloud")
plt.show()

