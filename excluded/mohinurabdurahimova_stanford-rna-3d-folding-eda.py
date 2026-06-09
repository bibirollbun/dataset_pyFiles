import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import plotly.express as px
from gensim.models import Word2Vec
from fastai.tabular.all import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.patches as mpatches

sns.set()


rna_sequence_data = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv', nrows = 25000)
rna_sequence_data.head()


# Sequence length distribution
rna_sequence_data['seq_len'] = rna_sequence_data['sequence'].str.len()

plt.figure(figsize=(10,5))
sns.histplot(rna_sequence_data['seq_len'], bins=50, kde=True)
plt.title("Distribution of RNA Sequence Lengths")
plt.xlabel("Sequence Length")
plt.ylabel("Frequency")
plt.show()

# Nucleotide frequency
all_bases = ''.join(rna_sequence_data['sequence'].values)
base_counts = Counter(all_bases)

plt.figure(figsize=(6,4))
sns.barplot(x=list(base_counts.keys()), y=list(base_counts.values()))
plt.title("Nucleotide Frequency in RNA Sequences")
plt.xlabel("Base")
plt.ylabel("Count")
plt.show()



rna_sequence_data = rna_sequence_data.drop(columns=rna_sequence_data.filter(like='error').columns)
rna_sequence_data = rna_sequence_data.drop(['description'], axis=1)
# reactivity_cols = rna_sequence_data.filter(like='reactivity').columns
# rna_sequence_data[reactivity_cols] = rna_sequence_data[reactivity_cols].clip(lower=0)
# rna_sequence_data['reactivity'] = rna_sequence_data[reactivity_cols].mean(axis=1)
# rna_sequence_data = rna_sequence_data.drop(columns=reactivity_cols)
# rna_sequence_data


from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

# Compute overall codon frequency
overall_frequency = Counter()

for index, row in rna_sequence_data.iterrows():
    sequence = row['target_id']
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3) if len(sequence[i:i+3]) == 3]
    frequency = Counter(codons)
    overall_frequency.update(frequency)

# Create custom color palette
custom_palette = sns.color_palette("husl", len(overall_frequency))

# Plot overall codon frequency
plt.figure(figsize=(16, 6))
sns.barplot(x=list(overall_frequency.keys()), y=list(overall_frequency.values()), palette=custom_palette)
plt.xticks(rotation=90)
plt.xlabel('Codon')
plt.ylabel('Frequency')
plt.title('Overall Codon Frequency')
plt.show()



overall_frequency = Counter()

for index, row in rna_sequence_data.iterrows():
    sequence = row['target_id']
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    frequency = Counter(codons)
    
    overall_frequency.update(frequency)

custom_palette = sns.color_palette("husl", len(overall_frequency))

plt.figure(figsize=(16, 6))
sns.barplot(x=list(overall_frequency.keys()), y=list(overall_frequency.values()), palette=custom_palette)
plt.xticks(rotation=90)
plt.xlabel('Codon')
plt.ylabel('Frequency')
plt.title('Overall Codon Frequency')
plt.show()


sequences = rna_sequence_data['target_id']

codon_freqs = []
for sequence in sequences:
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    frequency = Counter(codons)
    codon_freqs.append(frequency)

df = pd.DataFrame(codon_freqs).fillna(0)

plt.figure(figsize=(16, 8))
sns.heatmap(df, cmap="YlGnBu")
plt.title('Heatmap of Codon Frequency')
plt.xlabel('Codon')

interval = 10000
plt.yticks(range(0, len(sequences), interval), range(0, len(sequences), interval))

plt.ylabel('Sequence Index')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Extend the nucleotides mapping to cover all unique characters in the 'target_id' column
all_chars = set("".join(rna_sequence_data['target_id'].astype(str).values))

# Create a dynamic mapping from characters to indices
extended_mapping = {char: idx for idx, char in enumerate(sorted(all_chars))}

# Apply the extended mapping to the 'target_id' column and pad/truncate to ensure uniform sequence length
SEQUENCE_LENGTH = 170
def map_and_truncate_or_pad(seq):
    mapped = [extended_mapping.get(n, -1) for n in seq[:SEQUENCE_LENGTH]]  # -1 for unknown characters
    return mapped + [-1] * (SEQUENCE_LENGTH - len(mapped))  # Pad with -1 if the sequence is too short

numerical_sequences = rna_sequence_data['target_id'].apply(map_and_truncate_or_pad)

# Convert the list of sequences into a numpy array
rna_map = np.array(numerical_sequences.tolist())

# Set the color map and plot the data
CMAP = 'Set3'

plt.figure(figsize=(16, 4))
im = plt.imshow(rna_map, aspect='auto', cmap=CMAP, interpolation='nearest')

# Zoom in on the left by adjusting the xlim
plt.xlim(0, SEQUENCE_LENGTH // 4)  # Adjust this to zoom more or less

# Create a legend for the plot using the extended mapping
color_values = list(extended_mapping.values())
colors = [im.cmap(im.norm(color_value)) for color_value in color_values]
patches = [mpatches.Patch(color=colors[i], label=char) for i, char in enumerate(extended_mapping.keys())]

# Add the legend
plt.legend(handles=patches, bbox_to_anchor=(0.65, -0.1), ncol=5, borderaxespad=0.)

# Show the plot
plt.show()



nucleotide_counts = Counter(sequence)

plt.pie(nucleotide_counts.values(), labels=nucleotide_counts.keys(), autopct='%1.1f%%')
plt.title('Nucleotide Composition Pie Chart')
plt.show()


window_size = 10

# Extract one RNA sequence
sequence = rna_sequence_data['sequence'].iloc[0]

# Count G + C + A + U per window
gc_content = [
    sequence[i:i+window_size].count('G') +
    sequence[i:i+window_size].count('C') +
    sequence[i:i+window_size].count('A') +
    sequence[i:i+window_size].count('U')
    for i in range(len(sequence) - window_size + 1)
]

# Plot
plt.figure(figsize=(16, 6))
plt.plot(gc_content)
plt.xlabel('Window Start Position')
plt.ylabel('Total RNA Bases in Window')
plt.title('Sliding Window Base Composition (G+C+A+U)')
plt.show()



overall_pair_freqs = Counter()

for seq in rna_sequence_data['sequence']:
    pairs = [(seq[i:i+3], seq[i+3:i+6]) for i in range(0, len(seq) - 3, 3)]
    overall_pair_freqs.update(pairs)

index = list(set([pair[0] for pair in overall_pair_freqs.keys()]))
columns = list(set([pair[1] for pair in overall_pair_freqs.keys()]))
pair_df = pd.DataFrame(index=index, columns=columns).fillna(0)

for pair, freq in overall_pair_freqs.items():
    pair_df.at[pair[0], pair[1]] = freq
    
plt.figure(figsize=(10, 6))
sns.heatmap(pair_df, cmap='viridis')
plt.title('Overall Codon Pair Bias')
plt.show()


codon_counts = defaultdict(list)

for seq in rna_sequence_data['sequence']:
    counter = Counter([seq[i:i+3] for i in range(0, len(seq), 3)])
    
    for codon in counter:
        codon_counts[codon].append(counter[codon])

for codon, counts in codon_counts.items():
    while len(counts) < len(rna_sequence_data['sequence']):
        counts.append(0)

codon_count = pd.DataFrame(dict(codon_counts))

fig = px.scatter_3d(codon_count, x='AUG', y='GCA', z='GCU')
fig.show()

