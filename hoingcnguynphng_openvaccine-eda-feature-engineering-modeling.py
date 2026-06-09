import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import graphviz
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

np.random.seed(42)
torch.manual_seed(42)

# Cài đặt trực quan hóa
plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")


# Load datasets
train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
sample_sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTrain columns: {train.columns.tolist()}")
print(f"\nTest columns: {test.columns.tolist()}")


train.info()
print("\n" + "="*80)
test.info()


print("\n" + "="*80 + "\nTraining Set Statistical Summary:")
print(train.describe())
print("\n" + "="*80 + "\nTesting Set Statistical Summary:")
print(test.describe())



train.head(5)


test.head(5)


# Compute the mean value for each target variable
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']
train_targets = {}

for col in target_cols:
    train[f'mean_{col}'] = train[col].apply(lambda x: np.mean(x))
    train_targets[col] = train[f'mean_{col}'].values

# Visualize target distributions
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, col in enumerate(target_cols):
    axes[idx].hist(train[f'mean_{col}'], bins=50, alpha=0.7, edgecolor='black')
    axes[idx].set_title(f'Distribution of Mean {col}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Mean Value')
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(True, alpha=0.3)

# Remove unused subplot
fig.delaxes(axes[5])
plt.tight_layout()
plt.savefig('target_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nTarget Statistics:")
for col in target_cols:
    print(f"{col:15s}: mean={train[f'mean_{col}'].mean():.4f}, std={train[f'mean_{col}'].std():.4f}")



fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Train sequence length distribution
axes[0].bar(['seq_length=107'], [train['seq_length'].value_counts()[107]],
            color='steelblue', edgecolor='black', width=0.5)
axes[0].set_title('Train Set: Sequence Length Distribution', fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].grid(axis='y', alpha=0.3)

# Test sequence length distribution
test_seq_counts = test['seq_length'].value_counts().sort_index()
axes[1].bar(test_seq_counts.index.astype(str), test_seq_counts.values,
            color=['coral', 'lightseagreen'], edgecolor='black')
axes[1].set_title('Test Set: Sequence Length Distribution', fontweight='bold')
axes[1].set_xlabel('Sequence Length')
axes[1].set_ylabel('Count')
axes[1].grid(axis='y', alpha=0.3)

# Add percentage labels
for idx, (length, count) in enumerate(test_seq_counts.items()):
    pct = 100 * count / len(test)
    axes[1].text(idx, count + 50, f'{pct:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('sequence_length_distribution.png', dpi=300, bbox_inches='tight')
plt.show()



fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Violin plot of signal-to-noise ratio
axes[0].violinplot([train['signal_to_noise']], positions=[0], showmeans=True, showmedians=True)
axes[0].set_title('Signal-to-Noise Distribution', fontweight='bold')
axes[0].set_ylabel('Signal-to-Noise Ratio')
axes[0].grid(axis='y', alpha=0.3)

# SN_filter pie chart
sn_filter_counts = train['SN_filter'].value_counts()
colors = ['#ff9999', '#66b3ff']
explode = (0.05, 0)

axes[1].pie(sn_filter_counts.values,
            labels=['Passed Filter (1)', 'Failed Filter (0)'],
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            explode=explode,
            shadow=True,
            textprops={'fontsize': 11, 'fontweight': 'bold'})

axes[1].set_title('SN_filter Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig('signal_to_noise_analysis.png', dpi=300, bbox_inches='tight')
plt.show()


print(f"Samples with signal_to_noise greater than 1: {len(train.loc[(train['signal_to_noise'] > 1 )])}")
print(f"Samples with SN_filter = 1: {len(train.loc[(train['SN_filter'] == 1 )])}")
print(f"Samples with signal_to_noise greater than 1, but SN_filter == 0: {len(train.loc[(train['signal_to_noise'] > 1) & (train['SN_filter'] == 0)])}")


sample = train.iloc[0]
Counter(sample['sequence'])


def get_base_composition(sequence):
    """Calculate the percentage of each nucleotide base in the sequence."""
    total = len(sequence)
    return {
        'A': sequence.count('A') / total,
        'G': sequence.count('G') / total,
        'C': sequence.count('C') / total,
        'U': sequence.count('U') / total
    }

# Compute base composition for the training set
train_composition = train['sequence'].apply(get_base_composition)
composition_df = pd.DataFrame(train_composition.tolist())

# Visualize average base composition
fig, ax = plt.subplots(figsize=(10, 6))
composition_df.mean().plot(kind='bar', ax=ax,
                           color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'],
                           edgecolor='black', linewidth=1.5)

ax.set_title('Average RNA Base Composition in the Training Set', fontsize=14, fontweight='bold')
ax.set_xlabel('Nucleotide Base', fontsize=12)
ax.set_ylabel('Average Frequency', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)

# Add percentage labels on top of bars
for i, v in enumerate(composition_df.mean()):
    ax.text(i, v + 0.01, f'{v*100:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('base_composition.png', dpi=300, bbox_inches='tight')
plt.show()



def analyze_structure(structure_str):
    """Analyze RNA secondary structure statistics."""
    paired = structure_str.count('(') + structure_str.count(')')
    unpaired = structure_str.count('.')
    return {
        'paired': paired,
        'unpaired': unpaired,
        'pair_ratio': paired / len(structure_str),
    }

# Analyze RNA secondary structures
structure_analysis = train['structure'].apply(analyze_structure)
structure_df = pd.DataFrame(structure_analysis.tolist())

# Visualize distribution of base-pairing ratio
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(
    structure_df['pair_ratio'],
    bins=30,
    color='mediumpurple',
    edgecolor='black',
    alpha=0.7
)

ax.axvline(
    structure_df['pair_ratio'].mean(),
    color='red',
    linestyle='--',
    linewidth=2,
    label=f'Mean: {structure_df["pair_ratio"].mean():.3f}'
)

ax.set_title('Distribution of Base Pairing Ratio', fontsize=14, fontweight='bold')
ax.set_xlabel('Pairing Ratio', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('structure_pairing_ratio.png', dpi=300, bbox_inches='tight')
plt.show()


# Initialize containers
pairs = []           # to store normalized counts for each sequence
all_partners = []    # to store pairing partners for each nucleotide

# Iterate over all RNA sequences in the training set
for j in range(len(train)):
    partners = [-1 for _ in range(130)]  # initialize partners array (-1 = unpaired)
    pairs_dict = {}                       # dictionary to count base pairs
    queue = []                            # stack to track opening brackets '('
    
    structure = train.iloc[j]['structure']
    sequence = train.iloc[j]['sequence']
    
    # Identify base-pair positions
    for i, char in enumerate(structure):
        if char == '(':
            queue.append(i)
        elif char == ')':
            first = queue.pop()
            
            # Count the pair in the dictionary
            pair = (sequence[first], sequence[i])
            pairs_dict[pair] = pairs_dict.get(pair, 0) + 1
            
            # Store pairing partners
            partners[first] = i
            partners[i] = first
    
    all_partners.append(partners)
    
    # Calculate normalized frequencies for selected base-pair types
    pairs_num = sum(pairs_dict.values())
    pairs_unique = [('U', 'G'), ('C', 'G'), ('U', 'A'), ('G', 'C'), ('A', 'U'), ('G', 'U')]
    normalized_counts = [pairs_dict.get(item, 0) / pairs_num for item in pairs_unique]
    pairs.append(normalized_counts)

# Convert results to a DataFrame
pairs_df = pd.DataFrame(pairs, columns=['U-G', 'C-G', 'U-A', 'G-C', 'A-U', 'G-U'])
pairs_df.head()


# Sum all sequences to get total counts per pair type
total_counts = pairs_df.sum()

# Create bar plot
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(total_counts.index, total_counts.values, color=['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c'],
              edgecolor='black', linewidth=1.2)

# Add count labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 50, f'{int(height)}', ha='center', fontweight='bold')

# Titles and labels
ax.set_title('Total Base-Pair Counts Across All RNA Sequences', fontsize=14, fontweight='bold')
ax.set_xlabel('Base-Pair Type', fontsize=12)
ax.set_ylabel('Total Count', fontsize=12)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('base_pair_distribution.png', dpi=300, bbox_inches='tight')
plt.show()


def count_loop_types(loop_str):
    """Count occurrences of each RNA loop type in the sequence."""
    return {
        'S': loop_str.count('S'),  # Stem
        'M': loop_str.count('M'),  # Multiloop
        'I': loop_str.count('I'),  # Internal loop
        'B': loop_str.count('B'),  # Bulge
        'H': loop_str.count('H'),  # Hairpin
        'E': loop_str.count('E'),  # Dangling end
        'X': loop_str.count('X')   # External loop
    }

# Count loop types for all sequences
loop_counts = train['predicted_loop_type'].apply(count_loop_types)
loop_df = pd.DataFrame(loop_counts.tolist())

# Visualize total counts of each loop type
fig, ax = plt.subplots(figsize=(12, 6))
loop_totals = loop_df.sum().sort_values(ascending=False)
colors_loop = plt.cm.Set3(np.linspace(0, 1, len(loop_totals)))

bars = ax.bar(
    range(len(loop_totals)),
    loop_totals.values,
    color=colors_loop,
    edgecolor='black',
    linewidth=1.5
)

ax.set_xticks(range(len(loop_totals)))
ax.set_xticklabels(loop_totals.index, fontsize=11, fontweight='bold')
ax.set_title('Total Count of Loop Types Across All Sequences', fontsize=14, fontweight='bold')
ax.set_xlabel('Loop Type', fontsize=12)
ax.set_ylabel('Total Count', fontsize=12)
ax.grid(axis='y', alpha=0.3)

# Add count labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.,
        height,
        f'{int(height)}',
        ha='center',
        va='bottom',
        fontweight='bold'
    )

plt.tight_layout()
plt.savefig('loop_type_distribution.png', dpi=300, bbox_inches='tight')
plt.show()



DATA_DIR = Path("../input/stanford-covid-vaccine/")
BPPS_DIR = DATA_DIR / "bpps"
bppm_paths = list(BPPS_DIR.glob("*.npy"))
len(train) + len(test) == len(bppm_paths)


def get_bppm(id_):
    return np.load(BPPS_DIR / f"{id_}.npy")


def draw_structure(structure: str):
    pm = np.zeros((len(structure), len(structure)))
    start_token_indices = []
    for i, token in enumerate(structure):
        if token == "(":
            start_token_indices.append(i)
        elif token == ")":
            j = start_token_indices.pop()
            pm[i, j] = 1.0
            pm[j, i] = 1.0
    return pm


def plot_structures(bppm: np.ndarray, pm: np.ndarray, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Choose same colormap
    cmap = 'viridis'
    
    # BPPM: float values [0,1]
    im0 = axes[0].imshow(bppm, cmap=cmap, vmin=0, vmax=1, origin='lower')
    axes[0].set_title("BPPM")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Structure: mostly 0/1, scale to [0,1] to match BPPM
    im1 = axes[1].imshow(pm, cmap=cmap, vmin=0, vmax=1, origin='lower')
    axes[1].set_title("Structure")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    if axes is None:
        plt.show()


n_samples = 5
fig, all_axes = plt.subplots(n_samples, 2, figsize=(12, n_samples * 4))

# Ensure all_axes is 2D array
if n_samples == 1:
    all_axes = all_axes.reshape(1, 2)

for idx in range(n_samples):
    sample = train.loc[idx]
    bppm = get_bppm(sample.id)
    pm = draw_structure(sample.structure)
    
    # Pass the corresponding row of axes
    plot_structures(bppm, pm, axes=all_axes[idx])
    all_axes[idx, 0].set_title(f"Sample {idx}: BPPM", fontsize=10, fontweight='bold')
    all_axes[idx, 1].set_title(f"Sample {idx}: Structure", fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("combined_structures.png", dpi=300, bbox_inches='tight')
plt.show()


def visualize_graph(bppm: np.ndarray, sequence: str, threshold=0.1):
    indices = np.where(bppm > threshold)
    edges = list(zip(indices[0], indices[1], bppm[indices]))
    
    g = graphviz.Graph(format="png")
    for from_, to, coef in edges:
        if from_ > to:
            g.edge(sequence[from_] + f"({from_})",
                   sequence[to] + f"({to})",
                   label=f"{coef:.2f}",
                   penwidth=f"{int(max(1, abs(coef * 20)))}")
    g.render("./graph")
    return g


idx = 0
sample = train.loc[idx]

bppm = get_bppm(sample.id)
visualize_graph(bppm, sample.sequence, threshold=0.05)


train['reactivity'].head()


# Calculate mean per target for baseline
train['mean_reactivity'] = train['reactivity'].apply(lambda x: np.mean(x))
train['mean_deg_Mg_pH10'] = train['deg_Mg_pH10'].apply(lambda x: np.mean(x))
train['mean_deg_Mg_50C'] = train['deg_Mg_50C'].apply(lambda x: np.mean(x))


# Visualize distributions
fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=False)
color_pal = sns.color_palette("Set2", 5)

train['mean_reactivity'].plot(kind='hist', bins=50, ax=axs[0], color=color_pal[0], title='Distribution of Mean Reactivity')
train['mean_deg_Mg_pH10'].plot(kind='hist', bins=50, ax=axs[1], color=color_pal[1], title='Distribution of Mean deg_Mg_pH10')
train['mean_deg_Mg_50C'].plot(kind='hist', bins=50, ax=axs[2], color=color_pal[2], title='Distribution of Mean deg_Mg_50C')
plt.tight_layout()
plt.show()


# Split the 68 Reactivity values each into it's own column
for n in range(68):
    train[f'reactivity_{n}'] = train['reactivity'].apply(lambda x: x[n])
    
REACTIVITY_COLS = [r for r in train.columns if 'reactivity_' in r and 'error' not in r]

ax = train.set_index('id')[REACTIVITY_COLS] \
    .T \
    .plot(color='black',
          alpha=0.01,
          ylim=(-0.5, 5),
          title='reactivity of training set',
          figsize=(15, 5))
ax.get_legend().remove()


for n in range(68):
    train[f'deg_Mg_pH10_{n}'] = train['deg_Mg_pH10'].apply(lambda x: x[n])
    
DEG_MG_PH10_COLS = [r for r in train.columns if 'deg_Mg_pH10_' in r and 'error' not in r]

ax = train.set_index('id')[DEG_MG_PH10_COLS] \
    .T \
    .plot(color='c',
          alpha=0.01,
          ylim=(-0.5, 5),
          title='Deg Mg Ph10 of training set',
          figsize=(15, 5))
ax.get_legend().remove()


for n in range(68):
    train[f'deg_Mg_50C_{n}'] = train['deg_Mg_50C'].apply(lambda x: x[n])
    
DEG_MG_50C_COLS = [r for r in train.columns if 'deg_Mg_50C_' in r and 'error' not in r]

ax = train.set_index('id')[DEG_MG_50C_COLS] \
    .T \
    .plot(color='m',
          alpha=0.2,
          ylim=(-2, 7),
          title='Deg Mg 50C of training set',
          figsize=(15, 5)
         )
ax.get_legend().remove()


sns.pairplot(data=train,
             vars=['mean_reactivity',
                   'mean_deg_Mg_pH10',
                    'mean_deg_Mg_50C'],
            hue='SN_filter')
plt.show()


# https://www.kaggle.com/code/iamleonie/openvaccine-eda-feature-engineering-with-forgi#New-Features:-Stems,-Interior-Loops,-Hairpin-Loops,-etc.


print("Checking for missing values...")
print("\nTrain set:")
print(train.isnull().sum())
print("\nTest set:")
print(test.isnull().sum())
# No missing values in this dataset, but good practice to check


# Filter based on signal-to-noise ratio
signal_to_noise_threshold = 1.0
high_quality_indices = train[train['signal_to_noise'] > signal_to_noise_threshold].index
SN_filter_indices = train[train['SN_filter'] == 1].index

print(f"Total training samples: {len(train)}")
print(f"High quality samples (S/N > {signal_to_noise_threshold}): {len(high_quality_indices)}")
print(f"Samples passing SN_filter: {len(SN_filter_indices)}")
print(f"Overlap: {len(set(high_quality_indices) & set(SN_filter_indices))}")


def read_bpps_sum(df, bpps_dir='../input/stanford-covid-vaccine/bpps/'):
    bpps_arr = []
    for mol_id in df['id'].tolist():
        bpps = np.load(f"{bpps_dir}{mol_id}.npy")
        bpps_arr.append(bpps.sum(axis=1))
    return np.array(bpps_arr)

def read_bpps_max(df, bpps_dir='../input/stanford-covid-vaccine/bpps/'):
    bpps_arr = []
    for mol_id in df['id'].tolist():
        bpps = np.load(f"{bpps_dir}{mol_id}.npy")
        bpps_arr.append(bpps.max(axis=1))
    return np.array(bpps_arr)

def read_bpps_nb(df, bpps_dir='../input/stanford-covid-vaccine/bpps/'):
    bpps_nb_mean = 0.077522
    bpps_nb_std = 0.08914
    bpps_arr = []
    for mol_id in df['id'].tolist():
        bpps = np.load(f"{bpps_dir}{mol_id}.npy")
        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]  # fraction of nucleotides that form at least one pair
        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std  # normalize
        bpps_arr.append(bpps_nb)
    return np.array(bpps_arr)

print("Extracting BPPS features...")
train['bpps_sum'] = list(read_bpps_sum(train))
test['bpps_sum'] = list(read_bpps_sum(test))
train['bpps_max'] = list(read_bpps_max(train))
test['bpps_max'] = list(read_bpps_max(test))
train['bpps_nb'] = list(read_bpps_nb(train))
test['bpps_nb'] = list(read_bpps_nb(test))

print("BPPS feature extraction completed!")


def normalize_errors(df, error_cols):
    for col in error_cols:
        df[f'{col}_normalized'] = df[col].apply(
        lambda x: np.log(1 + 1.0 / np.array(x)) / 2.25
        )
    return df
error_cols = ['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10','deg_error_Mg_50C', 'deg_error_50C']
train = normalize_errors(train, error_cols)
print("Error normalization completed!")


def create_base_features(df):
    """Create features based on nucleotide composition"""
    print("Creating base composition features...")
    bases = ['A', 'G', 'C', 'U']
    for base in bases:
        df[f'{base}_percent'] = df['sequence'].apply(lambda x: x.count(base) / len(x))
    return df

train = create_base_features(train)
test = create_base_features(test)


def create_structure_features(df):
    """Create features from RNA secondary structure"""
    print("Creating structure features...")

    def get_pairing_partners(structure):
        """Find pairing partners for each position"""
        partners = [-1] * len(structure)
        stack = []
        for i, char in enumerate(structure):
            if char == '(':
                stack.append(i)
            elif char == ')' and stack:
                j = stack.pop()
                partners[i] = j
                partners[j] = i
        return partners

    df['partners'] = df['structure'].apply(get_pairing_partners)

    # Pairing ratio
    df['pair_ratio'] = df['structure'].apply(lambda x: (x.count('(') + x.count(')')) / len(x))
    # Unpaired ratio
    df['unpaired_ratio'] = df['structure'].apply(lambda x: x.count('.') / len(x))

    return df

train = create_structure_features(train)
test = create_structure_features(test)


def create_loop_features(df):
    """Create features from predicted loop types"""
    print("Creating loop type features...")
    loop_types = ['E', 'S', 'H', 'B', 'X', 'I', 'M']

    for loop_type in loop_types:
        df[f'loop_{loop_type}_percent'] = df['predicted_loop_type'].apply(
            lambda x: x.count(loop_type) / len(x)
        )
    return df

train = create_loop_features(train)
test = create_loop_features(test)


def create_positional_features(df):
    """Create position-based features for each nucleotide"""
    print("Creating positional features...")
    position_data = []

    for idx, row in df.iterrows():
        seq_len = row['seq_scored']
        sequence = row['sequence']
        structure = row['structure']
        loop_type = row['predicted_loop_type']
        partners = row['partners']

        for pos in range(seq_len):
            features = {
                'id': row['id'],
                'position': pos,
                'relative_position': pos / seq_len,
                'base': sequence[pos],
                'structure_char': structure[pos],
                'loop_type': loop_type[pos],
                'is_paired': 1 if structure[pos] in '()' else 0,
                'partner_distance': abs(partners[pos] - pos) if partners[pos] != -1 else 0,
                'partner_base': sequence[partners[pos]] if partners[pos] != -1 else 'None',
                'codon_position': pos % 3,
                'prev_base': sequence[pos-1] if pos > 0 else 'Start',
                'next_base': sequence[pos+1] if pos < len(sequence)-1 else 'End',
                'bpps_sum_pos': row['bpps_sum'][pos],
                'bpps_max_pos': row['bpps_max'][pos],
                'bpps_nb_pos': row['bpps_nb'][pos]
            }
            position_data.append(features)

    return pd.DataFrame(position_data)


# Create positional dataframes
print("\nCreating train positional features...")
train_pos = create_positional_features(train)
print("Creating test positional features...")
test_pos = create_positional_features(test)
print(f"\nTrain positional shape: {train_pos.shape}")
print(f"Test positional shape: {test_pos.shape}")


def encode_categorical_features(df):
    """One-hot encode categorical features"""
    print("Encoding categorical features...")

    # One-hot encode base
    base_dummies = pd.get_dummies(df['base'], prefix='base')
    # One-hot encode structure character
    struct_dummies = pd.get_dummies(df['structure_char'], prefix='struct')
    # One-hot encode loop type
    loop_dummies = pd.get_dummies(df['loop_type'], prefix='loop')
    # One-hot encode partner base
    partner_dummies = pd.get_dummies(df['partner_base'], prefix='partner')
    # One-hot encode codon position
    codon_dummies = pd.get_dummies(df['codon_position'], prefix='codon_pos')

    # Concatenate all features
    df_encoded = pd.concat([
        df[['id', 'position', 'relative_position', 'is_paired', 'partner_distance',
            'bpps_sum_pos', 'bpps_max_pos', 'bpps_nb_pos']],
        base_dummies,
        struct_dummies,
        loop_dummies,
        partner_dummies,
        codon_dummies
    ], axis=1)

    return df_encoded

train_encoded = encode_categorical_features(train_pos)
test_encoded = encode_categorical_features(test_pos)
print(f"\nEncoded train shape: {train_encoded.shape}")
print(f"Encoded test shape: {test_encoded.shape}")



def extract_targets(train_df, train_encoded_df):
    """Extract target values for each position"""
    print("Extracting target values...")
    targets_data = []

    for idx, row in train_df.iterrows():
        mol_id = row['id']
        seq_scored = row['seq_scored']

        for pos in range(seq_scored):
            targets = {
                'id': mol_id,
                'position': pos,
                'reactivity': row['reactivity'][pos],
                'deg_Mg_pH10': row['deg_Mg_pH10'][pos],
                'deg_pH10': row['deg_pH10'][pos],
                'deg_Mg_50C': row['deg_Mg_50C'][pos],
                'deg_50C': row['deg_50C'][pos],
                'reactivity_error': row['reactivity_error'][pos],
                'deg_error_Mg_pH10': row['deg_error_Mg_pH10'][pos],
                'deg_error_pH10': row['deg_error_pH10'][pos],
                'deg_error_Mg_50C': row['deg_error_Mg_50C'][pos],
                'deg_error_50C': row['deg_error_50C'][pos],
            }
            targets_data.append(targets)

    targets_df = pd.DataFrame(targets_data)

    # Merge with encoded features
    train_final = train_encoded_df.merge(targets_df, on=['id', 'position'], how='left')
    return train_final

train_final = extract_targets(train, train_encoded)
print(f"\nFinal training data shape: {train_final.shape}")
print(f"Features: {train_final.shape[1] - 10}")  # Subtract target columns


feature_cols = [col for col in train_final.columns
                if col not in ['id', 'position'] +
                ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C',
                 'reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10',
                 'deg_error_Mg_50C', 'deg_error_50C']]

print(f"\n{'='*80}")
print(f"FEATURE ENGINEERING SUMMARY")
print(f"{'='*80}")
print(f"Total features created: {len(feature_cols)}")
print(f"\nFeature categories:")
print(f"  - Positional features: 6")
print(f"  - BPPS features: 3")
print(f"  - One-hot encoded bases: {len([c for c in feature_cols if 'base_' in c])}")
print(f"  - One-hot encoded structure: {len([c for c in feature_cols if 'struct_' in c])}")
print(f"  - One-hot encoded loop types: {len([c for c in feature_cols if 'loop_' in c])}")
print(f"  - One-hot encoded partners: {len([c for c in feature_cols if 'partner_' in c])}")
print(f"  - Codon position: {len([c for c in feature_cols if 'codon_pos_' in c])}")
print(f"{'='*80}\n")


def mcrmse_metric(y_true, y_pred):
    """
    Compute MCRMSE (Mean Columnwise Root Mean Squared Error) 
    """
    # Use only the scored positions
    y_true_scored = y_true[:, :68]
    y_pred_scored = y_pred[:, :68]
    
    # RMSE per column (per target)
    colwise_rmse = np.sqrt(np.mean((y_true_scored - y_pred_scored)**2, axis=0))
    
    # Average RMSE across all target columns
    return np.mean(colwise_rmse)

class MCRMSELoss(nn.Module):
    def __init__(self, seq_len_target=68):
        super(MCRMSELoss, self).__init__()
        self.seq_len_target = seq_len_target

    def forward(self, y_pred, y_true):
        # Chỉ lấy các vị trí được chấm điểm
        y_true_scored = y_true[:, :self.seq_len_target]
        y_pred_scored = y_pred[:, :self.seq_len_target]

        # Tính toán MCRMSE
        # torch.mean(..., dim=0) -> tính trung bình theo chiều mẫu
        # torch.mean(..., dim=1) -> tính trung bình theo chiều vị trí
        loss = torch.mean(torch.sqrt(torch.mean((y_true_scored - y_pred_scored)**2, dim=1)))
        return loss

def calculate_metrics(y_true, y_pred, target_names):
    """
    Compute a comprehensive set of regression metrics per target:
    - RMSE
    - MAE
    - R²
    - NSE (Nash–Sutcliffe Efficiency)

    Also returns averaged metrics across all targets.
    """
    metrics = {}
    
    for i, target in enumerate(target_names):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        
        # Basic metrics
        metrics[f'{target}_rmse'] = np.sqrt(mean_squared_error(y_t, y_p))
        metrics[f'{target}_mae'] = mean_absolute_error(y_t, y_p)
        metrics[f'{target}_r2'] = r2_score(y_t, y_p)
        
        # NSE metric
        numerator = np.sum((y_t - y_p) ** 2)
        denominator = np.sum((y_t - np.mean(y_t)) ** 2)
        metrics[f'{target}_nse'] = 1 - (numerator / denominator)
    
    # Aggregate metrics across targets
    metrics['overall_rmse'] = np.mean([metrics[f'{t}_rmse'] for t in target_names])
    metrics['overall_mae'] = np.mean([metrics[f'{t}_mae'] for t in target_names])
    metrics['overall_r2'] = np.mean([metrics[f'{t}_r2'] for t in target_names])
    metrics['overall_nse'] = np.mean([metrics[f'{t}_nse'] for t in target_names])
    
    return metrics


class Covid19Dataset(Dataset):
    def __init__(self,X,bpps,mat,seq_length,scored_length,label=None,label_error=None,signal_to_noise=None,SN_filter_mask=None):
        self.X = X.astype(np.int)
#         self.bpps = np.log(bpps + 1e-8).astype(np.float32)
        self.bpps = bpps.astype(np.float32)
        
#         self.bpps = np.log(bpps + 1e-8)
#         self.bpps = np.concatenate([bpps.reshape([-1,130,130,1]),mat.reshape([-1,130,130,1])],axis=-1).astype(np.float32)
        if label is not None:
            self.label = label.astype(np.float32)
            self.signal_to_noise = signal_to_noise.astype(np.float32)
            self.label_error=label_error.astype(np.float32)
            self.SN_filter_mask = SN_filter_mask
        else:
            self.label = None
        self.mask = np.zeros([len(X),130],dtype=bool)
        for i in range(len(seq_length)):
            if seq_length[i] < 130:
                self.mask[i,seq_length[i]:] = True
        self.scored_mask = np.ones([len(X),130],dtype=bool)
        for i in range(len(scored_length)):
            if scored_length[i] < 130:
                self.scored_mask[i,scored_length[i]:] = False
        self.seq_length = seq_length


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        N = self.seq_length[idx]
        X = self.X[idx,:N]
        bpps = self.bpps[idx,:,:N,:N]
        mask = self.mask[idx,:N]
        scored_mask = self.scored_mask[idx,:N]
        if self.label is not None:
            label = self.label[idx,:N]
            label_error= self.label_error[idx,:N]
            signal_to_noise = self.signal_to_noise[idx]
            SN_filter_mask = self.SN_filter_mask[idx]
            return X,bpps,mask,scored_mask,label,label_error,signal_to_noise,SN_filter_mask
        else:
            return X,bpps,mask,scored_mask


# Tách các đặc trưng và mục tiêu
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']
feature_cols = [col for col in train_final.columns
                if col not in ['id', 'position'] + target_cols +
                ['deg_pH10', 'deg_50C'] +
                [c for c in train_final.columns if 'error' in c]]

# Xử lý các cột bị thiếu trong tập test
missing_cols = set(feature_cols) - set(test_encoded.columns)
if missing_cols:
    print(f"Cảnh báo: {len(missing_cols)} đặc trưng bị thiếu trong tập test. Thêm các cột bằng 0.")
    for col in missing_cols:
        test_encoded[col] = 0

def prepare_data_for_sequence_model(feature_df, target_df=None):
    """
    Nhóm dữ liệu theo ID để tạo thành các chuỗi và thực hiện padding.
    Trả về các tensor cho đặc trưng, mục tiêu (nếu có) và mặt nạ chú ý.
    """
    features_by_id = feature_df.groupby('id')
    
    feature_sequences = []
    target_sequences = []
    
    for mol_id, group in features_by_id:
        # Lấy đặc trưng và chuyển thành tensor
        feature_seq = torch.tensor(group[feature_cols].values, dtype=torch.float32)
        feature_sequences.append(feature_seq)
        
        # Lấy mục tiêu nếu có
        if target_df is not None:
            target_seq = torch.tensor(target_df.loc[target_df['id'] == mol_id, target_cols].values, dtype=torch.float32)
            target_sequences.append(target_seq)

    # Padding các chuỗi để có cùng độ dài
    # batch_first=True -> (batch_size, seq_len, features)
    padded_features = pad_sequence(feature_sequences, batch_first=True, padding_value=0.0)
    
    # Tạo mặt nạ chú ý (attention mask)
    # True cho các vị trí được đệm (padded), False cho các vị trí thực
    attention_mask = (padded_features.sum(dim=-1) == 0)
    
    if target_df is not None:
        padded_targets = pad_sequence(target_sequences, batch_first=True, padding_value=0.0)
        return padded_features, padded_targets, attention_mask
    else:
        return padded_features, attention_mask

# Chuẩn bị dữ liệu huấn luyện
X_train_seq, y_train_seq, train_masks = prepare_data_for_sequence_model(train_final[train_final['id'].isin(train['id'])], train_final)

# Chuẩn bị dữ liệu kiểm tra
X_test_seq, test_masks = prepare_data_for_sequence_model(test_encoded)

print(f"Kích thước X_train (chuỗi): {X_train_seq.shape}")
print(f"Kích thước y_train (chuỗi): {y_train_seq.shape}")
print(f"Kích thước Mặt nạ train:     {train_masks.shape}")
print(f"Kích thước X_test (chuỗi):  {X_test_seq.shape}")
print(f"Kích thước Mặt nạ test:      {test_masks.shape}")

# Chuẩn hóa đặc trưng
# Chúng ta sẽ chuẩn hóa trên dữ liệu không phải chuỗi và áp dụng lại
scaler = StandardScaler()
num_features = X_train_seq.shape[-1]
# Reshape to 2D for scaler -> (num_samples * seq_len, features)
X_train_flat = X_train_seq.reshape(-1, num_features)
# Fit trên dữ liệu không đệm
scaler.fit(X_train_flat[X_train_flat.sum(axis=1) != 0])

# Transform dữ liệu
X_train_seq_scaled = scaler.transform(X_train_flat).reshape(X_train_seq.shape)
X_test_flat = X_test_seq.reshape(-1, num_features)
X_test_seq_scaled = scaler.transform(X_test_flat).reshape(X_test_seq.shape)

# Chuyển đổi lại sang Tensor
X_train_tensor = torch.tensor(X_train_seq_scaled, dtype=torch.float32)
y_train_tensor = y_train_seq
X_test_tensor = torch.tensor(X_test_seq_scaled, dtype=torch.float32)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq, d_model)
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :].unsqueeze(0)      # -> (1, seq, d_model)
        x = x + pe
        return self.dropout(x)


class HybridModel(nn.Module):
    def __init__(self, input_dim, output_dim=3, d_model=128,
                 nhead=8, num_encoder_layers=3,
                 dim_feedforward=512, dropout=0.2):

        super(HybridModel, self).__init__()
        self.d_model = d_model

        # 1. Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # 2. CNN block
        self.cnn_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 3. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 4. Transformer encoder (PyTorch cũ không có batch_first)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # 5. Regression head
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, src, src_key_padding_mask=None):
        # src: (batch, seq, input_dim)

        # 1. Linear projection
        x = self.input_proj(src)  # (batch, seq, d_model)

        # 2. CNN block
        x_cnn = x.permute(0, 2, 1)      # -> (batch, d_model, seq)
        x_cnn = self.cnn_block(x_cnn)
        x = x_cnn.permute(0, 2, 1)      # -> (batch, seq, d_model)

        # 3. Positional encoding
        x = self.pos_encoder(x)

        # 4. Transformer encoder (no batch_first!)
        x = x.transpose(0, 1)           # -> (seq, batch, d_model)

        # key_padding_mask must remain (batch, seq)
        x = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )

        x = x.transpose(0, 1)           # -> (batch, seq, d_model)

        # 5. Output
        return self.output_layer(x)


input_dim = X_train_tensor.shape[2]
model = HybridModel(input_dim=input_dim)
print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device using: {device}")


n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Lưu trữ dự đoán và chỉ số
# Lưu ý: Chúng ta cần reshape lại các dự đoán sau này
oof_predictions = np.zeros(y_train.shape)
test_predictions_flat = np.zeros((X_test.shape[0], 3)) # Dữ liệu test phẳng
fold_metrics = []

print(f"\n{'='*80}")
print(f"HUẤN LUYỆN CROSS-VALIDATION VỚI MÔ HÌNH TRANSFORMER-CNN")
print(f"{'='*80}\n")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_tensor)):
    print(f"\n{'='*80}")
    print(f"FOLD {fold + 1}/{n_folds}")
    print(f"{'='*80}")
    
    # Chia dữ liệu theo chuỗi
    X_tr, X_val = X_train_tensor[train_idx], X_train_tensor[val_idx]
    y_tr, y_val = y_train_tensor[train_idx], y_train_tensor[val_idx]
    mask_tr, mask_val = train_masks[train_idx], train_masks[val_idx]
    
    # Tạo DataLoaders
    train_dataset = TensorDataset(X_tr, y_tr, mask_tr)
    val_dataset = TensorDataset(X_val, y_val, mask_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = HybridModel(input_dim=input_dim).to(device)
    criterion = MCRMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15

    for epoch in range(100):
        model.train()
        total_train_loss = 0
        for batch_X, batch_y, batch_mask in train_loader:
            batch_X, batch_y, batch_mask = batch_X.to(device), batch_y.to(device), batch_mask.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X, src_key_padding_mask=batch_mask)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y, batch_mask in val_loader:
                batch_X, batch_y, batch_mask = batch_X.to(device), batch_y.to(device), batch_mask.to(device)
                outputs = model(batch_X, src_key_padding_mask=batch_mask)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:03d}: Train Loss: {avg_train_loss:.5f}, Val Loss: {avg_val_loss:.5f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_hybrid_model_fold_{fold+1}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping tại epoch {epoch+1}")
                break
                
    # Tải trọng số tốt nhất và dự đoán
    model.load_state_dict(torch.load(f'best_hybrid_model_fold_{fold+1}.pth'))
    model.eval()
    
    # Dự đoán OOF và chuyển về dạng phẳng
    oof_preds_fold = []
    with torch.no_grad():
        for batch_X, _, batch_mask in val_loader:
            batch_X, batch_mask = batch_X.to(device), batch_mask.to(device)
            preds = model(batch_X, src_key_padding_mask=batch_mask).cpu().numpy()
            
            # Loại bỏ padding
            for i in range(preds.shape[0]):
                seq_len = int(torch.sum(~batch_mask[i]).item())
                oof_preds_fold.append(preds[i, :seq_len, :])
    
    oof_preds_fold_flat = np.concatenate(oof_preds_fold, axis=0)
    
    # Lấy các chỉ số đúng cho dữ liệu OOF phẳng
    # Lấy ra các chỉ số của các mẫu trong fold hiện tại từ `train_final`
    val_ids = train.iloc[val_idx]['id'].unique()
    oof_indices_flat = train_final[train_final['id'].isin(val_ids)].index
    oof_predictions[oof_indices_flat] = oof_preds_fold_flat
    
    # Dự đoán trên tập Test
    test_loader = DataLoader(TensorDataset(X_test_tensor, test_masks), batch_size=32, shuffle=False)
    test_preds_fold = []
    with torch.no_grad():
        for batch_X, batch_mask in test_loader:
            batch_X, batch_mask = batch_X.to(device), batch_mask.to(device)
            preds = model(batch_X, src_key_padding_mask=batch_mask).cpu().numpy()
            for i in range(preds.shape[0]):
                seq_len = int(torch.sum(~batch_mask[i]).item())
                test_preds_fold.append(preds[i, :seq_len, :])

    test_predictions_flat += np.concatenate(test_preds_fold, axis=0) / n_folds
    
    # Tính toán chỉ số cho fold (trên dữ liệu phẳng)
    y_val_flat = train_final.iloc[oof_indices_flat][target_cols].values
    fold_metric = calculate_metrics(y_val_flat, oof_preds_fold_flat, target_cols)
    fold_metrics.append(fold_metric)
    
    print(f"\nKết quả Fold {fold + 1}:")
    print(f"  RMSE: {fold_metric['overall_rmse']:.5f}")
    
# Tính toán chỉ số CV tổng thể
cv_metrics = calculate_metrics(y_train, oof_predictions, target_cols)
print(f"\n{'='*80}\nTÓM TẮT CROSS-VALIDATION\n{'='*80}")
print(f"Chỉ số CV Tổng thể:")
print(f"  RMSE: {cv_metrics['overall_rmse']:.5f} (±{np.std([m['overall_rmse'] for m in fold_metrics]):.5f})")


# Convert history to DataFrame
history_df = pd.DataFrame(history_dict)

# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Loss curves
for fold in range(1, n_folds + 1):
    fold_data = history_df[history_df['fold'] == fold]
    axes[0, 0].plot(fold_data['epoch'], fold_data['train_loss'], 
                    alpha=0.5, label=f'Fold {fold}')
    axes[0, 1].plot(fold_data['epoch'], fold_data['val_loss'], 
                    alpha=0.5, label=f'Fold {fold}')

axes[0, 0].set_title('Training Loss by Fold', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (MCRMSE)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].set_title('Validation Loss by Fold', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss (MCRMSE)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# MAE curves
for fold in range(1, n_folds + 1):
    fold_data = history_df[history_df['fold'] == fold]
    axes[1, 0].plot(fold_data['epoch'], fold_data['train_mae'], 
                    alpha=0.5, label=f'Fold {fold}')
    axes[1, 1].plot(fold_data['epoch'], fold_data['val_mae'], 
                    alpha=0.5, label=f'Fold {fold}')

axes[1, 0].set_title('Training MAE by Fold', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('MAE')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

axes[1, 1].set_title('Validation MAE by Fold', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('MAE')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()


# Prepare data for visualization
metrics_comparison = pd.DataFrame(fold_metrics)
metrics_comparison['fold'] = range(1, n_folds + 1)

# Plot metrics by fold
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics_to_plot = ['overall_rmse', 'overall_mae', 'overall_r2', 'overall_nse']
titles = ['RMSE by Fold', 'MAE by Fold', 'R² by Fold', 'NSE by Fold']
colors = ['coral', 'lightseagreen', 'gold', 'mediumpurple']

for idx, (metric, title, color) in enumerate(zip(metrics_to_plot, titles, colors)):
    ax = axes[idx // 2, idx % 2]
    
    bars = ax.bar(metrics_comparison['fold'], metrics_comparison[metric], 
                   color=color, edgecolor='black', linewidth=1.5, alpha=0.7)
    
    # Add mean line
    mean_val = metrics_comparison[metric].mean()
    ax.axhline(mean_val, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_val:.5f}')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.5f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel(metric.upper().replace('_', ' '), fontsize=12)
    ax.set_xticks(range(1, n_folds + 1))
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('fold_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# Create target-specific metrics visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for idx, metric_type in enumerate(['rmse', 'mae', 'r2', 'nse']):
    ax = axes[idx // 2, idx % 2]
    
    # Prepare data
    target_metrics = []
    for target in target_cols:
        values = [m[f'{target}_{metric_type}'] for m in fold_metrics]
        target_metrics.append({
            'target': target,
            'mean': np.mean(values),
            'std': np.std(values)
        })
    
    target_metrics_df = pd.DataFrame(target_metrics)
    
    # Plot
    x = np.arange(len(target_cols))
    bars = ax.bar(x, target_metrics_df['mean'], 
                   yerr=target_metrics_df['std'],
                   capsize=5, edgecolor='black', linewidth=1.5,
                   color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.7)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.5f}±{target_metrics_df["std"].iloc[i]:.5f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax.set_title(f'{metric_type.upper()} by Target', fontsize=14, fontweight='bold')
    ax.set_xlabel('Target Variable', fontsize=12)
    ax.set_ylabel(metric_type.upper(), fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(target_cols, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('target_specific_metrics.png', dpi=300, bbox_inches='tight')
plt.show()


# Create scatter plots for each target
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, target in enumerate(target_cols):
    ax = axes[idx]
    
    # Get predictions and actuals for this target
    y_true = y_train[:, idx]
    y_pred = oof_predictions[:, idx]
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.3, s=20, color='steelblue', edgecolor='none')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate and display metrics
    r2 = r2_score(y_true, y_pred)
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    
    ax.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse_val:.4f}',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title(f'{target} - Predictions vs Actuals', fontsize=12, fontweight='bold')
    ax.set_xlabel('Actual Values', fontsize=11)
    ax.set_ylabel('Predicted Values', fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('predictions_vs_actuals.png', dpi=300, bbox_inches='tight')
plt.show()


# Residual plots for each target
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, target in enumerate(target_cols):
    # Get predictions and actuals
    y_true = y_train[:, idx]
    y_pred = oof_predictions[:, idx]
    residuals = y_true - y_pred
    
    # Residual scatter plot
    ax1 = axes[0, idx]
    ax1.scatter(y_pred, residuals, alpha=0.3, s=20, color='coral', edgecolor='none')
    ax1.axhline(0, color='red', linestyle='--', linewidth=2)
    ax1.set_title(f'{target} - Residual Plot', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted Values', fontsize=11)
    ax1.set_ylabel('Residuals', fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Residual histogram
    ax2 = axes[1, idx]
    ax2.hist(residuals, bins=50, color='lightseagreen', edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_title(f'{target} - Residual Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Residuals', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.grid(alpha=0.3)
    
    # Add statistics
    mean_residual = residuals.mean()
    std_residual = residuals.std()
    ax2.text(0.98, 0.95, f'μ = {mean_residual:.5f}\nσ = {std_residual:.5f}',
             transform=ax2.transAxes, fontsize=10, fontweight='bold',
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=300, bbox_inches='tight')
plt.show()


# Create comprehensive metrics summary table
summary_data = []

for target in target_cols:
    row = {
        'Target': target,
        'RMSE': f"{cv_metrics[f'{target}_rmse']:.5f} ± {np.std([m[f'{target}_rmse'] for m in fold_metrics]):.5f}",
        'MAE': f"{cv_metrics[f'{target}_mae']:.5f} ± {np.std([m[f'{target}_mae'] for m in fold_metrics]):.5f}",
        'R²': f"{cv_metrics[f'{target}_r2']:.5f} ± {np.std([m[f'{target}_r2'] for m in fold_metrics]):.5f}",
        'NSE': f"{cv_metrics[f'{target}_nse']:.5f} ± {np.std([m[f'{target}_nse'] for m in fold_metrics]):.5f}"
    }
    summary_data.append(row)

# Add overall row
summary_data.append({
    'Target': 'Overall',
    'RMSE': f"{cv_metrics['overall_rmse']:.5f} ± {np.std([m['overall_rmse'] for m in fold_metrics]):.5f}",
    'MAE': f"{cv_metrics['overall_mae']:.5f} ± {np.std([m['overall_mae'] for m in fold_metrics]):.5f}",
    'R²': f"{cv_metrics['overall_r2']:.5f} ± {np.std([m['overall_r2'] for m in fold_metrics]):.5f}",
    'NSE': f"{cv_metrics['overall_nse']:.5f} ± {np.std([m['overall_nse'] for m in fold_metrics]):.5f}"
})

summary_df = pd.DataFrame(summary_data)

# Display as styled table
print(f"\n{'='*100}")
print(f"{'FINAL CROSS-VALIDATION METRICS SUMMARY':^100}")
print(f"{'='*100}\n")
print(summary_df.to_string(index=False))
print(f"\n{'='*100}\n")

# Save to CSV
summary_df.to_csv('cv_metrics_summary.csv', index=False)
print("Metrics summary saved to 'cv_metrics_summary.csv'")


# Prepare submission file
print("\nPreparing submission file...")

# Merge predictions with test IDs
test_encoded['reactivity'] = test_predictions_flat[:, 0]
test_encoded['deg_Mg_pH10'] = test_predictions_flat[:, 1]
test_encoded['deg_Mg_50C'] = test_predictions_flat[:, 2]

# Create submission in required format
submission = sample_sub.copy()

for idx, row in test_encoded.iterrows():
    mol_id = row['id']
    pos = int(row['position'])
    id_seqpos = f"{mol_id}_{pos}"
    
    if id_seqpos in submission['id_seqpos'].values:
        submission.loc[submission['id_seqpos'] == id_seqpos, 'reactivity'] = row['reactivity']
        submission.loc[submission['id_seqpos'] == id_seqpos, 'deg_Mg_pH10'] = row['deg_Mg_pH10']
        submission.loc[submission['id_seqpos'] == id_seqpos, 'deg_Mg_50C'] = row['deg_Mg_50C']

# Fill remaining columns with 0 (not scored)
submission['deg_pH10'] = 0
submission['deg_50C'] = 0

# Save submission
submission_filename = f'submission_hybrid_cv{cv_metrics["overall_rmse"]:.5f}.csv'
submission.to_csv(submission_filename, index=False)
print(f"\nSubmission đã được lưu vào '{submission_filename}'")

# Display sample
print(f"\nSubmission preview:")
print(submission.head(20))
print(f"\nSubmission shape: {submission.shape}")

