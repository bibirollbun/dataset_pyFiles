!pip install biopython
!pip install goatools


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import os
from Bio import SeqIO
from collections import defaultdict
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")


def load_sequences(file_path):
    """Load sequences into a dict {id: sequence_str}."""
    if not os.path.exists(file_path):
        return {}

    seq_dict = {}
    for record in SeqIO.parse(file_path, 'fasta'):
        seq_dict[record.id] = str(record.seq)
    return seq_dict


def analyze_fasta(file_path):
    """Analyze a FASTA file and return statistics."""
    if not os.path.exists(file_path):
        return None

    sequences = list(SeqIO.parse(file_path, 'fasta'))
    lengths = [len(seq.seq) for seq in sequences]

    stats = {
        'num_sequences': len(sequences),
        'total_length': sum(lengths),
        'avg_length': np.mean(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths)
    }

    return stats


# Define file paths
cafa5_train_fasta = '/kaggle/input/cafa-5-protein-function-prediction/Train/train_sequences.fasta'
cafa5_test_fasta = '/kaggle/input/cafa-5-protein-function-prediction/Test (Targets)/testsuperset.fasta'
cafa6_train_fasta = '/kaggle/input/cafa-6-protein-function-prediction/Train/train_sequences.fasta'
cafa6_test_fasta = '/kaggle/input/cafa-6-protein-function-prediction/Test/testsuperset.fasta'

# Label files
cafa5_terms = '/kaggle/input/cafa-5-protein-function-prediction/Train/train_terms.tsv'
cafa6_terms = '/kaggle/input/cafa-6-protein-function-prediction/Train/train_terms.tsv'

# Load sequences
cafa5_train_sequences = load_sequences(cafa5_train_fasta)
cafa5_test_sequences = load_sequences(cafa5_test_fasta)
cafa6_train_sequences = load_sequences(cafa6_train_fasta)
cafa6_test_sequences = load_sequences(cafa6_test_fasta)

# Load labels
cafa5_terms_df = pd.read_csv(cafa5_terms, sep='\t')
cafa6_terms_df = pd.read_csv(cafa6_terms, sep='\t')

# Create label dictionaries
cafa5_labels = defaultdict(set)
for _, row in cafa5_terms_df.iterrows():
    cafa5_labels[row['EntryID']].add(row['term'])

cafa6_labels = defaultdict(set)
for _, row in cafa6_terms_df.iterrows():
    cafa6_labels[row['EntryID']].add(row['term'])


# Create sets of sequences
cafa5_train_seqs = set(cafa5_train_sequences.values())
cafa5_test_seqs = set(cafa5_test_sequences.values())
cafa5_all_seqs = cafa5_train_seqs.union(cafa5_test_seqs)

cafa6_train_seqs = set(cafa6_train_sequences.values())
cafa6_test_seqs = set(cafa6_test_sequences.values())
cafa6_all_seqs = cafa6_train_seqs.union(cafa6_test_seqs)

print("Dataset Sizes:")
print("=" * 50)
print(f"CAFA-5 Train: {len(cafa5_train_sequences)} sequences")
print(f"CAFA-5 Test:  {len(cafa5_test_sequences)} sequences")
print(f"CAFA-6 Train: {len(cafa6_train_sequences)} sequences")
print(f"CAFA-6 Test:  {len(cafa6_test_sequences)} sequences")

print("\n" + "=" * 50)
print("Sequence Overlap Analysis:")
print("=" * 50)

print(f"\nCAFA-5 total unique sequences: {len(cafa5_all_seqs)}")
print(f"CAFA-6 total unique sequences: {len(cafa6_all_seqs)}")

seq_intersection = cafa5_all_seqs.intersection(cafa6_all_seqs)
print(f"Identical sequences in both CAFA-5 and CAFA-6: {len(seq_intersection)}")

# Detailed sequence overlaps
train_train_seq_overlap = cafa5_train_seqs.intersection(cafa6_train_seqs)
train_test_seq_overlap = cafa5_train_seqs.intersection(cafa6_test_seqs)
test_train_seq_overlap = cafa5_test_seqs.intersection(cafa6_train_seqs)
test_test_seq_overlap = cafa5_test_seqs.intersection(cafa6_test_seqs)

print(f"\nDetailed sequence overlaps:")
print(f"CAFA-5 Train ∩ CAFA-6 Train: {len(train_train_seq_overlap)} ({len(train_train_seq_overlap)/len(cafa5_train_seqs)*100:.2f}% of CAFA-5 Train, {len(train_train_seq_overlap)/len(cafa6_train_seqs)*100:.2f}% of CAFA-6 Train)")
print(f"CAFA-5 Train ∩ CAFA-6 Test: {len(train_test_seq_overlap)} ({len(train_test_seq_overlap)/len(cafa5_train_seqs)*100:.2f}% of CAFA-5 Train, {len(train_test_seq_overlap)/len(cafa6_test_seqs)*100:.2f}% of CAFA-6 Test)")
print(f"CAFA-5 Test ∩ CAFA-6 Train: {len(test_train_seq_overlap)} ({len(test_train_seq_overlap)/len(cafa5_test_seqs)*100:.2f}% of CAFA-5 Test, {len(test_train_seq_overlap)/len(cafa6_train_seqs)*100:.2f}% of CAFA-6 Train)")
print(f"CAFA-5 Test ∩ CAFA-6 Test: {len(test_test_seq_overlap)} ({len(test_test_seq_overlap)/len(cafa5_test_seqs)*100:.2f}% of CAFA-5 Test, {len(test_test_seq_overlap)/len(cafa6_test_seqs)*100:.2f}% of CAFA-6 Test)")

# Whole dataset overlaps
print(f"\nWhole dataset overlaps:")
cafa5_train_vs_cafa6_all = cafa5_train_seqs.intersection(cafa6_all_seqs)
cafa5_test_vs_cafa6_all = cafa5_test_seqs.intersection(cafa6_all_seqs)
cafa6_train_vs_cafa5_all = cafa6_train_seqs.intersection(cafa5_all_seqs)
cafa6_test_vs_cafa5_all = cafa6_test_seqs.intersection(cafa5_all_seqs)

print(f"CAFA-5 Train ∩ CAFA-6 (all): {len(cafa5_train_vs_cafa6_all)} ({len(cafa5_train_vs_cafa6_all)/len(cafa5_train_seqs)*100:.2f}% of CAFA-5 Train)")
print(f"CAFA-5 Test ∩ CAFA-6 (all): {len(cafa5_test_vs_cafa6_all)} ({len(cafa5_test_vs_cafa6_all)/len(cafa5_test_seqs)*100:.2f}% of CAFA-5 Test)")
print(f"CAFA-6 Train ∩ CAFA-5 (all): {len(cafa6_train_vs_cafa5_all)} ({len(cafa6_train_vs_cafa5_all)/len(cafa6_train_seqs)*100:.2f}% of CAFA-6 Train)")
print(f"CAFA-6 Test ∩ CAFA-5 (all): {len(cafa6_test_vs_cafa5_all)} ({len(cafa6_test_vs_cafa5_all)/len(cafa6_test_seqs)*100:.2f}% of CAFA-6 Test)")

# Combined train+test overlaps
print(f"\nCombined Train+Test overlaps:")
cafa5_all_vs_cafa6_all = cafa5_all_seqs.intersection(cafa6_all_seqs)
print(f"CAFA-5 (Train+Test) ∩ CAFA-6 (Train+Test): {len(cafa5_all_vs_cafa6_all)} ({len(cafa5_all_vs_cafa6_all)/len(cafa5_all_seqs)*100:.2f}% of CAFA-5 all, {len(cafa5_all_vs_cafa6_all)/len(cafa6_all_seqs)*100:.2f}% of CAFA-6 all)")

# New sequences in CAFA-6
new_in_cafa6 = cafa6_all_seqs - cafa5_all_seqs
print(f"New sequences in CAFA-6 (not in CAFA-5): {len(new_in_cafa6)} ({len(new_in_cafa6)/len(cafa6_all_seqs)*100:.2f}% of CAFA-6)")

# Sequences only in CAFA-5
only_in_cafa5 = cafa5_all_seqs - cafa6_all_seqs
print(f"Sequences only in CAFA-5 (not in CAFA-6): {len(only_in_cafa5)} ({len(only_in_cafa5)/len(cafa5_all_seqs)*100:.2f}% of CAFA-5)")


# Create DataFrame for overlapping sequences with labels
print("\n" + "=" * 50)
print("Creating DataFrame for Overlapping Sequences:")
print("=" * 50)

# Create reverse mapping: sequence -> list of IDs
seq_to_ids_cafa5 = defaultdict(list)
for id_, seq in cafa5_train_sequences.items():
    seq_to_ids_cafa5[seq].append(id_)
for id_, seq in cafa5_test_sequences.items():
    seq_to_ids_cafa5[seq].append(id_)

seq_to_ids_cafa6 = defaultdict(list)
for id_, seq in cafa6_train_sequences.items():
    seq_to_ids_cafa6[seq].append(id_)
for id_, seq in cafa6_test_sequences.items():
    seq_to_ids_cafa6[seq].append(id_)

# Create DataFrame
overlapping_data = []
for seq in tqdm(cafa5_all_vs_cafa6_all):
    cafa5_ids = seq_to_ids_cafa5.get(seq, [])
    cafa6_ids = seq_to_ids_cafa6.get(seq, [])

    # Get labels for these IDs
    cafa5_go_terms = set()
    cafa6_go_terms = set()

    for id_ in cafa5_ids:
        cafa5_go_terms.update(cafa5_labels.get(id_, set()))
    for id_ in cafa6_ids:
        cafa6_go_terms.update(cafa6_labels.get(id_, set()))

    overlapping_data.append({
        'sequence': seq,
        'cafa5_ids': cafa5_ids,
        'cafa6_ids': cafa6_ids,
        'cafa5_go_terms': list(cafa5_go_terms),
        'cafa6_go_terms': list(cafa6_go_terms),
        'num_cafa5_terms': len(cafa5_go_terms),
        'num_cafa6_terms': len(cafa6_go_terms)
    })

df_overlapping = pd.DataFrame(overlapping_data)
print(f"Created DataFrame with {len(df_overlapping)} overlapping sequences")
print(f"Columns: {list(df_overlapping.columns)}")

# Summary statistics
print(f"\nSummary:")
print(f"Total overlapping sequences: {len(df_overlapping)}")
print(f"Average CAFA-5 GO terms per sequence: {df_overlapping['num_cafa5_terms'].mean():.2f}")
print(f"Average CAFA-6 GO terms per sequence: {df_overlapping['num_cafa6_terms'].mean():.2f}")
print(f"Max CAFA-5 GO terms: {df_overlapping['num_cafa5_terms'].max()}")
print(f"Max CAFA-6 GO terms: {df_overlapping['num_cafa6_terms'].max()}")


# ID-based overlaps (as before)
print("\n" + "=" * 50)
print("ID Overlap Analysis:")
print("=" * 50)

cafa5_train_ids = set(cafa5_train_sequences.keys())
cafa5_test_ids = set(cafa5_test_sequences.keys())
cafa5_all_ids = cafa5_train_ids.union(cafa5_test_ids)

cafa6_train_ids = set(cafa6_train_sequences.keys())
cafa6_test_ids = set(cafa6_test_sequences.keys())
cafa6_all_ids = cafa6_train_ids.union(cafa6_test_ids)

print(f"\nCAFA-5 total unique IDs: {len(cafa5_all_ids)}")
print(f"CAFA-6 total unique IDs: {len(cafa6_all_ids)}")

id_intersection = cafa5_all_ids.intersection(cafa6_all_ids)
print(f"IDs present in both CAFA-5 and CAFA-6: {len(id_intersection)}")

# Detailed ID overlaps
id_train_train_overlap = cafa5_train_ids.intersection(cafa6_train_ids)
id_train_test_overlap = cafa5_train_ids.intersection(cafa6_test_ids)
id_test_train_overlap = cafa5_test_ids.intersection(cafa6_train_ids)
id_test_test_overlap = cafa5_test_ids.intersection(cafa6_test_ids)

print(f"\nDetailed ID overlaps:")
print(f"CAFA-5 Train ∩ CAFA-6 Train: {len(id_train_train_overlap)}")
print(f"CAFA-5 Train ∩ CAFA-6 Test: {len(id_train_test_overlap)}")
print(f"CAFA-5 Test ∩ CAFA-6 Train: {len(id_test_train_overlap)}")
print(f"CAFA-5 Test ∩ CAFA-6 Test: {len(id_test_test_overlap)}")

df_overlapping['cafa5_go_terms_len'] = df_overlapping['cafa5_go_terms'].apply(len)
df_overlapping['cafa6_go_terms_len'] = df_overlapping['cafa6_go_terms'].apply(len)
df = df_overlapping.loc[(df_overlapping['cafa5_go_terms_len'] > 0) & (df_overlapping['cafa6_go_terms_len'] > 0)]
df.head()


fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# CAFA-5 GO terms distribution
axes[0].hist(df['num_cafa5_terms'], bins=50, alpha=0.7, edgecolor='black')
axes[0].set_title('Distribution of CAFA-5 GO Terms per Sequence')
axes[0].set_xlabel('Number of GO Terms')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df['num_cafa5_terms'].mean(), color='red', linestyle='--', label=f'Mean: {df["num_cafa5_terms"].mean():.1f}')
axes[0].legend()

# CAFA-6 GO terms distribution
axes[1].hist(df['num_cafa6_terms'], bins=50, alpha=0.7, edgecolor='black')
axes[1].set_title('Distribution of CAFA-6 GO Terms per Sequence')
axes[1].set_xlabel('Number of GO Terms')
axes[1].set_ylabel('Frequency')
axes[1].axvline(df['num_cafa6_terms'].mean(), color='red', linestyle='--', label=f'Mean: {df["num_cafa6_terms"].mean():.1f}')
axes[1].legend()

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 8))
plt.scatter(df['num_cafa5_terms'], df['num_cafa6_terms'], alpha=0.6, s=30)
plt.xlabel('Number of CAFA-5 GO Terms')
plt.ylabel('Number of CAFA-6 GO Terms')
plt.title('CAFA-5 vs CAFA-6 GO Terms per Sequence')
plt.grid(True, alpha=0.3)

# Add diagonal line
max_terms = max(df['num_cafa5_terms'].max(), df['num_cafa6_terms'].max())
plt.plot([0, max_terms], [0, max_terms], 'r--', alpha=0.7, label='Equal terms')
plt.legend()
plt.show()


# Calculate differences
df['go_terms_diff'] = df['num_cafa6_terms'] - df['num_cafa5_terms']
df['go_terms_ratio'] = df.apply(lambda row: 
                               row['num_cafa6_terms'] / row['num_cafa5_terms'] if row['num_cafa5_terms'] > 0 
                               else (float('inf') if row['num_cafa6_terms'] > 0 else 0), axis=1)

print("GO Terms Difference Statistics:")
print(f"Mean difference (CAFA-6 - CAFA-5): {df['go_terms_diff'].mean():.2f}")
print(f"Median difference: {df['go_terms_diff'].median():.2f}")
print(f"Sequences with more CAFA-6 terms: {(df['go_terms_diff'] > 0).sum()}")
print(f"Sequences with equal terms: {(df['go_terms_diff'] == 0).sum()}")
print(f"Sequences with more CAFA-5 terms: {(df['go_terms_diff'] < 0).sum()}")


plt.figure(figsize=(12, 6))
plt.hist(df['go_terms_diff'], bins=100, alpha=0.7, edgecolor='black')
plt.xlabel('Difference (CAFA-6 GO Terms - CAFA-5 GO Terms)')
plt.ylabel('Frequency')
plt.title('Distribution of GO Terms Differences')
plt.axvline(df['go_terms_diff'].mean(), color='red', linestyle='--', label=f'Mean: {df["go_terms_diff"].mean():.2f}')
plt.axvline(df['go_terms_diff'].median(), color='green', linestyle='--', label=f'Median: {df["go_terms_diff"].median():.2f}')
plt.legend()
plt.show()


# Analyze GO terms that were skipped in CAFA-6
def flatten_go_terms(column):
    all_terms = []
    for terms in df[column]:
        if isinstance(terms, list):
            all_terms.extend(terms)
    return all_terms

cafa5_all_terms = flatten_go_terms('cafa5_go_terms')
cafa6_all_terms = flatten_go_terms('cafa6_go_terms')

cafa5_term_counts = Counter(cafa5_all_terms)
cafa6_term_counts = Counter(cafa6_all_terms)

# GO terms only in CAFA-5
only_cafa5_terms = set(cafa5_term_counts.keys()) - set(cafa6_term_counts.keys())
# GO terms only in CAFA-6
only_cafa6_terms = set(cafa6_term_counts.keys()) - set(cafa5_term_counts.keys())
# Common GO terms
common_terms = set(cafa5_term_counts.keys()) & set(cafa6_term_counts.keys())

print(f"GO terms only in CAFA-5: {len(only_cafa5_terms)}")
print(f"GO terms only in CAFA-6: {len(only_cafa6_terms)}")
print(f"Common GO terms: {len(common_terms)}")
print(f"Total unique GO terms in CAFA-5: {len(set(cafa5_term_counts.keys()))}")
print(f"Total unique GO terms in CAFA-6: {len(set(cafa6_term_counts.keys()))}")


# Sequence length analysis
df['sequence_length'] = df['sequence'].str.len()

plt.figure(figsize=(12, 6))
plt.scatter(df['sequence_length'], df['go_terms_diff'], alpha=0.6, s=30)
plt.xlabel('Sequence Length')
plt.ylabel('GO Terms Difference (CAFA-6 - CAFA-5)')
plt.title('Sequence Length vs GO Terms Difference')
plt.grid(True, alpha=0.3)
plt.show()


# Correlation analysis
correlation_matrix = df[['sequence_length', 'num_cafa5_terms', 'num_cafa6_terms', 'go_terms_diff']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

print("Correlation with sequence length:")
print(f"CAFA-5 GO terms: {df['sequence_length'].corr(df['num_cafa5_terms']):.3f}")
print(f"CAFA-6 GO terms: {df['sequence_length'].corr(df['num_cafa6_terms']):.3f}")
print(f"GO terms difference: {df['sequence_length'].corr(df['go_terms_diff']):.3f}")


# Calculate recall and precision metrics for GO terms
def calculate_go_recall_precision(row):
    cafa5_terms = set(row['cafa5_go_terms']) if isinstance(row['cafa5_go_terms'], list) else set()
    cafa6_terms = set(row['cafa6_go_terms']) if isinstance(row['cafa6_go_terms'], list) else set()
    
    if not cafa6_terms:
        return {'recall': 0, 'precision': 0, 'f1': 0, 'jaccard': 0}
    
    # Recall: how many CAFA-6 terms are in CAFA-5 (what fraction of CAFA-6 predictions are correct)
    recall = len(cafa6_terms & cafa5_terms) / len(cafa6_terms) if cafa6_terms else 0
    
    # Precision: how many CAFA-5 terms are in CAFA-6 (what fraction of CAFA-5 terms are predicted)
    precision = len(cafa5_terms & cafa6_terms) / len(cafa5_terms) if cafa5_terms else 0
    
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Jaccard similarity
    jaccard = len(cafa5_terms & cafa6_terms) / len(cafa5_terms | cafa6_terms) if (cafa5_terms | cafa6_terms) else 0
    
    return {'recall': recall, 'precision': precision, 'f1': f1, 'jaccard': jaccard}

# Apply metrics calculation
metrics_df = df.apply(calculate_go_recall_precision, axis=1, result_type='expand')
df = pd.concat([df, metrics_df], axis=1)

print("GO Terms Similarity Metrics:")
print(f"Mean Recall (CAFA-6 terms found in CAFA-5): {df['recall'].mean():.3f}")
print(f"Mean Precision (CAFA-5 terms found in CAFA-6): {df['precision'].mean():.3f}")
print(f"Mean F1 Score: {df['f1'].mean():.3f}")
print(f"Mean Jaccard Similarity: {df['jaccard'].mean():.3f}")

# Sequences with perfect recall (all CAFA-6 terms are in CAFA-5)
perfect_recall = (df['recall'] == 1.0).sum()
print(f"Sequences with perfect recall: {perfect_recall} ({perfect_recall/len(df)*100:.1f}%)")

# Sequences with perfect precision (all CAFA-5 terms are in CAFA-6)
perfect_precision = (df['precision'] == 1.0).sum()
print(f"Sequences with perfect precision: {perfect_precision} ({perfect_precision/len(df)*100:.1f}%)")

# Sequences with no overlap
no_overlap = (df['jaccard'] == 0).sum()
print(f"Sequences with no GO term overlap: {no_overlap} ({no_overlap/len(df)*100:.1f}%)")


# Distribution of recall and precision
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Recall distribution
axes[0,0].hist(df['recall'], bins=50, alpha=0.7, edgecolor='black')
axes[0,0].set_title('Distribution of Recall')
axes[0,0].set_xlabel('Recall')
axes[0,0].set_ylabel('Frequency')
axes[0,0].axvline(df['recall'].mean(), color='red', linestyle='--', label=f'Mean: {df["recall"].mean():.3f}')
axes[0,0].legend()

# Precision distribution
axes[0,1].hist(df['precision'], bins=50, alpha=0.7, edgecolor='black')
axes[0,1].set_title('Distribution of Precision')
axes[0,1].set_xlabel('Precision')
axes[0,1].set_ylabel('Frequency')
axes[0,1].axvline(df['precision'].mean(), color='red', linestyle='--', label=f'Mean: {df["precision"].mean():.3f}')
axes[0,1].legend()

# F1 distribution
axes[1,0].hist(df['f1'], bins=50, alpha=0.7, edgecolor='black')
axes[1,0].set_title('Distribution of F1 Score')
axes[1,0].set_xlabel('F1 Score')
axes[1,0].set_ylabel('Frequency')
axes[1,0].axvline(df['f1'].mean(), color='red', linestyle='--', label=f'Mean: {df["f1"].mean():.3f}')
axes[1,0].legend()

# Jaccard distribution
axes[1,1].hist(df['jaccard'], bins=50, alpha=0.7, edgecolor='black')
axes[1,1].set_title('Distribution of Jaccard Similarity')
axes[1,1].set_xlabel('Jaccard Similarity')
axes[1,1].set_ylabel('Frequency')
axes[1,1].axvline(df['jaccard'].mean(), color='red', linestyle='--', label=f'Mean: {df["jaccard"].mean():.3f}')
axes[1,1].legend()

plt.tight_layout()
plt.show()


# Scatter plot: Recall vs Precision
plt.figure(figsize=(10, 8))
plt.scatter(df['recall'], df['precision'], alpha=0.6, s=30, c=df['sequence_length'], cmap='viridis')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Recall vs Precision (colored by sequence length)')
plt.colorbar(label='Sequence Length')
plt.grid(True, alpha=0.3)
plt.show()


print("=" * 60)
print("ANALYSIS: Creating Fake CAFA-6 Predictions from CAFA-5")
print("=" * 60)

print(f"GO terms to remove from CAFA-5: {len(only_cafa5_terms)}")

# Create fake CAFA-6 predictions
def create_fake_cafa6(row):
    cafa5_terms = set(row['cafa5_go_terms']) if isinstance(row['cafa5_go_terms'], list) else set()
    # Remove terms that don't exist in CAFA-6
    fake_terms = cafa5_terms - only_cafa5_terms
    return list(fake_terms)

df['fake_cafa6_go_terms'] = df.apply(create_fake_cafa6, axis=1)
df['num_fake_cafa6_terms'] = df['fake_cafa6_go_terms'].apply(len)

# Calculate metrics between fake CAFA-6 and real CAFA-6
def calculate_fake_vs_real_metrics(row):
    fake_terms = set(row['fake_cafa6_go_terms'])
    real_terms = set(row['cafa6_go_terms']) if isinstance(row['cafa6_go_terms'], list) else set()
    
    if not real_terms:
        return {'fake_recall': 0, 'fake_precision': 0, 'fake_f1': 0, 'fake_jaccard': 0}
    
    # Recall: how many real CAFA-6 terms are in fake predictions
    recall = len(real_terms & fake_terms) / len(real_terms) if real_terms else 0
    
    # Precision: how many fake predictions are correct (in real CAFA-6)
    precision = len(fake_terms & real_terms) / len(fake_terms) if fake_terms else 0
    
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Jaccard similarity
    jaccard = len(fake_terms & real_terms) / len(fake_terms | real_terms) if (fake_terms | real_terms) else 0
    
    return {'fake_recall': recall, 'fake_precision': precision, 'fake_f1': f1, 'fake_jaccard': jaccard}

fake_metrics_df = df.apply(calculate_fake_vs_real_metrics, axis=1, result_type='expand')
df = pd.concat([df, fake_metrics_df], axis=1)

print(f"\nFake CAFA-6 predictions from CAFA-5:")
print(f"Average terms per sequence: {df['num_fake_cafa6_terms'].mean():.2f}")
print(f"Sequences with fake predictions: {(df['num_fake_cafa6_terms'] > 0).sum()}")
print(f"Sequences with no fake predictions: {(df['num_fake_cafa6_terms'] == 0).sum()}")

print(f"\nMetrics between Fake CAFA-6 and Real CAFA-6:")
print(f"Mean Recall (real terms found in fake): {df['fake_recall'].mean():.3f}")
print(f"Mean Precision (fake terms that are real): {df['fake_precision'].mean():.3f}")
print(f"Mean F1 Score: {df['fake_f1'].mean():.3f}")
print(f"Mean Jaccard Similarity: {df['fake_jaccard'].mean():.3f}")


# Calculate original metrics first
def calculate_original_metrics(row):
    cafa5_terms = set(row['cafa5_go_terms']) if isinstance(row['cafa5_go_terms'], list) else set()
    cafa6_terms = set(row['cafa6_go_terms']) if isinstance(row['cafa6_go_terms'], list) else set()
    
    if not cafa6_terms:
        return {'orig_recall': 0, 'orig_precision': 0, 'orig_f1': 0}
    
    # Recall: how many CAFA-6 terms are in CAFA-5
    recall = len(cafa6_terms & cafa5_terms) / len(cafa6_terms)
    
    # Precision: how many CAFA-5 terms are in CAFA-6
    precision = len(cafa5_terms & cafa6_terms) / len(cafa5_terms) if cafa5_terms else 0
    
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'orig_recall': recall, 'orig_precision': precision, 'orig_f1': f1}

orig_metrics_df = df.apply(calculate_original_metrics, axis=1, result_type='expand')
df = pd.concat([df, orig_metrics_df], axis=1)

# Compare with original CAFA-5 performance
print(f"\nComparison with original CAFA-5 vs CAFA-6:")
print(f"Original CAFA-5 Recall: {df['orig_recall'].mean():.3f} → Fake CAFA-6 Recall: {df['fake_recall'].mean():.3f}")
print(f"Original CAFA-5 Precision: {df['orig_precision'].mean():.3f} → Fake CAFA-6 Precision: {df['fake_precision'].mean():.3f}")
print(f"Original CAFA-5 F1: {df['orig_f1'].mean():.3f} → Fake CAFA-6 F1: {df['fake_f1'].mean():.3f}")




