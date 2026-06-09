## imports required for the basic code template below.

import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import glob
import sys
import argparse
from collections import defaultdict
from typing import Iterator, Tuple, Union, List

train_datasets_dir = "/kaggle/input/adaptive-immune-profiling-challenge-2025/train_datasets/train_datasets"
test_datasets_dir = "/kaggle/input/adaptive-immune-profiling-challenge-2025/test_datasets/test_datasets"
results_dir = "/kaggle/working/results"


## some utility functions such as data loaders, etc.

def load_data_generator(data_dir: str, metadata_filename='metadata.csv') -> Iterator[
    Union[Tuple[str, pd.DataFrame, bool], Tuple[str, pd.DataFrame]]]:
    """
    A generator to load immune repertoire data.

    This function operates in two modes:
    1.  If metadata is found, it yields data based on the metadata file.
    2.  If metadata is NOT found, it uses glob to find and yield all '.tsv'
        files in the directory.

    Args:
        data_dir (str): The path to the directory containing the data.

    Yields:
        An iterator of tuples. The format depends on the mode:
        - With metadata: (repertoire_id, pd.DataFrame, label_positive)
        - Without metadata: (filename, pd.DataFrame)
    """
    metadata_path = os.path.join(data_dir, metadata_filename)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        for row in metadata_df.itertuples(index=False):
            file_path = os.path.join(data_dir, row.filename)
            try:
                repertoire_df = pd.read_csv(file_path, sep='\t')
                yield row.repertoire_id, repertoire_df, row.label_positive
            except FileNotFoundError:
                print(f"Warning: File '{row.filename}' listed in metadata not found. Skipping.")
                continue
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        tsv_files = glob.glob(search_pattern)
        for file_path in sorted(tsv_files):
            try:
                filename = os.path.basename(file_path)
                repertoire_df = pd.read_csv(file_path, sep='\t')
                yield filename, repertoire_df
            except Exception as e:
                print(f"Warning: Could not read file '{file_path}'. Error: {e}. Skipping.")
                continue


def load_full_dataset(data_dir: str) -> pd.DataFrame:
    """
    Loads all TSV files from a directory and concatenates them into a single DataFrame.

    This function handles two scenarios:
    1. If metadata.csv exists, it loads data based on the metadata and adds
       'repertoire_id' and 'label_positive' columns.
    2. If metadata.csv does not exist, it loads all .tsv files and adds
       a 'filename' column as an identifier.

    Args:
        data_dir (str): The path to the data directory.

    Returns:
        pd.DataFrame: A single, concatenated DataFrame containing all the data.
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    df_list = []
    data_loader = load_data_generator(data_dir=data_dir)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        total_files = len(metadata_df)
        for rep_id, data_df, label in tqdm(data_loader, total=total_files, desc="Loading files"):
            data_df['ID'] = rep_id
            data_df['label_positive'] = label
            df_list.append(data_df)
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        total_files = len(glob.glob(search_pattern))
        for filename, data_df in tqdm(data_loader, total=total_files, desc="Loading files"):
            data_df['ID'] = os.path.basename(filename).replace(".tsv", "")
            df_list.append(data_df)

    if not df_list:
        print("Warning: No data files were loaded.")
        return pd.DataFrame()

    full_dataset_df = pd.concat(df_list, ignore_index=True)
    return full_dataset_df


def load_and_encode_kmers(data_dir: str, k: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loading and k-mer encoding of repertoire data.

    Args:
        data_dir: Path to data directory
        k: K-mer length

    Returns:
        Tuple of (encoded_features_df, metadata_df)
        metadata_df always contains 'ID', and 'label_positive' if available
    """
    from collections import Counter

    metadata_path = os.path.join(data_dir, 'metadata.csv')
    data_loader = load_data_generator(data_dir=data_dir)

    repertoire_features = []
    metadata_records = []

    search_pattern = os.path.join(data_dir, '*.tsv')
    total_files = len(glob.glob(search_pattern))

    for item in tqdm(data_loader, total=total_files, desc=f"Encoding {k}-mers"):
        if os.path.exists(metadata_path):
            rep_id, data_df, label = item
        else:
            filename, data_df = item
            rep_id = os.path.basename(filename).replace(".tsv", "")
            label = None

        kmer_counts = Counter()
        for seq in data_df['junction_aa'].dropna():
            for i in range(len(seq) - k + 1):
                kmer_counts[seq[i:i + k]] += 1

        repertoire_features.append({
            'ID': rep_id,
            **kmer_counts
        })

        metadata_record = {'ID': rep_id}
        if label is not None:
            metadata_record['label_positive'] = label
        metadata_records.append(metadata_record)

        del data_df, kmer_counts

    features_df = pd.DataFrame(repertoire_features).fillna(0).set_index('ID')
    features_df.fillna(0)
    metadata_df = pd.DataFrame(metadata_records)

    return features_df, metadata_df


def save_tsv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep='\t', index=False)


def get_repertoire_ids(data_dir: str) -> list:
    """
    Retrieves repertoire IDs from the metadata file or filenames in the directory.

    Args:
        data_dir (str): The path to the data directory.

    Returns:
        list: A list of repertoire IDs.
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        repertoire_ids = metadata_df['repertoire_id'].tolist()
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        tsv_files = glob.glob(search_pattern)
        repertoire_ids = [os.path.basename(f).replace('.tsv', '') for f in sorted(tsv_files)]

    return repertoire_ids


def generate_random_top_sequences_df(n_seq: int = 50000) -> pd.DataFrame:
    """
    Generates a random DataFrame simulating top important sequences.

    Args:
        n_seq (int): Number of sequences to generate.

    Returns:
        pd.DataFrame: A DataFrame with columns 'ID', 'dataset', 'junction_aa', 'v_call', 'j_call'.
    """
    seqs = set()
    while len(seqs) < n_seq:
        seq = ''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), size=15))
        seqs.add(seq)
    data = {
        'junction_aa': list(seqs),
        'v_call': ['TRBV20-1'] * n_seq,
        'j_call': ['TRBJ2-7'] * n_seq,
        'importance_score': np.random.rand(n_seq)
    }
    return pd.DataFrame(data)


def validate_dirs_and_files(train_dir: str, test_dirs: List[str], out_dir: str) -> None:
    assert os.path.isdir(train_dir), f"Train directory `{train_dir}` does not exist."
    train_tsvs = glob.glob(os.path.join(train_dir, "*.tsv"))
    assert train_tsvs, f"No .tsv files found in train directory `{train_dir}`."
    metadata_path = os.path.join(train_dir, "metadata.csv")
    assert os.path.isfile(metadata_path), f"`metadata.csv` not found in train directory `{train_dir}`."

    for test_dir in test_dirs:
        assert os.path.isdir(test_dir), f"Test directory `{test_dir}` does not exist."
        test_tsvs = glob.glob(os.path.join(test_dir, "*.tsv"))
        assert test_tsvs, f"No .tsv files found in test directory `{test_dir}`."

    try:
        os.makedirs(out_dir, exist_ok=True)
        test_file = os.path.join(out_dir, "test_write_permission.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        print(f"Failed to create or write to output directory `{out_dir}`: {e}")
        sys.exit(1)


def concatenate_output_files(out_dir: str) -> None:
    """
    Concatenates all test predictions and important sequences TSV files from the output directory.

    This function finds all files matching the patterns:
    - *_test_predictions.tsv
    - *_important_sequences.tsv

    and concatenates them to match the expected output format of submissions.csv.

    Args:
        out_dir (str): Path to the output directory containing the TSV files.

    Returns:
        pd.DataFrame: Concatenated DataFrame with predictions followed by important sequences.
                     Columns: ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']
    """
    predictions_pattern = os.path.join(out_dir, '*_test_predictions.tsv')
    sequences_pattern = os.path.join(out_dir, '*_important_sequences.tsv')

    predictions_files = sorted(glob.glob(predictions_pattern))
    sequences_files = sorted(glob.glob(sequences_pattern))

    df_list = []

    for pred_file in predictions_files:
        try:
            df = pd.read_csv(pred_file, sep='\t')
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not read predictions file '{pred_file}'. Error: {e}. Skipping.")
            continue

    for seq_file in sequences_files:
        try:
            df = pd.read_csv(seq_file, sep='\t')
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not read sequences file '{seq_file}'. Error: {e}. Skipping.")
            continue

    if not df_list:
        print("Warning: No output files were found to concatenate.")
        concatenated_df = pd.DataFrame(
            columns=['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call'])
    else:
        concatenated_df = pd.concat(df_list, ignore_index=True)
    submissions_file = os.path.join(out_dir, 'submissions.csv')
    concatenated_df.to_csv(submissions_file, index=False)
    print(f"Concatenated output written to `{submissions_file}`.")


def get_dataset_pairs(train_dir: str, test_dir: str) -> List[Tuple[str, List[str]]]:
    """Returns list of (train_path, [test_paths]) tuples for dataset pairs."""
    test_groups = defaultdict(list)
    for test_name in sorted(os.listdir(test_dir)):
        if test_name.startswith("test_dataset_"):
            base_id = test_name.replace("test_dataset_", "").split("_")[0]
            test_groups[base_id].append(os.path.join(test_dir, test_name))

    pairs = []
    for train_name in sorted(os.listdir(train_dir)):
        if train_name.startswith("train_dataset_"):
            train_id = train_name.replace("train_dataset_", "")
            train_path = os.path.join(train_dir, train_name)
            pairs.append((train_path, test_groups.get(train_id, [])))

    return pairs


## Explorative Datenanalyse (EDA) fÃ¼r Adaptive Immune Profiling Challenge

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Setze Stil fÃ¼r bessere Visualisierungen
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

## 1. Ãœberblick Ã¼ber die Datenstruktur

def get_data_overview(train_dir, test_dir):
    """Gibt einen Ãœberblick Ã¼ber die Datenstruktur"""
    
    print("=" * 80)
    print("DATENSTRUKTUR ÃœBERBLICK")
    print("=" * 80)
    
    # Training Datasets
    train_datasets = [d for d in os.listdir(train_dir) if d.startswith('train_dataset_')]
    print(f"\nğŸ“� Anzahl Training Datasets: {len(train_datasets)}")
    print(f"Training Datasets: {train_datasets[:5]}..." if len(train_datasets) > 5 else f"Training Datasets: {train_datasets}")
    
    # Test Datasets
    test_datasets = [d for d in os.listdir(test_dir) if d.startswith('test_dataset_')]
    print(f"\nğŸ“� Anzahl Test Datasets: {len(test_datasets)}")
    print(f"Test Datasets: {test_datasets[:5]}..." if len(test_datasets) > 5 else f"Test Datasets: {test_datasets}")
    
    return train_datasets, test_datasets


## 2. Analyse eines einzelnen Datensatzes

def analyze_single_dataset(dataset_path, dataset_name):
    """Detaillierte Analyse eines einzelnen Datensatzes"""
    
    print("\n" + "=" * 80)
    print(f"ANALYSE: {dataset_name}")
    print("=" * 80)
    
    # Lade Metadata
    metadata_path = os.path.join(dataset_path, 'metadata.csv')
    if os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path)
        print(f"\nğŸ“Š Metadata Info:")
        print(f"   - Anzahl Repertoires: {len(metadata)}")
        print(f"   - Spalten: {list(metadata.columns)}")
        
        if 'label_positive' in metadata.columns:
            label_dist = metadata['label_positive'].value_counts()
            print(f"\nğŸ�·ï¸�  Label-Verteilung:")
            print(f"   - Positive (1): {label_dist.get(1, 0)} ({label_dist.get(1, 0)/len(metadata)*100:.1f}%)")
            print(f"   - Negative (0): {label_dist.get(0, 0)} ({label_dist.get(0, 0)/len(metadata)*100:.1f}%)")
            
            # Visualisierung
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            
            # Label Distribution
            label_dist.plot(kind='bar', ax=ax[0], color=['#ff6b6b', '#4ecdc4'])
            ax[0].set_title(f'Label-Verteilung - {dataset_name}')
            ax[0].set_xlabel('Label')
            ax[0].set_ylabel('Anzahl')
            ax[0].set_xticklabels(['Negativ (0)', 'Positiv (1)'], rotation=0)
            
            # Pie Chart
            ax[1].pie(label_dist.values, labels=['Negativ', 'Positiv'], 
                     autopct='%1.1f%%', colors=['#ff6b6b', '#4ecdc4'])
            ax[1].set_title('Label-Proportion')
            
            plt.tight_layout()
            plt.show()
    
    # Lade ein Beispiel-Repertoire
    tsv_files = glob.glob(os.path.join(dataset_path, '*.tsv'))
    if tsv_files:
        print(f"\nğŸ“„ Anzahl TSV-Dateien: {len(tsv_files)}")
        
        # Lade erstes Repertoire
        sample_rep = pd.read_csv(tsv_files[0], sep='\t')
        print(f"\nğŸ”¬ Beispiel-Repertoire: {os.path.basename(tsv_files[0])}")
        print(f"   - Anzahl Sequenzen: {len(sample_rep)}")
        print(f"   - Spalten: {list(sample_rep.columns)}")
        print(f"\n   Erste Zeilen:")
        print(sample_rep.head(3))
        
        return metadata, sample_rep
    
    return metadata, None



## 3. CDR3-Sequenz Analyse

def analyze_cdr3_sequences(dataset_path, sample_size=5):
    """Analysiert CDR3-Sequenzen (junction_aa)"""
    
    print("\n" + "=" * 80)
    print("CDR3-SEQUENZ ANALYSE")
    print("=" * 80)
    
    # Sammle Daten aus mehreren Repertoires
    all_sequences = []
    all_lengths = []
    all_v_genes = []
    all_j_genes = []
    
    tsv_files = glob.glob(os.path.join(dataset_path, '*.tsv'))
    
    print(f"\nâ�³ Analysiere {min(sample_size, len(tsv_files))} Repertoires...")
    
    for tsv_file in tsv_files[:sample_size]:
        df = pd.read_csv(tsv_file, sep='\t')
        
        if 'junction_aa' in df.columns:
            sequences = df['junction_aa'].dropna()
            all_sequences.extend(sequences.tolist())
            all_lengths.extend(sequences.str.len().tolist())
        
        if 'v_call' in df.columns:
            all_v_genes.extend(df['v_call'].dropna().tolist())
        
        if 'j_call' in df.columns:
            all_j_genes.extend(df['j_call'].dropna().tolist())
    
    print(f"\nğŸ“Š Statistiken Ã¼ber {len(all_sequences)} Sequenzen:")
    print(f"   - Mittlere LÃ¤nge: {np.mean(all_lengths):.2f} Â± {np.std(all_lengths):.2f}")
    print(f"   - Min LÃ¤nge: {np.min(all_lengths)}")
    print(f"   - Max LÃ¤nge: {np.max(all_lengths)}")
    print(f"   - Median LÃ¤nge: {np.median(all_lengths):.0f}")
    
    # Visualisierungen
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. LÃ¤ngenverteilung
    axes[0, 0].hist(all_lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(all_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(all_lengths):.1f}')
    axes[0, 0].axvline(np.median(all_lengths), color='green', linestyle='--', label=f'Median: {np.median(all_lengths):.1f}')
    axes[0, 0].set_xlabel('CDR3 LÃ¤nge (AminosÃ¤uren)')
    axes[0, 0].set_ylabel('HÃ¤ufigkeit')
    axes[0, 0].set_title('Verteilung der CDR3-LÃ¤ngen')
    axes[0, 0].legend()
    
    # 2. AminosÃ¤ure-Komposition
    aa_counter = Counter()
    for seq in all_sequences[:10000]:  # Sample fÃ¼r Performance
        aa_counter.update(seq)
    
    aa_df = pd.DataFrame(aa_counter.most_common(20), columns=['AminosÃ¤ure', 'HÃ¤ufigkeit'])
    axes[0, 1].bar(aa_df['AminosÃ¤ure'], aa_df['HÃ¤ufigkeit'], color='coral')
    axes[0, 1].set_xlabel('AminosÃ¤ure')
    axes[0, 1].set_ylabel('HÃ¤ufigkeit')
    axes[0, 1].set_title('Top 20 AminosÃ¤uren in CDR3-Sequenzen')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. V-Gene Verteilung
    if all_v_genes:
        v_counter = Counter(all_v_genes)
        v_top = pd.DataFrame(v_counter.most_common(15), columns=['V-Gen', 'HÃ¤ufigkeit'])
        axes[1, 0].barh(v_top['V-Gen'], v_top['HÃ¤ufigkeit'], color='lightgreen')
        axes[1, 0].set_xlabel('HÃ¤ufigkeit')
        axes[1, 0].set_title('Top 15 V-Gene')
        axes[1, 0].invert_yaxis()
    
    # 4. J-Gene Verteilung
    if all_j_genes:
        j_counter = Counter(all_j_genes)
        j_top = pd.DataFrame(j_counter.most_common(15), columns=['J-Gen', 'HÃ¤ufigkeit'])
        axes[1, 1].barh(j_top['J-Gen'], j_top['HÃ¤ufigkeit'], color='plum')
        axes[1, 1].set_xlabel('HÃ¤ufigkeit')
        axes[1, 1].set_title('Top 15 J-Gene')
        axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    return all_sequences, all_lengths, all_v_genes, all_j_genes

## 4. Repertoire-Level Analyse

def analyze_repertoire_characteristics(dataset_path, sample_size=20):
    """Analysiert Charakteristiken auf Repertoire-Ebene"""
    
    print("\n" + "=" * 80)
    print("REPERTOIRE-LEVEL ANALYSE")
    print("=" * 80)
    
    metadata_path = os.path.join(dataset_path, 'metadata.csv')
    metadata = pd.read_csv(metadata_path)
    
    repertoire_stats = []
    
    print(f"\nâ�³ Analysiere {min(sample_size, len(metadata))} Repertoires...")
    
    for idx, row in metadata.head(sample_size).iterrows():
        file_path = os.path.join(dataset_path, row['filename'])
        df = pd.read_csv(file_path, sep='\t')
        
        stats = {
            'repertoire_id': row['repertoire_id'],
            'label': row.get('label_positive', None),
            'n_sequences': len(df),
            'mean_cdr3_length': df['junction_aa'].str.len().mean() if 'junction_aa' in df.columns else None,
            'unique_v_genes': df['v_call'].nunique() if 'v_call' in df.columns else None,
            'unique_j_genes': df['j_call'].nunique() if 'j_call' in df.columns else None,
        }
        repertoire_stats.append(stats)
    
    rep_df = pd.DataFrame(repertoire_stats)
    
    print("\nğŸ“Š Repertoire-Statistiken:")
    print(rep_df.describe())
    
    # Visualisierung
    if 'label' in rep_df.columns and rep_df['label'].notna().any():
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Anzahl Sequenzen pro Label
        rep_df.boxplot(column='n_sequences', by='label', ax=axes[0, 0])
        axes[0, 0].set_title('Anzahl Sequenzen pro Repertoire')
        axes[0, 0].set_xlabel('Label')
        axes[0, 0].set_ylabel('Anzahl Sequenzen')
        plt.sca(axes[0, 0])
        plt.xticks([1, 2], ['Negativ (0)', 'Positiv (1)'])
        
        # 2. Mittlere CDR3-LÃ¤nge pro Label
        rep_df.boxplot(column='mean_cdr3_length', by='label', ax=axes[0, 1])
        axes[0, 1].set_title('Mittlere CDR3-LÃ¤nge pro Repertoire')
        axes[0, 1].set_xlabel('Label')
        axes[0, 1].set_ylabel('Mittlere LÃ¤nge')
        plt.sca(axes[0, 1])
        plt.xticks([1, 2], ['Negativ (0)', 'Positiv (1)'])
        
        # 3. Unique V-Gene pro Label
        rep_df.boxplot(column='unique_v_genes', by='label', ax=axes[1, 0])
        axes[1, 0].set_title('Anzahl einzigartiger V-Gene')
        axes[1, 0].set_xlabel('Label')
        axes[1, 0].set_ylabel('Anzahl V-Gene')
        plt.sca(axes[1, 0])
        plt.xticks([1, 2], ['Negativ (0)', 'Positiv (1)'])
        
        # 4. Unique J-Gene pro Label
        rep_df.boxplot(column='unique_j_genes', by='label', ax=axes[1, 1])
        axes[1, 1].set_title('Anzahl einzigartiger J-Gene')
        axes[1, 1].set_xlabel('Label')
        axes[1, 1].set_ylabel('Anzahl J-Gene')
        plt.sca(axes[1, 1])
        plt.xticks([1, 2], ['Negativ (0)', 'Positiv (1)'])
        
        plt.tight_layout()
        plt.show()
    
    return rep_df



## 5. K-mer Analyse

def analyze_kmers(sequences, k=3, top_n=20):
    """Analysiert K-mer HÃ¤ufigkeiten"""
    
    print("\n" + "=" * 80)
    print(f"{k}-MER ANALYSE")
    print("=" * 80)
    
    kmer_counter = Counter()
    
    print(f"\nâ�³ Extrahiere {k}-mere aus {len(sequences)} Sequenzen...")
    
    for seq in sequences[:10000]:  # Sample fÃ¼r Performance
        if isinstance(seq, str):
            for i in range(len(seq) - k + 1):
                kmer_counter[seq[i:i+k]] += 1
    
    print(f"\nğŸ“Š Gefundene einzigartige {k}-mere: {len(kmer_counter)}")
    
    # Top K-mere
    top_kmers = pd.DataFrame(kmer_counter.most_common(top_n), 
                             columns=[f'{k}-mer', 'HÃ¤ufigkeit'])
    
    print(f"\nTop {top_n} {k}-mere:")
    print(top_kmers)
    
    # Visualisierung
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_kmers)), top_kmers['HÃ¤ufigkeit'], color='steelblue')
    plt.xticks(range(len(top_kmers)), top_kmers[f'{k}-mer'], rotation=45, ha='right')
    plt.xlabel(f'{k}-mer')
    plt.ylabel('HÃ¤ufigkeit')
    plt.title(f'Top {top_n} hÃ¤ufigste {k}-mere')
    plt.tight_layout()
    plt.show()
    
    return kmer_counter

## 6. Vergleich zwischen Training und Test Datasets

def compare_train_test_datasets(train_dir, test_dir, dataset_id):
    """Vergleicht korrespondierende Train/Test Datasets"""
    
    print("\n" + "=" * 80)
    print(f"TRAIN vs TEST VERGLEICH - Dataset {dataset_id}")
    print("=" * 80)
    
    train_path = os.path.join(train_dir, f'train_dataset_{dataset_id}')
    test_paths = glob.glob(os.path.join(test_dir, f'test_dataset_{dataset_id}*'))
    
    if not os.path.exists(train_path) or not test_paths:
        print("âš ï¸�  Korrespondierende Datasets nicht gefunden")
        return
    
    # Training Data
    train_metadata = pd.read_csv(os.path.join(train_path, 'metadata.csv'))
    train_tsv_count = len(glob.glob(os.path.join(train_path, '*.tsv')))
    
    print(f"\nğŸ�“ Training Dataset:")
    print(f"   - Repertoires: {len(train_metadata)}")
    print(f"   - TSV Files: {train_tsv_count}")
    if 'label_positive' in train_metadata.columns:
        print(f"   - Positive: {train_metadata['label_positive'].sum()}")
        print(f"   - Negative: {(train_metadata['label_positive'] == 0).sum()}")
    
    # Test Data
    print(f"\nğŸ§ª Test Dataset(s):")
    for test_path in test_paths:
        test_name = os.path.basename(test_path)
        test_tsv_count = len(glob.glob(os.path.join(test_path, '*.tsv')))
        print(f"   - {test_name}: {test_tsv_count} Repertoires")

## 7. Zusammenfassung aller Datasets

def summarize_all_datasets(train_dir):
    """Erstellt eine Zusammenfassung aller Datasets"""
    
    print("\n" + "=" * 80)
    print("ZUSAMMENFASSUNG ALLER DATASETS")
    print("=" * 80)
    
    summary_data = []
    
    train_datasets = [d for d in os.listdir(train_dir) if d.startswith('train_dataset_')]
    
    for dataset_name in tqdm(train_datasets, desc="Analysiere Datasets"):
        dataset_path = os.path.join(train_dir, dataset_name)
        metadata_path = os.path.join(dataset_path, 'metadata.csv')
        
        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
            
            summary = {
                'dataset': dataset_name,
                'n_repertoires': len(metadata),
                'n_positive': metadata['label_positive'].sum() if 'label_positive' in metadata.columns else None,
                'n_negative': (metadata['label_positive'] == 0).sum() if 'label_positive' in metadata.columns else None,
                'balance_ratio': metadata['label_positive'].mean() if 'label_positive' in metadata.columns else None
            }
            summary_data.append(summary)
    
    summary_df = pd.DataFrame(summary_data)
    
    print("\nğŸ“Š Dataset Ãœbersicht:")
    print(summary_df)
    
    # Visualisierung
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Anzahl Repertoires pro Dataset
    axes[0].bar(range(len(summary_df)), summary_df['n_repertoires'], color='skyblue')
    axes[0].set_xlabel('Dataset Index')
    axes[0].set_ylabel('Anzahl Repertoires')
    axes[0].set_title('Repertoires pro Dataset')
    
    # Balance Ratio
    if 'balance_ratio' in summary_df.columns:
        axes[1].bar(range(len(summary_df)), summary_df['balance_ratio'], color='coral')
        axes[1].axhline(y=0.5, color='red', linestyle='--', label='Perfect Balance')
        axes[1].set_xlabel('Dataset Index')
        axes[1].set_ylabel('Positive Label Ratio')
        axes[1].set_title('Class Balance pro Dataset')
        axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return summary_df


## 8. Speichere EDA-Ergebnisse

def save_eda_summary(summary_df, rep_stats, output_path='/kaggle/working/eda_summary.csv'):
    """Speichert EDA-Ergebnisse"""
    summary_df.to_csv(output_path, index=False)
    print(f"\nâœ… EDA-Zusammenfassung gespeichert: {output_path}")

# 1. 
train_datasets, test_datasets = get_data_overview(train_datasets_dir, test_datasets_dir)

# 2. Analysiere Training-Datensatz
for i in range(len(train_datasets)):
    first_train_dataset = os.path.join(train_datasets_dir, train_datasets[i])
    metadata_train, sample_repertoire = analyze_single_dataset(first_train_dataset, train_datasets[i])

# 3. CDR3-Sequenz Analyse
    sequences, lengths, v_genes, j_genes = analyze_cdr3_sequences(first_train_dataset, sample_size=10)

## 4. Repertoire-Level Analyse
    rep_stats = analyze_repertoire_characteristics(first_train_dataset, sample_size=30)

# 5. Analysiere verschiedene K-mer LÃ¤ngen
    for k in [3, 4, 5]:
        kmer_counts = analyze_kmers(sequences, k=k, top_n=15)

# 6. Vergleiche Datensatz
    if train_datasets:
        dataset_id = train_datasets[i].replace('train_dataset_', '')
        compare_train_test_datasets(train_datasets_dir, test_datasets_dir, dataset_id)

# 7. summary 
summary = summarize_all_datasets(train_datasets_dir)

# 8. save eda
save_eda_summary(summary, rep_stats)

print("\n" + "=" * 80)
print("âœ… EDA ABGESCHLOSSEN")
print("=" * 80)



## 9. Sequenz-DiversitÃ¤t Analyse
from scipy.stats import entropy

def calculate_diversity_metrics(dataset_path, sample_size=10):
    """Berechnet DiversitÃ¤ts-Metriken"""
    
    
    print("\n" + "=" * 80)
    print("DIVERSITÃ„TS-ANALYSE")
    print("=" * 80)
    
    metadata = pd.read_csv(os.path.join(dataset_path, 'metadata.csv'))
    
    diversity_stats = []
    
    for idx, row in metadata.head(sample_size).iterrows():
        file_path = os.path.join(dataset_path, row['filename'])
        df = pd.read_csv(file_path, sep='\t')
        
        # Shannon Entropy
        seq_counts = df['junction_aa'].value_counts()
        shannon_ent = entropy(seq_counts, base=2)
        
        # Clonality (1 - normalized entropy)
        max_entropy = np.log2(len(seq_counts))
        clonality = 1 - (shannon_ent / max_entropy) if max_entropy > 0 else 0
        
        diversity_stats.append({
            'repertoire_id': row['repertoire_id'],
            'label': row.get('label_positive', None),
            'unique_sequences': len(seq_counts),
            'total_sequences': len(df),
            'shannon_entropy': shannon_ent,
            'clonality': clonality
        })
    
    div_df = pd.DataFrame(diversity_stats)
    
    print("\nğŸ“Š DiversitÃ¤ts-Statistiken:")
    print(div_df.describe())
    
    # Visualisierung
    if 'label' in div_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        div_df.boxplot(column='shannon_entropy', by='label', ax=axes[0])
        axes[0].set_title('Shannon Entropy pro Label')
        axes[0].set_xlabel('Label')
        
        div_df.boxplot(column='clonality', by='label', ax=axes[1])
        axes[1].set_title('Clonality pro Label')
        axes[1].set_xlabel('Label')
        
        plt.tight_layout()
        plt.show()
    
    return div_df

diversity_df = calculate_diversity_metrics(first_train_dataset, sample_size=20)

## 10. Motif-Suche

def find_common_motifs(sequences, motif_length=5, top_n=10):
    """Findet hÃ¤ufige Motive in Sequenzen"""
    
    print("\n" + "=" * 80)
    print(f"MOTIF-ANALYSE (LÃ¤nge: {motif_length})")
    print("=" * 80)
    
    motif_positions = {}
    
    for seq in sequences[:5000]:
        if isinstance(seq, str) and len(seq) >= motif_length:
            for i in range(len(seq) - motif_length + 1):
                motif = seq[i:i+motif_length]
                position = 'N-terminal' if i < 3 else ('C-terminal' if i > len(seq) - motif_length - 3 else 'Middle')
                
                if motif not in motif_positions:
                    motif_positions[motif] = {'count': 0, 'positions': []}
                motif_positions[motif]['count'] += 1
                motif_positions[motif]['positions'].append(position)
    
    # Sortiere nach HÃ¤ufigkeit
    sorted_motifs = sorted(motif_positions.items(), key=lambda x: x[1]['count'], reverse=True)[:top_n]
    
    print(f"\nTop {top_n} Motive:")
    for motif, data in sorted_motifs:
        pos_dist = Counter(data['positions'])
        print(f"  {motif}: {data['count']}x - Positionen: {dict(pos_dist)}")

find_common_motifs(sequences, motif_length=5, top_n=15)



## 11. Sequenz-Ã„hnlichkeitsanalyse (Clustering)

def analyze_sequence_similarity(sequences, sample_size=1000, method='levenshtein'):
    """
    Analysiert Ã„hnlichkeiten zwischen Sequenzen
    """
    from sklearn.cluster import DBSCAN
    from sklearn.manifold import TSNE
    from Levenshtein import distance as lev_distance
    
    print("\n" + "=" * 80)
    print("SEQUENZ-Ã„HNLICHKEITSANALYSE")
    print("=" * 80)
    
    # Sample fÃ¼r Performance
    sample_seqs = [s for s in sequences if isinstance(s, str) and len(s) > 5][:sample_size]
    
    print(f"\nâ�³ Berechne Distanzmatrix fÃ¼r {len(sample_seqs)} Sequenzen...")
    
    # Erstelle Distanzmatrix
    n = len(sample_seqs)
    dist_matrix = np.zeros((n, n))
    
    for i in tqdm(range(n), desc="Berechne Distanzen"):
        for j in range(i+1, n):
            dist = lev_distance(sample_seqs[i], sample_seqs[j])
            # Normalisiere durch maximale LÃ¤nge
            max_len = max(len(sample_seqs[i]), len(sample_seqs[j]))
            dist_matrix[i, j] = dist / max_len if max_len > 0 else 0
            dist_matrix[j, i] = dist_matrix[i, j]
    
    print(f"\nğŸ“Š Distanz-Statistiken:")
    print(f"   - Mittlere Distanz: {dist_matrix[dist_matrix > 0].mean():.3f}")
    print(f"   - Std Distanz: {dist_matrix[dist_matrix > 0].std():.3f}")
    print(f"   - Min Distanz: {dist_matrix[dist_matrix > 0].min():.3f}")
    print(f"   - Max Distanz: {dist_matrix.max():.3f}")
    
    # DBSCAN Clustering
    print("\nâ�³ FÃ¼hre DBSCAN Clustering durch...")
    clustering = DBSCAN(eps=0.3, min_samples=5, metric='precomputed')
    clusters = clustering.fit_predict(dist_matrix)
    
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    
    print(f"\nğŸ”� Clustering-Ergebnisse:")
    print(f"   - Anzahl Cluster: {n_clusters}")
    print(f"   - Noise-Punkte: {n_noise}")
    print(f"   - GrÃ¶ÃŸter Cluster: {max(Counter(clusters).values())}")
    
    # t-SNE Visualisierung
    if len(sample_seqs) >= 50:
        print("\nâ�³ Erstelle t-SNE Visualisierung...")
        tsne = TSNE(n_components=2, random_state=42, metric='precomputed')
        coords = tsne.fit_transform(dist_matrix)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], 
                            c=clusters, cmap='viridis', 
                            alpha=0.6, s=50)
        plt.colorbar(scatter, label='Cluster')
        plt.title('t-SNE Visualisierung der Sequenz-Ã„hnlichkeit')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        plt.show()
    
    return dist_matrix, clusters

# FÃ¼hre Ã„hnlichkeitsanalyse durch
# Hinweis: Installiere python-Levenshtein falls noch nicht vorhanden
# !pip install python-Levenshtein

try:
    dist_matrix, clusters = analyze_sequence_similarity(sequences, sample_size=500)
except ImportError:
    print("âš ï¸�  python-Levenshtein nicht installiert. Ãœberspringe Ã„hnlichkeitsanalyse.")
    print("   Installiere mit: !pip install python-Levenshtein")

## 12. Positionsspezifische AminosÃ¤ure-Analyse

def analyze_positional_aa_distribution(sequences, max_length=20):
    """
    Analysiert AminosÃ¤ure-Verteilung an jeder Position
    """
    print("\n" + "=" * 80)
    print("POSITIONSSPEZIFISCHE AMINOSÃ„URE-ANALYSE")
    print("=" * 80)
    
    # Filter Sequenzen nach LÃ¤nge
    filtered_seqs = [s for s in sequences if isinstance(s, str) and len(s) == max_length]
    
    if len(filtered_seqs) < 100:
        print(f"âš ï¸�  Zu wenige Sequenzen mit LÃ¤nge {max_length}. Verwende variable LÃ¤ngen.")
        filtered_seqs = [s for s in sequences if isinstance(s, str) and 10 <= len(s) <= 25]
        max_length = max([len(s) for s in filtered_seqs])
    
    print(f"\nğŸ“Š Analysiere {len(filtered_seqs)} Sequenzen mit LÃ¤nge ~{max_length}")
    
    # Erstelle Positions-Matrix
    aa_list = list('ACDEFGHIKLMNPQRSTVWY')
    position_matrix = np.zeros((max_length, len(aa_list)))
    
    for seq in filtered_seqs[:5000]:  # Sample
        for pos, aa in enumerate(seq[:max_length]):
            if aa in aa_list:
                aa_idx = aa_list.index(aa)
                position_matrix[pos, aa_idx] += 1
    
    # Normalisiere
    position_matrix = position_matrix / position_matrix.sum(axis=1, keepdims=True)
    
    # Heatmap
    plt.figure(figsize=(16, 8))
    sns.heatmap(position_matrix.T, 
                xticklabels=range(1, max_length+1),
                yticklabels=aa_list,
                cmap='YlOrRd',
                cbar_kws={'label': 'HÃ¤ufigkeit'})
    plt.xlabel('Position in der Sequenz')
    plt.ylabel('AminosÃ¤ure')
    plt.title('Positionsspezifische AminosÃ¤ure-Verteilung')
    plt.tight_layout()
    plt.show()
    
    # Finde konservierte Positionen
    print("\nğŸ”� Konservierte Positionen (Entropie < 2.0):")
    for pos in range(max_length):
        pos_entropy = entropy(position_matrix[pos] + 1e-10, base=2)
        if pos_entropy < 2.0:
            top_aa = aa_list[np.argmax(position_matrix[pos])]
            top_freq = position_matrix[pos].max()
            print(f"   Position {pos+1}: {top_aa} ({top_freq*100:.1f}%), Entropie: {pos_entropy:.2f}")
    
    return position_matrix

pos_matrix = analyze_positional_aa_distribution(sequences, max_length=15)

## 13. V-J Gene Kombinationsanalyse

def analyze_vj_combinations(dataset_path, min_count=10):
    """
    Analysiert V-J Gen-Kombinationen und deren Assoziation mit Labels
    """
    print("\n" + "=" * 80)
    print("V-J GEN-KOMBINATIONSANALYSE")
    print("=" * 80)
    
    metadata = pd.read_csv(os.path.join(dataset_path, 'metadata.csv'))
    
    vj_combinations_pos = Counter()
    vj_combinations_neg = Counter()
    
    print(f"\nâ�³ Analysiere V-J Kombinationen...")
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
        file_path = os.path.join(dataset_path, row['filename'])
        df = pd.read_csv(file_path, sep='\t')
        
        if 'v_call' in df.columns and 'j_call' in df.columns:
            for _, seq_row in df.iterrows():
                v_gene = seq_row['v_call']
                j_gene = seq_row['j_call']
                
                if pd.notna(v_gene) and pd.notna(j_gene):
                    combination = f"{v_gene}_{j_gene}"
                    
                    if row.get('label_positive', None) == 1:
                        vj_combinations_pos[combination] += 1
                    else:
                        vj_combinations_neg[combination] += 1
    
    # Berechne Enrichment
    all_combinations = set(vj_combinations_pos.keys()) | set(vj_combinations_neg.keys())
    
    enrichment_data = []
    for combo in all_combinations:
        pos_count = vj_combinations_pos[combo]
        neg_count = vj_combinations_neg[combo]
        total = pos_count + neg_count
        
        if total >= min_count:
            enrichment = (pos_count / (pos_count + neg_count)) if total > 0 else 0
            enrichment_data.append({
                'combination': combo,
                'positive_count': pos_count,
                'negative_count': neg_count,
                'total': total,
                'enrichment_score': enrichment
            })
    
    enrichment_df = pd.DataFrame(enrichment_data).sort_values('enrichment_score', ascending=False)
    
    print(f"\nğŸ“Š Gefundene V-J Kombinationen: {len(enrichment_df)}")
    print(f"\nTop 10 in Positiven angereichert:")
    print(enrichment_df.head(10))
    print(f"\nTop 10 in Negativen angereichert:")
    print(enrichment_df.tail(10))
    
    # Visualisierung
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top enriched in positive
    top_pos = enrichment_df.head(15)
    axes[0].barh(range(len(top_pos)), top_pos['enrichment_score'], color='green', alpha=0.7)
    axes[0].set_yticks(range(len(top_pos)))
    axes[0].set_yticklabels(top_pos['combination'], fontsize=8)
    axes[0].set_xlabel('Enrichment Score (Positive)')
    axes[0].set_title('Top 15 V-J Kombinationen in Positiven')
    axes[0].invert_yaxis()
    
    # Top enriched in negative
    top_neg = enrichment_df.tail(15)
    axes[1].barh(range(len(top_neg)), top_neg['enrichment_score'], color='red', alpha=0.7)
    axes[1].set_yticks(range(len(top_neg)))
    axes[1].set_yticklabels(top_neg['combination'], fontsize=8)
    axes[1].set_xlabel('Enrichment Score (Negative)')
    axes[1].set_title('Top 15 V-J Kombinationen in Negativen')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    return enrichment_df

vj_enrichment = analyze_vj_combinations(first_train_dataset, min_count=5)

## 14. Sequenz-Logo (Motif Visualisierung)

def create_sequence_logo(sequences, position_range=(0, 15)):
    """
    Erstellt ein Sequenz-Logo fÃ¼r konservierte Regionen
    """
    print("\n" + "=" * 80)
    print("SEQUENZ-LOGO ANALYSE")
    print("=" * 80)
    
    # Filtere Sequenzen
    start, end = position_range
    filtered_seqs = [s[start:end] for s in sequences 
                     if isinstance(s, str) and len(s) >= end][:1000]
    
    if not filtered_seqs:
        print("âš ï¸�  Keine passenden Sequenzen gefunden")
        return
    
    # Berechne Positional Frequency Matrix
    aa_list = list('ACDEFGHIKLMNPQRSTVWY')
    length = end - start
    pfm = np.zeros((length, len(aa_list)))
    
    for seq in filtered_seqs:
        for pos, aa in enumerate(seq[:length]):
            if aa in aa_list:
                pfm[pos, aa_list.index(aa)] += 1
    
    # Konvertiere zu PWM (Position Weight Matrix)
    pfm = pfm + 0.01  # Pseudocount
    pwm = pfm / pfm.sum(axis=1, keepdims=True)
    
    # Berechne Information Content
    max_entropy = np.log2(len(aa_list))
    ic = np.zeros(length)
    
    for pos in range(length):
        pos_entropy = entropy(pwm[pos], base=2)
        ic[pos] = max_entropy - pos_entropy
    
    # Visualisierung
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # Information Content
    axes[0].bar(range(length), ic, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Information Content (bits)')
    axes[0].set_title('Sequenz-Konservierung pro Position')
    axes[0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Threshold')
    axes[0].legend()
    
    # Heatmap der AminosÃ¤ure-HÃ¤ufigkeiten
    sns.heatmap(pwm.T, ax=axes[1], 
                xticklabels=range(start+1, end+1),
                yticklabels=aa_list,
                cmap='Blues',
                cbar_kws={'label': 'Wahrscheinlichkeit'})
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('AminosÃ¤ure')
    axes[1].set_title('AminosÃ¤ure-Wahrscheinlichkeiten pro Position')
    
    plt.tight_layout()
    plt.show()
    
    # Konsensus-Sequenz
    consensus = ''.join([aa_list[np.argmax(pwm[pos])] for pos in range(length)])
    print(f"\nğŸ§¬ Konsensus-Sequenz: {consensus}")
    print(f"   Mittlerer Information Content: {ic.mean():.2f} bits")
    
    return pwm, ic, consensus

pwm, ic, consensus = create_sequence_logo(sequences, position_range=(0, 15))

## 15. Label-spezifische Sequenzanalyse

def compare_sequences_by_label(dataset_path, sample_size=1000):
    """
    Vergleicht Sequenz-Charakteristiken zwischen positiven und negativen Labels
    """
    print("\n" + "=" * 80)
    print("LABEL-SPEZIFISCHE SEQUENZANALYSE")
    print("=" * 80)
    
    metadata = pd.read_csv(os.path.join(dataset_path, 'metadata.csv'))
    
    pos_sequences = []
    neg_sequences = []
    
    print(f"\nâ�³ Sammle Sequenzen nach Label...")
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
        file_path = os.path.join(dataset_path, row['filename'])
        df = pd.read_csv(file_path, sep='\t')
        
        sequences = df['junction_aa'].dropna().tolist()
        
        if row.get('label_positive', None) == 1:
            pos_sequences.extend(sequences[:sample_size // len(metadata)])
        else:
            neg_sequences.extend(sequences[:sample_size // len(metadata)])
    
    print(f"\nğŸ“Š Gesammelte Sequenzen:")
    print(f"   - Positive: {len(pos_sequences)}")
    print(f"   - Negative: {len(neg_sequences)}")
    
    # Vergleiche LÃ¤ngenverteilungen
    pos_lengths = [len(s) for s in pos_sequences if isinstance(s, str)]
    neg_lengths = [len(s) for s in neg_sequences if isinstance(s, str)]
    
    # Vergleiche AminosÃ¤ure-Kompositionen
    pos_aa_counter = Counter()
    neg_aa_counter = Counter()
    
    for seq in pos_sequences[:5000]:
        if isinstance(seq, str):
            pos_aa_counter.update(seq)
    
    for seq in neg_sequences[:5000]:
        if isinstance(seq, str):
            neg_aa_counter.update(seq)
    
    # Normalisiere
    pos_total = sum(pos_aa_counter.values())
    neg_total = sum(neg_aa_counter.values())
    
    aa_comparison = []
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        pos_freq = pos_aa_counter[aa] / pos_total if pos_total > 0 else 0
        neg_freq = neg_aa_counter[aa] / neg_total if neg_total > 0 else 0
        fold_change = np.log2((pos_freq + 1e-10) / (neg_freq + 1e-10))
        
        aa_comparison.append({
            'aa': aa,
            'pos_freq': pos_freq,
            'neg_freq': neg_freq,
            'fold_change': fold_change
        })
    
    aa_comp_df = pd.DataFrame(aa_comparison).sort_values('fold_change', ascending=False)
    
    # Visualisierung
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. LÃ¤ngenverteilung
    axes[0, 0].hist(pos_lengths, bins=30, alpha=0.6, label='Positive', color='green', density=True)
    axes[0, 0].hist(neg_lengths, bins=30, alpha=0.6, label='Negative', color='red', density=True)
    axes[0, 0].set_xlabel('SequenzlÃ¤nge')
    axes[0, 0].set_ylabel('Dichte')
    axes[0, 0].set_title('LÃ¤ngenverteilung nach Label')
    axes[0, 0].legend()
    
    # 2. Boxplot LÃ¤ngen
    axes[0, 1].boxplot([pos_lengths, neg_lengths], labels=['Positive', 'Negative'])
    axes[0, 1].set_ylabel('SequenzlÃ¤nge')
    axes[0, 1].set_title('LÃ¤ngen-Vergleich')
    
    # 3. AminosÃ¤ure Fold-Change
    colors = ['green' if fc > 0 else 'red' for fc in aa_comp_df['fold_change']]
    axes[1, 0].bar(aa_comp_df['aa'], aa_comp_df['fold_change'], color=colors, alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].set_xlabel('AminosÃ¤ure')
    axes[1, 0].set_ylabel('Log2 Fold Change (Pos/Neg)')
    axes[1, 0].set_title('AminosÃ¤ure-Anreicherung')
    
    # 4. Frequenz-Vergleich
    x = np.arange(len(aa_comp_df))
    width = 0.35
    axes[1, 1].bar(x - width/2, aa_comp_df['pos_freq'], width, label='Positive', color='green', alpha=0.7)
    axes[1, 1].bar(x + width/2, aa_comp_df['neg_freq'], width, label='Negative', color='red', alpha=0.7)
    axes[1, 1].set_xlabel('AminosÃ¤ure')
    axes[1, 1].set_ylabel('Frequenz')
    axes[1, 1].set_title('AminosÃ¤ure-Frequenzen')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(aa_comp_df['aa'])
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Statistischer Test
    from scipy.stats import mannwhitneyu
    
    stat, pval = mannwhitneyu(pos_lengths, neg_lengths)
    print(f"\nğŸ“ˆ Mann-Whitney U Test (LÃ¤ngen):")
    print(f"   - Statistik: {stat:.2f}")
    print(f"   - P-Wert: {pval:.4e}")
    print(f"   - Signifikant: {'Ja' if pval < 0.05 else 'Nein'}")
    
    print(f"\nğŸ§¬ Top 5 in Positiven angereicherte AminosÃ¤uren:")
    print(aa_comp_df.head(5)[['aa', 'fold_change']])
    
    print(f"\nğŸ§¬ Top 5 in Negativen angereicherte AminosÃ¤uren:")
    print(aa_comp_df.tail(5)[['aa', 'fold_change']])
    
    return aa_comp_df, pos_sequences, neg_sequences

aa_comparison, pos_seqs, neg_seqs = compare_sequences_by_label(first_train_dataset, sample_size=2000)

## 16. K-mer Enrichment Analyse

def kmer_enrichment_analysis(pos_sequences, neg_sequences, k=4, top_n=30):
    """
    Findet K-mere, die in einer Klasse angereichert sind
    """
    print("\n" + "=" * 80)
    print(f"{k}-MER ENRICHMENT ANALYSE")
    print("=" * 80)
    
    pos_kmers = Counter()
    neg_kmers = Counter()
    
    print(f"\nâ�³ Extrahiere {k}-mere...")
    
    # Positive Sequenzen
    for seq in pos_sequences[:5000]:
        if isinstance(seq, str):
            for i in range(len(seq) - k + 1):
                pos_kmers[seq[i:i+k]] += 1
    
    # Negative Sequenzen
    for seq in neg_sequences[:5000]:
        if isinstance(seq, str):
            for i in range(len(seq) - k + 1):
                neg_kmers[seq[i:i+k]] += 1
    
    # Berechne Enrichment
    all_kmers = set(pos_kmers.keys()) | set(neg_kmers.keys())
    
    enrichment_data = []
    for kmer in all_kmers:
        pos_count = pos_kmers[kmer]
        neg_count = neg_kmers[kmer]
        total = pos_count + neg_count
        
        if total >= 10:  # Minimum count threshold
            # Berechne Odds Ratio
            pos_total = sum(pos_kmers.values())
            neg_total = sum(neg_kmers.values())
            
            pos_freq = pos_count / pos_total
            neg_freq = neg_count / neg_total
            
            if neg_freq > 0:
                odds_ratio = pos_freq / neg_freq
                log_odds = np.log2(odds_ratio)
            else:
                log_odds = 10  # Sehr hoher Wert wenn nur in pos
            
            enrichment_data.append({
                'kmer': kmer,
                'pos_count': pos_count,
                'neg_count': neg_count,
                'log2_odds_ratio': log_odds,
                'pos_freq': pos_freq,
                'neg_freq': neg_freq
            })
    
    enrich_df = pd.DataFrame(enrichment_data).sort_values('log2_odds_ratio', ascending=False)
    
    print(f"\nğŸ“Š Analysierte {k}-mere: {len(enrich_df)}")
    
    print(f"\nTop {top_n//2} in Positiven angereichert:")
    print(enrich_df.head(top_n//2)[['kmer', 'log2_odds_ratio', 'pos_count', 'neg_count']])
    
    print(f"\nTop {top_n//2} in Negativen angereichert:")
    print(enrich_df.tail(top_n//2)[['kmer', 'log2_odds_ratio', 'pos_count', 'neg_count']])
    
    # Visualisierung
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Volcano Plot
    axes[0].scatter(enrich_df['log2_odds_ratio'], 
                   -np.log10(1/(enrich_df['pos_count'] + enrich_df['neg_count'])),
                   alpha=0.5, s=20)
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Log2 Odds Ratio')
    axes[0].set_ylabel('-Log10(1/Total Count)')
    axes[0].set_title(f'{k}-mer Enrichment Volcano Plot')
    
    # Top enriched
    top_pos = enrich_df.head(15)
    top_neg = enrich_df.tail(15)
    combined = pd.concat([top_pos, top_neg])
    
    colors = ['green' if x > 0 else 'red' for x in combined['log2_odds_ratio']]
    axes[1].barh(range(len(combined)), combined['log2_odds_ratio'], color=colors, alpha=0.7)
    axes[1].set_yticks(range(len(combined)))
    axes[1].set_yticklabels(combined['kmer'], fontsize=8)
    axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel('Log2 Odds Ratio')
    axes[1].set_title(f'Top {k}-mere nach Enrichment')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    return enrich_df

# FÃ¼hre fÃ¼r verschiedene k-Werte durch
for k_val in [3, 4, 5]:
    kmer_enrich = kmer_enrichment_analysis(pos_seqs, neg_seqs, k=k_val, top_n=20)

## 17. HydrophobizitÃ¤ts- und Ladungsanalyse

def analyze_physicochemical_properties(sequences, labels=None):
    """
    Analysiert physikochemische Eigenschaften der Sequenzen
    """
    print("\n" + "=" * 80)
    print("PHYSIKOCHEMISCHE EIGENSCHAFTEN")
    print("=" * 80)
    
    # AminosÃ¤ure-Eigenschaften
    hydrophobicity = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
    }
    
    charge = {
        'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
        'G': 0, 'H': 0.5, 'I': 0, 'K': 1, 'L': 0,
        'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
        'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
    }
    
    properties = []
    
    print(f"\nâ�³ Berechne Eigenschaften fÃ¼r {len(sequences)} Sequenzen...")
    
    for seq in sequences[:5000]:
        if isinstance(seq, str) and len(seq) > 0:
            # HydrophobizitÃ¤t
            hydro = np.mean([hydrophobicity.get(aa, 0) for aa in seq])
            
            # Ladung
            net_charge = sum([charge.get(aa, 0) for aa in seq])
            
            # AromatizitÃ¤t (F, W, Y)
            aromatic = sum([1 for aa in seq if aa in 'FWY']) / len(seq)
            
            # PolaritÃ¤t
            polar = sum([1 for aa in seq if aa in 'STNQ']) / len(seq)
            
            # Basisch
            basic = sum([1 for aa in seq if aa in 'KRH']) / len(seq)
            
            # Sauer
            acidic = sum([1 for aa in seq if aa in 'DE']) / len(seq)
            
            properties.append({
                'hydrophobicity': hydro,
                'net_charge': net_charge,
                'aromatic_content': aromatic,
                'polar_content': polar,
                'basic_content': basic,
                'acidic_content': acidic,
                'length': len(seq)
            })
    
    prop_df = pd.DataFrame(properties)
    
    print(f"\nğŸ“Š Physikochemische Statistiken:")
    print(prop_df.describe())
    
    # Visualisierung
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    properties_to_plot = ['hydrophobicity', 'net_charge', 'aromatic_content', 
                          'polar_content', 'basic_content', 'acidic_content']
    
    for idx, prop in enumerate(properties_to_plot):
        axes[idx].hist(prop_df[prop], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[idx].axvline(prop_df[prop].mean(), color='red', linestyle='--', 
                         label=f'Mean: {prop_df[prop].mean():.3f}')
        axes[idx].set_xlabel(prop.replace('_', ' ').title())
        axes[idx].set_ylabel('HÃ¤ufigkeit')
        axes[idx].set_title(f'Verteilung: {prop.replace("_", " ").title()}')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Korrelationsanalyse
    corr_matrix = prop_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1)
    plt.title('Korrelation zwischen physikochemischen Eigenschaften')
    plt.tight_layout()
    plt.show()
    
    return prop_df

prop_df = analyze_physicochemical_properties(sequences)

## 18. Speichere umfassende EDA-Ergebnisse

def save_comprehensive_eda_results(output_dir='/kaggle/working/eda_results'):
    """
    Speichert alle EDA-Ergebnisse
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("SPEICHERE EDA-ERGEBNISSE")
    print("=" * 80)
    
    # Speichere DataFrames
    results_to_save = {
        'dataset_summary.csv': summary,
        'repertoire_stats.csv': rep_stats,
        'vj_enrichment.csv': vj_enrichment,
        'aa_comparison.csv': aa_comparison,
        'kmer_enrichment.csv': kmer_enrich,
        'physicochemical_properties.csv': prop_df,
        'diversity_metrics.csv': diversity_df
    }
    
    for filename, df in results_to_save.items():
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"âœ… Gespeichert: {filepath}")
    
    print(f"\nâœ… Alle EDA-Ergebnisse gespeichert in: {output_dir}")

save_comprehensive_eda_results()

print("\n" + "=" * 80)
print("ğŸ�‰ VOLLSTÃ„NDIGE EDA ABGESCHLOSSEN!")
print("=" * 80)
print("\nğŸ“‹ Zusammenfassung der durchgefÃ¼hrten Analysen:")
print("   1. âœ… Datenstruktur-Ãœberblick")
print("   2. âœ… Einzeldatensatz-Analyse")
print("   3. âœ… CDR3-Sequenz-Analyse")
print("   4. âœ… Repertoire-Level-Analyse")
print("   5. âœ… K-mer-Analyse")
print("   6. âœ… Train/Test-Vergleich")
print("   7. âœ… Dataset-Zusammenfassung")
print("   8. âœ… DiversitÃ¤ts-Metriken")
print("   9. âœ… Motif-Suche")
print("   10. âœ… Sequenz-Ã„hnlichkeit")
print("   11. âœ… Positionsspezifische Analyse")
print("   12. âœ… V-J Gen-Kombinationen")
print("   13. âœ… Sequenz-Logo")
print("   14. âœ… Label-spezifische Analyse")
print("   15. âœ… K-mer Enrichment")
print("   16. âœ… Physikochemische Eigenschaften")
print("   17. âœ… Ergebnisse gespeichert")



## 20. Erstelle einen HTML-Report

def create_eda_html_report(output_path='/kaggle/working/eda_report.html'):
    """
    Erstellt einen HTML-Report mit allen wichtigen Erkenntnissen
    """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>EDA Report - Adaptive Immune Profiling Challenge</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            .metric {{ background-color: white; padding: 20px; margin: 10px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
            .metric-label {{ color: #7f8c8d; font-size: 14px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background-color: white; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            .highlight {{ background-color: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>ğŸ§¬ Explorative Datenanalyse - Adaptive Immune Profiling Challenge 2025</h1>
        
        <div class="metric">
            <div class="metric-label">Anzahl Training Datasets</div>
            <div class="metric-value">{len(train_datasets)}</div>
        </div>
        
        <div class="metric">
            <div class="metric-label">Anzahl Test Datasets</div>
            <div class="metric-value">{len(test_datasets)}</div>
        </div>
        
        <div class="metric">
            <div class="metric-label">Durchschnittliche SequenzlÃ¤nge</div>
            <div class="metric-value">{np.mean(lengths):.1f} Â± {np.std(lengths):.1f}</div>
        </div>
        
        <h2>ğŸ“Š Wichtigste Erkenntnisse</h2>
        
        <div class="highlight">
            <strong>1. SequenzlÃ¤ngen:</strong> Die meisten CDR3-Sequenzen haben eine LÃ¤nge zwischen 
            {int(np.percentile(lengths, 25))} und {int(np.percentile(lengths, 75))} AminosÃ¤uren.
        </div>
        
        <div class="highlight">
            <strong>2. AminosÃ¤ure-Komposition:</strong> Die hÃ¤ufigsten AminosÃ¤uren sind 
            {', '.join([aa for aa, _ in Counter(''.join(str(s) for s in sequences[:1000])).most_common(5)])}.
        </div>
        
        <div class="highlight">
            <strong>3. Class Balance:</strong> Die Datasets zeigen unterschiedliche Balance-VerhÃ¤ltnisse 
            zwischen positiven und negativen Labels.
        </div>
        
        <h2>ğŸ“ˆ Empfehlungen fÃ¼r Modellierung</h2>
        <ul>
            <li>âœ… Verwenden Sie K-mer Features (k=3,4,5) als Basis-Features</li>
            <li>âœ… BerÃ¼cksichtigen Sie V-J Gen-Kombinationen</li>
            <li>âœ… Nutzen Sie Repertoire-Level Aggregationen</li>
            <li>âœ… Implementieren Sie Class-Balancing Strategien</li>
            <li>âœ… Extrahieren Sie physikochemische Eigenschaften</li>
        </ul>
        
        <h2>ğŸ“� Generierte Dateien</h2>
        <ul>
            <li>dataset_summary.csv</li>
            <li>repertoire_stats.csv</li>
            <li>vj_enrichment.csv</li>
            <li>aa_comparison.csv</li>
            <li>kmer_enrichment.csv</li>
            <li>physicochemical_properties.csv</li>
        </ul>
        
        <p style="margin-top: 50px; color: #7f8c8d; font-size: 12px;">
            Generiert am: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nâœ… HTML-Report erstellt: {output_path}")

create_eda_html_report()



## Optimized Feature Engineering with Dimensionality Reduction

class OptimizedFeatureEngineering(FeatureEngineering):
    """
    Optimized feature engineering with dimensionality reduction
    """
    
    def __init__(self, kmer_sizes: List[int] = [3, 4], 
                 use_vj_features: bool = True,
                 use_physicochemical: bool = True,
                 use_diversity: bool = True,
                 max_kmer_features: int = 1000,
                 use_pca: bool = False,
                 pca_components: int = 100):
        """
        Initialize optimized feature engineering
        
        Args:
            kmer_sizes: List of k-mer sizes (reduced to [3,4] for speed)
            use_vj_features: Whether to include V-J gene features
            use_physicochemical: Whether to include physicochemical properties
            use_diversity: Whether to include diversity metrics
            max_kmer_features: Maximum number of k-mer features to keep
            use_pca: Whether to use PCA for dimensionality reduction
            pca_components: Number of PCA components
        """
        super().__init__(kmer_sizes, use_vj_features, use_physicochemical, use_diversity)
        self.max_kmer_features = max_kmer_features
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca = None
        self.top_kmers = {}
        self.top_vj = {}
    
    def select_top_features(self, features_df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        """
        Select top features based on variance and correlation with labels
        
        Args:
            features_df: Features DataFrame
            labels: Target labels
            
        Returns:
            Reduced features DataFrame
        """
        print("Selecting top features...")
        
        # Separate feature types
        kmer_cols = [col for col in features_df.columns if col.startswith('kmer_')]
        vj_cols = [col for col in features_df.columns if col.startswith('vj_')]
        other_cols = [col for col in features_df.columns if not (col.startswith('kmer_') or col.startswith('vj_'))]
        
        selected_cols = other_cols.copy()  # Always keep physicochemical and diversity features
        
        # Select top k-mer features based on variance
        if kmer_cols:
            kmer_df = features_df[kmer_cols]
            variances = kmer_df.var()
            top_kmer_cols = variances.nlargest(min(self.max_kmer_features, len(kmer_cols))).index.tolist()
            selected_cols.extend(top_kmer_cols)
            
            # Store for transform
            self.top_kmers = {col: True for col in top_kmer_cols}
        
        # Select top V-J features based on variance
        if vj_cols:
            vj_df = features_df[vj_cols]
            variances = vj_df.var()
            top_vj_cols = variances.nlargest(min(500, len(vj_cols))).index.tolist()
            selected_cols.extend(top_vj_cols)
            
            # Store for transform
            self.top_vj = {col: True for col in top_vj_cols}
        
        reduced_df = features_df[selected_cols]
        
        print(f"Reduced features from {len(features_df.columns)} to {len(selected_cols)}")
        
        return reduced_df
    
    def fit_transform(self, data_dir: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit feature engineering pipeline and transform training data
        
        Args:
            data_dir: Path to training data directory
            
        Returns:
            Tuple of (features_df, labels)
        """
        print("Extracting features from training data (optimized)...")
        
        metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
        
        all_features = []
        all_labels = []
        repertoire_ids = []
        
        # Use parallel processing for feature extraction
        from joblib import Parallel, delayed
        
        def process_repertoire(row):
            """Process single repertoire"""
            file_path = os.path.join(data_dir, row['filename'])
            repertoire_df = pd.read_csv(file_path, sep='\t')
            features = self.extract_repertoire_features(repertoire_df)
            return features, row['label_positive'], row['repertoire_id']
        
        # Parallel processing
        results = Parallel(n_jobs=self.n_jobs if hasattr(self, 'n_jobs') else 4)(
            delayed(process_repertoire)(row) 
            for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing repertoires")
        )
        
        # Unpack results
        for features, label, rep_id in results:
            all_features.append(features)
            all_labels.append(label)
            repertoire_ids.append(rep_id)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        features_df.index = repertoire_ids
        features_df = features_df.fillna(0)
        
        labels = pd.Series(all_labels, index=repertoire_ids)
        
        # Feature selection
        features_df = self.select_top_features(features_df, labels)
        
        # Store feature names
        self.feature_names = features_df.columns.tolist()
        
        # Apply PCA if requested
        if self.use_pca:
            print(f"Applying PCA (n_components={self.pca_components})...")
            self.pca = PCA(n_components=min(self.pca_components, len(features_df.columns)))
            features_pca = self.pca.fit_transform(features_df)
            
            # Create new DataFrame with PCA components
            pca_cols = [f'pca_{i}' for i in range(features_pca.shape[1])]
            features_df = pd.DataFrame(features_pca, columns=pca_cols, index=features_df.index)
            self.feature_names = pca_cols
            
            print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        else:
            # Fit scaler
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features_df)
            features_df = pd.DataFrame(features_scaled, columns=features_df.columns, index=features_df.index)
        
        print(f"Final feature count: {len(self.feature_names)}")
        
        return features_df, labels
    
    def transform(self, data_dir: str) -> pd.DataFrame:
        """
        Transform test data using fitted pipeline (optimized)
        
        Args:
            data_dir: Path to test data directory
            
        Returns:
            Features DataFrame
        """
        print(f"Extracting features from test data: {data_dir}")
        
        metadata_path = os.path.join(data_dir, 'metadata.csv')
        
        all_features = []
        repertoire_ids = []
        
        # Use parallel processing
        from joblib import Parallel, delayed
        
        def process_repertoire(file_path, rep_id):
            """Process single repertoire"""
            repertoire_df = pd.read_csv(file_path, sep='\t')
            features = self.extract_repertoire_features(repertoire_df)
            return features, rep_id
        
        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
            
            results = Parallel(n_jobs=self.n_jobs if hasattr(self, 'n_jobs') else 4)(
                delayed(process_repertoire)(os.path.join(data_dir, row['filename']), row['repertoire_id'])
                for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing repertoires")
            )
        else:
            tsv_files = glob.glob(os.path.join(data_dir, '*.tsv'))
            
            results = Parallel(n_jobs=self.n_jobs if hasattr(self, 'n_jobs') else 4)(
                delayed(process_repertoire)(file_path, os.path.basename(file_path).replace('.tsv', ''))
                for file_path in tqdm(tsv_files, desc="Processing repertoires")
            )
        
        # Unpack results
        for features, rep_id in results:
            all_features.append(features)
            repertoire_ids.append(rep_id)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        features_df.index = repertoire_ids
        
        # Select only top features
        for col in features_df.columns:
            if col.startswith('kmer_') and col not in self.top_kmers:
                features_df = features_df.drop(columns=[col])
            elif col.startswith('vj_') and col not in self.top_vj:
                features_df = features_df.drop(columns=[col])
        
        # Align with training features
        for col in self.feature_names:
            if col not in features_df.columns and not col.startswith('pca_'):
                features_df[col] = 0
        
        if not self.use_pca:
            features_df = features_df[self.feature_names]
        
        features_df = features_df.fillna(0)
        
        # Apply transformation
        if self.use_pca:
            # First scale, then PCA
            temp_scaler = StandardScaler()
            features_scaled = temp_scaler.fit_transform(features_df)
            features_pca = self.pca.transform(features_scaled)
            features_df = pd.DataFrame(features_pca, columns=self.feature_names, index=features_df.index)
        else:
            features_scaled = self.scaler.transform(features_df)
            features_df = pd.DataFrame(features_scaled, columns=features_df.columns, index=features_df.index)
        
        return features_df

## Optimized Model Trainer with Early Stopping and Reduced CV

class OptimizedModelTrainer(ModelTrainer):
    """
    Optimized model trainer with faster training
    """
    
    def __init__(self, model_type: str = 'xgboost', n_jobs: int = 4, device: str = 'cpu',
                 fast_mode: bool = True):
        """
        Initialize optimized model trainer
        
        Args:
            model_type: Type of model
            n_jobs: Number of parallel jobs
            device: Device to use
            fast_mode: Whether to use faster hyperparameters
        """
        super().__init__(model_type, n_jobs, device)
        self.fast_mode = fast_mode
    
    def create_model(self):
        """Create model with optimized hyperparameters for speed"""
        
        if self.model_type == 'xgboost':
            if self.fast_mode:
                # Faster XGBoost settings
                self.model = xgb.XGBClassifier(
                    n_estimators=100,  # Reduced from 200
                    max_depth=4,       # Reduced from 6
                    learning_rate=0.1,  # Increased from 0.05
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=self.n_jobs,
                    tree_method='hist',  # Faster than exact
                    max_bin=256,  # Reduced from default
                    eval_metric='logloss'
                )
            else:
                # Standard settings
                self.model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=self.n_jobs,
                    tree_method='hist',
                    eval_metric='logloss'
                )
        
        elif self.model_type == 'lightgbm':
            if self.fast_mode:
                # Faster LightGBM settings
                self.model = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=self.n_jobs,
                    num_leaves=31,  # Reduced
                    verbose=-1
                )
            else:
                self.model = lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=self.n_jobs,
                    verbose=-1
                )
        
        elif self.model_type == 'rf':
            if self.fast_mode:
                # Faster Random Forest
                self.model = RandomForestClassifier(
                    n_estimators=50,  # Reduced from 200
                    max_depth=8,      # Reduced from 10
                    min_samples_split=10,  # Increased from 5
                    min_samples_leaf=4,    # Increased from 2
                    random_state=42,
                    n_jobs=self.n_jobs,
                    max_features='sqrt'  # Faster than 'auto'
                )
            else:
                self.model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=self.n_jobs
                )
        
        elif self.model_type == 'lr':
            # Logistic Regression is already fast
            self.model = LogisticRegression(
                C=1.0,
                max_iter=500,  # Reduced from 1000
                random_state=42,
                n_jobs=self.n_jobs,
                solver='saga'  # Faster for large datasets
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 3):
        """
        Perform cross-validation with reduced folds for speed
        
        Args:
            X: Features
            y: Labels
            cv: Number of folds (reduced to 3 for speed)
            
        Returns:
            Cross-validation scores
        """
        print(f"Performing {cv}-fold cross-validation...")
        
        if self.model is None:
            self.create_model()
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        scores = cross_val_score(
            self.model, X, y,
            cv=skf,
            scoring='roc_auc',
            n_jobs=1,  # Set to 1 to avoid nested parallelism
            verbose=0
        )
        
        print(f"Cross-validation AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return scores

## Fast Important Sequences Identifier

class FastImportantSequencesIdentifier(ImportantSequencesIdentifier):
    """
    Faster version of important sequences identifier
    """
    
    def identify_frequency_based(self, data_dir: str, top_k: int = 50000, 
                                 sample_size: int = None) -> pd.DataFrame:
        """
        Identify important sequences with optional sampling for speed
        
        Args:
            data_dir: Path to training data directory
            top_k: Number of top sequences to return
            sample_size: Maximum number of sequences to process per repertoire (None = all)
            
        Returns:
            DataFrame with important sequences
        """
        print("Identifying important sequences (frequency-based, optimized)...")
        
        metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
        
        # Collect sequences by label
        pos_sequences = Counter()
        neg_sequences = Counter()
        
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing repertoires"):
            file_path = os.path.join(data_dir, row['filename'])
            repertoire_df = pd.read_csv(file_path, sep='\t')
            
            sequences = repertoire_df['junction_aa'].dropna().tolist()
            
            # Sample if requested
            if sample_size and len(sequences) > sample_size:
                sequences = np.random.choice(sequences, sample_size, replace=False).tolist()
            
            if row['label_positive'] == 1:
                pos_sequences.update(sequences)
            else:
                neg_sequences.update(sequences)
        
        # Calculate enrichment scores (vectorized)
        all_sequences = set(pos_sequences.keys()) | set(neg_sequences.keys())
        
        # Pre-calculate totals
        pos_total = sum(pos_sequences.values())
        neg_total = sum(neg_sequences.values())
        
        sequence_scores = []
        
        # Batch processing
        for seq in tqdm(all_sequences, desc="Calculating enrichment scores"):
            pos_count = pos_sequences[seq]
            neg_count = neg_sequences[seq]
            total = pos_count + neg_count
            
            if total >= 5:  # Minimum count threshold
                pos_freq = pos_count / pos_total if pos_total > 0 else 0
                neg_freq = neg_count / neg_total if neg_total > 0 else 0
                
                if neg_freq > 0:
                    log_odds = np.log2((pos_freq + 1e-10) / (neg_freq + 1e-10))
                else:
                    log_odds = 10
                
                importance = abs(log_odds) * np.log(total + 1)
                
                sequence_scores.append({
                    'junction_aa': seq,
                    'pos_count': pos_count,
                    'neg_count': neg_count,
                    'log_odds_ratio': log_odds,
                    'importance_score': importance
                })
        
        # Sort and take top k
        scores_df = pd.DataFrame(sequence_scores).sort_values('importance_score', ascending=False)
        top_sequences = scores_df.head(top_k)
        
        # Add placeholder V and J genes
        top_sequences['v_call'] = 'TRBV20-1'
        top_sequences['j_call'] = 'TRBJ2-7'
        
        print(f"Identified {len(top_sequences)} important sequences")
        
        return top_sequences

## Updated ImmuneStatePredictor with Optimizations

class OptimizedImmuneStatePredictor(ImmuneStatePredictor):
    """
    Optimized immune state predictor with faster training
    """
    
    def __init__(self, n_jobs: int = 1, device: str = 'cpu', 
                 model_type: str = 'xgboost',
                 kmer_sizes: List[int] = [3, 4],  # Reduced from [3,4,5]
                 importance_method: str = 'frequency_based',
                 fast_mode: bool = True,
                 max_kmer_features: int = 1000,
                 use_pca: bool = False,
                 cv_folds: int = 3,  # Reduced from 5
                 **kwargs):
        """
        Initialize optimized predictor
        
        Args:
            n_jobs: Number of CPU cores
            device: Device to use
            model_type: Type of model to use
            kmer_sizes: K-mer sizes for feature extraction
            importance_method: Method for identifying important sequences
            fast_mode: Whether to use faster settings
            max_kmer_features: Maximum number of k-mer features
            use_pca: Whether to use PCA
            cv_folds: Number of cross-validation folds
        """
        total_cores = os.cpu_count()
        if n_jobs == -1:
            self.n_jobs = total_cores
        else:
            self.n_jobs = min(n_jobs, total_cores)
        
        self.device = device
        self.model_type = model_type
        self.kmer_sizes = kmer_sizes
        self.importance_method = importance_method
        self.fast_mode = fast_mode
        self.cv_folds = cv_folds
        
        # Initialize optimized components
        self.feature_eng = OptimizedFeatureEngineering(
            kmer_sizes=kmer_sizes,
            use_vj_features=True,
            use_physicochemical=True,
            use_diversity=True,
            max_kmer_features=max_kmer_features,
            use_pca=use_pca
        )
        self.feature_eng.n_jobs = self.n_jobs
        
        self.model_trainer = OptimizedModelTrainer(
            model_type=model_type,
            n_jobs=self.n_jobs,
            device=self.device,
            fast_mode=fast_mode
        )
        
        self.seq_identifier = FastImportantSequencesIdentifier(
            method=importance_method
        )
        
        self.important_sequences_ = None
        self.model = None
    
    def fit(self, train_dir_path: str):
        """
        Train the model on training data (optimized)
        
        Args:
            train_dir_path: Path to training data directory
            
        Returns:
            self
        """
        print(f"\n{'='*80}")
        print(f"TRAINING IMMUNE STATE PREDICTOR (OPTIMIZED)")
        print(f"{'='*80}\n")
        
        # Step 1: Feature extraction
        print("Step 1: Feature Extraction (Optimized)")
        start_time = time.time()
        X_train, y_train = self.feature_eng.fit_transform(train_dir_path)
        print(f"Feature extraction time: {time.time() - start_time:.2f}s")
        
        # Step 2: Model training
        print("\nStep 2: Model Training (Optimized)")
        start_time = time.time()
        self.model_trainer.train(X_train, y_train)
        self.model = self.model_trainer.model
        print(f"Training time: {time.time() - start_time:.2f}s")
        
        # Step 3: Cross-validation (optional, can be skipped for speed)
        if self.cv_folds > 0:
            print(f"\nStep 3: Cross-Validation ({self.cv_folds} folds)")
            start_time = time.time()
            cv_scores = self.model_trainer.cross_validate(X_train, y_train, cv=self.cv_folds)
            print(f"Cross-validation time: {time.time() - start_time:.2f}s")
        
        # Step 4: Identify important sequences
        print("\nStep 4: Identifying Important Sequences (Optimized)")
        start_time = time.time()
        self.important_sequences_ = self.seq_identifier.identify_frequency_based(
            train_dir_path,
            top_k=50000,
            sample_size=5000 if self.fast_mode else None  # Sample for speed
        )
        print(f"Sequence identification time: {time.time() - start_time:.2f}s")
        
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE!")
        print(f"{'='*80}\n")
        
        return self

## Usage Example with Timing

import time

print("\n" + "="*80)
print("STARTING OPTIMIZED IMMUNE STATE PREDICTION PIPELINE")
print("="*80 + "\n")

# Process each dataset pair
for train_dir, test_dirs in train_test_dataset_pairs:
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(train_dir)}")
    print(f"{'='*80}\n")
    
    total_start = time.time()
    
    # Initialize optimized predictor
    predictor = OptimizedImmuneStatePredictor(
        n_jobs=-1,  # Use all available cores
        device='cpu',
        model_type='xgboost',  # Or 'lightgbm' for even faster training
        kmer_sizes=[3, 4],  # Reduced from [3,4,5]
        importance_method='frequency_based',
        fast_mode=True,  # Enable fast mode
        max_kmer_features=1000,  # Limit k-mer features
        use_pca=False,  # Set to True for even more speed (but may reduce accuracy)
        cv_folds=3  # Reduced from 5
    )
    
    # Train
    predictor.fit(train_dir)
    
    # Save model
    model_save_dir = os.path.join(results_dir, 'models', os.path.basename(train_dir))
    predictor.save(model_save_dir)
    
    # Predict on test sets
    for test_dir in test_dirs:
        predictions = predictor.predict_proba(test_dir)
        
        # Save predictions
        pred_path = os.path.join(
            results_dir,
            f"{os.path.basename(train_dir)}_test_predictions.tsv"
        )
        save_tsv(predictions, pred_path)
    
    # Save important sequences
    important_seqs = predictor.identify_associated_sequences(
        dataset_name=os.path.basename(train_dir),
        top_k=50000
    )
    
    seq_path = os.path.join(
        results_dir,
        f"{os.path.basename(train_dir)}_important_sequences.tsv"
    )
    save_tsv(important_seqs, seq_path)
    
    total_time = time.time() - total_start
    print(f"\nTotal processing time for {os.path.basename(train_dir)}: {total_time:.2f}s ({total_time/60:.2f} min)")

# Concatenate all outputs
concatenate_output_files(out_dir=results_dir)

print("\n" + "="*80)
print("OPTIMIZED PIPELINE COMPLETE!")
print("="*80)








