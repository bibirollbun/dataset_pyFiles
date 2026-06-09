# heyy


!pip install -q /kaggle/input/rna-wheels/wheels/biopython-1.85-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


!ls "/kaggle/input/stanford-rna-3d-folding"


import os
from tqdm import tqdm
from collections import Counter

files = os.listdir("/kaggle/input/stanford-rna-3d-folding/PDB_RNA")
print(f"Total number of files: {len(files)}")

# Get file extensions
extensions = [os.path.splitext(file)[1] for file in files if os.path.splitext(file)[1]]

# Count occurrences of each extension
extension_counts = Counter(extensions)

print(f"File types and their counts:")
for ext, count in sorted(extension_counts.items()):
    print(f"  {ext}: {count}")


import pandas as pd
import os

# Find the CSV file name
files = os.listdir("/kaggle/input/stanford-rna-3d-folding/PDB_RNA")
csv_files = [f for f in files if f.endswith('.csv')]
print(f"CSV file(s): {csv_files}")

# Read the CSV file
csv_file_path = f"/kaggle/input/stanford-rna-3d-folding/PDB_RNA/{csv_files[0]}"

# Skip the problematic header lines
df = pd.read_csv(csv_file_path, on_bad_lines='skip')
print(f"CSV file shape: {df.shape}")

print(f"Column names: {list(df.columns)}")
print(df.head())


# Check unique counts for Entry ID
print(f"Total rows: {len(df)}")
print(f"Unique Entry IDs: {df['Entry ID'].nunique()}")
print(f"Duplicate Entry IDs: {len(df) - df['Entry ID'].nunique()}")

# Convert Release Date to datetime for better analysis
df['Release Date'] = pd.to_datetime(df['Release Date'])

# Get date range
print(f"\nRelease Date range:")
print(f"Earliest date: {df['Release Date'].min()}")
print(f"Latest date: {df['Release Date'].max()}")
print(f"Date span: {(df['Release Date'].max() - df['Release Date'].min()).days} days")

# Check unique release dates
print(f"\nUnique Release Dates: {df['Release Date'].nunique()}")

# Show some examples of duplicates if any exist
if len(df) > df['Entry ID'].nunique():
    print(f"\nExamples of duplicate Entry IDs:")
    duplicates = df[df['Entry ID'].duplicated(keep=False)].sort_values('Entry ID')
    print(duplicates.head(10))





import os

# Find the FASTA file
files = os.listdir("/kaggle/input/stanford-rna-3d-folding/PDB_RNA")
fasta_files = [f for f in files if f.endswith('.fasta')]
print(f"FASTA file(s): {fasta_files}")

# Read and examine the FASTA file
fasta_file_path = f"/kaggle/input/stanford-rna-3d-folding/PDB_RNA/{fasta_files[0]}"

# Read the file and look at its structure
with open(fasta_file_path, 'r') as f:
    lines = f.readlines()

print(f"Total lines in FASTA file: {len(lines)}")
print(f"\nFirst 20 lines:")
for i, line in enumerate(lines[:20]):
    print(f"Line {i+1}: {repr(line)}")


# Parse FASTA file to count sequences
sequences = []
current_seq = ""
headers = []

with open(fasta_file_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('>'):  # Header line
            if current_seq:  # Save previous sequence
                sequences.append(current_seq)
                current_seq = ""
            headers.append(line)
        else:  # Sequence line
            current_seq += line
    
    # Don't forget the last sequence
    if current_seq:
        sequences.append(current_seq)

print(f"\nFASTA file summary:")
print(f"Number of sequences: {len(sequences)}")
print(f"Number of headers: {len(headers)}")

if sequences:
    seq_lengths = [len(seq) for seq in sequences]
    print(f"Sequence lengths - Min: {min(seq_lengths)}, Max: {max(seq_lengths)}, Average: {sum(seq_lengths)/len(seq_lengths):.1f}")
    
    print(f"\nFirst few headers:")
    for i, header in enumerate(headers[:5]):
        print(f"  {header}")
    
    print(f"\nFirst sequence (first 100 chars):")
    print(f"  {sequences[0][:100]}...")


# Continue with the parsing code from before to see the complete statistics
print(f"\nFASTA file summary:")
print(f"Number of sequences: {len(sequences)}")
print(f"Sequence lengths - Min: {min(seq_lengths)}, Max: {max(seq_lengths)}")

# Count DNA vs RNA vs mixed
dna_count = sum(1 for h in headers if 'DNA' in h and 'RNA' not in h)
rna_count = sum(1 for h in headers if 'RNA' in h and 'DNA' not in h)
mixed_count = sum(1 for h in headers if 'DNA' in h and 'RNA' in h)
print(f"DNA sequences: {dna_count}")
print(f"RNA sequences: {rna_count}")  
print(f"Mixed DNA/RNA: {mixed_count}")

# Count unique PDB IDs
pdb_ids = [h.split('_')[0][1:] for h in headers]  # Remove '>' and chain part
unique_pdbs = len(set(pdb_ids))
print(f"Unique PDB structures: {unique_pdbs}")


# Count RNA sequences with less than 1000 nucleotides
rna_short_count = 0
rna_lengths = []

for i, header in enumerate(headers):
   if 'RNA' in header and 'DNA' not in header:  # Pure RNA sequences only
       seq_length = len(sequences[i])
       rna_lengths.append(seq_length)
       if seq_length < 1000:
           rna_short_count += 1

print(f"RNA sequences with less than 1000 nucleotides: {rna_short_count}")
print(f"Total RNA sequences: {len(rna_lengths)}")
print(f"Percentage of RNA sequences < 1000 nt: {rna_short_count/len(rna_lengths)*100:.1f}%")

# Additional statistics for RNA sequences
if rna_lengths:
   print(f"\nRNA sequence length statistics:")
   print(f"Min length: {min(rna_lengths)}")
   print(f"Max length: {max(rna_lengths)}")
   print(f"Average length: {sum(rna_lengths)/len(rna_lengths):.1f}")
   
   # Length distribution
   length_ranges = [
       (0, 100, "1-100"),
       (100, 500, "100-500"), 
       (500, 1000, "500-1000"),
       (1000, 2000, "1000-2000"),
       (2000, float('inf'), "2000+")
   ]
   
   print(f"\nRNA sequence length distribution:")
   for min_len, max_len, label in length_ranges:
       count = sum(1 for length in rna_lengths if min_len < length <= max_len)
       print(f"  {label} nt: {count}")


train_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")


train_sequences.head()


train_sequences_v2 = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.v2.csv")


train_sequences_v2.head()


# Check for target_ids in v1 that are not in v2
v1_only_targets = set(train_sequences['target_id']) - set(train_sequences_v2['target_id'])
print(f"Target IDs in v1 but not in v2: {len(v1_only_targets)}")
print(f"First 10: {list(v1_only_targets)[:10]}")


train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
train_labels_v2 = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.v2.csv")


train_labels.head()


train_labels_v2.head()


# Extract target_ids from labels by removing the residue suffix
v1_label_targets = set(train_labels['ID'].str.rsplit('_', n=1).str[0])
v2_label_targets = set(train_labels_v2['ID'].str.rsplit('_', n=1).str[0])

# Check for target_ids in v1 labels that are not in v2 labels
v1_only_label_targets = v1_label_targets - v2_label_targets
print(f"Target IDs in v1 labels but not in v2 labels: {len(v1_only_label_targets)}")
print(f"First 10: {list(v1_only_label_targets)[:10]}")


# Concatenate both sequence dataframes
combined_sequences = pd.concat([train_sequences, train_sequences_v2], ignore_index=True)

# Concatenate both label dataframes  
combined_labels = pd.concat([train_labels, train_labels_v2], ignore_index=True)

print(f"Combined sequences shape: {combined_sequences.shape}")
print(f"Combined labels shape: {combined_labels.shape}")
print(f"Unique target_ids in combined sequences: {combined_sequences['target_id'].nunique()}")


combined_sequences.head()


# Extract target_ids from FASTA headers for RNA sequences only
rna_headers = [h for h in headers if 'RNA' in h and 'DNA' not in h]
fasta_rna_targets = set()

for header in rna_headers:
    # Extract PDB_ID and chain from header like ">1SCL_A mol:na..."
    target_part = header.split()[0][1:]  # Remove '>' and take first part
    fasta_rna_targets.add(target_part)

print(f"RNA targets in FASTA: {len(fasta_rna_targets)}")

# Check overlap with combined sequences
combined_targets = set(combined_sequences['target_id'])
overlap = combined_targets.intersection(fasta_rna_targets)
missing_in_fasta = combined_targets - fasta_rna_targets

print(f"Combined sequence targets: {len(combined_targets)}")
print(f"Overlap (targets in both): {len(overlap)}")
print(f"Missing in FASTA: {len(missing_in_fasta)}")
print(f"Coverage: {len(overlap)/len(combined_targets)*100:.1f}%")


# Check unique target_id counts in combined sequences
target_counts = combined_sequences['target_id'].value_counts()
print(f"Total unique target_ids: {len(target_counts)}")
print(f"Target_ids appearing more than once: {(target_counts > 1).sum()}")
print(f"\nFirst 10 target_ids and their counts:")
print(target_counts.head(10))

print(f"\nSample target_ids from combined_sequences:")
print(combined_sequences['target_id'].head(10).tolist())

print(f"\nSample target_ids from FASTA RNA headers:")
sample_fasta_targets = list(fasta_rna_targets)[:10]
print(sample_fasta_targets)


# Compare the naming patterns more clearly
print("Combined sequences target_id pattern:")
print("Format appears to be: [4-char PDB]_[1-2 char chain]")
for target in combined_sequences['target_id'].head(5):
    print(f"  {target}")

print(f"\nFASTA RNA target_id pattern:")
print("Format appears to be: [4-char PDB]_[1-2 char chain]")
for target in list(fasta_rna_targets)[:5]:
    print(f"  {target}")

# Check if there's any case sensitivity or format difference
combined_lower = set(t.lower() for t in combined_sequences['target_id'])
fasta_lower = set(t.lower() for t in fasta_rna_targets)
overlap_lower = combined_lower.intersection(fasta_lower)

print(f"\nCase-insensitive check:")
print(f"Overlap when ignoring case: {len(overlap_lower)}")

# Check PDB codes only (without chain)
combined_pdbs = set(t.split('_')[0] for t in combined_sequences['target_id'])
fasta_pdbs = set(t.split('_')[0] for t in fasta_rna_targets)
pdb_overlap = combined_pdbs.intersection(fasta_pdbs)

print(f"\nPDB code overlap (ignoring chains):")
print(f"Combined PDB codes: {len(combined_pdbs)}")
print(f"FASTA PDB codes: {len(fasta_pdbs)}")
print(f"PDB overlap: {len(pdb_overlap)}")


# The case-insensitive overlap suggests the issue is case sensitivity
# Let's examine this more closely

print("Case comparison examples:")
combined_sample = list(combined_sequences['target_id'])[:5]
fasta_sample = list(fasta_rna_targets)[:5]

for target in combined_sample:
    print(f"Combined: {target} -> lowercase: {target.lower()}")

for target in fasta_sample:
    print(f"FASTA: {target} -> lowercase: {target.lower()}")

# Check if FASTA uses lowercase PDB codes
print(f"\nPDB code case analysis:")
combined_pdb_sample = [t.split('_')[0] for t in combined_sample]
fasta_pdb_sample = [t.split('_')[0] for t in fasta_sample]

print("Combined PDB codes:", combined_pdb_sample)
print("FASTA PDB codes:", fasta_pdb_sample)

# Check PDB overlap with case-insensitive comparison
combined_pdbs_lower = set(t.split('_')[0].lower() for t in combined_sequences['target_id'])
fasta_pdbs_lower = set(t.split('_')[0].lower() for t in fasta_rna_targets)
pdb_overlap_lower = combined_pdbs_lower.intersection(fasta_pdbs_lower)

print(f"\nCase-insensitive PDB overlap: {len(pdb_overlap_lower)}")
print(f"This explains the discrepancy - FASTA uses lowercase PDB codes!")


# !ls /kaggle/input/stanford-rna-3d-folding/PDB_RNA


# Extract RNA sequences and coordinates with deduplication
from Bio.PDB import MMCIFParser
import pandas as pd
from pathlib import Path

def extract_rna_data_from_cif(cif_file_path):
    """Extract unique RNA sequences and C1' coordinates from a CIF file"""
    parser = MMCIFParser(QUIET=True)
    
    try:
        structure = parser.get_structure('structure', cif_file_path)
        pdb_id = Path(cif_file_path).stem.upper()
        
        sequences_data = []
        coordinates_data = []
        seen_sequences = set()  # Track unique sequences
        
        for model in structure:
            for chain in model:
                chain_id = chain.id
                target_id = f"{pdb_id}_{chain_id}"
                
                # Check if chain contains RNA residues
                rna_residues = []
                for residue in chain:
                    if residue.get_resname() in ['A', 'U', 'G', 'C']:  # RNA nucleotides
                        rna_residues.append(residue)
                
                if rna_residues:  # Only process if RNA residues found
                    # Build sequence
                    sequence = ''.join([res.get_resname() for res in rna_residues])
                    
                    # Only add if sequence is unique
                    if sequence not in seen_sequences:
                        seen_sequences.add(sequence)
                        sequences_data.append({
                            'target_id': target_id,
                            'sequence': sequence
                        })
                        
                        # Extract C1' coordinates for this unique sequence
                        for i, residue in enumerate(rna_residues, 1):
                            if "C1'" in residue:
                                atom = residue["C1'"]
                                coordinates_data.append({
                                    'ID': f"{target_id}_{i}",
                                    'resname': residue.get_resname(),
                                    'resid': i,
                                    'x_1': atom.coord[0],
                                    'y_1': atom.coord[1], 
                                    'z_1': atom.coord[2]
                                })
        
        return sequences_data, coordinates_data
        
    except Exception as e:
        print(f"Error processing {cif_file_path}: {e}")
        return [], []


# Process all CIF files and save to CSV
import os
from tqdm import tqdm

def process_all_cif_files():
    """Process all CIF files in the directory and extract RNA data"""
    cif_dir = "/kaggle/input/stanford-rna-3d-folding/PDB_RNA"
    cif_files = [f for f in os.listdir(cif_dir) if f.endswith('.cif')]
    
    all_sequences = []
    all_coordinates = []
    
    print(f"Processing {len(cif_files)} CIF files...")
    
    for cif_file in tqdm(cif_files):
        cif_path = os.path.join(cif_dir, cif_file)
        sequences, coordinates = extract_rna_data_from_cif(cif_path)
        
        all_sequences.extend(sequences)
        all_coordinates.extend(coordinates)
    
    return all_sequences, all_coordinates

# Process all files
print("Starting full extraction...")
all_sequences, all_coordinates = process_all_cif_files()

print(f"\nFull extraction summary:")
print(f"Total unique RNA sequences: {len(all_sequences)}")
print(f"Total coordinate entries: {len(all_coordinates)}")

# Create DataFrames
sequences_df = pd.DataFrame(all_sequences)
coordinates_df = pd.DataFrame(all_coordinates)

# Save to CSV files
sequences_df.to_csv('rna_sequences.csv', index=False)
coordinates_df.to_csv('rna_coordinates.csv', index=False)

print(f"\nSaved files:")
print(f"rna_sequences.csv: {sequences_df.shape}")
print(f"rna_coordinates.csv: {coordinates_df.shape}")

print(f"\nFirst few entries:")
print(sequences_df.head())


# MSA     - 856
# MSA_v2  - 2534
# PDB_RNA - 8672

