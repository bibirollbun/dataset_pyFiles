!pip install biopython
!pip install ViennaRNA


import pandas as pd
from Bio.Seq import Seq
from Bio.SeqUtils import  molecular_weight
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils import MeltingTemp as mt
import re
import RNA  


df = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv', usecols = ('target_id', 'sequence'))
df.head(3)


def extract_rna_features(sequence):
    sequence = re.sub(r'[^ACGU]', '', sequence.upper())
    seq = Seq(sequence)
    
    # Basic statistics
    length = len(sequence)
    A_count = sequence.count('A')
    C_count = sequence.count('C')
    G_count = sequence.count('G')
    U_count = sequence.count('U')
    GC_content = gc_fraction(seq) * 100
    
    # Molecular weight
    molecular_weight_value = molecular_weight(seq, seq_type='RNA')
    
    
    # Purine/Pyrimidine Ratio
    purines = A_count + G_count
    pyrimidines = C_count + U_count
    purine_pyrimidine_ratio = purines / pyrimidines if pyrimidines != 0 else None
    
    # Palindromic Motifs
    palindromic_motifs = [sequence[i:j] for i in range(length) for j in range(i + 4, length + 1) if sequence[i:j] == sequence[i:j][::-1]]
    palindromic_motifs_count = len(palindromic_motifs)
    
    # Start/Stop Codons (for coding RNA)
    start_codon = sequence.startswith('AUG')
    stop_codon = bool(re.search(r'(UAA|UGA|UAG)', sequence))
    
    return length, A_count, C_count, G_count, U_count, GC_content, molecular_weight_value, purine_pyrimidine_ratio, palindromic_motifs_count, start_codon, stop_codon


df[['length', 'A_count', 'C_count', 'G_count', 'U_count', 
    'GC_content', 'molecular_weight', 
    'purine_pyrimidine_ratio', 'palindromic_motifs_count', 
    'start_codon', 'stop_codon']] = df['sequence'].apply(lambda x: pd.Series(extract_rna_features(x)))

df.head()


df.to_csv("protein_sequence_feature_etraction.csv", index=False)

