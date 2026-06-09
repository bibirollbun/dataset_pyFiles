!pip install pymsaviz


import numpy as np
import pandas as pd
import re

train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
train_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')
print(train_labels.shape), print(train_sequences.shape)
train_sequences


def extract_resid_resname(df):
    df = df.copy()  
    df['protein'] = None  
    df['chain'] = None

    for index, entry in df['all_sequences'].dropna().items():  
        if isinstance(entry, str):  
            split_data_pro = entry.split('|')[0] 
            split_data_pro = entry.split('>')[1]
            pro = split_data_pro.split('_')[0]  
            
            match = re.search(r"\|Chain (\w)\|", entry) 
            chain = match.group(1) if match else None

            df.at[index, 'protein'] = pro  
            df.at[index, 'chain'] = chain
    return df


train_sequences = extract_resid_resname(train_sequences)
train_sequences


!cat /kaggle/input/stanford-rna-3d-folding/MSA/17RA_A.MSA.fasta /kaggle/input/stanford-rna-3d-folding/MSA/1A1T_B.MSA.fasta /kaggle/input/stanford-rna-3d-folding/MSA/1A4T_A.MSA.fasta > MSA.fasta


from Bio import SeqIO

msa_file = "MSA.fasta"
output_file = "MSA_fixed.fasta"

# æœ€å¤§ã�®é…�åˆ—é•·ã‚’å�–å¾—
max_length = max(len(record.seq) for record in SeqIO.parse(msa_file, "fasta"))

# é…�åˆ—ã‚’æ�ƒã�ˆã‚‹
with open(output_file, "w") as out_f:
    for record in SeqIO.parse(msa_file, "fasta"):
        fixed_seq = str(record.seq).ljust(max_length, "-")  # ä¿®æ­£: str() ã�«å¤‰æ�›
        out_f.write(f">{record.id}\n{fixed_seq}\n")

print(f"âœ… ä¿®æ­£æ¸ˆã�¿ã�® MSA ãƒ•ã‚¡ã‚¤ãƒ«ã‚’ä¿�å­˜ã�—ã�¾ã�—ã�Ÿ: {output_file}")



from pymsaviz import MsaViz, get_msa_testdata
import matplotlib.pyplot as plt
import IPython.display as display


#msa_file = "/kaggle/input/stanford-rna-3d-folding/MSA/17RA_A.MSA.fasta"
msa_file = "MSA_fixed.fasta"
mv = MsaViz(msa_file, color_scheme="Taylor", wrap_length=50, show_grid=True, show_consensus=True)
mv.savefig("api_example01.png")

display.display(display.Image("api_example01.png"))


test_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')
test_sequences = extract_resid_resname(test_sequences)
test_sequences


df_merged = test_sequences.merge(train_sequences, on="protein", how="left")
df_merged


test_8TVZ = df_merged[df_merged['protein']=='8TVZ']['sequence_x']
train_8TVZ = df_merged[df_merged['protein']=='8TVZ']['sequence_y']

test_8BTZ = df_merged[df_merged['protein']=='8BTZ']['sequence_x']
train_8BTZ = df_merged[df_merged['protein']=='8BTZ']['sequence_x']

test_8UYS = df_merged[df_merged['protein']=='8UYS']['sequence_x']
train_8UYS = df_merged[df_merged['protein']=='8UYS']['sequence_x']

test_7YR7 = df_merged[df_merged['protein']=='7YR7']['sequence_x']
train_7YR7 = df_merged[df_merged['protein']=='7YR7']['sequence_x']


# FASTA ãƒ•ã‚¡ã‚¤ãƒ«ã�®ä½œæˆ�
fasta_filename = "sequences_8TVZ.fasta"

with open(fasta_filename, "w") as fasta_file:
    for i, seq in enumerate(test_8TVZ, 1):
        fasta_file.write(f">test_8TVZ_{i}\n{seq}\n")
    
    for i, seq in enumerate(train_8TVZ, 1):
        fasta_file.write(f">train_8TVZ_{i}\n{seq}\n")

print(f"âœ… FASTA ãƒ•ã‚¡ã‚¤ãƒ«ã�Œä½œæˆ�ã�•ã‚Œã�¾ã�—ã�Ÿ: {fasta_filename}")


msa_file = "sequences_8TVZ.fasta"
mv = MsaViz(msa_file, color_scheme="Taylor", wrap_length=50, show_grid=True, show_consensus=True)
mv.savefig("sequences_8TVZ.png")

display.display(display.Image("sequences_8TVZ.png"))


# FASTA ãƒ•ã‚¡ã‚¤ãƒ«ã�®ä½œæˆ�
fasta_filename = "sequences_8BTZ.fasta"

with open(fasta_filename, "w") as fasta_file:
    for i, seq in enumerate(test_8BTZ, 1):
        fasta_file.write(f">test_8BTZ_{i}\n{seq}\n")
    
    for i, seq in enumerate(train_8BTZ, 1):
        fasta_file.write(f">train_8BTZ_{i}\n{seq}\n")

print(f"âœ… FASTA ãƒ•ã‚¡ã‚¤ãƒ«ã�Œä½œæˆ�ã�•ã‚Œã�¾ã�—ã�Ÿ: {fasta_filename}")


msa_file = "sequences_8BTZ.fasta"
mv = MsaViz(msa_file, color_scheme="Taylor", wrap_length=50, show_grid=True, show_consensus=True)
mv.savefig("sequences_8BTZ.png")

display.display(display.Image("sequences_8BTZ.png"))


# FASTA ãƒ•ã‚¡ã‚¤ãƒ«ã�®ä½œæˆ�
fasta_filename = "sequences_8UYS.fasta"

with open(fasta_filename, "w") as fasta_file:
    for i, seq in enumerate(test_8UYS, 1):
        fasta_file.write(f">test_8UYS_{i}\n{seq}\n")
    
    for i, seq in enumerate(train_8UYS, 1):
        fasta_file.write(f">train_8UYS_{i}\n{seq}\n")

print(f"âœ… FASTA ãƒ•ã‚¡ã‚¤ãƒ«ã�Œä½œæˆ�ã�•ã‚Œã�¾ã�—ã�Ÿ: {fasta_filename}")



msa_file = "sequences_8UYS.fasta"
mv = MsaViz(msa_file, color_scheme="Taylor", wrap_length=50, show_grid=True, show_consensus=True)
mv.savefig("sequences_8UYS.png")

display.display(display.Image("sequences_8UYS.png"))


# FASTA ãƒ•ã‚¡ã‚¤ãƒ«ã�®ä½œæˆ�
fasta_filename = "sequences_7YR7.fasta"

with open(fasta_filename, "w") as fasta_file:
    for i, seq in enumerate(test_7YR7, 1):
        fasta_file.write(f">test_7YR7_{i}\n{seq}\n")
    
    for i, seq in enumerate(train_7YR7, 1):
        fasta_file.write(f">train_7YR7_{i}\n{seq}\n")

print(f"âœ… FASTA ãƒ•ã‚¡ã‚¤ãƒ«ã�Œä½œæˆ�ã�•ã‚Œã�¾ã�—ã�Ÿ: {fasta_filename}")



msa_file = "sequences_7YR7.fasta"
mv = MsaViz(msa_file, color_scheme="Taylor", wrap_length=50, show_grid=True, show_consensus=True)
mv.savefig("sequences_7YR7.png")

display.display(display.Image("sequences_7YR7.png"))




