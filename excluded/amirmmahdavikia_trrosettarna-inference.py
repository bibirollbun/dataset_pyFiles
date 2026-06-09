%%capture
!pip install /kaggle/input/trrosettarna-v1-1/trRosettaRNA_v1.1/pyrosetta_cache/pyrosetta-2025.19+release.fabfc12491-cp311-cp311-linux_x86_64.whl
!pip uninstall -y tensorflow tensorflow-gpu
!pip install --no-index --find-links=/kaggle/input/trrosettarna-v1-1/trRosettaRNA_v1.1/dependencies tensorflow==2.15.0
!pip install /kaggle/input/biopython/biopython-1.85-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


!cp -r /kaggle/input/trrosettarna-v1-1/trRosettaRNA_v1.1 /kaggle/working/
!rm -r /kaggle/working/trRosettaRNA_v1.1/pyrosetta_cache
%cd /kaggle/working/trRosettaRNA_v1.1


import os
import re
import gc
import subprocess
from datetime import datetime
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.PDB import PDBParser
from Bio import SeqIO


colors = ['#c1121f', '#ffb703', '#003049']
print('\n----- Color -----\n')
sns.palplot(sns.color_palette(colors))


PATH = '/kaggle/input/stanford-rna-3d-folding'
SS = '/kaggle/working/secondary_structure'
OUTPUT = '/kaggle/working/trRosettaRNA_pdb'

os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(SS, exist_ok=True)


test_seq = pd.read_csv(os.path.join(PATH, 'test_sequences.csv'))
test_seq.head()


def fasta_to_a3m_msa(input_file, output_file):
    records = list(SeqIO.parse(input_file, "fasta"))
    if not records:
        raise ValueError("Empty MSA")

    query = str(records[0].seq)
    output_lines = [f">{records[0].id}\n{query}"]  # Keep query as-is

    for record in records[1:]:
        aligned = str(record.seq)
        # Remove characters aligned to gaps in the query
        compressed = ''.join([
            res for res, q in zip(aligned, query) if q != '-'
        ])
        output_lines.append(f">{record.id}\n{compressed}")

    with open(output_file, "w") as f:
        f.write("\n".join(output_lines) + "\n")

def write_fasta(sequence, target_id, output_file):
    with open(output_file, "w") as f:
        f.write(f">{target_id}\n{sequence}\n")

for idx, rna in tqdm(test_seq.iterrows(), total=len(test_seq)):
    target_id = rna.target_id
    sequence = rna.sequence

    os.makedirs(os.path.join(SS, target_id), exist_ok=True)
    
    fasta_path = os.path.join(SS, target_id, f"{target_id}.fasta")
    msa_path = os.path.join(PATH, 'MSA', f'{target_id}.MSA.fasta')
    a3m_path = os.path.join(SS, target_id, f'{target_id}.a3m')
    
    write_fasta(sequence, target_id, fasta_path)
    fasta_to_a3m_msa(msa_path, a3m_path)


for idx, rna in test_seq.iterrows():
    target_id = rna.target_id
    length = len(rna.sequence)
    input_msa = os.path.join(SS, target_id, f'{target_id}.a3m')
    output_dir = os.path.join(OUTPUT, target_id)
    os.makedirs(output_dir, exist_ok=True)
    seq_npz = os.path.join(output_dir, f'{target_id}.npz')
        
    print(f'\n----- Predicting 2D Structure: {target_id} (#{length}) -----\n')
        
    cmd = [
        'python', 'predict.py',
        '-i', input_msa,
        '-o', seq_npz,
        '-mdir', './params/model_1',
        '-gpu', '0'
    ]
    print(f'Running: {" ".join(cmd)}')
        
    result = subprocess.run(cmd, capture_output=True, text=True)
        
    # Log stdout and stderr
    print(f'[STDOUT for {target_id}]\n{result.stdout}')
    #print(f'[STDERR for {target_id}]\n{result.stderr}')
        
    # Check for errors
    if result.returncode != 0:
        print(f'[ERROR] predict.py failed for {target_id} with return code {result.returncode}')
    
    gc.collect()
    torch.cuda.empty_cache()
    del result


for idx, rna in test_seq.iterrows():
    
    target_id = rna.target_id
    length = len(rna.sequence)
    output_dir = os.path.join(OUTPUT, target_id)
    fastafile = os.path.join(SS, target_id, f'{target_id}.fasta')
            
    print(f'\n{datetime.now()} [{idx+1}] Folding 3D Structure: {target_id} (#{length}) using tRrosettaRNA\n')
            
    npz_path = os.path.join(output_dir, f'{target_id}.npz')
    pdb_path = os.path.join(output_dir, f'{target_id}.pdb')
        
    cmd = [
        "python", "fold.py",
        "-npz", npz_path,
        "-fa", fastafile,
        "-out", pdb_path,
        "-nm", "5",
        "-dcut", "0.45",
        "-cpu", "4"
    ]
            
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired as e:
        print(f'\n{datetime.now()} [ERROR] tRrosettaRNA failed on {target_id} (#{length})\nException: {e}\n')
        
    print(f'[STDOUT for {target_id}]\n{result.stdout}')
    print(f'[STDERR for {target_id}]\n{result.stderr}')
    gc.collect()


!rm -r /kaggle/working/trRosettaRNA_v1.1
!rm -r {SS}
%cd /kaggle/working/


submission = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))


for idx, rna in test_seq.iterrows():
    target_id = rna.target_id
    prediction_path = os.path.join(OUTPUT, target_id)
    predictions = glob(os.path.join(prediction_path, '*.pdb'))

    for i, sample in enumerate(predictions):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("rna", sample)
        for model in structure:
            for chain in model:
                for residue in chain:
                    res_number = residue.get_id()[1]
                    res_id = f'{target_id}_{res_number}'
                    for atom in residue:
                        if atom.get_name() == "C1'":
                            x, y, z = atom.coord
                            submission.loc[submission.ID == res_id, f'x_{i+1}'] = float(x)
                            submission.loc[submission.ID == res_id, f'y_{i+1}'] = float(y)
                            submission.loc[submission.ID == res_id, f'z_{i+1}'] = float(z)


print('Submission.csv shape:', submission.shape)
submission.to_csv('/kaggle/working/submission.csv', index=False)


submission.head()


!rm -r {OUTPUT}


solution = pd.read_csv(os.path.join(PATH, 'validation_labels.csv'))
solution.head()


def parse_tmscore_output(output):
    # Extract TM-score based on length of reference structure (second)
    tm_score_match = re.findall(r'TM-score=\s+([\d.]+)', output)[1]
    if not tm_score_match:
        raise ValueError('No TM score found')
    return float(tm_score_match)


def write_target_line(
    atom_name, atom_serial, residue_name, chain_id, residue_num, x_coord, y_coord, z_coord, occupancy=1.0, b_factor=0.0, atom_type='P'
) -> str:
    """
    Writes a single line of PDB format based on provided atom information.

    Args:
        atom_name (str): Name of the atom (e.g., "N", "CA").
        atom_serial (int): Atom serial number.
        residue_name (str): Residue name (e.g., "ALA").
        chain_id (str): Chain identifier.
        residue_num (int): Residue number.
        x_coord (float): X coordinate.
        y_coord (float): Y coordinate.
        z_coord (float): Z coordinate.
        occupancy (float, optional): Occupancy value (default: 1.0).
        b_factor (float, optional): B-factor value (default: 0.0).

    Returns:
        str: A single line of PDB string.
    """
    return f'ATOM  {atom_serial:>5d}  {atom_name:<5s} {residue_name:<3s} {residue_num:>3d}    {x_coord:>8.3f}{y_coord:>8.3f}{z_coord:>8.3f}{occupancy:>6.2f}{b_factor:>6.2f}           {atom_type}\n'


def write2pdb(df: pd.DataFrame, xyz_id: str, target_path: str) -> int:
    resolved_cnt = 0
    with open(target_path, 'w') as target_file:
        for _, row in df.iterrows():
            x_coord = row[f'x_{xyz_id}']
            y_coord = row[f'y_{xyz_id}']
            z_coord = row[f'z_{xyz_id}']

            if x_coord > -1e17 and y_coord > -1e17 and z_coord > -1e17:
                # if True:
                resolved_cnt += 1
                target_line = write_target_line(
                    atom_name="C1'",
                    atom_serial=int(row['resid']),
                    residue_name=row['resname'],
                    chain_id='0',
                    residue_num=int(row['resid']),
                    x_coord=x_coord,
                    y_coord=y_coord,
                    z_coord=z_coord,
                    atom_type='C',
                )
                target_file.write(target_line)
    return resolved_cnt


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Computes the TM-score between predicted and native RNA structures using USalign.

    This function evaluates the structural similarity of RNA predictions to native structures
    by computing the TM-score. It uses USalign, a structural alignment tool, to compare
    the predicted structures with the native structures.

    Workflow:
    1. Copies the USalign binary to the working directory and grants execution permissions.
    2. Extracts the `target_id` from the `ID` column of both the solution and submission DataFrames.
    3. Iterates over each unique `target_id`, grouping the native and predicted structures.
    4. Writes PDB files for native and predicted structures.
    5. Runs USalign on each predicted-native pair and extracts the TM-score.
    6. Computes the highest TM-score per target and returns aggregated results.

    Args:
        solution (pd.DataFrame): A DataFrame containing the native RNA structures.
        submission (pd.DataFrame): A DataFrame containing the predicted RNA structures.
        row_id_column_name (str): The name of the column containing unique row identifiers.

    Returns:
        float: the average highest TM-scores.
    """

    os.system('cp //kaggle/input/usalign/USalign /kaggle/working/')
    os.system('sudo chmod u+x /kaggle/working//USalign')

    # Extract target_id from ID (target_resid)
    solution['target_id'] = solution['ID'].apply(lambda x: x.split('_')[0])
    submission['target_id'] = submission['ID'].apply(lambda x: x.split('_')[0])

    results = {}
    # Iterate through each target_id and generate PDB files for both clean and corrupted data
    for target_id, group_native in tqdm(solution.groupby('target_id'), desc='TM-scoring'):
        group_predicted = submission[submission['target_id'] == target_id]
        native_pdb = 'native.pdb'
        predicted_pdb = 'predicted.pdb'

        target_id_scores = []
        for pred_cnt in range(1, 6):
            prediction_scores = []
            for native_cnt in range(1, 41):
                # Write solution PDB
                resolved_cnt = write2pdb(group_native, native_cnt, native_pdb)

                # Write predicted PDB
                _ = write2pdb(group_predicted, pred_cnt, predicted_pdb)

                if resolved_cnt > 0:
                    command = f'/kaggle/working/USalign {predicted_pdb} {native_pdb} -atom " C1\'"'
                    usalign_output = os.popen(command).read()
                    prediction_scores.append(parse_tmscore_output(usalign_output))

            target_id_scores.append(max(prediction_scores))
        results[target_id] = max(target_id_scores)

    return results


scores = score(solution, submission, 'ID')
result = {'target_id': [],
          'length': [],
          'tm_score': []}

for idx, rna in test_seq.iterrows():
    
    target_id = rna.target_id
    length = len(rna.sequence)
    tm_score = scores[target_id]
    
    result['target_id'].append(target_id)
    result['length'].append(length)
    result['tm_score'].append(tm_score)

result_df = pd.DataFrame(result)
result_df


def plot_results(df, desc, axs=None):
    if axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 2]})
    else:
        fig = None

    df['color'] = df['length'].apply(lambda x: colors[0] if x > 400 else colors[-1])

    sns.scatterplot(data=df, x='length', y='tm_score', hue='color', palette={colors[0]: colors[0], colors[-1]: colors[-1]}, ax=axs[0], legend=False)
    axs[0].set_title('TM-score per length')
    axs[0].set_xlabel('RNA length')
    axs[0].set_ylabel('TM-score')
    axs[0].set_ylim(0., 1.)

    mean_tm = df['tm_score'].mean()
    axs[0].axhline(mean_tm, color=colors[1], linestyle='--', label=f'Mean = {mean_tm:.2f}')
    axs[0].legend()

    sns.barplot(data=df, x='target_id', y='tm_score', palette=df['color'].to_list(), ax=axs[1])
    axs[1].set_title('TM-score per target')
    axs[1].set_xlabel('Target id')
    axs[1].set_ylabel('TM-score')
    axs[1].set_ylim(0., 1.)
    axs[1].axhline(mean_tm, color=colors[1], linestyle='--', label=f'Mean = {mean_tm:.2f}')
    axs[1].legend()


plot_results(result_df, desc='Results')


def log_resluts(result_df):
    long_score = result_df.loc[result_df.length > 400, 'tm_score'].mean()
    short_score = result_df.loc[result_df.length <= 400, 'tm_score'].mean()
    all_score = result_df.tm_score.mean()

    print(f'[TM-score] > 400: {long_score:.4f}')
    print(f'[TM-score] <= 400: {short_score:.4f}')
    print(f'[TM-score] all: {all_score:.4f}')

log_resluts(result_df)

