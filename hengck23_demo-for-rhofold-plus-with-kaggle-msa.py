try:
    import Bio
except:
    #for rhofold+ #####################
    !pip install biopython
    !pip install ml-collections
    !pip install python-box
    !pip install dm-tree
    !pip install openmm[cuda12]





from copy import deepcopy

import pandas as pd
from Bio.PDB import Atom, Model, Chain, Residue, Structure, PDBParser
from Bio import SeqIO
import os, sys
import re
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

print('IMPORT OK !!!!')


PYTHON = sys.executable
print('PYTHON',PYTHON)

RHONET_DIR=\
'/kaggle/input/data-for-demo-for-rhofold-plus-with-kaggle-msa/RhoFold-main'
#'<your downloaded rhofold repo>/RhoFold-main'

USALIGN = \
'/kaggle/working//USalign'
#'<your us align path>/USalign'

os.system('cp /kaggle/input/usalign/USalign /kaggle/working/')
os.system('sudo chmod u+x /kaggle/working//USalign')


DATA_KAGGLE_DIR = '/kaggle/input/stanford-rna-3d-folding'
SEQ_DF = pd.read_csv(f'{DATA_KAGGLE_DIR}/train_sequences.csv')
LABEL_DF = pd.read_csv(f'{DATA_KAGGLE_DIR}/train_labels.csv')
LABEL_DF['target_id'] = LABEL_DF['ID'].apply(lambda x: '_'.join(x.split('_')[:-1]))


# helper ----
class dotdict(dict):
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

	def __getattr__(self, name):
		try:
			return self[name]
		except KeyError:
			raise AttributeError(name)

# visualisation helper ----
def set_aspect_equal(ax):
	x_limits = ax.get_xlim()
	y_limits = ax.get_ylim()
	z_limits = ax.get_zlim()

	# Compute the mean of each axis
	x_middle = np.mean(x_limits)
	y_middle = np.mean(y_limits)
	z_middle = np.mean(z_limits)

	# Compute the max range across all axes
	max_range = max(x_limits[1] - x_limits[0],
					y_limits[1] - y_limits[0],
					z_limits[1] - z_limits[0]) / 2.0

	# Set the new limits to ensure equal scaling
	ax.set_xlim(x_middle - max_range, x_middle + max_range)
	ax.set_ylim(y_middle - max_range, y_middle + max_range)
	ax.set_zlim(z_middle - max_range, z_middle + max_range)




# xyz df helper --------------------
def get_truth_df(target_id):
    truth_df = LABEL_DF[LABEL_DF['target_id'] == target_id]
    truth_df = truth_df.reset_index(drop=True)
    return truth_df

def parse_pdb_to_df(pdb_file, target_id):
    parser = PDBParser()
    structure = parser.get_structure('', pdb_file)

    df = []
    for model in structure:
        for chain in model:
            print(chain)
            chain_data = []
            for residue in chain:
                # print(residue)
                if residue.get_resname() in ['A', 'U', 'G', 'C']:
                    # Check if the residue has a C1' atom
                    if 'C1\'' in residue:
                        atom = residue['C1\'']
                        xyz = atom.get_coord()
                        resname = residue.get_resname()
                        resid = residue.get_id()[1]

                        #todo detect discontinous: resid = prev_resid+1
                        #ID	resname	resid	x_1	y_1	z_1
                        chain_data.append(dict(
                            ID = target_id+'_'+str(resid),
                            resname=resname,
                            resid=resid,
                            x_1=xyz[0],
                            y_1=xyz[1],
                            z_1=xyz[2],
                        ))
                        ##print(f"Residue {resname} {resid}, Atom: {atom.get_name()}, xyz: {xyz}")

            if len(chain_data)!=0:
                chain_df = pd.DataFrame(chain_data)
                df.append(chain_df)
                ##print(chain_df)
    return df

# usalign helper --------------------
def write_target_line(
    atom_name, atom_serial, residue_name, chain_id, residue_num, x_coord, y_coord, z_coord, occupancy=1.0, b_factor=0.0, atom_type='P'
):
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

def write_xyz_to_pdb(df, pdb_file, xyz_id = 1):
    resolved_cnt = 0
    with open(pdb_file, 'w') as target_file:
        for _, row in df.iterrows():
            x_coord = row[f'x_{xyz_id}']
            y_coord = row[f'y_{xyz_id}']
            z_coord = row[f'z_{xyz_id}']

            if x_coord > -1e17 and y_coord > -1e17 and z_coord > -1e17:
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

def parse_usalign_for_tm_score(output):
    # Extract TM-score based on length of reference structure (second)
    tm_score_match = re.findall(r'TM-score=\s+([\d.]+)', output)[1]
    if not tm_score_match:
        raise ValueError('No TM score found')
    return float(tm_score_match)

def parse_usalign_for_transform(output):
    # Locate the rotation matrix section
    matrix_lines = []
    found_matrix = False

    for line in output.splitlines():
        if "The rotation matrix to rotate Structure_1 to Structure_2" in line:
            found_matrix = True
        elif found_matrix and re.match(r'^\d+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+$', line):
            matrix_lines.append(line)
        elif found_matrix and not line.strip():
            break  # Stop parsing if an empty line is encountered after the matrix

    # Parse the rotation matrix values
    rotation_matrix = []
    for line in matrix_lines:
        parts = line.split()
        row_values = list(map(float, parts[1:]))  # Skip the first column (index)
        rotation_matrix.append(row_values)

    return np.array(rotation_matrix)

def call_usalign(predict_df, truth_df, verbose=1):
    truth_pdb = '~truth.pdb'
    predict_pdb = '~predict.pdb'
    write_xyz_to_pdb(predict_df, predict_pdb, xyz_id=1)
    write_xyz_to_pdb(truth_df, truth_pdb, xyz_id=1)

    command = f'{USALIGN} {predict_pdb} {truth_pdb} -atom " C1\'" -m -'
    output = os.popen(command).read()
    if verbose==1:
        print(output)
    tm_score = parse_usalign_for_tm_score(output)
    transform = parse_usalign_for_transform(output)
    return tm_score, transform


# msa helper --------------------
def read_msa(msa_file):
    f = open(msa_file, 'r')
    line = f.readlines()

    msa = []
    for i in range(0, len(line),2):
        m = dotdict(
            comment =line[i],
            seqence =line[i+1],
        )
        assert(m.comment[0]=='>')
        msa.append(m)
    return msa


def write_msa(msa_file, msa):
    line=[]
    for m in msa:
        line .append(m.comment)
        line .append(m.seqence)

    f = open(msa_file, 'wt')
    f.writelines(line)
    return msa
 
def msa_to_rhonet_file(msa_file, num_msa=5, out_dir='',target_id='xxx'):
    msa = read_msa(msa_file)
    msa0 = deepcopy(msa[0])
    msa0.comment =f'>{target_id}\n'
    msa0 = [msa0]

    a3m_file = f'{out_dir}/{target_id}.a3m'
    fasta_file = f'{out_dir}/{target_id}.fasta'
    os.makedirs(out_dir, exist_ok=True)

    write_msa(fasta_file, msa0)
    write_msa(a3m_file, msa[:num_msa])

print('HELPER OK!!!')


#start here!!!


out_dir   ='/kaggle/working/'
target_id = '1EIY_C'.upper()
sequence  = 'GCCGAGGUAGCUCAGUUGGUAGAGCAUGCGACUGAAAAUCGCAGUGUCCGCGGUUCGAUUCCGCGCCUCGGCACCA'
print('len(sequence):',len(sequence))


#1. prepare input
msa_file = f'{DATA_KAGGLE_DIR}/MSA/1EIY_C.MSA.fasta'
msa_to_rhonet_file(msa_file, num_msa=5, out_dir=out_dir,target_id=target_id)

cmd1 = f'cd {RHONET_DIR}'
#cmd2 = f'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6' #optional if you have lib error
cmd3 = f'{PYTHON} inference.py --input_fas {out_dir}/{target_id}.fasta --input_a3m {out_dir}/{target_id}.a3m --output_dir {out_dir}/ --ckpt ./pretrained/model_20221010_params.pt'

#follow rhofold repo, we use the cmdline:
#'python inference.py --input_fas ./example/input/3owzA/3owzA.fasta --input_a3m ./example/input/3owzA/3owzA.a3m --output_dir ./example/output/3owzA/ --ckpt ./pretrained/model_20221010_params.pt'

#do inference here!
#output = os.popen(cmd1+';'+cmd2+';'+cmd3).read()
output = os.popen(cmd1+';'+cmd3).read()
print(output)

#copy file from local results
#local_result = '/kaggle/input/data-for-demo-for-rhofold-plus-with-kaggle-msa/rhofold_input_output/.'
#!cp -a $local_result $out_dir

#expected ouput from rtx A6000 (non ada)
'''
2025-03-14 20:40:50,182 - INFO: Constructing RhoFold
2025-03-14 20:40:51,221 - INFO:     loading ./pretrained/model_20221010_params.pt
2025-03-14 20:40:51,743 - INFO: Input_fas /media/hp/c30d34ed-0d55-4077-82dc-b56cd13dd548/2025/kaggle/stanford-rna-3d-folding/result/rhofold_00/1EIY_C.fasta
2025-03-14 20:40:51,743 - INFO: Input_a3m /media/hp/c30d34ed-0d55-4077-82dc-b56cd13dd548/2025/kaggle/stanford-rna-3d-folding/result/rhofold_00/1EIY_C.a3m
2025-03-14 20:40:51,743 - INFO: Started RhoFold Inference
2025-03-14 20:40:51,755 - INFO:     Inference using device cuda
2025-03-14 20:40:54,523 - INFO:     Export PDB file to /media/hp/c30d34ed-0d55-4077-82dc-b56cd13dd548/2025/kaggle/stanford-rna-3d-folding/result/rhofold_00//unrelaxed_model.pdb
2025-03-14 20:40:54,523 - INFO: Finished RhoFold Inference in 2.780 seconds
2025-03-14 20:40:54,523 - INFO: Started Amber Relaxation : 1000 iterations
2025-03-14 20:40:54,523 - INFO:     AmberRelaxation: Using OpenCL
2025-03-14 20:41:09,410 - INFO:     Minimizing ...
2025-03-14 20:42:38,203 - INFO:     Energy at Minima is -505932.780 kcal/mol
2025-03-14 20:42:38,362 - INFO:     Export PDB file to /media/hp/c30d34ed-0d55-4077-82dc-b56cd13dd548/2025/kaggle/stanford-rna-3d-folding/result/rhofold_00//relaxed_1000_model.pdb
2025-03-14 20:42:38,363 - INFO: Finished Amber Relaxation : 1000 iterations in 103.840 seconds
'''

#expected ouput from P100
'''
len(sequence): 76
2025-03-14 16:03:33,110 - INFO: Constructing RhoFold
2025-03-14 16:03:34,429 - INFO:     loading ./pretrained/model_20221010_params.pt
2025-03-14 16:03:35,088 - INFO: Input_fas /kaggle/working//1EIY_C.fasta
2025-03-14 16:03:35,089 - INFO: Input_a3m /kaggle/working//1EIY_C.a3m
2025-03-14 16:03:35,089 - INFO: Started RhoFold Inference
2025-03-14 16:03:35,093 - INFO:     Inference using device cuda
2025-03-14 16:03:40,177 - INFO:     Export PDB file to /kaggle/working///unrelaxed_model.pdb
2025-03-14 16:03:40,177 - INFO: Finished RhoFold Inference in 5.088 seconds
2025-03-14 16:03:40,177 - INFO: Started Amber Relaxation : 1000 iterations
2025-03-14 16:03:40,177 - INFO:     AmberRelaxation: Using OpenCL
2025-03-14 16:04:03,056 - INFO:     Minimizing ...
2025-03-14 16:08:49,875 - INFO:     Energy at Minima is -497896.334 kcal/mol
2025-03-14 16:08:50,044 - INFO:     Export PDB file to /kaggle/working///relaxed_1000_model.pdb
2025-03-14 16:08:50,047 - INFO: Finished Amber Relaxation : 1000 iterations in 309.869 seconds

'''


#visualise prediction and compute tm score

predict_relax_df = parse_pdb_to_df(f'{out_dir}/relaxed_1000_model.pdb', target_id)
predict_unrelax_df = parse_pdb_to_df(f'{out_dir}/unrelaxed_model.pdb', target_id)

assert len(predict_relax_df)==1
assert len(predict_unrelax_df)==1
predict_relax_df = predict_relax_df[0]
predict_unrelax_df = predict_unrelax_df[0]

print(predict_relax_df)
print(predict_unrelax_df)

truth_df = get_truth_df(target_id)
print(truth_df)

tm_score_relax, transform_relax = call_usalign(predict_relax_df, truth_df, verbose=1)
tm_score_unrelax, transform_unrelax= call_usalign(predict_unrelax_df, truth_df, verbose=0)

print('tm_score_relax', tm_score_relax)
print('tm_score_unrelax', tm_score_unrelax)
print('transform_relax\n', transform_relax)
print('transform_unrelax\n', transform_unrelax)
zz=0

if 1:
    COLOR = ['red', 'blue', 'green', 'black', 'yellow', 'cyan', 'magenta']
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # ax.clear()

    #unrelax
    coord = predict_unrelax_df[['x_1', 'y_1', 'z_1']].to_numpy().astype('float32')
    coord = coord@transform_unrelax[:,1:].T + transform_unrelax[:,[0]].T
    x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]
    ax.scatter(x, y, z, c='red', s=30, alpha=1)
    ax.plot(x, y, z, color='red', linewidth=1, alpha=1, label=f'unrelax (tm:{tm_score_unrelax:0.3f})')


    #relax
    coord = predict_relax_df[['x_1', 'y_1', 'z_1']].to_numpy().astype('float32')
    coord = coord@transform_relax[:,1:].T + transform_relax[:,[0]].T
    x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]
    ax.scatter(x, y, z, c='orange', s=30, alpha=1)
    ax.plot(x, y, z, color='orange', linewidth=1, alpha=1, label=f'relax (tm:{tm_score_relax:0.3f})')

    # truth
    truth = truth_df[['x_1', 'y_1', 'z_1']].to_numpy().astype('float32')
    x, y, z = truth[:, 0], truth[:, 1], truth[:, 2]
    ax.scatter(x, y, z, c='black', s=30, alpha=1)
    ax.plot(x, y, z, color='black', linewidth=1, alpha=1, label=f'truth')

    set_aspect_equal(ax)
    plt.legend()
    plt.show()
    # plt.waitforbuttonpress()
    plt.close()
    

