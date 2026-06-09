from datetime import datetime
import pytz
print('LOGGING TIME OF START:',  datetime.strftime(datetime.now(pytz.timezone('Asia/Singapore')), "%Y-%m-%d %H:%M:%S"))


try:
    import Bio
except:
    #for drfold2 --------
    #!pip install biopython
    !pip install /kaggle/input/biopython/biopython-1.85-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

print('PIP INSTALL OK !!!!')


import os,sys

import pandas as pd
pd.set_option('display.max_columns', 20)
pd.set_option('display.expand_frame_repr', False)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer as timer
import re

import matplotlib 
import matplotlib.pyplot as plt


# helper--
class dotdict(dict):
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

	def __getattr__(self, name):
		try:
			return self[name]
		except KeyError:
			raise AttributeError(name)

def time_to_str(t, mode='min'):
	if mode=='min':
		t  = int(t)/60
		hr = t//60
		min = t%60
		return '%2d hr %02d min'%(hr,min) 
	elif mode=='sec':
		t   = int(t)
		min = t//60
		sec = t%60
		return '%2d min %02d sec'%(min,sec)

	else:
		raise NotImplementedError

def gpu_memory_use():
    if torch.cuda.is_available():
        device = torch.device(0)
        free, total = torch.cuda.mem_get_info(device)
        used= (total - free) / 1024 ** 3
        return int(round(used))
    else:
        return 0

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


print('torch',torch.__version__)
print('torch.cuda',torch.version.cuda)

print('IMPORT OK!!!')


MODE = 'submit' #'local' # submit

DATA_KAGGLE_DIR = '/kaggle/input/stanford-rna-3d-folding'
if MODE == 'local':
    valid_df = pd.read_csv(f'{DATA_KAGGLE_DIR}/validation_sequences.csv')
    label_df = pd.read_csv(f'{DATA_KAGGLE_DIR}/validation_labels.csv')
    label_df['target_id'] = label_df['ID'].apply(lambda x: '_'.join(x.split('_')[:-1]))

if MODE == 'submit':
	valid_df = pd.read_csv(f'{DATA_KAGGLE_DIR}/test_sequences.csv')

print('len(valid_df)',len(valid_df))
print(valid_df.iloc[0])
print('')


# cfg = dotdict(
#     num_conf = 5,
#     max_length=480,
# )
NUM_CONF=5
MAX_LENGTH=480
DEVICE='cuda'#'cuda' #'cpu'#

print('MODE:', MODE)
print('SETTING OK!!!')


sys.path.append('/kaggle/input/hengck23-rna-02/drfold2/cfg_97')
from EvoMSA2XYZ.Model import MSA2XYZ
from RNALM2.Model import RNA2nd
from data import parse_seq, Get_base, BASE_COOR
from data import (
    write_frame_coor_to_pdb,
    parse_pdb_to_xyz,
    frame_coor_to_C1
)
from energy_scorer import score_energy_one


###########################################################3
KAGGLE_TRUTH_PDB_DIR ='/kaggle/input/hengck23-drfold2-dummy-00/kaggle-casp15-truth'
USALIGN = '/kaggle/working/USalign' 
os.system('cp /kaggle/input/usalign/USalign /kaggle/working/')
os.system('sudo chmod u+x /kaggle/working/USalign')

# evaluate helper
def get_truth_df(target_id, label_df):
    truth_df = label_df[label_df['target_id'] == target_id]
    truth_df = truth_df.reset_index(drop=True)
    return truth_df

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



# data helper
def make_data(seq, device):
    aa_type = parse_seq(seq)
    base = Get_base(seq, BASE_COOR)
    seq_idx = np.arange(len(seq)) + 1

    msa = aa_type[None, :]
    msa = torch.from_numpy(msa)
    msa = torch.cat([msa, msa], 0)  # ???
    msa = F.one_hot(msa.long(), 6).float()

    base_x = torch.from_numpy(base).float()
    seq_idx = torch.from_numpy(seq_idx).long()

    msa, base_x, seq_idx = msa.to(device), base_x.to(device), seq_idx.to(device)
    return msa, base_x, seq_idx

    
def make_dummy_solution():
    solution=dotdict()
    for i, row in valid_df.iterrows():
        target_id = row.target_id
        sequence = row.sequence
        solution[target_id]=dotdict(
            target_id=target_id,
            sequence=sequence,
            coord=[],
        )
    return solution

def solution_to_submit_df(solution):
    submit_df = []
    for k,s in solution.items():
        df = coord_to_df(s.sequence, s.coord, s.target_id)
        submit_df.append(df)
    
    submit_df = pd.concat(submit_df)
    return submit_df
 

def coord_to_df(sequence, coord, target_id):
    L = len(sequence)
    df = pd.DataFrame()
    df['ID'] = [f'{target_id}_{i + 1}' for i in range(L)]
    df['resname'] = [s for s in sequence]
    df['resid'] = [i + 1 for i in range(L)]

    num_coord = len(coord)
    for j in range(num_coord):
        df[f'x_{j+1}'] = coord[j][:, 0]
        df[f'y_{j+1}'] = coord[j][:, 1]
        df[f'z_{j+1}'] = coord[j][:, 2]
    return df

################### start here !!! #######################################################3


out_dir = '/kaggle/working/model-output'
os.makedirs(out_dir, exist_ok=True)


def run_submit(valid_df):
    
    #load model (these are moified versions, not the same from their github repo)
    rnalm = RNA2nd(dict(
        s_in_dim=5,
        z_in_dim=2,
        s_dim= 512,
        z_dim= 128,
        N_elayers=18,
    ))
    rnalm_file = '/kaggle/input/hengck23-rna-02/weight/RCLM/epoch_67000'
    print(rnalm_file)
    print(
        rnalm.load_state_dict(torch.load(rnalm_file, map_location='cpu', weights_only=True), strict=False)
        #Unexpected key(s) in state_dict: "ss_head.linear.weight", "ss_head.linear.bias".
    )
    rnalm = rnalm.to(DEVICE)
    rnalm = rnalm.eval()
    
    #---
    msa2xyz = MSA2XYZ(dict(
        seq_dim=6,
        msa_dim=7,
        N_ensemble=1,
        N_cycle=8,  # 8
        m_dim=64,
        s_dim=64,
        z_dim=64,
    ))
    msa2xyz_file = [
        f'/kaggle/input/hengck23-rna-02/weight/cfg_97/model_{i}'
        for i in range(20)
    ]
    num_msa2xyz = len(msa2xyz_file) 
    msa2xyz_state_dict = []
    for c in range(num_msa2xyz):
        if c==0: print(msa2xyz_file[c])
        m = torch.load(msa2xyz_file[c], map_location='cpu', weights_only=True)
        msa2xyz_state_dict.append(m)
        
    print(msa2xyz.load_state_dict(msa2xyz_state_dict[0], strict=True))
    msa2xyz = msa2xyz.to(DEVICE)
    msa2xyz = msa2xyz.eval()
    msa2xyz.msaxyzone.premsa.rnalm = rnalm

    #---
    # start here !!!!!!!!!!!!!!!!!!!!!!
    #valid_df = valid_df.iloc[[0,1]].reset_index(drop=True)


    submit_df = [] 
    total_time_taken = 0
    max_gpu_mem_used = 0

    for i, row in valid_df.iterrows():
        start_timer = timer()
        target_id = row.target_id  # 'R1116' #casp15 R1116: len(157)
        sequence = row.sequence
        seq = row.sequence  
        L = len(seq)
        if L > MAX_LENGTH:
            seq = seq[:MAX_LENGTH]
        print(i, target_id, L, len(seq), seq[:75] + '...')

        msa, base_x, seq_idx = make_data(seq, DEVICE)
        secondary = None  # secondary structure

        if len(seq)>200:
            model_to_try = [0,1,2,8,9]
        elif  len(seq)>100:
            model_to_try = [0,2,4,6,8,10,12,14,16,18]#list(range(min(num_msa2xyz,10)))
        else:
            model_to_try = list(range(num_msa2xyz))

        energy = []
        netout = []
        coordinate=[]
        for c in model_to_try:
            msa2xyz.load_state_dict(msa2xyz_state_dict[c], strict=False)
            #print(c, msa2xyz.training, msa2xyz.msa_predor.linear.weight.device, msa2xyz.msa_predor.linear.weight.reshape(-1)[:5])

            #with torch.amp.autocast('cuda',dtype=torch.float16):
            with torch.no_grad():
                out = msa2xyz.pred(msa, seq_idx, secondary, base_x, np.array(list(seq)))

            if len(model_to_try)>5:
                e = score_energy_one(seq, target_id, out)
            else:
                e = 0
            energy.append(e) #tranucated sequence

 
            if L != len(seq):
                out['coor'] = np.pad(out['coor'] ,((0, L - len(seq)), (0, 0), (0, 0)), 'constant', constant_values=0)
            netout.append(out)
            
            xyz = frame_coor_to_C1(out['coor'], sequence)
            coordinate.append(xyz)
            
            #print('out:', out['coor'].shape)

            time_taken = timer() - start_timer
            total_time_taken += time_taken
            #print('time_taken:', time_to_str(time_taken, mode='sec'))

            gpu_mem_used = gpu_memory_use()
            max_gpu_mem_used = max(max_gpu_mem_used,gpu_mem_used)
            #print('gpu_mem_used:', gpu_mem_used, 'GB')

            print(f'{c:02d}   energy:{e:10.0f}   out{str(out["coor"].shape)}  time:{time_to_str(time_taken, mode="sec")}   gpu={gpu_mem_used} gb')


        #------- 
        torch.cuda.empty_cache()
        
        #select top5
        argsort = np.array(energy).argsort()
        argsort = argsort[:5]
        df = coord_to_df(row.sequence, [coordinate[k] for k in argsort], row.target_id)
        submit_df.append(df)
    
    print('----------------------------------------')
    print('MAX_LENGTH', MAX_LENGTH)
    print('### total_time_taken:', time_to_str(total_time_taken, mode='min'))
    print('### max_gpu_mem_used:', max_gpu_mem_used, 'GB')
    print('')

    submit_df = pd.concat(submit_df)
    submit_df.to_csv(f'submission.csv', index=False)
    print(submit_df) 
    return submit_df
  
run_submit(valid_df)

print('SUBMIT OK!!!')


if MODE == 'local':
    from data import write2pdb_for_tm

    # local validation
    submit_df = pd.read_csv('submission.csv')
    submit_df['target_id'] = submit_df['ID'].apply(lambda x: '_'.join(x.split('_')[:-1]))

    tm_score = []
    for i, row in valid_df.iterrows():
        target_id = row.target_id  # 'R1116' #casp15 R1116: len(157)
        seq = row.sequence
        # -----------------------------------------------
        print(i, target_id, len(seq), seq[:75] + '...')

        truth_pdb = f'{KAGGLE_TRUTH_PDB_DIR}/kaggle_truth_{target_id}_C1.pdb'
        # print(os.path.isfile(predict_pdb))

        tm = []
        for c in range(5):
            # predict_pdb = f'{out_dir}/{target_id}-coor.{c:02d}.pdb'
            # print(os.path.isfile(truth_pdb))
            df = submit_df[submit_df['target_id'] == target_id]
            xyz = df.loc[:, [f'x_{c+1}',f'y_{c+1}',f'z_{c+1}']].values
            predict_pdb =  '/kaggle/working/predict.pdb'
            write2pdb_for_tm(seq, xyz, target_id, predict_pdb)

            command = f'{USALIGN} {predict_pdb} {truth_pdb} -atom " C1\'" -m -'
            output = os.popen(command).read()
            # print(output)
            try:
                tm_c = parse_usalign_for_tm_score(output)
            except:
                tm_c = 0
            tm.append(tm_c)
        print('### tm:', tm)
        tm_score.append(max(tm))

    print('ALL\n', tm_score)
    print('MEAN', np.array(tm_score).mean())



