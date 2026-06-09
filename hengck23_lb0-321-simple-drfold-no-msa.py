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

    valid_df = valid_df.iloc[[0,1,6,7]].reset_index(drop=True) #for speedup debug

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
DEVICE='cuda' #'cpu'

print('MODE:', MODE)
print('SETTING OK!!!')


sys.path.append('/kaggle/input/hengck23-drfold2-dummy-00/drfold2/cfg_97')
from EvoMSA2XYZ.Model import MSA2XYZ
from RNALM2.Model import RNA2nd
from data import parse_seq, Get_base, BASE_COOR
from data import write_frame_coor_to_pdb, parse_pdb_to_xyz


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
def make_data(seq):
    aa_type = parse_seq(seq)
    base = Get_base(seq, BASE_COOR)
    seq_idx = np.arange(len(seq)) + 1

    msa = aa_type[None, :]
    msa = torch.from_numpy(msa)
    msa = torch.cat([msa, msa], 0) #???
    msa = F.one_hot(msa.long(), 6).float()

    base_x  = torch.from_numpy(base).float()
    seq_idx = torch.from_numpy(seq_idx).long()
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
solution = make_dummy_solution()


#load model (these are moified versions, not the same from their github repo)
rnalm = RNA2nd(dict(
    s_in_dim=5,
    z_in_dim=2,
    s_dim= 512,
    z_dim= 128,
    N_elayers=18,
))
rnalm_file = '/kaggle/input/hengck23-drfold2-dummy-00/RCLM/epoch_67000'
print(rnalm_file)
print(
    rnalm.load_state_dict(torch.load(rnalm_file, map_location='cpu', weights_only=True), strict=False)
    #Unexpected key(s) in state_dict: "ss_head.linear.weight", "ss_head.linear.bias".
)
rnalm = rnalm.to(DEVICE)
rnalm = rnalm.eval()


total_time_taken = 0
max_gpu_mem_used = 0
for c in range(NUM_CONF):

    msa2xyz = MSA2XYZ(dict(
        seq_dim=6,
        msa_dim=7,
        N_ensemble=1,#3
        N_cycle=8,
        m_dim=64,
        s_dim=64,
        z_dim=64,
    ))
    msa2xyz_file = [
        f'/kaggle/input/hengck23-drfold2-dummy-00/cfg_97/model_{k}' for k in [2,6,10,14,18]
    ][c]
    print(msa2xyz_file)
    print(
        msa2xyz.load_state_dict(torch.load(msa2xyz_file, map_location='cpu', weights_only=True), strict=True)
    )
    msa2xyz.msaxyzone.premsa.rnalm = rnalm
    msa2xyz = msa2xyz.to(DEVICE)
    msa2xyz = msa2xyz.eval()
 
    for i,row in valid_df.iterrows():
        start_timer = timer()
        
        target_id = row.target_id
        sequence = row.sequence
        seq = row.sequence    
        
        L = len(sequence)
        if L>MAX_LENGTH:
            i0 = 0 #np.random.choice(L-MAX_LENGTH+1)
            i1 = i0 + MAX_LENGTH
        else:
            i0 = 0
            i1 = L
        
        seq = sequence[i0:i1]
        print(c,i,target_id, L, seq[:75]+'...')
        
        msa, base_x, seq_idx = make_data(seq)
        msa, base_x, seq_idx = msa.to(DEVICE), base_x.to(DEVICE), seq_idx.to(DEVICE)
        secondary = None #secondary structure
    
        with torch.no_grad(): 
            out = msa2xyz.pred(msa, seq_idx, secondary, base_x, np.array(list(seq)))

        # key = list(out.keys()) # plddt(L,L), coor(L,3,3), dist_p(L,L,38), dist_c, dist_n,
        # for k in key:
        #     print(k, type(out[k]), out[k].shape)
 
        
        if L!=len(seq):
             out['coor'] = np.pad(out['coor'] ,((i0, L - i1), (0, 0), (0, 0)), 'constant', constant_values=0)


        print('out:',  out['coor'].shape)
        time_taken = timer()-start_timer
        total_time_taken += time_taken
        print('time_taken:', time_to_str(time_taken, mode='sec')) 
        
        gpu_mem_used = gpu_memory_use()
        max_gpu_mem_used = max(max_gpu_mem_used,gpu_mem_used)
        print('gpu_mem_used:', gpu_mem_used, 'GB')

        torch.cuda.empty_cache() 
        pdb_file = f'{out_dir}/{target_id}-coor.{c}.pdb'
        write_frame_coor_to_pdb(out['coor'], sequence, pdb_file) 
        xyz, resname, resid = parse_pdb_to_xyz(pdb_file)
        #assert(resname==row.sequence)
        #assert(resid==list(np.arange(L)+1))

        solution[target_id].coord.append(xyz)
        
        if MODE == 'local':
            pass  # save for local cv
        else:
            os.remove(pdb_file)
    print('')
    
#-----end of conformation generation ----
print('MAX_LENGTH', MAX_LENGTH)
print('### total_time_taken:', time_to_str(total_time_taken, mode='min'))
print('### max_gpu_mem_used:', max_gpu_mem_used, 'GB')
print('')

submit_df = solution_to_submit_df(solution)
submit_df.to_csv(f'submission.csv', index=False)
print(submit_df)
print('SUBMIT OK!!!!!!')
print('')


if 1: 
    print('debug: show first perdict')
    solution = list(solution.values())
    s = solution[0]
    target_id = s.target_id
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
 


    if MODE=='local':
        truth_df  = get_truth_df(target_id, label_df)
        truth = truth_df[['x_1', 'y_1', 'z_1']].to_numpy().astype('float32')
        truth_pdb = f'{KAGGLE_TRUTH_PDB_DIR}/kaggle_truth_{target_id}_C1.pdb' 
        # print(os.path.isfile(truth_pdb))
 
        x, y, z = truth[:, 0], truth[:, 1], truth[:, 2]
        ax.scatter(x, y, z, c='black', s=30, alpha=1)
        ax.plot(x, y, z, color='black', linewidth=1, alpha=1, label=f'truth')
        
    else:
        truth = s.coord[0] #align to first one
        truth_pdb = f'{out_dir}/{target_id}-coor.{c}.pdb' 
        # print(os.path.isfile(truth_pdb))
         
    aligned = []
    tm_score = []
    for c in range(5):
        predict_pdb = f'{out_dir}/{target_id}-coor.{c}.pdb'
        # print(os.path.isfile(predict_pdb))
         
        command = f'{USALIGN} {predict_pdb} {truth_pdb} -atom " C1\'" -m -'
        output = os.popen(command).read()
        tm = parse_usalign_for_tm_score(output)
        transform = parse_usalign_for_transform(output)
        aligned.append(s.coord[c]@transform[:,1:].T + transform[:,[0]].T)
        tm_score.append(tm)

    
    if MODE!='local':
        tm_score =['?']*5


    max_c = np.array(tm_score).argmax() 
    for c in range(5):
        x, y, z = aligned[c][:, 0], aligned[c][:, 1], aligned[c][:, 2]
        alpha =1 if c==max_c else 0.2
        ax.scatter(x, y, z, c='RED', s=30, alpha=alpha)
        ax.plot(x, y, z, color='RED', linewidth=1, alpha=alpha, label=f'{c}: tm {tm_score[c]}:0.5f')
        
    set_aspect_equal(ax)
    plt.legend()
    plt.show() 
    plt.close()





if MODE=='local':
    # local validation
 
    tm_score=[]
    for i,row in valid_df.iterrows(): 
        target_id = row.target_id#'R1116' #casp15 R1116: len(157)
        seq = row.sequence 
        #-----------------------------------------------
        print(i,target_id, len(seq), seq[:75]+'...')
    
        truth_pdb = f'{KAGGLE_TRUTH_PDB_DIR}/kaggle_truth_{target_id}_C1.pdb'
        # print(os.path.isfile(truth_pdb))
        
        tm = []
        for c in range(NUM_CONF):
            predict_pdb = f'{out_dir}/{target_id}-coor.{c}.pdb'
            # print(os.path.isfile(predict_pdb))
        
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
    
    print('ALL\n',tm_score)
    print('MEAN', np.array(tm_score).mean())



