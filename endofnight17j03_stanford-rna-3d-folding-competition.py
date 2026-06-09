from datetime import datetime
import pytz
import pandas as pd
print('LOGGING TIME OF START:',  datetime.strftime(datetime.now(pytz.timezone('Asia/Singapore')), "%Y-%m-%d %H:%M:%S"))


try:
    import Bio
except:
    pass
    #for drfold2 --------
    #!pip install biopython
    #!pip install /kaggle/input/biopython/biopython-1.85-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

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


d = pd.read_csv("/kaggle/input/submission/submission (1).csv")


MODE = 'submit' #'local' # submit

DATA_KAGGLE_DIR = '/kaggle/input/stanford-rna-3d-folding'
if MODE == 'local':
    valid_df = pd.read_csv(f'{DATA_KAGGLE_DIR}/validation_sequences.csv')
    label_df = pd.read_csv(f'{DATA_KAGGLE_DIR}/validation_labels.csv')
    label_df['target_id'] = label_df['ID'].apply(lambda x: '_'.join(x.split('_')[:-1]))

    valid_df = valid_df.iloc[[0,1,2]]

if MODE == 'submit':
	valid_df = pd.read_csv(f'{DATA_KAGGLE_DIR}/test_sequences.csv')
	#valid_df = pd.read_csv(f'/kaggle/input/hengck23-top-data/casp16_top1_sequence_df.csv')

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


# ###########################################################3
 
# # data helper
# def make_data(seq):
#     aa_type = parse_seq(seq)
#     base = Get_base(seq, BASE_COOR)
#     seq_idx = np.arange(len(seq)) + 1

#     msa = aa_type[None, :]
#     msa = torch.from_numpy(msa)
#     msa = torch.cat([msa, msa], 0) #???
#     msa = F.one_hot(msa.long(), 6).float()

#     base_x  = torch.from_numpy(base).float()
#     seq_idx = torch.from_numpy(seq_idx).long()
#     return msa, base_x, seq_idx
    
# def make_dummy_solution():
#     solution=dotdict()
#     for i, row in valid_df.iterrows():
#         target_id = row.target_id
#         sequence = row.sequence
#         solution[target_id]=dotdict(
#             target_id=target_id,
#             sequence=sequence,
#             coord=[
#                 np.zeros((len(sequence),3), dtype=np.float32) for j in range(5)
#             ],
#         )
#     return solution

# def solution_to_submit_df(solution):
#     submit_df = []
#     for k,s in solution.items():
#         df = coord_to_df(s.sequence, s.coord, s.target_id)
#         submit_df.append(df)
    
#     submit_df = pd.concat(submit_df)
#     return submit_df
 

# def coord_to_df(sequence, coord, target_id):
#     L = len(sequence)
#     df = pd.DataFrame()
#     df['ID'] = [f'{target_id}_{i + 1}' for i in range(L)]
#     df['resname'] = [s for s in sequence]
#     df['resid'] = [i + 1 for i in range(L)]

#     num_coord = len(coord)
#     for j in range(num_coord):
#         df[f'x_{j+1}'] = coord[j][:, 0]
#         df[f'y_{j+1}'] = coord[j][:, 1]
#         df[f'z_{j+1}'] = coord[j][:, 2]
#     return df

# ################### start here !!! #######################################################3
# #---
# # import pickle
# # top_file =\
# # '/kaggle/input/hengck23-top-data/af3-casp16_out.pkl'
# #     #'/kaggle/input/hengck23-top-data/vfold-casp16_out.pkl'
# #     #'/kaggle/input/hengck23-top-data/drfold2-casp16_out.pkl'
# # with open(top_file, 'rb') as f:
# # 	CASP_TOP = pickle.load(f)
# # print('CASP_TOP read !!!!!')
# #----

# solution = make_dummy_solution()
# #----
# for i,row in valid_df.iterrows():
#     target_id = row.target_id
#     sequence = row.sequence
#     #sequence='AAGUACCCUCCAAGCCCUACAGGUUGGAAGAGGGGGCUAUCAGUCCUGUAGGCAGACUC'
#     #if len(sequence)<200: continue    
#     #if len(sequence)>400: continue
    
#     top = CASP_TOP.get(sequence,None)
#     if top is None: continue
#     if len(top)==0: continue
    
#     #print(top)
#     print('\r', i, len(top), target_id, sequence[:10] + '...','found!', end='', flush=True)
 
#     nL = min(len(row.sequence), len(top[0]))
#     for k in range(len(top)):
#         solution[target_id].coord[k][:nL]=top[k][:nL]
        
# print('')
# #----


# submit_df = solution_to_submit_df(solution)
# submit_df.to_csv(f'submission.csv', index=False)
# print(submit_df)
# print('SUBMIT OK!!!!!!')
# print('')
 





d.to_csv('submission.csv', index=False)

