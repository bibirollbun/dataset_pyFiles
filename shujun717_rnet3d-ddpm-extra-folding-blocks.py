import sys
sys.path.append("/kaggle/input/rnet3d-ddpm-test80/test80_improved_inference_upload")





import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import pickle
from utils import *
import os
#from Diffusion import Diffusion
import argparse
from tqdm import tqdm


# If running in a notebook, replace argparse with manual assignment
class Args:
    target_csv = '/kaggle/input/stanford-rna-3d-folding/test_sequences.csv'
    config = '/kaggle/input/rnet3d-ddpm-test80/test80_improved_inference_upload/recycle.yaml'
    weights = '/kaggle/input/rnet3d-ddpm-test80/test80_improved_inference_upload/weights/recycle.yaml_RibonanzaNet_3D.pt'

args = Args()


test_data=pd.read_csv(args.target_csv)#.loc[2:].reset_index(drop=True)

from torch.utils.data import Dataset, DataLoader

class RNADataset(Dataset):
    def __init__(self,data):
        self.data=data
        self.tokens={nt:i for i,nt in enumerate('ACGU')}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence=[self.tokens[nt] for nt in (self.data.loc[idx,'sequence'])]
        sequence=np.array(sequence)
        sequence=torch.tensor(sequence)




        return {'sequence':sequence}

test_dataset=RNADataset(test_data)


import torch.nn as nn
from Diffusion import finetuned_RibonanzaNet



args.config

config=load_config_from_yaml(args.config)

model=finetuned_RibonanzaNet(load_config_from_yaml("/kaggle/input/rnet3d-ddpm-test80/test80_improved_inference_upload/pairwise.yaml"),config,pretrained=False).cuda()
#model.decode(torch.ones(1,10).long().cuda(),torch.ones(1,10).long().cuda())


import torch
state_dict=torch.load(args.weights,map_location='cpu')
#state_dict=torch.load("RibonanzaNet-3D-v2.pt",map_location='cpu')

#get rid of module. from ddp state dict
new_state_dict={}

for key in state_dict:
    new_state_dict[key[7:]]=state_dict[key]

model.load_state_dict(new_state_dict)



model.eval()
preds=[]
for i in tqdm(range(len(test_dataset))):
    src=test_dataset[i]['sequence'].long()
    src=src.unsqueeze(0).cuda()
    target_id=test_data.loc[i,'target_id']

    predicted_dm=[]
    #for _ in range(5):
    with torch.no_grad():
        #xyz,distogram=model.sample_euler(src,5,200,N_cycle=config.max_cycles)
        with torch.cuda.amp.autocast():
            #xyz,distogram=model.sample_euler(src,5,200,N_cycle=10)
            #xyz,distogram=model.sample_euler(src,5,200,N_cycle=config.max_cycles)
            #xyz,distogram=model.sample_euler(src,5,200,N_cycle=10)
            xyz,distogram=model.sample_euler(src,5,200,N_cycle=1)
    preds.append(xyz.cpu().numpy())


ID=[]
resname=[]
resid=[]
x=[]
y=[]
z=[]

data=[]

for i in range(len(test_data)):
    #print(test_data.loc[i])

    
    for j in range(len(test_data.loc[i,'sequence'])):
        # ID.append(test_data.loc[i,'sequence_id']+f"_{j+1}")
        # resname.append(test_data.loc[i,'sequence'][j])
        # resid.append(j+1) # 1 indexed
        row=[test_data.loc[i,'target_id']+f"_{j+1}",
             test_data.loc[i,'sequence'][j],
             j+1]

        for k in range(5):
            for kk in range(3):
                row.append(preds[i][k][j][kk])
        data.append(row)

columns=['ID','resname','resid']
for i in range(1,6):
    columns+=[f"x_{i}"]
    columns+=[f"y_{i}"]
    columns+=[f"z_{i}"]


submission=pd.DataFrame(data,columns=columns)


csv_filename=args.config.split('/')[-1].replace('.csv','')
submission.to_csv(f'{csv_filename}_predictions.csv',index=False)



import plotly.graph_objects as go
import numpy as np

# Step 1: Load predicted and native coordinates
index = 6
ID = test_data.loc[index, 'target_id']
xyz = preds[index][0]  # Nx3 predicted coordinates

# Step 5: Plot
fig = go.Figure()

# Predicted structure (xyz)
fig.add_trace(go.Scatter3d(
    x=xyz[:, 0],
    y=xyz[:, 1],
    z=xyz[:, 2],
    mode='markers',
    marker=dict(size=5, color='blue', opacity=0.7),
    name='Predicted'
))


# Layout
fig.update_layout(
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    ),
    title=f"3D Alignment: {ID}",
    legend=dict(x=0.01, y=0.99)
)

# Show
fig.show(renderer='iframe')




