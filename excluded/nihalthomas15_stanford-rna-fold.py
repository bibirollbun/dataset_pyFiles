import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import pickle


config = {
    "seed": 0,
    "cutoff_date": "2020-01-01",
    "test_cutoff_date": "2022-05-01",
    "max_len": 384,
    "batch_size": 1,
    "learning_rate": 1e-5,
    "weight_decay": 0.0, #altered from 0.0
    "mixed_precision": "bf16",
    "model_config_path": "../working/configs/pairwise.yaml",  # Adjust path as needed
    "epochs": 10,
    "cos_epoch": 5,
    "loss_power_scale": 1.0,
    "max_cycles": 1,
    "grad_clip": 0.1,
    "gradient_accumulation_steps": 1,
    "d_clamp": 30,
    "max_len_filter": 9999999,
    "structural_violation_epoch": 50,
    "balance_weight": False,
}


config = {
    "seed": 42,  # More common for reproducibility
    "cutoff_date": "2020-01-01",
    "test_cutoff_date": "2022-05-01",
    "max_len": 512,  # Increase max sequence length for better long-range dependencies
    "batch_size": 2,  # Increase batch size to stabilize training (adjust based on GPU memory)
    "learning_rate": 1.5e-5,  # Slightly higher to improve convergence speed
    "weight_decay": 1e-3,  # Introduce small weight decay to prevent overfitting
    "mixed_precision": "bf16",  # Keeps efficiency without losing precision
    "model_config_path": "../working/configs/pairwise.yaml",  
    "epochs": 15,  # Increase epochs to allow better convergence
    "cos_epoch": 7,  # Align cosine annealing to longer training
    "loss_power_scale": 1.0,
    "max_cycles": 2,  # Increase cycles to allow multiple learning phases
    "grad_clip": 1.0,  # Increase gradient clipping for more stability
    "gradient_accumulation_steps": 2,  # Helps manage memory while maintaining batch effect
    "d_clamp": 30,
    "max_len_filter": 9999999,
    "structural_violation_epoch": 40,  # Reduce slightly for faster feedback
    "balance_weight": True,  # Balance loss weighting if dataset is imbalanced
}


test_data=pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")


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
test_dataset[0]


import sys

sys.path.append("/kaggle/input/ribonanzanet2d-final")


from Network import *
import yaml



class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)

class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, config, pretrained=False):
        config.dropout=0.25
        super(finetuned_RibonanzaNet, self).__init__(config)
        if pretrained:
            self.load_state_dict(torch.load("/kaggle/input/ribonanzanet-weights/RibonanzaNet.pt",map_location='cpu'))
        # self.ct_predictor=nn.Sequential(nn.Linear(64,256),
        #                                 nn.ReLU(),
        #                                 nn.Linear(256,64),
        #                                 nn.ReLU(),
        #                                 nn.Linear(64,1)) 
        self.dropout=nn.Dropout(0.2)
        self.xyz_predictor=nn.Linear(256,3)

    def forward(self,src):
        
        #with torch.no_grad():
        sequence_features, pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))

        xyz=self.xyz_predictor(sequence_features)

        return xyz


model=finetuned_RibonanzaNet(load_config_from_yaml("/kaggle/input/ribonanzanet2d-final/configs/pairwise.yaml"),pretrained=False).cuda()

model.load_state_dict(torch.load("/kaggle/input/ribonanzanet-3d-finetune/RibonanzaNet-3D.pt"))


test_dataset[0]['sequence'].shape


model.eval()
preds=[]
for i in range(len(test_dataset)):
    src=test_dataset[i]['sequence'].long()
    src=src.unsqueeze(0).cuda()

    model.train()

    tmp=[]
    for i in range(4):
        with torch.no_grad():
            xyz=model(src).squeeze()
        tmp.append(xyz.cpu().numpy())

    model.eval()
    with torch.no_grad():
        xyz=model(src).squeeze()
    tmp.append(xyz.cpu().numpy())

    tmp=np.stack(tmp,0)
    #exit()
    preds.append(tmp)


tmp.shape


preds[0]


preds[7][0].shape


import plotly.graph_objects as go
import numpy as np

# Example: Generate an Nx3 matrix

xyz = preds[7][0]  # Replace this with your actual Nx3 data
N = len(xyz)

# Extract columns
x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

# Create the 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(
        size=5,
        color=z,  # Coloring based on z-value
        colorscale='Viridis',  # Choose a colorscale
        opacity=0.8
    )
)])

# Customize layout
fig.update_layout(
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    ),
    title="3D Scatter Plot"
)

# Show figure
fig.show(renderer='iframe')


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


submission
submission.to_csv('submission.csv',index=False)


submission


import pandas as pd
import torch
import numpy as np
import random
import yaml

# Set seed for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

config = {
    "seed": 0,
    "cutoff_date": "2020-01-01",
    "test_cutoff_date": "2022-05-01",
    "max_len": 384,
    "batch_size": 1,
    "learning_rate": 1e-5,
    "weight_decay": 1e-5,  # Small value to prevent overfitting
    "mixed_precision": "bf16",
    "model_config_path": "../working/configs/pairwise.yaml",
    "epochs": 10,
    "cos_epoch": 5,
    "loss_power_scale": 1.0,
    "max_cycles": 1,
    "grad_clip": 0.1,
    "gradient_accumulation_steps": 1,
    "d_clamp": 30,
    "max_len_filter": 9999999,
    "structural_violation_epoch": 50,
    "balance_weight": False,
}

# Load test dataset
test_data = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")

# RNA Dataset Class
class RNADataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.tokens = {nt: i for i, nt in enumerate('ACGU')}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = [self.tokens[nt] for nt in self.data.loc[idx, 'sequence']]
        sequence = torch.tensor(sequence, dtype=torch.long)
        return {'sequence': sequence}

# Load Dataset
test_dataset = RNADataset(test_data)

# Import model dependencies
import sys
sys.path.append("/kaggle/input/ribonanzanet2d-final")

from Network import RibonanzaNet

# Config Loader
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries = entries

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)

# Fine-tuned Model Class
class FinetunedRibonanzaNet(RibonanzaNet):
    def __init__(self, config, pretrained=False):
        config.dropout = 0.2  # Restore dropout
        super(FinetunedRibonanzaNet, self).__init__(config)
        
        if pretrained:
            self.load_state_dict(torch.load(
                "/kaggle/input/ribonanzanet-weights/RibonanzaNet.pt",
                map_location='cpu',
                weights_only=True  # SAFE loading
            ))
        
        self.dropout = torch.nn.Dropout(0.2)  # Keep some dropout for stability
        self.xyz_predictor = torch.nn.Linear(256, 3)  

    def forward(self, src):
        sequence_features, _ = self.get_embeddings(
            src, torch.ones_like(src).long().to(src.device)
        )
        xyz = self.xyz_predictor(sequence_features)
        return xyz

# Load Model
model = FinetunedRibonanzaNet(
    load_config_from_yaml("/kaggle/input/ribonanzanet2d-final/configs/pairwise.yaml"),
    pretrained=False
).cuda()

# Load Weights Securely
model.load_state_dict(torch.load(
    "/kaggle/input/ribonanzanet-3d-finetune/RibonanzaNet-3D.pt",
    map_location='cuda',
    weights_only=True  # SAFE loading
))

# Ensure model is in evaluation mode
model.eval()

# Inference
preds = []
with torch.no_grad():  # Disable gradient tracking for inference
    for i in range(len(test_dataset)):
        src = test_dataset[i]['sequence'].unsqueeze(0).cuda()
        xyz = model(src).squeeze().cpu().numpy()
        preds.append(xyz)

# Convert predictions to submission format
data = []
for i in range(len(test_data)):
    for j, res in enumerate(test_data.loc[i, 'sequence']):
        row = [test_data.loc[i, 'target_id'] + f"_{j+1}", res, j+1]
        row.extend(preds[i][j])
        data.append(row)

# Create DataFrame
columns = ['ID', 'resname', 'resid', 'x', 'y', 'z']
submission = pd.DataFrame(data, columns=columns)

# Save to CSV
submission.to_csv('submission.csv', index=False)


