import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import pickle
import os
import sys


config = {
    "seed": 0,
    "cutoff_date": "2020-01-01",
    "test_cutoff_date": "2022-05-01",
    "max_len": 384,
    "batch_size": 1,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
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


test_data=pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")
test_data.head()


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


sys.path.append("/kaggle/input/ribonanzanet2/pytorch/alpha/1")

import torch.nn as nn
from Network import *

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, rnet_config, config, pretrained=False):
        rnet_config.dropout=0.1
        rnet_config.use_grad_checkpoint=True
        super(finetuned_RibonanzaNet, self).__init__(rnet_config)
        if pretrained:
            self.load_state_dict(torch.load(config.pretrained_weight_path,map_location='cpu'))
        # self.ct_predictor=nn.Sequential(nn.Linear(64,256),
        #                                 nn.ReLU(),
        #                                 nn.Linear(256,64),
        #                                 nn.ReLU(),
        #                                 nn.Linear(64,1)) 
        self.dropout=nn.Dropout(0.0)

        decoder_dim=config.decoder_dim
        self.structure_module=[SimpleStructureModule(d_model=decoder_dim, nhead=config.decoder_nhead, 
                 dim_feedforward=decoder_dim*4, pairwise_dimension=rnet_config.pairwise_dimension, dropout=0.0) for i in range(config.decoder_num_layers)]
        self.structure_module=nn.ModuleList(self.structure_module)

        self.xyz_embedder=nn.Linear(3,decoder_dim)
        self.xyz_norm=nn.LayerNorm(decoder_dim)
        self.xyz_predictor=nn.Linear(decoder_dim,3)
        
        self.adaptor=nn.Sequential(nn.Linear(rnet_config.ninp,decoder_dim),nn.LayerNorm(decoder_dim))

        self.distogram_predictor=nn.Sequential(nn.LayerNorm(rnet_config.pairwise_dimension),
                                                nn.Linear(rnet_config.pairwise_dimension,40))

        self.time_embedder=SinusoidalPosEmb(decoder_dim)

        self.time_mlp=nn.Sequential(nn.Linear(decoder_dim,decoder_dim),
                                    nn.ReLU(),  
                                    nn.Linear(decoder_dim,decoder_dim))
        self.time_norm=nn.LayerNorm(decoder_dim)

        self.distance2pairwise=nn.Linear(1,rnet_config.pairwise_dimension,bias=False)

        self.pair_mlp=nn.Sequential(nn.Linear(rnet_config.pairwise_dimension,rnet_config.pairwise_dimension),
                                    nn.ReLU(),
                                    nn.Linear(rnet_config.pairwise_dimension,rnet_config.pairwise_dimension))


        #hyperparameters for diffusion
        self.n_times = config.n_times

        #self.model = model
        
        # define linear variance schedule(betas)
        beta_1, beta_T = config.beta_min, config.beta_max
        betas = torch.linspace(start=beta_1, end=beta_T, steps=config.n_times)#.to(device) # follows DDPM paper
        self.sqrt_betas = torch.sqrt(betas)
                                     
        # define alpha for forward diffusion kernel
        self.alphas = 1 - betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1-alpha_bars)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)

        self.data_std=config.data_std


    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(*inputs)
            return inputs
        return custom_forward
    
    def embed_pair_distance(self,inputs):
        pairwise_features,xyz=inputs
        distance_matrix=xyz[:,None,:,:]-xyz[:,:,None,:]
        distance_matrix=(distance_matrix**2).sum(-1).clip(2,37**2).sqrt()
        distance_matrix=distance_matrix[:,:,:,None]
        pairwise_features=pairwise_features+self.distance2pairwise(distance_matrix)

        return pairwise_features

    def forward(self,src,xyz,t):
        
        #with torch.no_grad():
        sequence_features, pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))
        
        distogram=self.distogram_predictor(pairwise_features)

        sequence_features=self.adaptor(sequence_features)

        decoder_batch_size=xyz.shape[0]
        sequence_features=sequence_features.repeat(decoder_batch_size,1,1)
        

        pairwise_features=pairwise_features.expand(decoder_batch_size,-1,-1,-1)

        pairwise_features= checkpoint.checkpoint(self.custom(self.embed_pair_distance), [pairwise_features,xyz],use_reentrant=False)

        time_embed=self.time_embedder(t).unsqueeze(1)
        tgt=self.xyz_norm(sequence_features+self.xyz_embedder(xyz)+time_embed)

        tgt=self.time_norm(tgt+self.time_mlp(tgt))

        for layer in self.structure_module:
            #tgt=layer([tgt, sequence_features,pairwise_features,xyz,None])
            tgt=checkpoint.checkpoint(self.custom(layer),
            [tgt, sequence_features,pairwise_features,xyz,None],
            use_reentrant=False)
            # xyz=xyz+self.xyz_predictor(sequence_features).squeeze(0)
            # xyzs.append(xyz)
            #print(sequence_features.shape)
        
        xyz=self.xyz_predictor(tgt).squeeze(0)
        #.squeeze(0)

        return xyz, distogram
    

    def denoise(self,sequence_features,pairwise_features,xyz,t):
        decoder_batch_size=xyz.shape[0]
        sequence_features=sequence_features.expand(decoder_batch_size,-1,-1)
        pairwise_features=pairwise_features.expand(decoder_batch_size,-1,-1,-1)

        pairwise_features=self.embed_pair_distance([pairwise_features,xyz])

        sequence_features=self.adaptor(sequence_features)
        time_embed=self.time_embedder(t).unsqueeze(1)
        tgt=self.xyz_norm(sequence_features+self.xyz_embedder(xyz)+time_embed)
        tgt=self.time_norm(tgt+self.time_mlp(tgt))
        #xyz_batch_size=xyz.shape[0]
        


        for layer in self.structure_module:
            tgt=layer([tgt, sequence_features,pairwise_features,xyz,None])
            # xyz=xyz+self.xyz_predictor(sequence_features).squeeze(0)
            # xyzs.append(xyz)
            #print(sequence_features.shape)
        xyz=self.xyz_predictor(tgt).squeeze(0)
        # print(xyz.shape)
        # exit()
        return xyz


    def extract(self, a, t, x_shape):
        """
            from lucidrains' implementation
                https://github.com/lucidrains/denoising-diffusion-pytorch/blob/beb2f2d8dd9b4f2bd5be4719f37082fe061ee450/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L376
        """
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
    def scale_to_minus_one_to_one(self, x):
        # according to the DDPMs paper, normalization seems to be crucial to train reverse process network
        return x * 2 - 1
    
    def reverse_scale_to_zero_to_one(self, x):
        return (x + 1) * 0.5
    
    def make_noisy(self, x_zeros, t): 
        # assume we get raw data, so center and scale by 35
        x_zeros = x_zeros - torch.nanmean(x_zeros,1,keepdim=True)
        x_zeros = x_zeros/self.data_std
        #rotate randomly
        x_zeros = random_rotation_point_cloud_torch_batch(x_zeros)


        # perturb x_0 into x_t (i.e., take x_0 samples into forward diffusion kernels)
        epsilon = torch.randn_like(x_zeros).to(x_zeros.device)
        
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars.to(x_zeros.device), t, x_zeros.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars.to(x_zeros.device), t, x_zeros.shape)
        
        # Let's make noisy sample!: i.e., Forward process with fixed variance schedule
        #      i.e., sqrt(alpha_bar_t) * x_zero + sqrt(1-alpha_bar_t) * epsilon
        noisy_sample = x_zeros * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar
    
        return noisy_sample.detach(), epsilon
    
    
    # def forward(self, x_zeros):
    #     x_zeros = self.scale_to_minus_one_to_one(x_zeros)
        
    #     B, _, _, _ = x_zeros.shape
        
    #     # (1) randomly choose diffusion time-step
    #     t = torch.randint(low=0, high=self.n_times, size=(B,)).long().to(x_zeros.device)
        
    #     # (2) forward diffusion process: perturb x_zeros with fixed variance schedule
    #     perturbed_images, epsilon = self.make_noisy(x_zeros, t)
        
    #     # (3) predict epsilon(noise) given perturbed data at diffusion-timestep t.
    #     pred_epsilon = self.model(perturbed_images, t)
        
    #     return perturbed_images, epsilon, pred_epsilon
    
    
    def denoise_at_t(self, x_t, sequence_features, pairwise_features, timestep, t):
        B, _, _ = x_t.shape
        if t > 1:
            z = torch.randn_like(x_t).to(sequence_features.device)
        else:
            z = torch.zeros_like(x_t).to(sequence_features.device)
        
        # at inference, we use predicted noise(epsilon) to restore perturbed data sample.
        epsilon_pred = self.denoise(sequence_features, pairwise_features, x_t, timestep)
        
        alpha = self.extract(self.alphas.to(x_t.device), timestep, x_t.shape)
        sqrt_alpha = self.extract(self.sqrt_alphas.to(x_t.device), timestep, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars.to(x_t.device), timestep, x_t.shape)
        sqrt_beta = self.extract(self.sqrt_betas.to(x_t.device), timestep, x_t.shape)
        
        # denoise at time t, utilizing predicted noise
        x_t_minus_1 = 1 / sqrt_alpha * (x_t - (1-alpha)/sqrt_one_minus_alpha_bar*epsilon_pred) + sqrt_beta*z
        
        return x_t_minus_1#.clamp(-1., 1)
                
    def sample(self, src, N):
        # start from random noise vector, NxLx3
        x_t = torch.randn((N, src.shape[1], 3)).to(src.device)
        
        # autoregressively denoise from x_T to x_0
        #     i.e., generate image from noise, x_T

        #first get conditioning
        sequence_features, pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))
        # sequence_features=sequence_features.expand(N,-1,-1)
        # pairwise_features=pairwise_features.expand(N,-1,-1,-1)
        distogram=self.distogram_predictor(pairwise_features).squeeze()
        distogram=distogram.squeeze()[:,:,2:40]*torch.arange(2,40).float().cuda() 
        distogram=distogram.sum(-1)  

        for t in range(self.n_times-1, -1, -1):
            timestep = torch.tensor([t]).repeat_interleave(N, dim=0).long().to(src.device)
            x_t = self.denoise_at_t(x_t, sequence_features, pairwise_features, timestep, t)
        
        # denormalize x_0 into 0 ~ 1 ranged values.
        #x_0 = self.reverse_scale_to_zero_to_one(x_t)
        x_0 = x_t * self.data_std
        return x_0, distogram




class SimpleStructureModule(nn.Module):

    def __init__(self, d_model, nhead, 
                 dim_feedforward, pairwise_dimension, dropout=0.1,
                 ):
        super(SimpleStructureModule, self).__init__()
        #self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = MultiHeadAttention(d_model, nhead, d_model//nhead, d_model//nhead, dropout=dropout)
        #self.cross_attn = MultiHeadAttention(d_model, nhead, d_model//nhead, d_model//nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.pairwise2heads=nn.Linear(pairwise_dimension,nhead,bias=False)
        self.pairwise_norm=nn.LayerNorm(pairwise_dimension)

        #self.distance2heads=nn.Linear(1,nhead,bias=False)
        #self.pairwise_norm=nn.LayerNorm(pairwise_dimension)

        self.activation = nn.GELU()

        
    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(*inputs)
            return inputs
        return custom_forward

    def forward(self, input):
        tgt , src,  pairwise_features, pred_t, src_mask = input
        
        #src = src*src_mask.float().unsqueeze(-1)

        pairwise_bias=self.pairwise2heads(self.pairwise_norm(pairwise_features)).permute(0,3,1,2)

        


        #print(pairwise_bias.shape,distance_bias.shape)

        #pairwise_bias=pairwise_bias+distance_bias


        res=tgt
        tgt,attention_weights = self.self_attn(tgt, tgt, tgt, mask=pairwise_bias, src_mask=src_mask)
        tgt = res + self.dropout1(tgt)
        tgt = self.norm1(tgt)

        # print(tgt.shape,src.shape)
        # exit()

        res=tgt
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = res + self.dropout2(tgt)
        tgt = self.norm2(tgt)


        return tgt






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


diffusion_config=load_config_from_yaml("/kaggle/input/ribonanzanet2-ddpm-v2/diffusion_config.yaml")
rnet_config=load_config_from_yaml("/kaggle/input/ribonanzanet2/pytorch/alpha/1/pairwise.yaml")

model=finetuned_RibonanzaNet(rnet_config,diffusion_config).cuda()



state_dict=torch.load("/kaggle/input/ribonanzanet2-ddpm-v2/RibonanzaNet-DDPM-v2.pt",map_location='cpu')

#get rid of module. from ddp state dict
new_state_dict={}

for key in state_dict:
    new_state_dict[key[7:]]=state_dict[key]

model.load_state_dict(new_state_dict)


from tqdm import tqdm
model.eval()
preds=[]
for i in tqdm(range(len(test_dataset))):
    src=test_dataset[i]['sequence'].long()
    src=src.unsqueeze(0).cuda()
    target_id=test_data.loc[i,'target_id']

    #tmp=[]
    predicted_dm=[]
    #for _ in range(5):
    with torch.no_grad():
        xyz,distogram=model.sample(src,5)

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


submission
submission.to_csv('submission.csv',index=False)


#score val
import pandas as pd
import pandas.api.types
import os
import re

# Function to parse TMscore output
def parse_tmscore_output(output):
    result = {}

    # Extract TM-score based on length of reference structure (second)
    tm_score_match = re.findall(r"TM-score=\s+([\d.]+)", output)[1]
    result['TM-score'] = float(tm_score_match) if tm_score_match else None

    return result

def write_pdb_line(atom_name, atom_serial, residue_name, chain_id, residue_num, x_coord, y_coord, z_coord, occupancy=1.0, b_factor=0.0, atom_type='P'):
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
    line = f"ATOM  {atom_serial:>5d}  {atom_name:<5s} {residue_name:<3s} {residue_num:>3d}    {x_coord:>8.3f}{y_coord:>8.3f}{z_coord:>8.3f}{occupancy:>6.2f}{b_factor:>6.2f}           {atom_type}\n"
    return line

def write2pdb(df, xyz_id, pdb_path):
    resolved_cnt=0
    with open(pdb_path, "w") as pdb_file:
        for _, row in df.iterrows():
            x_coord=row[f"x_{xyz_id}"]
            y_coord=row[f"y_{xyz_id}"]
            z_coord=row[f"z_{xyz_id}"]

            if x_coord>-1e17 and y_coord>-1e17 and z_coord>-1e17:
            #if True:
                resolved_cnt+=1
                pdb_line = write_pdb_line(
                    atom_name="C1'", 
                    atom_serial=int(row["resid"]), 
                    residue_name=row['resname'], 
                    chain_id='0', 
                    residue_num=int(row["resid"]), 
                    x_coord=x_coord, 
                    y_coord=y_coord, 
                    z_coord=z_coord,
                    atom_type="C"
                )
                pdb_file.write(pdb_line)
    return resolved_cnt


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    Computes the TM-score between predicted and native RNA structures using USalign.

    This function evaluates the structural similarity of RNA predictions to native structures
    by computing the TM-score. It uses USalign, a structural alignment tool, to compare
    the predicted structures with the native structures.

    Workflow:
    1. Copies the USalign binary to the working directory and grants execution permissions.
    2. Extracts the `pdb_id` from the `ID` column of both the solution and submission DataFrames.
    3. Iterates over each unique `pdb_id`, grouping the native and predicted structures.
    4. Writes PDB files for native and predicted structures.
    5. Runs USalign on each predicted-native pair and extracts the TM-score.
    6. Computes the highest TM-score per target and returns aggregated results.

    Args:
        solution (pd.DataFrame): A DataFrame containing the native RNA structures.
        submission (pd.DataFrame): A DataFrame containing the predicted RNA structures.
        row_id_column_name (str): The name of the column containing unique row identifiers.

    Returns:
        tuple:
            - results (list): The highest TM-score for each `pdb_id`.
            - results_per_sub (list): TM-scores for each predicted-native pair.
            - outputs (list): Raw output logs from USalign for debugging.
    '''

    os.system("cp /kaggle/input/usalign/USalign /kaggle/working/")
    os.system("sudo chmod u+x /kaggle/working//USalign")


    # Extract pdb_id from ID (pdb_resid)
    solution["pdb_id"] = solution["ID"].apply(lambda x: x.split("_")[0])
    submission["pdb_id"] = submission["ID"].apply(lambda x: x.split("_")[0])

    #fix pdb_ids comment out later
    # solution.loc[solution['pdb_id']=="R1138v1",'pdb_id']='R1138'
    # solution.loc[solution['pdb_id']=="R1117",'pdb_id']='R1117v2'
    
    results=[]
    outputs=[]
    results_per_sub=[]
    # Iterate through each pdb_id and generate PDB files for both clean and corrupted data
    for pdb_id, group_native in solution.groupby("pdb_id"):
        group_predicted = submission[submission["pdb_id"] == pdb_id]
        #print(group_native,group_predicted)
        # Define output file paths
        # clean_pdb_path = os.path.join(output_folder, f"{pdb_id}_C3_clean.pdb")
        # corrupted_pdb_path = os.path.join(output_folder, f"{pdb_id}_C3_corrupted.pdb")
        native_pdb=f'native.pdb'
        predicted_pdb=f'predicted.pdb'

        all_scores=[]
        for pred_cnt in range(1,6):
            tmp=[]
            for native_cnt in range(1,41):
                # Write solution PDB
                resolved_cnt=write2pdb(group_native, native_cnt, native_pdb)
                
                # Write predicted PDB
                _=write2pdb(group_predicted, pred_cnt, predicted_pdb)

                if resolved_cnt>0:
                    command = f"/kaggle/working/USalign {predicted_pdb} {native_pdb} -atom \" C1'\""
                    output = os.popen(command).read()
                    outputs.append(output)
                    parsed_data = parse_tmscore_output(output)
                    tmp.append(parsed_data['TM-score'])
                    
            all_scores.append(max(tmp))
        # print(output)
        # stop
        print(pdb_id)
        print(all_scores)
        results_per_sub.append(all_scores)
        results.append(max(all_scores))
    
    print(results)
    #return sum(results)/len(results), outputs
    return results, results_per_sub, outputs
    #return outputs

if 'R1107' in set(test_data['target_id']):
    solution=pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_labels.csv")
    
    scores,results_per_sub,outputs=score(solution,submission,'ID')
    print(np.mean(scores))

