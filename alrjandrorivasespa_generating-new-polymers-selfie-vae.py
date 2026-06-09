!pip install -q rdkit selfies


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import selfies as sf
import torch
import torch.optim as optim
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Draw, MolFromSmiles, rdMolDescriptors
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from torch import nn
from torch.functional import F
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

# The vocabulary used by the SELFIES tokenizer
vocab = {'<PAD>': 0, '<UNK>': 1, '[=P]': 2, '[\\N+1]': 3, '[O]': 4, '[Branch2]': 5, '[I with 2 bond(s) - a max. of 1 bond(s) was specified]': 6, 
         '[\\Si]': 7, '[/SiH1]': 8, '[P with 7 bond(s) - a max. of 5 bond(s) was specified]': 9, '[S]': 10, '[Fe]': 11, '[[PH1]': 12, 
         '[PH1]': 13, '[=P+1]': 14, '[/I]': 15, '[\\SiH1]': 16, '[NH1+1]': 17, '[P+]': 18, 
         '[#Branch2]': 19, '[NH2+1]': 20, '[#Se]': 21, '[Ge]': 22, '[/SH1]': 23, '[Sn]': 24, '[=Zn]': 25, '[/Cl]': 26, '[[IH1]': 27,
         '[\\B]': 28, '[=B]': 29, '[/O]': 30, '[\\C@H1]': 31, '[#N+1]': 32, '[GeH1]': 33, '[/S]': 34, '[P]': 35, '[#PH1]': 36,
         '[=Branch1]': 37, '[SiH]': 38, '[C@@]': 39, '[Ca]': 40, '[I with 3 bond(s) - a max. of 1 bond(s) was specified]': 41,
         '[2H]': 42, '[Pb]': 43, '[C@@H1]': 44, '[/Sn]': 45, '[=SiH1]': 46, '[IH2]': 47, '[/C@H1]': 48, '[n+]': 49, '[Cl]': 50,
         '[O-]': 51, '[N+1]': 52, '[#S]': 53, '[\\SH1]': 54, '[\\Cd]': 55, '[SH1]': 56, '[\\C]': 57, '[\\I]': 58, '[=Si]': 59,
         '[/Br]': 60, '[/P+1]': 61, '[\\N-1]': 62, '[F]': 63, '[#SH1]': 64, '[/2H]': 65, '[=Branch2]': 66, '[SiH1]': 67, 
         '[/N+1]': 68, '[Ni]': 69, '[\\K]': 70, '[\\Ca]': 71, '[#Ge]': 72, '[=PH1]': 73, '[=Fe]': 74, '[-\\Ring1]': 75, '[/N]': 76,
         '[SiH2]': 77, '[[IH2]': 78, '[\\F]': 79, '[O-1]': 80, '[B]': 81, '[#C]': 82, '[/C]': 83, '[N]': 84, '[Ring1]': 85, 
         '[\\2H]': 86, '[=Se]': 87, '[Na]': 88, '[=NH2+1]': 89, '[C@]': 90, '[I]': 91, '[/Si]': 92, '[/Zn]': 93, '[/NH1+1]': 94, 
         '[\\Cl]': 95, '[\\S]': 96, '[/Se]': 97, '[/B]': 98, '[/PH1]': 99, '[SiH3]': 100, '[Ring2]': 101, '[=As]': 102,
         '[\\PH1]': 103, '[/As]': 104, '[NH3+1]': 105, '[#P]': 106, '[=N+1]': 107, '[nH]': 108, '[/F]': 109, '[=NH1+1]': 110,
         '[=Sn]': 111, '[=Ring1]': 112, '[Co]': 113, '[Te]': 114, '[K]': 115, '[\\Na]': 116, '[-/Ring1]': 117, '[Br]': 118,
         '[=N]': 119, '[Se]': 120, '[\\N]': 121, '[#B]': 122, '[/Co]': 123, '[N-1]': 124, '[\\Se]': 125, '[=Ca]': 126, 
         '[NH+]': 127, '[/N-1]': 128, '[Si]': 129, '[/SiH3]': 130, '[=Ring2]': 131, '[Branch1]': 132, '[=S]': 133, '[\\Br]': 134,
         '[=SH1]': 135, '[SH]': 136, '[IH]': 137, '[\\O]': 138, '[/P]': 139, '[As]': 140, '[=Ge]': 141, '[C@H1]': 142, '[#Si]': 143,
         '[\\Sn]': 144, '[C]': 145, '[Zn]': 146, '[N+]': 147, '[=N-1]': 148, '[NH2+]': 149, '[=C]': 150, '[-\\Ring2]': 151,
         '[#Branch1]': 152, '[-/Ring2]': 153, '[NH1]': 154, '[\\P]': 155, '[=SiH3]': 156, '[/Ge]': 157, '[#N]': 158, '[=O]': 159,
         '[N-]': 160, '[Cd]': 161, '[P+1]': 162, '[#SiH1]': 163, '[PH]': 164}


import selfies as sf
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def smiles_to_selfies(smiles: str) -> str:
    """
    Encodes a SMILES string into SELFIES format.
    Replaces '*' with 'I' to allow conversion, as SELFIES doesn't support '*'.
    """
    smiles = smiles.replace('*', 'I')
    try:
        return sf.encoder(smiles)
    except Exception as e:
        return f"Error processing SMILES '{smiles}': {e}"

def selfies_to_smiles(selfies: str) -> str:
    """
    Decodes a SELFIES string into SMILES format.
    Replaces 'I' back to '*' after decoding.
    """
    try:
        return sf.decoder(selfies).replace('I', '*')
    except Exception as e:
        return f"Error processing SELFIES '{selfies}': {e}"

def parallel_processing(smiles_list, conversion_function, num_processes: int = None):

    if num_processes is None:
        num_processes = cpu_count()

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(conversion_function, smiles_list), total=len(smiles_list)))
    return results

# Apply the conversions
train['SELFIES'] = parallel_processing(train['SMILES'], smiles_to_selfies)
test['SELFIES'] = parallel_processing(test['SMILES'], smiles_to_selfies)



import torch
import re
from collections import Counter

def detokenize_selfies(tokenized_tensor, vocab_dict):
    # create reverse tokenizer
    id_to_token = {i: token for token, i in vocab_dict.items()}
    
    sequences = []
    for seq_ids in tokenized_tensor:
        tokens = []
        for token_id in seq_ids:
            token = id_to_token[token_id.item()]
            if token == '<PAD>':
                break  # For first padding
            if token != '<UNK>':
                tokens.append(token)
        sequences.append(''.join(tokens))
    
    return sequences


def tokenize_selfies(sequences, vocab=None, max_len=310):
    # Create a vocab if not provided 
    if vocab is None:
        valid_sequences = [str(seq) for seq in sequences if seq is not None and str(seq) != 'nan']
        all_tokens = []
        for seq in valid_sequences:
            tokens = re.findall(r'\[[^\]]+\]', seq)
            all_tokens.extend(tokens)
        
        vocab = ['<PAD>', '<UNK>'] + list(set(all_tokens))

    token_to_id = {token: i for i, token in enumerate(vocab)}
    
    # Tokenization of the sequences
    tokenized = []
    for seq in tqdm(sequences, total=len(sequences)):
        if seq is None or str(seq) == 'nan':
            ids = [0] * max_len  # Only padding
        else:
            tokens = re.findall(r'\[[^\]]+\]', str(seq))
            ids = [token_to_id.get(token, 1) for token in tokens]  # 1 = <UNK>
            ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
        
        tokenized.append(ids)
    
    return torch.tensor(tokenized, dtype=torch.long), token_to_id



import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Self-attention mechanism combined with convolutions"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.query = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.key = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.value = nn.Conv1d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable parameter to control attention intensity
        
    def forward(self, x):
        batch_size, C, length = x.size()
        
        # Linear projections
        proj_query = self.query(x).permute(0, 2, 1)  # B x L x C'
        proj_key = self.key(x)  # B x C' x L
        energy = torch.bmm(proj_query, proj_key)  # B x L x L
        attention = F.softmax(energy, dim=-1)  # B x L x L
        
        proj_value = self.value(x).permute(0, 2, 1)  # B x L x C
        out = torch.bmm(attention, proj_value)  # B x L x C
        out = out.permute(0, 2, 1)  # B x C x L
        
        return self.gamma * out + x
    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.relu  = nn.LeakyReLU(0.1)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)
    

class SelfiesVAE(nn.Module):
    def __init__(self, z_dim=32, vocab_size=len(vocab), max_size=310, emb_mol=64): 
        super().__init__()
        # Only retain the molecule embedding
        self.molecule_emb = nn.Embedding(vocab_size, emb_mol)
        self.max_size = max_size

        self.encoder = nn.Sequential(
            ResidualBlock(emb_mol, 256),     
            nn.MaxPool1d(kernel_size=4),     
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            SelfAttention(256),
            
            ResidualBlock(256, 128),         
            nn.MaxPool1d(kernel_size=4),     
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            SelfAttention(128),
            
            nn.Flatten()                
        )
        
        # Layers to obtain latent parameters (adjusted for new dimension)
        self.fc_mu = nn.Linear(128 * 19, z_dim)
        self.fc_logvar = nn.Linear(128 * 19, z_dim)

        # Decoder: reconstructs from latent space
        self.fc_dec = nn.Linear(z_dim, 128 * 19)
        
        self.decoder = nn.Sequential(
            ResidualBlock(128, 128),          
            nn.Upsample(scale_factor=4, mode='linear'), 
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            SelfAttention(256),
            
            ResidualBlock(256, 256),         
            nn.Upsample(scale_factor=4, mode='linear'), 
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            SelfAttention(256),
            
            # Final adjustment to reach exactly 310 sequence lenght(max_size)
            nn.Conv1d(256, vocab_size, kernel_size=3, padding=1), 
            nn.Upsample(size=max_size, mode='linear')  
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        # Only process the molecule embedding
        emb_x = self.molecule_emb(x).transpose(1, 2) 
        enc = self.encoder(emb_x)

        # Project to mu and logvar
        mu = self.fc_mu(enc)
        logvar = self.fc_logvar(enc)
        
        return mu, logvar

    def decode(self, z):
        
        x = self.fc_dec(z)         # (batch, 128*19)
        x = x.view(-1, 128, 19)    # Reshape to (batch, 128, 19)
        x = self.decoder(x)        # (batch, vocab_size, 310)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SelfiesVAE(vocab_size=len(vocab), max_size=310).to(device)
model.load_state_dict(torch.load('/kaggle/input/vae-weights-finetuned-for-kaggle-polymers/kaggle-vae-pretrain25.pth', weights_only=True))



from rdkit import Chem
from rdkit.Chem import DataStructs, rdMolDescriptors
RDLogger.DisableLog('rdApp.*')


def calculate_similarity(smile1, smile2):
    morgan_generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)

    mol1 = Chem.MolFromSmiles(smile1)
    mol2 = Chem.MolFromSmiles(smile2)
    
    if mol1 is None or mol2 is None:
        return 0.0
    
    fp1 = morgan_generator.GetFingerprint(mol1)
    fp2 = morgan_generator.GetFingerprint(mol2)

    return DataStructs.TanimotoSimilarity(fp1, fp2)


def generate_new_polymers(base_smile, n_generations=10000, noise_factor=0.15):
    model.eval()
    batch_size = 128
    all_smiles = set()

    with torch.no_grad():
        base_tokenized, _ = tokenize_selfies([smiles_to_selfies(base_smile)], vocab=vocab)
        mu, logvar = model.encode(base_tokenized.to(device))
        z_base = model.reparameterize(mu, logvar)

        for _ in tqdm(range(0, n_generations, batch_size), total=(n_generations // batch_size), desc='Generating Molecules'):
            current_batch_size = min(batch_size, n_generations - len(all_smiles))
            if current_batch_size <= 0: break

            z_new = z_base.repeat(current_batch_size, 1) + torch.randn_like(z_base.repeat(current_batch_size, 1)) * noise_factor
            decoded = model.decode(z_new)
            tokens = F.softmax(decoded, dim=1).argmax(dim=1).cpu().numpy()
            detok = detokenize_selfies(tokens, vocab)
            smiles_batch = [selfies_to_smiles(s) for s in detok]
            all_smiles.update(smiles_batch)

    # Filter Invalid Molecules
    valid_smiles = [s for s in all_smiles if s and Chem.MolFromSmiles(s)]
    print(f'Valid Unique Molecules: {len(valid_smiles)} -> {len(valid_smiles)/n_generations*100:.2f}%')

    # Calculate similarity and visualize top 30 simimilar polymers
    similarities = [(s, calculate_similarity(base_smile, s)) for s in tqdm(valid_smiles, total=len(valid_smiles), desc='Calculating Similarities')]
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    top_smiles = [s for s, _ in similarities[:30]]
    labels = [f"Similarity: {sim:.3f}" for _, sim in similarities[:30]]
    mols = [Chem.MolFromSmiles(s) for s in top_smiles]
    
    return similarities, Draw.MolsToGridImage(mols, molsPerRow=6, legends=labels)
    

# Test Molecule selected to generate new similar polymers
goal_smile = test.loc[0, 'SMILES']

# Generation of new 10k new polimers with low noise
generated_smiles, img = generate_new_polymers(goal_smile, n_generations=10000, noise_factor=0.075)
img


# Generation of 50k new polimers with high noise
generated_smiles, img = generate_new_polymers(goal_smile, n_generations=50000, noise_factor=0.25)
img

