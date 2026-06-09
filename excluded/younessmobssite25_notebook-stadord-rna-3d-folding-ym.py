!ls /kaggle/input/my-ribonanzanet2d-final-2


model = finetuned_RibonanzaNet(load_config_from_yaml("/kaggle/input/my-ribonanzanet2d-final-2/pairwise.yaml"), pretrained=False).to(device)


!ls /kaggle/input/ribonanzanet-weights
!ls /kaggle/input/ribonanzanet-3d-finetune


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import sys
import yaml

# Installer les dépendances nécessaires
!pip install pyyaml

# Configurer la gestion de la mémoire PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Définir le device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du device : {device}")

# Vérifier les fichiers dans votre dataset personnalisé
print("Contenu de /kaggle/input/my-ribonanzanet2d-final-2 :")
!ls /kaggle/input/my-ribonanzanet2d-final || echo "Dataset my-ribonanzanet2d-final non trouvé. Veuillez l'ajouter à votre notebook."

# Ajouter le chemin vers RibonanzaNet
sys.path.append("/kaggle/input/my-ribonanzanet2d-final-2")

# Importer RibonanzaNet
try:
    from network import RibonanzaNet
    print("Imported RibonanzaNet from 'network'")
except ModuleNotFoundError:
    try:
        from Network import RibonanzaNet
        print("Imported RibonanzaNet from 'Network'")
    except ModuleNotFoundError:
        print("Erreur : 'network' ou 'Network' non trouvé. Vérifiez le nom du module avec :")
        print("!ls /kaggle/input/my-ribonanzanet2d-final-2")
        raise

# Définir les classes nécessaires pour charger la configuration
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries = entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)

# Définir une version affinée de RibonanzaNet pour la prédiction 3D
class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, config, pretrained=False):
        config.dropout = 0.2
        super(finetuned_RibonanzaNet, self).__init__(config)
        if pretrained:
            self.load_state_dict(torch.load("/kaggle/input/ribonanzanet-weights/RibonanzaNet.pt", map_location='cpu'))
        self.dropout = nn.Dropout(0.0)
        self.xyz_predictor = nn.Linear(256, 3)

    def forward(self, src):
        sequence_features, pairwise_features = self.get_embeddings(src, torch.ones_like(src).long().to(src.device))
        xyz = self.xyz_predictor(sequence_features)
        return xyz

# Vérifier les fichiers des poids
print("Contenu de /kaggle/input/ribonanzanet-weights :")
!ls /kaggle/input/ribonanzanet-weights || echo "Dataset ribonanzanet-weights non trouvé. Veuillez l'ajouter à votre notebook."

print("Contenu de /kaggle/input/ribonanzanet-3d-finetune :")
!ls /kaggle/input/ribonanzanet-3d-finetune || echo "Dataset ribonanzanet-3d-finetune non trouvé. Veuillez l'ajouter à votre notebook."

# Charger le modèle RibonanzaNet
model = finetuned_RibonanzaNet(load_config_from_yaml("/kaggle/input/my-ribonanzanet2d-final-2/pairwise.yaml"), pretrained=False).to(device)
model.load_state_dict(torch.load("/kaggle/input/ribonanzanet-3d-finetune/RibonanzaNet-3D.pt"))

# Étape 1 : Charger les données de test
test = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')

# Définir la fonction is_valid_sequence
valid_nucleotides = set('ACGU')
def is_valid_sequence(seq):
    return all(nucleotide in valid_nucleotides for nucleotide in seq)

test['is_valid'] = test['sequence'].apply(is_valid_sequence)
test_clean = test[test['is_valid']].drop(columns=['is_valid'])

# Créer le dataset de test (sans structure 2D pour l'instant)
class RNADataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokens = {nt: i for i, nt in enumerate('ACGU')}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]['sequence']
        sequence_encoded = [self.tokens[nt] for nt in sequence]
        sequence_encoded = torch.tensor(sequence_encoded, dtype=torch.long)
        return {'sequence': sequence_encoded}

test_dataset = RNADataset(test_clean)

# Définir lengths_test
lengths_test_list = [len(seq) for seq in test_clean['sequence']]
lengths_test = torch.tensor(lengths_test_list, dtype=torch.long).to(device)

# Calculer le nombre total de nucléotides attendu
total_nucleotides = test_clean['sequence'].str.len().sum()
print(f"Nombre total de nucléotides attendu : {total_nucleotides}")

# Prédire les coordonnées
model.eval()
preds = []
for i in range(len(test_dataset)):
    src = test_dataset[i]['sequence'].long()
    src = src.unsqueeze(0).to(device)

    # Générer 5 prédictions différentes
    tmp = []
    # 4 prédictions en mode train pour introduire de la variabilité
    model.train()
    for _ in range(4):
        with torch.no_grad():
            xyz = model(src).squeeze()
        tmp.append(xyz.cpu().numpy())

    # 1 prédiction en mode eval pour une prédiction stable
    model.eval()
    with torch.no_grad():
        xyz = model(src).squeeze()
    tmp.append(xyz.cpu().numpy())

    tmp = np.stack(tmp, axis=0)  # Shape: (5, seq_len, 3)
    preds.append(tmp)

    # Vider le cache de la mémoire GPU
    torch.cuda.empty_cache()

# Construire le fichier submission.csv
data = []
for i in range(len(test_clean)):
    sequence = test_clean.iloc[i]['sequence']
    target_id = test_clean.iloc[i]['target_id']
    seq_len = len(sequence)
    
    for j in range(seq_len):
        row = [f"{target_id}_{j+1}", sequence[j], j+1]
        for k in range(5):  # 5 prédictions
            x, y, z = preds[i][k][j]
            row.extend([x, y, z])
        data.append(row)

# Définir les colonnes
columns = ['ID', 'resname', 'resid']
for i in range(1, 6):
    columns.extend([f'x_{i}', f'y_{i}', f'z_{i}'])

# Créer le DataFrame
submission_df = pd.DataFrame(data, columns=columns)

# Vérifier les NaN
print("Vérification des valeurs NaN dans submission_df :")
print(submission_df.isna().sum())
submission_df = submission_df.fillna(0)

# Vérifier le nombre de lignes
print(f"Nombre de lignes dans submission_df : {len(submission_df)}")
if len(submission_df) != total_nucleotides:
    print("Erreur : Le nombre de lignes ne correspond pas au nombre de nucléotides attendu !")

# Vérifier les colonnes
expected_columns = ['ID', 'resname', 'resid'] + [f'{coord}_{i}' for i in range(1, 6) for coord in ['x', 'y', 'z']]
print("Colonnes attendues :", expected_columns)
print("Colonnes dans submission_df :", submission_df.columns.tolist())

# Sauvegarder le fichier
submission_df.to_csv('submission.csv', index=False)
print("Soumission créée : submission.csv")

# Afficher un aperçu
print("Aperçu de submission.csv :")
print(submission_df.head())


!ls /kaggle/input/my-ribonanzanet2d-final -R




