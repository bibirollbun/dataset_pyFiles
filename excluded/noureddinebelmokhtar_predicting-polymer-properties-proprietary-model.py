# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install /kaggle/input/rdkit-dataset/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import re
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdmolops import GetDistanceMatrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import linecache
import math
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# === Electronegativity table of common atoms ===
electronegativity_table = {
    'C': 2.55, 'N': 3.04, 'O': 3.44, 'S': 2.58, 'P': 2.19
}

# === 1. Calculating the backbone length (longest linear path) ===
def longest_linear_backbone(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return 0
    longest = 0
    for i in range(n_atoms):
        visited = set()
        stack = [(i, 0)]
        while stack:
            current, length = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            longest = max(longest, length)
            for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                nbr_idx = neighbor.GetIdx()
                if nbr_idx not in visited:
                    stack.append((nbr_idx, length + 1))
    return longest

# === 2. Calculation of the backbone length (longest linear path)2. 
#Number of branch points (≥ 3 neighbors) ===
def count_branch_points(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return sum(1 for atom in mol.GetAtoms() if atom.GetDegree() >= 3)

# === 3. Counting connections by type ===
def count_bonds(smiles, bond_type):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return sum(1 for bond in mol.GetBonds() if bond.GetBondType() == bond_type)

def count_single_bonds(smiles):
    return count_bonds(smiles, Chem.rdchem.BondType.SINGLE)

def count_double_bonds(smiles):
    return count_bonds(smiles, Chem.rdchem.BondType.DOUBLE)

def count_triple_bonds(smiles):
    return count_bonds(smiles, Chem.rdchem.BondType.TRIPLE)

# === 4. Counting aromatic and non-aromatic rings ===
def count_aromatic_rings(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    ri = mol.GetRingInfo()
    aromatic_rings = 0
    for ring in ri.AtomRings():
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            aromatic_rings += 1
    return aromatic_rings

def count_non_aromatic_rings(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    ri = mol.GetRingInfo()
    non_aromatic_rings = 0
    for ring in ri.AtomRings():
        if any(not mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            non_aromatic_rings += 1
    return non_aromatic_rings

# === 5. Molar mass ===
def compute_molar_mass(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.MolWt(mol)

# === 6. Polymerization scores ===
def compute_two_polymerization_scores(smiles):
    scores = []

    matches = re.finditer(r'([=#]?)([A-Z][a-z]?|\*)\*', smiles)
    for m in matches:
        bond, atom = m.group(1), m.group(2)
        weight = {'': 1, '=': 2, '#': 3}.get(bond, 1)
        if atom == '*':
            score = weight * 1
        elif atom in electronegativity_table:
            score = weight * electronegativity_table[atom]
        else:
            score = weight
        scores.append(score)

    matches2 = re.finditer(r'([=#]?)(\()(\*)', smiles)
    for match in matches2:
        bond_type = match.group(1)
        open_paren_pos = match.start(2)
        atom_before = ''
        i = open_paren_pos - 1
        while i >= 0 and smiles[i].islower():
            atom_before = smiles[i] + atom_before
            i -= 1
        if i >= 0 and smiles[i].isalpha():
            atom_before = smiles[i] + atom_before
        weight = {'': 1, '=': 2, '#': 3}.get(bond_type, 1)
        if atom_before in electronegativity_table:
            score = weight * electronegativity_table[atom_before]
        else:
            star_pos = match.start(3)
            atom_after = ''
            j = star_pos + 1
            while j < len(smiles) and smiles[j] in ['*', '(', ')']:
                j += 1
            if j < len(smiles):
                atom_after = smiles[j]
                if j + 1 < len(smiles) and smiles[j+1].islower():
                    atom_after += smiles[j+1]
            if atom_after in electronegativity_table:
                score = weight * electronegativity_table[atom_after]
            else:
                score = weight
        scores.append(score)

    all_star_positions = [m.start() for m in re.finditer(r'\*', smiles)]
    used_positions = set()
    for m in re.finditer(r'([=#]?)([A-Z][a-z]?|\*)\*', smiles):
        used_positions.add(m.end() - 1)
    for m in re.finditer(r'([=#]?)(\()(\*)', smiles):
        used_positions.add(m.end() - 1)

    for pos in all_star_positions:
        if pos not in used_positions:
            scores.append(1)

    scores_sorted = sorted(scores, reverse=True)
    while len(scores_sorted) < 2:
        scores_sorted.append(0.0)

    return scores_sorted[0], scores_sorted[1]

# === 7. Chemical groups and their electronegativity (excluding halogens) ===
groupements = {
    'hydroxyle_OH': ('[OX2H]', 3.44),
    'carboxyle_COOH': ('C(=O)[OX2H1]', 3.44),
    'amine_NH2': ('[NX3;H2,H1;!$(NC=O)]', 3.04),
    'ether_RO_R': ('[OD2]([#6])[#6]', 3.44),
    'ester_COO': ('C(=O)O[#6]', 3.44),
    'phenyl': ('c1ccccc1', 2.55),
    'amide': ('C(=O)N', 3.04),
    'nitrate': ('[$([ON]=O)]', 3.44),
    'thiol_SH': ('[SH]', 2.58),
    'alkyne_CC_triple': ('C#C', 2.55),
}

def compute_group_electronegativity(smiles, smarts, electronegativity):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    pattern = Chem.MolFromSmarts(smarts)
    matches = mol.GetSubstructMatches(pattern)
    return len(matches) * electronegativity

# === Fonction average distance between branch pointst ===
def average_branch_points_distance(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    branch_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetDegree() >= 3]
    n = len(branch_indices)
    if n <= 1:
        return 0.0
    dist_matrix = GetDistanceMatrix(mol)
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            distances.append(dist_matrix[branch_indices[i], branch_indices[j]])
    return np.mean(distances)

# === Fonction distance moyenne entre points de polymérisation (atomes '*') ===
def average_polymerization_points_distance(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    poly_points_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
    n = len(poly_points_indices)
    if n <= 1:
        return 0.0
    dist_matrix = GetDistanceMatrix(mol)
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            distances.append(dist_matrix[poly_points_indices[i], poly_points_indices[j]])
    return np.mean(distances)

# === Fonction pour électro-négativité combinée des halogènes ===
def compute_halogenes_electronegativity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    halogens = {
        'fluor': ('[F]', 3.98),
        'chlore': ('[Cl]', 3.16),
        'brome': ('[Br]', 2.96),
        'iode': ('[I]', 2.66),
    }
    total = 0
    for name, (smarts, elec) in halogens.items():
        pattern = Chem.MolFromSmarts(smarts)
        matches = mol.GetSubstructMatches(pattern)
        total += len(matches) * elec
    return total

# Suppression uniquement de la colonne encombrement (calculée mais pas gardée)
def compute_encombrement(row):
    if row['longueur'] and row['longueur'] != 0:
        return row['ramifications'] / row['longueur']
    else:
        return 0
        
# === Chargement des données ===
list_datasets = ['/kaggle/input/neurips-open-polymer-prediction-2025/test.csv',
                #'/kaggle/input/datasets-targets/train_Density.csv',
                 #'/kaggle/input/datasets-targets/train_FFV.csv',
                 #'/kaggle/input/datasets-targets/train_Rg.csv',
                 #'/kaggle/input/datasets-targets/train_Tc.csv',
                 #'/kaggle/input/datasets-targets/train_Tg.csv'
]
list_fils = ['test',
            'Density',
            'FFV',
            'Rg',
            'Tc',
            'Tg'
]
list_num_features=[23,28]
for Num in list_num_features:
    for i in range(0, len(list_datasets)):
        df_tg = pd.read_csv(list_datasets[i])
        
        # === Application des fonctions ===
        
        df_tg['molar_mass'] = df_tg['SMILES'].apply(compute_molar_mass)
        df_tg['longueur'] = df_tg['SMILES'].apply(longest_linear_backbone)
        df_tg['ramifications'] = df_tg['SMILES'].apply(count_branch_points)
        df_tg['liaisons_simples'] = df_tg['SMILES'].apply(count_single_bonds)
        df_tg['liaisons_doubles'] = df_tg['SMILES'].apply(count_double_bonds)
        df_tg['liaisons_triples'] = df_tg['SMILES'].apply(count_triple_bonds)
        df_tg['cycles_aromatiques'] = df_tg['SMILES'].apply(count_aromatic_rings)
        df_tg['cycles_non_aromatiques'] = df_tg['SMILES'].apply(count_non_aromatic_rings)
        df_tg[['polymerization_score_1', 'polymerization_score_2']] = df_tg['SMILES'].apply(
            lambda smi: pd.Series(compute_two_polymerization_scores(smi))
        )
        df_tg['distance_moy_branch'] = df_tg['SMILES'].apply(average_branch_points_distance)
        df_tg['distance_moy_polymerization_points'] = df_tg['SMILES'].apply(average_polymerization_points_distance)
        
        
        
        df_tg['encombrement'] = df_tg.apply(compute_encombrement, axis=1)
        df_tg.drop(columns=['encombrement'], inplace=True)
        
        # Calcul électro-négativité des groupements chimiques (hors halogènes)
        for nom, (smarts, elec) in groupements.items():
            df_tg[nom] = df_tg['SMILES'].apply(lambda smi: compute_group_electronegativity(smi, smarts, elec))
        
        # Calcul électro-négativité combinée des halogènes
        df_tg['halogenes'] = df_tg['SMILES'].apply(compute_halogenes_electronegativity)
        
        # Suppression colonnes individuelles halogènes si elles existaient (sécurité)
        for halogen in ['fluor', 'chlore', 'brome', 'iode']:
            if halogen in df_tg.columns:
                df_tg.drop(columns=[halogen], inplace=True)
        
        # === Colonnes finales ===
        colonnes = [
            'id', 'molar_mass', 'polymerization_score_1', 'polymerization_score_2',
            'longueur', 'ramifications', 'distance_moy_branch', 'distance_moy_polymerization_points',
            'liaisons_simples', 'liaisons_doubles', 'liaisons_triples',
            'cycles_aromatiques', 'cycles_non_aromatiques',
        ] + list(groupements.keys()) + ['halogenes']
        df_features = df_tg[colonnes].copy()
        
        if Num==28:
            df_features['flexibility'] = df_features['liaisons_simples'] + df_features['ether_RO_R'] + df_features['cycles_non_aromatiques']
            df_features['flexibility'] = df_features['flexibility']
            df_features['heavy_atoms'] = df_features['halogenes'] + df_features['nitrate'] + df_features['phenyl'] + df_features['ester_COO']
            df_features['polarity'] = df_features['hydroxyle_OH'] + df_features['carboxyle_COOH'] + df_features['amine_NH2'] + df_features['amide'] + df_features['nitrate'] + df_features['thiol_SH']
            df_features['polarity'] = df_features['polarity']
            df_features['rigidity'] = df_features['liaisons_doubles'] + df_features['liaisons_triples'] + df_features['cycles_aromatiques'] + df_features['phenyl']
            df_features['rigidity'] = df_features['rigidity'] 
            df_features['structure_score'] = df_features['molar_mass'] + df_features['longueur'] - df_features['ramifications']
        
        
        # === Sauvegarde ===
        df_features.to_csv(f'/kaggle/working/df_{list_fils[i]}_features_{Num}.csv', index=False)
        print(f'/kaggle/working/df_{list_fils[i]}_features.csv')
# === Aperçu ===
print(df_features.head())


def normalization(param, dataset):
    
    if param=='Tg' or param=='FFV':   # or param=='Density'
        if dataset == 'test':
            df = pd.read_csv(f'/kaggle/working/df_test_features_28.csv')
            txt_fil =f"/kaggle/working/df_test_{param}_features.txt"
        else:
            df = pd.read_csv(f'/kaggle/working/df_{param}_features.csv')
            txt_fil =f"/kaggle/working/df_train_{param}_features.txt"
        filename = f'/kaggle/input/scaler-train-val-datasets-scaled/df_{param}_features.joblib'
        
    else :
       if dataset == 'test':
            df = pd.read_csv(f'/kaggle/working/df_test_features_23.csv')
            txt_fil =f"/kaggle/working/df_test_{param}_features.txt"
       else:
            df = pd.read_csv(f'/kaggle/working/df_{param}_features.csv')
            txt_fil =f"/kaggle/working/df_train_{param}_features.txt"
       filename = f'/kaggle/input/polymers-scalers/{param}_scaler.joblib'
        
    X_test = df.drop(columns=['id'])
    id_ = df[['id']]

    # Load the scaler from the file
    loaded_scaler = joblib.load(filename)
    print(f"Scaler loaded from '{filename}'")

    X_test_scaled = loaded_scaler.transform(X_test)
    
    # Convert back to DataFrames for saving
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # Step 4: Save to TXT (tab-separated or comma-separated)
    X_test_scaled.to_csv(txt_fil, header=None, index=False, sep='\t')

    print("Datasets saved as CSV and TXT.")
    print(txt_fil)
    return id_
params = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
for param in params:
    dataset = 'test'
    id_ = normalization(param, dataset)
"""
for param in params:
    dataset = 'train'
    id_ = normalization(param, dataset)
"""


def neuron(type_fonc, n1):
    try:
        if type_fonc == 0:  #sigmoid
            a = 1/(1+math.exp(-n1))
        elif type_fonc == 1:   #identity
            a = n1
        elif type_fonc == 2:  #tanh
            a = math.tanh(n1) #2/(1+math.exp(-2 *n1))-1
        elif type_fonc == 3:  #relu
                if n1<=0:
                    a = 0
                if n1>0:
                    a = n1
        elif type_fonc == 4:  #leaky_relu
            if n1 > 0 :
                a = n1
            else:
                a = 0.01 * n1
    except OverflowError:
        a = float('inf')

    return (a)

def valid(test_fill, param):
    
    nn_fille_2=test_fill
    prediction = []
    final = 0
    r = 0
    for df in pd.read_csv(nn_fille_2, sep = "\t", header = None, iterator=True, chunksize =1 ):
 
        incrimente=0
        n_output_lear = 0
        k = 0
        j = 0
        i = 0
        
        var=0
        h_lears = None
        
        input_1 = 0
        outt = None
        n11=0
        type_f_a = []
        type_f_w_b = []
        n_neron = []
        
        #fichier1[0]=[]
        output_nerons_num=[]
        p=[]

        if param=='Tg' or param=='FFV':   # or param=='Density'
            file_path = f"/kaggle/input/polymers-models/seting_{param}_1.txt"
        else:
            file_path = f"/kaggle/input/polymers-models/seting_{param}.txt"
        line_number = 1 
        line = linecache.getline(file_path, line_number)
        fichier1 = line.strip().split("\t")
        fichier1 = line.strip().split("\t")
        input_1 = int(fichier1[0])
        output = int(fichier1[1])
        h_lears = int(fichier1[2])
        n_output_lear=h_lears+1
        k = 3
        
        while k<3+h_lears+2:
            n_neron.append(int(fichier1[k]))
            k += 1
        data_kind = int(fichier1[k])
        k=k+1  #7
        output_nerons_num_size=int(fichier1[k])
        
        k=k+1
        
        for i in range(0,output_nerons_num_size):
            output_nerons_num.append(int(fichier1[k]))
           
            k += 1
        #n_neron.append(0)
        i=0
        p=[]
        #k=9
        n_type_f = int(fichier1[k])
        k += 1
        j = 0
        while j<data_kind:
            i = 0
            p=[]
            while i<n_type_f:
                p.append(int(fichier1[k]))
                i += 1
                k += 1
            p.append(p)
            p=[]
            j += 1
        j = 0
        while j<output_nerons_num_size:
            i = 0
            p=[]
            while i<n_type_f:
                p.append(int(fichier1[k]))
                i += 1
                k += 1
            p.append(p)
            p=[]
            j += 1
        pages = int(fichier1[k])
        rows = int(fichier1[k+1])
        colons = int(fichier1[k+2])
        x_factor = int(fichier1[k+3]) 
       
        k=k+3
        x_factor = int(fichier1[k+1])
        x_factor = int(fichier1[k+2])
        x_factor = int(fichier1[k+3])
        x_factor = int(fichier1[k+4])
        x_factor = (fichier1[k+5]) 
        x_factor = (fichier1[k+6]) 
        k=k+6
        x_factor = int(fichier1[k+1]) 
        x_factor = int(fichier1[k+2])
        x_factor = float(fichier1[k+3])
   
        x_factor = fichier1[k+4]
        
        x_factor = int(fichier1[k+5])
        x_factor = int(fichier1[k+6]) 
        x_factor = float(fichier1[k+7])
        x_factor = int(fichier1[k+8])
        k=k+9
        p=0
        pp=[]
        for p in range(0,h_lears):
            for i in range(0, data_kind):
                k+=1
                
        for p in range(0,output_nerons_num_size):
            k+=1
        
        for p in range(0,h_lears):
            for i in range(0, data_kind):
                pp.append(int(fichier1[k]))
                k+=1
            type_f_a.append(pp)
            pp=[]
            
        for p in range(0,output_nerons_num_size):
            pp.append(int(fichier1[k]))
            k+=1
        type_f_a.append(pp)
        p=[]
     
        pini2= [0 for _ in range(input_1 + output)]
        pp=[]
        weigth_01=[]
        p=[]
        for i in range(0,pages):
            for j in range(0,n_neron[i]):
                for k in range(0,n_neron[i+1]):
                    p.append(258)
                pp.append(p)
                p=[]
            weigth_01.append(pp)
            pp=[]
        weigth=[]
        for i in range(0,pages):
            for j in range(0,n_neron[i]):
                for k in range(0,n_neron[i+1]):
                    p.append(258)
                pp.append(p)
                p=[]
            weigth.append(pp)
            pp=[]
        weigthT= []
        for i in range(0,pages):
            for k in range(0,n_neron[i+1]):
                for j in range(0,n_neron[i]):
                    p.append(258)
                pp.append(p)
                p=[]
            weigthT.append(pp)
            pp=[]
        bais=[]
        for i in range(0,pages):
            for j in range(0,n_neron[i+1]):
                p.append(0)
            bais.append(p)
            p=[]
        a=[]
        for i in range(0,pages+1):
            for j in range(0,n_neron[i]):
                p.append(0)
            a.append(p)
            p=[]
        n=[]
        for i in range(0,pages):
            for j in range(0,n_neron[i+1]):
                p.append(0)
            n.append(p)
            p=[]
        #initialisation de "out"
        out_file = open(nn_fille_2, 'r')
        in_out_validation=len(out_file.readlines())
        out=np.zeros((in_out_validation,output))
        out_file.close()
        
        #fin
        if param=='Tg' or param=='FFV':   # or param=='Density'
            file_path = f"/kaggle/input/polymers-models/w_{param}_1.txt"
        else:
            file_path = f"/kaggle/input/polymers-models/w_{param}.txt"
        line_number = 1
        line = linecache.getline(file_path, line_number)
        data = list(map(float, line.strip().split("\t")))
        incrimente=0
        for i in range(0,pages):
            colon =n_neron[i+1]
            row = n_neron[i]
            for j in range(0,row):
                for k in range(0,colon):
                    weigth[i][j][k] =float(data[incrimente])
                    incrimente+=1
        incrimente=0
     
        #fin
        
        if param=='Tg' or param=='FFV':   # or param=='Density'
            file_path = f"/kaggle/input/polymers-models/b_{param}_1.txt"
        else:
            file_path = f"/kaggle/input/polymers-models/b_{param}.txt"
        line_number = 1
        line = linecache.getline(file_path, line_number)
        data = list(map(float, line.strip().split("\t")))
        for i in range(0,pages):
            colon =n_neron[i+1]
            for k in range(0,colon):
                bais[i][k] =float(data[incrimente])
                incrimente+=1
        incrimente=0
        
        #fin
        
            
        if param=='Tg' or param=='FFV':   # or param=='Density'
            file_path = f"/kaggle/input/polymers-models/01table_{param}_1.txt"
        else:
            file_path = f"/kaggle/input/polymers-models/01table_{param}.txt"
            
        line_number = 1
        line = linecache.getline(file_path, line_number)
        data = list(map(int, line.strip().split("\t")))  
        for i in range(0,pages):
            colon =n_neron[i+1]
            row = n_neron[i]
            for j in range(0,row):
                for k in range(0,colon):
                    weigth_01[i][j][k] =int(data[incrimente])
                    incrimente+=1
        incrimente=0
        
        data = []
    
        #fin
        for i in range(0,pages):
            colon =n_neron[i+1]
            row = n_neron[i]
            for j in range(0,row):
                for k in range(0,colon):
                    weigth[i][j][k] =weigth[i][j][k]*weigth_01[i][j][k]
    
   
        trans_a=df.values.tolist()
        #input append
        for i in range(0, input_1):
            pini2[i]=trans_a[0][i]
            trans_2=pini2[i]
            a[0][i]=trans_2
 
        for i in range(0,pages):
            colon =n_neron[i+1]
            row = n_neron[i]
            for j in range(0,row):
                for k in range(0,colon):
                    weigthT[i][k][j] =weigth[i][j][k]
       
        for i in range(0,pages):
            colon =n_neron[i+1]
            row = n_neron[i]
            sweetch=0
            kind_neron=0
            for j in range(0,colon):
                for k in range(0,row):
                    n11 = a[i][k] * weigthT[i][j][k] + n11
                    
                n[i][j]=n11+bais[i][j]
                n11=0
                #calcul des a[i][j]
                
                a[i+1][j]=neuron(type_f_a[i][sweetch],n[i][j])
                kind_neron= kind_neron + 1
                if i<pages-1:
                    if kind_neron==n_neron[i+1]/data_kind:
                        sweetch= sweetch + 1
                        kind_neron=0
                if i==pages-1:
                    if kind_neron==output_nerons_num[sweetch]:
                        sweetch= sweetch + 1
                        kind_neron=0
                if i==h_lears and j<output:
                    out[r][j]=a[i+1][j]
                
                if i==0:
                    if j==input_1 :
                        j=input_1 
                    if j==input_1 :
                        j=input_1 
       
        prediction.append(a[n_output_lear][0])
        
        r=r+1
  
    return prediction #outt
prediction_test = []

for i in range(0, 5):    
    param = params[i]
    test_fill= f"/kaggle/working/df_test_{param}_features.txt"
    prediction_test.append(valid(test_fill, param))
    prediction_test[i] = pd.DataFrame({param : prediction_test[i]})
    



"""
# --- Step 1: Initialize a list to store all results ---
results_data = []

print("--- Starting Model Evaluation ---")

# --- Loop 1: Training Set Evaluation ---
print("Evaluating on Training Set...")
prediction_train = [] # Reset list
for i in range(len(params)):
    param = params[i]
    
    # Get predictions
    if param=='Tg' or param=='FFV':   # or param=='Density'
        train_fill_path = f"/kaggle/input/train-val-datasets-scaled/train_df_extra_{param}_1.txt"
    else:
        train_fill_path = f"/kaggle/input/polymers-train-test-datasets/train_df_{param}_features_1.txt"
    
    y_pred_train = valid(train_fill_path, param)
    prediction_train.append(y_pred_train)
    
    # Get true values
    if param=='Tg' or param=='FFV':   # or param=='Density'
        train_fill_path = f"/kaggle/input/train-val-datasets-scaled/train_df_extra_{param}_1.csv"
    else:
        train_fill_path = f"/kaggle/input/polymers-train-test-datasets/train_df_{param}_features_1.csv"
    
    train_dataset = pd.read_csv(train_fill_path)
    y_true_train = train_dataset[param]
    
    # Calculate and store metrics
    mae = mean_absolute_error(y_true_train, y_pred_train)
    rmse = np.sqrt(mean_squared_error(y_true_train, y_pred_train))
    r2 = r2_score(y_true_train, y_pred_train)
    results_data.append({'Dataset': 'Train', 'Property': param, 'MAE': mae, 'RMSE': rmse, 'R²': r2})

# --- Loop 2: Validation Set Evaluation ---
print("Evaluating on Validation Set...")
prediction_val = [] # Reset list
for i in range(len(params)):
    param = params[i]
    
    # Get predictions
    if param=='Tg' or param=='FFV':   # or param=='Density'
        val_fill_path = f"/kaggle/input/train-val-datasets-scaled/test_df_extra_{param}_1.txt"
    else:
        val_fill_path = f"/kaggle/input/polymers-train-test-datasets/test_df_{param}_features_1.txt"
    
    y_pred_val = valid(val_fill_path, param)
    prediction_val.append(y_pred_val)

    # Get true values
    if param=='Tg' or param=='FFV':   # or param=='Density'
        val_fill_path = f"/kaggle/input/train-val-datasets-scaled/test_df_extra_{param}_1.csv"
    else:
        val_fill_path = f"/kaggle/input/polymers-train-test-datasets/test_df_{param}_features_1.csv"
    
    val_dataset = pd.read_csv(val_fill_path)
    y_true_val = val_dataset[param]

    # Calculate and store metrics
    mae = mean_absolute_error(y_true_val, y_pred_val)
    rmse = np.sqrt(mean_squared_error(y_true_val, y_pred_val))
    r2 = r2_score(y_true_val, y_pred_val)
    results_data.append({'Dataset': 'Validation', 'Property': param, 'MAE': mae, 'RMSE': rmse, 'R²': r2})


# --- Step 3: Create and display the final results DataFrame ---
results_df = pd.DataFrame(results_data)

# Pivot the table for a wide, easy-to-compare format
results_df_styled = results_df.pivot(index='Property', columns='Dataset', values=['MAE', 'RMSE', 'R²']).round(4)

print("\n" + "="*60)
print("                Model Performance Summary")
print("="*60)
print(results_df_styled)
print("="*60 + "\n")
"""


test_df = pd.concat([id_, prediction_test[0], prediction_test[1], prediction_test[2], prediction_test[3], prediction_test[4]], axis=1)

test_df['Tg'] =test_df['Tg']*10 #-273.15
test_df['Tg'] = test_df['Tg'] * 9/5 + 32
test_df.to_csv("/kaggle/working/submission.csv", index=False)
print(test_df)




