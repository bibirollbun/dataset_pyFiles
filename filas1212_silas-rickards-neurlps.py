%pip install /kaggle/input/rdkit-librarys/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


#Import Libraries
import numpy as np
import pandas as pd

#Exploratory Data Analysis
import matplotlib.pyplot as plt
import seaborn as sns

#Chem Libraries
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

#Machine Learning Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#Helpful Preprocessing
from sklearn.model_selection import train_test_split

#Suppress Warnings
from rdkit import RDLogger  
import warnings
RDLogger.DisableLog('rdApp.*') 
warnings.simplefilter(action = "ignore", category = RuntimeWarning)



#Reading our data
train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')

df_1 = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv')
df_3 = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv')
df_4 = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv')

df_submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')




#Data cleaning the supplementary datasets.
df_1 = df_1.groupby('SMILES').mean().reset_index()
df_3 = df_3.groupby('SMILES').mean().reset_index()
df_4 = df_4.groupby('SMILES').mean().reset_index()


#Saving to a single dataframe
df = pd.merge(train_df, df_1, on='SMILES', how='outer')
df = pd.merge(df, df_3, on='SMILES', how='outer')
df = pd.merge(df, df_4, on='SMILES', how='outer')


print(df['SMILES'].duplicated().sum())

df.sample(10, random_state=64)


#Cleaning up the duplicate columns

#Making single 'Tg' column
df['Tg'] = df.apply(lambda x: x['Tg_y'] if pd.isnull(x['Tg_x']) else x['Tg_x'], axis=1)

#Making single 'FFV' column
df['FFV'] = df.apply(lambda x: x['FFV_y'] if pd.isnull(x['FFV_x']) else x['FFV_x'], axis=1)

#Making single 'Tc' column
df['Tc'] = df.apply(lambda x: x['TC_mean'] if pd.isnull(x['Tc']) else x['Tc'], axis=1)


df.drop(['Tg_x', 'Tg_y', 'FFV_x', 'FFV_y', 'TC_mean'], axis=1, inplace=True)

df = df[['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]

df.head()


can_smiles = []
for smile in df['SMILES']:
    mol = Chem.MolFromSmiles(smile)
    can_smiles.append(Chem.MolToSmiles(mol))
    
df['SMILES'] = can_smiles


can_smiles = []
for smile in df_submission['SMILES']:
    mol = Chem.MolFromSmiles(smile)
    can_smiles.append(Chem.MolToSmiles(mol))

df_submission['SMILES'] = can_smiles



print(df['SMILES'].duplicated().sum())


df[df['SMILES'].duplicated(keep=False)]


df.at[2503, 'Tc'] = df.at[8968, 'Tc']

df.at[8972, 'Tc'] = np.mean([df.at[8972, 'Tc'], df.at[8973, 'Tc']])

df.drop_duplicates(subset='SMILES', inplace=True)


df.head()


df.describe()


# Randomly sample 10 SMILES
sample_smiles = df['SMILES'].sample(10, random_state=64).to_list()

# Convert SMILES to RDKit Mol objects
mols = [Chem.MolFromSmiles(smi) for smi in sample_smiles]

# Draw and display molecules inline
Draw.MolsToGridImage(
    mols,
    molsPerRow=5,
    subImgSize=(200, 200),
    legends=[f"{sample_smiles[i]}" for i in range(len(sample_smiles))]
)


df_submission.head()


#Creating a molecular weight column
df['MW'] = df.apply(lambda x: Descriptors.ExactMolWt(Chem.MolFromSmiles(x['SMILES'])), axis=1)
df_submission['MW'] = df_submission.apply(lambda x: Descriptors.ExactMolWt(Chem.MolFromSmiles(x['SMILES'])), axis=1)


df['MW'].describe()


#This function 
def analyze_rings(SMILE):
    
    mol = Chem.MolFromSmiles(SMILE)
    
    def rewire_neighbors(ring, rw_mol):
        
        ring_set = ring.get('atoms')
        
        for atom in ring_set:
            for neighbor in rw_mol.GetAtomWithIdx(atom).GetNeighbors():
                if neighbor.GetIdx() not in ring_set:
                    rw_mol.RemoveBond(atom, neighbor.GetIdx())
                    if rw_mol.GetBondBetweenAtoms(neighbor.GetIdx(), ring.get('dummy_idx')) is None:
                        rw_mol.AddBond(neighbor.GetIdx(), ring.get('dummy_idx'))
            if atom not in atoms_to_remove:
                atoms_to_remove.append(atom)
                        
        return rw_mol
    
    
    
    #We will be modifying a molecular diagram
    rw_mol = Chem.RWMol(mol)
    
    # 1. Identify rings and atoms in rings
    ring_info = rw_mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    atoms_to_remove = []
    
    num_rings = ring_info.NumRings()
    
    
    # 2. Create a dummy carbon atom for each ring and store mapping
    ring_to_dummy = []
    
    for i, ring in enumerate(atom_rings):
        dummy_atom = Chem.Atom('C')  # Use dummy carbon atom
        dummy_idx = rw_mol.AddAtom(dummy_atom)
        ring_to_dummy.append({'atoms': set(ring),'dummy_idx': dummy_idx})
    
    # 3. For each non-ring neighbor, rewire their bond to the cooresponding carbon atom for each ring
    for ring in ring_to_dummy:
        rw_mol = rewire_neighbors(ring, rw_mol)
    
    # 4. Remove rings
    for idx in sorted(atoms_to_remove, reverse=True):
        rw_mol.RemoveAtom(idx)
    
    
    # 5. Count Heavy Neighbors

    branch_count = 0

    for atom in rw_mol.GetAtoms():
        neighbors = atom.GetNeighbors()

        # Count heavy neighbors
        heavy_neighbors = len([n for n in neighbors if n.GetAtomicNum() > 1])
        
        if heavy_neighbors > 2:
            branch_count += heavy_neighbors - 2
    
    
        
    return [num_rings, branch_count]

    


df[['NRings', 'HBranches']] = df['SMILES'].apply(analyze_rings).to_list()
df_submission[['NRings', 'HBranches']] = df_submission['SMILES'].apply(analyze_rings).to_list()


df.head()


df['numRotate'] = df.apply(lambda x: rdMolDescriptors.CalcNumRotatableBonds(Chem.MolFromSmiles(x['SMILES'])), axis=1)
df_submission['numRotate'] = df_submission.apply(lambda x: rdMolDescriptors.CalcNumRotatableBonds(Chem.MolFromSmiles(x['SMILES'])), axis=1)


def analyze_3D(SMILE):
    
    mol_volume = np.nan
    vdw_volume = np.nan
    dipole_moment = np.nan
    

    
    #Generate mol from SMILE
    mol = Chem.MolFromSmiles(SMILE)
    
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*':
            atom.SetAtomicNum(1)  # Replace dummy with hydrogen
    
    mol = Chem.AddHs(mol)
    
    try:
        #Generate conformer 
        conformer = rdDistGeom.EmbedMolecule(mol, useBasicKnowledge=False)
        

        # Compute volumes
        mol_volume = rdMolDescriptors.DoubleCubicLatticeVolume(mol).GetVolume()
        vdw_volume = rdMolDescriptors.DoubleCubicLatticeVolume(mol).GetVDWVolume()
        
        ComputeGasteigerCharges(mol)
        dipole_vector = np.zeros(3)
        conf = mol.GetConformer()
        
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            
            try:
                charge = float(atom.GetProp('_GasteigerCharge'))
            except Exception:
                charge = 0
            
            pos = conf.GetAtomPosition(idx)
            r = np.array([pos.x, pos.y, pos.z])
            dipole_vector += charge * r

        # Convert e·Å to Debye: 1 e·Å ≈ 4.8032 Debye
        dipole_moment = np.linalg.norm(dipole_vector) * 4.80320  
    except TimeoutError as e:
        print(e)
    finally:
        return [mol_volume, vdw_volume, dipole_moment]

    
    


df[['MV', 'VDWV', 'DM']] = df['SMILES'].apply(analyze_3D).to_list()
df_submission[['MV', 'VDWV', 'DM']] = df_submission['SMILES'].apply(analyze_3D).to_list()


# Code credit goes to [Brian Kelley](https://github.com/bp-kelley)

def _dfs(bond, atom, visited, path, longest):
    """Depth first search to find the longest path from an atom
    Args:
       bond:  the bond we are traversing (None for the first atom)
       atom: the atom we are traversing to
       visited: set of atom indices we have seen on the this path
       path: The list of [(from_bond, to_atom), ...] alreadh in the path
       longest: holds the longest path found so far
    """
    path.append((bond, atom))
    visited.add(atom.GetIdx())
    if not longest or len(path) > len(longest[0]):
        longest.clear()
        longest.append( path.copy() )

    for bond in atom.GetBonds():
        nbr = bond.GetOtherAtom(atom)
        if nbr.GetIdx() not in visited:
            _dfs(bond, nbr, visited, path, longest)

            path.pop()
            visited.remove(nbr.GetIdx())
            
            
def longest_path(SMILE):
    """Start at any atom (a), go to the atom farthest from that atom (b)
    And then find the atom farthest from (b).  This path is the longest path
    Args:
      mol: the molecule to search
      
    Returns:
       [atom indices for atoms in the longest path], pbond indices for bonds in the longest path]
    """
    
    #Generate mol from SMILE
    mol = Chem.MolFromSmiles(SMILE) 
    
    for atom in mol.GetAtoms():
        # go to the farthest atom
        visited, path, longest = set(), [], []
        _dfs(None, atom, visited, path, longest)
        
        # from the farthest atom, go to the farthest atom from it
        visited, path, longest = set(), [], []
        _dfs(None, atom, visited, path, longest)
        
        # this is the longest path, return
        bond_path = [b.GetIdx() for b,_ in longest[0] if b is not None]
        return len(bond_path)


df['Longest_Path_Length'] = df['SMILES'].apply(longest_path)
df_submission['Longest_Path_Length'] = df_submission['SMILES'].apply(longest_path)


df.head()


df_submission.head()


df.info()


df.info()

df.isnull().sum()


df['id'] = df['id'].astype(object)


df.info()


#Takes an arbitrary amount of numeric dataframe columns and creates pairs of boxplots+histograms arranged in at most three columns.
def numericUA(dFrame, bins = "auto"):

  ncols = len(dFrame.columns)
  nsets = (ncols-1) // 3 + 1

  f, aplot = plt.subplots(

    nrows = 2*nsets,
    ncols=  min(3, ncols),
    height_ratios=(.25, .75)*nsets,
    figsize = (18, 10*nsets)
  )

  #Turns off axis' by default so that there are no blank plots
  for a in aplot.flatten():
      a.axis("off")

  colIndex = 0

  #Iterates through each column and creates both the box plot and histogram in their correct positions
  for col in dFrame.columns:
    sns.boxplot(x = dFrame[col], ax=aplot[2*(colIndex // 3), colIndex % 3], showmeans=True, color = 'orange')

    sns.histplot(x = dFrame[col], ax = aplot[2*(colIndex // 3) + 1, colIndex % 3], kde=False, bins = bins)

    #Creates mean line
    aplot[2*(colIndex // 3)+1, colIndex % 3].axvline(np.mean(dFrame[col]), color='g', linestyle='--')

    #Creates median line
    aplot[2*(colIndex // 3)+1, colIndex % 3].axvline(np.median(dFrame[col]), color='black', linestyle='-')

    #If a plot exists, turns on axis'
    aplot[2*(colIndex // 3), colIndex % 3].set_axis_on()
    aplot[2*(colIndex // 3)+1, colIndex % 3].set_axis_on()

    colIndex += 1


  plt.show()


df.select_dtypes(include='number')


numericUA(df.select_dtypes(include='number'))


df[df['Tc'] > 1.0]


#Removing abnormally high Tc
df.drop(index=8750, axis=0, inplace=True)


df[df['DM'] > 30]


#Removing abnormally high DM
df.drop(index=[2842, 3643], axis=0, inplace=True)


df[df['numRotate'] > 60]


#Removing abnormally high DM
df.drop(index=[4081], axis=0, inplace=True)


numericUA(df.select_dtypes(include='number'))


df.drop(['id', 'SMILES'], axis=1, inplace=True)


targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
features = ['MW', 'NRings', 'HBranches', 'numRotate', 'MV', 'VDWV', 'DM', 'Longest_Path_Length']


for target in targets:
    for feature in features:
        
        
        
        sns.scatterplot(df, x=target, y=feature)
        
        plt.title(f'{target} vs {feature}')
        plt.tight_layout()
        plt.show()
        



plt.figure(figsize=(10, 6))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap of Titanic")
plt.show()


df.drop('MV', axis=1, inplace=True)
df_submission.drop('MV', axis=1, inplace=True)

features.remove('MV')


df.info()

df.isnull().sum()


from sklearn.impute import KNNImputer

knn_imputer = KNNImputer(n_neighbors=5)

df[features] = pd.DataFrame(knn_imputer.fit_transform(df[features]),index=df.index)

knn_imputer = KNNImputer(n_neighbors=5)

df_submission[features] = pd.DataFrame(knn_imputer.fit_transform(df_submission[features]),index=df_submission.index)


X_train, X_val, y_train, y_val = train_test_split(
    df[features], df[targets], test_size=0.20, random_state=64
    )


X_train.shape, X_val.shape


df_tuned = pd.DataFrame({'n_estimators': [129, 186, 124, 113, 167],
                        'max_depth': [10, 20, 14, 18, 16],
                        'min_samples_split': [5, 5, 3, 4, 3],
                        'min_samples_leaf': [1, 2, 1, 2, 2],
                        'bootstrap': [True, True, True, True, True]}, index= targets)

df_tuned



def rf_model(X_train, y_train, X_test, y_test, df_tuned):
    models = {}
    y_pred = np.zeros_like(y_test)

    # Train one random forest per task
    for idx, name in enumerate(targets):
        print('Training Random Tree regressor for the task:', name)
        y_col = y_train.iloc[:, idx]
        mask  = ~np.isnan(y_col)
        model = RandomForestRegressor(n_estimators=df_tuned.loc[name, 'n_estimators'],
                                      max_depth=df_tuned.loc[name, 'max_depth'],
                                      min_samples_split=df_tuned.loc[name, 'min_samples_split'],
                                      min_samples_leaf=df_tuned.loc[name, 'min_samples_leaf'],
                                      bootstrap=df_tuned.loc[name, 'bootstrap'],
                                      random_state=64)
        model.fit(X_train[mask], y_col[mask])
        models[name] = model
        # Predict on test set
        y_pred[:, idx] = model.predict(X_test)
        
    return models, y_pred


models2, y_pred = rf_model(X_train, y_train, X_val, y_val, df_tuned)


X_test = df_submission[features]

df_submission[targets] = 0.0


y_test_pred= np.zeros_like(df_submission[targets])
y_test_pred = y_test_pred.astype(float)

for idx, name in enumerate(targets):
    # Predict on test set
    y_test_pred[:, idx] = models2[name].predict(X_test)


df_submission = df_submission[['id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]

df_submission[targets] = y_test_pred


df_submission.to_csv('submission.csv', index=False)


df_submission

