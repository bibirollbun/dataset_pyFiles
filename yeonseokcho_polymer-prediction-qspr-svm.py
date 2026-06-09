import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
                   
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
        
import warnings
warnings.filterwarnings("ignore", category=Warning)


sample_submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
print(sample_submission.shape)
sample_submission.head()


test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
print(test.shape)
test.head()

# a hidden test set


train= pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
print(train.shape)
train.head()


train.info()


train.describe(include='all')


# 1. Tg (Glass Transition Temperature)
# Relevance: Molecular weight, polarity, hydrogen bonding, molecular flexibility, ring structure, etc.
descriptor_names_tg = [
    'MolWt',                # Molecular weight
    'MolLogP',              # Hydrophobicity/Polarity
    'NumHDonors',           # Number of hydrogen bond donors
    'NumHAcceptors',        # Number of hydrogen bond acceptors
    'TPSA',                 # Topological polar surface area
    'NumRotatableBonds',    # Number of rotatable bonds (molecular flexibility)
    'RingCount',            # Number of rings
    'HeavyAtomCount',       # Number of heavy atoms
    'FractionCSP3',         # Fraction of sp3 carbons (structural diversity)
    'BalabanJ',             # Topological index (molecular complexity)
    'Chi0',                 # Connectivity index
    'Kappa3',               # Shape index
]

# 4-2. FFV (Fractional Free Volume)
# Relevance: Molecular size, density, structural complexity, branching, etc.
descriptor_names_ffv = [
    'MolWt',
    'TPSA',
    'LabuteASA',           # Accessible surface area
    'MolMR',               # Molecular refractivity (volume-related)
    'NumAliphaticRings',   # Number of aliphatic rings
    'NumAromaticRings',    # Number of aromatic rings
    'FractionCSP3',
    'NumRotatableBonds',
    'HeavyAtomCount',
    'PEOE_VSA1',           # Partial charge surface area 1
    'PEOE_VSA6',           # Partial charge surface area 6
]

# 3. Tc (Thermal Conductivity)
# Relevance: Molecular size, polarity, structural rigidity, electron distribution, etc.
descriptor_names_tc = [
    'MolWt',
    'MolLogP',
    'TPSA',
    'LabuteASA',
    'NumAromaticRings',
    'NumAliphaticRings',
    'BalabanJ',
    'Chi3n',               # Connectivity index
    'Kappa2',              # Shape index
    'HeavyAtomCount',
    'EState_VSA1',         # EState surface area 1
    'EState_VSA9',         # EState surface area 9
]

# 4. Density
# Relevance: Molecular weight, volume, bonding structure, atomic composition, etc.
descriptor_names_density = [
    'MolWt',
    'MolMR',
    'TPSA',
    'LabuteASA',
    'HeavyAtomCount',
    'FractionCSP3',
    'NumAliphaticRings',
    'NumAromaticRings',
    'PEOE_VSA2',
    'PEOE_VSA8',
    'Kappa1',
]

# 5. Rg (Radius of Gyration)
# Relevance: Molecular size, chain length, flexibility, structural complexity, etc.
descriptor_names_rg = [
    'MolWt',
    'NumRotatableBonds',
    'HeavyAtomCount',
    'FractionCSP3',
    'LabuteASA',
    'BalabanJ',
    'Kappa3',
    'Chi1v',
    'TPSA',
    'NumAliphaticCarbocycles',
    'NumAromaticRings',
]


# To install RDKit on Kaggle without internet access, you need to upload the RDKit wheel file as a dataset.

!pip install /kaggle/input/rdkit-install-whl/rdkit_wheel/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from tqdm import tqdm


# descriiptors to use
descriptor_names_all= list(set(
    descriptor_names_tg +
    descriptor_names_ffv +
    descriptor_names_tc +
    descriptor_names_density +
    descriptor_names_rg
))

calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names_all)
calc


# for example
ethanol_smiles = 'CCO'

# Convert SMILES to Mol object
ethanol_mol = Chem.MolFromSmiles(ethanol_smiles)

ethanol_desc_values = calc.CalcDescriptors(ethanol_mol)

for name, value in zip(descriptor_names_all, ethanol_desc_values):
    print(f"{name}: {value}")


# convert smiles to descriptors
def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan]*len(descriptor_names_all)
    return list(calc.CalcDescriptors(mol))

smiles_to_descriptors


# Apply descriptors to the train dataset
train_desc_list = []
for smi in tqdm(train['SMILES'], desc='Calculating descriptors'):
    train_desc_list.append(smiles_to_descriptors(smi))
train_desc_df = pd.DataFrame(train_desc_list, columns=descriptor_names_all)

# combine train dataset
train_desc = pd.concat([train, train_desc_df], axis=1)

print(train_desc.shape)
train_desc.head()


# Apply descriptors to the test dataset
test_desc_list = []
for smi in tqdm(test['SMILES'], desc='Calculating descriptors'):
    test_desc_list.append(smiles_to_descriptors(smi))
test_desc_df = pd.DataFrame(test_desc_list, columns=descriptor_names_all)

# combine test dataset
test_desc = pd.concat([test, test_desc_df], axis=1)

print(test_desc.shape)
test_desc.head()


# simple_descriptors from smiles
def simple_descriptors(smiles):
    return [
        len(smiles),            # Length of the SMILES string
        smiles.count('C'),      # Number of 'C' atoms
        smiles.count('O'),      # Number of 'O' atoms
        smiles.count('N'),      # Number of 'N' atoms
        smiles.count('='),      # Number of '=' atoms
        smiles.count('('),      # Number of '(', branch
        smiles.count('1'),      # Number of ring indicators '1'
        smiles.count('2'),      # Number of ring indicators '2'
        smiles.count('3'),      # Number of ring indicators '3'
    ]

simple_desc_names = [
    'SMILES_len', 'C_count', 'O_count', 'N_count', 'double_bond_count',
    'branch_count', 'ring1_count', 'ring2_count', 'ring3_count'
]

# 2. Calculate simple_descriptors and add to train_desc
train_simple_desc = train_desc['SMILES'].apply(simple_descriptors).tolist()
train_simple_desc_df = pd.DataFrame(train_simple_desc, columns=simple_desc_names)

test_simple_desc = test_desc['SMILES'].apply(simple_descriptors).tolist()
test_simple_desc_df = pd.DataFrame(test_simple_desc, columns=simple_desc_names)

print(train_simple_desc_df.shape, test_simple_desc_df.shape)
train_simple_desc_df.head()


test_simple_desc_df.isna().sum()


train_desc_all = pd.concat([train_desc.reset_index(drop=True), train_simple_desc_df], axis=1)
test_desc_all = pd.concat([test_desc.reset_index(drop=True), test_simple_desc_df], axis=1)

print(train_desc_all.shape, test_desc_all.shape)
train_desc_all.head()


# 1. features and target data for Tg
train_Tg = train_desc_all[~train_desc_all['Tg'].isna()].copy()

train_Tg_features = train_Tg.drop(['id', 'SMILES',	'Tg', 'FFV', 'Tc',	'Density',	'Rg'], axis=1)
train_Tg_target = train_Tg['Tg']

print(train_Tg_features.shape, train_Tg_target.shape)
train_Tg_features.head()


# 2. features selection
corr = train_Tg_features.corrwith(train_Tg_target).abs()
low_corr_features = corr[corr < 0.05].index  
train_Tg_features = train_Tg_features.drop(low_corr_features, axis=1)
train_Tg_features.shape


train_Tg_features.isna().sum().sum()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

# 3. Train & Validation set 
X_train_Tg, X_val_Tg, y_train_Tg, y_val_Tg = train_test_split(
    train_Tg_features, train_Tg_target, test_size=0.2, random_state=42
)

# 4. Scaling
scaler = StandardScaler()
X_train_Tg_scaled = scaler.fit_transform(X_train_Tg)
X_val_Tg_scaled = scaler.transform(X_val_Tg)

test_features_Tg = test_desc_all[X_train_Tg.columns]  
test_features_Tg_scaled = scaler.transform(test_features_Tg)

X_train_Tg_scaled.shape, X_val_Tg_scaled.shape, test_features_Tg_scaled.shape


# 5. Hyperparameter Tuning
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'C': np.logspace(-2, 3, 20),
    'epsilon': np.logspace(-3, 0, 10),
    'gamma': ['scale', 'auto'] + list(np.logspace(-3, 1, 10))
}

svr = SVR(kernel='rbf')
random_search = RandomizedSearchCV(
    svr, param_distributions=param_dist, 
    n_iter=50, cv=10, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42
)
random_search.fit(X_train_Tg_scaled, y_train_Tg)

print("Best parameters:", random_search.best_params_)


# 6. SVM Regression Model
svm_Tg = SVR(kernel='rbf', C=297.63514416313194, epsilon=1.0, gamma=0.007742636826811269)
svm_Tg.fit(X_train_Tg_scaled, y_train_Tg)

# 7. Prediction
y_train_Tg_pred = svm_Tg.predict(X_train_Tg_scaled)
y_val_Tg_pred = svm_Tg.predict(X_val_Tg_scaled)

# 8. Evaluation
train_mae = mean_absolute_error(y_train_Tg, y_train_Tg_pred)
train_r2 = r2_score(y_train_Tg, y_train_Tg_pred)
val_mae = mean_absolute_error(y_val_Tg, y_val_Tg_pred)
val_r2 = r2_score(y_val_Tg, y_val_Tg_pred)

print(f"Train MAE: {train_mae:.6f}")
print(f"Train R2: {train_r2:.6f}")
print(f"Validation MAE: {val_mae:.6f}")
print(f"Validation R2: {val_r2:.6f}")

# 9. Predicted Values
import pandas as pd
result_df = pd.DataFrame({
    "True_Tg": y_val_Tg,
    "Pred_Tg": y_val_Tg_pred
})
print(result_df.head())

"""
Train MAE: 42.511104
Train R2: 0.720988
Validation MAE: 48.653386
Validation R2: 0.572868
"""


# learning_curve for Tg
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    svm_Tg, X_train_Tg_scaled, y_train_Tg, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_scores_mean = -np.mean(train_scores, axis=1)
val_scores_mean = -np.mean(val_scores, axis=1)

plt.figure(figsize=(6, 3))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Train MAE')
plt.plot(train_sizes, val_scores_mean, 'o-', color='red', label='Validation MAE')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Absolute Error')
plt.title('Learning Curve')
plt.legend()
plt.show()





# 1. features and target data for FFV
train_FFV = train_desc_all[~train_desc_all['FFV'].isna()].copy()

train_FFV_features = train_FFV.drop(['id', 'SMILES', 'Tg', 'FFV', 'Tc',	'Density',	'Rg'], axis=1)
train_FFV_target = train_FFV['FFV']

print(train_FFV_features.shape, train_FFV_target.shape)
train_FFV_features.head()


# 2. features selection
corr = train_FFV_features.corrwith(train_FFV_target).abs()
low_corr_features = corr[corr < 0.05].index  # 상관계수 0.05 미만 피처 제거 예시
train_FFV_features = train_FFV_features.drop(low_corr_features, axis=1)
train_FFV_features.shape


train_FFV_features.isna().sum().sum()


# 3. Train & Validation set 
X_train_FFV, X_val_FFV, y_train_FFV, y_val_FFV = train_test_split(
    train_FFV_features, train_FFV_target, test_size=0.2, random_state=42
)

# 4. Scaling
scaler = StandardScaler()
X_train_FFV_scaled = scaler.fit_transform(X_train_FFV)
X_val_FFV_scaled = scaler.transform(X_val_FFV)

test_features_FFV = test_desc_all[X_train_FFV.columns]  
test_features_FFV_scaled = scaler.transform(test_features_FFV)

X_train_FFV_scaled.shape, X_val_FFV_scaled.shape, test_features_FFV_scaled.shape


# 5. Hyperparameter Tuning
param_dist = {
    'C': np.logspace(-2, 3, 20),
    'epsilon': np.logspace(-3, 0, 10),
    'gamma': ['scale', 'auto'] + list(np.logspace(-3, 1, 10))
}

svr = SVR(kernel='rbf')
random_search = RandomizedSearchCV(
    svr, param_distributions=param_dist, 
    n_iter=50, cv=10, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42
)
random_search.fit(X_train_FFV_scaled, y_train_FFV)

print("Best parameters:", random_search.best_params_)


# 6. SVM Regression Model
svm_FFV = SVR(kernel='rbf', C=2.3357214690901213, epsilon=0.004641588833612777, gamma='auto')
svm_FFV.fit(X_train_FFV_scaled, y_train_FFV)

# 7. Prediction
y_train_FFV_pred = svm_FFV.predict(X_train_FFV_scaled)
y_val_FFV_pred = svm_FFV.predict(X_val_FFV_scaled)

# 8. Evaluation
train_mae = mean_absolute_error(y_train_FFV, y_train_FFV_pred)
train_r2 = r2_score(y_train_FFV, y_train_FFV_pred)
val_mae = mean_absolute_error(y_val_FFV, y_val_FFV_pred)
val_r2 = r2_score(y_val_FFV, y_val_FFV_pred)

print(f"Train MAE: {train_mae:.6f}")
print(f"Train R2: {train_r2:.6f}")
print(f"Validation MAE: {val_mae:.6f}")
print(f"Validation R2: {val_r2:.6f}")

# 9. Predicted Values
import pandas as pd
result_df = pd.DataFrame({
    "True_FFV": y_val_FFV,
    "Pred_FFV": y_val_FFV_pred
})
print(result_df.head())

"""
Train MAE: 0.005327
Train R2: 0.833889
Validation MAE: 0.008119
Validation R2: 0.650542
"""


# learning_curve for FFV
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    svm_FFV, X_train_FFV_scaled, y_train_FFV, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_scores_mean = -np.mean(train_scores, axis=1)
val_scores_mean = -np.mean(val_scores, axis=1)

plt.figure(figsize=(6, 3))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Train MAE')
plt.plot(train_sizes, val_scores_mean, 'o-', color='red', label='Validation MAE')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Absolute Error')
plt.title('Learning Curve')
plt.legend()
plt.show()


# 1. features and target data for Tc
train_Tc = train_desc_all[~train_desc_all['Tc'].isna()].copy()

train_Tc_features = train_Tc.drop(['id', 'SMILES', 'Tg', 'FFV', 'Tc',	'Density',	'Rg'], axis=1)
train_Tc_target = train_Tc['Tc']

print(train_Tc_features.shape, train_Tc_target.shape)
train_Tc_features.head()


# 2. features selection
corr = train_Tc_features.corrwith(train_Tc_target).abs()
low_corr_features = corr[corr < 0.05].index  
train_Tc_features = train_Tc_features.drop(low_corr_features, axis=1)
train_Tc_features.shape


train_Tc_features.isna().sum().sum()


# 3. Train & Validation set 
X_train_Tc, X_val_Tc, y_train_Tc, y_val_Tc = train_test_split(
    train_Tc_features, train_Tc_target, test_size=0.2, random_state=42
)

# 4. Scaling
scaler = StandardScaler()
X_train_Tc_scaled = scaler.fit_transform(X_train_Tc)
X_val_Tc_scaled = scaler.transform(X_val_Tc)

test_features_Tc = test_desc_all[X_train_Tc.columns]  
test_features_Tc_scaled = scaler.transform(test_features_Tc)

X_train_Tc_scaled.shape, X_val_Tc_scaled.shape, test_features_Tc_scaled.shape


# 5. Hyperparameter Tuning
param_dist = {
    'C': np.logspace(-2, 3, 20),
    'epsilon': np.logspace(-3, 0, 10),
    'gamma': ['scale', 'auto'] + list(np.logspace(-3, 1, 10))
}

svr = SVR(kernel='rbf')
random_search = RandomizedSearchCV(
    svr, param_distributions=param_dist, 
    n_iter=50, cv=10, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42
)
random_search.fit(X_train_Tc_scaled, y_train_Tc)

print("Best parameters:", random_search.best_params_)


# 6. SVM Regression Model
svm_Tc = SVR(kernel='rbf', C=0.6951927961775606, epsilon=0.0021544346900318843, gamma=0.0027825594022071257)
svm_Tc.fit(X_train_Tc_scaled, y_train_Tc)

# 7. Prediction
y_train_Tc_pred = svm_Tc.predict(X_train_Tc_scaled)
y_val_Tc_pred = svm_Tc.predict(X_val_Tc_scaled)

# 8. Evaluation
train_mae = mean_absolute_error(y_train_Tc, y_train_Tc_pred)
train_r2 = r2_score(y_train_Tc, y_train_Tc_pred)
val_mae = mean_absolute_error(y_val_Tc, y_val_Tc_pred)
val_r2 = r2_score(y_val_Tc, y_val_Tc_pred)

print(f"Train MAE: {train_mae:.6f}")
print(f"Train R2: {train_r2:.6f}")
print(f"Validation MAE: {val_mae:.6f}")
print(f"Validation R2: {val_r2:.6f}")

# 9. Predicted Values
import pandas as pd
result_df = pd.DataFrame({
    "True_Tc": y_val_Tc,
    "Pred_Tc": y_val_Tc_pred
})
print(result_df.head())

"""
Train MAE: 0.026053
Train R2: 0.798491
Validation MAE: 0.029231
Validation R2: 0.772919
"""


# learning_curve for Tc
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    svm_Tc, X_train_Tc_scaled, y_train_Tc, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_scores_mean = -np.mean(train_scores, axis=1)
val_scores_mean = -np.mean(val_scores, axis=1)

plt.figure(figsize=(6, 3))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Train MAE')
plt.plot(train_sizes, val_scores_mean, 'o-', color='red', label='Validation MAE')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Absolute Error')
plt.title('Learning Curve')
plt.legend()
plt.show()


# 1. features and target data for Density
train_Density = train_desc_all[~train_desc_all['Density'].isna()].copy()

train_Density_features = train_Density.drop(['id', 'SMILES', 'Tg', 'FFV', 'Tc',	'Density',	'Rg'], axis=1)
train_Density_target = train_Density['Density']

print(train_Density_features.shape, train_Density_target.shape)
train_Density_features.head()


# 2. features selection
corr = train_Density_features.corrwith(train_Density_target).abs()
low_corr_features = corr[corr < 0.05].index  
train_Density_features = train_Density_features.drop(low_corr_features, axis=1)
train_Density_features.shape


train_Density_features.isna().sum().sum()


# 3. Train & Validation set 
X_train_Density, X_val_Density, y_train_Density, y_val_Density = train_test_split(
    train_Density_features, train_Density_target, test_size=0.2, random_state=42
)

# 4. Scaling
scaler = StandardScaler()
X_train_Density_scaled = scaler.fit_transform(X_train_Density)
X_val_Density_scaled = scaler.transform(X_val_Density)

test_features_Density = test_desc_all[X_train_Density.columns]  
test_features_Density_scaled = scaler.transform(test_features_Density)

X_train_Density_scaled.shape, X_val_Density_scaled.shape, test_features_Density_scaled.shape


# 5. Hyperparameter Tuning
param_dist = {
    'C': np.logspace(-2, 3, 20),
    'epsilon': np.logspace(-3, 0, 10),
    'gamma': ['scale', 'auto'] + list(np.logspace(-3, 1, 10))
}

svr = SVR(kernel='rbf')
random_search = RandomizedSearchCV(
    svr, param_distributions=param_dist, 
    n_iter=50, cv=10, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42
)
random_search.fit(X_train_Density_scaled, y_train_Density)

print("Best parameters:", random_search.best_params_)


# 6. SVM Regression Model
svm_Density = SVR(kernel='rbf', C=2.3357214690901213, epsilon=0.004641588833612777, gamma='auto')
svm_Density.fit(X_train_Density_scaled, y_train_Density)

# 7. Prediction
y_train_Density_pred = svm_Density.predict(X_train_Density_scaled)
y_val_Density_pred = svm_Density.predict(X_val_Density_scaled)

# 8. Evaluation
train_mae = mean_absolute_error(y_train_Density, y_train_Density_pred)
train_r2 = r2_score(y_train_Density, y_train_Density_pred)
val_mae = mean_absolute_error(y_val_Density, y_val_Density_pred)
val_r2 = r2_score(y_val_Density, y_val_Density_pred)

print(f"Train MAE: {train_mae:.6f}")
print(f"Train R2: {train_r2:.6f}")
print(f"Validation MAE: {val_mae:.6f}")
print(f"Validation R2: {val_r2:.6f}")

# 9. Predicted Values
import pandas as pd
result_df = pd.DataFrame({
    "True_Density": y_val_Density,
    "Pred_Density": y_val_Density_pred
})
print(result_df.head())

"""
Train MAE: 0.015570
Train R2: 0.889586
Validation MAE: 0.040491
Validation R2: 0.674534
"""


# learning_curve for DEnsity
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    svm_Density, X_train_Density_scaled, y_train_Density, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_scores_mean = -np.mean(train_scores, axis=1)
val_scores_mean = -np.mean(val_scores, axis=1)

plt.figure(figsize=(6, 3))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Train MAE')
plt.plot(train_sizes, val_scores_mean, 'o-', color='red', label='Validation MAE')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Absolute Error')
plt.title('Learning Curve')
plt.legend()
plt.show()


# 1. features and target data for Rg
train_Rg = train_desc_all[~train_desc_all['Rg'].isna()].copy()

train_Rg_features = train_Rg.drop(['id', 'SMILES', 'Tg', 'FFV', 'Tc',	'Density',	'Rg'], axis=1)
train_Rg_target = train_Rg['Rg']

print(train_Rg_features.shape, train_Rg_target.shape)
train_Rg_features.head()


# 2. features selection
corr = train_Rg_features.corrwith(train_Rg_target).abs()
low_corr_features = corr[corr < 0.05].index  
train_Rg_features = train_Rg_features.drop(low_corr_features, axis=1)
train_Rg_features.shape


train_Rg_features.isna().sum().sum()


# 3. Train & Validation set 
X_train_Rg, X_val_Rg, y_train_Rg, y_val_Rg = train_test_split(
    train_Rg_features, train_Rg_target, test_size=0.2, random_state=42
)

# 4. Scaling
scaler = StandardScaler()
X_train_Rg_scaled = scaler.fit_transform(X_train_Rg)
X_val_Rg_scaled = scaler.transform(X_val_Rg)

test_features_Rg = test_desc_all[X_train_Rg.columns]  
test_features_Rg_scaled = scaler.transform(test_features_Rg)

X_train_Rg_scaled.shape, X_val_Rg_scaled.shape, test_features_Rg_scaled.shape


# 5. Hyperparameter Tuning
param_dist = {
    'C': np.logspace(-2, 3, 20),
    'epsilon': np.logspace(-3, 0, 10),
    'gamma': ['scale', 'auto'] + list(np.logspace(-3, 1, 10))
}

svr = SVR(kernel='rbf')
random_search = RandomizedSearchCV(
    svr, param_distributions=param_dist, 
    n_iter=50, cv=10, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42
)
random_search.fit(X_train_Rg_scaled, y_train_Rg)

print("Best parameters:", random_search.best_params_)


# 6. SVM Regression Model
svm_Rg = SVR(kernel='rbf', C=48.32930238571752, epsilon=1.0, gamma='auto')
svm_Rg.fit(X_train_Rg_scaled, y_train_Rg)

# 7. Prediction
y_train_Rg_pred = svm_Rg.predict(X_train_Rg_scaled)
y_val_Rg_pred = svm_Rg.predict(X_val_Rg_scaled)

# 8. Evaluation
train_mae = mean_absolute_error(y_train_Rg, y_train_Rg_pred)
train_r2 = r2_score(y_train_Rg, y_train_Rg_pred)
val_mae = mean_absolute_error(y_val_Rg, y_val_Rg_pred)
val_r2 = r2_score(y_val_Rg, y_val_Rg_pred)

print(f"Train MAE: {train_mae:.6f}")
print(f"Train R2: {train_r2:.6f}")
print(f"Validation MAE: {val_mae:.6f}")
print(f"Validation R2: {val_r2:.6f}")

# 9. Predicted Values
import pandas as pd
result_df = pd.DataFrame({
    "True_Rg": y_val_Rg,
    "Pred_Rg": y_val_Rg_pred
})
print(result_df.head())
"""
Train MAE: 1.401032
Train R2: 0.792006
Validation MAE: 2.173191
Validation R2: 0.517837
"""


# learning_curve for Rg
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    svm_Rg, X_train_Rg_scaled, y_train_Rg, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_scores_mean = -np.mean(train_scores, axis=1)
val_scores_mean = -np.mean(val_scores, axis=1)

plt.figure(figsize=(6, 3))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Train MAE')
plt.plot(train_sizes, val_scores_mean, 'o-', color='red', label='Validation MAE')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Absolute Error')
plt.title('Learning Curve')
plt.legend()
plt.show()


test_pred_Tg = svm_Tg.predict(test_features_Tg_scaled)
test_pred_FFV = svm_FFV.predict(test_features_FFV_scaled)
test_pred_Tc = svm_Tc.predict(test_features_Tc_scaled)
test_pred_Density = svm_Density.predict(test_features_Density_scaled)
test_pred_Rg = svm_Rg.predict(test_features_Rg_scaled)
test_pred_Tg


submission = pd.DataFrame()
submission['id'] = test['id']

submission['Tg'] = test_pred_Tg
submission['FFV'] = test_pred_FFV
submission['Tc'] = test_pred_Tc
submission['Density'] = test_pred_Density
submission['Rg'] = test_pred_Rg

print(submission.shape)
submission.head()


submission.to_csv("submission.csv", index=False)


submission = pd.read_csv('/kaggle/working/submission.csv')
print(submission.shape)
submission.head()

