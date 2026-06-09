import kagglehub
path = kagglehub.dataset_download("senkin13/rdkit-2025-3-3-cp311")
print("Path to dataset files", path)
!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')


import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski
import numpy as np


# Functional group SMARTS patterns
smarts_patterns = {
    'Hydroxyl': '[OX2H]',             # -OH
    'CarboxylicAcid': 'C(=O)[OH]',    # -COOH
    'PrimaryAmine': '[NX3;H2]',       # -NH2
    'SecondaryAmine': '[NX3;H1]',     # -NHR
    'Amide': 'C(=O)N',                # Amide
    'Ether': '[OD2]([#6])[#6]',       # R-O-R
    'AromaticRings': 'a1aaaaa1'       # Simple benzene
}

# Descriptor labels
descriptor_names = [
    'MolWt', 'LogP', 'TPSA', 'RotatableBonds',
    'HDonors', 'HAcceptors', 'FracCSP3', 'HeavyAtoms',
    'RingCount', 'AliphaticRings', 'AromaticRings',
    'Heteroatoms', 'BalabanJ', 'BertzCT', 'LabuteASA', 'MolMR', 'ValenceElectrons'
] + list(smarts_patterns.keys())


# Compute descriptors
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * len(descriptor_names)

    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.RingCount(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumHeteroatoms(mol),
        Descriptors.BalabanJ(mol),
        Descriptors.BertzCT(mol),
        Descriptors.LabuteASA(mol),
        Descriptors.MolMR(mol),
        Descriptors.NumValenceElectrons(mol),
    ]

    for name, pattern in smarts_patterns.items():
        patt = Chem.MolFromSmarts(pattern)
        descriptors.append(len(mol.GetSubstructMatches(patt)))

    return descriptors

# Flattened feature matrix
flattened_features = df['SMILES'].apply(compute_descriptors).apply(pd.Series)
flattened_features.columns = descriptor_names

# Final dataframe: smiles + descriptors
df = pd.concat([df[['SMILES']], flattened_features], axis=1)

# Show result
print(df.head())


data = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
df.insert(loc = len(df.columns), column='Tg',value=data['Tg'])
df1 = df.dropna(subset=['Tg'])
y_train1 = df1['Tg']
X_train1 = df1.drop('Tg',axis=1)
X_train1 = X_train1.drop('SMILES',axis=1)
duplicates = X_train1.columns[X_train1.columns.duplicated()]
print("Duplicate columns:", duplicates.tolist())
X_train1 = X_train1.loc[:, ~X_train1.columns.duplicated()]


X_test  = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
# Ensure pandas and RDKit are imported
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski

# If not already defined
smarts_patterns = {
    'Hydroxyl': '[OX2H]',
    'CarboxylicAcid': 'C(=O)[OH]',
    'PrimaryAmine': '[NX3;H2]',
    'SecondaryAmine': '[NX3;H1]',
    'Amide': 'C(=O)N',
    'Ether': '[OD2]([#6])[#6]',
    'AromaticRings': 'a1aaaaa1'
}

descriptor_names = [
    'MolWt', 'LogP', 'TPSA', 'RotatableBonds',
    'HDonors', 'HAcceptors', 'FracCSP3', 'HeavyAtoms',
    'RingCount', 'AliphaticRings', 'AromaticRings',
    'Heteroatoms', 'BalabanJ', 'BertzCT', 'LabuteASA', 'MolMR', 'ValenceElectrons'
] + list(smarts_patterns.keys())

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * len(descriptor_names)

    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.RingCount(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumHeteroatoms(mol),
        Descriptors.BalabanJ(mol),
        Descriptors.BertzCT(mol),
        Descriptors.LabuteASA(mol),
        Descriptors.MolMR(mol),
        Descriptors.NumValenceElectrons(mol),
    ]

    for name, pattern in smarts_patterns.items():
        patt = Chem.MolFromSmarts(pattern)
        descriptors.append(len(mol.GetSubstructMatches(patt)))

    return descriptors

# ðŸš¨ Ensure X_test has a SMILES column or Series
# Example if X_test is a DataFrame with a SMILES column:
flattened_test_features = X_test['SMILES'].apply(compute_descriptors).apply(pd.Series)
flattened_test_features.columns = descriptor_names

# Combine with SMILES if needed:
X_test_with_descriptors = pd.concat([X_test[['SMILES']].reset_index(drop=True), flattened_test_features], axis=1)

# Display result
print(X_test_with_descriptors.head())
X_test=X_test_with_descriptors
X_test = X_test.drop('SMILES',axis=1)
duplicates = X_test.columns[X_test.columns.duplicated()]
print("Duplicate columns:", duplicates.tolist())
X_test = X_test.loc[:, ~X_test.columns.duplicated()]  # If using test set




#insert model 1 here with X_train1 and y_train1


import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

cat_features = X_train1.select_dtypes(include=["object", "category"]).columns.tolist()

# Initialize model
model1 = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=4,
    l2_leaf_reg=10,
    early_stopping_rounds=10,
    verbose=0
)

# K-Fold CV
cv = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_list = []
r2_list = []

for train_idx, val_idx in cv.split(X_train1):
    X_train, X_val = X_train1.iloc[train_idx], X_train1.iloc[val_idx]
    y_train, y_val = y_train1.iloc[train_idx], y_train1.iloc[val_idx]

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    model1.fit(train_pool, eval_set=val_pool)
    
    preds = model1.predict(X_val)
    
    rmse = mean_squared_error(y_val, preds, squared=False)
    r2 = r2_score(y_val, preds)

    rmse_list.append(rmse)
    r2_list.append(r2)

print(f"Mean RMSE: {sum(rmse_list)/len(rmse_list):.4f}")
print(f"Mean RÂ²: {sum(r2_list)/len(r2_list):.4f}")



y_pred1 = model1.predict(X_test)


df.insert(loc = len(df.columns), column='FVV',value=data['FFV'])
df = df.drop('Tg',axis=1)
df2 = df.dropna(subset=['FVV'])
y_train2 = df2['FVV']
X_train2 = df2.drop('FVV',axis=1)#CHANGE NAME FFV
X_train2 = X_train2.drop('SMILES',axis =1)
# Check for duplicates
duplicates = X_train2.columns[X_train2.columns.duplicated()]
X_train2 = X_train2.loc[:, ~X_train2.columns.duplicated()]



#insert model 2 here with X_train2 and y_train2


import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

cat_features = X_train1.select_dtypes(include=["object", "category"]).columns.tolist()

# Initialize model
model2 = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=4,
    l2_leaf_reg=10,
    early_stopping_rounds=10,
    verbose=0
)

# K-Fold CV
cv = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_list = []
r2_list = []

for train_idx, val_idx in cv.split(X_train2):
    X_train, X_val = X_train2.iloc[train_idx], X_train2.iloc[val_idx]
    y_train, y_val = y_train2.iloc[train_idx], y_train2.iloc[val_idx]

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    model2.fit(train_pool, eval_set=val_pool)
    
    preds = model2.predict(X_val)
    
    rmse = mean_squared_error(y_val, preds, squared=False)
    r2 = r2_score(y_val, preds)

    rmse_list.append(rmse)
    r2_list.append(r2)

print(f"Mean RMSE: {sum(rmse_list)/len(rmse_list):.4f}")
print(f"Mean RÂ²: {sum(r2_list)/len(r2_list):.4f}")


y_pred2 = model2.predict(X_test)


df.insert(loc = len(df.columns), column='Tc',value=data['Tc'])
df = df.drop('FVV',axis=1)
df3 = df.dropna(subset=['Tc'])
y_train3 = df3['Tc']
X_train3 = df3.drop('Tc',axis=1)#CHANGE NAME FFV
X_train3 = X_train3.drop('SMILES',axis =1)
# Check for duplicates
duplicates = X_train3.columns[X_train3.columns.duplicated()]
X_train3 = X_train3.loc[:, ~X_train3.columns.duplicated()]



#insert model 3 here with X_train3 and y_train3


import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

cat_features = X_train1.select_dtypes(include=["object", "category"]).columns.tolist()

# Initialize model
model3 = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=4,
    l2_leaf_reg=10,
    early_stopping_rounds=10,
    verbose=0
)

# K-Fold CV
cv = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_list = []
r2_list = []

for train_idx, val_idx in cv.split(X_train3):
    X_train, X_val = X_train3.iloc[train_idx], X_train3.iloc[val_idx]
    y_train, y_val = y_train3. iloc[train_idx], y_train3.iloc[val_idx]

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    model3.fit(train_pool, eval_set=val_pool)
    
    preds = model3.predict(X_val)
    
    rmse = mean_squared_error(y_val, preds, squared=False)
    r2 = r2_score(y_val, preds)

    rmse_list.append(rmse)
    r2_list.append(r2)

print(f"Mean RMSE: {sum(rmse_list)/len(rmse_list):.4f}")
print(f"Mean RÂ²: {sum(r2_list)/len(r2_list):.4f}")


y_pred3 = model3.predict(X_test)


df.insert(loc = len(df.columns), column='Density',value=data['Density'])
df = df.drop('Tc',axis=1)
df4 = df.dropna(subset=['Density'])
y_train4 = df4['Density']
X_train4 = df4.drop('Density',axis=1)#CHANGE NAME FFV
X_train4 = X_train4.drop('SMILES',axis =1)
# Check for duplicates
duplicates = X_train4.columns[X_train4.columns.duplicated()]
X_train4 = X_train4.loc[:, ~X_train4.columns.duplicated()]



#insert model 4 here with X_train4 and y_train4


import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

cat_features = X_train4.select_dtypes(include=["object", "category"]).columns.tolist()

# Initialize model
model4 = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=4,
    l2_leaf_reg=10,
    early_stopping_rounds=10,
    verbose=0
)

# K-Fold CV
cv = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_list = []
r2_list = []

for train_idx, val_idx in cv.split(X_train4):
    X_train, X_val = X_train4.iloc[train_idx], X_train4.iloc[val_idx]
    y_train, y_val = y_train4.iloc[train_idx], y_train4.iloc[val_idx]

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    model4.fit(train_pool, eval_set=val_pool)
    
    preds = model4.predict(X_val)
    
    rmse = mean_squared_error(y_val, preds, squared=False)
    r2 = r2_score(y_val, preds)

    rmse_list.append(rmse)
    r2_list.append(r2)

print(f"Mean RMSE: {sum(rmse_list)/len(rmse_list):.4f}")
print(f"Mean RÂ²: {sum(r2_list)/len(r2_list):.4f}")


y_pred4 = model4.predict(X_test)


df.insert(loc = len(df.columns), column='Rg',value=data['Rg'])
df = df.drop('Density',axis=1)
df5 = df.dropna(subset=['Rg'])
y_train5 = df5['Rg']
X_train5 = df5.drop('Rg',axis=1)#CHANGE NAME FFV
X_train5 = X_train5.drop('SMILES',axis =1)
# Check for duplicates
duplicates = X_train5.columns[X_train5.columns.duplicated()]
X_train5 = X_train5.loc[:, ~X_train5.columns.duplicated()]



#insert model 5 here with X_train5 and y_train5


import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

cat_features = X_train4.select_dtypes(include=["object", "category"]).columns.tolist()

# Initialize model
model5 = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=4,
    l2_leaf_reg=10,
    early_stopping_rounds=10,
    verbose=0
)

# K-Fold CV
cv = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_list = []
r2_list = []

for train_idx, val_idx in cv.split(X_train5):
    X_train, X_val = X_train5.iloc[train_idx], X_train5.iloc[val_idx]
    y_train, y_val = y_train5.iloc[train_idx], y_train5.iloc[val_idx]

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    model5.fit(train_pool, eval_set=val_pool)
    
    preds = model5.predict(X_val)
    
    rmse = mean_squared_error(y_val, preds, squared=False)
    r2 = r2_score(y_val, preds)

    rmse_list.append(rmse)
    r2_list.append(r2)

print(f"Mean RMSE: {sum(rmse_list)/len(rmse_list):.4f}")
print(f"Mean RÂ²: {sum(r2_list)/len(r2_list):.4f}")


y_pred5 = model5.predict(X_test)


Submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
Submission = Submission.drop('SMILES',axis = 1)
Submission['Tg'] = y_pred1
Submission['FFV'] = y_pred2
Submission['Tc']= y_pred3
Submission['Density'] = y_pred4
Submission['Rg'] = y_pred5


Submission


submission = Submission
submission.to_csv("submission.csv", index=False)





