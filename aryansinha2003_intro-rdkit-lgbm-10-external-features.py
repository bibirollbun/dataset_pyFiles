!pip install /kaggle/input/rdkit-install-whl/rdkit_wheel/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error


train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')


train_df.isnull().sum()


TARGETS = ["Density", "Tc", "Tg", "Rg", "FFV"]
SMILES_COL = "SMILES"


# Feature extraction function using RDKit
def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * 10

    atom_counts = {}
    atomic_mass = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        atom_counts[symbol] = atom_counts.get(symbol, 0) + 1
        atomic_mass.append(atom.GetMass())

    bond_counts = {'single': 0, 'double': 0, 'triple': 0, 'aromatic': 0}
    for bond in mol.GetBonds():
        btype = bond.GetBondType()
        if btype.name == 'SINGLE':
            bond_counts['single'] += 1
        elif btype.name == 'DOUBLE':
            bond_counts['double'] += 1
        elif btype.name == 'TRIPLE':
            bond_counts['triple'] += 1
        elif bond.GetIsAromatic():
            bond_counts['aromatic'] += 1

    features = [
        Descriptors.MolWt(mol),
        Descriptors.RingCount(mol),
        atom_counts.get('C', 0),
        atom_counts.get('O', 0),
        atom_counts.get('N', 0),
        np.mean(atomic_mass),
        bond_counts['single'],
        bond_counts['double'],
        bond_counts['triple'],
        bond_counts['aromatic']
    ]
    return features



# Apply feature extraction
feature_names = ['MolWt', 'RingCount', 'C_count', 'O_count', 'N_count', 'AvgMass', 'SingleBonds', 'DoubleBonds', 'TripleBonds', 'AromaticBonds']
train_features = train_df[SMILES_COL].apply(extract_features).to_list()
test_features = test_df[SMILES_COL].apply(extract_features).to_list()


train_features[:10]


X = pd.DataFrame(train_features, columns=feature_names)
y = train_df[TARGETS]


X


X_new= train_df.join(X,how="left")
X_new


y


X_test = pd.DataFrame(test_features, columns=feature_names)
X_test


from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

models = {}
scalers = {}
imputers = {}
test_predictions = {}

for target in TARGETS:
    y = train_df[target]
    valid_indices = y.dropna().index
    X_valid = X.loc[valid_indices].copy()
    y_valid = y.loc[valid_indices]

    # Standardize features before KNN imputation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X_scaled)

    X_train, X_val, y_train, y_val = train_test_split(X_imputed, y_valid, test_size=0.2, random_state=121)

    model = LGBMRegressor(learning_rate=0.1, max_depth=7,n_estimators=200, num_leaves=63,random_state=121)
    model.fit(X_train, y_train)

    models[target] = model
    scalers[target] = scaler
    imputers[target] = imputer

    # Predict on the real test set
    X_test_scaled = scaler.transform(X_test)
    X_test_imputed = imputer.transform(X_test_scaled)
    test_predictions[target] = model.predict(X_test_imputed)

# Prepare submission DataFrame with test IDs and predictions for each target
submission = test_df[['id']].copy()
for target in TARGETS:
    submission[target] = test_predictions[target]

submission.to_csv('submission.csv', index=False)


