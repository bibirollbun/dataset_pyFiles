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


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor



train = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")



train.tail()


import re


def extract_features(df):
    features = pd.DataFrame()
    smiles = df['SMILES']
    
    # Basic features
    features['smiles_len'] = smiles.apply(len)
    features['num_atoms'] = smiles.apply(lambda x: len(re.findall(r'[A-Z][a-z]?', x)))
    
    # Count specific atoms
    for atom in ['C', 'O', 'N', 'H', 'F', 'Cl', 'Br', 'S', 'Si']:
        features[f'num_{atom}'] = smiles.str.count(atom)

    # Bond types
    features['num_single_bonds'] = smiles.str.count('-')
    features['num_double_bonds'] = smiles.str.count('=')
    features['num_triple_bonds'] = smiles.str.count('#')
    features['num_aromatic'] = smiles.str.count('c')  # lowercase c = aromatic carbon

    # Ring indicators
    for i in range(1, 10):
        features[f'ring_{i}'] = smiles.str.count(str(i))
    features['total_rings'] = features[[f'ring_{i}' for i in range(1, 10)]].sum(axis=1)

    # Branching & complexity
    features['num_branches'] = smiles.str.count(r'\(')
    features['num_close_branches'] = smiles.str.count(r'\)')
    features['branch_depth'] = features['num_branches'] - features['num_close_branches']

    # stereochemistry
    features['has_chirality'] = smiles.str.contains('@').astype(int)
    features['has_charge'] = smiles.str.contains('[+-]').astype(int)
    features['has_dot'] = smiles.str.contains('\.').astype(int)  # multiple fragments

    return features



X_train = extract_features(train)
X_test = extract_features(test)



target_features = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']


submission = pd.DataFrame()
submission['id'] = test['id']



for target in target_features:
    y = train[target]
    mask = y.notna()
    
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train[mask], y[mask])

    submission[target] = model.predict(X_test)

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Submission file created: submission.csv")


import shutil
shutil.move("submission.csv", "/kaggle/working/submission.csv")



print(submission.shape)
print(submission.head())
print(submission.columns)





