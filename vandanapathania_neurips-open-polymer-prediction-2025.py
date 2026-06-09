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


import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# 1. Load train and test data
train_path = '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv'
test_path = '/kaggle/input/neurips-open-polymer-prediction-2025/test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# 2. Define target columns and feature extraction from SMILES
target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

def smiles_string_features(smiles):
    length = len(smiles)
    num_rings = smiles.count('c1')
    num_branches = smiles.count('(')
    num_stars = smiles.count('*')
    num_C = smiles.count('C')
    num_O = smiles.count('O')
    num_N = smiles.count('N')
    return [length, num_rings, num_branches, num_stars, num_C, num_O, num_N]

feature_names = ['smiles_len', 'num_rings', 'num_branches', 'num_stars', 'num_C', 'num_O', 'num_N']

for df in [train, test]:
    df[feature_names] = df['SMILES'].apply(lambda x: pd.Series(smiles_string_features(x)))

# 3. Train a separate RandomForest for each target property using available data
models = {}
for col in target_cols:
    df = train[~train[col].isnull()].copy()
    X = df[feature_names]
    y = df[col]
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    models[col] = model

# 4. Predict on test set
X_test = test[feature_names]
test_preds = pd.DataFrame({'id': test['id']})
for col in target_cols:
    test_preds[col] = models[col].predict(X_test)

# 5. Prepare submission file
submission = test_preds[['id'] + target_cols]
submission.to_csv('submission.csv', index=False)
print(submission)




