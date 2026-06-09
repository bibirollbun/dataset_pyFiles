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


train_original = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
test_original = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")


col, cnt, length = "column", "count", "total"
print(f"{col:<10} {cnt:<10} {length:<10}")
for col in train_original.columns:
    print(f"{col:<10} {train_original[col].notna().sum():<10} {len(train_original[col]):<10}")


!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

def cal_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    desc = Descriptors.CalcMolDescriptors(mol)
    return desc


desc = pd.concat((train_original["SMILES"], test_original["SMILES"]), axis=0).apply(cal_descriptors).apply(pd.Series)


col, cnt, length, unik = "column", "count", "total", "unique"
print(f"{col:<35} {cnt:<10} {length:<10} {unik:<10}")
for col in desc.columns:
    print(f"{col:<35} {desc[col].notna().sum():<10} {len(desc[col]):<10} {desc[col].nunique():<10}")
    if desc[col].notna().sum() < len(desc[col]) or desc[col].nunique() == 1:
        desc.drop(columns=col, axis=1, inplace=True)


mean = desc.mean()
std = desc.std()
col, str_mean, str_std = "column", "mean", "std"
print(f"{col:<35} {str_mean:<35} {str_std:<35}")
for col in desc.columns:
    print(f"{col:<35} {mean[col]:<35} {std[col]:<35}")


desc_z = (desc - mean) / std

n_feature = 50
from sklearn.decomposition import PCA
pca = PCA(n_components=n_feature)

desc_pca = pca.fit_transform(desc_z)
ratio = pca.explained_variance_ratio_


import matplotlib.pyplot as plt

plt.figure()
plt.bar([i for i in range(n_feature)], ratio)
plt.xlabel("PCA feature")
plt.ylabel("explained variance ratio")
plt.show()


desc_pca_df = pd.DataFrame(desc_pca)
features = [f"PCA_{i}" for i in range(n_feature)]
desc_pca_df.columns = features

to_train = ["Tg", "FFV", "Tc", "Density", "Rg"]
train_full = pd.concat((desc_pca_df.iloc[:len(train_original), :], train_original[to_train]), axis=1)

train_dict = {}
for col in to_train:
    train_dict[col] = train_full[[*features, col]][train_full[col].notna()]


from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error


model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=7,
    reg_lambda=1,
    random_state=114514
)

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=114514)

for key in train_dict:
    print(key)

    X, y = train_dict[key][features], train_dict[key].iloc[:, -1]
    fold = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2, mse = r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred)
        print(f"fold {fold}: r2: {r2}, mse: {mse}")
        fold += 1
    print("############################################################")


model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=7,
    reg_lambda=1,
    random_state=114514
)

model_dict = {}
X_test = desc_pca_df.iloc[len(train_original):, :]
submission = pd.DataFrame(None)
submission["id"] = test_original["id"]

for key in train_dict:
    model_dict[key] = model
    
    X_train, y_train = train_dict[key][features], train_dict[key].iloc[:, -1]

    model_dict[key].fit(X_train, y_train)
    y_pred = model.predict(X_train)

    r2, mse = r2_score(y_train, y_pred), mean_squared_error(y_train, y_pred)
    print(f"{key}: r2: {r2}, mse: {mse}")

    submission[key] = model_dict[key].predict(X_test)


submission.to_csv("/kaggle/working/submission.csv", index=None)

