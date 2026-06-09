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





# install RDKit for offline use
!pip install /kaggle/input/rdkit-install-whl/rdkit_wheel/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


import pandas as pd
from sklearn.model_selection import train_test_split

import pandas as pd

csv_path = '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv'
train_df = pd.read_csv(csv_path)
csv2_path = '/kaggle/input/neurips-open-polymer-prediction-2025/test.csv'
test_df = pd.read_csv(csv2_path)





train_df.head(4)


# データの行数・列数確認
print(train_df.shape)

# 列名とデータ型の確認
print(train_df.info())

# 先頭5行を表示
print(train_df.head())


# 各列の欠損数・割合を確認
missing_counts = train_df.isnull().sum()
missing_ratio = train_df.isnull().mean()
print(pd.concat([missing_counts, missing_ratio], axis=1, keys=['missing_count', 'missing_ratio']))


print(train_df.describe())





from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import pandas as pd
from tqdm import tqdm

# 1. 208個のdescriptor名を取得
descriptor_names = [desc[0] for desc in Descriptors._descList]

# 2. 計算器を準備
calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

# 3. 記述子を計算する関数
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * len(descriptor_names)
    return calc.CalcDescriptors(mol)

# 4. SMILES列からdescriptorを計算
descriptor_df = pd.DataFrame(
    [compute_descriptors(smi) for smi in tqdm(train_df['SMILES'])],
    columns=descriptor_names
)

# 5. 元のtrain_dfと連結（インデックス揃え）
train_df_with_desc = pd.concat([train_df.reset_index(drop=True), descriptor_df], axis=1)

# Optional: 欠損を確認
print(train_df_with_desc.isnull().sum().sort_values(ascending=False).head())


import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

X_all = train_df_with_desc[descriptor_names]
y_all = train_df_with_desc['FFV']

# NaNでない部分（学習用）
train_mask = y_all.notnull()
X_train = X_all.loc[train_mask]
y_train = y_all.loc[train_mask]

# NaN部分（予測用）
pred_mask = ~train_mask
X_pred = X_all.loc[pred_mask]

# 学習
model = LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42)
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_pred)

# 元のDataFrameのNaN部分を予測値で埋める
y_filled = y_all.copy()
y_filled.loc[pred_mask] = y_pred

# 結果を確認
print(y_filled.isnull().sum())  # 0 なら埋まっている


from lightgbm import LGBMRegressor

for target in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
    target_series = train_df_with_desc[target]
    non_null_mask = target_series.notnull()

    X_train = train_df_with_desc.loc[non_null_mask, descriptor_names]
    y_train = target_series.loc[non_null_mask]

    X_pred = train_df_with_desc.loc[~non_null_mask, descriptor_names]

    if X_pred.shape[0] == 0:
        print(f"No missing values in {target}")
        continue

    model = LGBMRegressor(n_estimators=500, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_pred)

    train_df_with_desc.loc[~non_null_mask, target] = y_pred
    print(f"Filled missing values in {target} with predicted values")


# 補完後の分布の確認

import matplotlib.pyplot as plt
import seaborn as sns

targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

for target in targets:
    plt.figure(figsize=(6,4))
    sns.histplot(train_df_with_desc[target].dropna(), bins=50, kde=True)
    plt.title(f'{target} Distribution After Imputation')
    plt.xlabel(target)
    plt.ylabel('Count')
    plt.show()





test_df.head()


# 1. SMILES列からdescriptorを計算
descriptor_df = pd.DataFrame(
    [compute_descriptors(smi) for smi in tqdm(test_df['SMILES'])],
    columns=descriptor_names
)

# 2. 元のtrain_dfと連結（インデックス揃え）
test_df_with_desc = pd.concat([test_df.reset_index(drop=True), descriptor_df], axis=1)

# Optional: 欠損を確認
print(test_df_with_desc.isnull().sum().sort_values(ascending=False).head())


from lightgbm import LGBMRegressor

targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

for target in targets:
    # 欠損がないデータで学習
    non_null_mask = train_df_with_desc[target].notnull()
    X_train = train_df_with_desc.loc[non_null_mask, descriptor_names]
    y_train = train_df_with_desc.loc[non_null_mask, target]

    # testデータ（欠損値想定なし）
    X_test = test_df_with_desc[descriptor_names]

    # モデル構築と学習
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 予測
    y_pred = model.predict(X_test)

    # test_df_with_desc に保存
    test_df_with_desc[f'{target}_pred'] = y_pred
    print(f"✅ {target} 予測完了")


# 1. 提出に必要な列のみ抽出
submission = test_df_with_desc[['id'] + [f'{t}_pred' for t in targets]].copy()

# 2. カラム名を 'Tg', 'FFV', ... に戻す
submission.columns = ['id'] + targets

# 3. CSV出力
submission.to_csv('submission.csv', index=False)

print("✅ 提出用ファイル 'submission.csv' を作成しました。")
print(submission.head())




