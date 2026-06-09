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


import glob
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# ========= データ読み込み =========
path = Path('/kaggle/input/tabular-playground-series-nov-2022/')
submission_format = pd.read_csv(path / 'sample_submission.csv', index_col='id')
train = pd.read_csv(path / 'train_labels.csv', index_col='id')

# ========= インデックス確認 =========
sub_ids = submission_format.index
gt_ids = train.index
y_true = train.loc[gt_ids]['label'].values

print("データ長")
print(f"sub_ids: {len(sub_ids)}")
print(f"gt_ids:  {len(gt_ids)}")
print(f"y_true:  {len(y_true)}\n")

# ========= サブミッションファイル読み込み =========
files = sorted(glob.glob(str(path / 'submission_files/*.csv')))
dfs = [pd.read_csv(f).set_index('id') for f in files]
print(f"ファイル数: {len(files)}")
print(files[:3])  # 最初の3ファイル名

# ========= 予測行列作成 =========
assert all(len(df) == 40000 for df in dfs), "40000 行なし"
assert all("pred" in df.columns for df in dfs), "列 'pred' が必要"

pred_matrix = np.column_stack([df["pred"].values for df in dfs])  # shape: (40000, n_models)
index_all = dfs[0].index

# ========= LogLoss 評価用関数 =========
results = {}
blends = {}

def safe_log_loss(pred, name):
    pred_series = pd.Series(pred, index=index_all)
    clipped = np.clip(pred_series.loc[gt_ids], 1e-5, 1 - 1e-5)
    loss = log_loss(y_true, clipped)
    results[name] = loss
    blends[name] = pred
    print(f"{name}: LogLoss = {loss:.6f}")

# ========= 重み付き平均（手動設定） =========
weights = np.linspace(1, 0.1, len(dfs))
weights /= weights.sum()
weighted_avg = np.average(pred_matrix, axis=1, weights=weights)
safe_log_loss(weighted_avg, "Weighted Average")

# ========= スタッキング（メタ特徴量） =========
X_meta_train = pred_matrix[:20000]
y_train = y_true

# ========== 最良手法の決定 ==========
best_method = min(results, key=results.get)
print(f"\n最良手法: {best_method}, LogLoss = {results[best_method]:.6f}")

# ========= 提出ファイル作成 =========
best_pred = blends[best_method]
final_submission = pd.DataFrame({"pred": best_pred}, index=index_all)
final_submission.loc[sub_ids].to_csv("submission.csv")
print("✅ 提出ファイル 'submission.csv' を保存しました")

# ========= 再確認 =========
clipped_pred = np.clip(final_submission.loc[gt_ids]['pred'].values, 1e-5, 1 - 1e-5)
final_score = log_loss(y_true, clipped_pred)
print(f"✅ 再評価 LogLoss: {final_score:.6f}")

# ========= ファイル確認 =========
print("提出ファイルの先頭:")
print(final_submission.head())


