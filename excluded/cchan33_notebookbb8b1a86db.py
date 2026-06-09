import glob
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from scipy.optimize import minimize


path = Path('/kaggle/input/tabular-playground-series-nov-2022/')
submission_format = pd.read_csv(path / 'sample_submission.csv', index_col='id')
train = pd.read_csv(path / 'train_labels.csv', index_col='id')


sub_ids = submission_format.index         
gt_ids = train.index                     
y_true = train.loc[gt_ids]['label'].values


files = sorted(glob.glob(str(path / 'submission_files/*.csv')))
dfs = [pd.read_csv(f).set_index('id') for f in files]


assert all(len(df) == 40000 for df in dfs), "40000 行なし"


# 確かめ
print("データ長")
print(f"sub_ids: {len(sub_ids)}")
print(f"gt_ids:  {len(gt_ids)}")
print(f"y_true:  {len(y_true)}\n")

print(f"ファイル数: {len(files)}")
print(files[:5]) 


pred_matrix = np.column_stack([df['pred'].values for df in dfs])
index_all = dfs[0].index 


print(f"pred_matrix shape: {pred_matrix.shape}")


# 各手法の結果（LogLoss）を保存する辞書
results = {}
blends = {}


def safe_log_loss(pred, name):
    pred_series = pd.Series(pred, index=dfs[0].index)  # 40,000件のID（全体）
    clipped = np.clip(pred_series.loc[gt_ids], 1e-5, 1-1e-5)  # 評価は前半20,000件
    loss = log_loss(y_true, clipped)
    results[name] = loss
    blends[name] = pred
    print(f"{name}: LogLoss = {loss:.6f}")


# 各モデルの予測値を rank に変換
rank_matrix = np.apply_along_axis(
    lambda x: pd.Series(x).rank().values, axis=0, arr=pred_matrix
)


rank_avg = rank_matrix.mean(axis=1) / rank_matrix.max()
safe_log_loss(rank_avg, "Rank Average")


rank_pred = blends['Rank Average']
final_submission = pd.DataFrame({'pred': rank_pred}, index=dfs[0].index)
final_submission.loc[sub_ids].to_csv('submission.csv')
print(f"手法：ランク平均 'Rank Average', 提出ファイル 'submission.csv' を出力完了")
print(rank_pred)


pred_series = pd.Series(rank_pred, index=dfs[0].index)
clipped_pred = np.clip(pred_series.loc[gt_ids], 1e-5, 1 - 1e-5)
print(clipped_pred)
final_score = log_loss(y_true, clipped_pred)
print(f"ランク平均 'Rank Average', 評価 LogLoss = {final_score:.6f}")

