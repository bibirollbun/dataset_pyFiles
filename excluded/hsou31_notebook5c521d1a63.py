import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import os
import pandas as pd
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
from sklearn.metrics import log_loss
from scipy.optimize import minimize


submission_files = glob.glob("submission_files/*.csv")


dfs = [pd.read_csv(f).set_index("id") for f in submission_files]
print(dfs)


model_names = [f"model_{i}" for i in range(len(dfs))]


labels = pd.read_csv("/kaggle/input/tabular-playground-series-nov-2022/train_labels.csv").set_index("id")
print(labels.head())


train_df = pd.read_csv("/kaggle/input/tabular-playground-series-nov-2022/train_labels.csv").set_index("id")
print(train_df.head())


y_true = labels.loc[train_df.index].values 
print(y_true)


n_samples = 1000
n_models = 5000

dfs = [pd.DataFrame({"pred": np.random.rand(n_samples)}) for _ in range(n_models)]
pred_matrix = np.column_stack([df["pred"].values for df in dfs])


def correlation_loss(preds, targets):
    """予測と正解ラベルとの相関損失（1 - 相関係数）"""
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    return 1 - np.corrcoef(preds, targets)[0, 1]


# 例の予測値と正解
y_pred = np.random.rand(100)
y_true = np.random.rand(100)


# 相関損失を計算
corr_loss = correlation_loss(y_pred, y_true)


# 結果保存
results = {}
results["Correlation Average"] = corr_loss


plt.figure(figsize=(10, 6))
bars = plt.bar(results.keys(), results.values(), color="skyblue")
best_method = min(results, key=results.get)
bars[list(results.keys()).index(best_method)].set_color("orange")
plt.xticks(rotation=45)
plt.title("Log Loss Comparison of Ensemble Methods")
plt.ylabel("Log Loss (lower is better)")
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()


# 仮の予測値を生成（実際はCSV読み込みでもOK）
n = 1000
model1 = pd.Series(np.random.rand(n), name="model1")
model2 = pd.Series(np.random.rand(n), name="model2")
model3 = pd.Series(np.random.rand(n), name="model3")

# DataFrameにまとめる
preds = pd.concat([model1, model2, model3], axis=1)

# 各平均の計算
simple_avg = preds.mean(axis=1)                    # 単純平均
weighted_avg = preds.dot([0.2, 0.3, 0.5])           # 重み付き平均（例：model3を重視）
logit_avg = (np.log(preds / (1 - preds))).mean(axis=1)
logit_avg = 1 / (1 + np.exp(-logit_avg))            # ロジット平均
rank_avg = preds.rank().mean(axis=1) / len(preds)   # 順位平均（スケーリング）

# 最良結果を格納
best_pred = {
    "Simple Average": simple_avg,
    "Weighted Average": weighted_avg,
    "Logit Average": logit_avg,
    "Rank Average": rank_avg,
}


final_df = dfs[0].copy()
final_df["pred"] = best_pred


final_df = final_df.reset_index()
output_filename = f"{best_method.replace(' ', '_')}_submission.csv"
final_df.to_csv(output_filename, index=False)


print(f"\nBest Ensemble Method: {best_method}")
print(f"LogLoss = {results[best_method]:.6f}")
print(f"提出ファイル: {output_filename}")


print(os.listdir("/kaggle/input/tabular-playground-series-nov-2022"))


sample = pd.read_csv("./Correlation_Average_submission.csv")  # 自分の作業フォルダから読み込む


# 提出ファイルのテンプレートを読み込み
sample = pd.read_csv("/kaggle/input/tabular-playground-series-nov-2022/sample_submission.csv")

# 提出用CSVとして保存
sample.to_csv("submission.csv", index=False)


# 提出用データ確認
print(sample.head())
print(sample.columns)
print(sample.shape)

# 書き出し
sample.to_csv("submission.csv", index=False)


path = Path('/kaggle/input/tabular-playground-series-nov-2022/')
submission_format = pd.read_csv(path / 'sample_submission.csv', index_col='id')
train = pd.read_csv(path / 'train_labels.csv', index_col='id')


sub_ids = submission_format.index         
gt_ids = train.index                     
y_true = train.loc[gt_ids]['label'].values


print("データ長")
print(f"sub_ids: {len(sub_ids)}")
print(f"gt_ids:  {len(gt_ids)}")
print(f"y_true:  {len(y_true)}\n")


print(f"ファイル数: {len(files)}")
print(files[:3]) 
assert all(len(df) == 1000 for df in dfs)


pred_matrix = np.column_stack([df['pred'].values for df in dfs])
index_all = dfs[0].index 


print(f"pred_matrix shape: {pred_matrix.shape}")


results = {}
blends = {}


def safe_log_loss(pred, name):
    pred_series = pd.Series(pred, index=dfs[0].index)  # 40,000件のID（全体）
    clipped = np.clip(pred_series.loc[gt_ids], 1e-5, 1-1e-5)  # 評価は前半20,000件
    loss = log_loss(y_true, clipped)
    results[name] = loss
    blends[name] = pred
    print(f"{name}: LogLoss = {loss:.6f}")


X_meta_train = pred_matrix[:20000]


X_meta_full = pred_matrix


X_train = X_meta_train
y_train = y_true


print(len(y_true))
print(len(X_train))


C_values = np.logspace(-2, 0.5, 5)


cv_scores = []


ensemble_df = pd.DataFrame({"id": index_all})


# /kaggle/input 配下のフォルダ一覧を表示
print(os.listdir('/kaggle/input'))


print(os.listdir('/kaggle/input/tabular-playground-series-nov-2022'))


sample_sub = pd.read_csv('/kaggle/input/tabular-playground-series-nov-2022/sample_submission.csv')
print(sample_sub.head())


sample_sub = pd.read_csv('/kaggle/input/tabular-playground-series-nov-2022/sample_submission.csv')

# ダミー予測を用意
your_predictions = [0.5] * len(sample_sub)

# 提出ファイル作成
final_submission = sample_sub.copy()

final_submission.to_csv('submission.csv', index=False)
print(final_submission.head())


# 例: sample_subのidを使い、モデル予測結果を'prediction'カラムに入れる場合
final_submission = sample_sub.copy()

# ファイルに保存
final_submission.to_csv('submission.csv', index=False)


final_submission.to_csv('submission.csv', index=False)


print("提出ファイルの先頭:")
print(final_submission.head())


import pandas as pd

# 読み込み
df = pd.read_csv("submission.csv")

# 列名に余計なスペースがあれば削除
df.columns = df.columns.str.strip()

# 必須チェック
assert list(df.columns) == ['id', 'pred'], "列名が不正（id, pred）である必要があります"
assert len(df) == 20000, f"行数が不正（20000行であるべきだが、{len(df)}行）"
assert df['id'].is_unique, "id列が一意でない"
assert df['id'].min() == 20000 and df['id'].max() == 39999, "idが20000～39999でない"
assert df.isnull().sum().sum() == 0, "NaN値が含まれています"

# 再保存（安全のため）
df.to_csv("submission_fixed.csv", index=False)
print("✅ 提出用ファイルが正しく整いました：submission_fixed.csv")


# 問題なければ新しいファイル名で保存
df.to_csv("submission_fixed.csv", index=False)

print("✅ submission_fixed.csv を作成しました。")
print(df.head())

