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


path = Path('/kaggle/input/tabular-playground-series-nov-2022/')
submission_format = pd.read_csv(path / 'sample_submission.csv', index_col='id')
train = pd.read_csv(path / 'train_labels.csv', index_col='id')


sub_ids = submission_format.index         
gt_ids = train.index                     
y_true = train.loc[gt_ids]['label'].values


print("Data-Length")
print(f"sub_ids: {len(sub_ids)}")
print(f"gt_ids:  {len(gt_ids)}")
print(f"y_true:  {len(y_true)}\n")



files = sorted(glob.glob(str(path / 'submission_files/*.csv')))
dfs = [pd.read_csv(f).set_index('id') for f in files]


print(f"Number of files: {len(files)}")
print(files[:3]) 
assert all(len(df) == 40000 for df in dfs), "No 40000 rows"


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


simple_avg = np.mean(pred_matrix, axis=1)
safe_log_loss(simple_avg, "Simple Average")


weights = np.linspace(1, 0.1, len(dfs))
weights /= weights.sum()
weighted_avg = np.average(pred_matrix, axis=1, weights=weights)
safe_log_loss(weighted_avg, "Weighted Average")


rank_matrix = np.apply_along_axis(lambda x: pd.Series(x).rank().values, axis=0, arr=pred_matrix)
rank_avg = rank_matrix.mean(axis=1) / rank_matrix.max()
safe_log_loss(rank_avg, "Rank Average")


def logit(p):
    return np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


logits = np.array([logit(p) for p in pred_matrix.T])
logit_avg = sigmoid(np.mean(logits, axis=0))
safe_log_loss(logit_avg, "Logit Average")


X_meta_train = pred_matrix[:20000]


X_meta_full = pred_matrix


X_train = X_meta_train
y_train = y_true


print(len(y_true))
print(len(X_train))


"""
model = make_pipeline(
    MinMaxScaler(), # pred_matrixが 0〜1 のときの標準的スケーラ
    LogisticRegression(
        penalty='l1', # L1正則化
        solver='saga', # モデルのパラメータ
        C=1, # 正則化強度
        max_iter=200,
        random_state=1234
    )
)
"""


C_values = np.logspace(-2, 0.5, 5)


cv_scores = []


"""
for C in C_values:
    model = make_pipeline(
        MinMaxScaler(),
        LogisticRegression(C=C, max_iter=200, solver='liblinear', random_state=1234)
    )
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    cv_scores.append(scores.mean())
"""


"""
plt.figure(figsize=(8,5))
plt.semilogx(C_values, cv_scores, marker='o')
plt.xlabel('C (Inverse of regularization strength)')
plt.ylabel('Mean CV ROC-AUC')
plt.title('Effect of C on Logistic Regression Performance')
plt.grid(True)
plt.show()
"""


model = make_pipeline(
    MinMaxScaler(), 
    LogisticRegression(
        penalty='l1',
        C=0.1, 
        max_iter=200, 
        solver='saga', 
        random_state=1234
    )
)
model.fit(X_meta_train, y_true)
lr_pred_full = model.predict_proba(X_meta_full)[:, 1]




safe_log_loss(lr_pred_full, "Logistic Regression")



ensemble_df = pd.DataFrame({"id": index_all})


for method_name, pred_array in blends.items():
    ensemble_df[method_name] = pred_array
    print(method_name)


print("各手法の予測値（先頭5行）:")
print(ensemble_df.head())


for name, score in results.items():
    print(f"{name}: {score:.6f}")


plt.figure(figsize=(10, 6))
bars = plt.bar(results.keys(), results.values(), color='skyblue')
best_method = min(results, key=results.get)
bars[list(results.keys()).index(best_method)].set_color("orange")
plt.ylabel("Log Loss")
plt.title("Ensemble Methods Comparison")
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()



best_pred = blends[best_method]
final_submission = pd.DataFrame({'pred': best_pred}, index=dfs[0].index)
final_submission.loc[sub_ids].to_csv('submission.csv')
print(f"提出ファイル 'submission.csv' を出力完了")


pred_series = pd.Series(best_pred, index=dfs[0].index)
clipped_pred = np.clip(pred_series.loc[gt_ids], 1e-5, 1 - 1e-5)
print(clipped_pred)
final_score = log_loss(y_true, clipped_pred)
print(f"最良手法 '{best_method}', 再評価 LogLoss = {final_score:.6f}")


print("提出ファイルの先頭:")
print(final_submission.loc[sub_ids])

