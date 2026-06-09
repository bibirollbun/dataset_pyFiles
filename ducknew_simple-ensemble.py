import pandas as pd

# 0.43916
df1 = pd.read_csv("/kaggle/input/catboost-ranker-baseline-flightrank-2025/submission.csv")
# 0.42163
df2 = pd.read_csv("/kaggle/input/lightgbm-ranker-ndcg-3/submission.csv")
# 0.47635
df3 = pd.read_csv("/kaggle/input/ensemble-with-polars/submission.csv")
# 0.48388
df4 = pd.read_csv("/kaggle/input/xgboost-ranker-with-polars/submission.csv")


import pandas as pd

dfs = [
    df1,df2,df3,df4
]

def rank2score(sr, eps=1e-6):
    n = sr.max()
    return 1.0 - (sr - 1) / (n + eps)

score_frames = []
for i, df in enumerate(dfs):
    tmp = df[['Id', 'ranker_id', 'selected']].copy()
    tmp['score'] = tmp.groupby('ranker_id')['selected'].transform(rank2score)
    score_frames.append(tmp[['Id', 'ranker_id', 'score']].rename(columns={'score': f'score_{i}'}))

merged = score_frames[0]
for i in range(1, 4):
    merged = merged.merge(score_frames[i], on=['Id', 'ranker_id'], how='left')

weights = [0.1, 0.1, 0.1, 0.7]
score_cols = [f'score_{i}' for i in range(4)]
w = pd.Series(weights, index=score_cols)
merged['score_mean'] = (merged[score_cols] * w).sum(axis=1) / w.sum()

def score2rank(s):
    return s.rank(method='first', ascending=False).astype(int)

merged['selected'] = merged.groupby('ranker_id')['score_mean'].transform(score2rank)

out = merged[['Id', 'ranker_id', 'selected']]
out.to_csv("submission.csv", index=False, float_format='%.0f')

