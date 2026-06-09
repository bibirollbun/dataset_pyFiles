import numpy as np
import pandas as pd

df1 = pd.read_csv("/kaggle/input/memory-optimized-transformers-for-impostor-hunt/submission.csv")
df2 = pd.read_csv("/kaggle/input/combining-feature-extraction-bert/submission.csv")
df3 = pd.read_csv("/kaggle/input/0-87759-fake-or-real-bert-pca-randomforest/submission.csv")
df4 = pd.read_csv("/kaggle/input/truthgpt-spotting-real-in-this-fake-world/submission.csv")
df5 = pd.read_csv("/kaggle/input/0-84232-enssenbel-4-model-impostor-hunt/submission.csv")


dfs = [df1, df2, df3, df4, df5]
for df in dfs:
    df.sort_values('id', inplace=True)
    df.reset_index(drop=True, inplace=True)

weights = [0.35, 0.35, 0.1, 0.1, 0.1]
score_1 = np.zeros(len(df1))
score_2 = np.zeros(len(df1))

for df, w in zip(dfs, weights):
    score_1 += (df['real_text_id'] == 1) * w
    score_2 += (df['real_text_id'] == 2) * w

final_pred = np.where(score_1 >= score_2, 1, 2).astype(int)
submission = pd.DataFrame({'id': df1['id'], 'real_text_id': final_pred})
submission.to_csv("submission.csv", index=False)

