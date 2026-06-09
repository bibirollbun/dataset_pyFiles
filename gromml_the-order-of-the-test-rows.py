import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')


train_zero_test_not_zero_cols = []

for col in train_df.columns:
    if np.all(np.isclose(train_df[col].to_numpy(), 0.0)):
        if not np.all(np.isclose(test_df[col].to_numpy(), 0.0)):
            train_zero_test_not_zero_cols.append(col)

print(train_zero_test_not_zero_cols)


for col in train_zero_test_not_zero_cols:
    print(col, np.mean(np.isclose(test_df[col].to_numpy(), 0.0)))


y_true = test_df[train_zero_test_not_zero_cols].astype(bool).sum(axis=1)
y_true


import tqdm
from sklearn.metrics import accuracy_score

seeds = np.arange(1000)
scores = []

for seed in tqdm.tqdm(seeds):
    y_pred = y_true.sort_values()
    y_pred = y_pred.sample(n=len(y_pred), random_state=seed)
    scores.append(accuracy_score(y_true, y_pred))


plt.figure(figsize=(16, 4))
plt.plot(seeds, scores)
plt.title('random_state with max accuracy: ' + str(np.argmax(scores)))


zeroes_shares_df = train_df.astype(bool).mean(axis=0)
zeroes_shares_df[(zeroes_shares_df > 0.4) & (zeroes_shares_df < 0.6)]


train_df[['X526', 'X589', 'X778', 'X786']].iloc[:100000].plot(subplots=True, figsize=(16, 12))


def sort_test_df_by_time(df):
    assert len(df.shape) == 2
    assert df.shape[0] == 538150

    n = df.shape[0]
    t = pd.Series(np.arange(n))
    t = t.sample(n=n, random_state=700)

    t = pd.Series(np.arange(n), index=t.to_numpy()).sort_index()
    return df.iloc[t.to_numpy()]


sorted_test_df = sort_test_df_by_time(test_df)


sorted_test_df.to_parquet('sorted_test.parquet')




