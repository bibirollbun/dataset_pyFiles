import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


df_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col = 'id')
target_column = 'rainfall'
y_train = df_train[target_column]
y_train.head()


preds = np.zeros(y_train.shape[0])
roc_auc_score(y_train, preds)


preds = np.ones(y_train.shape[0])
roc_auc_score(y_train, preds)



preds = np.random.randint(0, 2, size=y_train.shape[0])
roc_auc_score(y_train, preds)


preds = np.random.rand(y_train.shape[0])
roc_auc_score(y_train, preds)


preds = 0.9 * y_train
print(f' Predictions:', preds)
print(f'ROC AUC:', roc_auc_score(y_train, preds))


sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
sub[target_column] = 0
sub.to_csv('submission.csv', index = False)


sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
sub[target_column] = np.random.rand(sub.shape[0])
sub.to_csv('submission.csv', index = False)


sub[target_column] = np.linspace(0, 1, sub.shape[0])  
sub.to_csv('submission.csv', index=False)


half = sub.shape[0] // 2
sub[target_column][:half] = 0.1
sub[target_column][half:] = 0.9
sub.to_csv('submission.csv', index=False)


y_train = df_train[['day', 'rainfall']]
y_train["year"] = y_train.index// 365 + 1 


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

for year in range(1, 7):  
    yearly_data = y_train[y_train["year"] == year]
    plt.scatter(yearly_data["day"], [year] * len(yearly_data), c=yearly_data["rainfall"], cmap="bwr", label=f"Year {year}")

plt.xlabel("Day of the Year")
plt.ylabel("Year")
plt.title("Days with Value 1 (Red) and 0 (Blue) per Year")
plt.colorbar(label="Value (0=Blue, 1=Red)")
plt.show()


zero_rainfall_counts = y_train[y_train["rainfall"] == 0].groupby("day")["rainfall"].count()

days_with_five_zeros = zero_rainfall_counts[zero_rainfall_counts >= 4].index.tolist()

print("Days where rainfall = 0 at least 4 times:", days_with_five_zeros)


df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col = 'id')
df_test["rainfall"] = 1
df_test.loc[df_test["day"].isin(days_with_five_zeros), "rainfall"] = 0
df_test.head(2)


sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
sub['rainfall'] = df_test['rainfall'].values
sub.to_csv('submission.csv', index = False)


zero_rainfall_counts = y_train[y_train["rainfall"] == 0].groupby("day")["rainfall"].count()

days_with_five_zeros = zero_rainfall_counts[zero_rainfall_counts >= 5].index.tolist()

print("Days where rainfall = 0 at least 5 times:", days_with_five_zeros)


df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col = 'id')
df_test["rainfall"] = 1
df_test.loc[df_test["day"].isin(days_with_five_zeros), "rainfall"] = 0

sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
sub['rainfall'] = df_test['rainfall'].values
sub.to_csv('submission.csv', index = False)


zero_rainfall_counts = y_train[y_train["rainfall"] == 0].groupby("day")["rainfall"].count()

days_with_five_zeros = zero_rainfall_counts[zero_rainfall_counts >= 3].index.tolist()

print("Days where rainfall = 0 at least 3 times:", days_with_five_zeros)


df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col = 'id')
df_test["rainfall"] = 1
df_test.loc[df_test["day"].isin(days_with_five_zeros), "rainfall"] = 0

sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
sub['rainfall'] = df_test['rainfall'].values
sub.to_csv('submission.csv', index = False)


zero_rainfall_counts = y_train[y_train["rainfall"] == 0].groupby("day")["rainfall"].count()

days_with_five_zeros = zero_rainfall_counts[zero_rainfall_counts >= 2].index.tolist()

print("Days where rainfall = 0 at least 2 times:", days_with_five_zeros)


df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col = 'id')
df_test["rainfall"] = 1
df_test.loc[df_test["day"].isin(days_with_five_zeros), "rainfall"] = 0

sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
sub['rainfall'] = df_test['rainfall'].values
sub.to_csv('submission.csv', index = False)


preds = y_train['rainfall'].iloc[-730:]
sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
sub['rainfall'] = preds.values
sub.to_csv('submission.csv', index = False)


test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
condition = (test.cloud > 73.5) & (test.sunshine < 0.5) & (test.pressure<=1020.35)

test['pred'] = condition.astype(int)
print(sum(test.pred))


sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
sub['rainfall'] = test.pred
sub.to_csv('submission.csv', index = False)

