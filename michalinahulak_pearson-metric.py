from scipy.stats import pearsonr

import random
import numpy as np
import pandas as pd


train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')


def pearson_metric(y_true, y_pred):
    corr, _ = pearsonr(y_true, y_pred)
    return corr


y_true = train['label'].values


preds = {
    "Perfect": y_true.copy(),
    "Shifted": y_true + 100,
    "Noisy": y_true + np.random.normal(0, 0.1 * np.std(y_true), size=len(y_true)),
    "Random": np.random.permutation(y_true),
    "Inverted": -y_true,
}

print("Pearson Correlation Scores:")
for name, pred in preds.items():
    corr, _ = pearsonr(y_true, pred)
    print(f"{name:10s}: {corr:.4f}")


sub = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')

y_min = train['label'].min()
y_max = train['label'].max()

sub['prediction'] = np.random.uniform(low=y_min, high=y_max, size=len(sub))

sub.to_csv('submission.csv', index = False)
sub

