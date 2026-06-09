# Baseline for those who want to get a Shengen visa
# in a very unusual way :)
# Good luck!
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import warnings
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')

ytrain = pd.read_csv('/kaggle/input/iaio-2026-sfr-timeseries-classification/ytrain.csv')

def extract_mean_features(file_path):
    df = pd.read_csv(file_path, header=None)
    features = []
    features.extend(df.mean(axis=0).values)
    features.extend(df.std(axis=0).values)
    features.extend(df.max(axis=0).values)
    features.extend(df.min(axis=0).values)
    features.extend(df.median(axis=0).values)
    features.extend(df.quantile(0.25, axis=0).values)
    features.extend(df.quantile(0.75, axis=0).values)
    features.extend(df.sem(axis=0).values)
    features.extend(df.skew(axis=0).values)
    features.extend(df.kurt(axis=0).values)
    features.extend((df**2).mean(axis=0).values)
    features.extend((df**3).mean(axis=0).values)
    features.extend((df**4).mean(axis=0).values)
#    features.extend((df**3).mean(axis=0).values)
    features.extend(df.diff().mean(axis=0).values)
#   features.extend(df.mad(axis=0).values)
    features.extend(df.quantile(0.05, axis=0).values)
    features.extend(df.quantile(0.10, axis=0).values)
    features.extend(df.quantile(0.90, axis=0).values)
    features.extend(df.quantile(0.95, axis=0).values)
    features.extend(df.quantile(0.25, axis=0).values)
    features.extend(df.quantile(0.20, axis=0).values)
    features.extend(df.quantile(0.80, axis=0).values)
    features.extend((df > df.mean()).mean(axis=0).values)
    features.extend(df.var(axis=0).values)
    features.extend(np.sqrt(df.var(axis=0)).values)
    features.extend((df.max() - df.min()).values)
#    features.extend((df.quantile(0.95) - df.quantile(0.25)).values)
#    features.extend((df.quantile(0.95) - df.quantile(0.10)).values)
#   features.extend((df.quantile(0.95) - df.quantile(0.05)).values)
#    features.extend((df.quantile(0.75) - df.quantile(0.25)).values)
#   features.extend((df.quantile(0.75) - df.quantile(0.10)).values)
#    features.extend((df.quantile(0.75) - df.quantile(0.05)).values)
#    features.extend((df.quantile(0.75) - df.quantile(0.25)).values)
    features.extend(np.mean(np.diff(df.values, axis=0), axis=0))
    features.extend(np.std(np.diff(df.values, axis=0), axis=0))
    features.extend(((df - df.mean())**2).mean(axis=0).values)
    features.extend(((df - df.mean())**3).mean(axis=0).values)
    features.extend(((df - df.mean())**4).mean(axis=0).values)
    features.extend(((df - df.mean())**5).mean(axis=0).values)
    features.extend(np.max(np.abs(df.values), axis=0))
    features.extend(np.log1p(np.abs(df.values)).mean(axis=0))
    features.extend(np.mean(np.gradient(df.values, axis=0), axis=0))
    features.extend(np.ptp(df.values, axis=0))
 #   features.extend(np.log1p(np.abs(df.diff().std(axis=0).values)))
 #   features.extend(np.log1p(np.abs(df.diff().mean(axis=0).values)))
 #   features.extend(np.log1p(df.var(axis=0)).values)
 #   features.extend(np.log1p(np.sqrt(df.var(axis=0)).values))
 # You can generate more feactures, max score is ~0.99 avaliable to get. Any feature can upgrade your score :)
    return np.array(features)

#X_train.append(extract_mean_features(file_path))
X_train = []
y_train = []
train_dir = Path('/kaggle/input/iaio-2026-sfr-timeseries-classification/xtrain/xtrain')

for idx, row in tqdm(ytrain.iterrows(), total=len(ytrain)):
    file_path = train_dir / f"{row['Id']}.csv"
    if file_path.exists():
        X_train.append(extract_mean_features(file_path))
        y_train.append(row['Attack'])

X_train.append(extract_mean_features(file_path))
y_train.append(row['Attack'])
X_train = np.array(X_train)
y_train = np.array(y_train)
#min_samples_split=5,min_samples_leaf=2 0.79
rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

sample_submission = pd.read_csv('/kaggle/input/iaio-2026-sfr-timeseries-classification/sample_submission.csv')
test_ids = sample_submission['Id'].values
X_test = []
test_dir = Path('/kaggle/input/iaio-2026-sfr-timeseries-classification/xtest/xtest')

for test_id in tqdm(test_ids):
    file_path = test_dir / f"{test_id}.csv"
    if file_path.exists():
        X_test.append(extract_mean_features(file_path))
    else:
        X_test.append(np.zeros(X_train.shape[1]))

X_test = np.array(X_test)

predictions = rf_model.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({'Id': test_ids, 'Attack': predictions})
submission.to_csv('submission.csv', index=False)

