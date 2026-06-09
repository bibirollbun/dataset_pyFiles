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
    return df.mean(axis=0).values

X_train = []
y_train = []
train_dir = Path('/kaggle/input/iaio-2026-sfr-timeseries-classification/xtrain/xtrain')

for idx, row in tqdm(ytrain.iterrows(), total=len(ytrain)):
    file_path = train_dir / f"{row['Id']}.csv"
    if file_path.exists():
        X_train.append(extract_mean_features(file_path))
        y_train.append(row['Attack'])

X_train = np.array(X_train)
y_train = np.array(y_train)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
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




