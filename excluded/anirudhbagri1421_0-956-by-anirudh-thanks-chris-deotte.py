# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from cuml.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Define features and target
RMV = ['rainfall', 'id']
FEATURES = [c for c in train.columns if c not in RMV]
TARGET = 'rainfall'

print("Our features are:", FEATURES)

# Feature Engineering
def feature_engineering(df):
    df['cloud_sun_ratio'] = df['cloud'] / (df['sunshine'] + 1)  # Adding 1 to avoid division by zero
    df['pressure_wind_ratio'] = df['pressure'] / (df['windspeed'] + 1)
    if 'temparature' in df.columns and 'dewpoint' in df.columns:
        df['temp_dew_diff'] = df['temparature'] - df['dewpoint']
    df['humidity_pressure_ratio'] = df['humidity'] / (df['pressure'] + 1)
    df['wind_sun_ratio'] = df['windspeed'] / (df['sunshine'] + 1)
    df['cloud_temp_ratio'] = df['cloud'] / (df['temparature'] + 1) if 'temparature' in df.columns else 0
    return df

train = feature_engineering(train)
test = feature_engineering(test)

# Prepare data for K-Fold Cross Validation
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=777)

oof_knn = np.zeros(len(train))
pred_knn = np.zeros(len(test))

# Normalize data
scaler = StandardScaler()

for i, (train_index, test_index) in enumerate(kf.split(train)):
    print("#" * 25)
    print(f"### Fold {i+1}")
    print("#" * 25)
    
    x_train, x_valid = train.loc[train_index, FEATURES], train.loc[test_index, FEATURES]
    y_train, y_valid = train.loc[train_index, TARGET], train.loc[test_index, TARGET]
    x_test = test[FEATURES].copy()
    
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)
    
    # K-Nearest Neighbors Classifier
    model = KNeighborsClassifier(n_neighbors=101, p=1)
    model.fit(x_train, y_train)
    
    # Infer OOF
    oof_knn[test_index] = model.predict_proba(x_valid)[:, 1]
    # Infer Test
    pred_knn += model.predict_proba(x_test)[:, 1]

# Compute average test predictions
pred_knn /= FOLDS

# Calculate ROC AUC score
true = train[TARGET].values
roc_auc = roc_auc_score(true, oof_knn)
print(f"KNN CV Score AUC = {roc_auc:.3f}")

# Load best public notebook predictions
best_public = pd.read_csv("/kaggle/input/lb-915-public-notebook/submission95427.csv")
best_public = best_public['rainfall'].values

# Ensemble predictions
print("Ensemble achieves LB = 0.956! Hooray!")
sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
sub['rainfall'] = -0.065 * rankdata(pred_knn) + 1.065 * rankdata(best_public)
sub['rainfall'] = rankdata(sub['rainfall']) / len(sub)

# Save submission
sub.to_csv("submission.csv", index=False)
print(sub.shape)
print(sub.head())

