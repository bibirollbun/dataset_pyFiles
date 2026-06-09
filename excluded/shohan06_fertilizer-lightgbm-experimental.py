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


# import libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import optuna

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings('ignore')


# import dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
origin = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")


# drop 'id' from train and test
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)


# concat train and origin
train = pd.concat([train, origin], axis=0, ignore_index=True)
train.info()


# numeric, categorical and target features
numerics = [col for col in train.columns if train[col].dtype in ['int64', 'float64']]
category = [col for col in train.columns if train[col].dtype in ['object']]
target = 'Fertilizer Name'

# remove target from category
category.remove(target)

print(f"Numeric Features:\t{numerics}")
print(f"Categorical Features:\t{category}")
print(f"Target:\t{target}")


# encode categorical features
def encode(df, columns):
    df_copy = df.copy()
    le = LabelEncoder()
    for col in columns:
        df_copy[col] = le.fit_transform(df_copy[col])

    return df_copy

train = encode(train, category)
test = encode(test, category)


# encode target
fertilizer_mapping = {
    '28-28': 0,
    '17-17-17': 1,
    '10-26-26': 2,
    'DAP': 3,
    '20-20': 4,
    '14-35-14': 5,
    'Urea': 6
}

train['Fertilizer Name'] = train['Fertilizer Name'].map(fertilizer_mapping)


# for MAP@3
def apk(actual, predicted, k=3):
    predicted = list(predicted)  # Convert to list for .index()
    if actual in predicted:
        return 1.0 / (predicted.index(actual) + 1)
    return 0.0

def mapk(y_true, y_pred_probs, k=3):
    top_k_preds = np.argsort(y_pred_probs, axis=1)[:, ::-1][:, :k]
    return np.mean([apk(a, p) for a, p in zip(y_true, top_k_preds)])


# prepare data for model
X = train.drop('Fertilizer Name', axis=1)
y = train['Fertilizer Name']

# KFold setup
n_splits = 30
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


# lightGBM parameters (choosen by OPTUNA in another notebook)
best_params = {
    'learning_rate': 0.09580403245370023,
    'num_leaves': 94,
    'min_child_samples': 83,
    'feature_fraction': 0.5321646008412426,
    'bagging_fraction': 0.9490495735621516,
    'bagging_freq': 7,
    'lambda_l1': 4.040585831894955,
    'lambda_l2': 0.5060511556240237,
    'objective': 'multiclass',
    'num_class': len(fertilizer_mapping),
    'boosting_type': 'gbdt',
    'metric': 'multi_logloss',
    'verbosity': -1
}

# store test preds
test_preds = np.zeros((test.shape[0], len(fertilizer_mapping)))

# KFold loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"{'-'*15} Fold: {fold+1} {'-'*15}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = LGBMClassifier(**best_params, n_estimators=2000)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)])

    val_preds = model.predict_proba(X_val)
    val_score = mapk(y_val.values, val_preds, k=3)
    print(f"Validation MAP@3:\t{val_score}")

    # test predictions
    test_preds += model.predict_proba(test) / n_splits


print(fertilizer_mapping.keys())
print(fertilizer_mapping.values())


# Convert predictions to top 3 labels per sample
id_to_label = {v: k for k, v in fertilizer_mapping.items()}
top_3_preds = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]  # top 3 in descending order
predicted_labels = [' '.join([id_to_label[idx] for idx in row]) for row in top_3_preds]


# submission
submission['Fertilizer Name'] = predicted_labels
submission.to_csv('submission.csv', index=False)
submission.head()




