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
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv').set_index('id')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv').set_index("id")
orig_data = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")

# Rename temperature column
def rename_temperature_column(df):
    df = df.rename(columns={'Temparature': 'Temperature'})
    return df

train = rename_temperature_column(train)
test = rename_temperature_column(test)
orig_data = rename_temperature_column(orig_data)

# Merge datasets
train = pd.concat([train, orig_data], ignore_index=True)

# Feature Encoding
cat_cols = train.select_dtypes(include=['object', 'category']).columns.drop('Fertilizer Name')
oe = OrdinalEncoder()
train[cat_cols] = oe.fit_transform(train[cat_cols])
test[cat_cols] = oe.transform(test[cat_cols])

# Target Encoding
le = LabelEncoder()
train['Fertilizer Name'] = le.fit_transform(train['Fertilizer Name'])

# Optimize dtypes
for df in [train, test]:
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int16')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float16')

# Prepare data
X = train.drop('Fertilizer Name', axis=1)
y = train['Fertilizer Name']
X_test = test

# Class weights
class_weights = np.bincount(y)
class_weights = class_weights.max() / class_weights

# MAP@3 metric
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

# Pseudo-labeling
initial_model = XGBClassifier(
    max_depth=17, colsample_bytree=0.467, subsample=0.86, n_estimators=1000,
    learning_rate=0.03, gamma=0.26, max_delta_step=4, reg_alpha=2.7,
    reg_lambda=1.4, early_stopping_rounds=50, objective='multi:softprob',
    random_state=13, enable_categorical=True, tree_method='hist', device='cuda'
)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
initial_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
test_probs = initial_model.predict_proba(X_test)
conf_threshold = 0.85
conf_mask = test_probs.max(axis=1) > conf_threshold
X_pseudo = X_test[conf_mask].copy()
y_pseudo = np.argmax(test_probs[conf_mask], axis=1)
X = pd.concat([X, X_pseudo], ignore_index=True)
y = pd.concat([pd.Series(y), pd.Series(y_pseudo)], ignore_index=True)
print(f"Added {len(X_pseudo)} pseudo-labeled samples")

# XGBoost with class weights
FOLDS = 10
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
oof = np.zeros((len(X), y.nunique()))
pred_prob = np.zeros((len(X_test), y.nunique()))

xgb_model = XGBClassifier(
    max_depth=17, colsample_bytree=0.467, subsample=0.86, n_estimators=1000,
    learning_rate=0.03, gamma=0.26, max_delta_step=4, reg_alpha=2.7,
    reg_lambda=1.4, early_stopping_rounds=50, objective='multi:softprob',
    random_state=13, enable_categorical=True, tree_method='hist', device='cuda',
    scale_pos_weight=class_weights  # Apply class weights
)

for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {i+1}")
    x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    xgb_model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=0)
    oof[valid_idx] = xgb_model.predict_proba(x_valid)
    pred_prob += xgb_model.predict_proba(X_test) / FOLDS
    
    top_3_preds = np.argsort(oof[valid_idx], axis=1)[:, -3:][:, ::-1]
    map3_score = mapk([[label] for label in y_valid], top_3_preds)
    print(f"Fold {i+1} MAP@3: {map3_score:.5f}")

# Submission
top_k_indices = np.argsort(pred_prob, axis=1)[:, -3:][:, ::-1]
top_k_labels = le.inverse_transform(top_k_indices.ravel()).reshape(top_k_indices.shape)
submission = pd.DataFrame({
    'id': pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")['id'],
    'Fertilizer Name': [' '.join(row) for row in top_k_labels]
})
submission.to_csv('submission_xgb.csv', index=False)
print("Submission saved: submission_xgb.csv")

