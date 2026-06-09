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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


df_train.columns = df_train.columns.str.replace(' ', '_').str.lower()
df_test.columns = df_test.columns.str.replace(' ', '_').str.lower()


df_train.head(3)


import matplotlib.pyplot as plt

# Define the columns you want to plot
cat_cols = ['soil_type', 'crop_type']

# Initialize LabelEncoder for categorical columns
label_encoders = {col: LabelEncoder() for col in cat_cols}

# Apply LabelEncoder to each categorical column
for col in cat_cols:
    df_train[col] = label_encoders[col].fit_transform(df_train[col])
    df_test[col] = label_encoders[col].transform(df_test[col])

# Encode the target separately
target_le = LabelEncoder()
df_train['fertilizer_name'] = target_le.fit_transform(df_train['fertilizer_name'])



df_train.head(3)


X = df_train.drop(columns=['fertilizer_name'])
y = df_train['fertilizer_name']


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

for col in df_train.columns:
    # Reshape the column to 2D
    df_train[[col]] = scaler.fit_transform(df_train[[col]])



for col in df_test.columns:
    df_test[[col]] = scaler.fit_transform(df_test[[col]])


num_classes = len(target_le.classes_)
num_classes


import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer
import numpy as np

# Your MAP@3 scoring function (using make_scorer if needed)
def map3_eval(y_true, y_proba):
    top_3 = np.argsort(y_proba, axis=1)[:, -3:][:, ::-1]
    score = 0.0
    for i in range(len(y_true)):
        if y_true[i] in top_3[i]:
            rank = np.where(top_3[i] == y_true[i])[0][0]
            score += 1 / (rank + 1)
    return score / len(y_true)

# Use this in objective for Optuna
def objective(trial):
    params = {
        'objective': 'multi:softprob',
        'num_class': 7,
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear']),
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }

    model = XGBClassifier(**params, random_state=42)

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, valid_idx in skf.split(X, y):
        x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model.fit(x_train, y_train)
        proba = model.predict_proba(x_valid)
        scores.append(map3_eval(y_valid.values, proba))

    return np.mean(scores)

# Run Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best score:", study.best_value)
print("Best params:", study.best_params)



xgb_model = XGBClassifier(
    **study.best_params,
    objective='multi:softprob',
    num_class=7,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)


from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import numpy as np

# Define MAP@3
def map3(actual, predicted_proba, k=3):
    top_k_preds = np.argsort(predicted_proba, axis=1)[:, ::-1][:, :k]
    score = 0.0
    for i in range(len(actual)):
        if actual[i] in top_k_preds[i]:
            rank = np.where(top_k_preds[i] == actual[i])[0][0]
            score += 1 / (rank + 1)
    return score / len(actual)

FOLDS = 15
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
xgb_scores = []

oof = np.zeros((len(df_train), y.nunique()))
pred_prob = np.zeros((len(df_test), y.nunique()))


test = df_test[X.columns]  # Ensure test features match

for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    xgb_model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=0)

    proba = xgb_model.predict_proba(x_valid)
    oof[valid_idx] = proba
    pred_prob += xgb_model.predict_proba(test)

    score = map3(y_valid.values, proba)
    xgb_scores.append(score)

print("Mean MAP@3 score:", np.mean(xgb_scores))


submission_data = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


top_3_preds = np.argsort(pred_prob, axis=1)[:, -3:][:, ::-1]
top_3_labels = target_le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)

submission = pd.DataFrame({
    'id': submission_data.id,
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv('submission.csv', index=False)
submission.head()

