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


# ğŸ“¦ Install Required Libraries
# !pip install pandas numpy scikit-learn xgboost optuna matplotlib seaborn

import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# ğŸ“� File Paths
DATA_DIR = '/kaggle/input/playground-series-s5e7/'  # or your local path
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# ğŸ§¹ Drop 'Personality' from test if exists
test_df = test_df.drop(columns='Personality', errors='ignore')

# Save IDs
train_id = train_df['id']
test_id = test_df['id']



# ğŸ§ª Add constant feature
train_df['constant_zero_feature'] = 0
test_df['constant_zero_feature'] = 0

# ğŸ§¼ Numerical Imputation
numerical_cols = train_df.select_dtypes(include=np.number).columns.difference(['id', 'Personality', 'constant_zero_feature'])
imputer = IterativeImputer(random_state=42)

train_df[numerical_cols] = imputer.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = imputer.transform(test_df[numerical_cols])

# ğŸ§¼ Categorical Imputation & One-Hot Encoding
cat_cols = train_df.select_dtypes(include='object').columns.difference(['Personality'])

for col in cat_cols:
    train_df[col] = train_df[col].fillna('Missing')
    test_df[col] = test_df[col].fillna('Missing')

train_df = pd.get_dummies(train_df, columns=cat_cols, drop_first=False)
test_df = pd.get_dummies(test_df, columns=cat_cols, drop_first=False)

# ğŸ§± Align Features
common_cols = sorted(set(train_df.columns) | set(test_df.columns))
common_cols = [col for col in common_cols if col not in ['id', 'Personality']]

X = train_df[common_cols].reindex(columns=common_cols, fill_value=0)
X_test = test_df[common_cols].reindex(columns=common_cols, fill_value=0)
y = train_df['Personality']



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Label Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))



import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import xgboost as xgb

def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.01),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5)
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    accs = []

    for train_idx, val_idx in cv.split(X, y_encoded):
        model = xgb.XGBClassifier(n_estimators=1000, **params)
        model.fit(X.iloc[train_idx], y_encoded[train_idx],
                  eval_set=[(X.iloc[val_idx], y_encoded[val_idx])],
                  early_stopping_rounds=50, verbose=False)
        preds = model.predict(X.iloc[val_idx])
        accs.append(accuracy_score(y_encoded[val_idx], preds))

    return np.mean(accs)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print("Best Score:", study.best_value)
print("Best Params:", study.best_params)



final_xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'gpu_hist',               # Use 'hist' if you don't have GPU
    'predictor': 'gpu_predictor',
    'random_state': 42,
    'n_estimators': 5000,                    # High value + early stopping
    'learning_rate': 0.007233211266709561,
    'max_depth': 5,
    'subsample': 0.7637445257868068,
    'colsample_bytree': 0.5177249032101772,
    'reg_lambda': 6.5143762971841594,
    'reg_alpha': 5.565061464579081,
    'min_child_weight': 5,
    'gamma': 4.973579122262935,

    # Additional regularizing params (optional)
    'max_delta_step': 1,
    'scale_pos_weight': 1,                   # Use class balancing if necessary
    'grow_policy': 'depthwise',
    'sampling_method': 'uniform',
    'verbosity': 0
}



kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y_encoded))
test_preds = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_encoded)):
    model = xgb.XGBClassifier(**final_xgb_params)  # âœ… Correct
    model.fit(X.iloc[train_idx], y_encoded[train_idx],
              eval_set=[(X.iloc[val_idx], y_encoded[val_idx])],
              verbose=False)

    val_preds = model.predict(X.iloc[val_idx])
    oof_preds[val_idx] = val_preds
    test_preds.append(model.predict(X_test))

    acc = accuracy_score(y_encoded[val_idx], val_preds)
    print(f"Fold {fold + 1} Accuracy: {acc:.4f}")

final_acc = accuracy_score(y_encoded, oof_preds)
print(f"\nFinal OOF Accuracy: {final_acc:.4f}")



# ğŸ“¤ Submission
final_preds = np.round(np.mean(test_preds, axis=0)).astype(int)
submission = pd.DataFrame({
    'id': test_id,
    'Personality': le.inverse_transform(final_preds)
})
submission.to_csv("submission_1.csv", index=False)
print(submission.head())

# ğŸ“Š Feature Importance
feature_importance = model.feature_importances_
imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
imp_df = imp_df.sort_values(by='Importance', ascending=False)
print(imp_df.head(10))


