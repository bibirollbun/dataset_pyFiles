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


# 2. Imports
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


# 4. Label Encoding
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_target = LabelEncoder()

le_soil.fit(pd.concat([train['Soil Type'], test['Soil Type']]))
le_crop.fit(pd.concat([train['Crop Type'], test['Crop Type']]))
le_target.fit(train['Fertilizer Name'])

train['Soil Type Enc'] = le_soil.transform(train['Soil Type'])
train['Crop Type Enc'] = le_crop.transform(train['Crop Type'])
train['Fertilizer Name Enc'] = le_target.transform(train['Fertilizer Name'])
test['Soil Type Enc'] = le_soil.transform(test['Soil Type'])
test['Crop Type Enc'] = le_crop.transform(test['Crop Type'])

feature_cols = [
    'Soil Type Enc', 'Crop Type Enc',
    'Temparature', 'Humidity', 'Moisture',
    'Nitrogen', 'Potassium', 'Phosphorous'
]
X = train[feature_cols]
y = train['Fertilizer Name Enc'].values
X_test = test[feature_cols]


# 5. Optuna Objective Function (with correct callbacks)
def objective(trial):
    params = {
        'objective': 'multiclass',
        'num_class': len(le_target.classes_),
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 16, 64),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 5),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 5),
        'verbosity': -1,
        'n_jobs': -1,
        'seed': 42
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    accs = []
    for train_idx, valid_idx in cv.split(X, y):
        dtrain = lgb.Dataset(X.iloc[train_idx], label=y[train_idx])
        dvalid = lgb.Dataset(X.iloc[valid_idx], label=y[valid_idx])
        model = lgb.train(
        params, dtrain, valid_sets=[dvalid],
        num_boost_round=100,
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        preds = np.argmax(model.predict(X.iloc[valid_idx]), axis=1)
        accs.append(accuracy_score(y[valid_idx], preds))
    return np.mean(accs)


# 6. Run Optuna Study (adjust n_trials if you want more tuning)
print("ðŸ”Ž Running Optuna hyperparameter search (15 trials, can increase)...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15)
print("Best trial:", study.best_trial.value, study.best_trial.params)


# 7. Train final LightGBM model with best parameters
params = study.best_trial.params
params.update({
    'objective': 'multiclass',
    'num_class': len(le_target.classes_),
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'n_jobs': -1,
    'seed': 42
})
lgb_train = lgb.Dataset(X, label=y)
final_model = lgb.train(params, lgb_train, num_boost_round=200, callbacks=[lgb.log_evaluation(20)])


# 8 OOF Cross-Validation Accuracy (for leaderboard estimate)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X), dtype=int)
for train_idx, valid_idx in cv.split(X, y):
    lgb_tr = lgb.Dataset(X.iloc[train_idx], label=y[train_idx])
    lgb_va = lgb.Dataset(X.iloc[valid_idx], label=y[valid_idx])
    model = lgb.train(
        params, lgb_tr, num_boost_round=final_model.best_iteration or 200,
        callbacks=[lgb.log_evaluation(0)]
    )
    preds = np.argmax(model.predict(X.iloc[valid_idx]), axis=1)
    oof_preds[valid_idx] = preds
cv_acc = accuracy_score(y, oof_preds)
print(f"\nLightGBM OOF CV Accuracy: {cv_acc:.4f}")
print(classification_report(y, oof_preds, target_names=le_target.classes_))


# 9. Predict test set and make submission
test_pred = np.argmax(final_model.predict(X_test), axis=1)
test_pred_label = le_target.inverse_transform(test_pred)
sample_submission['Fertilizer Name'] = test_pred_label
sample_submission.to_csv('/kaggle/working/submission_lgb_optuna.csv', index=False)
print("âœ… submission_lgb_optuna.csv created!")




