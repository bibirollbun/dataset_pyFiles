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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool


SEED = 42
N_SPLITS = 5

features = [
    'age','alcohol_consumption_per_week','physical_activity_minutes_per_week',
    'diet_score','sleep_hours_per_day','screen_time_hours_per_day','bmi',
    'waist_to_hip_ratio','systolic_bp','diastolic_bp','heart_rate',
    'cholesterol_total','hdl_cholesterol','ldl_cholesterol','triglycerides',
    'gender','smoking_status','family_history_diabetes',
    'hypertension_history','cardiovascular_history'
]

train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

y = train['diagnosed_diabetes'].values
X = train[features].copy()
X_test = test[features].copy()

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in features if c not in num_cols]


zero_cols = [
    'bmi','systolic_bp','diastolic_bp','heart_rate',
    'cholesterol_total','hdl_cholesterol','ldl_cholesterol','triglycerides'
]

for c in zero_cols:
    X[c] = X[c].replace(0, np.nan)
    X_test[c] = X_test[c].replace(0, np.nan)


for c in num_cols:
    med = X[c].median()
    X[c].fillna(med, inplace=True)
    X_test[c].fillna(med, inplace=True)

for c in cat_cols:
    X[c] = X[c].fillna('Missing').astype(str)
    X_test[c] = X_test[c].fillna('Missing').astype(str)



enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
enc.fit(pd.concat([X[cat_cols], X_test[cat_cols]]))

X_enc = X.copy()
X_test_enc = X_test.copy()
X_enc[cat_cols] = enc.transform(X[cat_cols])
X_test_enc[cat_cols] = enc.transform(X_test[cat_cols])


LGB_PARAMS = {
    'n_estimators': 700,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': True,
    'random_state': SEED,
    'n_jobs': -1
}


CAT_PARAMS = {
    'iterations': 700,
    'learning_rate': 0.05,
    'depth': 6,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': SEED,
    'verbose': 0
}


oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

test_lgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (tr, val) in enumerate(skf.split(X_enc, y), 1):
    X_tr, X_val = X_enc.iloc[tr], X_enc.iloc[val]
    y_tr, y_val = y[tr], y[val]

    lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
    lgb_model.fit(X_tr, y_tr)
    oof_lgb[val] = lgb_model.predict_proba(X_val)[:,1]
    test_lgb += lgb_model.predict_proba(X_test_enc)[:,1] / N_SPLITS

    train_pool = Pool(X.iloc[tr], y_tr, cat_features=cat_cols)
    val_pool = Pool(X.iloc[val], y_val, cat_features=cat_cols)

    cat_model = CatBoostClassifier(**CAT_PARAMS)
    cat_model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    oof_cat[val] = cat_model.predict_proba(X.iloc[val])[:,1]
    test_cat += cat_model.predict_proba(X_test)[:,1] / N_SPLITS

    print(f"Fold {fold} | LGB AUC: {roc_auc_score(y_val, oof_lgb[val]):.5f} "
          f"| CAT AUC: {roc_auc_score(y_val, oof_cat[val]):.5f}")


print("OOF LGB AUC:", roc_auc_score(y, oof_lgb))
print("OOF CAT AUC:", roc_auc_score(y, oof_cat))

final_test_preds = (0.6 * test_cat) + (0.4 * test_lgb)

submission = pd.DataFrame({
    'id': test['id'],
    'loan_paid_back': final_test_preds
})

submission.to_csv('submission.csv', index=False)
print("Submission saved")

