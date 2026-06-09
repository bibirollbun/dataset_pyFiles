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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


SEED = 42
N_SPLITS = 5
features = [ 'annual_income','debt_to_income_ratio','credit_score', 'loan_amount', 
            'interest_rate', 'employment_status','loan_purpose']


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')

test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

sample = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


y = train['loan_paid_back']
X = train[features].copy()
X_test = test[features].copy()


num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in features if c not in num_cols]


for c in num_cols:
    med = X[c].median()
    X[c].fillna(med, inplace=True)
    X_test[c].fillna(med, inplace=True)

for c in cat_cols:
    X[c] = X[c].fillna('Missing').astype(str)
    X_test[c] = X_test[c].fillna('Missing').astype(str)


if y.dtype == object or y.dtype == 'bool':
    le = LabelEncoder()
    y = le.fit_transform(y)


from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
enc.fit(pd.concat([X[cat_cols], X_test[cat_cols]], axis=0))
X_enc = X.copy()
X_test_enc = X_test.copy()
X_enc[cat_cols] = enc.transform(X[cat_cols])
X_test_enc[cat_cols] = enc.transform(X_test[cat_cols])


LGB_PARAMS = {
    'n_estimators': 2000,
    'learning_rate': 0.03,
    'num_leaves': 31,
    'colsample_bytree': 0.7,
    'subsample': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': SEED,
    'n_jobs': -1,
    'importance_type': 'gain'
}

CAT_PARAMS = {
    'iterations': 2000,
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 3,
    'bagging_temperature': 0.2,
    'random_state': SEED,
    'eval_metric': 'AUC',
    'verbose': 200,
    'early_stopping_rounds': 100
}

RF_PARAMS = {
    'n_estimators': 600,
    'max_depth': None,
    'n_jobs': -1,
    'random_state': SEED,
    'class_weight': None  
}


oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_rf  = np.zeros(len(X))

test_lgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))
test_rf  = np.zeros(len(X_test))

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_enc, y), 1):
    X_tr, X_val = X_enc.iloc[tr_idx], X_enc.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model_lgb = lgb.LGBMClassifier(**LGB_PARAMS)
    model_lgb.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  eval_metric='auc')
                  #early_stopping_rounds=100,
                  #verbose=False)
    oof_lgb[val_idx] = model_lgb.predict_proba(X_val)[:,1]
    test_lgb += model_lgb.predict_proba(X_test_enc)[:,1] / N_SPLITS

    X_tr_cat = X.iloc[tr_idx][features]    
    X_val_cat = X.iloc[val_idx][features]
    model_cat = CatBoostClassifier(**CAT_PARAMS)
    train_pool = Pool(X_tr_cat, y_tr, cat_features=cat_cols)
    val_pool = Pool(X_val_cat, y_val, cat_features=cat_cols)
    model_cat.fit(train_pool, eval_set=val_pool, use_best_model=True)
    oof_cat[val_idx] = model_cat.predict_proba(X_val_cat)[:,1]
    test_cat += model_cat.predict_proba(X_test[features])[:,1] / N_SPLITS

   
    model_rf = RandomForestClassifier(**RF_PARAMS)
    model_rf.fit(X_tr, y_tr)
    oof_rf[val_idx] = model_rf.predict_proba(X_val)[:,1]
    test_rf += model_rf.predict_proba(X_test_enc)[:,1] / N_SPLITS

    
    print(f"Fold {fold} -> LGB AUC: {roc_auc_score(y_val, oof_lgb[val_idx]):.5f}, "
          f"Cat AUC: {roc_auc_score(y_val, oof_cat[val_idx]):.5f}, "
          f"RF AUC: {roc_auc_score(y_val, oof_rf[val_idx]):.5f}")


print("Overall LGB OOF AUC:", roc_auc_score(y, oof_lgb))
print("Overall Cat OOF AUC:", roc_auc_score(y, oof_cat))
print("Overall RF OOF AUC:", roc_auc_score(y, oof_rf))


w_lgb = 0.3
w_cat = 0.5
w_rf  = 0.2

final_test_preds = (w_lgb * test_lgb) + (w_cat * test_cat) + (w_rf * test_rf)


submission = pd.DataFrame({
    'id': test['id'],
    'loan_paid_back': final_test_preds
})

submission.to_csv("submission.csv",index=False)
print("Submission made successfully!!")

