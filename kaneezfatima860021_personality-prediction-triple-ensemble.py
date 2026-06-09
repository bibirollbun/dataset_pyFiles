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


import os, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
import warnings; warnings.filterwarnings('ignore')

DATA_PATH = "/kaggle/input/playground-series-s5e7"
train = pd.read_csv(f"{DATA_PATH}/train.csv")
test  = pd.read_csv(f"{DATA_PATH}/test.csv")
print(train.shape, test.shape)


train.head()
print(train['Personality'].value_counts())


# Separate numeric / categorical
num_cols = train.select_dtypes(include=['int64','float64']).columns
cat_cols = train.select_dtypes('object').drop(['Personality'], axis=1).columns.tolist()


# Median impute numerics
medians = train[num_cols].median()
train[num_cols] = train[num_cols].fillna(medians)
test[num_cols]  = test[num_cols].fillna(medians)


# Mode impute categoricals
for col in cat_cols:
    mode_val = train[col].mode()[0]
    train[col].fillna(mode_val, inplace=True)
    test[col].fillna(mode_val,  inplace=True)


# Encode target
le_target = LabelEncoder(); train['y'] = le_target.fit_transform(train['Personality']); y = train['y']



# Labelâ€‘encode cats for XGB & LGBM
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))

X      = train.drop(columns=['id','Personality','y'])
X_test = test.drop(columns=['id'])
cat_idx = [X.columns.get_loc(c) for c in cat_cols]


SEED, N_FOLDS = 42, 10
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Hyperâ€‘params
cat_params = dict(iterations=2500, learning_rate=0.025, depth=8, l2_leaf_reg=4,
                  eval_metric='Accuracy', loss_function='MultiClass', random_state=SEED,
                  early_stopping_rounds=200, verbose=False)

xgb_params = dict(n_estimators=2000, learning_rate=0.02, max_depth=10, subsample=0.9,
                  colsample_bytree=0.9, objective='multi:softprob', num_class=len(np.unique(y)),
                  eval_metric='mlogloss', tree_method='hist', random_state=SEED)

lgb_params = dict(objective='multiclass', num_class=len(np.unique(y)), metric='multi_logloss',
                  learning_rate=0.02, n_estimators=3000, max_depth=-1, subsample=0.9,
                  colsample_bytree=0.9, random_state=SEED)


# Placeholders
cat_oof = np.zeros((len(train), len(np.unique(y)))); cat_prob = np.zeros((len(test), len(np.unique(y))))
xgb_oof = cat_oof.copy(); xgb_prob = cat_prob.copy()
lgb_oof = cat_oof.copy(); lgb_prob = cat_prob.copy()


for fold,(tr,val) in enumerate(skf.split(X,y),1):
    print(f"Fold {fold}")
    # CatBoost
    cat = CatBoostClassifier(**cat_params)
    cat.fit(X.iloc[tr],y.iloc[tr], eval_set=(X.iloc[val],y.iloc[val]), cat_features=cat_idx, use_best_model=True)
    cat_oof[val] = cat.predict_proba(X.iloc[val]); cat_prob += cat.predict_proba(X_test)/N_FOLDS
    del cat; gc.collect()
    # XGBoost
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X.iloc[tr],y.iloc[tr], eval_set=[(X.iloc[val],y.iloc[val])], early_stopping_rounds=200, verbose=False)
    xgb_oof[val] = xgb.predict_proba(X.iloc[val]); xgb_prob += xgb.predict_proba(X_test)/N_FOLDS
    del xgb; gc.collect()

    


# LightGBM
lgb_train = lgb.Dataset(X.iloc[tr], label=y.iloc[tr])
lgb_val   = lgb.Dataset(X.iloc[val], label=y.iloc[val])

lgbm = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=3000,
    valid_sets=[lgb_val],
    callbacks=[
        lgb.early_stopping(200),
        lgb.log_evaluation(False)
    ]
)

lgb_oof[val] = lgbm.predict(X.iloc[val])
lgb_prob    += lgbm.predict(X_test) / N_FOLDS
del lgbm; gc.collect()



# Soft blend weights
w_cat, w_xgb, w_lgb = 0.5, 0.3, 0.2

blended_oof = w_cat*cat_oof + w_xgb*xgb_oof + w_lgb*lgb_oof
blended_pred = np.argmax(blended_oof, axis=1)
print('OOF Acc:', accuracy_score(y, blended_pred).round(4))
print(classification_report(y, blended_pred, target_names=le_target.classes_))


blended_test = w_cat*cat_prob + w_xgb*xgb_prob + w_lgb*lgb_prob
sub_pred = np.argmax(blended_test, axis=1)
submission = pd.DataFrame({'id': test['id'], 'Personality': le_target.inverse_transform(sub_pred)})
submission.to_csv('submission.csv', index=False); print('submission.csv saved')




