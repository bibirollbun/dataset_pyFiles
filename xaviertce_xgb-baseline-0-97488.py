import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,StratifiedKFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score
print("Using XGBoost version",xgb.__version__)


train = pd.read_csv(r'/kaggle/input/playground-series-s5e7/train.csv')


test = pd.read_csv(r'/kaggle/input/playground-series-s5e7/test.csv')


train.head()


train['Personality'] = train['Personality'].map({'Extrovert':0,'Introvert':1})


train.info()


FEATURES = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']


cat_col = ['Stage_fear', 'Drained_after_socializing']
for col in cat_col:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")





%%time
FOLDS = 5
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=1)
val_scores = []
train_scores = []
oof = np.zeros(len(train))
pred = np.zeros(len(test))
for i, (train_index, test_index) in enumerate(kf.split(train,train['Personality'].values)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"Personality"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"Personality"]
    x_test = test[FEATURES].copy()

    xgb_params = {
                  'n_estimators': 2500, 
                  'eta': 0.017579962549289938, 
                  'alpha': 2.321837605269201, 
                  'subsample': 0.7882697285800268,
                  'colsample_bytree': 0.9490715823210952, 
                  'max_depth': 12, 
                  'min_child_weight': 6, 
                  'gamma': 1.1312486129566937, 
                  'max_bin': 78406,
                  'device': 'cuda',
                  'eval_metric': 'auc',
                  'random_state' : 42,
                  'enable_categorical':True
                 }
    model_xgb = XGBClassifier(**xgb_params)
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=500 
    )

    # INFER OOF
    val_preds_proba = model_xgb.predict_proba(x_valid)[:, 1]
    # INFER TEST
    train_preds_proba = model_xgb.predict_proba(x_train)[:, 1]
    pred += model_xgb.predict_proba(test[FEATURES])[:, 1]

    val_scores.append(roc_auc_score(y_valid, val_preds_proba))
    train_scores.append(roc_auc_score(y_train, train_preds_proba))
    print(f'Fold {i}: train_scores - {train_scores[-1]:.5f} val_scores - {val_scores[-1]:.5f}')

pred /= FOLDS





sub = pd.read_csv(r'/kaggle/input/playground-series-s5e7/sample_submission.csv')


sub['Personality'] = np.where(pred >= 0.50, 1, 0)


sub['Personality'] = sub['Personality'].map({0:'Extrovert',1:'Introvert'})


sub.to_csv(r'sub_xgb_97488.csv',index=False)







