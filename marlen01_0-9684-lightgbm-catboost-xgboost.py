import numpy as np 
import pandas as pd 
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import  StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
train_df


X = train_df.drop(columns=["id", "y"])
y = train_df["y"]
X_test = test_df.drop(columns=["id"])


ts_tr = pd.concat((X,X_test)).reset_index(drop = True)
#one-hot encoding
ts_tr = pd.get_dummies(ts_tr, columns = ['job','marital','education','default','housing','loan','contact','month','poutcome'])


ts_tr.columns


ts_tr.info()


# после one-hot
X = ts_tr.iloc[:len(train_df), :]
X_test = ts_tr.iloc[len(train_df):, :]



#Cross validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_scores = []
test_preds = np.zeros(len(X_test))
for fold, (train_idx, val_idx) in enumerate(kf.split(X,y)):
    print(f"Fold {fold + 1}")
    x_train,x_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train,y_val = y.iloc[train_idx], y.iloc[val_idx]
   #define models
    xgb = XGBClassifier(n_estimators=1000, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
    lgb = LGBMClassifier(n_estimators = 1000,learning_rate=0.1)
    cat = CatBoostClassifier(n_estimators=1000,learning_rate = 0.1,verbose=0)
    xgb.fit(x_train,y_train)
    lgb.fit(x_train,y_train)
    cat.fit(x_train,y_train)
    val_preds = (
        xgb.predict_proba(x_val)[:,1] +
        lgb.predict_proba(x_val)[:,1] +
        cat.predict_proba(x_val)[:,1]     
    ) / 3



#Metric evaluation
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    auc = roc_auc_score(y_val,val_preds)
    val_scores.append(auc)
    print(f"ROC-AUC for fold {fold + 1}: {auc:.4f}")


for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    test_preds += (
    xgb.predict_proba(X_test)[:,1] +
    lgb.predict_proba(X_test)[:,1] +
    cat.predict_proba(X_test)[:,1]
) / kf.n_splits
print(f"\nMean ROC-AUC on folds: {np.mean(val_scores):.4f}")


#save submission
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
sample_submission['y'] = test_preds
sample_submission.to_csv("cv_ensemble_submission.csv",index=False)

