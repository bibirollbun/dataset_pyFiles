import pandas as pd
import numpy as np
from catboost import CatBoostClassifier,Pool
from sklearn.model_selection import StratifiedKFold,cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score


train= pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sample=pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")


TARGET = 'diagnosed_diabetes'
test_id = test['id'].copy()


X = train.drop(TARGET, axis=1)
y = train[TARGET]
test_X = test.copy()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


cat_params = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 8000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 6,
    "random_strength": 1.0,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.8,
    "min_data_in_leaf": 50,
    "od_type": "Iter",
    "od_wait": 300,
    "random_seed": 42,
    "verbose": 0,
    'task_type':"GPU"
}


oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test_X))

fold_auc = []


N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)


cat_cols= train.select_dtypes(include='object').columns.to_list()
for fold, (train_idx, val_idx) in enumerate(skf.split(X,y)):
    print(f"\nFold {fold + 1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_pool = Pool(X_train, y_train,cat_features=cat_cols)
    val_pool = Pool(X_val, y_val,cat_features=cat_cols)

    model = CatBoostClassifier(**cat_params)
    model.fit(
        train_pool,
        eval_set = val_pool,
        use_best_model = True
        
    )

    val_pred = model.predict_proba(X_val)[:,1]
    oof_preds[val_idx] = val_pred

    auc = roc_auc_score(y_val, val_pred)
    fold_auc.append(auc)
    print(f"AUC: {auc:.5f}")


y_pred=model.predict_proba(test)[:,1]


sample["diagnosed_diabetes"]=y_pred
sample.to_csv("submission.csv",index= False)




