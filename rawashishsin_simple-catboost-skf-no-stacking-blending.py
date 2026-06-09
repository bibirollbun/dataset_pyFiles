import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


target = 'diagnosed_diabetes'
X_train = train.drop(target, axis=1)
y_train = train[target]
cat_columns = X_train.select_dtypes(include='object').columns.to_list()


print(X_train.shape)
print(y_train.shape)
print(cat_columns)
display(X_train.head())


X_test = test


print(X_train.columns)
print(X_test.columns)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\n===== Fold {fold+1} =====")

    X_train_v, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_v, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        eval_metric="AUC",
        random_seed=42,
        verbose=100,
        thread_count=-1
    )

    model.fit(
        X_train_v, y_train_v,
        cat_features=cat_columns,
        eval_set=(X_val, y_val),
        early_stopping_rounds=150,
        use_best_model=True
    )

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits



roc_auc_score(y_train, oof_preds)


# submission
submission = pd.DataFrame({
    "id": X_test["id"],
    "diagnosed_diabetes": test_preds
})

submission.to_csv("CatBoostskf.csv", index=False)

