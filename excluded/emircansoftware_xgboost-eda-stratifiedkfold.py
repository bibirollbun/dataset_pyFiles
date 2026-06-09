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


train=pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


train.head()


train.info()


cat_cols=train.select_dtypes(include="object").columns.tolist()


num_cols=train.select_dtypes(include=np.number).columns.tolist()


num_cols.remove("y")


for i in cat_cols:
    print(train[i].value_counts())


import seaborn as sns
import matplotlib.pyplot as plt

for i in num_cols:
    plt.figure()
    sns.boxplot(x=train[i])
    plt.show()


train=train[train["previous"]<150]


train.info()


train.drop("id",axis=1,inplace=True)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

X=train.drop(columns=["y"])
y=train["y"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=42)

num_cols.remove("id")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", RobustScaler(), num_cols),
        ("cat", OrdinalEncoder(), cat_cols)
    ]
)


test.info()


test.drop("id",axis=1,inplace=True)


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from xgboost import XGBClassifier
import numpy as np
import time

FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nğŸ”� Fold {fold}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    test_transformed = preprocessor.transform(test)

    model = XGBClassifier(
        max_depth=12,
        learning_rate=0.02,
        n_estimators=5000,
        subsample=0.6,
        colsample_bytree=0.6,
        use_label_encoder=False,
        eval_metric="logloss",
        early_stopping_rounds=25,
        verbosity=0
    )

    start = time.time()

    model.fit(
        X_train_transformed, y_train,
        eval_set=[(X_val_transformed, y_val)],
        verbose=100
    )

    val_probs = model.predict_proba(X_val_transformed)[:, 1]
    test_probs = model.predict_proba(test_transformed)[:, 1]

    oof_preds[val_idx] = val_probs
    test_preds += test_probs

    val_preds_label = (val_probs > 0.5).astype(int)
    auc = roc_auc_score(y_val, val_probs)
    acc = accuracy_score(y_val, val_preds_label)
    f1 = f1_score(y_val, val_preds_label)

    print(f"Fold {fold} ROC-AUC: {auc:.4f} | Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
    print(f"â�±ï¸� SÃ¼re: {time.time() - start:.1f} saniye")

test_preds /= FOLDS

final_preds_label = (oof_preds > 0.5).astype(int)
final_auc = roc_auc_score(y, oof_preds)
final_acc = accuracy_score(y, final_preds_label)
final_f1 = f1_score(y, final_preds_label)

print(f"\nğŸ“Š Final ROC-AUC: {final_auc:.4f}")
print(f"âœ… Final Accuracy: {final_acc:.4f}")
print(f"ğŸ�¯ Final F1-Score: {final_f1:.4f}")



sub=pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


sub["y"]=test_preds


sub.head()


sub.to_csv("submission.csv",index=False)




