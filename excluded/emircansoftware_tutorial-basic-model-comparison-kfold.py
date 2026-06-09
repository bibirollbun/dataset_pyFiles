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


train=pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sub=pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


train.head()


train.info()


train.drop("id",axis=1,inplace=True)


num_cols=train.select_dtypes(include=np.number).columns.tolist()
cat_cols=train.select_dtypes(include=["object","bool"]).columns.tolist()


for i in cat_cols:
    print(train[i].value_counts())


import matplotlib.pyplot as plt 
import seaborn as sns

for i in num_cols:
    plt.figure()
    sns.boxplot(x=train[i])
    plt.title(f"{i} Outlier")
    plt.show()


test.drop("id",axis=1,inplace=True)


from sklearn.preprocessing import OrdinalEncoder
encode=OrdinalEncoder()

for i in cat_cols:
    train[i]=encode.fit_transform(train[[i]])
    test[i]=encode.transform(test[[i]])


num_cols.remove("loan_paid_back")


from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()

for i in num_cols:
    train[i]=scaler.fit_transform(train[[i]])
    test[i]=scaler.transform(test[[i]])


X=train.drop("loan_paid_back",axis=1)
y=train["loan_paid_back"]


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def test_binary_classification_models(X_train, X_test, y_train, y_test):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RidgeClassifier': RidgeClassifier(),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'GradientBoosting': GradientBoostingClassifier(),
        'CatBoost': CatBoostClassifier(verbose=0),
        'KNN': KNeighborsClassifier(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    results = []
    plt.figure(figsize=(10, 8))

    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name} training is finished")
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc_score = roc_auc_score(y_test, y_score)

        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc_score
        })

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Binary Classification Models')
    plt.legend(loc='lower right')
    plt.show()

    return pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False).reset_index(drop=True)


#df_results = test_binary_classification_models(X_train, X_test, y_train, y_test)
#print(df_results)


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from catboost import CatBoostClassifier
import time

FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros((len(test), FOLDS))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nFold {fold}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostClassifier(
        iterations=2000,
        depth=8,
        learning_rate=0.01,
        subsample=0.9,
        colsample_bylevel=0.9,
        random_seed=42,
        loss_function="Logloss",
        eval_metric="AUC",
        od_type="Iter",
        verbose=100
    )

    start = time.time()

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )

    val_preds_proba = model.predict_proba(X_val)[:, 1]
    val_preds = (val_preds_proba >= 0.5).astype(int)

    test_preds[:, fold - 1] = model.predict_proba(test)[:, 1]

    oof_preds[val_idx] = val_preds_proba

    auc = roc_auc_score(y_val, val_preds_proba)
    acc = accuracy_score(y_val, val_preds)
    f1 = f1_score(y_val, val_preds)

    print(f"Fold {fold} AUC: {auc:.4f} | ACC: {acc:.4f} | F1: {f1:.4f}")
    print(f"Time: {time.time() - start:.1f} sec")

test_preds_mean = test_preds.mean(axis=1)

overall_auc = roc_auc_score(y, oof_preds)
print(f"\nOverall OOF ROC-AUC: {overall_auc:.4f}")


sub.head()


sub["loan_paid_back"]=test_preds_mean


sub.to_csv("submission.csv",index=False)




