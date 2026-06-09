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


import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import seaborn as sns  


train_df=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission_df=pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


cat_cols = ['Soil Type', 'Crop Type']
for col in cat_cols:
    train_df[col] = LabelEncoder().fit_transform(train_df[col])
    test_df[col] = LabelEncoder().fit_transform(test_df[col])


le = LabelEncoder()
train_df["Fertilizer Name"] = le.fit_transform(train_df["Fertilizer Name"])
target_classes = le.classes_


X=train_df.drop(columns=['id','Fertilizer Name'])
y=train_df['Fertilizer Name']
X_test=test_df.drop(columns='id')


numeric_cols=X.select_dtypes(include=np.number).columns.tolist()


scaler=StandardScaler()
scaler.fit(X[numeric_cols])
X[numeric_cols]=scaler.transform(X[numeric_cols])
X_test[numeric_cols]=scaler.transform(X_test[numeric_cols])


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        score = 0.0
        for i in range(min(k, len(p))):
            if p[i] == a:
                score += 1.0 / (i + 1)
                break 
        return score
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])



param={'n_estimators': 617, 'max_depth': 8, 'learning_rate': 0.072994289007024, 'subsample': 0.6235403064167543, 'colsample_bytree': 0.7791783409048931, 'gamma': 0.6012356106903359, 'reg_alpha': 0.9727998785517237, 'reg_lambda': 4.036402441816036}

model = XGBClassifier(
    **param,
    objective='multi:softprob',  
    num_class=7,  
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    device="cuda" 
)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in kf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_val)
    top_3_preds = np.argsort(y_proba, axis=1)[:, ::-1][:, :3]

    map3 = mapk(y_val.tolist(), top_3_preds.tolist(), k=3)
    scores.append(map3)

print(f"Cross-Validation MAP@3 scores: {scores}")
print(f"Mean MAP@3: {np.mean(scores):.4f}")


global3 = None
desired_fold = 4

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    if fold == desired_fold:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_proba_val = model.predict_proba(X_val)
        top_3_preds_val = np.argsort(y_proba_val, axis=1)[:, ::-1][:, :3]
        map3_score = mapk(y_val.tolist(), top_3_preds_val.tolist(), k=3)

        y_proba_test = model.predict_proba(X_test)
        global3 = y_proba_test


from IPython.display import FileLink

top_3_preds = np.argsort(global3, axis=1)[:, -3:][:, ::-1] 

flat_preds = top_3_preds.ravel()
top_3_labels = le.inverse_transform(flat_preds).reshape(top_3_preds.shape)

submission_df['Fertilizer Name'] = [' '.join(row) for row in top_3_labels]


submission_df.to_csv('fsubmission9.csv', index=False)
FileLink("fsubmission9.csv")

