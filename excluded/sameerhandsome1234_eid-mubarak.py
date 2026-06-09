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
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


train_df=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission_df=pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


def new_column(df):
    df['Total_Compound']= df['Nitrogen']+df['Potassium']+df['Phosphorous']
    df['Nitrogen_Prop']= df['Nitrogen']/df['Total_Compound']
    df['Potassium_Prop']= df['Potassium']/df['Total_Compound']
    df['Phosphorous_Prop']= df['Phosphorous']/df['Total_Compound']
    df['Temp_Humidity']= (df['Temparature']*df['Humidity'])/100
    df['Temp_Moisture']= (df['Temparature']*df['Moisture'])/100
    df['Humi_Moisture']= (df['Humidity']*df['Moisture'])/100

new_column(train_df)
new_column(test_df)



cat_cols = ['Soil Type', 'Crop Type']
for col in cat_cols:
    train_df[col] = LabelEncoder().fit_transform(train_df[col])
    test_df[col] = LabelEncoder().fit_transform(test_df[col])


train_df.head()


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
        p = p[:k]
        score = 0.0
        for i, pi in enumerate(p):
            if pi == a and pi not in p[:i]:
                score += 1.0 / (i + 1)
        return score

    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


import optuna


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10)
    }

    model = XGBClassifier(
        **params,
        objective='multi:softprob',
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        device="cuda"
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)
        top_3_preds = np.argsort(y_proba, axis=1)[:, ::-1][:, :3]
        map3 = mapk(y_val.tolist(), top_3_preds.tolist(), k=3)
        scores.append(map3)

    return np.mean(scores)


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)



params= {'n_estimators': 398, 'max_depth': 7, 'learning_rate': 0.09868377392213469, 'subsample': 0.8994067915071924, 'colsample_bytree': 0.7785922596888081, 'gamma': 0.00283612996100735, 'min_child_weight': 7}

model = XGBClassifier(
    **params,
    objective='multi:softprob',  
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    device="cuda" 
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
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


global2 = None
scores = []
desired_fold = 3  

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    if fold == desired_fold:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_proba_val = model.predict_proba(X_val)
        top_3_preds_val = np.argsort(y_proba_val, axis=1)[:, ::-1][:, :3]
        map3_score = mapk(y_val.tolist(), top_3_preds_val.tolist(), k=3)
        scores.append(map3_score)

        y_proba_test = model.predict_proba(X_test)
        global2 = y_proba_test

print(f"Fold {desired_fold} MAP@3 score: {scores}")



top_pred_indices = np.argmax(global2, axis=1)

top_pred_labels = le.inverse_transform(top_pred_indices)

submission_df['Fertilizer Name']= top_pred_labels


submission_df.to_csv('fsubmission1.csv', index=False)
from IPython.display import FileLink

FileLink("fsubmission1.csv")

