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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import optuna


train_main = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv", index_col="id")
submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


train_main


train_extra = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")


train_extra


train_extra.columns = train_extra.columns.str.strip().str.replace(" ", "_")
train_extra = train_extra.rename(columns={"FertilizerName": "Fertilizer Name"})


common_features = list(set(train_main.columns) & set(train_extra.columns))
train_combined = pd.concat([train_main, train_extra[common_features]], ignore_index=True)
train_combined = train_combined.drop_duplicates().reset_index(drop=True)


label_encoder = LabelEncoder()
train_combined['Target'] = label_encoder.fit_transform(train_combined['Fertilizer Name'])


cat_cols = train_combined.select_dtypes(include='object').drop(columns=['Fertilizer Name']).columns.tolist()
num_cols = train_combined.select_dtypes(include=['int64', 'float64']).drop(columns=['Target']).columns.tolist()


train_combined[cat_cols] = train_combined[cat_cols].fillna('NaN')
test[cat_cols] = test[cat_cols].fillna('NaN')


ord_enc = OrdinalEncoder()
train_combined[cat_cols] = ord_enc.fit_transform(train_combined[cat_cols])
test[cat_cols] = ord_enc.transform(test[cat_cols])


scaler = StandardScaler()
train_combined[num_cols] = scaler.fit_transform(train_combined[num_cols])
test[num_cols] = scaler.transform(test[num_cols])


X = train_combined[cat_cols + num_cols]
y = train_combined['Target']
X_test = test[cat_cols + num_cols]


def map3(actual, pred_proba):
    top3 = np.argsort(pred_proba, axis=1)[:, -3:][:, ::-1]
    return np.mean([int(act in top3[i]) / (np.where(top3[i] == act)[0][0] + 1) if act in top3[i] else 0
                    for i, act in enumerate(actual)])

map3_scorer = make_scorer(lambda y_true, y_pred: map3(y_true, y_pred), greater_is_better=True, needs_proba=True)


def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
    }
    model = XGBClassifier(
        tree_method="hist",
        objective='multi:softprob',
        num_class=len(np.unique(y)),
        n_estimators=500,
        random_state=42,
        verbosity=0,
        **params
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring=map3_scorer)
    return np.mean(scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

best_params = study.best_params
print("Best Params:", best_params)


final_model = XGBClassifier(
    tree_method="hist",
    objective='multi:softprob',
    num_class=len(np.unique(y)),
    n_estimators=1000,
    random_state=42,
    verbosity=0,
    **study.best_params
)
final_model.fit(X, y)


y_pred = final_model.predict_proba(X_test)
top3_preds = np.argsort(y_pred, axis=1)[:, -3:][:, ::-1]
top3_labels = label_encoder.inverse_transform(np.ravel(top3_preds))
top3_df = pd.DataFrame(top3_preds, columns=['1', '2', '3']).applymap(lambda i: label_encoder.inverse_transform([i])[0])
submission['Fertilizer Name'] = top3_df.apply(lambda row: ' '.join(row.values), axis=1)
submission.to_csv("submission.csv", index=False)
submission.head()


submission


import joblib
joblib.dump(study, "study.pkl")


study = joblib.load("study.pkl")




