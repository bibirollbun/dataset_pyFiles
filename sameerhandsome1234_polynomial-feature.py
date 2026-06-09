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


train_df=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission_df=pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.preprocessing import StandardScaler


def new_column(df):
    df['BMI'] = df['Weight'] / (df['Height']/100)**2
    df['Cardio_Effort_Score'] = (df['Heart_Rate'] * df['Duration']) / df['Body_Temp']
new_column(train_df)
new_column(test_df)


from sklearn.preprocessing import PolynomialFeatures
selected_cols=['Duration', 'Heart_Rate','Body_Temp', 'BMI']

poly = PolynomialFeatures(degree=2, include_bias=False)

X_train_poly = poly.fit_transform(train_df[selected_cols])
X_test_poly= poly.transform(test_df[selected_cols])

poly_feature_names = poly.get_feature_names_out(selected_cols)

df_train_poly = pd.DataFrame(X_train_poly, columns=poly_feature_names, index=train_df.index)
df_test_poly = pd.DataFrame(X_test_poly, columns=poly_feature_names, index=test_df.index)

train_df = pd.concat([train_df, df_train_poly], axis=1)
test_df = pd.concat([test_df, df_test_poly], axis=1)


train_df.columns


mapping={
    'male':1,
    'female':0
}
train_df['Sex']=train_df['Sex'].map(mapping)
test_df['Sex']=test_df['Sex'].map(mapping)


X=train_df.drop(columns=['id','Calories'])
y=train_df['Calories']
X_test=test_df.drop(columns=['id'])


numeric_cols=X.select_dtypes(include=np.number).columns.tolist()


scaler=StandardScaler()
scaler.fit(X[numeric_cols])
X[numeric_cols]=scaler.transform(X[numeric_cols])
X_test[numeric_cols]=scaler.transform(X_test[numeric_cols])


# Remove duplicate columns by name
X = X.loc[:, ~X.columns.duplicated()]
X_test = X_test.loc[:, ~X_test.columns.duplicated()]


import optuna
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error

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

    model = XGBRegressor(**params, random_state=42, n_jobs=-1)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_pred = np.clip(y_pred, 0, None)  # clip negative predictions to zero

        rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
        scores.append(rmsle)

    return np.mean(scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)



params={'n_estimators': 866, 'max_depth': 10, 'learning_rate': 0.02911342413265998, 'subsample': 0.9572495530185205, 'colsample_bytree': 0.9364410928684441, 'gamma': 4.297302159227095, 'min_child_weight': 1}
model=XGBRegressor(**params,random_state=42,n_jobs=-1)
kf = KFold(n_splits=5, shuffle=True, random_state=42) 

scores = []

for train_idx, val_idx in kf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_pred = np.clip(y_pred, 0, None)  

    rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
    scores.append(rmsle)
print(f"Cross Validation RMSLE: {scores}")


global2 = None
scores = []
desired_fold = 2 

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    if fold == desired_fold:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_pred = np.clip(y_pred, 0, None)

        global2 = model.predict(X_test)
        global2 = np.clip(global2, 0, None)

        rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
        scores.append(rmsle)

print(f"Cross-validation RMSLE score: {scores}")


#submission_df=pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submission_df['Calories']=global2
submission_df.to_csv('csubmission8.csv',index=False)
from IPython.display import FileLink

FileLink("csubmission8.csv")




