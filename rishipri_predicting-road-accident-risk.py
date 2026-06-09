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


train=pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sub=pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


train.info()


train.head()


num_cols = train.select_dtypes(include=np.number).columns.tolist()
cat_cols = train.select_dtypes(include=["object", "bool"]).columns.tolist()


num_cols.remove('id')


for i in cat_cols:
    print(train[i].value_counts())


import matplotlib.pyplot as plt
import seaborn as sns

for i in num_cols:
    plt.figure()
    sns.boxplot(x=train[i])
    plt.title(f'{i} Outlier')
    plt.show()


train.info()


test.info()



train.drop("id",axis=1,inplace=True)
test.drop("id",axis=1,inplace=True)


from sklearn.preprocessing import OrdinalEncoder
encode = OrdinalEncoder()


for i in cat_cols:
      train[i]=encode.fit_transform(train[[i]])
      test[i]= encode.transform(test[[i]])
    


train.head()


num_cols.remove("accident_risk")


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

for i in num_cols:
    train[i]=scaler.fit_transform(train[[i]])
    test[i]=scaler.transform(test[[i]])


train.head()


X=train.drop("accident_risk",axis=1)
y=train["accident_risk"]


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def test_regression_models(X_train, X_test, y_train, y_test):
    models={
         'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'DecisionTree': DecisionTreeRegressor(),
        'RandomForest': RandomForestRegressor(),
        'GradientBoosting': GradientBoostingRegressor(),
        'CatBoost':CatBoostRegressor(),
        'KNN': KNeighborsRegressor(),
        'XGBoost': XGBRegressor(verbosity=0) 
    }
    result = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name} training is finished")
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        })

    return pd.DataFrame(results).sort_values(by='R2', ascending=False).reset_index(drop=True)
    
    



from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
import numpy as np
import time


FOLDS = 3
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=8)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\nFold {fold}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBRegressor(
        max_depth=8,
        learning_rate=0.01,
        n_estimators=2000,
        subsample=0.9,
        colsample_bytree=0.9,
        early_stopping_rounds=25,
        eval_metric="rmse"
    )
    start = time.time()

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    val_preds = model.predict(X_val)
    test_preds += model.predict(test)

    oof_preds[val_idx] = val_preds
    rmse = np.sqrt(((val_preds - y_val) ** 2).mean())
    print(f"Fold {fold} RMSE: {rmse:.4f}")
    print(f"Time: {time.time() - start:.1f} sec")

test_preds /= FOLDS




test_preds


sub.head()


sub["accident_risk"]=test_preds


sub.to_csv("submission.csv",index=False)




