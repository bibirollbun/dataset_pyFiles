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


train=pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


train.head()


import matplotlib.pyplot as plt
import seaborn as sns

for i in train.columns:
    plt.figure()
    sns.boxplot(x=train[i])
    plt.show()
    


train.info()


train.drop("id",axis=1,inplace=True)


from sklearn.model_selection import train_test_split
X=train.drop("BeatsPerMinute",axis=1)
y=train["BeatsPerMinute"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=42)


from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
y_train_transformed=np.log1p(y_train)
y_test_transformed=np.log1p(y_test)


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def test_regression_models(X_train, X_test, y_train, y_test):
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'DecisionTree': DecisionTreeRegressor(),
        'GradientBoosting': GradientBoostingRegressor(),
        'XGBoost': XGBRegressor(verbosity=0)  
    }

    results = []

    for name, model in models.items():
        print(f"{name} traning")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"{name} finished")

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': np.expm1(rmse),
            'R2': r2
        })

    return pd.DataFrame(results).sort_values(by='R2', ascending=False).reset_index(drop=True)


#df_results = test_regression_models(X_train_scaled, X_test_scaled, y_train_transformed, y_test_transformed)
#print(df_results)


test.head()


test.drop("id",axis=1,inplace=True)


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import numpy as np
import time

FOLDS = 3
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\nğŸ”� Fold {fold}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    model = XGBRegressor(
        max_depth=12,
        learning_rate=0.03,
        n_estimators=10000,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=25,
        eval_metric="rmse"
    )

    start = time.time()

    model.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        verbose=100
    )

    val_preds_log = model.predict(X_val)
    test_preds += model.predict(test)

    oof_preds[val_idx] = val_preds_log

    rmse = np.sqrt(((val_preds_log - y_val_log) ** 2).mean())
    print(f"Fold {fold} Log-RMSE: {rmse:.4f}")
    print(f"â�±ï¸� Time: {time.time() - start:.1f} sec")

test_preds /= FOLDS

final_oof = np.expm1(oof_preds)

rmsle = np.sqrt(mean_squared_log_error(y, final_oof))
print(f"\nğŸ“Š Final RMSLE: {rmsle:.4f}")


final_oof


sub=pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")


sub.head()


test_preds=np.expm1(test_preds)


sub["BeatsPerMinute"]=test_preds


sub.head()


sub.to_csv("submission.csv",index=False)

