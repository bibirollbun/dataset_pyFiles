import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display
import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/playground-series-s3e1/train.csv')
df_train


df_train.info()


df_train.drop('id', axis=1, inplace=True)


df_train.duplicated().sum()


df_train.isna().sum()


sns.pairplot(df_train)


sns.boxplot(df_train['AveOccup'])


### Removing outlier
df_train = df_train[df_train['AveOccup']<5]


df_train.reset_index(inplace=True, drop=True)


def regression_pipeline(df):
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    models = {
        "LinearRegression": (LinearRegression(), {}),
        "Ridge": (Ridge(), {"alpha": [0.1, 1, 10, 100]}),
        "Lasso": (Lasso(), {"alpha": [0.001, 0.01, 0.1, 1]}),
        # "RandomForest": (RandomForestRegressor(),
        #                   {"n_estimators": [50, 100, 200], "max_depth": [10, 20]}),
        # "GradientBoosting": (GradientBoostingRegressor(),
        #                       {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]})
    }
    
    best_model = None
    best_rmse = float("inf")
    best_model_name = ""
    
    for name, (model, param_grid) in models.items():
        if param_grid:
            search = RandomizedSearchCV(model, param_grid, n_iter=20, scoring='neg_root_mean_squared_error', cv=5)
            search.fit(X_train, y_train)
            best_model_for_type = search.best_estimator_
        else:
            best_model_for_type = model.fit(X_train, y_train)
        
        cv_rmse = -cross_val_score(best_model_for_type, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5).mean()
        print(name, cv_rmse)
        
        if cv_rmse < best_rmse:
            best_rmse = cv_rmse
            best_model = best_model_for_type
            best_model_name = name
    
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return {"best_model_name": best_model_name, "best_model": best_model, "cv_rmse": best_rmse, "test_rmse": test_rmse}



model = regression_pipeline(df_train)
model





df_test = pd.read_csv('/kaggle/input/playground-series-s3e1/test.csv')
df_test


y_pred = model['best_model'].predict(df_test.drop('id', axis=1))
y_pred


submission = pd.DataFrame({'id': df_test['id'], 'MedHouseVal': y_pred})
submission


submission.to_csv('submission.csv', index=False)




