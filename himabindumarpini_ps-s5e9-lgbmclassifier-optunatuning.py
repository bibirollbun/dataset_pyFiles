import optuna

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor,early_stopping


train_data = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
train_data.head(3)


test_data = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
test_data.head(3)


train_data.shape , test_data.shape


train_data['AudioAmp'] = 10 ** (train_data['AudioLoudness'] / 20)
test_data['AudioAmp'] = 10 ** (test_data['AudioLoudness'] / 20)


print(train_data.isnull().sum())
print("------------------------------")
print(test_data.isnull().sum())


print(train_data.info())
print("------------------------------")
print(test_data.info())


train_data.describe()


test_data.describe()


correlation = train_data.corr(method='spearman')

plt.figure(figsize=(11,5))
plt.title('Correlation heatmap')
sns.heatmap(
    correlation,
    annot=True,
    vmin=-1,
    vmax=1,
    cmap="GnBu_r",   
    center=0         
)
plt.show()



X = train_data.drop(['id','BeatsPerMinute'],axis=1)
y = train_data['BeatsPerMinute']
X.shape , y.shape
X


# scaler = StandardScaler()
# X = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)
# X 


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42,stratify=y)
X_train.shape , y_train.shape , X_test.shape , y_test.shape


# Model Training


def objective_lgbm(trial):
    params_lgbm = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 15),  
        "subsample": trial.suggest_float("subsample", 0.1, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 12),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
        "n_estimators": 3000,
        "objective": "regression",
        "boosting_type": "gbdt",
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, valid_idx in kf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model_lgb = LGBMRegressor(**params_lgbm)
        model_lgb.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="mse",
            callbacks=[early_stopping(100)]
        )

        preds = model_lgb.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, preds))
        scores.append(rmse)

    return np.mean(scores)  


study_lgbm = optuna.create_study(direction="minimize")  
study_lgbm.optimize(objective_lgbm, n_trials=20, show_progress_bar=True)

print(study_lgbm.best_params)


optuna.visualization.plot_optimization_history(study_lgbm)


optuna.visualization.plot_slice(study_lgbm,params=['learning_rate','max_depth','min_child_samples','subsample'])


optuna.visualization.plot_param_importances(study_lgbm)


# Best is trial 34 with value: 26.458431884671.
# {'learning_rate': 0.044487954953164516, 'max_depth': 15, 'min_child_samples': 14,
# 'subsample': 0.929477270124114, 'colsample_bytree': 0.9863994510127474, 
# 'reg_lambda': 5.736133624275964, 'reg_alpha': 1.6289994443777565}

best_params_lgbm = {
        "learning_rate": 0.044487954953164516,
        "max_depth": 15,
        "min_child_samples": 14,  
        "subsample": 0.929477270124114,
        "colsample_bytree": 0.9863994510127474,
        "reg_lambda": 5.736133624275964,
        "reg_alpha": 1.6289994443777565,
        "n_estimators": 10000,
        "objective": "regression",
        "boosting_type": "gbdt",
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1
    }


model_lgb = LGBMRegressor(**best_params_lgbm)
model_lgb.fit(X,y)


mean_squared_error(y_train,model_lgb.predict(X_train)) , mean_squared_error(y_test,model_lgb.predict(X_test))


test_predict = model_lgb.predict(test_data.drop('id',axis=1))
test_predict


Submission = pd.DataFrame({'id':test_data['id'],'BeatsPerMinute':test_predict})
Submission


Submission.to_csv('submission.csv',index = False)

