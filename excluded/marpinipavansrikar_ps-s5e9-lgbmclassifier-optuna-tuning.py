import optuna

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import StandardScaler,FunctionTransformer,PowerTransformer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor,early_stopping
from catboost import CatBoostRegressor


train_data = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
train_data.head(3)


train_data.shape


print(train_data.isnull().sum())


print(train_data.info())


train_data.describe()


train_data['AudioAmp'] = 10 ** (train_data['AudioLoudness'] / 20)


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

print('\nCorrelation with target:')
print(train_data.corr()['BeatsPerMinute'].sort_values(ascending=False))


num_features = train_data.select_dtypes(include=[np.number]).columns.tolist()
num_features.remove("id")

train_data[num_features].hist(bins=30, figsize=(15,12), layout=(4,3))
plt.suptitle("Feature Distributions")
plt.show()


for col in train_data.columns.drop('id'):
    plt.figure(figsize=(6,4))
    sns.boxplot(x=train_data[col])
    plt.title(f"Outliers in {col}")
    plt.show()


pt = PowerTransformer(method='yeo-johnson')
train_data[["AcousticQuality","InstrumentalScore","LivePerformanceLikelihood","VocalContent"]] = \
    pt.fit_transform(train_data[["AcousticQuality","InstrumentalScore","LivePerformanceLikelihood","VocalContent"]])


X = train_data.drop(['id','BeatsPerMinute'],axis=1)
y = train_data['BeatsPerMinute']
X.shape , y.shape
X


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42,stratify=y)
X_train.shape , y_train.shape , X_test.shape , y_test.shape


# def objective_lgbm(trial):
#     params_lgbm = {
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
#         "max_depth": trial.suggest_int("max_depth", 5, 20),
#         "min_child_samples": trial.suggest_int("min_child_samples", 1, 15),  
#         "subsample": trial.suggest_float("subsample", 0.1, 1),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1),
#         "reg_lambda": trial.suggest_float("reg_lambda", 1, 12),
#         "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
#         "n_estimators": 3000,
#         "objective": "regression",
#         "boosting_type": "gbdt",
#         "n_jobs": -1,
#         "random_state": 42,
#         "verbose": -1
#     }

#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     scores = []
    
#     for train_idx, valid_idx in kf.split(X, y):
#         X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
#         y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

#         model_lgb = LGBMRegressor(**params_lgbm)
#         model_lgb.fit(
#             X_train, y_train,
#             eval_set=[(X_valid, y_valid)],
#             eval_metric="mse",
#             callbacks=[early_stopping(100)]
#         )

#         preds = model_lgb.predict(X_valid)
#         rmse = np.sqrt(mean_squared_error(y_valid, preds))
#         scores.append(rmse)

#     return np.mean(scores)  


# study_lgbm = optuna.create_study(direction="minimize")  
# study_lgbm.optimize(objective_lgbm, n_trials=20, show_progress_bar=True)

# print(study_lgbm.best_params)


# optuna.visualization.plot_optimization_history(study_lgbm)


# optuna.visualization.plot_slice(study_lgbm,params=['learning_rate','max_depth','min_child_samples','subsample'])


# optuna.visualization.plot_param_importances(study_lgbm)


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


# model_lgb = LGBMRegressor(**best_params_lgbm)
# model_lgb.fit(X,y)


# def objective_cat(trial):
#     bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"])
    
#     params_cat = {
#         "iterations": 10000,
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
#         "depth": trial.suggest_int("depth", 4, 10),
#         "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
#         "random_strength": trial.suggest_float("random_strength", 0.1, 2.0),
#         "rsm": trial.suggest_float("rsm", 0.5, 1.0),
#         "bootstrap_type": bootstrap_type,
#         "loss_function": "RMSE",
#         "eval_metric": "RMSE",
#         "task_type": "CPU",
#         "random_seed": 42,
#         "verbose": False
#     }

#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     scores = []
    
#     for train_idx, valid_idx in kf.split(X, y):
#         X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
#         y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

#         model_cat = CatBoostRegressor(**params_cat)
#         model_cat.fit(
#             X_train, y_train,
#             eval_set=(X_valid, y_valid),
#             early_stopping_rounds=300,
#             use_best_model=True
#         )

    #     preds = model_cat.predict(X_valid)
    #     rmse = np.sqrt(mean_squared_error(y_valid, preds))
    #     scores.append(rmse)

    # return np.mean(scores)



# study_cat = optuna.create_study(direction="minimize")
# study_cat.optimize(objective_cat, n_trials=5, show_progress_bar=True)

# print("Best CatBoost Params:", study_cat.best_params)
# print("Best CatBoost RMSE:", study_cat.best_value)


# optuna.visualization.plot_param_importances(study_cat)


# optuna.visualization.plot_optimization_history(study_cat)


# Best CatBoost Params: {'bootstrap_type': 'Bernoulli', 'learning_rate': 0.05434275211871993
#, 'depth': 6, 'l2_leaf_reg': 6.7074408645594605, 'random_strength': 0.38489725742493125
#, 'rsm': 0.9279763638059497}
# Best CatBoost RMSE: 26.459522422011855


best_params_cat = {
        "iterations": 10000,
        "learning_rate": 0.05434275211871993,
        "depth": 6,
        "l2_leaf_reg": 6.7074408645594605,
        "random_strength": 0.38489725742493125,
        "rsm": 0.9279763638059497,
        "bootstrap_type": 'Bernoulli',
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "task_type": "CPU",
        "random_seed": 42,
        "verbose": False
    }


model_cat = CatBoostRegressor(**best_params_cat)
model_cat.fit(X,y)


# mean_squared_error(y_train,model_lgbm.predict(X_train)) , mean_squared_error(y_test,model_lgbm.predict(X_test))


# mean_squared_error(y_train,model_cat.predict(X_train)) , mean_squared_error(y_test,model_cat.predict(X_test))


test_data = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

train_data['AudioAmp'] = 10 ** (train_data['AudioLoudness'] / 20)
pt = PowerTransformer(method='yeo-johnson')
train_data[["AcousticQuality","InstrumentalScore","LivePerformanceLikelihood","VocalContent"]] = \
    pt.fit_transform(train_data[["AcousticQuality","InstrumentalScore","LivePerformanceLikelihood","VocalContent"]])

test_predict = model_cat.predict(test_data.drop('id',axis=1))
test_predict

Submission = pd.DataFrame({'id':test_data['id'],'BeatsPerMinute':test_predict})
Submission


Submission.to_csv('submission.csv',index = False)

