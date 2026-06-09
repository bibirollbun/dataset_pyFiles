import pandas as pd
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter('ignore')

from cuml.preprocessing import TargetEncoder

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression

import optuna


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv", index_col='id')
training_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv", index_col='id')


print(f"Train shape: {train.shape}")
print(f"Training extra shape: {training_extra.shape}")
print(f"Test shape: {test.shape}")


print("Train columns:", train.columns)
print("Training extra columns:", training_extra.columns)
print("Test columns:", test.columns)


merged_data = pd.concat([train, training_extra], axis=0, ignore_index=True)

print(f"Merged data shape: {merged_data.shape}")


discrete = [
    var for var in merged_data.columns if merged_data[var].dtype != 'O' and var != 'Price'
    and merged_data[var].nunique() <= 10
]
continuous = [
    var for var in merged_data.columns
    if merged_data[var].dtype != 'O' and var != 'Price' and var not in discrete
]

# categorical
categorical = [var for var in merged_data.columns if merged_data[var].dtype == 'O']

print('There are {} discrete variables'.format(len(discrete)))
print('There are {} continuous variables'.format(len(continuous)))
print('There are {} categorical variables'.format(len(categorical)))


merged_data.isnull().mean()[merged_data.isnull().mean() > 0]


test.isnull().mean()[test.isnull().mean() > 0]


for col in categorical:
    print(merged_data[col].value_counts())
    print()


def diagnostic_plots(df, variable):

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 3, 1)
    sns.histplot(df[variable], bins=30)
    plt.title('Histogram')

    plt.subplot(1, 3, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.ylabel('RM quantiles')

    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[variable])
    plt.title('Boxplot')

    plt.show()


diagnostic_plots(merged_data, 'Compartments')


diagnostic_plots(merged_data, 'Weight Capacity (kg)')


for var in discrete:
    merged_data.groupby(var)['Price'].median().plot()
    plt.ylabel('Median Price per label')
    plt.title(var)
    plt.show()


merged_data.describe()


merged_data[categorical] = merged_data[categorical].fillna('Missing').astype('category')
merged_data[continuous] = merged_data[continuous].fillna(merged_data[continuous].median())

test[categorical] = test[categorical].fillna('Missing').astype('category')
test[continuous] = test[continuous].fillna(test[continuous].median())


merged_data['Weight_sin'] = np.sin(merged_data['Weight Capacity (kg)'])
merged_data['Weight_cos'] = np.cos(merged_data['Weight Capacity (kg)'])

test['Weight_sin'] = np.sin(test['Weight Capacity (kg)'])
test['Weight_cos'] = np.cos(test['Weight Capacity (kg)'])


te_params = {'n_folds': 20, 'smooth': 20, 'split_method': 'random', 'stat': 'mean'}
TE = TargetEncoder(**te_params)


features = [x for x in merged_data.columns if x!='Price']
target = 'Price'

for col in features:
    merged_data[f"TE_{col}"] = TE.fit_transform(merged_data[col], merged_data[target])
    test[f"TE_{col}"] = TE.transform(test[col])


for cat_col in categorical:
    for num_col in continuous + discrete:
        intera_col = f'{cat_col}_x_{num_col}'
        merged_data[intera_col] = merged_data[cat_col].astype(str) + '_' + merged_data[num_col].astype(str)
        test[intera_col] = test[cat_col].astype(str) + '_' + test[num_col].astype(str)
        merged_data[intera_col] = merged_data[intera_col].astype('category').cat.codes
        test[intera_col] = test[intera_col].astype('category').cat.codes


for col in categorical:
        merged_data[col] = merged_data[col].astype('category').cat.codes
        test[col] = test[col].astype('category').cat.codes


scaler = StandardScaler().set_output(transform="pandas")

price_train = merged_data['Price']

scaler.fit(merged_data.drop(columns=['Price']))

merged_data = scaler.transform(merged_data.drop(columns=['Price']))
test = scaler.transform(test)

merged_data['Price'] = price_train


train_sample = merged_data.sample(frac=0.5, random_state=42)


train_sample.shape


def objective(trial):
    
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 500, 2500),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "random_state": 42,
        "verbose": -1,
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": -1
    }

    X_train, X_valid, y_train, y_valid = train_test_split(
        train_sample.drop(columns=['Price']), 
        train_sample['Price'], 
        test_size=0.2, 
        random_state=42
    )

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    
    preds = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    
    return rmse



study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

best_params = study.best_trial.params
best_cv_rmse = study.best_value


print(f"Best params: {best_params}")
print(f"Best CV RMSE: {best_cv_rmse}")


model_lgm = lgb.LGBMRegressor(
    **best_params,
    boosting_type="gbdt",
    objective="regression",
    metric="rmse",
    random_state=42,
    verbose=-1,
    device="gpu",
    gpu_platform_id=0,
    gpu_device_id=-1
)


model_lgm.fit(
    merged_data.drop(columns=['Price']), merged_data['Price'],
)


importance = model_lgm.feature_importances_

sorted_idx = np.argsort(importance)[::-1]
features = merged_data.columns

plt.figure(figsize=(10, 6))
plt.barh([features[i] for i in sorted_idx], importance[sorted_idx])
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("LGBM Feature Importance") 
plt.gca().invert_yaxis()  
plt.show()


test_preds = model_lgm.predict(test)


submission = pd.DataFrame({ "id": test.index, target: test_preds })

submission.to_csv("submission-ss.csv", index=False)




