!pip install lightgbm mlflow


import numpy as np
import pandas as pd
from time import sleep

import matplotlib.pyplot as plt
import seaborn as sns

import json

from itertools import permutations



pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
# accident_risk - Ñ‚Ğ°Ñ€Ğ³ĞµÑ‚
train_data = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col='id')

test_data = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv', index_col='id')


trgt = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col='id')
trgt = trgt['accident_risk']



categorial_vals = {
    c : {
        y: x for x, y in enumerate(list(train_data[c].unique()))
    } 
    for c in train_data.select_dtypes('object').columns
}
cat_data = categorial_vals




for c in cat_data.keys():
    print(c)
    print(*[f'\t{k}: {cat_data[c][k]}' for k in cat_data[c]], sep='\n')


print(f"ĞŸÑ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¸ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğ¹ Ğ² Ñ‚Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡ĞºĞ¾Ğ¼ Ğ´Ğ°Ñ‚Ğ°Ñ�ĞµÑ‚Ğµ {train_data.isna().sum().sum() + train_data.isnull().sum().sum()}")
print(f"ĞŸÑ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¸ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğ¹ Ğ² Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ğ¾Ğ¼ Ğ´Ğ°Ñ‚Ğ°Ñ�ĞµÑ‚Ğµ {test_data.isna().sum().sum() + test_data.isnull().sum().sum()}")


train_data.info()


train_data.select_dtypes('object').describe()


train_data.select_dtypes(['int', 'float']).describe()


train_data.select_dtypes('bool').describe()


sns.histplot(train_data['accident_risk'], kde = True, bins = 200)


train_data.dtypes.unique()
r = train_data.select_dtypes('int', 'float').columns.shape[0]//3 + (train_data.select_dtypes('int', 'float').columns.shape[0]%3 > 0)*1
fig, axes = plt.subplots(r , 3, figsize=(12, 6))
axes = axes.flatten()
for i, col in enumerate(train_data.select_dtypes('int', 'float').columns):
    sns.barplot(
        data=train_data,
        hue=col,
        y='accident_risk',
        ax=axes[i],
        palette='viridis', 
        estimator="mean"
    )
    axes[i].set_ylabel('Mean accident_risk')
    axes[i].set_xlabel(col)


sns.boxplot(train_data['accident_risk'])


train_data[['accident_risk']].merge(train_data.select_dtypes(include='bool'), on=train_data.index).drop('key_0', axis=1)
train_data['accident_risk'].std()
def rpb(data: pd.DataFrame, target, column):
    y_0 = np.array(data[data[column] == False].groupby(column).mean())[0,0]
    y_1 = np.array(data[data[column] != False].groupby(column).mean())[0,0]
    n = len(data)
    n_0 = len(data[data[column] == False])
    n_1 = n - n_0
    sigma = data[target].std()
    return (y_1 - y_0)/sigma * ((n_1 * n_0)/(n*(n-1)))**0.5
rpb_coefs = {
    x : rpb(
        train_data[['accident_risk']].merge(
            train_data.select_dtypes(include='bool'), 
            on=train_data.index).drop('key_0', axis=1
            ), 
            'accident_risk', 
            x)
      for x in train_data.select_dtypes(include='bool').columns
}


pd.DataFrame(rpb_coefs, index=[0])


train_data_numeric = train_data.select_dtypes('number')
train_data_numeric.columns
corr_ = train_data_numeric.corr(method='spearman').iloc[:, -1]
corr_ = np.array(corr_).reshape(len(train_data_numeric.columns), 1 )

ax = sns.heatmap(corr_, cmap='CMRmap', yticklabels=list(train_data_numeric.columns),
                 xticklabels=['accident_risk'], 
                 annot=True
                 )


import shap
from sklearn.ensemble import RandomForestRegressor


for k in cat_data.keys():
    train_data[k] = train_data[k].map(cat_data[k])
data = train_data.copy()


X = data.drop(columns=['num_reported_accidents', 'accident_risk'], axis=1)
y = data['num_reported_accidents']

model = RandomForestRegressor(
    n_estimators=10,
    max_depth=6,
    random_state=42
    # n_jobs=-1
)
model.fit(X, y)

# Ğ¡Ğ¾Ğ·Ğ´Ğ°Ñ‘Ğ¼ Ğ¾Ğ±ÑŠÑ�Ñ�Ğ½Ğ¸Ñ‚ĞµĞ»ÑŒ SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Ğ“Ğ»Ğ¾Ğ±Ğ°Ğ»ÑŒĞ½Ğ°Ñ� Ğ²Ğ°Ğ¶Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²
shap.summary_plot(shap_values, X, plot_type="bar")
bool_cols = ['holiday', 'public_road', 'school_season', 'road_signs_present']

for col in bool_cols:
    X[col] = X[col].astype(int)

bucket_cols = [ 'time_of_day', 'weather', 'lighting', 'road_type']

for col in bucket_cols:
    X[col] = X[col].cat.codes if X[col].dtype.name == 'category' else X[col]

for c in data.columns:
    print(f'{c}: {data[c].unique()} {data[c].dtype}')
data.info()
data.select_dtypes(["int", "float"]).describe()
shap.summary_plot(shap_values, X, plot_type="dot")
## Ğ’Ñ‹Ğ²Ğ¾Ğ´


import featuretools as ft


feat_lst = list(data.columns)


for c in ['accident_risk'] + list(data.select_dtypes(include='bool').columns):
    feat_lst.remove(c)
    
for feat in feat_lst:
    data[feat] = data[feat] / data[feat].max()
data.head()


numerical_primitives = [
    'add_numeric',
    'subtract_numeric',
    'multiply_numeric'
]


transformation_primitives = [
    'and',
    'or',
    'not',
    'multiply_numeric_boolean',
    'multiply_boolean',
    'equal',
    'greater_than',
    'less_than',
    'isin',
    'is_null'    
]


es = ft.EntitySet(id="data_feat")
es = es.add_dataframe(dataframe_name="acc_df", dataframe=data[feat_lst], index="id")

feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name='acc_df',
    max_depth=2,
    trans_primitives=transformation_primitives + numerical_primitives,
    agg_primitives=[],  # Ğ±ĞµĞ· Ğ°Ğ³Ñ€ĞµĞ³Ğ°Ñ†Ğ¸Ğ¹
    verbose=True
    )
    


feature_matrix.head()
# feature_matrix['accident_risk'] = data['accident_risk']


from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb


shap.initjs()

X = feature_matrix
y = data['accident_risk']

model = RandomForestRegressor(
    n_estimators=10,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)


# Ğ¡Ğ¾Ğ·Ğ´Ğ°Ñ‘Ğ¼ Ğ¾Ğ±ÑŠÑ�Ñ�Ğ½Ğ¸Ñ‚ĞµĞ»ÑŒ SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)


# Ğ²Ğ°Ğ¶Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²
shap.summary_plot(shap_values, X, plot_type="bar")
shap.summary_plot(shap_values, X, plot_type="dot")
# 3. ĞŸĞ¾Ğ»ÑƒÑ‡Ğ°ĞµĞ¼ Ğ°Ğ±Ñ�Ğ¾Ğ»Ñ�Ñ‚Ğ½Ñ‹Ğµ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ñ� Ğ²Ğ°Ğ¶Ğ½Ğ¾Ñ�Ñ‚Ğ¸
shap_importance = np.abs(shap_values).mean(axis=0)
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'shap_importance': shap_importance
}).sort_values('shap_importance', ascending=False)


print("Ğ¢Ğ¾Ğ¿-3 Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ² Ğ¿Ğ¾ Ğ·Ğ½Ğ°Ñ‡Ğ¸Ğ¼Ğ¾Ñ�Ñ‚Ğ¸:")
feature_importance_df.head(3)


threshold = 0.02 #Ğ²Ğ°Ğ¶Ğ½Ğ¾Ñ�Ñ‚ÑŒ
selected_features = feature_importance_df[feature_importance_df['shap_importance'] > threshold]['feature'].tolist()


print(f"\nĞ�Ñ‚Ğ¾Ğ±Ñ€Ğ°Ğ½Ğ¾ {len(selected_features)} Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ² Ñ�Ğ¾ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸ĞµĞ¼ threshold > {threshold}:")
for feature in selected_features:
    importance = feature_importance_df[feature_importance_df['feature'] == feature]['shap_importance'].values[0]
    print(f"  {feature}: {importance:.4f}")


selected_features = ['curvature + speed_limit',
  'lighting + num_reported_accidents',
  'lighting * speed_limit']


train_data = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col='id')

test_data = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv', index_col='id')


for k in cat_data.keys():
    test_data[k] = test_data[k].map(cat_data[k])
    test_data[k].astype(int)
    
for k in cat_data.keys():
    train_data[k] = train_data[k].map(cat_data[k])
    train_data[k].astype(int)
    



# test_data = test_data[list(selected_features)]
test_data.head()
for feat in test_data.select_dtypes('number').columns:
    test_data[feat] = test_data[feat] / test_data[feat].max()


features_important = ['curvature', 'num_reported_accidents', 'speed_limit', 'lighting']


test_data = test_data[list(features_important)]
train_data = train_data[list(features_important)]

test_data['curvature + speed_limit'] = test_data['curvature'] +  test_data['speed_limit']
test_data['lighting + num_reported_accidents'] = test_data['lighting'] + test_data['num_reported_accidents']
test_data['lighting * speed_limit'] = test_data['lighting'] * test_data['speed_limit']

train_data['curvature + speed_limit'] = train_data['curvature'] +  train_data['speed_limit']
train_data['lighting + num_reported_accidents'] = train_data['lighting'] + train_data['num_reported_accidents']
train_data['lighting * speed_limit'] = train_data['lighting'] * train_data['speed_limit']


train_data.info()


test_data = test_data[selected_features]
train_data = train_data[selected_features]


train_data.info()


train_data.info()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test ,  y_train, y_test = train_test_split(train_data, trgt, test_size=0.3, random_state=42)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')



X_test.describe()


X_train.describe()


import mlflow
import mlflow.sklearn
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

from joblib import Parallel, delayed
import numpy as np
import pandas as pd





print("Ğ’Ğ°Ğ¶Ğ½Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸:", selected_features)


models = [

    {
        "name": "LightGBM",
        "estimator": LGBMRegressor(random_state=42),
        "params": {
            "learning_rate": [0.05, 0.01],
            "n_estimators": [100, 300, 600, 1000],
            "max_depth": [5, 8, 12],
            "num_leaves": [31, 63],
            "boosting_type": ['gbdt'],
            "min_data_in_leaf": [20, 50],
            "subsample": [0.8, 1.0]
            }
    },
    {
        "name": "LightGBM",
        "estimator": LGBMRegressor(random_state=42),
        "params": {
            "learning_rate": [0.05, 0.01],
            "n_estimators": [100, 300, 600, 1000],
            "max_depth": [5, 8, 12],
            "num_leaves": [31, 63],
            "boosting_type": ['goss'],
            "min_data_in_leaf": [20, 50],
            "subsample": [0.8, 1.0]
            }
    },

    {
        "name": "LightGBM",
        "estimator": LGBMRegressor(random_state=42),
        "params": {
            "learning_rate": [0.05, 0.01],
            "n_estimators": [100, 300, 600, 1000],
            "max_depth": [5, 8, 12],
            "num_leaves": [31, 63],
            "boosting_type": ['gbdt'],
            "min_data_in_leaf": [20, 50],
            "subsample": [0.8, 1.0]
            }
    },
    {
        "name": "LightGBM",
        "estimator": LGBMRegressor(random_state=42),
        "params": {
            "learning_rate": [0.05, 0.01],
            "n_estimators": [100, 300, 600, 1000],
            "max_depth": [5, 8, 12],
            "num_leaves": [31, 63],
            "boosting_type": ['goss'],
            "min_data_in_leaf": [20, 50],
            "subsample": [0.8, 1.0]
            }
    },

    {
        "name": "LightGBM",
        "estimator": LGBMRegressor(random_state=42),
        "params": {
            "learning_rate": [0.05, 0.01],
            "n_estimators": [100, 300, 600, 1000],
            "max_depth": [5, 8, 12],
            "num_leaves": [31, 63],
            "boosting_type": ['gbdt'],
            "min_data_in_leaf": [20, 50],
            "subsample": [0.8, 1.0]
            }
    }
]


models_final_list = []


mlflow.set_tracking_uri("file:./mlruns") 
mlflow.set_experiment("accident_risk_model_LGBMReg_3")


def train_and_log_model(
    model_name, estimator, param_distributions, 
    X_train, y_train, X_test, y_test, 
    n_iter: int=20
):

    with mlflow.start_run(run_name=model_name):
        try:

            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_distributions,
                n_iter=n_iter,               # ĞºĞ¾Ğ»-Ğ²Ğ¾ Ñ�Ğ»ÑƒÑ‡Ğ°Ğ¹Ğ½Ñ‹Ñ… ĞºĞ¾Ğ¼Ğ±Ğ¸Ğ½Ğ°Ñ†Ğ¸Ğ¹
                cv=3,
                scoring='r2',
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            
            y_pred = best_model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            # ğŸ“¤ Ğ›Ğ¾Ğ³Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¾Ğ² Ğ¸ Ğ¼ĞµÑ‚Ñ€Ğ¸Ğº
            mlflow.log_params(search.best_params_)
            metrics = {
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2
            }
            mlflow.log_metrics({
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2
            })

            # ğŸ“¤ Ğ›Ğ¾Ğ³Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸
            mlflow.sklearn.log_model(best_model, artifact_path="model")
            models_final_list.append([best_model, str(metrics)])

            print(f"âœ… {model_name} â€” Ğ»ÑƒÑ‡ÑˆĞ°Ñ� Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ: {best_model}")
            print(f"ğŸ“Š MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

            return {
                "model": model_name,
                "best_params": search.best_params_,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            }
        except Exception as e:
            print(e)


        
        
import os
n_jobs = os.cpu_count()
n_jobs


results = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
    delayed(train_and_log_model)(
        m["name"],
        m["estimator"],
        m["params"],
        X_train, y_train, X_test, y_test,
        n_iter=3 
    )
    for m in models
)


results_df = pd.DataFrame(results)


for b, r in zip(results_df['best_params'], results_df['R2']):
    print(r)
    print(b)


indx = 0
mxr2 = -1
for i, best in enumerate(models_final_list):
    if mxr2 < float(best[1].replace('{', '').replace('}', '').split()[-1]):
        mxr2 = float(best[1].replace('{', '').replace('}', '').split()[-1])
        indx = i


indx,mxr2 


# Ğ­Ñ‚Ğ¾ Ğ´Ğ»Ñ� Ğ·Ğ°Ğ¿ÑƒÑ�ĞºĞ° Ñ� ĞºĞ¾Ğ¼Ğ¿Ğ°, Ñ‡Ñ‚Ğ¾Ğ±Ñ‹ Ğ±Ñ‹Ğ»Ğ¾ ÑƒĞ´Ğ¾Ğ±Ğ½ĞµĞµ Ñ�Ğ¼Ğ¾Ñ‚Ñ€ĞµÑ‚ÑŒ Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒĞºĞ¸
# !mlflow ui

# with open('/kaggle/working/mlruns/841679242729576967/models/m-11469a5fc2e04d0f977710c6a6d8f178/metrics/R2') as f:
#     print(f.read())


# import joblib

# # Ğ—Ğ°Ğ³Ñ€ÑƒĞ·ĞºĞ° Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ… Ğ¸Ğ»Ğ¸ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸
# loaded_model = joblib.load('/kaggle/working/mlruns/841679242729576967/models/m-11469a5fc2e04d0f977710c6a6d8f178/artifacts/model.pkl')


# test_pred = loaded_model.predict(X_test[:5])
# print(test_pred)


# y_pred = loaded_model.predict(test_data)


# submission = pd.DataFrame({
#     'id': test_data.index, 
#     'accident_risk': y_pred
# }).reset_index(drop=True)


# submission.to_csv('/kaggle/working/submission.csv', index=False)


model = models_final_list[indx][0]
test_pred = model.predict(X_test[:5])
print(test_pred)


y_pred = model.predict(test_data)

submission = pd.DataFrame({
    'id': test_data.index, 
    'accident_risk': y_pred
}).reset_index(drop=True)


submission.to_csv('/kaggle/working/submission.csv', index=False)

