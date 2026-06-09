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


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv', index_col='id')
submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import VotingRegressor
from catboost import CatBoostRegressor


categorical_cols = train.select_dtypes(include="object").columns
numerical_cols = train.select_dtypes(include=["int64", "float64"]).columns

print("Categorical Columns:", list(categorical_cols))
print("Numerical Columns:", list(numerical_cols))


for col in categorical_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(data=train, x=col)
    plt.title(f"Bar Plot for {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


for col in numerical_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(train[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.show()


categorical_col = train.select_dtypes('object').columns.tolist()
for col in categorical_col:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


bool_col = train.select_dtypes('bool').columns.tolist()
train[bool_col] = train[bool_col].astype(int)
test[bool_col] = test[bool_col].astype(int)


plt.figure(figsize=(12, 12))
corr_matrix = train.corr()

sns.heatmap(corr_matrix,annot=True,
            fmt='.2f',cmap='coolwarm',
            center=0,square=True,
            linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()


def add_engineered_features(df):
    
    df["speed_curvature_ratio"] = df["speed_limit"] / (df["curvature"] + 1e-5)
    df["speed_curvature"] = df["speed_limit"] * df["curvature"]
    df["lanes_curvature"] = df["num_lanes"] * df["curvature"]
    df["speed_lanes"] = df["speed_limit"] * df["num_lanes"]
    df["curvature_sq"] = df["curvature"] ** 2
    df["traffic_complexity"] = (df["num_lanes"] * df["curvature"]) / (df["speed_limit"] + 1e-5)
    df["is_dark"] = df["lighting"].apply(lambda x: 1 if x in [1, 2] else 0)
    df["bad_weather"] = df["weather"].apply(lambda x: 1 if x in [1, 2] else 0)

    curv_mean = df["curvature"].mean()
    speed_mean = df["speed_limit"].mean()
    df["visibility_risk_curvature"] = df.apply(
        lambda row: 1 if row["is_dark"] == 1 and row["bad_weather"] == 1 and row["curvature"] > curv_mean else 0,
        axis=1
    )
    df["visibility_risk_speed"] = df.apply(
        lambda row: 1 if row["is_dark"] == 1 and row["bad_weather"] == 1 and row["speed_limit"] > speed_mean else 0,
        axis=1
    )
    
    return df


train = add_engineered_features(train)
test = add_engineered_features(test)


X = train.drop(columns=['accident_risk'])
y = train['accident_risk']

X_test = test.copy()


from sklearn.preprocessing import StandardScaler


num_col = X.select_dtypes(include=np.number).columns.tolist()
scaler = StandardScaler()
X[num_col] = scaler.fit_transform(X[num_col])
X_test[num_col] = scaler.transform(X_test[num_col])


import mlflow
import mlflow.xgboost
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("file:///kaggle/working/mlruns")
mlflow.set_experiment("Loan_Approval_XGBoost_Regression")

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'device': 'cuda',
    'n_jobs': -1,
    'random_state': 42,
    'n_estimators': 10000,
    'learning_rate': 0.014368221881254603,
    'max_depth': 7,
    'subsample': 0.7726730454314866,
    'colsample_bytree': 0.6298027410701106,
    'min_child_weight': 8.035739733368358,
    'gamma': 0.008651922110094842,
    'reg_alpha': 1.2204301402022693,
    'reg_lambda': 1.1288166455831825,
    'num_parallel_tree': 3
}

model = XGBRegressor(**params)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

with mlflow.start_run():
    mlflow.log_params(params)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, preds))
        scores.append(rmse)

        mlflow.log_metric(f"rmse_fold_{fold+1}", rmse)
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")

    avg_rmse = np.mean(scores)
    mlflow.log_metric("avg_rmse", avg_rmse)


    mlflow.xgboost.log_model(model, artifact_path="model")

print(" Average RMSE:", np.mean(scores))


global2=None

for fold_num, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    if fold_num == 4:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)

        global2 = model.predict(X_test)
        break


import mlflow
import mlflow.catboost

mlflow.set_tracking_uri("file:///kaggle/working/mlruns")
mlflow.set_experiment("Loan_Approval_CatBoost_Regression")

final_params = {
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'task_type': 'GPU',
    'devices': '0',
    'random_seed': 42,
    'iterations': 16000,
    'learning_rate': 0.006310591662509489,
    'depth': 7,
    'l2_leaf_reg': 2.4055075455358415,
    'bagging_temperature': 0.07633077474909343,
    'random_strength': 4.52216948206277,
    'border_count': 146,
    'min_data_in_leaf': 15
}

model1 = CatBoostRegressor(**final_params, verbose=False)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores1 = []

with mlflow.start_run():
    mlflow.log_params(final_params)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model1.fit(X_train, y_train)
        preds = model1.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, preds))
        scores1.append(rmse)

        mlflow.log_metric(f"rmse_fold_{fold+1}", rmse)
        print(f"Fold {fold+1} RMSE: {rmse:.5f}")

    avg_rmse = np.mean(scores1)
    mlflow.log_metric("avg_rmse", avg_rmse)
    print(" Average RMSE:", avg_rmse)

    mlflow.catboost.log_model(model1, artifact_path="catboost_model")


global3=None

for fold_num, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    if fold_num == 4:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model1.fit(X_train, y_train)

        global3 = model1.predict(X_test)
        break


import mlflow.sklearn

mlflow.set_tracking_uri("file:///kaggle/working/mlruns")
mlflow.set_experiment("VotingRegressor_Ensemble")

cat_model = CatBoostRegressor(
    **final_params,
    verbose=False 
)

xgb_model = XGBRegressor(
    **params,
    verbosity=0     
)

voting_model = VotingRegressor(
    estimators=[('cat', cat_model), ('xgb', xgb_model)],
    weights=[0.5, 0.5],
    n_jobs=-1
)


voting_model = VotingRegressor(**voting_params)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

global4 = np.zeros((X_test.shape[0], kf.get_n_splits()))

with mlflow.start_run():
    mlflow.log_param("voting_weights", voting_params['weights'])
    mlflow.log_param("n_models", len(voting_params['estimators']))
    mlflow.log_param("n_splits", kf.get_n_splits())

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fit model
        voting_model.fit(X_train, y_train)

        # Validation prediction
        preds = voting_model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        scores.append(rmse)
        mlflow.log_metric(f"rmse_fold_{fold+1}", rmse)
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")

        global4[:, fold] = voting_model.predict(X_test)

    global4_mean = np.mean(global4, axis=1)

    avg_rmse = np.mean(scores)
    mlflow.log_metric("avg_rmse", avg_rmse)
    print("\n Average RMSE:", avg_rmse)

    mlflow.sklearn.log_model(voting_model, artifact_path="voting_model")


from IPython.display import FileLink

submission['accident_risk']= global4_mean

submission.to_csv("asubmission7.csv",index=False)
FileLink("asubmission7.csv")


global5 = (0.3 * global2) + (0.3 * global3) + (0.4 * global4_mean)

submission['accident_risk']= global5

submission.to_csv("asubmission8.csv",index=False)
FileLink("asubmission8.csv")

