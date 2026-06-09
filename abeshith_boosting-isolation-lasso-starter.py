import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
import catboost
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

import warnings
warnings.simplefilter(action = "ignore")


df_train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
df_extra_train = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")


submission_data = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")


submission_data


def data_details(data):
    print(data.shape)
    return data.head()

print("Train Data")
data_details(df_train)


data_details(test_data)


data_details(submission_data)


print("Extra Train Data")
data_details(df_extra_train)


train_data = pd.concat([df_train,df_extra_train])


data_details(train_data)


train_data.isnull().sum()


test_data.isnull().sum()


categorical_cols = test_data.select_dtypes(include=['object']).columns
numerical_cols = test_data.select_dtypes(include=['number']).columns


for col in categorical_cols:
    train_data[col].fillna(train_data[col].mode()[0], inplace=True)
    test_data[col].fillna(test_data[col].mode()[0], inplace=True)

    encoder = LabelEncoder()
    train_data[col] = encoder.fit_transform(train_data[col])
    test_data[col] = encoder.transform(test_data[col])


num_imputer = SimpleImputer(strategy="median")
train_data[numerical_cols] = num_imputer.fit_transform(train_data[numerical_cols])
test_data[numerical_cols] = num_imputer.transform(test_data[numerical_cols])


X = train_data.drop(columns=['Price'])  
y = train_data['Price']
test = test_data.copy()


test.shape


# Handling outliers using IsolationForest 
%time
iso = IsolationForest(contamination=0.02, random_state=42)
outliers = iso.fit_predict(X)
mask = outliers == 1
X, y = X[mask], y[mask]


scaler = RobustScaler() 
X = scaler.fit_transform(X)
test = scaler.transform(test)


test.shape


%time
lasso = Lasso(alpha=0.01)
lasso.fit(X, y)
selector = SelectFromModel(lasso, prefit=True)
X_train_selected = selector.transform(X)
X_test_selected = selector.transform(test)


X_test_selected.shape


X_train, X_valid, y_train, y_valid = train_test_split(X_train_selected, y, test_size=0.2, random_state=42)


lgb_model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=8,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42
)


%time
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="rmse",
)


lgb_pred = lgb_model.predict(X_valid)
lgb_rmse = mean_squared_error(y_valid, lgb_pred, squared=False)
print(f"LightGBM RMSE: {lgb_rmse}")


cat_model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.01,
    depth=8,
    l2_leaf_reg=0.1,
    subsample=0.8,
    colsample_bylevel=0.8,
    loss_function="RMSE",
    random_seed=42,
    verbose=200
)


%time
cat_model.fit(
    X_train, y_train,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=100,
    verbose=200
)


cat_pred = cat_model.predict(X_valid)
cat_rmse = mean_squared_error(y_valid, cat_pred, squared=False)
print(f"CatBoost RMSE: {cat_rmse}")


xgb_model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    objective="reg:squarederror",
    random_state=42
)


xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="rmse",
    early_stopping_rounds=100,
    verbose=200
)


xgb_pred = xgb_model.predict(X_valid)
xgb_rmse = mean_squared_error(y_valid, xgb_pred, squared=False)
print(f"XGBoost RMSE: {xgb_rmse}")


%time
rmse_scores = {"LightGBM": lgb_rmse, "CatBoost": cat_rmse, "XGBoost": xgb_rmse}
best_model_name = min(rmse_scores, key=rmse_scores.get)


%time
if best_model_name == "CatBoost":
    print("CatBoost is the best model")
    best_model = cat_model
    test_predictions = cat_model.predict(X_test_selected)
elif best_model_name == "LightGBM":
    print("LightGBM is the best model")
    best_model = lgb_model
    test_predictions = lgb_model.predict(X_test_selected)
else:
    print("XGBoost is the best model")
    best_model = xgb_model
    test_predictions = xgb_model.predict(X_test_selected)


submission_data["Price"] = test_predictions  
submission_data.to_csv("submission.csv", index=False)
print("Submission file saved successfully!")


submission.shape


test_predictions.shape




