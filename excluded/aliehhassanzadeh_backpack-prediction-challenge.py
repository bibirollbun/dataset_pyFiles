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


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')


from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


from sklearn.model_selection import train_test_split



train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
train_extra_df = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


train_df.head()


test_df.head()


print("Shape of the train dataset: ", train_df.shape)
print("Shape of the test dataset: ", test_df.shape)


train_df.info()


train_df.describe()


train_df.corr(numeric_only=True)


train_df.isna().sum()


test_df.isna().sum()


train_df.duplicated().sum()


test_df_ids = test_df["id"]

train_df.drop("id", inplace=True, axis=1)
test_df.drop("id", inplace=True, axis=1)


from sklearn.model_selection import train_test_split

# Separate Features (X) and Target (y) from train_df
X = train_df.drop(columns=["Price"])  # Drop "Price" from features
y = train_df["Price"]  # Target variable

# Split train dataset into Train and Validation Sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_test = test_df


#!pip install catboost


rmse_results = {}
r2_score_results = {}


from catboost import Pool

cat_features = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
num_features = ["Compartments", "Weight Capacity (kg)" ]


# Train-Validation Split (Train: 80%, Val: 20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply KNN Imputer for Numerical Features
knn_imputer = KNNImputer(n_neighbors=5)
X_train[num_features] = knn_imputer.fit_transform(X_train[num_features])
X_val[num_features] = knn_imputer.transform(X_val[num_features]) 
X_test[num_features] = knn_imputer.transform(X_test[num_features]) 

# Apply Simple Imputer for Categorical Features
cat_imputer = SimpleImputer(strategy='most_frequent')  
X_train[cat_features] = cat_imputer.fit_transform(X_train[cat_features])
X_val[cat_features] = cat_imputer.transform(X_val[cat_features]) 
X_test[cat_features] = cat_imputer.transform(X_test[cat_features])

# Ensure categorical features are of type string
X_train[cat_features] = X_train[cat_features].astype(str)
X_val[cat_features] = X_val[cat_features].astype(str)
X_test[cat_features] = X_test[cat_features].astype(str)

# Create CatBoost Pools
train_pool = Pool(X_train, label=y_train, cat_features=cat_features)
val_pool = Pool(X_val, label=y_val, cat_features=cat_features) 
test_pool = Pool(X_test, cat_features=cat_features)  



from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

cb_model = CatBoostRegressor(
    task_type="GPU" ,
    loss_function="RMSE",         
    verbose=100,                  
    iterations=1000,              
    learning_rate=0.01,          
    l2_leaf_reg=10                
)



cb_model.fit(train_pool, eval_set=val_pool)


y_pred = cb_model.predict(X_val)


catboost_rmse = mean_squared_error(y_val, y_pred, squared=False)
catboost_r2 = r2_score(y_val, y_pred)

print(f"CatBoost Validation RMSE: {catboost_rmse:.2f}")
print(f"CatBoost RÂ² Score: {catboost_r2:.4f}")


rmse_results['CatBoost'] = catboost_rmse
r2_score_results['CatBoost'] = catboost_r2


y_test_pred = cb_model.predict(X_test)


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
for cat_feature in cat_features:
    
    X_train[cat_feature] = label_encoder.fit_transform(X_train[cat_feature])
    
   
    X_val[cat_feature] = label_encoder.transform(X_val[cat_feature])
    X_test[cat_feature] = label_encoder.transform(X_test[cat_feature])


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Train RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict
y_pred_rf = rf_model.predict(X_val)

# Evaluate
rf_rmse = mean_squared_error(y_val, y_pred_rf, squared=False)
rf_r2 = r2_score(y_val, y_pred_rf)

print(f"Random Forest RMSE: {rf_rmse:.2f}")
print(f"Random Forest RÂ² Score: {rf_r2:.4f}")

rmse_results['Random Forest'] = rf_rmse
r2_score_results['Random Forest'] = rf_r2

y_test_pred = rf_model.predict(X_test)


import xgboost as xgb

# Train XGBoost Regressor
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict
y_pred_xgb = xgb_model.predict(X_val)

# Evaluate
xgb_rmse = mean_squared_error(y_val, y_pred_xgb, squared=False)
xgb_r2 = r2_score(y_val, y_pred_xgb)

print(f"XGBoost RMSE: {xgb_rmse:.2f}")
print(f"XGBoost RÂ² Score: {xgb_r2:.4f}")

rmse_results['XGBoost'] = xgb_rmse
r2_score_results['XGBoost'] = xgb_r2

y_test_pred = xgb_model.predict(X_test)


import lightgbm as lgb


lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_val)
lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_pred))
lgb_r2 = r2_score(y_val, lgb_pred)


print(f"LightGBM RMSE: {lgb_rmse:.2f}")
print(f"LightGBM RÂ² Score: {lgb_r2:.4f}") 


rmse_results['LightGBM'] = lgb_rmse
r2_score_results['LightGBM'] = lgb_r2

y_test_pred = xgb_model.predict(X_test)


models_results = pd.DataFrame({
    'Model': list(rmse_results.keys()),
    'RMSE': list(rmse_results.values()),
    'R2_Score': list(r2_score_results.values())
})
print(models_results.sort_values(by="RMSE"))


results = pd.DataFrame({
    'id': test_df_ids, 
    'Price': y_test_pred  
})


results.to_csv("submission.csv", index=False)

print(results.head())  





