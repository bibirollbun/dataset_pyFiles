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
train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train.drop(columns = 'id',inplace=True)
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
test_id = test['id']
test.drop(columns = 'id',inplace=True)
train.info()


for col in train.columns:
    print(train[col].value_counts(),"\n","=="*30)


%%time
def rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def eda(df):
    if 'Size' in df.columns:
        df['Size'].replace({"Small": 0, "Medium": 1, "Large": 2}, inplace=True)

    categorical_columns = ['Brand', 'Material', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
    
    existing_categorical_columns = [col for col in categorical_columns if col in df.columns]

    df = pd.get_dummies(df, columns=existing_categorical_columns, dummy_na=True)

    df['Weight Capacity (kg)'].fillna(df['Weight Capacity (kg)'].mode()[0], inplace=True)

    from sklearn.ensemble import RandomForestRegressor

    train = df.dropna(subset=['Size'])  # Rows where 'Size' is not missing
    test = df[df['Size'].isna()]  # Rows where 'Size' is missing
    
    X_train = train.drop(columns=['Size'])
    y_train = train['Size']
    X_test = test.drop(columns=['Size'])
    
    # Train Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=16)
    rf.fit(X_train, y_train)
    
    # Predict missing values
    df.loc[df['Size'].isna(), 'Size'] = rf.predict(X_test)

    return df


train_df = eda(train)
test_df = eda(test)


%%time
train_df.info()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet,Ridge,SGDRegressor,BayesianRidge,TheilSenRegressor,RANSACRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


x = train_df.drop(columns = ['Price'])
y = train_df['Price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=16)


%%time
def model_list(models, random_state_num, x_train, x_test, y_train, y_test):
    rmse_score_dict = {}
    for model_class in models:
        model = model_class(random_state=random_state_num)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        model_name = model_class.__name__
        
        rmse_score_dict[model_name] = rmse(y_test, y_pred)
    
    return rmse_score_dict

models = [DecisionTreeRegressor, GradientBoostingRegressor, RandomForestRegressor, XGBRegressor, LGBMRegressor]

rmse_scores = model_list(models, 16, x_train, x_test, y_train, y_test)

print("RMSE Scores for different models:")
for model_name, score in rmse_scores.items():
    print(f"{model_name}: {score:.4f}")


%%time
def linear_model_list(models, x_train, x_test, y_train, y_test):
    rmse_score_dict = {}
    for model_class in models:
        model = model_class()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        model_name = model_class.__name__
        
        rmse_score_dict[model_name] = rmse(y_test, y_pred)
    
    return rmse_score_dict

models = [LinearRegression, Lasso, ElasticNet,Ridge,SGDRegressor,BayesianRidge,TheilSenRegressor,RANSACRegressor]

rmse_scores = linear_model_list(models, x_train, x_test, y_train, y_test)

print("RMSE Scores for different models:")
for model_name, score in rmse_scores.items():
    print(f"{model_name}: {score:.4f}")


params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.08,
    'max_depth': 13,
    'reg_alpha': 0.01,
    'lambda_l2': 0.01,  
    'min_child_samples' : 32,
    'colsample_bytree': 0.93,
    'subsample': 0.7, 
    'seed': 42,
    'verbose': -1,
    'device' : 'cpu' 
}
LGBM_model = LGBMRegressor(**params)
LGBM_model.fit(x_train, y_train)
y_pred = LGBM_model.predict(x_test)

# Calculate RMSLE score
score = rmse(y_test, y_pred)
print(f"RMSE for LGBMRegressor: {score:.4f}")


params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.08,
    'max_depth': 13,
    'reg_alpha': 0.01,
    'lambda_l2': 0.01,  
    'min_child_samples' : 32,
    'colsample_bytree': 0.93,
    'subsample': 0.7, 
    'seed': 42,
    'verbose': -1,
    'device' : 'cpu' 
}
LGBM_model = LGBMRegressor(**params)
LGBM_model.fit(x_train, y_train)
LGBM_preds = LGBM_model.predict(test_df)
predictions = pd.DataFrame({
    'id': test_id,
    'Price': LGBM_preds
})
predictions.to_csv('LGBM_ans.csv', index=False)


%%time
TheilSenRegressor_model = TheilSenRegressor()
TheilSenRegressor_model.fit(x_train, y_train)
TSR_preds = TheilSenRegressor_model.predict(test_df)
predictions = pd.DataFrame({
    'id': test_id,
    'Price': TSR_preds
})
predictions.to_csv('TSR_ans.csv', index=False)




