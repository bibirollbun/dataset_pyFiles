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
import optuna
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


print(train.info())
print(train.isnull().sum())


cat_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
for col in cat_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(test[col].mode()[0])

train['Weight Capacity (kg)'] = train['Weight Capacity (kg)'].fillna(train['Weight Capacity (kg)'].median())
test['Weight Capacity (kg)'] = test['Weight Capacity (kg)'].fillna(test['Weight Capacity (kg)'].median())


train['Compartments'] = train['Compartments'].astype(int)
test['Compartments'] = test['Compartments'].astype(int)


train['Waterproof'] = train['Waterproof'].map({'Yes': 1, 'No': 0})
train['Laptop Compartment'] = train['Laptop Compartment'].map({'Yes': 1, 'No': 0})
test['Waterproof'] = test['Waterproof'].map({'Yes': 1, 'No': 0})
test['Laptop Compartment'] = test['Laptop Compartment'].map({'Yes': 1, 'No': 0})

size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
train['Size'] = train['Size'].map(size_mapping)
test['Size'] = test['Size'].map(size_mapping)



cat_col = ['Brand', 'Material', 'Style', 'Color']
le = LabelEncoder()
for col in cat_col:
    train[col] = le.fit_transform(train[col])
    test[col] = le.fit_transform(test[col])



numerical_cols = ['Compartments', 'Weight Capacity (kg)']
scaler = StandardScaler()
train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
test[numerical_cols] = scaler.transform(test[numerical_cols])



train['Size_Weight_Interaction'] = train['Size'] * train['Weight Capacity (kg)']
test['Size_Weight_Interaction'] = test['Size'] * test['Weight Capacity (kg)']


poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)


train = train.drop(columns=['Compartments'], axis=1)  # Drop 'Compartments' before polynomial features
test = test.drop(columns=['Compartments'], axis=1)  # Drop 'Compartments' before polynomial features


numerical_cols = ['Weight Capacity (kg)', 'Size']  # Update this to include only remaining columns



poly_features = poly.fit_transform(train[numerical_cols])
poly_feature_names = poly.get_feature_names_out(numerical_cols)

train_poly = pd.DataFrame(poly_features, columns=poly_feature_names)
test_poly = pd.DataFrame(poly.transform(test[numerical_cols]), columns=poly_feature_names)


train = pd.concat([train, train_poly], axis=1)
test = pd.concat([test, test_poly], axis=1)



train = train.loc[:, ~train.columns.duplicated()]  # Remove duplicate columns in train
test = test.loc[:, ~test.columns.duplicated()]  # Remove duplicate columns in test



X = train.drop(columns=['id', 'Price'])
y = train['Price']
X_test = test.drop(columns=['id'])



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)



def objective(trial):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'device': 'gpu',
        'n_estimators': trial.suggest_int('n_estimators', 1000, 2000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 0.1),
        'max_depth': trial.suggest_int('max_depth', 10, 20),
        'reg_alpha': trial.suggest_uniform('reg_alpha', 0.1, 1.0),
        'reg_lambda': trial.suggest_uniform('reg_lambda', 0.5, 1.5),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 15),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.7, 0.9),
        'subsample': trial.suggest_uniform('subsample', 0.7, 0.9),
        'num_leaves': trial.suggest_int('num_leaves', 31, 60),
        'min_split_gain': trial.suggest_uniform('min_split_gain', 0.1, 0.3)
    }
    
    model = lgb.LGBMRegressor(**param)

    # **Train the model without early stopping or verbose (CORRECTION)**
    model.fit(X_train, y_train)  # Just fitting without additional arguments
    
    # Predict on validation set
    y_pred = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, y_pred, squared=False)
    
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)  # Adjust n_trials as needed

best_params = study.best_params
print(f"Best parameters: {best_params}")


model = lgb.LGBMRegressor(**best_params)
model.fit(X, y)


y_pred = model.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
print(f"Validation RMSE: {rmse}")


kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    cv_scores.append(rmse)

print(f"Cross-Validation RMSE: {np.mean(cv_scores)}")


import matplotlib.pyplot as plt

feature_importance = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance)
plt.xlabel('Feature Importance')
plt.ylabel('Feature Name')
plt.title('Feature Importance Plot')
plt.show()


lgb_model = lgb.LGBMRegressor(**best_params)
xgb_model = XGBRegressor()
cat_model = CatBoostRegressor(verbose=0)

ensemble_model = VotingRegressor(estimators=[
    ('lgb', lgb_model),
    ('xgb', xgb_model),
    ('cat', cat_model)
])

ensemble_model.fit(X, y)


predictions = ensemble_model.predict(X_test)


submission = pd.DataFrame({
    'id': test['id'],
    'num_sold': predictions
})
submission.to_csv('submission.csv', index=False)
print("Submission file saved!")

