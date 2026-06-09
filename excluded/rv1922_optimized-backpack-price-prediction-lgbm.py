import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train.head()


train.info()


train.isnull().sum()


cat_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']


for col in cat_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(test[col].mode()[0])


train['Compartments'] = train['Compartments'].astype(int)
test['Compartments'] = test['Compartments'].astype(int)


train['Waterproof'] = train['Waterproof'].map({'Yes': 1, 'No': 0})
train['Laptop Compartment'] = train['Laptop Compartment'].map({'Yes': 1, 'No': 0})

size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
train['Size'] = train['Size'].map(size_mapping)


test['Waterproof'] = test['Waterproof'].map({'Yes': 1, 'No': 0})
test['Laptop Compartment'] = test['Laptop Compartment'].map({'Yes': 1, 'No': 0})

size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
test['Size'] = test['Size'].map(size_mapping)


cat_col = ['Brand', 'Material','Style', 'Color']


le = LabelEncoder()

for col in cat_col:
    train[col] = le.fit_transform(train[col])
    test[col] = le.fit_transform(test[col])


train.head()


X = train.drop(columns=['Price'])
y = train['Price']


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

    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])

    y_pred = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, y_pred, squared=False)
    
    return rmse


#study = optuna.create_study(direction='minimize')
#study.optimize(objective, n_trials=100)

#best_params = study.best_params
#print(f"Best parameters: {best_params}")


train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid)


params = {
    'n_estimators': 1609,
    'learning_rate': 0.005049978711721023,
    'max_depth': 15,
    'reg_alpha': 0.7150416723617203,
    'reg_lambda': 1.0138609177771931,
    'min_child_samples': 9,
    'colsample_bytree': 0.7490172199155802,
    'subsample': 0.7716379803944933,
    'num_leaves': 33,
    'min_split_gain': 0.19258368404480392,
    'objective': 'regression',
    'metric': 'rmse',
    'device': 'gpu'  # Enable GPU acceleration
}
model = lgb.LGBMRegressor(**params)
model.fit(X, y)


test.head()


submission_ids = test['id']
predictions = model.predict(test)


submission = pd.DataFrame({
    'id': submission_ids,
    'num_sold': predictions 
})


submission.to_csv('submission.csv', index=False)
print("File Saved!")
print(submission.head())

