import pandas as pd
import numpy as np
import optuna
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import catboost
from catboost import CatBoostRegressor, Pool
import warnings
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train.head()


train.info()


train.isnull().sum()


test.isnull().sum()


train['Compartments'] = train['Compartments'].fillna(train['Compartments'].mean())
test['Compartments'] = test['Compartments'].fillna(test['Compartments'].mean())

train['Weight Capacity (kg)'] = train['Weight Capacity (kg)'].fillna(train['Weight Capacity (kg)'].median()).astype(int)
test['Weight Capacity (kg)'] = test['Weight Capacity (kg)'].fillna(test['Weight Capacity (kg)'].median()).astype(int)


cat_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']


for col in cat_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(train[col].mode()[0])


le = LabelEncoder()

for col in cat_cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.fit_transform(test[col])


train.head()


X = train.drop(columns=['Price'])
y = train['Price']


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)


def tune_catboost(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1500, 3500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 3, 15),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-6, 1e-2),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_state': 42,
        'task_type': 'GPU'
    }
    
    model = CatBoostRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        verbose=0,
        early_stopping_rounds=100  
    )
    
    preds = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, preds, squared=False)
    
    return rmse


#study = optuna.create_study(direction='minimize')
#study.optimize(tune_catboost, n_trials=100)

#print("Best Parameters:", study.best_params)#


model = catboost.CatBoostRegressor(
    n_estimators=1592,
    learning_rate= 0.08374385801321514,
    depth=4,
    l2_leaf_reg=0.0001256521912858505,
    bagging_temperature= 0.12300491216704376,
    random_strength= 0.11539317258746727,
    task_type="GPU",
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42
)
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

