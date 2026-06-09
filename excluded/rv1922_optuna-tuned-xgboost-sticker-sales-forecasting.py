import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
import optuna
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb
from xgboost import XGBRegressor


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


train.head()


train.info()


train = train.dropna()
test = test.dropna()


def transform_date(df, col):
    df[col] = pd.to_datetime(df[col])
    
    df['year'] = df[col].dt.year.astype('int')
    df['quarter'] = df[col].dt.quarter.astype('int')
    df['month'] = df[col].dt.month.astype('int')
    df['day'] = df[col].dt.day.astype('int')
    df['day_of_week'] = df[col].dt.dayofweek.astype('int')
    df['week_of_year'] = df[col].dt.isocalendar().week.astype('int')
    
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['year_sin'] = np.sin(2 * np.pi * df['year'] / 7)
    df['year_cos'] = np.cos(2 * np.pi * df['year'] / 7)
    
    df['group'] = (df['year'] - 2010) * 48 + df['month'] * 4 + df['day'] // 7
    
    return df


train = transform_date(train, 'date')
test = transform_date(test, 'date')


train = train.drop(columns=['date'], axis=1)
test = test.drop(columns=['date'], axis=1)


cat_cols = ['country','store','product']


label_encoders = {}  
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    label_encoders[col] = le


train.head()


train['num_sold'] = np.log1p(train['num_sold'])


X = train.drop(columns=['num_sold'])
y = train['num_sold']


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 6, 20),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 0.3),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'objective': 'reg:squarederror',
        'eval_metric': 'mape',
        'tree_method': 'hist',
        'n_jobs': -1
    }

    model = xgb.XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    mape = mean_absolute_percentage_error(y_valid, preds)

    return mape


#study = optuna.create_study(direction='minimize')  
#study.optimize(objective, n_trials=100)  

#best_params = study.best_params
#print(f"Best hyperparameters: {best_params}")


model = xgb.XGBRegressor(
    booster='gbtree',
    n_estimators=756,
    learning_rate=0.04904806078551087,
    max_depth=9,
    min_child_weight=183,
    gamma=1.1575719123385967e-05,
    subsample=0.9133647509179157,
    colsample_bytree=0.6156377400418497,
    reg_alpha=0.746087025727205,
    reg_lambda=0.06959880783708906,
    random_state=42
)

model.fit(X, y)


test.head()


label_encoders = {}  
for col in cat_cols:
    le = LabelEncoder()
    test[col] = le.fit_transform(test[col])
    label_encoders[col] = le


test.head()


submission_ids = test['id']
predictions = model.predict(test)


predictions = np.expm1(predictions)


submission = pd.DataFrame({
    'id': submission_ids,
    'num_sold': predictions 
})


submission.to_csv('submission.csv', index=False)
print("File Saved!")
print(submission.head())

