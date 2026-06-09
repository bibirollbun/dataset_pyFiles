import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt  
import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


train.head()


train.info()


train.isnull().sum()


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


label_encoders = {}  
for col in cat_cols:
    le = LabelEncoder()
    test[col] = le.fit_transform(test[col])
    label_encoders[col] = le


train.head()


train['num_sold'] = np.log1p(train['num_sold'])


X = train.drop(columns=['id','num_sold'])
y = train['num_sold']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 5000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.05),
        'depth': trial.suggest_int('depth', 4, 12),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-5, 10),
        'border_count': trial.suggest_int('border_count', 50, 255),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'random_strength': trial.suggest_loguniform('random_strength', 1, 20),
        'eval_metric': 'MAPE',
    }
    
    model = CatBoostRegressor(**params, verbose=0)

    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)

    y_pred = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, y_pred)

    return mape


#study = optuna.create_study(direction='minimize')
#study.optimize(objective, n_trials=50)

#print("Best MAPE:", study.best_value)
#print("Best Parameters:", study.best_params)


best_params = {
    'iterations': 4130,
    'learning_rate': 0.045367725598447144,
    'depth': 9,
    'l2_leaf_reg': 0.11044931448456394,
    'border_count': 253,
    'subsample': 0.9209882418913344,
    'random_strength': 2.5948687010212663,
    'eval_metric': 'MAPE'
}

model = CatBoostRegressor(**best_params, verbose=0)

model.fit(X, y, early_stopping_rounds=50, verbose=0)


test.head()


final_predictions = model.predict(test)  
final_predictions = np.expm1(final_predictions)


submission_ids = test['id']
submission = pd.DataFrame({
    'id': submission_ids,
    'num_sold': final_predictions
})


submission.to_csv('submission.csv', index=False)
print(submission.head())
print("File Saved!")

