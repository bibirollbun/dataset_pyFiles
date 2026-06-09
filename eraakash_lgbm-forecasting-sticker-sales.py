import pandas as pd 
import numpy as np 


import warnings 
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


train_df.head()


import pickle 

with open('/kaggle/input/gdp-dictionary/gdp_dict.pkl', 'rb') as f:
    gdp_dict = pickle.load(f)


pd.DataFrame(gdp_dict)


%pip install holidays pycountry


import holidays 
import pycountry as pc 


def find_transformable_methods(col):
    # find all possible methods that can be applied to the column (date methods)
    col_date = pd.to_datetime(col, errors='coerce')

    all_attrs = dir(col_date.dt)
    transformable = [attr for attr in all_attrs if not callable(getattr(col_date.dt, attr)) and not attr.startswith('_')]

    return transformable


def preprocessing(df):
    # == Train Set Tweaks == 
    if 'num_sold' in df.columns:
        df.drop(columns=['id'], inplace=True)
        df['num_sold'] = np.log(df['num_sold'])

    # == Transforming Date ==
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    all_methods = [] 
    
    for method in find_transformable_methods(df['date']):
        all_methods.append(method) 
        if method in [
            'date', 'time', 'timetz', 'unit', 'freq', 'tz', 
            'weekday', 'dayofyear', 'dayofweek',
            'is_month_start', 'is_month_end', 'is_quarter_start',
            'is_quarter_end', 'is_year_start', 'is_year_end'
        ]:
            continue 

        df[method] = getattr(df['date'].dt, method)
        df[method] = df[method].astype(np.float32)

        if df[method].isnull().any():
            print(f'Nulls in {method}: {df[method].isnull().sum()}')

        # convert to sine and cosine 
        if 'is_' not in method:
            df[f'{method}_sin'] = np.sin(2 * np.pi * df[method] / df[method].max())
            df[f'{method}_cos'] = np.cos(2 * np.pi * df[method] / df[method].max())

            # drop original column
            df.drop(columns=[method], inplace=True)

    for col in df.columns:
        if 'is_leap_year' == col:
            continue

        if df[col].nunique() <= 1:
            df.drop(columns=[col], inplace=True)


    # == GDP Features ==
    df['year'] = df['date'].dt.year
    df['gdp'] = df.apply(lambda row: gdp_dict[row['country']][row['year']], axis=1)


    # == Country Features ==
    def get_country_abbr(country):
        try:
            return pc.countries.get(name=country).alpha_2
        except:
            return None
        
    df['country'] = df['country'].map(get_country_abbr)


    # == Holiday Features ==
    def get_holiday(date_, country):
        try:
            country_holidays = holidays.country_holidays(country)
            return 1 if type(country_holidays.get(date_)) == str else 0
        except:
            return None
        
    df['is_holiday'] = df.apply(lambda x: get_holiday(x['date'], x['country']), axis=1)
    df['pre_holiday'] = df['is_holiday'].shift(1)
    df['post_holiday'] = df['is_holiday'].shift(-1)
    
    df['pre_holiday'].fillna(0, inplace=True)
    df['post_holiday'].fillna(0, inplace=True)

    # == Label Encoding ==
    cat_cols = df.select_dtypes(include='object').columns

    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes
        
    print('---')
    return df.drop(columns=['date']).dropna()

train = preprocessing(train_df.copy())
test = preprocessing(test_df.copy())


from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit

X = train.drop(columns=['num_sold'])
y = train['num_sold']

import optuna
from sklearn.metrics import mean_absolute_percentage_error as mape_

def mape(y_true, y_pred):
    y_true_inv = np.exp(y_true)
    y_pred_inv = np.exp(y_pred)
    
    score = mape_(y_true_inv, y_pred_inv)
    return score


def train_model(X, y, params, oof=False):
    model = LGBMRegressor(**params)
    
    # Use TimeSeriesSplit for time-series aware cross-validation
    cv = TimeSeriesSplit(n_splits=5)
    scores = []

    oof_preds = []
    
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        score = mape(y_test, y_pred)
        print(f'Fold Score: {score}')

        if oof:
            test_preds = model.predict(test.drop(columns=['id']))
            oof_preds.append(test_preds)

        scores.append(score)

    if oof:
        return oof_preds, np.mean(scores), model
    return [], np.mean(scores), model


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_uniform('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_uniform('reg_lambda', 0.0, 1.0),
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1,
        'metric': 'mape',
        # use gpu
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0
    }

    _, score, _ = train_model(X, y, params)
    return score


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)


study.best_value


params = study.best_params

oof_preds, scores, model = train_model(X, y, params, oof=True)


import matplotlib.pyplot as plt

def plot_feature_importance(model, X):
    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)

    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure(figsize=(12, 6))
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, X.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

plot_feature_importance(model, X)


def build_submission():
    oof_preds_inv = np.exp(oof_preds)
    oof_preds_inv = np.mean(oof_preds_inv, axis=0)

    submission = pd.DataFrame({
        'id': test['id'],
        'num_sold': np.ceil(oof_preds_inv)
    })

    submission.to_csv('/kaggle/working/submission.csv', index=False)
    return submission

build_submission()

