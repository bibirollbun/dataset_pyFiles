import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

import sklearn
sklearn.set_config(transform_output='pandas')

from sklearn.preprocessing import LabelEncoder, FunctionTransformer, OneHotEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin, clone

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from xgboost import XGBRegressor, XGBRFRegressor, DMatrix
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor, early_stopping

from statsmodels.graphics.tsaplots import plot_acf

import category_encoders as ce

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=28, titlepad=10,
       titlecolor='red'
      )

warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', index_col='id')

train['date'] = pd.to_datetime(train['date'], format='%Y-%m-%d')
test['date'] = pd.to_datetime(test['date'], format='%Y-%m-%d')


train.head()


test.head()


train.isna().sum()


train[train['num_sold'].isna()][['country', 'store', 'product']].value_counts()


train = train.dropna().reset_index(drop=True)


train.duplicated().sum()


plt.figure(figsize=(24,6))
train.groupby('date')['num_sold'].sum().plot(xlabel='Date', 
                                             ylabel='Number of Products Sold', 
                                             title='Total Sales Over Time')
plt.grid()
plt.show()


plt.figure(figsize=(24, 6))
sns.lineplot(x=train['date'].dt.year, y=train['num_sold'], hue=train['country'], estimator='sum')
plt.title('Sales Trends by Country (Year-wise)')
plt.xlabel('Year')
plt.ylabel('Number of Products Sold')
plt.legend(title='Country')
plt.grid()
plt.show()


plt.figure(figsize=(24, 6))
sns.lineplot(x=train['date'].dt.year, y=train['num_sold'], hue=train['store'], estimator='sum')
plt.title('Sales Trends by Store-Type (Year-wise)')
plt.xlabel('Year')
plt.ylabel('Number of Products Sold')
plt.legend(title='Store Type')
plt.grid()
plt.show()


plt.figure(figsize=(16, 6))
plot_acf(train.groupby('date')['num_sold'].sum().dropna(), lags=20)
plt.title('Autocorrelation Plot')
plt.show()


def model_trainer(model, X, y, test, n_splits=5, random_state=42, verbose=0, model_name=None):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    print("="*80)
    model_name_ = model[-1].__class__.__name__ if isinstance(model, Pipeline) else model.__class__.__name__
    print(f"Model: {model_name_}")
    print("="*80 + '\n')

    oof_mape = []
    oof_test_preds = np.zeros(len(test))
    oof_train_preds = np.zeros(len(y))
    
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(X)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y[valid_idx]

        if model_name == 'xgb':
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=verbose)
            booster = model.get_booster()
            
            y_pred = booster.predict(DMatrix(X_valid), iteration_range=(0, model.best_iteration+1))
            test_pred = booster.predict(DMatrix(test), iteration_range=(0, model.best_iteration+1))
            oof_train_preds[train_idx] = booster.predict(DMatrix(X_train), iteration_range=(0, model.best_iteration+1))

        elif model_name == 'cat':
            trainPool = Pool(X_train ,y_train)
            testPool = Pool(test)
            validPool = Pool(X_valid, y_valid)

            model.fit(X=trainPool, eval_set=validPool, verbose=verbose, early_stopping_rounds=200)
            y_pred = model.predict(validPool)
            test_pred = model.predict(testPool)
            oof_train_preds[train_idx] = model.predict(Pool(X_train))

        elif model_name == 'lgb':
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='rmse', callbacks=[early_stopping(200, verbose=0)])
            y_pred = model.predict(X_valid, num_iteration=model.best_iteration_)
            test_pred = model.predict(test, num_iteration=model.best_iteration_)
            oof_train_preds[train_idx] = model.predict(X_train, num_iteration=model.best_iteration_)

        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            test_pred = model.predict(test)
            oof_train_preds[train_idx] = model.predict(X_train)

        oof_test_preds += test_pred
        mape = mean_absolute_percentage_error(np.expm1(y_valid), np.expm1(y_pred))
        print(f"Fold {fold+1} --> MAPE: {mape:.4f}")
        oof_mape.append(mape)
    
    print()
    print(f"Average Fold MAPE: {np.mean(oof_mape):.4f} \xb1 {np.std(oof_mape):.4f}")
    return oof_test_preds/n_splits, oof_train_preds



class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame()
        df['day'] = X.date.dt.day
        df['month'] = X.date.dt.month
        df['year'] = X.date.dt.year
        df['quarter'] = X.date.dt.quarter
        df['dayofyear'] = X.date.dt.dayofyear
        df['weekday'] = X.date.dt.weekday
        df['sine_day'] = np.sin(2 * np.pi * df['day'] / 31)
        df['cos_day'] = np.cos(2 * np.pi * df['day'] / 31)
        df['sine_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        df['sine_year'] = np.sin(2 * np.pi * df['year'])
        df['cos_year'] = np.cos(2 * np.pi * df['year'])
        df['sine_quarter'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['cos_quarter'] = np.cos(2 * np.pi * df['quarter'] / 4)
        df['sine_dayofyear'] = np.sin(2 * np.pi * df['dayofyear'] / 366)
        df['cos_dayofyear'] = np.cos(2 * np.pi * df['dayofyear'] / 366)
        df['sine_weekday'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['cos_weekday'] = np.cos(2 * np.pi * df['weekday'] / 7)
        return df


preprocessing = ColumnTransformer([
    ('categorical', 
     OneHotEncoder(handle_unknown='ignore', sparse_output=False),
     ['country', 'store', 'product']),
    ('date', DateTransformer(),['date'])
], remainder='drop')


target = 'num_sold'


X = train.copy()
y = np.log1p(X.pop(target))

X = preprocessing.fit_transform(X)
test = preprocessing.transform(test)


test_preds, train_preds = pd.DataFrame(), pd.DataFrame()


xgb_params = {
    'n_estimators': 3000,
    'learning_rate': 0.00990161328639894,
    'max_depth': 17,
    'min_child_weight': 58,
    'subsample': 0.7373527286687829,
    'colsample_bytree': 0.4544157822113165,
    'gamma': 0.0019767061497068528,
    'reg_alpha': 0.7647218923252306,
    'device': 'cuda',
    'tree_method': 'hist',
    'random_state': 0,
    'early_stopping_rounds': 200
}

xgb_reg = XGBRegressor(**xgb_params)

test_preds['xgb'], train_preds['xgb'] = model_trainer(xgb_reg, X, y, 
                                                      test, 
                                                      random_state=0, verbose=0, model_name='xgb')


cat_params = {
    'n_estimators': 10000,
    'learning_rate': 0.05, 
    'task_type': 'GPU', 
    'verbose': False, 
    'allow_writing_files': False,
}

cat_reg = CatBoostRegressor(**cat_params)

test_preds['cat'], train_preds['cat'] = model_trainer(
    cat_reg,
    X, y, test, random_state=0, model_name='cat'
)


lgb_reg = LGBMRegressor(verbosity=-1, device='gpu',
                        n_estimators=5000, learning_rate=0.1
                       )

test_preds['lgb'], train_preds['lgb'] = model_trainer(
    lgb_reg,
    X, y, test, random_state=101, model_name='lgb'
)


hgb_reg = HistGradientBoostingRegressor(max_iter=1000)

test_preds['hgb'], train_preds['hgb'] = model_trainer(
    hgb_reg, 
    X, y,
    test, random_state=0
)


import optuna

def objective(trial):
    xw = trial.suggest_float('xgb', 0.1, 5)
    cw = trial.suggest_float('cat', 0.1, 5)
    lw = trial.suggest_float('lgb', 0.1, 5)
    hw = trial.suggest_float('hgb', 0.1, 5)

    pred = np.average(train_preds.to_numpy(), weights=[xw, cw, lw, hw], axis=1)

    score = mean_absolute_percentage_error(np.expm1(y), np.expm1(pred))
    return score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=500)

print(study.best_params)


mean_absolute_percentage_error(np.expm1(y), np.expm1(np.mean(train_preds.to_numpy(), axis=1)))


weights = study.best_params

test_pred = np.average(test_preds.to_numpy(), weights=list(weights.values()),axis=1)


sub = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
sub[target] = np.round(np.expm1(test_pred))
sub.to_csv('submission1.csv', index=False)
sub.head()


ext1 = pd.read_csv('/kaggle/input/s5e1-eda-and-linear-regression-baseline/submission.csv')[target].ravel()
ext2 = pd.read_csv('/kaggle/input/no-model-pss5e1/no_model.csv')[target].ravel()
ext3 = pd.read_csv('/kaggle/input/s5e1-previous-years-baseline-no-model/submission.csv')[target].ravel()


final_pred = np.average([ext1, ext2, ext3, sub[target]], weights=[15, 10, 20, 1], axis=0)


sub[target] = np.round(final_pred)
sub.to_csv('submission.csv', index=False)
sub.head()

