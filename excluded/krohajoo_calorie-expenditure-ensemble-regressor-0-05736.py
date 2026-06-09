import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold

import time
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')


def root_mean_squared_log_error(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


root = Path('/kaggle/input/playground-series-s5e5')

train = pd.read_csv(root / 'train.csv')
test = pd.read_csv(root / 'test.csv')
submission = pd.read_csv(root / 'sample_submission.csv')


def infotable(df):
    return pd.concat([
        df.dtypes.rename('dtype'),
        df.describe().T,
        df.isna().sum().rename('null'),
        df.nunique().rename('unique'),
        df.mode().iloc[0].rename('most-freq')],
        axis=1
    )

    
infotable(train).style.format(precision=2)


train.sample(3)


cat_columns = ['Sex']
num_columns = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']


def enumerate_steps(xs, start=0, step=1):
    for x in xs:
        yield (start, x)
        start += step


nrows, ncols = 4, 4

_, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
ax = ax.flatten()

for i, col in enumerate_steps(num_columns, step=2):
    bins = train[col].nunique() // 10 if train[col].nunique() > 100 else train[col].nunique() // 3   # prevent 'Calories' to show precise
    sns.histplot(train, x=col, ax=ax[i], bins=bins, kde=True)
    ax[i].grid()
    ax[i].set_title(f'Histogram - {col}')
    
    sns.boxplot(train, x=col, ax=ax[i+1])
    ax[i+1].grid()
    ax[i+1].set_title(f'Boxplot - {col}')


for x in ax[(ncols * nrows - 2):]:
    x.set_visible(False)
    
plt.tight_layout()


_, ax = plt.subplots(nrows=2, ncols=4, figsize=(16, 6))
ax = ax.flatten()

for i, col in enumerate(num_columns):
    sns.violinplot(train, x='Sex', y=col, ax=ax[i])
    ax[i].set_title(f'{col} distribution by Sex')

ax[7].set_visible(False)
plt.tight_layout()


palette = sns.color_palette("crest")


def to_string(interval):
    return f'{interval.left} - {interval.right}'
    

calories = pd.cut(train['Calories'], bins=5).map(to_string)
calories = pd.concat([train.drop('Calories', axis=1), calories.rename('C_bins')], axis=1)

plt.figure(figsize=(5, 5))

sns.jointplot(
    calories, 
    x='Heart_Rate', 
    y='Body_Temp', 
    hue='C_bins', 
    palette=palette
)

plt.suptitle('Heart_Rate vs. Body_Temp by Calories (binned)')
plt.tight_layout()
plt.show()


duration = pd.cut(train['Duration'], bins=5).map(to_string)
duration = pd.concat([train.drop('Duration', axis=1), duration.rename('D_bins')], axis=1)

plt.figure(figsize=(5, 5))

sns.jointplot(
    duration, 
    x='Heart_Rate', 
    y='Body_Temp', 
    hue='D_bins', 
    palette=palette
)

plt.suptitle('Heart_Rate vs. Body_Temp by Duration')
plt.tight_layout()
plt.show()


plt.figure(figsize=(5, 5))

sns.jointplot(
    train, 
    x='Height', 
    y='Weight', 
    hue='Sex', 
    palette=palette
)

plt.suptitle('Height vs. Weight by Sex')
plt.tight_layout()
plt.show()


plt.figure(figsize=(5, 5))

sns.jointplot(train, x='Duration', y='Calories', hue='Sex', palette=palette)

plt.suptitle('Duration vs. Calories by Sex')
plt.tight_layout()
plt.show()


X, y = train.drop('Calories', axis=1), train['Calories']


def preprocess(df, inplace=False):
    if not inplace:
        df = df.copy()
        
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Temp_Deviation'] = df['Body_Temp'] - 37.0
    df['W_H_ration'] = df['Weight'] / df['Height']

    df = pd.get_dummies(df, columns=['Sex'], drop_first=True)

    df = df.drop(['id'], axis=1)
    return df


X = preprocess(X)
X_test = preprocess(test)


# Scale data
exclude = ['W_H_ratio', 'Sex_male']
to_scale = X.columns.difference(exclude)

scaler = StandardScaler()
X[to_scale] = scaler.fit_transform(X[to_scale])
X_test[to_scale] = scaler.transform(X_test[to_scale])


class Trainer:
    def __init__(self, model, params, n_splits=5, random_state=42):
        self.model = model
        self.params = params
        self.n_splits = n_splits
        self.random_state = random_state
        self.name = self.model.__name__

    def fit(self, X, y, test):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        oof_preds = np.zeros(len(X))
        test_preds = np.zeros(len(test))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            y_train = np.log1p(y_train)
            y_val = np.log1p(y_val)

            start = time.time()

            model = self.model(**self.params)
            model.fit(X_train, y_train)

            val_preds = model.predict(X_val)
            rmsle = root_mean_squared_log_error(np.expm1(y_val), np.expm1(val_preds))
            oof_preds[val_idx] = np.expm1(val_preds)
            test_preds += np.expm1(model.predict(test))

            end = time.time()
            print(f'Fold {fold+1} / {self.n_splits} (Runtime: {end - start:1f}s)')
            print(f'RMSLE: {rmsle:.5f}\n--------------')

        test_preds /= self.n_splits
        return oof_preds, test_preds


lgbm_params = {
    "boosting_type": "gbdt",
    "colsample_bytree": 0.50,
    "learning_rate": 0.046,
    "min_child_samples": 39,
    "min_child_weight": 0.58,
    "n_estimators": 10000,
    "n_jobs": -1,
    "num_leaves": 94,
    "random_state": 42,
    "reg_alpha": 13.4,
    "reg_lambda": 3.5,
    "subsample": 0.86,
    "verbose": -1
}


xgb_params = {
    "colsample_bylevel": 0.56,
    "colsample_bynode": 1.0,
    "colsample_bytree": 0.66,
    "eval_metric": "rmse",
    "gamma": 6.9,
    "learning_rate": 0.04,
    "max_depth": 79,
    "max_leaves": 94,
    "min_child_weight": 54,
    "n_estimators": 10000,
    "n_jobs": -1,
    "random_state": 42,
    "reg_alpha": 40.5,
    "reg_lambda": 50.5,
    "subsample": 0.99,
    "verbosity": 0
}

ridge_params = {
    'alpha': 1.0,
    'max_iter': 50000
}


cb_params = {
    "border_count": 197,
    "colsample_bylevel": 0.55,
    "depth": 10,
    "eval_metric": "RMSE",
    "iterations": 5000,
    "l2_leaf_reg": 0.85,
    "learning_rate": 0.05,
    "min_child_samples": 248,
    "random_state": 42,
    "random_strength": 0.27,
    "verbose": False
}

rf_params = {
    'n_estimators': 75,
    'min_samples_split': 3
}


n_splits = 5
random_state = 42

lgbm_trainer = Trainer(
    LGBMRegressor,
    lgbm_params, 
    n_splits=n_splits,
    random_state=random_state
)


oof_lgbm, test_lgbm = lgbm_trainer.fit(X, y, X_test)


xgb_trainer = Trainer(
    XGBRegressor,
    xgb_params,
    n_splits=n_splits,
    random_state=random_state
)

oof_xgb, test_xgb = xgb_trainer.fit(X, y, X_test)


ridge_trainer = Trainer(
    Ridge,
    ridge_params,
    n_splits=n_splits,
    random_state=random_state
)

oof_ridge, test_ridge = ridge_trainer.fit(X, y, X_test)


rf_trainer = Trainer(
    RandomForestRegressor,
    rf_params,
    n_splits=n_splits,
    random_state=random_state
)

oof_rf, test_rf = rf_trainer.fit(X, y, X_test)


cb_trainer = Trainer(
    CatBoostRegressor,
    cb_params,
    n_splits=n_splits,
    random_state=random_state
)

oof_cb, test_cb = cb_trainer.fit(X, y, X_test)


from scipy.optimize import minimize


class EnsembleRegressor:
    def __init__(self, preds, y_true):
        self.models = list(preds.keys())
        self.oof_preds = np.stack(list(preds.values()))
        self.y_true = y_true
        self.bounds = [(0,1)] * self.oof_preds.shape[0]                          # Ensure each model get minimum weight of 0.015: maybe try different values
        self.constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
        
        # call fit-method to setup weights
        self.fit()

    def fit(self):
        weights = [1/self.oof_preds.shape[0]] * self.oof_preds.shape[0]
        result = minimize(self.loss_fn, weights,
                          bounds=self.bounds, constraints=self.constraints)
        self.weights = result.x

    def loss_fn(self, weights):
        pred = np.zeros(self.oof_preds.shape[1])
        for i in range(self.oof_preds.shape[0]):
            pred += weights[i] * self.oof_preds[i]
        rmsle = root_mean_squared_log_error(self.y_true, pred)
        # l2_reg = 0.1 * np.sum(np.square(weights))                 # L2-regularization to prevent extreme weighting
        return rmsle   # + l2_reg
        
    def predict(self, oof_preds):
        pred = np.zeros(oof_preds.shape[1])
        for i in range(oof_preds.shape[0]):
            pred += self.weights[i] * oof_preds[i]
        return pred


models = {
    'lgbm': oof_lgbm,
    'xgb': oof_xgb,
    'ridge': oof_ridge,
    'catboost': oof_cb,
    'random_forest': oof_rf
}


ensemble = EnsembleRegressor(models, y)


final_pred = ensemble.predict(np.stack(list(models.values())))
final_rmsle = root_mean_squared_log_error(final_pred, y)

print(f'final val rmsle: {final_rmsle}')


import math

def show_weights(models, weights):
    plt.figure(figsize=(6, 3))
    ax = sns.barplot(y=list(models), x=weights)
    for bar in ax.containers:
        labels = [f'{round(i, 4) * 100}%' for i in bar.datavalues]
        ax.bar_label(bar, labels=labels)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_xlabel('Model')
    ax.set_ylabel('Weight')
    plt.tight_layout()
    plt.show()
    


show_weights(ensemble.models, 
            ensemble.weights)


test_preds = np.stack([
    test_lgbm,
    test_xgb,
    test_ridge,
    test_cb,
    test_rf
])

ensemble_preds = ensemble.predict(test_preds)


submission['Calories'] = ensemble_preds
submission.to_csv('submission.csv', index=False)

