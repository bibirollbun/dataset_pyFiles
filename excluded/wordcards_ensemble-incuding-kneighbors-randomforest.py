from IPython.display import display, clear_output
import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import polars as pl

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor, early_stopping
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from cuml.neighbors import KNeighborsRegressor


PATH = '../input/playground-series-s5e10/'
train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'test.csv')
sub_df = pd.read_csv(PATH + 'sample_submission.csv')

y = train['accident_risk'].copy()
stratify1 = train['speed_limit']
stratify2 = train['lighting']


def create_dic(col):
    unq = list(train[col].unique())
    return dict(zip(unq, [str(v) for v in range(len(unq))]))

road_dic = create_dic('road_type')
lighting_dic = create_dic('lighting')
weather_dic = create_dic('weather')
time_dic = create_dic('time_of_day')

in_cols = test.drop('id', axis=1).columns
cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']

# For GBDTs
dics = [road_dic, lighting_dic, weather_dic, time_dic]

def base_encoder(input_df):
    out_df = input_df[in_cols].copy()
    for col, dic in zip(cat_cols, dics):
        out_df[col] = input_df[col].map(dic).astype('category')

    return out_df

x = base_encoder(train)
test_x = base_encoder(test)
feature_name = x.columns.to_list()


# Simple encoding for KNeighbors
in_cols2 = ['curvature', 'speed_limit', 'lighting', 'weather', 'num_reported_accidents']

def kn_encoder():
    all_df = pd.concat([train[in_cols2], test[in_cols2]])
    all_df['higher_speed_limit'] = (all_df['speed_limit'] >= 60).astype(int)
    all_df['night_lighting'] = (all_df['lighting'] == 'night').astype(int)
    all_df['clear_weather'] = (all_df['weather'] == 'clear').astype(int)
    all_df['many_reported_accidents'] = (all_df['num_reported_accidents'] >= 3).astype(int)

    all_df.drop(['speed_limit', 'lighting', 'weather', 'num_reported_accidents'], axis=1, inplace=True)

    x = all_df[:len(train)]
    test_x = all_df[len(train):].reset_index(drop=True)

    return x, test_x

x_kn, test_x_kn = kn_encoder()


# One Hot Encoding for other scikit-learn models
oh_cols = ['road_type', 'num_lanes', 'speed_limit', 'lighting', 'weather', 'time_of_day', 'num_reported_accidents']

def onehot_encoder():
    oh_df = pd.get_dummies(pd.concat([train[in_cols], test[in_cols]]), columns=oh_cols, drop_first=True)

    x = oh_df[:len(train)]
    test_x = oh_df[len(train):].reset_index(drop=True)

    return x, test_x

x_oh, test_x_oh = onehot_encoder()

x.shape, x_kn.shape, x_oh.shape


class BaseWrapper:
    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def get_feature_importances(self):
        return None

    def get_feature_importance_df(self, fold_id):
        fi = self.get_feature_importances()
        if fi is None:
            return None
        return pl.DataFrame({
            'feature': self.feature_names,
            'importance': fi,
            'fold': fold_id
        })


class GBDTWrapper(BaseWrapper):
    def __init__(self, model_type:str, seed:int, params:dict, feature_names=None, cat_columns=None):
        self.model_type = model_type
        self.seed = seed
        self.params = params
        self.feature_names = feature_names
        self.cat_columns = cat_columns

        if model_type == 'LGB':
            self.model = LGBMRegressor(
                n_estimators=10000,
                importance_type='gain',
                random_state=self.seed,
                verbose=-1,
                **self.params)

        elif model_type == 'XGB':
            self.model = XGBRegressor(
                n_estimators=10000,
                tree_method='gpu_hist',
                enable_categorical=True,
                early_stopping_rounds=100,
                random_state=self.seed,
                **self.params)

        elif model_type == 'Cat':
            self.model = CatBoostRegressor(
                iterations=10000,
                task_type='GPU',
                random_seed=self.seed,
                verbose=False,
                **self.params)

        else:
            raise ValueError("Invalid model type. Please choose 'LGB', 'XGB' or 'Cat'")

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        if self.model_type == 'LGB':
            self.model.fit(
                x_train, y_train,
                eval_set=(x_valid, y_valid),
                feature_name=self.feature_names,
                categorical_feature=self.cat_columns,
                callbacks=[early_stopping(stopping_rounds=100, verbose=False)]
            )

        elif self.model_type == 'XGB':
            self.model.fit(
                x_train, y_train,
                eval_set=[(x_valid, y_valid)],
                verbose=False
            )

        elif self.model_type == 'Cat':
            self.model.fit(
                x_train, y_train,
                eval_set=(x_valid, y_valid),
                cat_features=self.cat_columns,
                use_best_model=True
            )

    def predict(self, x):
        return self.model.predict(x)

    def get_feature_importances(self):
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None

    def get_best_iteration(self):
        if self.model_type == 'XGB':
            return self.model.best_iteration
        elif self.model_type == 'Cat':
            return self.model.get_best_iteration()
        else:
            return self.model.best_iteration_


class SklearnWrapper(BaseWrapper):
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)



def run_cv(model, x, test_x, y, folds, seed, stratify):
    oof = np.zeros(len(x))
    pred = np.zeros(len(test_x))
    fi_df = pl.DataFrame()

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    for i, (train_idx, valid_idx) in enumerate(skf.split(x, stratify)):
        x_train, y_train = x.loc[train_idx], y[train_idx]
        x_valid, y_valid = x.loc[valid_idx], y[valid_idx]

        model.fit(x_train, y_train, x_valid, y_valid)
        oof[valid_idx] = model.predict(x_valid)
        pred += model.predict(test_x) / folds

        fold_rmse = np.sqrt(mean_squared_error(y_valid, oof[valid_idx]))

        msg = f'fold {i} RMSE: {fold_rmse:.5f}'
        if hasattr(model, "get_best_iteration"):
            try:
                best_it = model.get_best_iteration()
                if best_it is not None:
                    msg += f" @{best_it}"
            except Exception:
                pass
        print(msg)

        # feature importance がある場合のみ
        fi = model.get_feature_importance_df(i)
        if fi is not None:
            fi_df = pl.concat([fi_df, fi], how='vertical')

    tot_rmse = np.sqrt(mean_squared_error(y, oof))
    print(f'\ntotal RMSE: {tot_rmse:.5f}\n')

    return oof, pred, fi_df if len(fi_df) > 0 else None


LGB_params = {'max_depth':5, 'colsample_bytree':0.8}
XGB_params = {'max_depth': 8, 'colsample_bytree': 0.8, 'learning_rate': 0.04}
Cat_params = {'learning_rate': 0.1}


seed = 0
model = GBDTWrapper('LGB', seed, LGB_params, feature_names=feature_name, cat_columns=cat_cols)
oof_L0, pred_L0, fi_L0 = run_cv(model, x, test_x, y, folds=5, seed=seed, stratify=stratify1)


seed=42
model = GBDTWrapper('LGB', seed, LGB_params, feature_names=feature_name, cat_columns=cat_cols)
oof_L1, pred_L1, fi_L1 = run_cv(model, x, test_x, y, folds=5, seed=seed, stratify=stratify2)


seed = 27
model = GBDTWrapper('XGB', seed, XGB_params, feature_names=feature_name, cat_columns=cat_cols)
oof_X0, pred_X0, fi_X0 = run_cv(model, x, test_x, y, folds=5, seed=seed, stratify=stratify1)


seed=36
model = GBDTWrapper('XGB', seed, XGB_params, feature_names=feature_name, cat_columns=cat_cols)
oof_X1, pred_X1, fi_X1 = run_cv(model, x, test_x, y, folds=5, seed=seed, stratify=stratify2)


%%time
seed=9
model = GBDTWrapper('Cat', seed, Cat_params, feature_names=feature_name, cat_columns=cat_cols)
oof_C0, pred_C0, fi_C0 = run_cv(model, x, test_x, y, folds=5, seed=seed, stratify=stratify1)


model = SklearnWrapper(LinearRegression())
oof_LR, pred_LR, _ = run_cv(model, x_oh, test_x_oh, y, folds=5, seed=22, stratify=stratify2)


%%time
model = SklearnWrapper(RandomForestRegressor(max_features=0.4, n_jobs=-1))
oof_RF, pred_RF, _ = run_cv(model, x_oh, test_x_oh, y, folds=5, seed=73, stratify=stratify1)


model = SklearnWrapper(KNeighborsRegressor(n_neighbors=128))
oof_KN, pred_KN, _ = run_cv(model, x_kn, test_x_kn, y, folds=5, seed=19, stratify=stratify2)


oof_df = pd.DataFrame({'LGB1': oof_L0, 'LGB2': oof_L1, 'XGB1': oof_X0, 'XGB2': oof_X1,
                       'Cat': oof_C0, 'KN': oof_KN, 'RF':oof_RF, 'LR': oof_LR})
pred_df = pd.DataFrame({'LGB1': pred_L0, 'LGB2': pred_L1, 'XGB1': pred_X0, 'XGB2': pred_X1,
                       'Cat': pred_C0, 'KN': pred_KN, 'RF':pred_RF, 'LR': pred_LR})


N_FOLDS = 5
oof = np.zeros(len(train))
pred = np.zeros(len(test))
rmses = []
coefs = []
ridge = Ridge(positive=True, random_state=55)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)

for i, (train_idx, valid_idx) in enumerate(skf.split(oof_df, stratify1)):
    x_train, y_train = oof_df.loc[train_idx], y[train_idx]
    x_valid, y_valid = oof_df.loc[valid_idx], y[valid_idx]

    ridge.fit(x_train, y_train)

    oof[valid_idx] = ridge.predict(x_valid)
    pred += ridge.predict(pred_df) / N_FOLDS
    coefs.append(ridge.coef_)

    fold_rmse = np.sqrt(mean_squared_error(y_valid, oof[valid_idx]))
    rmses.append(fold_rmse)

tot_rmse = np.sqrt(mean_squared_error(y, oof))
print(f'Total RMSE: {tot_rmse:.5f}\n')

print("RMSE and Ridge weights by folds")
display(
    pd.concat([pd.DataFrame({'Fold': [f for f in range(1,N_FOLDS+1)], 'RMSE': rmses}),
           pd.DataFrame(coefs).rename(
               columns={i: oof_df.columns[i] for i in range(len(oof_df.columns))})],
          axis=1).set_index('Fold')
)


pl.DataFrame({'id':test['id'], 'accident_risk': pred}).write_csv('submission.csv')


fig, ax = plt.subplots(1, 3, sharey=True, figsize=(12, 5))
plt.suptitle('feature importances')
titles = ['LGB', 'XGB', 'Cat']
_order = fi_L0.group_by('feature').agg(pl.col('importance').mean().alias('MU')
    ).sort('MU', descending=True)['feature'].to_list()

for i, _df in enumerate([fi_L0, fi_X0, fi_C0]):
    sns.boxenplot(y='feature', x='importance', data=_df.to_pandas(), orient='h', order=_order, ax=ax[i])
    ax[i].set_ylabel('')
    ax[i].set_xticklabels([])
    ax[i].set_title(titles[i])
plt.tight_layout()

