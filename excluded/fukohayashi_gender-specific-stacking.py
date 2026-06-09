import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
import optuna

import logging
logging.getLogger('optuna').setLevel(logging.ERROR)

import warnings
warnings.simplefilter("ignore")
pd.set_option('display.max_colwidth', None)


test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv' )


train_df


y_col = 'Calories'


train_df.columns


train_df.describe()


train_df.info()


test_df.info()


numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != 'Sex']

n_cols = 2
n_rows = int(np.ceil(len(numeric_cols) / n_cols))

plt.figure(figsize=(12, 4 * n_rows))

for i, col in enumerate(numeric_cols):
    plt.subplot(n_rows, n_cols, i + 1)
    
    sns.kdeplot(data=train_df, x=col, hue='Sex', fill=True, common_norm=False, alpha=0.3)
    
    plt.title(f'Distribution of {col} by Sex')
    plt.xlabel(col)
    plt.ylabel('Density')

plt.tight_layout()
plt.show()


def df_pre(df):
    le = LabelEncoder()
    # df['Sex'] = le.fit_transform(df['Sex'])
    df['duration_cross_body_temp'] = df['Duration'] * df['Body_Temp']
    df['duration_cross_weight'] = df['Duration'] * df['Weight']

    # height_bins = range(120, 211, 10) 
    # height_labels = [i for i in range(120, 210, 10)]
    # df['HeightGroup'] = pd.cut(df['Height'], bins=height_bins, labels=height_labels, right=False)
    
    # median_weights = df.groupby('HeightGroup')['Weight'].median()
    
    # df['Weight_Median_by_HeightGroup'] = df['HeightGroup'].map(median_weights).astype(float)
    # df['Weight_Median_Diff'] = df['Weight'] / df['Weight_Median_by_HeightGroup']

    bins = range(0, 101, 10)
    labels = [i for i in range(0, 100, 10)]
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    median_rate = df.groupby('AgeGroup')['Heart_Rate'].median()
    
    df['Rate_Median_by_AgeGroup'] = df['AgeGroup'].map(median_rate).astype(float)
    df['HeartRate_Median_Diff'] = df['Heart_Rate'] / df['Rate_Median_by_AgeGroup']

    df['HeartRate_per_min'] = df['Heart_Rate'] / df['Duration']
    
    df = df.drop(['Rate_Median_by_AgeGroup', 'AgeGroup'], axis = 1)
    return df


def rmsle_score(y, pred):
    y = np.expm1(y)
    pred = np.expm1(pred)
    y = np.maximum(0, y)
    pred = np.maximum(0, pred)
    return np.sqrt(mean_squared_log_error(y, pred))


seed = 42
n_splits = 10


train_df = df_pre(train_df) 
test_df = df_pre(test_df)


test_id = test_df["id"]


all_predictions = []


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
import optuna

base_model_count = 3

def get_lgb_model():
    params = {
        'n_estimators':1000,
        'learning_rate':0.05,
        'num_leaves':31,
        'random_state':seed,
        'objective':'regression',
        'metric':'rmse',
        'verbose':-1
        }
    return lgb.LGBMRegressor(**params)

def get_xgb_model():
    params = {
        'n_estimators':1000,
        'learning_rate':0.05,
        'max_depth':6,
        'random_state':seed,
        'objective':'reg:squarederror',
        'verbosity':0
        }
    return xgb.XGBRegressor(**params)

def get_cat_model():
    params = {
        'iterations':1000,
        'learning_rate':0.05,
        'depth':6,
        'random_seed':seed,
        'verbose':0,
        'loss_function':'RMSE'
        }
    return CatBoostRegressor(**params)
        


# def get_nn_model():
#     params = {
#         'hidden_layer_sizes':(100,), 
#         'max_iter':1000, 
#         'random_state':seed
#         }
#     return MLPRegressor(**params)

# def get_linear_model():
#     return Ridge(alpha=1.0, random_state=seed)


for sex_value in ['female', 'male']:
    print(f"\n--- Processing for Sex = {sex_value} ---")
    train_sex = train_df[train_df['Sex'] == sex_value]
    test_sex = test_df[test_df['Sex'] == sex_value]

    y = np.log1p(train_sex[y_col])
    X = train_sex.drop([y_col, 'id', 'Sex'], axis=1)
    test_X = test_sex.drop(['id', 'Sex'], axis=1)

    oof_preds = np.zeros((X.shape[0], base_model_count))
    test_preds = np.zeros((test_X.shape[0], base_model_count))

    base_models = [get_lgb_model(), get_xgb_model(), get_cat_model()]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for m_idx, model in enumerate(base_models):
        print(f"Training base model {m_idx + 1}/{base_model_count}...")
        fold_test_preds = np.zeros((test_X.shape[0], n_splits))
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            x_tr, x_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
            model.fit(x_tr, y_tr)
            val_pred = model.predict(x_val)
            oof_preds[val_idx, m_idx] = model.predict(x_val)
            fold_test_preds[:, fold] = model.predict(test_X)

            fold_rmsle = rmsle_score(y_val, val_pred)
            print(f"  Fold {fold + 1} RMSLE: {fold_rmsle:.5f}")
            
        test_preds[:, m_idx] = fold_test_preds.mean(axis=1)


    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'num_leaves': trial.suggest_int('num_leaves', 20, 64),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'random_state': seed,
            'verbose': -1
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(oof_preds, y)
        preds = model.predict(oof_preds)
        return rmsle_score(y, preds)

    print("Tuning meta-model with Optuna...")
    study = optuna.create_study(direction='minimize')
    best_params = study.optimize(objective, n_trials=30, n_jobs=-1)
    

    meta_model = lgb.LGBMRegressor(best_params)
    meta_model.fit(oof_preds, y)

    final_preds_log = meta_model.predict(test_preds)
    final_preds = np.expm1(final_preds_log)

    df_pred = pd.DataFrame({
        'id': test_sex['id'].values,
        y_col: final_preds
    })
    all_predictions.append(df_pred)

submission = pd.concat(all_predictions).sort_values("id")
submission.to_csv("submission.csv", index=False)




submission

