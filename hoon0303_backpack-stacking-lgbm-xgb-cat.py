from cuml.preprocessing import TargetEncoder
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import polars as pl

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)


dict_fen = {'Material':'NaN','Style':'NaN','Brand':'NaN','Size':'NaN','Waterproof':'NaN','Color':'NaN','Laptop Compartment':'NaN'}


def feh(df):
    df = df.fillna(dict_fen)

    map_size       = {'Small':1.1,'Medium':1.2,'Large':1.3,'NaN':0}
    map_brand      = {'Jansport':1.1,'Adidas':1.2,'Nike':1.3,'Puma':1.4,'Under Armour':1.5,'NaN':0}
    map_color      = {'Black':1.1,'Green':1.2,'Red':1.3,'Blue':1.4,'Gray':1.05,'Pink':1.5,'NaN':0}
    map_style      = {'Messenger':1.1,'Backpack':1.2,'Tote':1.3,'NaN':0}
    map_material   = {'Polyester':1.1,'Leather':1.2,'Nylon':1.3,'Canvas':1.4,'NaN':0}
    map_waterproof = {'Yes':1.1,'No':1.0,'NaN':0}
    map_laptop     = {'Yes':1.1,'No':1.0,'NaN':0}
    
    df['Size_map']        = df['Size'].map(map_size)
    df['Brand_map']       = df['Brand'].map(map_brand)
    df['Color_map']       = df['Color'].map(map_color)
    df['Style_map']       = df['Style'].map(map_style)
    df['Material_map']    = df['Material'].map(map_material)
    df['Waterproof_map']  = df['Waterproof'].map(map_waterproof)
    df['Laptop_map']      = df['Laptop Compartment'].map(map_laptop)
    df['Compartments_map']= df['Compartments'].apply(lambda x: x/1.1)

    df = df.rename(columns={'Size_map':'x1', 'Brand_map':'x2', 'Color_map':'x3', 
                            'Style_map':'x4', 'Material_map':'x5', 'Waterproof_map':'x6', 
                            'Laptop_map':'x7', 'Compartments_map':'x8'}) 

    polar_df = pl.from_pandas(df)
    polar_df = polar_df.with_columns(
        _2_1=((pl.col('x1')-pl.col('x3'))**2 + (pl.col('x2')-pl.col('x4'))**2).sqrt(),
        _2_2=((pl.col('x1')-pl.col('x5'))**2 + (pl.col('x2')-pl.col('x6'))**2).sqrt(),
        _2_3=((pl.col('x1')-pl.col('x7'))**2 + (pl.col('x2')-pl.col('x8'))**2).sqrt(),
        _3_1=((pl.col('x1')-pl.col('x4'))**2 + (pl.col('x2')-pl.col('x5'))**2 + (pl.col('x3')-pl.col('x6'))**2).sqrt(),
        _3_2=((pl.col('x1')-pl.col('x7'))**2 + (pl.col('x2')-pl.col('x8'))**2).sqrt(),
        _3_3=((pl.col('x4')-pl.col('x7'))**2 + (pl.col('x5')-pl.col('x8'))**2).sqrt(),
        _4_1=((pl.col('x1')-pl.col('x5'))**2 + (pl.col('x2')-pl.col('x6'))**2 + 
              (pl.col('x3')-pl.col('x7'))**2 + (pl.col('x4')-pl.col('x8'))**2).sqrt()
    )
    df = polar_df.to_pandas()
    return df


df_test  = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv", index_col='id')
df_train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv", index_col='id')
df_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col='id')



df_train = pd.concat([df_train, df_extra], axis=0, ignore_index=True)

df_train = feh(df_train)
df_test  = feh(df_test)

target = "Price"
features = [col for col in df_train.columns if col != target]
CATS = ['Material', 'Style', 'Brand', 'Size', 'Waterproof', 'Color', 'Laptop Compartment']

RANDOM_SEED = 42
N_ESTIMATORS = 3500

estimators = [
    ('lgbm', LGBMRegressor(
        random_state=RANDOM_SEED,
        n_estimators=N_ESTIMATORS,
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        verbose=-1
    )),
    ('xgb', XGBRegressor(
        random_state=RANDOM_SEED,
        n_estimators=N_ESTIMATORS,
        tree_method='gpu_hist',
        enable_categorical=True,
        predictor='gpu_predictor',
        device='cuda',
        verbosity=0
    )),
    ('catboost', CatBoostRegressor(
        random_seed=RANDOM_SEED,
        iterations=N_ESTIMATORS,
        task_type='GPU',
        devices='0',
        verbose=0,
        cat_features=CATS
    ))
]

meta_model = Ridge(alpha=0.1)


X_train, y_train = df_train[features], df_train[target]
X_test = df_test[features]

TE = TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')
for col in features:
    TE.fit(X_train[col], y_train)
    X_train[f"TE_{col}"] = TE.transform(X_train[col])
    X_test [f"TE_{col}"] = TE.transform(X_test[col])

X_train[CATS] = X_train[CATS].fillna('--').astype('category')
X_test [CATS] = X_test [CATS].fillna('--').astype('category')

all_features = features + [f"TE_{col}" for col in features]

stacking_model = StackingRegressor(estimators=estimators, final_estimator=meta_model)
stacking_model.fit(X_train[all_features], y_train)

test_preds = stacking_model.predict(X_test[all_features])


sub1 = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")

sub1['Price'] = test_preds 
sub1.to_csv("stacking.csv", index=False)

sub1.head()


sub2 = pd.read_csv("/kaggle/input/feature-engineering-with-rapids-lb-38-847/submission_v1.csv")


df_ensemble = sub1.copy()
df_ensemble.iloc[:, 1:] = (sub1.iloc[:, 1:] * 0.4) + (sub2.iloc[:, 1:] * 0.6)

df_ensemble.to_csv("submission.csv", index=False)

