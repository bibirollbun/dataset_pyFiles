! pip install cir_model pytorch_tabnet ftfy


if 1 == 2:
    source_file_path = '/kaggle/input/yunbase/Yunbase/baseline.py'
    target_file_path = '/kaggle/working/baseline.py'
    with open(source_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    with open(target_file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    !pip install -q --requirement /kaggle/input/yunbase/Yunbase/requirements.txt --no-index --find-links file:/kaggle/input/yunbase/


import polars as pl
import pandas as pd
import numpy as np
import random
from  lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import sys, os, gc

sys.path.append("/kaggle/input/yunbase")
sys.path.append("/kaggle/input/yunbase/Yunbase")
sys.path.append("/kaggle/usr/lib/yunbase_baseline_04_12_2024")
#from baseline import Yunbase
from yunbase_baseline_04_12_2024 import Yunbase 

sys.path.append("/kaggle/input/jane-street-real-time-market-data-forecasting")
import kaggle_evaluation.jane_street_inference_server

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
seed_everything(seed=2025)


yunbase=Yunbase()
data=[]
for i in [6,7,8,9]:
    train=pl.read_parquet(f"/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id={i}/part-0.parquet")
    train=train.to_pandas()
    train['sin_time_id']=np.sin(2*np.pi*train['time_id']/967)
    train['cos_time_id']=np.cos(2*np.pi*train['time_id']/967)
    train['sin_time_id_halfday']=np.sin(2*np.pi*train['time_id']/483)
    train['cos_time_id_halfday']=np.cos(2*np.pi*train['time_id']/483)
    #train=train.fillna(-1)
    train=yunbase.reduce_mem_usage(train,float16_as32=False)
    data.append(train)
train=pd.concat(data)
print(f"train.shape:{train.shape}")
del data
gc.collect()
final_feature=['symbol_id','sin_time_id','cos_time_id','sin_time_id_halfday','cos_time_id_halfday']+[f'feature_0{i}' if i<10 else f'feature_{i}' for i in range(79)]
train=train[['responder_6']+final_feature]
train.head()


lgb_params={"boosting_type": "gbdt","metric": 'rmse',
            'random_state': 2025,  "max_depth": 10,"learning_rate": 0.1,
            "n_estimators": 120,"colsample_bytree": 0.6,"colsample_bynode": 0.6,"verbose": -1,"reg_alpha": 0.2,
            "reg_lambda": 5,"extra_trees":True,'num_leaves':64,"max_bin":255,
            'device':'gpu','gpu_use_dp':True,
            }

cat_params={'task_type':'GPU',
           'random_state':2025,
           'eval_metric'         : 'RMSE',
           'bagging_temperature' : 0.50,
           'iterations'          : 200,
           'learning_rate'       : 0.1,
           'max_depth'           : 12,
           'l2_leaf_reg'         : 1.25,
           'min_data_in_leaf'    : 24,
           'random_strength'     : 0.25, 
           'verbose'             : 0,
          }
xgb_params={'random_state': 2025, 'n_estimators': 125, 
            'learning_rate': 0.1, 'max_depth': 10,
            'reg_alpha': 0.08, 'reg_lambda': 0.8, 
            'subsample': 0.95, 'colsample_bytree': 0.6, 
            'min_child_weight': 3,
            'tree_method':'gpu_hist',
           }

print("lgb")
lgb=LGBMRegressor(**lgb_params)
lgb.fit(train[final_feature].values,train['responder_6'].values)
lgb.booster_.save_model("./lgb.model")

print("cat")
cat=CatBoostRegressor(**cat_params)
cat.fit(train[final_feature].values,train['responder_6'].values)
cat.save_model("./cat.model", pool=None)

print("xgb")
xgb=XGBRegressor(**xgb_params)
xgb.fit(train[final_feature].values,train['responder_6'].values)
xgb.save_model("./xgb.model")


def predict(test,lags):
    global lgb,cat,xgb
    
    predictions = test.select(
        'row_id',
        pl.lit(0.0).alias('responder_6'),
    )
    test=test.to_pandas()
    test['sin_time_id']=np.sin(2*np.pi*test['time_id']/967)
    test['cos_time_id']=np.cos(2*np.pi*test['time_id']/967)
    test['sin_time_id_halfday']=np.sin(2*np.pi*test['time_id']/483)
    test['cos_time_id_halfday']=np.cos(2*np.pi*test['time_id']/483)
    test=test.fillna(-1)
    test=test[final_feature]
    eps=1e-10
    test_preds=0.55*lgb.predict(test)+0.2*cat.predict(test)+0.25*xgb.predict(test)
    test_preds=np.clip(test_preds,-5+eps,5-eps)
    predictions = predictions.with_columns(pl.Series('responder_6', test_preds.ravel()))
    return predictions

inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(predict)

#if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
#    inference_server.serve()
#else:
#    inference_server.run_local_gateway(
#        (
#            '/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet',
#            '/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet',
#        )
#    )

