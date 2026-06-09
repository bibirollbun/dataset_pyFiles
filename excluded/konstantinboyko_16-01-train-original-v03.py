import os, sys
import random
import dill
import polars as pl
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

import warnings
warnings.filterwarnings('ignore')

sys.path.append("/kaggle/input/jane-street-real-time-market-data-forecasting")
import kaggle_evaluation.jane_street_inference_server

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
seed_everything(seed=2025)

def save_as_dill(model_name, model_object, file_ext='.dill'):
    with open(f"./{model_name}{file_ext}", "wb") as file_handle:
        dill.dump(model_object, file_handle, protocol=4)

def custom_metric(y_true,y_pred,weight):
    weighted_r2=1-(np.sum(weight*(y_true-y_pred)**2)/np.sum(weight*y_true**2))
    return weighted_r2
    
print("< read parquet >")
datas=[]
for i in range(7,10):
    train=pl.read_parquet(f"/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id={i}/part-0.parquet")
    train=train.to_pandas()
    datas.append(train)
train=pd.concat(datas)
print(f"train.shape:{train.shape}")

print("< get X,y >")
cols=[f'feature_0{i}' if i<10 else f'feature_{i}' for i in range(79)]
X=train[cols].fillna(3).values
y=train['responder_6'].values

print("< train test split >")
split=1300000#around 10%
weights=train['weight'].values
train_X,train_y,test_X,test_y,train_weight,test_weight=X[:-split],y[:-split],X[-split:],y[-split:],weights[:-split],weights[-split:]
print(f"train_X.shape:{train_X.shape},test_X.shape:{test_X.shape}")

print("< fit and predict >")
model=Ridge()
model.fit(train_X,train_y)
save_as_dill('ridge', model)
train_pred=model.predict(train_X)
test_pred=model.predict(test_X)
print(f"train weighted_r2:{custom_metric(train_y,train_pred,weight=train_weight)}")
print(f"test weighted_r2:{custom_metric(test_y,test_pred,weight=test_weight)}")

def predict(test,lags):
    cols=[f'feature_0{i}' if i<10 else f'feature_{i}' for i in range(79)]
    predictions = test.select(
        'row_id',
        pl.lit(0.0).alias('responder_6'),
    )
    test_preds=model.predict(test[cols].to_pandas().fillna(3).values)
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

