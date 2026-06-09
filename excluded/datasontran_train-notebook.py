
# !pip install mlflow


# import pandas as pd
import numpy as np
import xgboost
path = "/kaggle/working/"




import pandas as pd

df = pd.read_pickle("/kaggle/input/kaggle-predict-future-sales-first-place-solution/abubakar_VVIP/abubakar_VVIP/checkpoint_final_0.84.pkl")  
#pkl is much faster and much lower in memory
df['item_cnt_month'] = df['item_cnt_month'].clip(0,20)
df=df.rename(columns={"item_cnt_month":"item_cnt"})
df=df[df!=np.inf]


import warnings

warnings.filterwarnings("ignore", module="lightgbm")

import lightgbm as lgbm


def fit_booster(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    params=None,
    test_run=False,
    categoricals=[],
    dropcols=[],
    early_stopping=True,
):
    if params is None:
        params = {"learning_rate": 0.1, "subsample_for_bin": 300000, "n_estimators": 50}

    early_stopping_rounds = None
    if early_stopping == True:
        early_stopping_rounds = 30

    if test_run:
        eval_set = [(X_train, y_train)]
    else:
        eval_set = [(X_train, y_train), (X_test, y_test)]

    booster = lgbm.LGBMRegressor(**params)

    categoricals = [c for c in categoricals if c in X_train.columns]

    booster.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        eval_metric=["rmse"],
        verbose=100,
        categorical_feature=categoricals,
        early_stopping_rounds=early_stopping_rounds,
    )

    return booster


params = {
    "num_leaves": 966,
    "cat_smooth": 45.01680827234465,
    "min_child_samples": 27,
    "min_child_weight": 0.021144950289224463,
    "max_bin": 214,
    "learning_rate": 0.01,
    "subsample_for_bin": 300000,
    "min_data_in_bin": 7,
    "colsample_bytree": 0.8,
    "subsample": 0.6,
    "subsample_freq": 5,
    "n_estimators": 750,
}




#designating the categorical features which should be focused on
import pandas as pd
import numpy as np
import lightgbm as lgb


import warnings
warnings.filterwarnings("ignore")

def build_lgb_model(X_train,y_train):
    params = {
        'objective': 'rmse',
        'metric': 'rmse',
        'num_leaves': 1023,
        'min_data_in_leaf':10,
        'feature_fraction':0.7,
        'learning_rate': 0.01,
        'num_rounds': 500,
    #     'early_stopping_rounds': 30,
        'seed': 1
    }
        
    cat_features = ['item_category_id','month','shop_id']
    lgb_train = lgb.Dataset(X_train, y_train)
    model = lgb.train(params=params, train_set=lgb_train,valid_sets=lgb_train,verbose_eval=10,
                     categorical_feature=cat_features)
    return model


keep_from_month = 2  # The first couple of months are dropped because of distortions to their features (e.g. wrong item age)
test_month = 33
dropcols = [
    "shop_id",
    "item_id",
    "new_item",
]  # The features are dropped to reduce overfitting



n_splits=1
SEED=42
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from IPython.display import clear_output
from scipy.optimize import minimize
import numpy as np
from colorama import Fore, Style
from tqdm import tqdm
def root_mean_squared_error(y, y_val):
    mse = mean_squared_error(y, y_val)
    return np.sqrt(mse)

def TrainML(model_class,X,y, test_data):
    SKF = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    train_S = []
    test_S = []
    
    test_preds = np.zeros((len(test_data), n_splits))
    oof_non_rounded = np.zeros(len(y), dtype=float) 
    for fold, (train_idx, test_idx) in enumerate(tqdm(SKF.split(X, y), desc="Training Folds", total=n_splits)):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx] 
        model = clone(model_class)

        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        test_pred = model.predict(test_data)

        oof_non_rounded[test_idx] = y_val_pred
        train_rmse = root_mean_squared_error(y_train, y_train_pred)
        val_rmse = root_mean_squared_error(y_val, y_val_pred)

        train_S.append(train_rmse)
        test_S.append(val_rmse)
        
        test_preds[:, fold] = test_pred
        
        print(f"Fold {fold+1} - Train RMSE: {train_rmse:.4f}, Validation RMSE: {val_rmse:.4f}")
        clear_output(wait=True)

    print(f"Mean Train  --> {np.mean(train_S):.4f}")
    print(f"Mean Validation RMSE ---> {np.mean(test_S):.4f}")
    
    tRMSE = root_mean_squared_error(y, oof_non_rounded)

    print(f"----> || Optimized RMSE SCORE :: {Fore.CYAN}{Style.BRIGHT} {tRMSE:.3f}{Style.RESET_ALL}")
    tpm = test_preds.mean(axis=1)
    return tpm


CatBoost_Params = {
    'learning_rate': 0.05,
    'depth': 6,
    'iterations': 200,
    'l2_leaf_reg': 10,  
}
Light_Params = {
    'learning_rate': 0.046,
    'max_depth': 12,
    'num_leaves': 478,
    'min_data_in_leaf': 13,
    'feature_fraction': 0.893,
    'bagging_fraction': 0.784,
    'bagging_freq': 4,
    'lambda_l1': 10,  # Increased from 6.59
    'lambda_l2': 0.01  # Increased from 2.68e-06,
    # "device": "gpu"
}
XGB_Params = {
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1,  # Increased from 0.1
    'reg_lambda': 5,  # Increased from 1
}


# !mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
# !sudo apt install nvidia-driver-460 nvidia-cuda-toolkit clinfo
# !apt-get update --fix-missing
# !pip install -q  lightgbm==4.3.0 \
#   --config-settings=cmake.define.USE_GPU=ON \
#   --config-settings=cmake.define.OpenCL_INCLUDE_DIR="/usr/local/cuda/include/" \
#   --config-settings=cmake.define.OpenCL_LIBRARY="/usr/local/cuda/lib64/libOpenCL.so";


import xgboost as xgb
preds_arr_lgb=[]
vals_arr_lgb=[]
vals_arr_lgb_84=[]
preds_arr_xgb=[]
vals_arr_xgb=[]
preds_arr_nn=[]
shop_id=[]
item_id=[]
cat_id=[]
month_arr=[]
# from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
print(df.shape)
print(np.sum(df[df["date_block_num"]>33]['item_cnt']))
for i in range(25,35):
    X_train=df[df["date_block_num"]<i].drop(['item_cnt',"date_block_num"],axis=1)
    y_train=df.loc[(df["date_block_num"]<i),'item_cnt']
    X_val=df[df["date_block_num"]==i].drop(['item_cnt',"date_block_num"],axis=1)
    y_val=df.loc[(df["date_block_num"]==i),'item_cnt']
    #########################XGB################################
    # model_name="XGB_iterations_100"+str(i)
    # mlflow.xgboost.autolog(registered_model_name=model_name)
    # SEED=0
    # xgb_model = xgb.XGBRegressor(learning_rate=0.05,max_leaves=800,num_round=1000,n_estimators=100,max_depth=10,early_stopping_rounds=10)
    # xgb_model.fit(X_train,y_train,eval_metric="rmse",eval_set=[(X_train,y_train)])
    # val_pred=xgb_model.predict(X_val).clip(0,20)
    # vals_arr_xgb.append(val_pred)
    ######################################################################
    CatBoost_Model = CatBoostRegressor(**CatBoost_Params,random_seed=SEED, task_type='GPU',verbose=0)
    # val_pred = TrainML(CatBoost_Model,X_train,y_train,X_val)
    model = clone(CatBoost_Model)
    model.fit(X_train,y_train)
    val_pred=model.predict(X_val)
    vals_arr_lgb.append(val_pred)
    #########################LGB###############################
    # Light = LGBMRegressor(**Light_Params, random_state=SEED, verbose=-1, n_estimators=300)
    # val_pred = TrainML(Light,X_train,y_train,X_val)
    # vals_arr_lgb_84.append(val_pred)
    
#     _ = joblib.dump(xgb_model, "xgb_model_"+str(i)+"_.pkl")
#     _ = joblib.dump(lgb_model, "lgb_model"+str(i)+"_.pkl")
#     _ = joblib.dump(lgbooster, "lgbooster"+str(i)+"_.pkl")
    
  
# vals_arr_xgb_series=pd.Series(vals_arr_xgb)
# vals_arr_xgb_series.to_pickle(path+'vals_arr_xgb_84.pkl')
    

vals_arr_lgb_series=pd.Series(vals_arr_lgb)
vals_arr_lgb_series.to_pickle(path+'vals_arr_cast.pkl')

# vals_arr_lgb_series_84s=pd.Series(vals_arr_lgb_84)
# vals_arr_lgb_series_84s.to_pickle(path+'vals_arr_lgbm-25-30.pkl')


