!pip install scikit-learn==1.6.1 


!pip install -q hillclimbers


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd 
import lightgbm as lgb
from xgboost import XGBRegressor
from xgboost import XGBRFRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from hillclimbers import climb_hill, partial
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv").drop(['id'],axis=1) 
print("Train shape:",train.shape)

test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv").drop(['id'],axis=1) 
print("Test shape:", test.shape ) 

oof = {}
test_pred = {} 

FOLDS = 5
TARGET = 'Calories'
train.head(3) 


train.info() 


test.info() 


train.nunique() 


test.nunique() 


train.duplicated().value_counts()


test.duplicated().value_counts() 


FEATURES = [c for c in train.columns if not c in [TARGET]] 
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}") 


train_num = train.copy() 
test_num = test.copy()

train_num['Sex'] = train_num['Sex'].map({'male': 0, 'female':1})
test_num['Sex'] = test_num['Sex'].map({'male': 0, 'female':1}) 

# train_num = considering train dataset and treating all features as numerical
# test_num = considering test dataset and treating all features as numerical


train_cat = train.copy()
test_cat = test.copy()

combined = pd.concat([train_cat,test_cat],axis=0,ignore_index=True) 

for c in FEATURES:

    combined[c],_ = combined[c].factorize()
    combined[c] -= combined[c].min()
    combined[c] = combined[c].astype("int32")
    combined[c] = combined[c].astype("category")

train_cat = combined.iloc[:len(train_cat)].copy()
test_cat = combined.iloc[len(train_cat):].reset_index(drop=True).copy() 

# train_cat = considering train dataset and treating all features as categorical
# test_cat = considering test dataset and treating all features as categorical 


def cross_validation(model, label, df1, df2): 
    
    train_copy = df1.copy()
    test_copy = df2.copy()
 
    oof_model = np.zeros(len(train_copy))
    pred_model = np.zeros(len(test_copy)) 
                             
    FEATURES = [c for c in df1.columns if not c in [TARGET]] 

    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=1) 

    for i, (train_index, valid_index) in enumerate(kf.split(train_copy)):
    
        print("#"*25)
        print(f"### Fold {i+1}")
        print("#"*25)
    
        x_train = df1.loc[train_index,FEATURES].copy()
        y_train = df1.loc[train_index, TARGET]
        x_valid = df1.loc[valid_index,FEATURES].copy()
        y_valid = df1.loc[valid_index, TARGET] 
        x_test = df2[FEATURES].copy() 
    
        # Transform the target variable
        y_train_log = np.log1p(y_train)
        y_val_log = np.log1p(y_valid) 
        
        if label in ['lgbm_num', 'lgbm_cat']:
            model.fit(
                x_train, y_train_log,
                eval_set=[(x_valid, y_val_log)],  
                callbacks=[
                lgb.log_evaluation(500),
                lgb.early_stopping(stopping_rounds=100)
                ],
            )   
        elif label in ['xgb_num', 'cb_num', 'xgb_cat', 'cb_cat']: 
            
            model.fit(
                x_train, y_train_log,
                eval_set=[(x_valid, y_val_log)],  
                verbose=1000,
            )
        else: 
            model.fit(x_train, y_train_log) 
    
    
        # OOF
        predictions_log = model.predict(x_valid)    
        oof_model[valid_index] = predictions_log
        
        rmsle = root_mean_squared_error(y_val_log , predictions_log) 
        print(f'RMSLE = {rmsle:.5f}')
    
    
        # Test
        predictions_log_test = model.predict(x_test)   
        pred_model += predictions_log_test
    

    pred_model /= FOLDS # Compute average test preds

    
    oof[label] = oof_model 
    test_pred[label] = pred_model   
    
    print("#"*25)
    print(f"### OOF Evaluation")
    print("#"*25)
    
    rmsle = root_mean_squared_error(np.log1p(train_copy[TARGET]) , oof_model) 
    print(f'RMSLE = {rmsle:.5f}') 


%%time

model = CatBoostRegressor(
        
            random_seed= 1, 
            task_type="GPU",
            eval_metric='RMSE',
            loss_function='RMSE', 
            score_function= 'NewtonL2',
            bootstrap_type= 'Bernoulli',
            early_stopping_rounds=100,
            cat_features= FEATURES,
            iterations=10000,
            l2_leaf_reg= 6,
            depth=10,

        )
cross_validation(model, 'cb_cat', train_cat, test_cat) 


%%time 

model = XGBRegressor(
        
        device="cuda",
        learning_rate=0.01,
        n_estimators=10000,
        eval_metric="rmse",
        colsample_bytree=0.6,
        enable_categorical=True,
        early_stopping_rounds=100,

    )
cross_validation(model, 'xgb_cat', train_cat, test_cat) 


%%time

model = lgb.LGBMRegressor(
        
        n_iter= 10000,
        max_depth= -1,
        num_leaves= 148,
        learning_rate= 0.01,
        colsample_bytree= 0.6,
        colsample_bynode= 0.8,
        objective= 'rmse',
        metric= 'rmse', 
        verbosity= -1,

    )
cross_validation(model, 'lgbm_cat', train_cat, test_cat) 


%%time
 
model = CatBoostRegressor(
        
            random_seed= 1, 
            task_type="GPU",
            eval_metric='RMSE',
            loss_function='RMSE', 
            score_function= 'NewtonL2',
            bootstrap_type= 'Bernoulli',
            early_stopping_rounds=100,
            iterations=10000,
            l2_leaf_reg= 6,
            depth=11,

        )
cross_validation(model, 'cb_num', train_num, test_num) 


%%time

model = XGBRegressor(
        
        max_depth=8, 
        reg_lambda=3,
        device="cuda",
        learning_rate=0.01,
        n_estimators=10000,
        eval_metric="rmse",
        colsample_bytree=0.7,
        enable_categorical=True,
        early_stopping_rounds=100,
        colsample_bynode=0.8,
        random_seed= 1,
        subsample=0.95,

    )
cross_validation(model, 'xgb_num', train_num, test_num) 


%%time

model = lgb.LGBMRegressor(
        
        n_iter= 10000,
        max_depth= -1,
        num_leaves= 148,
        learning_rate= 0.01,
        colsample_bytree= 0.6,
        colsample_bynode= 0.8,
        objective= 'rmse',
        metric= 'rmse', 
        verbosity= -1,

    )
cross_validation(model, 'lgbm_num', train_num, test_num) 


%%time

model = lgb.LGBMRegressor(
        
        n_iter= 400,
        max_depth= -1,
        boosting= 'dart',
        learning_rate= 0.5,
        colsample_bytree= 0.6,
        objective= 'rmse',
        metric= 'rmse', 
        verbosity= -1,
        drop_rate=0.1, 
        max_drop=50,
        skip_drop=0.5,

    )
cross_validation(model, 'dart_num', train_num, test_num) 


%%time

model = HistGradientBoostingRegressor(
        
        max_iter=10000,
        max_features=0.6,
        learning_rate=0.01,
        n_iter_no_change=100,
        l2_regularization=10,
        max_leaf_nodes=100,
        early_stopping='auto',
        categorical_features='from_dtype',

    )
cross_validation(model, 'hgb_num', train_num, test_num) 


df_oof_et = pd.read_csv("/kaggle/input/flaml-extratrees-0-06114-cv-lb-of-0-05892/oof.csv")['0']
oof['flaml_et'] = np.log1p(df_oof_et)  

df_test_et = pd.read_csv("/kaggle/input/flaml-extratrees-0-06114-cv-lb-of-0-05892/submission.csv")['Calories']
test_pred['flaml_et'] = np.log1p(df_test_et) 


df_oof_rf = pd.read_csv("/kaggle/input/fast-rapids-cuml-randomforest-cv-0-0667/oof.csv")['Calories']
oof['rf'] = df_oof_rf

df_test_rf = pd.read_csv("/kaggle/input/fast-rapids-cuml-randomforest-cv-0-0667/submission.csv")['Calories']
test_pred['rf'] = np.log1p(df_test_rf) 


df_oof_mlp = pd.read_csv("/kaggle/input/neural-net-sklearn-mlp-cv-0-06080-lb-0-05775/oof_nn.csv")['OOF_NN']
oof['mlp'] = df_oof_mlp

df_test_mlp = pd.read_csv("/kaggle/input/neural-net-sklearn-mlp-cv-0-06080-lb-0-05775/submission_nn.csv")['Calories']
test_pred['mlp'] = np.log1p(df_test_mlp)


df_oof_tabnet = pd.read_csv("/kaggle/input/pytorch-tabnet-cv-0-06182-lb-0-05799/oof_tabnet.csv")['Calories']
oof['tabnet'] = np.log1p(df_oof_tabnet)

df_test_tabnet = pd.read_csv("/kaggle/input/pytorch-tabnet-cv-0-06182-lb-0-05799/submission_tabnet.csv")['Calories']
test_pred['tabnet'] = np.log1p(df_test_tabnet)


oof = pd.DataFrame(oof)
test_pred =pd.DataFrame(test_pred) 

train_copy = train.copy() 
train_copy[TARGET] = np.log1p(train_copy[TARGET]) 

hc_test, hc_oof = climb_hill(train=train_copy, target=TARGET, objective='minimize', 
                             eval_metric=partial(root_mean_squared_error),oof_pred_df= oof, 
                             test_pred_df= test_pred,plot_hill=True,plot_hist=False, 
                             precision=0.001,negative_weights=True,return_oof_preds=True) 


%%time

oof['Calories'] = train['Calories'] 

model = Ridge()

cross_validation(model, 'ridge', oof, test_pred)  


oof_hill_climber = np.expm1(pd.Series(hc_oof)) # Inverse transform the predictions
oof_hill_climber[oof_hill_climber < 0] = 0 # Ensure non-negative predictions
oof_hill_climber.to_csv("oof_hill_climbing.csv",index=False) 


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

rmsle(train[TARGET], oof_hill_climber) 


oof_ridge = np.expm1(oof['ridge']) # Inverse transform the predictions
oof_ridge[oof_ridge < 0] = 0 # Ensure non-negative predictions
oof_ridge.to_csv("oof_ridge.csv",index=False) 


rmsle(train[TARGET], oof_ridge) 


sub_df = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
sub_df_hc = sub_df.copy() 

df_hc = np.expm1(hc_test) # Inverse transform the predictions
df_hc[df_hc < 0] = 0 # Ensure non-negative predictions

sub_df_hc[TARGET] = df_hc
sub_df_hc.to_csv("submission_hill_climbing.csv",index=False) 

print("Sub shape:",sub_df_hc.shape)
sub_df_hc.head() 


sub_df_ridge = sub_df.copy() 

test_pred_ridge = np.expm1(test_pred['ridge']) # Inverse transform the predictions
test_pred_ridge[test_pred_ridge < 0] = 0 # Ensure non-negative predictions

sub_df_ridge[TARGET] = test_pred_ridge
sub_df_ridge.to_csv("submission_ridge.csv",index=False)

print("Sub shape:",sub_df_ridge.shape)
sub_df_ridge.head() 

