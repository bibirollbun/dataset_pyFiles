import catboost as catb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from lightgbm import early_stopping, log_evaluation


import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier , Pool
import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from lightgbm import early_stopping



data = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
data.columns


test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')





cat_features = [
    "gender", "ethnicity", "education_level",
    "income_level", "smoking_status", "employment_status",
]


X = data.drop(columns='diagnosed_diabetes')
y = data['diagnosed_diabetes'].copy()
y.sum()/len(y)


for col in cat_features:
    X[col] = X[col].astype("category")
    test[col] = test[col].astype("category")


X = data.drop(columns='diagnosed_diabetes')
y = data['diagnosed_diabetes'].copy()
y.sum()/len(y)






# Optuna optimization for CatBoost using GPU.
# Objective: maximize ROC-AUC on validation set.


# def objective(trial):

#     params = {
#         "iterations": 5000,
#         "depth": trial.suggest_categorical("depth", [4, 5, 6, 7, 8, 9, 10]),
#         "learning_rate": trial.suggest_categorical("learning_rate", [ 0.03,0.05,0.07,0.1]),
#         "l2_leaf_reg": trial.suggest_categorical("l2_leaf_reg", [18,19, 20,21,22]),
#         "bagging_temperature": trial.suggest_categorical("bagging_temperature", [0.1, 0.5, 1.0, 2.0, 5.0]),
#         "random_strength": trial.suggest_categorical("random_strength", [ 2,2.5, 3]),
#         "border_count": trial.suggest_categorical("border_count", [ 254,400,300]),
#         "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide"]),
        
#         "loss_function": "Logloss",
#         "eval_metric": "AUC",
#         "bootstrap_type": "Bayesian",
#         "task_type": "GPU",     # GPU tuning
#         "devices": "0",
#         "random_seed": 42,
#     }

#     model = CatBoostClassifier(**params)

#     model.fit(
#         train_pool,
#         eval_set=valid_pool,
#         use_best_model=True,
#         verbose=500
#     )

#     preds = model.predict_proba(valid_pool)[:, 1]
#     auc = roc_auc_score(y_test, preds)

#     return auc



# study = optuna.create_study(direction="maximize")

# study.optimize(objective, n_trials=50)

# print("Best AUC:", study.best_value)
# print("Best params:", study.best_params)



# #sgb

# def objective(trial):

#     params = {
#         "n_estimators": 4000,

#         "max_depth": trial.suggest_categorical("max_depth", [3, 4, 5, 6, 7, 8, 9, 10]),
#         "min_child_weight": trial.suggest_categorical("min_child_weight", [1, 2, 3, 4, 5, 6, 8, 10]),
#         "gamma": trial.suggest_categorical("gamma", [0, 0.1, 0.2, 0.5, 1, 2, 5]),

#         "reg_alpha": trial.suggest_categorical("reg_alpha", [0, 0.1, 0.5, 1, 3, 5, 10]),
#         "reg_lambda": trial.suggest_categorical("reg_lambda", [0.1, 0.5, 1, 2, 5, 10, 20]),

#         "learning_rate": trial.suggest_categorical("learning_rate", [0.005, 0.01, 0.02, 0.03, 0.08, 0.1]),

#         "subsample": trial.suggest_categorical("subsample", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
#         "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),

#         "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),

#         "booster": "gbtree",
#         "tree_method": "gpu_hist",
#         "predictor": "gpu_predictor",

#         "objective": "binary:logistic",
#         "eval_metric": "auc",

#         "random_state": 42,
#         "enable_categorical":True
#     }

#     modelX = xgb.XGBClassifier(**params)

#     modelX.fit(X_train , y_train , verbose=500  , eval_set=[(X_test,y_test)])

#     proba = modelX.predict_proba(X_test)[:, 1]
#     auc = roc_auc_score(y_test, proba)

#     return auc     
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100)

# print("Best AUC:", study.best_value)
# print("Best params:", study.best_params)




# Generate Out-of-Fold (OOF) predictions for CatBoost.
# These predictions will later be used for stacking.
import gc
from sklearn.model_selection import StratifiedKFold , KFold
pred_cat = np.zeros(len(test))
oof_cat = np.zeros(len(X))
k = 5
startK = StratifiedKFold(n_splits=k , shuffle = True , random_state=42)
for fold,(tr_idx , vl_idx)  in enumerate(startK.split(X,y)):
    print('fold ---------------- '+str(fold))
    X_tr_raw = X.iloc[tr_idx].reset_index(drop=True)
    y_tr_raw=y.iloc[tr_idx].reset_index(drop=True)

    X_vl_raw  = X.iloc[vl_idx].reset_index(drop=True)
    y_vl=y.iloc[vl_idx].reset_index(drop=True)
    X_tr , X_vl  = X_tr_raw.copy(), X_vl_raw.copy()
    X_ts = test.copy(deep=True)
    print('creation model ===============')
    model_cat =   CatBoostClassifier(
            cat_features=cat_features,
            iterations=5000,
            learning_rate=0.05,
            grow_policy='Lossguide',
            l2_leaf_reg=21,
            depth=4,
            bagging_temperature=1.0,
            random_strength=3,
            border_count=254,
            loss_function='Logloss',
            eval_metric='AUC',
            bootstrap_type='Bayesian',   
            task_type='GPU',
            devices='0',                
            random_seed=42,
            verbose=False               
        )
    print('=================fitting=========model')
    model_cat.fit(
    X_tr,
    y_tr_raw,
    eval_set=(X_vl, y_vl),
    early_stopping_rounds=400,
    use_best_model=True
    )

    y_vl_pred_cat = model_cat.predict_proba(X_vl)[:,1]
    pred_cat += model_cat.predict_proba(X_ts)[:,1]
    oof_cat[vl_idx] = model_cat.predict_proba(X_vl)[:,1]
    auc = roc_auc_score(y_vl,y_vl_pred_cat)
    print('auc ============='+str(fold) + '==========' +str(auc))
    del  X_tr_raw , y_tr_raw,X_vl_raw 
    # y_vl ,X_tr , X_vl ,X_ts
pred_cat /= k
score_final_cat = roc_auc_score(y , oof_cat)
print("CatBoost OOF AUC:", roc_auc_score(y, oof_cat))




# Target encoding with leakage prevention.
# Encoding is done inside CV folds only.

def target_encoder(df_train, df_val, col, target):
    mean = df_train.groupby(col)[target].mean()
    global_mean = df_train[target].mean()
    col_name = f'{col}_mean'
    df_val = df_val.copy()
    df_val[col_name] = df_val[col].map(mean).fillna(global_mean).astype(float)
    return df_val



# """
# High-order categorical interactions.
# These capture complex non-linear relationships.
# """

selected_interactions = [
    ["age","bmi","systolic_bp"],
    ["bmi","cholesterol_total","ldl_cholesterol"],
    ["family_history_diabetes","waist_to_hip_ratio","triglycerides"],
    ["heart_rate","bmi","cholesterol_total","systolic_bp"],
]










interaction_features =[]



for col in cat_features:
    X[col] = X[col].astype("category")
    test[col] = test[col].astype("category")



# 
# XGBoost trained on numerical + target-encoded interaction features.
# GPU acceleration + early stopping.
# 
pred_xgb = np.zeros(len(test))
oof_xgb = np.zeros(len(X))
k = 5
startK = StratifiedKFold(n_splits=k , shuffle = True , random_state=88)
for fold,(tr_idx , vl_idx)  in enumerate(startK.split(X,y)):
    print('fold ---------------- '+str(fold))
    X_tr_raw = X.iloc[tr_idx].reset_index(drop=True)
    y_tr_raw=y.iloc[tr_idx].reset_index(drop=True)

    X_vl_raw  = X.iloc[vl_idx].reset_index(drop=True)
    y_vl=y.iloc[vl_idx].reset_index(drop=True)
    X_tr , X_vl  = X_tr_raw.copy(), X_vl_raw.copy()
    X_ts = test.copy(deep=True)
    for col in cat_features:
        X_tr[col] = X_tr[col].astype("category")
        X_vl[col] = X_vl[col].astype("category")
        X_ts[col] = X_ts[col].astype("category")

    for col in interaction_features:
        X_tr[f'{col}_mean'] = np.nan
        X_vl[f'{col}_mean'] = np.nan
        X_ts[f'{col}_mean'] = np.nan


    inner_startkfold = KFold(n_splits = k , shuffle=True , random_state=88)
    

    for _ ,(tr_idx_in,vl_idx_in) in enumerate(inner_startkfold.split(X_tr_raw)):
        in_tr = pd.concat([X_tr_raw.iloc[tr_idx_in],y_tr_raw.iloc[tr_idx_in]],axis=1)
        in_vl = X_tr_raw.iloc[vl_idx_in]
        for col in interaction_features :
           

            
            te_temp = target_encoder(in_tr, in_vl.copy(),col,'diagnosed_diabetes')
            te_col = f'{col}_mean'
            
            X_tr.loc[vl_idx_in,te_col] = te_temp[te_col].values
    assert not X_tr[[f'{c}_mean' for c in interaction_features]].isnull().any().any(), \
    "NaN detected in X_tr after OOF target encoding"
    tr_with_y = pd.concat([X_tr_raw, y_tr_raw], axis=1)

    for col in interaction_features :
       
        
        te_col=f'{col}_mean'
        tmp = target_encoder(tr_with_y, X_vl[[col]].copy(), col, 'diagnosed_diabetes')
        X_vl[f'{col}_mean'] = tmp[f'{col}_mean'].values

        tmp = target_encoder(tr_with_y, X_ts[[col]].copy(), col, 'diagnosed_diabetes')
        X_ts[f'{col}_mean'] = tmp[f'{col}_mean'].values

    X_tr.drop(interaction_features ,axis = 1 , inplace =True)
    X_vl.drop(interaction_features ,axis = 1 , inplace =True)
    X_ts.drop(interaction_features ,axis = 1 , inplace =True)
    print('=================fitting=========model')
    xgb_int = xgb.XGBClassifier(
            n_estimators = 6000,
            max_depth=5,
            min_child_weight=2,
            gamma=0,
            reg_alpha=5,
            reg_lambda=10,
            learning_rate=0.02,
            subsample=0.9,
            colsample_bytree=0.5,
            grow_policy='lossguide',
            booster='gbtree',
            tree_method='hist',      
           
            enable_categorical=True,     
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            early_stopping_rounds=400
    
        )
    xgb_int.fit(X_tr,y_tr_raw , eval_set = [(X_vl,y_vl)],verbose=False  )
    y_vl_pred = xgb_int.predict_proba(X_vl)[:,1]
    pred_xgb += xgb_int.predict_proba(X_ts)[:,1]
    oof_xgb[vl_idx] = xgb_int.predict_proba(X_vl)[:,1]
    auc = roc_auc_score(y_vl,y_vl_pred)
    print('auc ============='+str(fold) + '==========' +str(auc))
    del  X_tr_raw ,X_vl_raw 
pred_xgb /= k
score_final_xgb = roc_auc_score(y , oof_xgb)




# LightGBM model trained on interaction-enhanced dataset.
# Optimized with Optuna.

pred_lightgbm = np.zeros(len(test))
oof_lightgbm = np.zeros(len(X))
k = 5
startK = StratifiedKFold(n_splits=k , shuffle = True , random_state=88)
for fold,(tr_idx , vl_idx)  in enumerate(startK.split(X,y)):
    print('fold ---------------- '+str(fold))
    X_tr_raw = X.iloc[tr_idx].reset_index(drop=True)
    y_tr_raw=y.iloc[tr_idx].reset_index(drop=True)

    X_vl_raw  = X.iloc[vl_idx].reset_index(drop=True)
    y_vl=y.iloc[vl_idx].reset_index(drop=True)
    X_tr , X_vl  = X_tr_raw.copy(), X_vl_raw.copy()
    X_ts = test.copy(deep=True)
    for col in cat_features:
        X_tr[col] = X_tr[col].astype("category")
        X_vl[col] = X_vl[col].astype("category")
        X_ts[col] = X_ts[col].astype("category")

    for col in interaction_features:
        X_tr[f'{col}_mean'] = np.nan
        X_vl[f'{col}_mean'] = np.nan
        X_ts[f'{col}_mean'] = np.nan


    inner_startkfold = KFold(n_splits = k , shuffle=True , random_state=88)
    

    for _ ,(tr_idx_in,vl_idx_in) in enumerate(inner_startkfold.split(X_tr_raw)):
        in_tr = pd.concat([X_tr_raw.iloc[tr_idx_in],y_tr_raw.iloc[tr_idx_in]],axis=1)
        in_vl = X_tr_raw.iloc[vl_idx_in]
        for col in interaction_features :
           

            
            te_temp = target_encoder(in_tr, in_vl.copy(),col,'diagnosed_diabetes')
            te_col = f'{col}_mean'
            
            X_tr.loc[vl_idx_in,te_col] = te_temp[te_col].values
    assert not X_tr[[f'{c}_mean' for c in interaction_features]].isnull().any().any(), \
    "NaN detected in X_tr after OOF target encoding"
    tr_with_y = pd.concat([X_tr_raw, y_tr_raw], axis=1)

    for col in interaction_features :
       
        
        te_col=f'{col}_mean'
        tmp = target_encoder(tr_with_y, X_vl[[col]].copy(), col, 'diagnosed_diabetes')
        X_vl[f'{col}_mean'] = tmp[f'{col}_mean'].values

        tmp = target_encoder(tr_with_y, X_ts[[col]].copy(), col, 'diagnosed_diabetes')
        X_ts[f'{col}_mean'] = tmp[f'{col}_mean'].values

    X_tr.drop(interaction_features ,axis = 1 , inplace =True)
    X_vl.drop(interaction_features ,axis = 1 , inplace =True)
    X_ts.drop(interaction_features ,axis = 1 , inplace =True)
    print('creation model ===============')
    model_light = LGBMClassifier( learning_rate = 0.005 , num_leaves =  64 , max_depth =  6 , min_child_samples= 200 , min_gain_to_split = 1.43 ,
                       colsample_bytree = 0.447 , reg_alpha = 1.5829 , reg_lambda = 65.66 , max_bin=98 ,
                       objective = 'binary', n_estimators = 10000  ,n_jobs = -1,
                        random_state = 42 ,    verbosity= -1)
    print('=================fitting=========model')
    model_light.fit(X_tr,y_tr_raw , eval_set = [(X_vl,y_vl)] , callbacks=[
        early_stopping(stopping_rounds=400),
        log_evaluation(period=0)  
    ] , eval_metric="auc",)
    y_vl_pred = model_light.predict_proba(X_vl)[:,1]
    pred_lightgbm += model_light.predict_proba(X_ts)[:,1]
    oof_lightgbm[vl_idx] = model_light.predict_proba(X_vl)[:,1]
    auc = roc_auc_score(y_vl,y_vl_pred)
    print('auc ============='+str(fold) + '==========' +str(auc))
    del  X_tr_raw , y_tr_raw,X_vl_raw 
    # y_vl ,X_tr , X_vl ,X_ts
pred_lightgbm /= k
score_final = roc_auc_score(y , oof_lightgbm)
X_tr.head(3)




# Check correlation between base models.
# High correlation -> diminishing returns for stacking.


corr = np.corrcoef(oof_cat, oof_xgb)[0,1]
print("OOF correlation:", corr)




from sklearn.linear_model import LogisticRegressionCV


# Final stacking using Logistic Regression.
# Optimized directly for ROC-AUC.


dataset_stack = np.column_stack((oof_cat, oof_lightgbm, oof_xgb))

stacker = LogisticRegressionCV(cv=5, scoring="roc_auc")
stacker.fit(dataset_stack, y)

final_oof = stacker.predict_proba(dataset_stack)[:,1]
print("Stacked OOF AUC:", roc_auc_score(y, final_oof))



test_stack = np.column_stack((pred_cat, pred_lightgbm, pred_xgb))
final_pred = stacker.predict_proba(test_stack)[:,1]
test = test.set_index('id')
submission = pd.DataFrame({
    "id": test.index,
    "diagnosed_diabetes": final_pred
})

submission.to_csv("submission.csv", index=False)


