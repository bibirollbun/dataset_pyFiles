import warnings
warnings.filterwarnings("ignore")


!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from metric import score


train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')


RMV = ["ID","efs","efs_time"]
RACES = train['race_group']
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
        train[c] = train[c].astype('category')
        test[c] = test[c].astype('category')
    else:
        train[c] = train[c].fillna(-1)
        test[c] = test[c].fillna(-1) 
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


from sklearn.model_selection import StratifiedKFold
FOLDS = 5
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)


from lifelines import BreslowFlemingHarringtonFitter

def transform_survival_probability_bfh(df):

    oof = np.zeros(len(train))
    
    kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    for i, (train_index, test_index) in enumerate(kf.split(df, RACES)):

        df_train = df.iloc[train_index,:].copy()
        df_val = df.iloc[test_index,:].copy()
        
        bfh = BreslowFlemingHarringtonFitter()
        bfh.fit(durations=df_train['efs_time'], event_observed=df_train['efs'])

        oof[test_index] = bfh.survival_function_at_times(df_val['efs_time']).values

    return oof
    
train["Breslow"] = transform_survival_probability_bfh(train)
train.loc[train.efs == 0, 'Breslow'] -= 0.15


plt.hist(train.loc[train.efs==1,"Breslow"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"Breslow"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Transformed Target y")
plt.ylabel("Density")
plt.title("BreslowFlemingHarrington Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()


from metric import score
def res(df, preds):
    y_true = df[["ID","efs","efs_time","race_group"]].copy()
    y_pred = df[["ID"]].copy()
    y_pred["prediction"] = preds
    return score(y_true.copy(), y_pred.copy(), "ID")


param_cat = {
        'loss_function': 'RMSE',
        'learning_rate': 0.03,
        'random_state': 42,
        'task_type': 'CPU',
        'num_trees': 6000,
        'reg_lambda': 8.0,
        'depth': 8
    }

param_lgb = {
        'objective': 'regression',
        'metric': 'rmse',
        'device': 'cpu',
        'verbose': -1,
        "bagging_freq": 1,
        'n_estimators': 9800,
        'learning_rate': 0.0025562611410098906,
        'max_depth': 11,
        'subsample': 0.7000302358347922,
        'colsample_bytree': 0.34454787171802054,
        'min_data_in_leaf': 52
    }


param_xgb = {
        'device': 'cpu',
        'enable_categorical': True, 
        "objective": "reg:squarederror",
        "verbosity": 0,
        'n_estimators': 9400,
        'learning_rate': 0.01462545658882346,
        'max_depth': 4,
        'subsample': 0.8427706960687078,
        'colsample_bytree': 0.2630880900000106,
        'min_child_weight': 50,
        'reg_lambda': 29.0
}


from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
print("Using XGBoost version",xgb.__version__)


%%time
    
oof_xgb_bfh = np.zeros(len(train))
pred_xgb_bfh= np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train, train['race_group'])):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    X_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"Breslow"]
    X_val = train.loc[test_index,FEATURES].copy()
    y_val = train.loc[test_index,"Breslow"]
    X_test = test[FEATURES].copy()
    
    model_xgb_bfh = XGBRegressor(**param_xgb)
    model_xgb_bfh.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],  
        verbose=0,
        early_stopping_rounds=300,
    )

    # INFER OOF
    oof_xgb_bfh[test_index] = model_xgb_bfh.predict(X_val)
    # INFER TEST
    pred_xgb_bfh += model_xgb_bfh.predict(X_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb_bfh /= FOLDS


print('CV for XGBoost BreslowFlemingHarrington', res(train, oof_xgb_bfh))


import catboost as cb
from catboost import CatBoostRegressor
import optuna
print("Using CatBoost version",cb.__version__)


%%time
    
oof_cat_bfh = np.zeros(len(train))
pred_cat_bfh = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train, RACES)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    X_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"Breslow"]
    X_val = train.loc[test_index,FEATURES].copy()
    y_val = train.loc[test_index,"Breslow"]
    X_test = test[FEATURES].copy()


    model_cat_bfh = CatBoostRegressor(**param_cat)
    
    model_cat_bfh.fit(X_train, y_train, eval_set=[(X_val, y_val)], cat_features=CATS, verbose=0, early_stopping_rounds=300)
    
    # INFER OOF
    oof_cat_bfh[test_index] = model_cat_bfh.predict(X_val)
    # INFER TEST
    pred_cat_bfh += model_cat_bfh.predict(X_test)

# COMPUTE AVERAGE TEST PREDS
pred_cat_bfh /= FOLDS


print('CV for CatBoost BreslowFlemingHarrington', res(train, oof_cat_bfh))


from lightgbm import LGBMRegressor
import lightgbm as lgb
print("Using LightGBM version",lgb.__version__)


%%time


    
oof_lgb_bfh = np.zeros(len(train))
pred_lgb_bfh = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train, RACES)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    X_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"Breslow"]
    X_val = train.loc[test_index,FEATURES].copy()
    y_val = train.loc[test_index,"Breslow"]
    X_test = test[FEATURES].copy()

    model_lgb_bfh = LGBMRegressor(**param_lgb)
    
    model_lgb_bfh.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(300, verbose=0), lgb.log_evaluation(0)]
    )
    
    # INFER OOF
    oof_lgb_bfh[test_index] = model_lgb_bfh.predict(X_val)
    # INFER TEST
    pred_lgb_bfh += model_lgb_bfh.predict(X_test)

# COMPUTE AVERAGE TEST PREDS
pred_lgb_bfh /= FOLDS


print('CV for LightGBM BreslowFlemingHarrington', res(train, oof_lgb_bfh))


print(10 * '#')
print('RESULTS')
print(10 * '#')
print('CV for XGBoost   BreslowFlemingHarrington', res(train, oof_xgb_bfh))
print('CV for CatBoost  BreslowFlemingHarrington', res(train, oof_cat_bfh))
print('CV for LightGBM  BreslowFlemingHarrington', res(train, oof_lgb_bfh))


from scipy.stats import rankdata


models = [
    rankdata(oof_xgb_bfh),
    rankdata(oof_cat_bfh),
    rankdata(oof_lgb_bfh),
]

pred_models = [
    rankdata(pred_xgb_bfh),
    rankdata(pred_cat_bfh),
    rankdata(pred_lgb_bfh),
]

coefs = [8, 4, 1]

oof = 0
    
for coef, model in zip(coefs, models):
    oof += coef * model
print('CV for ensemble', res(train, oof))


ids = test['ID']
preds = 0    
for coef, model in zip(coefs, pred_models):
        preds += coef * model

output = pd.DataFrame(data={'ID': ids, 'prediction': preds})
output.to_csv('submission.csv', index=False)

