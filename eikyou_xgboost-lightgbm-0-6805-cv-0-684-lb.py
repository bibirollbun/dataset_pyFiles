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


import warnings
warnings.filterwarnings("ignore")


RMV = ["ID","efs","efs_time"]
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


#filling hla_nmdp_6
train["hla_nmdp_6"].fillna(train[["hla_match_a_low", "hla_match_b_low", "hla_match_drb1_high"]].sum(axis=1), inplace=True)
test["hla_nmdp_6"].fillna(test[["hla_match_a_low", "hla_match_b_low", "hla_match_drb1_high"]].sum(axis=1), inplace=True)  

#filling hla_low_res_6
train["hla_low_res_6"].fillna(train[["hla_match_a_low", "hla_match_b_low", "hla_match_drb1_low"]].sum(axis=1), inplace=True)
test["hla_low_res_6"].fillna(test[["hla_match_a_low", "hla_match_b_low", "hla_match_drb1_low"]].sum(axis=1), inplace=True)  

#filling hla_high_res_6
train["hla_high_res_6"].fillna(train[["hla_match_a_high", "hla_match_b_high", "hla_match_drb1_high"]].sum(axis=1), inplace=True)
test["hla_high_res_6"].fillna(test[["hla_match_a_high", "hla_match_b_high", "hla_match_drb1_high"]].sum(axis=1), inplace=True)  

#filling hla_low_res_8
train['hla_low_res_8'].fillna(train[["hla_match_a_low", "hla_match_b_low", "hla_match_c_low", "hla_match_drb1_low"]].sum(axis=1), inplace=True)
test['hla_low_res_8'].fillna(test[["hla_match_a_low", "hla_match_b_low", "hla_match_c_low", "hla_match_drb1_low"]].sum(axis=1), inplace=True)

#filling hla_high_res_8
train["hla_high_res_8"].fillna(train[["hla_match_a_high", "hla_match_b_high", "hla_match_c_high", "hla_match_drb1_high"]].sum(axis=1), inplace=True)
test["hla_high_res_8"].fillna(test[["hla_match_a_high", "hla_match_b_high", "hla_match_c_high", "hla_match_drb1_high"]].sum(axis=1), inplace=True)

#filling hla_low_res_10
train["hla_low_res_10"].fillna(train[["hla_match_a_low", "hla_match_b_low", "hla_match_c_low", "hla_match_drb1_low", "hla_match_dqb1_low"]].sum(axis=1), inplace=True)
test["hla_low_res_10"].fillna(test[["hla_match_a_low", "hla_match_b_low", "hla_match_c_low", "hla_match_drb1_low", "hla_match_dqb1_low"]].sum(axis=1), inplace=True)

#filling hla_high_res_10
train["hla_high_res_10"].fillna(train[["hla_match_a_high", "hla_match_b_high", "hla_match_c_high", "hla_match_drb1_high", "hla_match_dqb1_high"]].sum(axis=1), inplace=True)
test["hla_high_res_10"].fillna(test[["hla_match_a_high", "hla_match_b_high", "hla_match_c_high", "hla_match_drb1_high", "hla_match_dqb1_high"]].sum(axis=1), inplace=True)


CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
    else:
        train[c] = train[c].fillna(-1)
        test[c] = test[c].fillna(-1) 
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


combined = pd.concat([train,test],axis=0,ignore_index=True)
#print("Combined data shape:", combined.shape )

# LABEL ENCODE CATEGORICAL FEATURES
print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
for c in FEATURES:

    # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
    if c in CATS:
        print(f"{c}, ",end="")
        combined[c],_ = combined[c].factorize()
        combined[c] -= combined[c].min()
        combined[c] = combined[c].astype("int32")
        combined[c] = combined[c].astype("category")
    
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


from lifelines import KaplanMeierFitter
def transform_survival_probability_kmf(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return y
train["KaplanMeier"] = transform_survival_probability_kmf(train, time_col='efs_time', event_col='efs')


plt.hist(train.loc[train.efs==1,"KaplanMeier"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"KaplanMeier"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Transformed Target y")
plt.ylabel("Density")
plt.title("KaplanMeier Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()


from lifelines import NelsonAalenFitter
def transform_survival_probability_naf(df, time_col='efs_time', event_col='efs'):
    naf = NelsonAalenFitter()
    naf.fit(durations =df[time_col], event_observed=df[event_col])
    y = -naf.cumulative_hazard_at_times(df[time_col]).values
    return y
train["NelsonAalen"] = transform_survival_probability_naf(train, time_col='efs_time', event_col='efs')


plt.hist(train.loc[train.efs==1,"NelsonAalen"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"NelsonAalen"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Transformed Target")
plt.ylabel("Density")
plt.title("NelsonAalen Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()


from lifelines import CoxPHFitter
def transform_survival_probability_cox(df, time_col='efs_time', event_col='efs'):
    cph = CoxPHFitter()
    cph.fit(df=df, duration_col=time_col, event_col=event_col)
    y = []
    for a in train['efs_time']:
        y.append(cph.baseline_survival_.loc[a, :].item())
    return y
train["CoxPH"] = transform_survival_probability_cox(train, time_col='efs_time', event_col='efs')


plt.hist(train.loc[train.efs==1,"CoxPH"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"CoxPH"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Transformed Target")
plt.ylabel("Density")
plt.title("CoxPH Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()


param_xgb = {
    'device': 'cpu',
    'enable_categorical': True, 
    "objective": "reg:squarederror",
    "n_estimators": 6000,
    "verbosity": 0,
    'learning_rate': 0.009,
    'max_depth': 4,
    'subsample': 0.91,
    'colsample_bytree': 0.24,
    'min_child_weight': 5,
    'reg_lambda': 13
}

param_cox = {
    'device': 'cpu',
    'enable_categorical': True, 
    "objective": "survival:cox",
    "n_estimators": 6000,
    "verbosity": 0,
    'learning_rate': 0.038,
    'max_depth': 4,
    'subsample': 0.84,
    'colsample_bytree': 0.27,
    'min_child_weight': 16,
    'reg_lambda': 15
}

param_lgb = {
        'n_estimators': 2100,
        'learning_rate': 0.013,
        'max_depth': 10,
        'subsample': 0.87,
        'colsample_bytree': 0.12,
        'min_data_in_leaf': 17,
        'device': 'cpu',
        'objective': "regression",
        'metric': "rmse",
        'verbosity': -1,
        'bagging_freq': 1,
}


import warnings
warnings.filterwarnings("ignore")


from metric import score
def res(df, preds):
    y_true = df[["ID","efs","efs_time","race_group"]].copy()
    y_pred = df[["ID"]].copy()
    y_pred["prediction"] = preds
    return score(y_true.copy(), y_pred.copy(), "ID")


from sklearn.model_selection import KFold
FOLDS = 5


from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
print("Using XGBoost version",xgb.__version__)


from lightgbm import LGBMRegressor
import lightgbm as lgb
print("Using LightGBM version",lgb.__version__)


from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
print("Using XGBoost version",xgb.__version__)


%%time
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_xgb_km = np.zeros(len(train))
pred_xgb_km = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    X_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"KaplanMeier"]
    X_val = train.loc[test_index,FEATURES].copy()
    y_val = train.loc[test_index,"KaplanMeier"]
    X_test = test[FEATURES].copy()
    
    model_xgb_km = XGBRegressor(**param_xgb)
    model_xgb_km.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],  
        verbose=0,
        early_stopping_rounds=300,
    )

    # INFER OOF
    oof_xgb_km[test_index] = model_xgb_km.predict(X_val)
    # INFER TEST
    pred_xgb_km += model_xgb_km.predict(X_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb_km /= FOLDS


print('CV for XGBoost KaplanMeier', res(train, oof_xgb_km))


%%time
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_xgb_na = np.zeros(len(train))
pred_xgb_na = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    X_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"NelsonAalen"]
    X_val = train.loc[test_index,FEATURES].copy()
    y_val = train.loc[test_index,"NelsonAalen"]
    X_test = test[FEATURES].copy()
    
    model_xgb_na = XGBRegressor(**param_xgb)
    
    model_xgb_na.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],  
        verbose=0,
        early_stopping_rounds=300,
    )

    # INFER OOF
    oof_xgb_na[test_index] = model_xgb_na.predict(X_val)
    # INFER TEST
    pred_xgb_na += model_xgb_na.predict(X_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb_na /= FOLDS


print('CV for XGBoost NelsonAalen', res(train, oof_xgb_na))


%%time
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_xgb_coxph = np.zeros(len(train))
pred_xgb_coxph = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    X_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"CoxPH"]
    X_val = train.loc[test_index,FEATURES].copy()
    y_val = train.loc[test_index,"CoxPH"]
    X_test = test[FEATURES].copy()
    
    model_xgb_coxph = XGBRegressor(**param_xgb)
    
    model_xgb_coxph.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],  
        verbose=0,
        early_stopping_rounds=300,
    )

    # INFER OOF
    oof_xgb_coxph[test_index] = model_xgb_coxph.predict(X_val)
    # INFER TEST
    pred_xgb_coxph += model_xgb_coxph.predict(X_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb_coxph /= FOLDS


print('CV for XGBoost CoxPH', res(train, oof_xgb_coxph))


# SURVIVAL COX NEEDS THIS TARGET (TO DIGEST EFS AND EFS_TIME)
train["efs_time2"] = train.efs_time.copy()
train.loc[train.efs==0,"efs_time2"] *= -1


%%time
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_xgb_cox = np.zeros(len(train))
pred_xgb_cox = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    X_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"efs_time2"]    
    X_val = train.loc[test_index,FEATURES].copy()
    y_val = train.loc[test_index,"efs_time2"]
    X_test = test[FEATURES].copy()

    model_xgb_cox = XGBRegressor(**param_cox)
    
    model_xgb_cox.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0, early_stopping_rounds=300)
    
    # INFER OOF
    oof_xgb_cox[test_index] = model_xgb_cox.predict(X_val)
    # INFER TEST
    pred_xgb_cox += model_xgb_cox.predict(X_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb_cox /= FOLDS


print('CV for XGBoost Cox', res(train, oof_xgb_cox))


%%time
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_lgb_km = np.zeros(len(train))
pred_lgb_km = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    X_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"KaplanMeier"]
    X_val = train.loc[test_index,FEATURES].copy()
    y_val = train.loc[test_index,"KaplanMeier"]
    X_test = test[FEATURES].copy()

    model_lgb_km = LGBMRegressor(**param_lgb)
    
    model_lgb_km.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )
    
    # INFER OOF
    oof_lgb_km[test_index] = model_lgb_km.predict(X_val)
    # INFER TEST
    pred_lgb_km += model_lgb_km.predict(X_test)

# COMPUTE AVERAGE TEST PREDS
pred_lgb_km /= FOLDS


print('CV for LightGBM KaplanMeier', res(train, oof_lgb_km))


%%time
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_lgb_na = np.zeros(len(train))
pred_lgb_na = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    X_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"NelsonAalen"]
    X_val = train.loc[test_index,FEATURES].copy()
    y_val = train.loc[test_index,"NelsonAalen"]
    X_test = test[FEATURES].copy()

    model_lgb_na = LGBMRegressor(**param_lgb)
    
    model_lgb_na.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )
    
    # INFER OOF
    oof_lgb_na[test_index] = model_lgb_na.predict(X_val)
    # INFER TEST
    pred_lgb_na += model_lgb_na.predict(X_test)

# COMPUTE AVERAGE TEST PREDS
pred_lgb_na /= FOLDS


print('CV for LightGBM NelsonAalen', res(train, oof_lgb_na))


%%time
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_lgb_coxph = np.zeros(len(train))
pred_lgb_coxph = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    X_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"CoxPH"]
    X_val = train.loc[test_index,FEATURES].copy()
    y_val = train.loc[test_index,"CoxPH"]
    X_test = test[FEATURES].copy()

    model_lgb_coxph = LGBMRegressor(**param_lgb)
    
    model_lgb_coxph.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )
    
    # INFER OOF
    oof_lgb_coxph[test_index] = model_lgb_coxph.predict(X_val)
    # INFER TEST
    pred_lgb_coxph += model_lgb_coxph.predict(X_test)

# COMPUTE AVERAGE TEST PREDS
pred_lgb_coxph /= FOLDS


print('CV for LightGBM CoxPH', res(train, oof_lgb_coxph))


print(10 * '#')
print('RESULTS')
print(10 * '#')
print('CV for XGBoost KaplanMeier ', res(train, oof_xgb_km))
print('CV for XGBoost NelsonAalen ', res(train, oof_xgb_na))
print('CV for XGBoost CoxPH       ', res(train, oof_xgb_coxph))
print('CV for XGBoost Cox Loss    ', res(train, oof_xgb_cox))
print(10 * '#')
print('CV for LightGBM KaplanMeier', res(train, oof_lgb_km))
print('CV for LightGBM NelsonAalen', res(train, oof_lgb_na))
print('CV for LightGBM CoxPH      ', res(train, oof_lgb_coxph))


from scipy.stats import rankdata

oof_models = [
    rankdata(oof_xgb_km),
    rankdata(oof_xgb_na),
    rankdata(oof_xgb_coxph),
    rankdata(oof_xgb_cox),
    rankdata(oof_lgb_km),
    rankdata(oof_lgb_na),
    rankdata(oof_lgb_coxph)
]

pred_models = [
    rankdata(pred_xgb_km),
    rankdata(pred_xgb_na),
    rankdata(pred_xgb_coxph),
    rankdata(pred_xgb_cox),
    rankdata(pred_lgb_km),
    rankdata(pred_lgb_na),
    rankdata(pred_lgb_coxph)
]


# def objective(trial):
#     coefs = [
#         trial.suggest_int("xgb_km", 1, 10),
#         trial.suggest_int("xgb_na", 1, 10),
#         trial.suggest_int("xgb_coxph", 1, 10),
#         trial.suggest_int("xgb_cox", 1, 10),
#         trial.suggest_int("lgb_km", 1, 10),
#         trial.suggest_int("lgb_na", 1, 10),
#         trial.suggest_int("lgb_coxph", 1, 10),
#     ]

#     preds = 0
    
#     for coef, model in zip(coefs, oof_models):
#         preds += coef * model
        
#     out = res(train, preds)
    
#     return out


# import optuna
# study = optuna.create_study(direction="maximize")
# optuna.logging.set_verbosity(optuna.logging.WARNING)
# study.optimize(objective, n_trials=1000, show_progress_bar=True)


coefs = [3, 2, 5, 10, 1, 1, 2]
oof = 0
for coef, model in zip(coefs, oof_models):
        oof += coef * model
print('Overall CV', res(train , oof))


ids = test['ID']
preds = 0    
for coef, model in zip(coefs, pred_models):
        preds += coef * model

output = pd.DataFrame(data={'ID': ids, 'prediction': preds})
output.to_csv('submission.csv', index=False)

