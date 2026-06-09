!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl



"""
To evaluate the equitable prediction of transplant survival outcomes,
we use the concordance index (C-index) between a series of event
times and a predicted score across each race group.

It represents the global assessment of the model discrimination power:
this is the model’s ability to correctly provide a reliable ranking
of the survival times based on the individual risk scores.

The concordance index is a value between 0 and 1 where:

0.5 is the expected result from random predictions,
1.0 is perfect concordance (with no censoring, otherwise <1.0),
0.0 is perfect anti-concordance (with no censoring, otherwise >0.0)

"""

import pandas as pd
import pandas.api.types
import numpy as np
from lifelines.utils import concordance_index

class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> y_pred = {'prediction': {0: 1.0, 1: 0.0, 2: 1.0}}
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred.insert(0, row_id_column_name, range(len(y_pred)))
    >>> y_true = { 'efs': {0: 1.0, 1: 0.0, 2: 0.0}, 'efs_time': {0: 25.1234,1: 250.1234,2: 2500.1234}, 'race_group': {0: 'race_group_1', 1: 'race_group_1', 2: 'race_group_1'}}
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true.insert(0, row_id_column_name, range(len(y_true)))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    0.75
    """

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
    # Merging solution and submission dfs on ID
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(
                        merged_df_race[interval_label],
                        -merged_df_race[prediction_label],
                        merged_df_race[event_label])
        metric_list.append(c_index_race)
    # return float(np.mean(metric_list)-np.sqrt(np.var(metric_list))),metric_list,merged_df_race_dict.keys()
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))



import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
print("Test shape:", test.shape )

# test = add_features(test)

train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
print("Train shape:",train.shape)
train.head()

# train=add_features(train)


RMV = ["ID","efs","efs_time","y","stratify_label","donor_age_bin",'age_at_hct_bin','donor_age_bin_missing']
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


# CATS = []
# for c in FEATURES:
#     if train[c].nunique()<25:
#         CATS.append(c)
#         train[c] = train[c].fillna("NAN")
#         test[c] = test[c].fillna("NAN")
# print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


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
        
    # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")
    
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


# from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
print("Using XGBoost version",xgb.__version__)


from scipy.stats import gamma
k, beta = 4, -0.5 #4,0.5
train['y'] = 1 - gamma.cdf(train.efs_time / np.exp(-beta), k)


%%time
from sklearn.model_selection import KFold
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)



# 5分割のStratified K-Fold
# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)

    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"y"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"y"]
    x_test = test[FEATURES].copy()

    model_xgb = XGBRegressor(
        device="cuda",
        objective='reg:tweedie',
        max_depth=6,#3 6
        colsample_bytree=0.5,#0.5
        subsample=0.9,#0.8 0.9
        n_estimators=2000,#2000
        learning_rate=0.02,#0.02
        enable_categorical=True,
        min_child_weight=45,#80,45
        max_cat_to_onehot=8,#7 8
        reg_lambda=1,#1
        reg_alpha=0.1,#0.1
        gamma=0.9,
        eta=0.0,
        # early_stopping_rounds=200,
        monotone_constraints={
                    # 'comorbidity_score': 1,
                    # 'hla_match_c_high': -1,
                    # 'hla_high_res_10': -1,
                    'hla_high_res_6': -1,
                    'hla_high_res_8': -1,
                    # 'hla_low_res_10': -1,
                    'hla_low_res_6': -1,
                    # 'hla_low_res_8': -1,
                    'hla_match_a_high': -1,
                    # 'hla_match_a_low': -1,
                    # 'hla_match_b_high': -1,
                    'hla_match_drb1_low': -1,
                    'hla_match_c_low': -1,
                    # 'hla_match_c_high': -1,
                    # 'donor_age': -1,
                    # 'hla_match_drb1_high': -1,
                    'hla_match_dqb1_low': -1,
                    'hla_nmdp_6': -1,
                    # 'karnofsky_score': -1,

                }
        #early_stopping_rounds=25,
    )
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=500
    )

    # INFER OOF
    oof_xgb[test_index] = model_xgb.predict(x_valid)
    # INFER TEST
    pred_xgb += model_xgb.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb /= FOLDS


RMV=['oof_xgb_c','oof_xgb','oof_lgb']
FEATURES2=[c for c in FEATURES if not c in RMV]


from lightgbm import LGBMRegressor
import lightgbm as lgb
print("Using LightGBM version",lgb.__version__)


from sklearn.model_selection import GroupKFold
FOLDS = 10
# fold2=custom_group_stratified_kfold(train,train['y'],train['stratify_label'],train['dri_score'],n_splits=10)
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# gkf=GroupKFold(n_splits=5)
oof_lgb = np.zeros(len(train))
pred_lgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)

    x_train = train.loc[train_index,FEATURES2].copy()
    y_train = train.loc[train_index,"y"]
    x_valid = train.loc[test_index,FEATURES2].copy()
    y_valid = train.loc[test_index,"y"]
    # x_test = test[FEATURES2].copy()

    model_lgb = LGBMRegressor(
        # device="gpu",
        max_depth=4,#3 4
        colsample_bytree=0.2,#0.4 0.2
        # subsample=0.8,
        n_estimators=2500,#2500
        learning_rate=0.02,#0.02
        objective="tweedie",#tweedie
        verbose=-1,
        max_cat_to_onehot=7,#9 7
        cat_smooth=100,#100
        # monotone_constraints=monotone_constraints_list,
        lambda_l1=0.75,#0.75
        lambda_l2=1,#1
        #early_stopping_rounds=25,
    )
    model_lgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
    )

    # INFER OOF
    oof_lgb[test_index] = model_lgb.predict(x_valid)
    # INFER TEST
    pred_lgb += model_lgb.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_lgb /= FOLDS


%%time
from catboost import CatBoostRegressor
import numpy as np

FOLDS = 10
# skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_cat3 = np.zeros(len(train))
pred_cat3 = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):
    print("#" * 25)
    print(f"### Fold {i+1}")
    print("#" * 25)

    x_train = train.loc[train_index, FEATURES2].copy()
    y_train = train.loc[train_index, "y"]
    x_valid = train.loc[test_index, FEATURES2].copy()
    y_valid = train.loc[test_index, "y"]
    # x_test = test[FEATURES].copy()

    model_cat = CatBoostRegressor(
        task_type="GPU",
        boosting_type='Plain',  # 線形モード
        feature_border_type='GreedyLogSum',  # 線形回帰に適した特徴処理
        learning_rate=0.022249526148312184,
        verbose=250,
        iterations=4491,
        l2_leaf_reg=0.7357305158548999, 
        border_count= 246, 
        bagging_temperature= 0.6896908609682827,
        depth=5,
    )

    model_cat.fit(
        x_train, y_train,
        eval_set=(x_valid, y_valid),
        cat_features=CATS,
        verbose=250
    )

    # OOF 予測
    oof_cat3[test_index] = model_cat.predict(x_valid)
    # テストデータ予測
    pred_cat3 += model_cat.predict(x_test)

# 平均化
pred_cat3 /= FOLDS



import pickle

file_path = "/kaggle/input/oof-nn/oof_nn.pkl"

with open(file_path, "rb") as f:
    oof_nn = pickle.load(f)



from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

FOLDS = 5
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_xgb_c = np.zeros(len(train))
pred_efs_c = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train, train["efs"])):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index, FEATURES].copy()
    y_train = train.loc[train_index, "efs"]
    x_valid = train.loc[test_index, FEATURES].copy()
    y_valid = train.loc[test_index, "efs"]
    x_test = test[FEATURES].copy()

    model_xgb = XGBClassifier(
        device="cuda",
        max_depth=3,  
        colsample_bytree=0.7129400756425178, 
        subsample=0.8185881823156917, 
        n_estimators=20_000, 
        learning_rate=0.04425768131771064,  
        eval_metric="auc", 
        early_stopping_rounds=50, 
        objective='binary:logistic',
        scale_pos_weight=1.5379160847615545,  
        min_child_weight=4,
        enable_categorical=True,
        gamma=3.1330719334577584
    )
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=100
    )

    # INFER OOF (Probabilities -> Binary)
    oof_xgb_c[test_index] = (model_xgb.predict_proba(x_valid)[:, 1] > 0.5).astype(int)
    # INFER TEST (Probabilities -> Average Probs)
    pred_efs_c += model_xgb.predict_proba(x_test)[:, 1]

# COMPUTE AVERAGE TEST PREDS
pred_efs_c = (pred_efs_c / FOLDS > 0.5).astype(int)

# EVALUATE PERFORMANCE
accuracy = accuracy_score(train["efs"], oof_xgb_c)
f1 = f1_score(train["efs"], oof_xgb_c)
roc_auc = roc_auc_score(train["efs"], oof_xgb_c)
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")


from scipy.stats import rankdata

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_nn
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost KaplanMeier =",m)


%%time
SAVE = False
SKIP = False # skip train CoxPhFilter

import os,pickle
DIR1 = "models/"
DIR2 = "/kaggle/input/save-load-cibmtr-data/models/"
if SAVE: os.mkdir(DIR1)

SAVE=0
OUT_FOLDS = 10
IN_FOLDS = 10
kf = KFold(n_splits=OUT_FOLDS, shuffle=True, random_state=42)
    
oof4 = np.zeros(len(train))
pred4 = np.zeros(len(test))
all_oof4 = []
all_pred4 = []

if SAVE: 
    models = {}
else:
    with open(f"{DIR2}models4.pkl", "rb") as f:
        models = pickle.load(f)

for i, (train_index, test_index) in enumerate(kf.split(train)):

    all_oof4.append( np.zeros(len(train)) )
    all_pred4.append( np.zeros(len(test)) )

    print("#"*25)
    print(f"### OUT Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES2].copy()
    y_train = train.loc[train_index,"efs_time"].copy()
    x_valid = train.loc[test_index,FEATURES2].copy()
    x_test = test[FEATURES2].copy()

    kf2 = KFold(n_splits=IN_FOLDS, shuffle=True, random_state=42)
    for i2, (train_index2, test_index2) in enumerate(kf.split(train_index)):

        print(" ",f"### OUT Fold {i+1} => IN Fold {i2+1} ###")

        x_train2 = x_train.iloc[train_index2]
        y_train2a = y_train.iloc[train_index2].values
        y_train2b = np.where( train.loc[train_index,"efs"].values==1, train.loc[train_index,"efs_time"].values, -1 )[train_index2]
        y_train2 = np.column_stack([y_train2a,y_train2b])

        x_valid2 = x_train.iloc[test_index2]
        y_valid2a = y_train.iloc[test_index2]
        y_valid2b = np.where( train.loc[train_index,"efs"].values==1, train.loc[train_index,"efs_time"].values, -1 )[test_index2]
        y_valid2 = np.column_stack([y_valid2a,y_valid2b])

        n = f"model_f{i}_f{i2}"
        if SAVE:
            model = CatBoostRegressor(
                #task_type="GPU",  
                learning_rate=0.1,     
                grow_policy='Lossguide',
                loss_function='SurvivalAft',
            )
            model.fit(x_train2,y_train2,
                      eval_set=(x_valid2, y_valid2),
                      cat_features=CATS,
                      verbose=250)
            models[n] = model
        else:
            model = models[n]
        
        # INFER INSIDE OOF
        all_oof4[i][train_index[test_index2]] = model.predict(x_valid2)
        # INFER OUTSIDE OOF
        all_oof4[i][test_index] += model.predict(x_valid)
        # INFER INSIDE TEST
        all_pred4[i] += model.predict(x_test)
        # INFER OUTSIDE TEST
        pred4 += model.predict(x_test)
        
    all_oof4[i][test_index] /= IN_FOLDS
    oof4[test_index] = all_oof4[i][test_index]
    all_pred4[i] /= IN_FOLDS

# COMPUTE AVERAGE TEST PREDS
pred4 /= OUT_FOLDS * IN_FOLDS

# PRINT CV SCORES
print()
print("#"*25)
y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = -oof4
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"Overall CV = {m}")

for k in range(OUT_FOLDS):
    y_true = train[["ID","efs","efs_time","race_group"]].copy()
    y_pred = train[["ID"]].copy()
    y_pred["prediction"] = -all_oof4[k]
    m = score(y_true.copy(), y_pred.copy(), "ID")
    print(f"Out Fold {k+1} CV = {m}")


%%time
OUT_FOLDS = 10
IN_FOLDS = 10
kf = KFold(n_splits=OUT_FOLDS, shuffle=True, random_state=42)
    
oof77 = np.zeros(len(train))
pred77 = np.zeros(len(test))
all_oof7 = []
all_pred7 = []

if SAVE: 
    models = {}
else:
    with open(f"{DIR2}models7.pkl", "rb") as f:
        models = pickle.load(f)

for i, (train_index, test_index) in enumerate(kf.split(train)):

    all_oof7.append( np.zeros(len(train)) )
    all_pred7.append( np.zeros(len(test)) )

    print("#"*25)
    print(f"### OUT Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index].copy()
    x_valid = train.loc[test_index].copy()
    x_test = test.copy()

    kf2 = KFold(n_splits=IN_FOLDS, shuffle=True, random_state=42)
    for i2, (train_index2, test_index2) in enumerate(kf.split(train_index)):
        
        print(" ",f"### OUT Fold {i+1} => IN Fold {i2+1} ###")

        x_train2 = x_train.iloc[train_index2]
        x_valid2 = x_train.iloc[test_index2]

        n = f"model_f{i}_f{i2}"
        if SAVE:
            model = XGBClassifier(
                device="cuda",
                max_depth=3,  
                colsample_bytree=0.5,  
                subsample=0.8,  
                n_estimators=2000,  
                learning_rate=0.02,  
                enable_categorical=True,
                min_child_weight=80,
            )
            model.fit(
                x_train2[FEATURES2], x_train2.efs, 
                eval_set=[(x_valid2[FEATURES2], x_valid2.efs)], 
                verbose=500 
            )
            models[n] = model
        else:
            model = models[n]
        
        # INFER INSIDE OOF
        all_oof7[i][train_index[test_index2]] = model.predict_proba(x_valid2[FEATURES2])[:,0]
        # INFER OUTSIDE OOF
        all_oof7[i][test_index] += model.predict_proba(x_valid[FEATURES2])[:,0]
        # INFER INSIDE TEST
        all_pred7[i] += model.predict_proba(x_test[FEATURES2])[:,0]
        # INFER OUTSIDE TEST
        pred77 += model.predict_proba(x_test[FEATURES2])[:,0]
        
    all_oof7[i][test_index] /= IN_FOLDS
    oof77[test_index] = all_oof7[i][test_index]
    all_pred7[i] /= IN_FOLDS

# COMPUTE AVERAGE TEST PREDS
pred77 /= OUT_FOLDS * IN_FOLDS

# PRINT CV SCORES
print()
print("#"*25)
y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = -oof77
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"Overall CV = {m}")

for k in range(OUT_FOLDS):
    y_true = train[["ID","efs","efs_time","race_group"]].copy()
    y_pred = train[["ID"]].copy()
    y_pred["prediction"] = -all_oof7[k]
    m = score(y_true.copy(), y_pred.copy(), "ID")
    print(f"Out Fold {k+1} CV = {m}")





from lifelines import CoxPHFitter
from cuml.preprocessing.TargetEncoder import TargetEncoder
#from sklearn.preprocessing import TargetEncoder


train4 = train.copy()
test4 = test.copy()
RR = ["y2","y3","y","efs_time2","OOF1","OOF2","OOF3","OOF4","OOF5","ID","oof_99","oof_lgb","oof_xgb","oof_cat3","oof_xgb_c","oof_77","oof_all4"]
train4 = train4.drop([c for c in train4.columns if c in RR],axis=1)
test4 = test4.drop([c for c in test4.columns if c in RR],axis=1)

for c in FEATURES2:
    if c in CATS: continue
    m = train4[c].median() #pd.concat([train4[c],test4[c]],axis=0).median()
    train4[c] = train4[c].fillna(m)
    test4[c] = test4[c].fillna(m)
    print(c,", ",end="")


%%time

OUT_FOLDS = 10
IN_FOLDS = 10
kf = KFold(n_splits=OUT_FOLDS, shuffle=True, random_state=42)
    
oof66 = np.zeros(len(train4))
pred66 = np.zeros(len(test4))
all_oof6 = []
all_pred6 = []

if not SKIP:
    if SAVE: 
        models = {}
    else:
        with open(f"{DIR2}models6.pkl", "rb") as f:
            models = pickle.load(f)
    
    for i, (train_index, test_index) in enumerate(kf.split(train4)):
    
        all_oof6.append( np.zeros(len(train4)) )
        all_pred6.append( np.zeros(len(test4)) )
    
        print("#"*25)
        print(f"### OUT Fold {i+1}")
        print("#"*25)
        
        x_train = train4.loc[train_index].copy()
        x_valid = train4.loc[test_index].copy()
        x_test = test4.copy()
    
        kf2 = KFold(n_splits=IN_FOLDS, shuffle=True, random_state=42)
        for i2, (train_index2, test_index2) in enumerate(kf.split(train_index)):
    
            print(" ",f"### OUT Fold {i+1} => IN Fold {i2+1} ###")
    
            x_train2 = x_train.iloc[train_index2].copy()
            x_valid2 = x_train.iloc[test_index2].copy()
    
            if 1:
                FF = [f for f in x_train2.columns] 
                TE = []
                for c in FF:
                    if c not in CATS: continue
                    print(c,", ",end="")
                    #enc_auto = TargetEncoder(smooth='auto',
                    #                         random_state=42, 
                    #                         cv=10) 
                    enc_auto = TargetEncoder(smooth=5,
                                             split_method="random", 
                                             seed=42, 
                                             stat="mean",
                                             n_folds=10)
    
                    x_train2[f"{c}_efs"] = enc_auto.fit_transform(x_train2[[c]], x_train2.efs)
                    x_valid2[f"{c}_efs"] = enc_auto.transform(x_valid2[[c]])
                    x_valid[f"{c}_efs"] = enc_auto.transform(x_valid[[c]])
                    x_test[f"{c}_efs"] = enc_auto.transform(x_test[[c]])
                    
                    x_train2[f"{c}_efs_time"] = enc_auto.fit_transform(x_train2[[c]], x_train2.efs_time)
                    x_valid2[f"{c}_efs_time"] = enc_auto.transform(x_valid2[[c]])
                    x_valid[f"{c}_efs_time"] = enc_auto.transform(x_valid[[c]])
                    x_test[f"{c}_efs_time"] = enc_auto.transform(x_test[[c]])
                    TE.append(c)
                    
                #x_train2 = x_train2.drop(TE,axis=1)
                #x_valid2 = x_valid2.drop(TE,axis=1)
                print()
    
            n = f"model_f{i}_f{i2}"
            if SAVE:
                model = CoxPHFitter()
                model.fit(x_train2, duration_col='efs_time', event_col='efs')
                models[n] = model
            else:
                model = models[n]
            
            # INFER INSIDE OOF
            all_oof6[i][train_index[test_index2]] = model.predict_partial_hazard(x_valid2)
            # INFER OUTSIDE OOF
            COLS = x_valid2.columns
            all_oof6[i][test_index] += model.predict_partial_hazard(x_valid[COLS])
            # INFER INSIDE TEST
            all_pred6[i] += model.predict_partial_hazard(x_test[COLS])
            # INFER OUTSIDE TEST
            pred66 += model.predict_partial_hazard(x_test[COLS])
            
        all_oof6[i][test_index] /= IN_FOLDS
        oof66[test_index] = all_oof6[i][test_index]
        all_pred6[i] /= IN_FOLDS
    
    # COMPUTE AVERAGE TEST PREDS
    pred66 /= OUT_FOLDS * IN_FOLDS
    
    # PRINT CV SCORES
    print()
    print("#"*25)
    y_true = train[["ID","efs","efs_time","race_group"]].copy()
    y_pred = train[["ID"]].copy()
    y_pred["prediction"] = oof66
    m = score(y_true.copy(), y_pred.copy(), "ID")
    print(f"Overall CV = {m}")
    
    for k in range(OUT_FOLDS):
        y_true = train[["ID","efs","efs_time","race_group"]].copy()
        y_pred = train[["ID"]].copy()
        y_pred["prediction"] = all_oof6[k]
        m = score(y_true.copy(), y_pred.copy(), "ID")
        print(f"Out Fold {k+1} CV = {m}")
else:
    oof66 = np.load("/kaggle/input/save-load-cibmtr-data/oof2/oof66.npy")
    pred66 = pred99.copy()
    all_oof6 = np.load("/kaggle/input/save-load-cibmtr-data/oof3/all_oof6.npy")
    all_pred6 = all_pred9.copy()


%%time
OUT_FOLDS = 10
IN_FOLDS = 10
kf = KFold(n_splits=OUT_FOLDS, shuffle=True, random_state=42)
    
oof99 = np.zeros(len(train))
pred99 = np.zeros(len(test))
all_oof9 = []
all_pred9 = []

if SAVE: 
    models = {}
else:
    with open(f"{DIR2}models9.pkl", "rb") as f:
        models = pickle.load(f)

for i, (train_index, test_index) in enumerate(kf.split(train)):

    all_oof9.append( np.zeros(len(train)) )
    all_pred9.append( np.zeros(len(test)) )

    print("#"*25)
    print(f"### OUT Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index].copy()
    x_valid = train.loc[test_index].copy()
    x_test = test.copy()

    kf2 = KFold(n_splits=IN_FOLDS, shuffle=True, random_state=42)
    for i2, (train_index2, test_index2) in enumerate(kf.split(train_index)):
        
        print(" ",f"### OUT Fold {i+1} => IN Fold {i2+1} ###")

        x_train2 = x_train.iloc[train_index2]
        x_valid2 = x_train.iloc[test_index2]

        n = f"model_f{i}_f{i2}"
        if SAVE:
            model = CatBoostClassifier(
                task_type="GPU",  
                learning_rate=0.1,  
                grow_policy='Lossguide',
                iterations=250
            )
            model.fit(x_train2[FEATURES],x_train2.efs,
                      eval_set=(x_valid2[FEATURES2], x_valid2.efs),
                      cat_features=CATS,
                      verbose=50)
            models[n] = model
        else:
            model = models[n]
        
        # INFER INSIDE OOF
        all_oof9[i][train_index[test_index2]] = model.predict_proba(x_valid2[FEATURES2])[:,0]
        # INFER OUTSIDE OOF
        all_oof9[i][test_index] += model.predict_proba(x_valid[FEATURES2])[:,0]
        # INFER INSIDE TEST
        all_pred9[i] += model.predict_proba(x_test[FEATURES2])[:,0]
        # INFER OUTSIDE TEST
        pred99 += model.predict_proba(x_test[FEATURES2])[:,0]
        
    all_oof9[i][test_index] /= IN_FOLDS
    oof99[test_index] = all_oof9[i][test_index]
    all_pred9[i] /= IN_FOLDS

# COMPUTE AVERAGE TEST PREDS
pred99 /= OUT_FOLDS * IN_FOLDS

# PRINT CV SCORES
print()
print("#"*25)
y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = -oof99
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"Overall CV = {m}")

for k in range(OUT_FOLDS):
    y_true = train[["ID","efs","efs_time","race_group"]].copy()
    y_pred = train[["ID"]].copy()
    y_pred["prediction"] = -all_oof9[k]
    m = score(y_true.copy(), y_pred.copy(), "ID")
    print(f"Out Fold {k+1} CV = {m}")


train['oof_xgb']=oof_xgb
FEATURES.append('oof_xgb')


# train['oof_xgb_c']=oof_xgb_c
# FEATURES.append('oof_xgb_c')


train['oof_lgb']=oof_lgb
FEATURES.append('oof_lgb')


train['oof_cat3']=oof_cat3
FEATURES.append('oof_cat3')


train['oof_all4']=-oof4
FEATURES.append('oof_all4')


train['oof_77']=-oof77
FEATURES.append('oof_77')


train['oof_99']=-oof99
FEATURES.append('oof_99')


train['oof_66']=-oof66
FEATURES.append('oof_66')


train['oof_nn']=oof_nn
FEATURES.append('oof_nn')


oof_nn = ["oof_nn"]  # 除外したい要素（複数の要素がある場合はリストに追加）

# oof_nn に含まれる要素を除外
FEATURES = [feat for feat in FEATURES if feat not in oof_nn]


test['oof_xgb']=pred_xgb
test['oof_lgb']=pred_lgb
test['oof_cat3']=pred_cat3
test['oof_all4']=pred4
test['oof_77']=pred77
test['oof_99']=pred99
test['oof_66']=pred66


test['y']=1





from catboost import CatBoostRanker, Pool
from scipy.stats import rankdata
import catboost as cb
print(f"CatBoost version",cb.__version__)


yes = [['cyto_score_detail', 'hla_match_a_low', 'prod_type', '1'],
 ['hla_match_drb1_low', 'rituximab', '3'],
 ['hla_high_res_10', 'hla_match_dqb1_low', '1'],
 ['melphalan_dose', 'obesity', '2'],
 ['hla_nmdp_6', 'prim_disease_hct', '0'],
 ['donor_related', 'prod_type', '2'],
 ['hla_low_res_8', 'hla_match_b_low', '0'],
 ['conditioning_intensity', 'hepatic_mild', '1'],
 ['hla_match_drb1_low', 'pulm_severe', '2'],
 ['dri_score', 'hla_low_res_10', '2'],
 ['rheum_issue', 'year_hct', '1'],
 ['arrhythmia', 'hla_low_res_6', '2'],
 ['sex_match', 'tce_div_match', '2'],
 ['comorbidity_score', 'cyto_score', '2'],
 ['diabetes', 'in_vivo_tcd', 'pulm_severe', '3'],
 ['cyto_score', 'prod_type', '1'],
 ['hepatic_severe', 'in_vivo_tcd', '0'],
 ['cyto_score_detail', 'graft_type', 'hla_high_res_8', '4'],
 ['comorbidity_score', 'tce_imm_match', '2'],
 ['gvhd_proph', 'prod_type', '2'],
 ['conditioning_intensity', 'hla_match_c_low', '2'],
 ['hepatic_mild', 'hla_low_res_10', '2'],
 ['cyto_score', 'pulm_severe', '4'],
 ['dri_score', 'year_hct', '2'],
 ['pulm_moderate', 'sex_match', '0'],
 ['graft_type', 'hla_match_drb1_high', '1'],
 ['prior_tumor', 'sex_match', 'tce_div_match', '1'],
 ['hla_match_a_low', 'hla_nmdp_6', '2'],
 ['donor_age', 'ethnicity', '0'],
 ['conditioning_intensity', 'hla_high_res_10', 'hla_match_c_low', '4'],
 ['ethnicity', 'peptic_ulcer', '3'],
 ['ethnicity', 'tce_div_match', '2'],
 ['karnofsky_score', 'prod_type', '3'],
 ['diabetes', 'graft_type', 'prod_type', '2'],
 ['conditioning_intensity', 'tce_match', '3'],
 ['ethnicity', 'hla_match_c_low', '2'],
 ['hla_match_drb1_low', 'tce_imm_match', '3'],
 ['hla_low_res_10', 'sex_match', 'vent_hist', '4'],
 ['hla_match_a_low', 'mrd_hct', '4'],
 ['dri_score', 'prim_disease_hct', '1'],
 ['dri_score', 'tbi_status', '0'],
 ['dri_score', 'ethnicity', '0'],
 ['dri_score', 'year_hct', '0'],
 ['conditioning_intensity', 'hla_match_b_low', '1'],
 ['cyto_score', 'hla_high_res_8', '1'],
 ['cyto_score', 'cyto_score_detail', '1'],
 ['conditioning_intensity', 'karnofsky_score', '4'],
 ['cyto_score', 'mrd_hct', '4'],
 ['age_at_hct', 'cyto_score', '4'],
 ['dri_score', 'hla_low_res_6', '4'],
 ['cardiac', 'dri_score', '4']]

yes += [['peptic_ulcer', 'prod_type', '1'],
 ['ethnicity', 'mrd_hct', '2'],
 ['ethnicity', 'prod_type', '0'],
 ['diabetes', 'prim_disease_hct', '0'],
 ['mrd_hct', 'prior_tumor', 'tce_imm_match', '3'],
 ['dri_score', 'hepatic_severe', '1'],
 ['prod_type', 'renal_issue', '1'],
 ['conditioning_intensity', 'peptic_ulcer', '1'],
 ['gvhd_proph', 'prod_type', '3'],
 ['cardiac', 'prim_disease_hct', '2'],
 ['cyto_score_detail', 'donor_related', '0'],
 ['arrhythmia', 'prim_disease_hct', '1'],
 ['cardiac', 'prim_disease_hct', '4'],
 ['prim_disease_hct', 'tce_div_match', '4'],
 ['cyto_score', 'diabetes', '4'],
 ['rheum_issue', 'sex_match', '0'],
 ['conditioning_intensity', 'prod_type', 'psych_disturb', '2'],
 ['hepatic_mild', 'pulm_moderate', '1'],
 ['in_vivo_tcd', 'sex_match', '1'],
 ['ethnicity', 'mrd_hct', 'vent_hist', '3'],
 ['conditioning_intensity', 'prim_disease_hct', 'tce_imm_match', '3'],
 ['cyto_score', 'dri_score', 'prior_tumor', '2'],
 ['rituximab', 'tce_div_match', '3'],
 ['mrd_hct', 'obesity', '2'],
 ['rheum_issue', 'vent_hist', '1']]

yes += [['cyto_score_detail', 'ethnicity', '1'],
 ['conditioning_intensity', 'in_vivo_tcd', '1'],
 ['prim_disease_hct', 'pulm_severe', '0'],
 ['in_vivo_tcd', 'melphalan_dose', 'renal_issue', '0'],
 ['peptic_ulcer', 'prim_disease_hct', '2'],
 ['cardiac', 'conditioning_intensity', 'vent_hist', '2'],
 ['cardiac', 'in_vivo_tcd', '2']]
yes += [['mrd_hct', 'race_group', '4'],
        ['in_vivo_tcd', 'pulm_severe', '3'],
        ['cardiac', 'donor_related', 'melphalan_dose', '4']]

yes += [['peptic_ulcer', 'prim_disease_hct', '2'],
 ['peptic_ulcer', 'prod_type', '0'],
 ['cyto_score', 'hla_nmdp_6', '0'],
 ['hepatic_mild', 'psych_disturb', '2'],
 ['gvhd_proph', 'hla_match_a_high', '2'],
 ['cyto_score', 'melphalan_dose', '0'],
 ['hla_low_res_6', 'hla_match_dqb1_low', '1'],
 ['conditioning_intensity', 'graft_type', '0'],
 ['ethnicity', 'hla_low_res_8', '3'],
 ['conditioning_intensity', 'donor_related', '2'],
 ['arrhythmia', 'hla_match_a_high', '1'],
 ['arrhythmia', 'ethnicity', '2'],
 ['donor_related', 'prod_type', '4'],
 ['hla_match_a_low', 'hla_match_dqb1_low', '2'],
 ['hla_match_a_high', 'hla_match_a_low', '3'],
 ['peptic_ulcer', 'prim_disease_hct', '1'],
 ['prim_disease_hct', 'tce_imm_match', '1'],
 ['hla_low_res_8', 'hla_match_dqb1_high', '3'],
 ['obesity', 'pulm_severe', '2']]

yes += [['cyto_score_detail', 'prim_disease_hct', '1'],
 ['tbi_status', 'vent_hist', '2'],
 ['conditioning_intensity', 'renal_issue', '0'],
 ['diabetes', 'prior_tumor', '3'],
 ['prior_tumor', 'pulm_moderate', '3'],
 ['graft_type', 'melphalan_dose', '3'],
 ['conditioning_intensity', 'sex_match', '0'],
 ['cyto_score_detail', 'donor_related', 'graft_type', '0'],
 ['donor_related', 'psych_disturb', '2'],
 ['conditioning_intensity', 'tce_imm_match', 'tce_match', '0'],
 ['obesity', 'prim_disease_hct', '3'],
 ['mrd_hct', 'sex_match', '3'],
 ['peptic_ulcer', 'tce_match', '3'],
 ['donor_related', 'ethnicity', 'pulm_moderate', '4'],
 ['rituximab', 'tce_match', '3'],
 ['cyto_score_detail', 'dri_score', '2'],
 ['mrd_hct', 'prod_type', '1'],
 ['arrhythmia', 'in_vivo_tcd', '1'],
 ['cardiac', 'prod_type', '3'],
 ['cardiac', 'obesity', '3'],
 ['hepatic_severe', 'prim_disease_hct', '0'],
 ['hepatic_severe', 'in_vivo_tcd', 'peptic_ulcer', '4'],
 ['hepatic_mild', 'tce_div_match', '3']]
yes += [['gvhd_proph', 'prim_disease_hct', '4']]

yes += [['hla_match_b_low', 'in_vivo_tcd', '2'],
 ['in_vivo_tcd', 'peptic_ulcer', '1'],
 ['arrhythmia', 'prior_tumor', 'psych_disturb', '1'],
 ['hla_match_dqb1_low', 'hla_match_drb1_low', 'pulm_moderate', '3'],
 ['hla_high_res_10', 'hla_match_dqb1_high', 'hla_match_drb1_high', '1'],
 ['cyto_score', 'mrd_hct', '4'],
 ['hla_high_res_10', 'hla_match_drb1_low', '3'],
 ['hla_match_drb1_high', 'tbi_status', 'tce_match', '4'],
 ['cyto_score', 'tce_div_match', '1'],
 ['hla_match_a_high', 'prim_disease_hct', '0'],
 ['hla_match_c_high', 'prior_tumor', '3'],
 ['prior_tumor', 'prod_type', '0'],
 ['pulm_severe', 'year_hct', '4'],
 ['hla_low_res_6', 'rheum_issue', '4'],
 ['hla_match_a_high', 'in_vivo_tcd', '2'],
 ['hla_match_c_high', 'prim_disease_hct', '1'],
 ['hla_high_res_10', 'hla_match_a_low', '0'],
 ['gvhd_proph', 'pulm_severe', '0'],
 ['peptic_ulcer', 'vent_hist', '2'],
 ['race_group', 'tbi_status', '3']]


if 1:
    yes += [['cyto_score', 'hepatic_mild','0'],
     ['cardiac', 'conditioning_intensity','0'],
     ['renal_issue', 'sex_match','0'],
     ['cyto_score_detail', 'rituximab','0'],
     ['ethnicity', 'prod_type','0'],
     ['hepatic_mild', 'prior_tumor','0'],
     ['diabetes', 'prim_disease_hct','0'],
     ['in_vivo_tcd', 'peptic_ulcer','0'],
     ['mrd_hct', 'prim_disease_hct','0'],
     ['conditioning_intensity', 'tce_match','0'],
     ['arrhythmia', 'prim_disease_hct','0'],
     ['conditioning_intensity', 'obesity','0']]
    
    yes += [['cardiac', 'graft_type','1'],
     ['graft_type', 'prod_type','1'],
     ['conditioning_intensity', 'diabetes','1'],
     ['melphalan_dose', 'prim_disease_hct','1'],
     ['conditioning_intensity', 'sex_match','1'],
     ['dri_score', 'in_vivo_tcd','1']]
    
    yes += [['cmv_status', 'tce_imm_match','2'],
     ['prod_type', 'pulm_severe','2'],
     ['diabetes', 'donor_related','2'],
     ['prim_disease_hct', 'prod_type','2'],
     ['conditioning_intensity', 'hepatic_mild','2'],
     ['conditioning_intensity', 'sex_match','2']]

    yes += [['cardiac', 'hepatic_mild','3'],
     ['obesity', 'vent_hist','3'],
     ['cyto_score_detail', 'sex_match','3'],
     ['cyto_score', 'cyto_score_detail','3']]


import warnings
from cuml.common.exceptions import NotFittedError
from cuml.internals.safe_imports import cpu_only_import
from cuml.internals.safe_imports import gpu_only_import

cudf = gpu_only_import("cudf")
pandas = cpu_only_import("pandas")
cp = gpu_only_import("cupy")
np = cpu_only_import("numpy")


def get_stat_func(stat):
    def func(ds):
        if hasattr(ds, stat):
            return getattr(ds, stat)()
        else:
            # implement stat
            raise ValueError(f"{stat} function is not implemented.")

    return func


class TargetEncoder2:
    """
    A cudf based implementation of target encoding [1]_, which converts
    one or multiple categorical variables, 'Xs', with the average of
    corresponding values of the target variable, 'Y'. The input data is
    grouped by the columns `Xs` and the aggregated mean value of `Y` of
    each group is calculated to replace each value of `Xs`. Several
    optimizations are applied to prevent label leakage and parallelize
    the execution.

    Parameters
    ----------
    n_folds : int (default=4)
        Default number of folds for fitting training data. To prevent
        label leakage in `fit`, we split data into `n_folds` and
        encode one fold using the target variables of the remaining folds.
    smooth : int or float (default=0)
        Count of samples to smooth the encoding. 0 means no smoothing.
    seed : int (default=42)
        Random seed
    split_method : {'random', 'continuous', 'interleaved'}, \
        (default='interleaved')
        Method to split train data into `n_folds`.
        'random': random split.
        'continuous': consecutive samples are grouped into one folds.
        'interleaved': samples are assign to each fold in a round robin way.
        'customize': customize splitting by providing a `fold_ids` array
        in `fit()` or `fit_transform()` functions.
    output_type : {'cupy', 'numpy', 'auto'}, default = 'auto'
        The data type of output. If 'auto', it matches input data.
    stat : {'mean','var','median'}, default = 'mean'
        The statistic used in encoding, mean, variance or median of the
        target.

    References
    ----------
    .. [1] https://maxhalford.github.io/blog/target-encoding/

    Examples
    --------
    Converting a categorical implementation to a numerical one

    >>> from cudf import DataFrame, Series
    >>> from cuml.preprocessing import TargetEncoder
    >>> train = DataFrame({'category': ['a', 'b', 'b', 'a'],
    ...                    'label': [1, 0, 1, 1]})
    >>> test = DataFrame({'category': ['a', 'c', 'b', 'a']})

    >>> encoder = TargetEncoder()
    >>> train_encoded = encoder.fit_transform(train.category, train.label)
    >>> test_encoded = encoder.transform(test.category)
    >>> print(train_encoded)
    [1. 1. 0. 1.]
    >>> print(test_encoded)
    [1.   0.75 0.5  1.  ]
    """

    def __init__(
        self,
        n_folds=4,
        smooth=0,
        seed=42,
        split_method="interleaved",
        output_type="auto",
        stat="mean",
    ):
        if smooth < 0:
            raise ValueError(f"smooth {smooth} is not zero or positive")
        if n_folds < 0 or not isinstance(n_folds, int):
            raise ValueError(
                "n_folds {} is not a positive integer".format(n_folds)
            )
        if output_type not in {"cupy", "numpy", "auto"}:
            msg = (
                "output_type should be either 'cupy'"
                " or 'numpy' or 'auto', "
                "got {0}.".format(output_type)
            )
            raise ValueError(msg)
        if stat not in {"mean", "var", "median","min","max","nunique"}:
            msg = "stat should be 'mean', 'var' or 'median'." f"got {stat}."
            raise ValueError(msg)

        if not isinstance(seed, int):
            raise ValueError("seed {} is not an integer".format(seed))

        if split_method not in {
            "random",
            "continuous",
            "interleaved",
            "customize",
        }:
            msg = (
                "split_method should be either 'random'"
                " or 'continuous' or 'interleaved', or 'customize'"
                "got {0}.".format(split_method)
            )
            raise ValueError(msg)

        self.n_folds = n_folds
        self.seed = seed
        self.smooth = smooth
        self.split_method = split_method
        self.y_col = "__TARGET__"
        self.y_col2 = "__TARGET__SQUARE__"
        self.x_col = "__FEA__"
        self.out_col = "__TARGET_ENCODE__"
        self.out_col2 = "__TARGET_ENCODE__SQUARE__"
        self.fold_col = "__FOLD__"
        self.id_col = "__INDEX__"
        self.train = None
        self.output_type = output_type
        self.stat = stat

    def fit(self, x, y, fold_ids=None):
        """
        Fit a TargetEncoder instance to a set of categories

        Parameters
        ----------
        x : cudf.Series or cudf.DataFrame or cupy.ndarray
           categories to be encoded. It's elements may or may
           not be unique
        y : cudf.Series or cupy.ndarray
            Series containing the target variable.
        fold_ids : cudf.Series or cupy.ndarray
            Series containing the indices of the customized
            folds. Its values should be integers in range
            `[0, N-1]` to split data into `N` folds. If None,
            fold_ids is generated based on `split_method`.

        Returns
        -------
        self : TargetEncoder
            A fitted instance of itself to allow method chaining
        """
        if self.split_method == "customize" and fold_ids is None:
            raise ValueError(
                "`fold_ids` is required "
                "since split_method is set to"
                "'customize'."
            )
        if fold_ids is not None and self.split_method != "customize":
            self.split_method == "customize"
            warnings.warn(
                "split_method is set to 'customize'"
                "since `fold_ids` are provided."
            )
        if fold_ids is not None and len(fold_ids) != len(x):
            raise ValueError(
                f"`fold_ids` length {len(fold_ids)}"
                "is different from input data length"
                f"{len(x)}"
            )

        res, train = self._fit_transform(x, y, fold_ids=fold_ids)
        self.train_encode = res
        self.train = train
        self._fitted = True
        return self

    def fit_transform(self, x, y, fold_ids=None):
        """
        Simultaneously fit and transform an input

        This is functionally equivalent to (but faster than)
        `TargetEncoder().fit(y).transform(y)`

        Parameters
        ----------
        x : cudf.Series or cudf.DataFrame or cupy.ndarray
           categories to be encoded. It's elements may or may
           not be unique
        y : cudf.Series or cupy.ndarray
            Series containing the target variable.
        fold_ids : cudf.Series or cupy.ndarray
            Series containing the indices of the customized
            folds. Its values should be integers in range
            `[0, N-1]` to split data into `N` folds. If None,
            fold_ids is generated based on `split_method`.

        Returns
        -------
        encoded : cupy.ndarray
            The ordinally encoded input series

        """
        self.fit(x, y, fold_ids=fold_ids)
        return self.train_encode

    def transform(self, x):
        """
        Transform an input into its categorical keys.

        This is intended for test data. For fitting and transforming
        the training data, prefer `fit_transform`.

        Parameters
        ----------
        x : cudf.Series
            Input keys to be transformed. Its values doesn't have to
            match the categories given to `fit`

        Returns
        -------
        encoded : cupy.ndarray
            The ordinally encoded input series

        """
        self._check_is_fitted()
        test = self._data_with_strings_to_cudf_dataframe(x)
        if self._is_train_df(test):
            return self.train_encode
        x_cols = [i for i in test.columns.tolist() if i != self.id_col]
        test = test.merge(self.encode_all, on=x_cols, how="left")
        return self._impute_and_sort(test)

    def _fit_transform(self, x, y, fold_ids):
        """
        Core function of target encoding
        """
        self.output_type = self._get_output_type(x)
        cp.random.seed(self.seed)
        train = self._data_with_strings_to_cudf_dataframe(x)
        x_cols = [i for i in train.columns.tolist() if i != self.id_col]
        train[self.y_col] = self._make_y_column(y)

        self.n_folds = min(self.n_folds, len(train))
        train[self.fold_col] = self._make_fold_column(len(train), fold_ids)

        self.y_stat_val = get_stat_func(self.stat)(train[self.y_col])
        if self.stat in ["median","min","max","nunique"]:
            return self._fit_transform_for_loop(train, x_cols)

        self.mean = train[self.y_col].mean()
        if self.stat == "var":
            y_cols = [self.y_col, self.y_col2]
            train[self.y_col2] = self._make_y_column(y * y)
            self.mean2 = train[self.y_col2].mean()
        else:
            y_cols = [self.y_col]

        y_count_each_fold, y_count_all = self._groupby_agg(
            train, x_cols, op="count", y_cols=y_cols
        )

        y_sum_each_fold, y_sum_all = self._groupby_agg(
            train, x_cols, op="sum", y_cols=y_cols
        )

        """
        Note:
            encode_each_fold is used to encode train data.
            encode_all is used to encode test data.
        """
        cols = [self.fold_col] + x_cols
        encode_each_fold = self._compute_output(
            y_sum_each_fold,
            y_count_each_fold,
            cols,
            f"{self.y_col}_x",
            f"{self.y_col2}_x",
        )
        encode_all = self._compute_output(
            y_sum_all, y_count_all, x_cols, self.y_col, self.y_col2
        )

        self.encode_all = encode_all

        train = train.merge(encode_each_fold, on=cols, how="left")
        del encode_each_fold
        return self._impute_and_sort(train), train

    def _fit_transform_for_loop(self, train, x_cols):
        def _rename_col(df, col):
            df.columns = [col]
            return df.reset_index()

        res = []
        unq_vals = train[self.fold_col].unique()
        if not isinstance(unq_vals, (cp.ndarray, np.ndarray)):
            unq_vals = unq_vals.values_host
        for f in unq_vals:
            mask = train[self.fold_col].values == f
            dg = train.loc[~mask].groupby(x_cols).agg({self.y_col: self.stat})
            dg = _rename_col(dg, self.out_col)
            res.append(train.loc[mask].merge(dg, on=x_cols, how="left"))
        res = cudf.concat(res, axis=0)
        self.encode_all = train.groupby(x_cols).agg({self.y_col: self.stat})
        self.encode_all = _rename_col(self.encode_all, self.out_col)
        return self._impute_and_sort(res), train

    def _make_y_column(self, y):
        """
        Create a target column given y
        """
        if isinstance(y, cudf.Series) or isinstance(y, pandas.Series):
            return y.values
        elif isinstance(y, cp.ndarray) or isinstance(y, np.ndarray):
            if len(y.shape) == 1:
                return y
            elif y.shape[1] == 1:
                return y[:, 0]
            else:
                raise ValueError(
                    f"Input of shape {y.shape} " "is not a 1-D array."
                )
        else:
            raise TypeError(
                f"Input of type {type(y)} is not cudf.Series, "
                "or pandas.Series"
                "or numpy.ndarray"
                "or cupy.ndarray"
            )

    def _make_fold_column(self, len_train, fold_ids):
        """
        Create a fold id column for each split
        """

        if self.split_method == "random":
            return cp.random.randint(0, self.n_folds, len_train)
        elif self.split_method == "continuous":
            return (
                cp.arange(len_train) / (len_train / self.n_folds)
            ) % self.n_folds
        elif self.split_method == "interleaved":
            return cp.arange(len_train) % self.n_folds
        elif self.split_method == "customize":
            if fold_ids is None:
                raise ValueError(
                    "fold_ids can't be None"
                    "since split_method is set to"
                    "'customize'."
                )
            return fold_ids
        else:
            msg = (
                "split_method should be either 'random'"
                " or 'continuous' or 'interleaved', "
                "got {0}.".format(self.split_method)
            )
            raise ValueError(msg)

    def _compute_output(self, df_sum, df_count, cols, y_col, y_col2=None):
        """
        Compute the output encoding based on aggregated sum and count
        """
        df_sum = df_sum.merge(df_count, on=cols, how="left")
        smooth = self.smooth
        df_sum[self.out_col] = (df_sum[f"{y_col}_x"] + smooth * self.mean) / (
            df_sum[f"{y_col}_y"] + smooth
        )
        if self.stat == "var":
            df_sum[self.out_col2] = (
                df_sum[f"{y_col2}_x"] + smooth * self.mean2
            ) / (df_sum[f"{y_col2}_y"] + smooth)
            df_sum[self.out_col] = (
                df_sum[self.out_col2] - df_sum[self.out_col] ** 2
            )
            df_sum[self.out_col] = (
                df_sum[self.out_col]
                * df_sum[f"{y_col2}_y"]
                / (df_sum[f"{y_col2}_y"] - 1)
            )
        return df_sum

    def _groupby_agg(self, train, x_cols, op, y_cols):
        """
        Compute aggregated value of each fold and overall dataframe
        grouped by `x_cols` and agg by `op`
        """
        cols = [self.fold_col] + x_cols
        df_each_fold = train.groupby(cols, as_index=False).agg(
            {y_col: op for y_col in y_cols}
        )
        df_all = df_each_fold.groupby(x_cols, as_index=False).agg(
            {y_col: "sum" for y_col in y_cols}
        )

        df_each_fold = df_each_fold.merge(df_all, on=x_cols, how="left")
        for y_col in y_cols:
            df_each_fold[f"{y_col}_x"] = (
                df_each_fold[f"{y_col}_y"] - df_each_fold[f"{y_col}_x"]
            )
        return df_each_fold, df_all

    def _check_is_fitted(self):
        if not self._fitted or self.train is None:
            msg = (
                "This LabelEncoder instance is not fitted yet. Call 'fit' "
                "with appropriate arguments before using this estimator."
            )
            raise NotFittedError(msg)

    def _is_train_df(self, df):
        """
        Return True if the dataframe `df` is the training dataframe, which
        is used in `fit_transform`
        """
        if len(df) != len(self.train):
            return False
        self.train = self.train.sort_values(self.id_col).reset_index(drop=True)
        for col in df.columns:
            if col not in self.train.columns:
                raise ValueError(
                    f"Input column {col} " "is not in train data."
                )
            if not (df[col] == self.train[col]).all():
                return False
        return True

    def _impute_and_sort(self, df):
        """
        Impute and sort the result encoding in the same row order as input
        """
        df[self.out_col] = df[self.out_col].nans_to_nulls()
        df[self.out_col] = df[self.out_col].fillna(self.y_stat_val)
        df = df.sort_values(self.id_col)
        res = df[self.out_col].values.copy()
        if self.output_type == "numpy":
            return cp.asnumpy(res)
        return res

    def _data_with_strings_to_cudf_dataframe(self, x):
        """
        Convert input data with strings to cudf dataframe.
        Supported data types are:
            1D or 2D numpy/cupy arrays
            pandas/cudf Series
            pandas/cudf DataFrame
        Input data could have one or more string columns.
        """
        if isinstance(x, cudf.DataFrame):
            df = x.copy()
        elif isinstance(x, cudf.Series):
            df = x.to_frame().copy()
        elif isinstance(x, cp.ndarray) or isinstance(x, np.ndarray):
            df = cudf.DataFrame()
            if len(x.shape) == 1:
                df[self.x_col] = x
            else:
                df = cudf.DataFrame(
                    x, columns=[f"{self.x_col}_{i}" for i in range(x.shape[1])]
                )
        elif isinstance(x, pandas.DataFrame):
            df = cudf.from_pandas(x)
        elif isinstance(x, pandas.Series):
            df = cudf.from_pandas(x.to_frame())
        else:
            raise TypeError(
                f"Input of type {type(x)} is not cudf.Series, cudf.DataFrame "
                "or pandas.Series or pandas.DataFrame"
                "or cupy.ndarray or numpy.ndarray"
            )
        df[self.id_col] = cp.arange(len(x))
        return df.reset_index(drop=True)

    def _get_output_type(self, x):
        """
        Infer output type if 'auto'
        """
        if self.output_type != "auto":
            return self.output_type
        if (
            isinstance(x, np.ndarray)
            or isinstance(x, pandas.DataFrame)
            or isinstance(x, pandas.Series)
        ):
            return "numpy"
        return "cupy"

    @classmethod
    def _get_param_names(cls):
        return [
            "n_folds",
            "smooth",
            "seed",
            "split_method",
        ]

    def get_params(self, deep=False):
        """
        Returns a dict of all params owned by this class.
        """
        params = dict()
        variables = self._get_param_names()
        for key in variables:
            var_value = getattr(self, key, None)
            params[key] = var_value
        return params


%%time
from cuml.preprocessing.TargetEncoder import TargetEncoder
from sklearn.model_selection import KFold
from itertools import combinations
import time
import random
oof_cat_ranker = np.zeros((len(train)))
pred_cat_ranker = np.zeros((len(test)))
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
for i, (train_index, test_index) in enumerate(kf.split(train)):
    start = time.time()
    # train['y']=train['y'].rank(method='dense').astype(int)
    tmp_valid=train.iloc[test_index].reset_index(drop=True)
    tmp_train=train.iloc[train_index].reset_index(drop=True)
    tmp_test = test.copy()
    tmp_train['y'] = -1*tmp_train['efs_time'] 
    tmp_train.loc[tmp_train.efs==0,'y'] = -1e9
    tmp_valid['y']=tmp_valid['y'].rank(method='dense').astype(int)
    tmp_train['y']=tmp_train['y'].rank(method='dense').astype(int)
    TE2 = []
    train_features = []
    valid_features = []
    test_features = []
    for c0 in yes:
        c = c0[:-1]
        x = int(c0[-1])
        
        if x == 0:
            TE = TargetEncoder2(n_folds=25, smooth=5, split_method='random', stat='mean')
            n = "TE1-" + "-".join(c)
            train_col = TE.fit_transform(tmp_train[c], tmp_train['efs'])
            valid_col = TE.transform(tmp_valid[c])
            test_col = TE.transform(tmp_test[c])
        elif x == 1:
            TE = TargetEncoder2(n_folds=25, smooth=5, split_method='random', stat='mean')
            n = "TE2-" + "-".join(c)
            train_col = TE.fit_transform(tmp_train[c], tmp_train['efs_time'])
            valid_col = TE.transform(tmp_valid[c])
            test_col = TE.transform(tmp_test[c])
        elif x == 2:
            TE = TargetEncoder2(n_folds=25, smooth=0, split_method='random', stat='min')
            n = "TE3-" + "-".join(c)
            train_col = TE.fit_transform(tmp_train[c], tmp_train['efs_time'])
            valid_col = TE.transform(tmp_valid[c])
            test_col = TE.transform(tmp_test[c])
        elif x == 3:
            TE = TargetEncoder2(n_folds=25, smooth=0, split_method='random', stat='max')
            n = "TE4-" + "-".join(c)
            train_col = TE.fit_transform(tmp_train[c], tmp_train['efs_time'])
            valid_col = TE.transform(tmp_valid[c])
            test_col = TE.transform(tmp_test[c])
        elif x == 4:
            TE = TargetEncoder2(n_folds=25, smooth=0, split_method='random', stat='nunique')
            n = "TE5-" + "-".join(c)
            train_col = TE.fit_transform(tmp_train[c], tmp_train['efs_time'])
            valid_col = TE.transform(tmp_valid[c])
            test_col = TE.transform(tmp_test[c])
        # Append the new columns to the lists
        if not n in TE2:
            train_features.append(pd.DataFrame({n: train_col}, index=tmp_train.index))
            valid_features.append(pd.DataFrame({n: valid_col}, index=tmp_valid.index))
            test_features.append(pd.DataFrame({n: test_col}, index=tmp_test.index))
            TE2.append(n)
            print(f"{n}, ", end="")
    tmp_train = pd.concat([tmp_train] + train_features, axis=1)
    tmp_valid = pd.concat([tmp_valid] + valid_features, axis=1)
    tmp_test = pd.concat([tmp_test] + test_features, axis=1)
    tmp_valid['group'] = list(np.arange(len(tmp_valid)//2))*2
    tmp_valid['ranked_target'] = tmp_valid.groupby('group')['y'].rank(method='dense').astype(int) - 1
    
    tmp_train['group'] = np.repeat( np.arange(len(tmp_train)//2),2 )
    tmp_train['ranked_target'] = tmp_train.groupby('group')['y'].rank(method='dense').astype(int) - 1
    X_valid=tmp_valid[FEATURES+TE2]
    # X_valid=tmp_valid[FEATURES]
    y_valid=tmp_valid['y']
    X_train=tmp_train[FEATURES+TE2]
    # X_train=tmp_train[FEATURES]
    y_train=tmp_train['y']

    # indices = tmp_train.index.values
    # efs = tmp_train["efs"].values
    
    # # 目標ペア数（100万個のペアを生成したい）
    # target_pair_count = 1000000
    
    # # サンプルされたペアを格納するリスト
    # pairs = set()  # 重複を防ぐために set を使用
    
    # # サンプリングを繰り返し
    # while len(pairs) < target_pair_count:
    #     # 2つのインデックスをランダムに選択
    #     i, j = random.sample(range(len(indices)), 2)
        
    #     # `efs` が 0 同士のペアを除外
    #     if efs[i] != 0 or efs[j] != 0:
    #         # ペアが逆順でも同一とみなす
    #         if (indices[i], indices[j]) not in pairs and (indices[j], indices[i]) not in pairs:
    #             pairs.add((indices[i], indices[j]))  # ペアをセットに追加
    # pairs=list(pairs)
    train_pool = Pool(
        data=X_train, 
        label=y_train,
        group_id=[1] * len(X_train),
        cat_features=CATS,
    )
    
    
    # indices = tmp_valid.index.values
    # efs = tmp_valid["efs"].values
    
    # # 目標ペア数（100万個のペアを生成したい）
    # target_pair_count = 1000000
    
    # # サンプルされたペアを格納するリスト
    # pairs = set()  # 重複を防ぐために set を使用
    
    # # サンプリングを繰り返し
    # while len(pairs) < target_pair_count:
    #     # 2つのインデックスをランダムに選択
    #     i, j = random.sample(range(len(indices)), 2)
        
    #     # `efs` が 0 同士のペアを除外
    #     if efs[i] != 0 or efs[j] != 0:
    #         # ペアが逆順でも同一とみなす
    #         if (indices[i], indices[j]) not in pairs and (indices[j], indices[i]) not in pairs:
    #             pairs.add((indices[i], indices[j]))  # ペアをセットに追加
    # pairs=list(pairs)
    valid_pool = Pool(
        data=X_valid, 
        label=y_valid,
        group_id=[1] * len(X_valid),
        cat_features=CATS,
    )
    test_pool = Pool(
        data=tmp_test[FEATURES+TE2], 
        cat_features=CATS
    )
    params = {
        #'loss_function': 'YetiRankPairwise',  # Pairwise ranking objective
        'loss_function': 'PairLogitPairwise:max_pairs=500000',
        #'loss_function': 'PairLogit',
        'depth': 5,
        'colsample_bylevel': 0.8,
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.5,
        'learning_rate': 0.0025,#0.03
        'eval_metric': 'NDCG',
        'task_type': 'GPU',
        'use_best_model' : False,
        # 'one_hot_max_size':20,
    }
    model = CatBoostRanker(
        iterations=1_500,
        **params
    )
    elapsed = (time.time()-start)/60.
    print(f"### Created Pool (elapsed {elapsed:.2f} minutes) ###")
    print("#"*25)
    start = time.time()
    model.fit(
       train_pool,
       eval_set=valid_pool,
       verbose=100
    )
    elapsed = (time.time()-start)/60.
    p = model.predict(valid_pool)
    oof_cat_ranker[test_index] = rankdata(p)
    y_true = train.loc[test_index][["ID","efs","efs_time","race_group"]].copy()
    y_pred = train.loc[test_index][["ID"]].copy()
    y_pred["prediction"] = p
    s = score(y_true.copy(), y_pred.copy(), "ID")

    print(f" => CV = {s} (elapsed {elapsed:.1f} minutes)")
    p2 = model.predict(test_pool)
    pred_cat_ranker += p2
    
    


# %%time
# from cuml.preprocessing.TargetEncoder import TargetEncoder
# from sklearn.model_selection import KFold
# from itertools import combinations
# import time
# import random
# oof_cat_ranker = np.zeros((len(train)))
# FOLDS = 10
# kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
# for i, (train_index, test_index) in enumerate(kf.split(train)):
#     start = time.time()
#     # train['y']=train['y'].rank(method='dense').astype(int)
#     tmp_valid=train.iloc[test_index].reset_index(drop=True)
#     tmp_train=train.iloc[train_index].reset_index(drop=True)
#     tmp_valid['group'] = list(np.arange(len(tmp_valid)//2))*2
#     tmp_valid['ranked_target'] = tmp_valid.groupby('group')['y'].rank(method='dense').astype(int) - 1
#     tmp_train['y2'] = -1*tmp_train['efs_time'] 
#     tmp_train.loc[tmp_train.efs==0,'y2'] = -1e9
#     idx1 = tmp_train.loc[tmp_train.efs==1].index.values
#     idx0 = tmp_train.loc[tmp_train.efs==0].index.values
    
#     idx_a = np.random.choice(idx1,1_000,replace=True)
#     #idx_a1 = np.repeat(idx1,len(idx1))
#     #idx_a2 = np.tile(idx1,len(idx1))
#     #idx_a = np.array( [elem for pair in zip(idx_a1, idx_a2) for elem in pair] )
    
#     idx_b = np.random.choice(idx1,1_000,replace=True)
#     idx_c = np.random.choice(idx0,1_000,replace=True)
#     idx_d = np.array( [elem for pair in zip(idx_b, idx_c) for elem in pair] )
#     #idx_b1 = np.repeat(idx1,len(idx0))
#     #idx_b2 = np.tile(idx0,len(idx1))
#     #idx_d = np.array( [elem for pair in zip(idx_b1, idx_b2) for elem in pair] )

#     idx_e = np.concatenate([idx_a,idx_d])
#     tmp_train = tmp_train.iloc[idx_e].reset_index(drop=True)
    
#     tmp_train['x'] = tmp_train['efs_time'].diff(1)
#     x = tmp_train.x.values
#     x[::2] = x[1::2]
#     tmp_train.x = x
    
#     tmp_train['x2'] = tmp_train['efs'].diff(1)
#     x2 = tmp_train.x2.values
#     x2[::2] = x2[1::2]
#     tmp_train.x2 = x2
#     tmp_train['x3'] = tmp_train['ID'].diff(1)
#     x3 = tmp_train.x3.values
#     x3[::2] = x3[1::2]
#     tmp_train.x3 = x3
    
#     before = tmp_train.shape
#     tmp_train = tmp_train.loc[(tmp_train.x3!=0)&((tmp_train.x>0)|(tmp_train.x2==0))].reset_index(drop=True)
#     after = tmp_train.shape
#     tmp_train['group'] = np.repeat( np.arange(len(tmp_train)//2),2 )
#     tmp_train['ranked_target'] = tmp_train.groupby('group')['y2'].rank(method='dense').astype(int) - 1
#     # tmp_valid['y']=tmp_valid['y'].rank(method='dense').astype(int)
#     # tmp_train['y']=tmp_train['y'].rank(method='dense').astype(int)
#     TE2 = []
#     train_features = []
#     valid_features = []
#     # test_features = []
#     for c0 in yes:
#         c = c0[:-1]
#         x = int(c0[-1])
        
#         if x == 0:
#             TE = TargetEncoder2(n_folds=25, smooth=5, split_method='random', stat='mean')
#             n = "TE1-" + "-".join(c)
#             train_col = TE.fit_transform(tmp_train[c], tmp_train['efs'])
#             valid_col = TE.transform(tmp_valid[c])
#             # test_col = TE.transform(tmp_test[c])
#         elif x == 1:
#             TE = TargetEncoder2(n_folds=25, smooth=5, split_method='random', stat='mean')
#             n = "TE2-" + "-".join(c)
#             train_col = TE.fit_transform(tmp_train[c], tmp_train['efs_time'])
#             valid_col = TE.transform(tmp_valid[c])
#             # test_col = TE.transform(tmp_test[c])
#         elif x == 2:
#             TE = TargetEncoder2(n_folds=25, smooth=0, split_method='random', stat='min')
#             n = "TE3-" + "-".join(c)
#             train_col = TE.fit_transform(tmp_train[c], tmp_train['efs_time'])
#             valid_col = TE.transform(tmp_valid[c])
#             # test_col = TE.transform(tmp_test[c])
#         elif x == 3:
#             TE = TargetEncoder2(n_folds=25, smooth=0, split_method='random', stat='max')
#             n = "TE4-" + "-".join(c)
#             train_col = TE.fit_transform(tmp_train[c], tmp_train['efs_time'])
#             valid_col = TE.transform(tmp_valid[c])
#             # test_col = TE.transform(tmp_test[c])
#         elif x == 4:
#             TE = TargetEncoder2(n_folds=25, smooth=0, split_method='random', stat='nunique')
#             n = "TE5-" + "-".join(c)
#             train_col = TE.fit_transform(tmp_train[c], tmp_train['efs_time'])
#             valid_col = TE.transform(tmp_valid[c])
#             # test_col = TE.transform(tmp_test[c])
#         # Append the new columns to the lists
#         if not n in TE2:
#             train_features.append(pd.DataFrame({n: train_col}, index=tmp_train.index))
#             valid_features.append(pd.DataFrame({n: valid_col}, index=tmp_valid.index))
#             # test_features.append(pd.DataFrame({n: test_col}, index=tmp_test.index))
#             TE2.append(n)
#             print(f"{n}, ", end="")
#     tmp_train = pd.concat([tmp_train] + train_features, axis=1)
#     tmp_valid = pd.concat([tmp_valid] + valid_features, axis=1)
    
#     tmp_valid['group'] = list(np.arange(len(tmp_valid)//2))*2
#     tmp_valid['ranked_target'] = tmp_valid.groupby('group')['y'].rank(method='dense').astype(int) - 1
    
#     tmp_train['group'] = np.repeat( np.arange(len(tmp_train)//2),2 )
#     tmp_train['ranked_target'] = tmp_train.groupby('group')['y'].rank(method='dense').astype(int) - 1
#     X_valid=tmp_valid[FEATURES+TE2]
#     # X_valid=tmp_valid[FEATURES]
#     y_valid=tmp_valid['ranked_target']
#     X_train=tmp_train[FEATURES+TE2]
#     # X_train=tmp_train[FEATURES]
#     y_train=tmp_train['ranked_target']

#     # indices = tmp_train.index.values
#     # efs = tmp_train["efs"].values
    
#     # # 目標ペア数（100万個のペアを生成したい）
#     # target_pair_count = 1000000
    
#     # # サンプルされたペアを格納するリスト
#     # pairs = set()  # 重複を防ぐために set を使用
    
#     # # サンプリングを繰り返し
#     # while len(pairs) < target_pair_count:
#     #     # 2つのインデックスをランダムに選択
#     #     i, j = random.sample(range(len(indices)), 2)
        
#     #     # `efs` が 0 同士のペアを除外
#     #     if efs[i] != 0 or efs[j] != 0:
#     #         # ペアが逆順でも同一とみなす
#     #         if (indices[i], indices[j]) not in pairs and (indices[j], indices[i]) not in pairs:
#     #             pairs.add((indices[i], indices[j]))  # ペアをセットに追加
#     # pairs=list(pairs)
#     train_pool = Pool(
#         data=X_train, 
#         label=y_train,
#         group_id=[i // 2 for i in range(len(X_train))],
#         cat_features=CATS,
#     )
    
    
#     # indices = tmp_valid.index.values
#     # efs = tmp_valid["efs"].values
    
#     # # 目標ペア数（100万個のペアを生成したい）
#     # target_pair_count = 1000000
    
#     # # サンプルされたペアを格納するリスト
#     # pairs = set()  # 重複を防ぐために set を使用
    
#     # # サンプリングを繰り返し
#     # while len(pairs) < target_pair_count:
#     #     # 2つのインデックスをランダムに選択
#     #     i, j = random.sample(range(len(indices)), 2)
        
#     #     # `efs` が 0 同士のペアを除外
#     #     if efs[i] != 0 or efs[j] != 0:
#     #         # ペアが逆順でも同一とみなす
#     #         if (indices[i], indices[j]) not in pairs and (indices[j], indices[i]) not in pairs:
#     #             pairs.add((indices[i], indices[j]))  # ペアをセットに追加
#     # pairs=list(pairs)
#     valid_pool = Pool(
#         data=X_valid, 
#         label=y_valid,
#         group_id=[i // 2 for i in range(len(X_valid))],
#         cat_features=CATS,
#     )
#     params = {
#         #'loss_function': 'YetiRankPairwise',  # Pairwise ranking objective
#         'loss_function': 'PairLogitPairwise:max_pairs=1000000',
#         #'loss_function': 'PairLogit',
#         'depth': 5,
#         'colsample_bylevel': 0.5,
#         'bootstrap_type': 'Bernoulli',
#         'subsample': 0.8,
#         'learning_rate': 0.03,#0.03
#         'eval_metric': 'NDCG',
#         'task_type': 'GPU',
#         'use_best_model' : False,
#         # 'one_hot_max_size':7,
#     }
#     model = CatBoostRanker(
#         iterations=1_500,
#         **params
#     )
#     elapsed = (time.time()-start)/60.
#     print(f"### Created Pool (elapsed {elapsed:.2f} minutes) ###")
#     print("#"*25)
#     start = time.time()
#     model.fit(
#        train_pool,
#        eval_set=valid_pool,
#        verbose=100
#     )
#     elapsed = (time.time()-start)/60.
#     p = model.predict(valid_pool)
#     oof_cat_ranker[test_index] = rankdata(p)
#     y_true = train.loc[test_index][["ID","efs","efs_time","race_group"]].copy()
#     y_pred = train.loc[test_index][["ID"]].copy()
#     y_pred["prediction"] = p
#     s = score(y_true.copy(), y_pred.copy(), "ID")

#     print(f" => CV = {s} (elapsed {elapsed:.1f} minutes)")
    
    


# #post-processing
# mask = oof_xgb_c == 1

# # そのインデックスに対して oof_xgb を +0.1 する

# oof_xgb[mask]+=0.1


# oof_nn[mask]+=0.1


# oof_cat3[mask]+=0.1


from scipy.stats import rankdata

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_cat_ranker
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost KaplanMeier =",m)


sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")

sub['prediction'] = rankdata(pred_cat_ranker)


sub.to_csv("submission.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()

