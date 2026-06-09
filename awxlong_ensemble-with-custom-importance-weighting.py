!pip install -q /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install -q /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install -q /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install -q /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install -q /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
print("Test shape:", test.shape )

train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
print("Train shape:",train.shape)
train.head()


train['race_group'].value_counts()


plt.hist(train.loc[train.efs==1,"efs_time"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"efs_time"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Time of Observation, efs_time")
plt.ylabel("Density")
plt.title("Times of Observation. Either time to event, or time observed without event.")
plt.legend()
plt.show()


from lifelines import KaplanMeierFitter
from scipy.stats import rankdata
from sklearn.preprocessing import quantile_transform

def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return y

def transform_rank_log(time, event):
    """Transform the target by stretching the range of eventful efs_times and compressing the range of event_free efs_times
    
    From https://www.kaggle.com/code/cdeotte/nn-mlp-baseline-cv-670-lb-676"""
    transformed = time.values.copy()
    mx = transformed[event == 1].max() # last patient who dies
    mn = transformed[event == 0].min() # first patient who survives
    transformed[event == 0] = time[event == 0] + mx - mn
    transformed = rankdata(transformed)
    transformed[event == 0] += len(transformed) * 2
    transformed = transformed / transformed.max()
    transformed = np.log(transformed)
    return - transformed

def transform_quantile(time, event):
    """Transform the target by stretching the range of eventful efs_times and compressing the range of event_free efs_times
    
    From https://www.kaggle.com/code/ambrosm/esp-eda-which-makes-sense"""
    transformed = np.full(len(time), np.nan)
    transformed_dead = quantile_transform(- time[event == 1].values.reshape(-1, 1)).ravel()
    transformed[event == 1] = transformed_dead
    transformed[event == 0] = transformed_dead.min() - 0.3
    return transformed


race_group=sorted(train['race_group'].unique())
for race in race_group:
    # KP Meier
    train.loc[train['race_group']==race,"y"] = transform_survival_probability(train[train['race_group']==race], time_col='efs_time', event_col='efs')
    gap = 0.7*(train.loc[(train['race_group']==race)&(train['efs']==0)]['y'].max()-train.loc[(train['race_group']==race)&(train['efs']==1)]['y'].min())/2
    train.loc[(train['race_group']==race)&(train['efs']==0),'y']-=gap
    
    # Quantile KP Meier
    train.loc[train['race_group']==race,"quantile_kp_meier"] = transform_quantile(time = train[train['race_group']==race].efs_time, event=train[train['race_group']==race].efs)
    
    # Rank Loss KP Meier
    train.loc[train['race_group']==race,"rank_kp_meier"] = transform_rank_log(time = train[train['race_group']==race].efs_time, event=train[train['race_group']==race].efs)
    
plt.hist(train.loc[train.efs==1,"y"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"y"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Transformed Target y")
plt.ylabel("Density")
plt.title("KaplanMeier Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()

plt.hist(train.loc[train.efs==1,"quantile_kp_meier"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"quantile_kp_meier"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Transformed Quantile Target y")
plt.ylabel("Density")
plt.title("Quantile KaplanMeier Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()

plt.hist(train.loc[train.efs==1,"rank_kp_meier"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"rank_kp_meier"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Transformed Rank Log Target y")
plt.ylabel("Density")
plt.title("Rank Log KaplanMeier Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()


MIN_YEAR = train['year_hct'].min() # 2008
nunique2=[col for col in train.columns if train[col].nunique()==2 and col!='efs'] 
#nunique<50
nunique50=[col for col in train.columns if train[col].nunique()<50 and col not in ['efs','weight', 'year_hct']]+['age_group','dri_score_NA'] + ['year_hct_relative']

def FE(df):
    print("< deal with outlier >")
    df['nan_value_each_row'] = df.isnull().sum(axis=1)
    #year_hct=2020 only 4 rows.
    print("<convert to year_hct relative")
    df['year_hct_relative'] = df['year_hct'] - MIN_YEAR
    df.drop(columns=['year_hct'], inplace=True)
    # df['year_hct']=df['year_hct'].replace(2020,2019)
    df['age_group']=df['age_at_hct']//10
    #karnofsky_score 40 only 10 rows.
    df['karnofsky_score']=df['karnofsky_score'].replace(40,50)
    #hla_high_res_8=2 only 2 rows.
    df['hla_high_res_8']=df['hla_high_res_8'].replace(2,3)
    #hla_high_res_6=0 only 1 row.
    df['hla_high_res_6']=df['hla_high_res_6'].replace(0,2)
    #hla_high_res_10=3 only 1 row.
    df['hla_high_res_10']=df['hla_high_res_10'].replace(3,4)
    #hla_low_res_8=2 only 1 row.
    df['hla_low_res_8']=df['hla_low_res_8'].replace(2,3)
    df['dri_score']=df['dri_score'].replace('Missing disease status','N/A - disease not classifiable')
    df['dri_score_NA']=df['dri_score'].apply(lambda x:int('N/A' in str(x)))
    for col in ['diabetes','pulm_moderate','cardiac']:
        df.loc[df[col].isna(),col]='Not done'

    print("< cross feature >")
    df['donor_age-age_at_hct']=df['donor_age']-df['age_at_hct']
    df['comorbidity_score+karnofsky_score']=df['comorbidity_score']+df['karnofsky_score']
    df['comorbidity_score-karnofsky_score']=df['comorbidity_score']-df['karnofsky_score']
    df['comorbidity_score*karnofsky_score']=df['comorbidity_score']*df['karnofsky_score']
    df['comorbidity_score/karnofsky_score']=df['comorbidity_score']/df['karnofsky_score']
    
    print("< fillna >")
    df[nunique50]=df[nunique50].astype(str).fillna('NaN')
    
    print("< combine category feature >")
    for i in range(len(nunique2)):
        for j in range(i+1,len(nunique2)):
            df[nunique2[i]+nunique2[j]]=df[nunique2[i]].astype(str)+df[nunique2[j]].astype(str)
    
    # print("< drop useless columns >")
    # df.drop(['ID'],axis=1,inplace=True,errors='ignore')
    return df

train = FE(train)
test = FE(test)


RMV = ["ID","efs","efs_time","y", 'quantile_kp_meier', 'rank_kp_meier']
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


# combined = pd.concat([train,test],axis=0,ignore_index=True)
# print("Combined data shape:", combined.shape )
# # Store the original race group names and their corresponding codes
# race_group_categories = combined['race_group'].astype('category').cat.categories

# # Label encode race_group and capture the mapping
# combined['race_group'], _ = combined['race_group'].factorize()
# race_code_to_name = {code: name for code, name in enumerate(race_group_categories)}

# # Print the mapping for verification
# print("Race group encoding mapping:")
# for code, name in race_code_to_name.items():
#     print(f"{code}: {name}")


combined = pd.concat([train,test],axis=0,ignore_index=True)
#print("Combined data shape:", combined.shape )
# Store the original race group names and their corresponding codes
# race_group_categories = combined['race_group'].astype('category').cat.categories

# # Label encode race_group and capture the mapping
# combined['race_group'], _ = combined['race_group'].factorize()
# race_code_to_name = {code: name for code, name in enumerate(race_group_categories)}

# # Print the mapping for verification
# print("Race group encoding mapping:")
# for code, name in race_code_to_name.items():
#     print(f"{code}: {name}")
# # LABEL ENCODE CATEGORICAL FEATURES
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


from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
print("Using XGBoost version",xgb.__version__)


race_code_to_name = {
    0: 'American Indian or Alaska Native',
1: 'Asian',
2: 'Black or African-American',
3: 'More than one race',
4: 'Native Hawaiian or other Pacific Islander',
5: 'White'

}
# 0: American Indian or Alaska Native
# 1: Asian
# 2: Black or African-American
# 3: More than one race
# 4: Native Hawaiian or other Pacific Islander
# 5: White





# Define target distribution and calculate importance weights
target_props = {
    'White': 1,  # Adjusted to realistic proportion from paper
    'Black or African-American': 0.09,
    'Asian': 0.04,
    'Native Hawaiian or other Pacific Islander': 0.02,
    'American Indian or Alaska Native': 0.03,
    'More than one race': 0.02
}

# Calculate importance weights for each sample
train_props = 1/len(target_props)  # Balanced training assumption
# Calculate importance weights using the correct mapping
importance_weights = (
    train['race_group']
    .map(lambda x:  train_props/ target_props[race_code_to_name[x]])
    .astype(float)
    .values
)

# Clip extreme weights for stability
# importance_weights = np.clip(importance_weights, 0.1, 5)

importance_weights


IGNORE = ['psych_disturb', 'graft_type', 'prod_type', 'in_vivo_tcd', 'dri_score_NA', 'rituximab']
FEATURES = [i for i in FEATURES if i not in IGNORE]
CATS = [i for i in CATS if i not in IGNORE]
len(FEATURES), len(CATS)


lgb_params={"boosting_type": "gbdt","metric": 'mae',
            'random_state': 42,  "max_depth": 9,"learning_rate": 0.1,
            "n_estimators": 768,"colsample_bytree": 0.6,"colsample_bynode": 0.6,
            "verbose": -1,"reg_alpha": 0.2,
            "reg_lambda": 5,"extra_trees":True,'num_leaves':64,"max_bin":255,
            'importance_type': 'gain',#better than 'split'
            'device':'gpu','gpu_use_dp':True
           }

cat_params={'random_state':42,'eval_metric' : 'MAE',
            'bagging_temperature': 0.50,'iterations': 650,
            'learning_rate': 0.1,'max_depth': 8,
            'l2_leaf_reg': 1.25,'min_data_in_leaf': 24,
            'random_strength' : 0.25, 'verbose': 0,
            'task_type':'CPU',
            }
xgb_params={'random_state': 42, 'n_estimators': 256, 
            'learning_rate': 0.1, 'max_depth': 6,
            'reg_alpha': 0.08, 'reg_lambda': 0.8, 
            'subsample': 0.95, 'colsample_bytree': 0.6, 
            'min_child_weight': 3,'early_stopping_rounds':1024,
             'enable_categorical':True,'tree_method':'gpu_hist'
            }


xgb_race_params = {'random_state':42, 'n_estimators': 1799, 'max_depth': 4, 'learning_rate': 0.015986963575324597, 'subsample': 0.679547272652044, 'colsample_bytree': 0.8141976825388598, 'min_child_weight': 7, 'reg_lambda': 0.2988980595840879, 'reg_alpha': 0.1419698467602532, 'enable_categorical':True,'tree_method':'hist', 'device':'cuda', 'early_stopping_rounds':1024}


%%time
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)


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

    # Get weights for current training fold
    fold_weights = importance_weights[train_index]
    
    model_xgb = XGBRegressor(
        **xgb_race_params
    )
    
    model_xgb.fit(
        x_train, y_train,
        sample_weight=fold_weights,  # Apply importance weights
        eval_set=[(x_valid, y_valid)],  
        verbose=500 
    )

    # model_xgb.save_model(f"xgb_kaggle_weights_{i}.bin")

    # INFER OOF
    oof_xgb[test_index] = model_xgb.predict(x_valid)
    # INFER TEST
    pred_xgb += model_xgb.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb /= FOLDS


from metric import score

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_xgb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost KaplanMeier =",m)


feature_importance = model_xgb.feature_importances_
importance_df = pd.DataFrame({
    "Feature": FEATURES,  # Replace FEATURES with your list of feature names
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 15))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost KaplanMeier Feature Importance")
plt.gca().invert_yaxis()  # Flip features for better readability
plt.show()


from catboost import CatBoostRegressor, CatBoostClassifier
import catboost as cb
print("Using CatBoost version",cb.__version__)


cat_race_params = {'random_state':42, 'bagging_temperature': 0.46163620004791045, 'iterations': 982, 'learning_rate': 0.13757438841964445, 'max_depth': 7, 'l2_leaf_reg': 4.895675709091795, 'min_data_in_leaf': 31, 'random_strength': 0.7333645049208688, 'eval_metric' : 'MAE', 'task_type' : 'CPU'} # trial 61 with value 0.67


%%time
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_cat = np.zeros(len(train))
pred_cat = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"y"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"y"]
    x_test = test[FEATURES].copy()

    # Get weights for current training fold
    fold_weights = importance_weights[train_index]
    
    model_cat = CatBoostRegressor(
        **cat_race_params
    )
    model_cat.fit(x_train,y_train,
              eval_set=(x_valid, y_valid),
              sample_weight=fold_weights,  # Apply importance weights
              cat_features=CATS,
              early_stopping_rounds=100,
              verbose=250)

    # INFER OOF
    oof_cat[test_index] = model_cat.predict(x_valid)
    # INFER TEST
    pred_cat += model_cat.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_cat /= FOLDS


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_cat
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for CatBoost KaplanMeier =",m)


feature_importance = model_cat.get_feature_importance()
importance_df = pd.DataFrame({
    "Feature": FEATURES, 
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 15))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("CatBoost KaplanMeier Feature Importance")
plt.gca().invert_yaxis()  # Flip features for better readability
plt.show()


from lightgbm import LGBMRegressor
import lightgbm as lgb
print("Using LightGBM version",lgb.__version__)


lgb_race_params = {'max_depth': 7, 'learning_rate': 0.015201335432187617, 'n_estimators': 1479, 'colsample_bytree': 0.40606089552680164, 'colsample_bynode': 0.7422814146406744, 'reg_alpha': 0.5706578926746176, 'reg_lambda': 5.686408171263541, 'num_leaves': 33, "boosting_type": "gbdt","metric": 'mae', 'random_state': 42,'importance_type': 'gain', "extra_trees":True, "max_bin":255}


FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_lgb = np.zeros(len(train))
pred_lgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"y"]    
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"y"]
    x_test = test[FEATURES].copy()

    # Get weights for current training fold
    fold_weights = importance_weights[train_index]
    
    model_lgb = LGBMRegressor(
        **lgb_race_params
    )
    model_lgb.fit(
        x_train, y_train,
        sample_weight=fold_weights,  # Apply importance weights
        eval_set=[(x_valid, y_valid)],
        categorical_feature=CATS,  # Specify categorical features
    )
    
    # INFER OOF
    oof_lgb[test_index] = model_lgb.predict(x_valid)
    # INFER TEST
    pred_lgb += model_lgb.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_lgb /= FOLDS


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_lgb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for LightGBM KaplanMeier =",m)


feature_importance = model_lgb.feature_importances_ 
importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 15))
plt.barh(importance_df["Feature"], importance_df["Importance"], color='skyblue')
plt.xlabel("Importance (Gain)")
plt.ylabel("Feature")
plt.title("LightGBM KaplanMeier Feature Importance")
plt.gca().invert_yaxis()  # Flip features for better readability
plt.show()


xgb_quantile_params = {'random_state':42, 'n_estimators': 1732, 'max_depth': 5, 'learning_rate': 0.012281557858698075, 'subsample': 0.7393452028316118, 'colsample_bytree': 0.5020340288862665, 'min_child_weight': 2, 'reg_lambda': 0.9941198189202145, 'reg_alpha': 0.13689353159386, 'enable_categorical':True,'tree_method':'hist', 'device':'cuda', 'early_stopping_rounds':1024} # trial 51 with 0.6731063939813396


%%time
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)


oof_xgb_quant = np.zeros(len(train))
pred_xgb_quant = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):
    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"quantile_kp_meier"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"quantile_kp_meier"]
    x_test = test[FEATURES].copy()

    # Get weights for current training fold
    fold_weights = importance_weights[train_index]
    
    model_xgb = XGBRegressor(
        **xgb_quantile_params
    )
    
    model_xgb.fit(
        x_train, y_train,
        sample_weight=fold_weights,  # Apply importance weights
        eval_set=[(x_valid, y_valid)],  
        verbose=500 
    )

    # model_xgb.save_model(f"xgb_kaggle_weights_{i}.bin")

    # INFER OOF
    oof_xgb_quant[test_index] = model_xgb.predict(x_valid)
    # INFER TEST
    pred_xgb_quant += model_xgb.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb_quant /= FOLDS


# from metric import score

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_xgb_quant
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost Quantile KaplanMeier =",m)


cat_quantile_params = {'random_state':42, 'bagging_temperature': 0.7451417507892211, 'iterations': 925, 'learning_rate': 0.12184110055030929, 'max_depth': 5, 'l2_leaf_reg': 7.9915859450648155, 'min_data_in_leaf': 49, 'random_strength': 0.46138124901842115, 'eval_metric' : 'MAE', 'task_type' : 'CPU'} # trial 24 with 0.67


%%time
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_cat_quant = np.zeros(len(train))
pred_cat_quant = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"quantile_kp_meier"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"quantile_kp_meier"]
    x_test = test[FEATURES].copy()

    # Get weights for current training fold
    fold_weights = importance_weights[train_index]
    
    model_cat = CatBoostRegressor(
        **cat_quantile_params
    )
    model_cat.fit(x_train,y_train,
              eval_set=(x_valid, y_valid),
              sample_weight=fold_weights,  # Apply importance weights
              cat_features=CATS,
              early_stopping_rounds=100,
              verbose=250)

    # INFER OOF
    oof_cat_quant[test_index] = model_cat.predict(x_valid)
    # INFER TEST
    pred_cat_quant += model_cat.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_cat_quant /= FOLDS


lgb_quantile_params = {'max_depth': 8, 'learning_rate': 0.011294930758885664, 'n_estimators': 951, 'colsample_bytree': 0.6300880012505659, 'colsample_bynode': 0.530630067506899, 'reg_alpha': 0.8874275150467659, 'reg_lambda': 4.2448482343518315, 'num_leaves': 63, "boosting_type": "gbdt","metric": 'mae', 'random_state': 42,'importance_type': 'gain', "extra_trees":True, "max_bin":255}


FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_lgb_quant = np.zeros(len(train))
pred_lgb_quant = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"quantile_kp_meier"]    
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"quantile_kp_meier"]
    x_test = test[FEATURES].copy()

    # Get weights for current training fold
    fold_weights = importance_weights[train_index]
    
    model_lgb = LGBMRegressor(
        **lgb_quantile_params
    )
    model_lgb.fit(
        x_train, y_train,
        sample_weight=fold_weights,  # Apply importance weights
        eval_set=[(x_valid, y_valid)],
        categorical_feature=CATS,  # Specify categorical features
    )
    
    # INFER OOF
    oof_lgb_quant[test_index] = model_lgb.predict(x_valid)
    # INFER TEST
    pred_lgb_quant += model_lgb.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_lgb_quant /= FOLDS


xgb_ranklog_params = {'random_state':42, 'n_estimators': 1790, 'max_depth': 3, 'learning_rate': 0.028636340198073568, 'subsample': 0.6978477605213024, 'colsample_bytree': 0.5501202649932605, 'min_child_weight': 3, 'reg_lambda': 0.6885645513731737, 'reg_alpha': 0.5284388429562188, 'enable_categorical':True,'tree_method':'hist', 'device':'cuda', 'early_stopping_rounds':1024} # trial  63 with 0.6684286852831739


%%time
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)


oof_xgb_rank = np.zeros(len(train))
pred_xgb_rank = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):
    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"rank_kp_meier"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"rank_kp_meier"]
    x_test = test[FEATURES].copy()

    # Get weights for current training fold
    fold_weights = importance_weights[train_index]
    
    model_xgb = XGBRegressor(
        **xgb_ranklog_params
    )
    
    model_xgb.fit(
        x_train, y_train,
        sample_weight=fold_weights,  # Apply importance weights
        eval_set=[(x_valid, y_valid)],  
        verbose=500 
    )

    # model_xgb.save_model(f"xgb_kaggle_weights_{i}.bin")

    # INFER OOF
    oof_xgb_rank[test_index] = model_xgb.predict(x_valid)
    # INFER TEST
    pred_xgb_rank += model_xgb.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb_rank /= FOLDS


cat_rank_params = {'random_state':42, 'bagging_temperature': 0.725639649924309, 'iterations': 905, 'learning_rate': 0.07529291232235219, 'max_depth': 10, 'l2_leaf_reg': 5.5105061580513155, 'min_data_in_leaf': 16, 'random_strength': 0.9315340290021121, 'eval_metric' : 'MAE', 'task_type' : 'CPU'}


%%time
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_cat_rank = np.zeros(len(train))
pred_cat_rank = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"rank_kp_meier"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"rank_kp_meier"]
    x_test = test[FEATURES].copy()

    # Get weights for current training fold
    fold_weights = importance_weights[train_index]
    
    model_cat = CatBoostRegressor(
        **cat_quantile_params
    )
    model_cat.fit(x_train,y_train,
              eval_set=(x_valid, y_valid),
              sample_weight=fold_weights,  # Apply importance weights
              cat_features=CATS,
              early_stopping_rounds=100,
              verbose=250)

    # INFER OOF
    oof_cat_rank[test_index] = model_cat.predict(x_valid)
    # INFER TEST
    pred_cat_rank += model_cat.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_cat_rank /= FOLDS


FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_lgb_rank = np.zeros(len(train))
pred_lgb_rank = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"rank_kp_meier"]    
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"rank_kp_meier"]
    x_test = test[FEATURES].copy()

    # Get weights for current training fold
    fold_weights = importance_weights[train_index]
    
    model_lgb = LGBMRegressor(
        **lgb_quantile_params
    )
    model_lgb.fit(
        x_train, y_train,
        sample_weight=fold_weights,  # Apply importance weights
        eval_set=[(x_valid, y_valid)],
        categorical_feature=CATS,  # Specify categorical features
    )
    
    # INFER OOF
    oof_lgb_rank[test_index] = model_lgb.predict(x_valid)
    # INFER TEST
    pred_lgb_rank += model_lgb.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_lgb_rank /= FOLDS


# SURVIVAL COX NEEDS THIS TARGET (TO DIGEST EFS AND EFS_TIME)
train["efs_time2"] = train.efs_time.copy()
train.loc[train.efs==0,"efs_time2"] *= -1


FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_xgb_cox = np.zeros(len(train))
pred_xgb_cox = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"efs_time2"]    
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"efs_time2"]
    x_test = test[FEATURES].copy()

    # Get weights for current training fold
    fold_weights = importance_weights[train_index]

    model_xgb_cox = XGBRegressor(
        device="cuda",
        max_depth=3,  
        colsample_bytree=0.5,  
        subsample=0.8,  
        n_estimators=2000,  
        learning_rate=0.02,  
        enable_categorical=True,
        min_child_weight=80,
        objective='survival:cox',
        eval_metric='cox-nloglik',
    )
    model_xgb_cox.fit(
        x_train, y_train,
        sample_weight=fold_weights,  # Apply importance weights
        eval_set=[(x_valid, y_valid)],  
        verbose=500  
    )
    
    # INFER OOF
    oof_xgb_cox[test_index] = model_xgb_cox.predict(x_valid)
    # INFER TEST
    pred_xgb_cox += model_xgb_cox.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb_cox /= FOLDS


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_xgb_cox
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost Survival:Cox =",m)


feature_importance = model_xgb_cox.feature_importances_
importance_df = pd.DataFrame({
    "Feature": FEATURES,  # Replace FEATURES with your list of feature names
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 15))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Survival:Cox Feature Importance")
plt.gca().invert_yaxis()  # Flip features for better readability
plt.show()


cox2_params = {
        'grow_policy': 'Lossguide',
        'min_child_samples': 2,
        'loss_function': 'Cox',
        'learning_rate': 0.03,
        'random_state': 42,
        'task_type': 'CPU',
        'num_trees': 6000,
        'reg_lambda': 8.0,
        'num_leaves': 32,
        'depth': 8
    }


FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_cat_cox = np.zeros(len(train))
pred_cat_cox = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"efs_time2"]    
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"efs_time2"]
    x_test = test[FEATURES].copy()

    # Get weights for current training fold
    fold_weights = importance_weights[train_index]
    
    model_cat_cox = CatBoostRegressor(
        **cox2_params
    )
    model_cat_cox.fit(x_train,y_train,
              sample_weight=fold_weights,  # Apply importance weights
              eval_set=(x_valid, y_valid),
              cat_features=CATS,
              verbose=100)
    
    # INFER OOF
    oof_cat_cox[test_index] = model_cat_cox.predict(x_valid)
    # INFER TEST
    pred_cat_cox += model_cat_cox.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_cat_cox /= FOLDS


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_cat_cox
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for CatBoost Survival:Cox =",m)


feature_importance = model_cat_cox.get_feature_importance()
importance_df = pd.DataFrame({
    "Feature": FEATURES, 
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 15))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("CatBoost Survival:Cox Feature Importance")
plt.gca().invert_yaxis()  # Flip features for better readability
plt.show()


from scipy.stats import rankdata 

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = rankdata(oof_xgb) + rankdata(oof_cat) + rankdata(oof_lgb)\
                     + rankdata(oof_xgb_quant) + rankdata(oof_cat_quant) + rankdata(oof_lgb_quant)\
                     + rankdata(oof_xgb_rank) + rankdata(oof_cat_rank) + rankdata(oof_lgb_rank)\
                     + rankdata(oof_xgb_cox) + rankdata(oof_cat_cox)
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for Ensemble =",m)


sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
# sub.prediction = rankdata(pred_xgb) + rankdata(pred_cat) + rankdata(pred_lgb) + rankdata(pred_xgb_quant) + rankdata(pred_cat_quant) + rankdata(pred_lgb_quant) + rankdata(pred_xgb_rank) + rankdata(pred_cat_rank) + rankdata(pred_lgb_rank) + rankdata(pred_xgb_cox) + rankdata(pred_cat_cox)

# Calculate ranks for each prediction
rank1 = rankdata(pred_xgb)
rank2 = rankdata(pred_cat)
rank3 = rankdata(pred_lgb)
rank4 = rankdata(pred_xgb_quant)
rank5 = rankdata(pred_cat_quant)
rank6 = rankdata(pred_lgb_quant)
rank7 = rankdata(pred_xgb_rank)
rank8 = rankdata(pred_cat_rank)
rank9 = rankdata(pred_lgb_rank)
rank10 = rankdata(pred_xgb_cox)
rank11 = rankdata(pred_cat_cox)

# Create DataFrame of ranks
rank_df = pd.DataFrame({
    'rank1': rank1,
    'rank2': rank2,
    'rank3': rank3,
    'rank4': rank4,
    'rank5': rank5,
    'rank6': rank6,
    'rank7': rank7,
    'rank8': rank8,
    'rank9': rank9,
    'rank10': rank10,
    'rank11': rank11
})

# Average the ranks along axis=1 (across columns)
ensemble_rank = rank_df.mean(axis=1)

# Assign the averaged ranks to the submission dataframe
sub.prediction = ensemble_rank

sub.to_csv("submission.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()

