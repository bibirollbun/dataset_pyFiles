# Installing libraries

!pip install -q /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install -q /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install -q /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install -q /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install -q /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


# Importing libraries

import numpy as np 
import pandas as pd 

from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter

from sklearn.model_selection import train_test_split, KFold

import warnings
warnings.filterwarnings("ignore")


# Helper functions

# evaluation funciton
def score(dfl) -> float:

    dfl.reset_index(inplace=True)
    race_dict = dict(dfl.groupby(['race_group'], observed=False).groups)
    metric_list = []
    for race in dfl['race_group'].unique():

        df_race = dfl.loc[dfl['race_group']==race]
        # Calculate the concordance index
        c_index_race = concordance_index(df_race['efs_time'],
                                         -df_race['prediction'],
                                         df_race['efs'])
        metric_list.append(c_index_race)
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))

def gpt(efs: pd.Series, efs_time: pd.Series, alpha=1, epsilon=1e-6) -> pd.Series:
    efs_time = efs_time + epsilon
    risk_score = (1 / efs_time) * (efs + alpha)
    return risk_score

def kmf(efs: pd.Series, efs_time: pd.Series) -> pd.Series:
    kmf = KaplanMeierFitter()
    kmf.fit(durations=efs_time, event_observed=efs)
    return kmf.survival_function_at_times(efs_time).values

def naf(efs: pd.Series, efs_time: pd.Series) -> pd.Series:
    naf = NelsonAalenFitter()
    naf.fit(durations=efs_time, event_observed=efs)
    return (-naf.cumulative_hazard_at_times(efs_time).values)

def cox(efs: pd.Series, efs_time: pd.Series) -> pd.Series:
    cox = efs_time
    cox[efs==0] *= -1
    return cox


#loading data

trn = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
tst = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
sub = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')
des = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')

trn['src']='trn'
tst['src']='tst'

# train test combined
df = pd.concat([trn, tst], ignore_index=True)

# data format and removing trailing and leading white spaces
df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

#with pd.option_context('display.max_columns', None): # setting the max rows
#    display(df.head())

df['gpt'] = 0.0
df['kmf'] = 0.0
df['naf'] = 0.0
df['cox'] = 0.0

df.loc[df['src']=='trn','gpt'] = gpt(df.loc[df['src']=='trn','efs'],df.loc[df['src']=='trn','efs_time'])
df.loc[df['src']=='trn','kmf'] = kmf(df.loc[df['src']=='trn','efs'],df.loc[df['src']=='trn','efs_time'])
df.loc[df['src']=='trn','naf'] = naf(df.loc[df['src']=='trn','efs'],df.loc[df['src']=='trn','efs_time'])
df.loc[df['src']=='trn','cox'] = cox(df.loc[df['src']=='trn','efs'],df.loc[df['src']=='trn','efs_time'])


# Features list
feature_list = [
# 'ID',
 'dri_score',
 'psych_disturb',
 'cyto_score',
 'diabetes',
 'hla_match_c_high',
 'hla_high_res_8',
 'tbi_status',
 'arrhythmia',
 'hla_low_res_6',
 'graft_type',
 'vent_hist',
 'renal_issue',
 'pulm_severe',
 'prim_disease_hct',
 'hla_high_res_6',
 'cmv_status',
 'hla_high_res_10',
 'hla_match_dqb1_high',
 'tce_imm_match',
 'hla_nmdp_6',
 'hla_match_c_low',
 'rituximab',
 'hla_match_drb1_low',
 'hla_match_dqb1_low',
 'prod_type',
 'cyto_score_detail',
 'conditioning_intensity',
 'ethnicity',
 'year_hct',
 'obesity',
 'mrd_hct',
 'in_vivo_tcd',
 'tce_match',
 'hla_match_a_high',
 'hepatic_severe',
 'donor_age',
 'prior_tumor',
 'hla_match_b_low',
 'peptic_ulcer',
 'age_at_hct',
 'hla_match_a_low',
 'gvhd_proph',
 'rheum_issue',
 'sex_match',
 'hla_match_b_high',
 'race_group',
 'comorbidity_score',
 'karnofsky_score',
 'hepatic_mild',
 'tce_div_match',
 'donor_related',
 'melphalan_dose',
 'hla_low_res_8',
 'cardiac',
 'hla_match_drb1_high',
 'pulm_moderate',
 'hla_low_res_10',
# 'efs',
# 'efs_time',
# 'src',
]

cat_cols = [
'dri_score',
'psych_disturb',
'cyto_score',
'diabetes',
'tbi_status',
'arrhythmia',
'graft_type',
'vent_hist',
'renal_issue',
'pulm_severe',
'prim_disease_hct',
'cmv_status',
'tce_imm_match',
'rituximab',
'prod_type',
'cyto_score_detail',
'conditioning_intensity',
'ethnicity',
'obesity',
'mrd_hct',
'in_vivo_tcd',
'tce_match',
'hepatic_severe',
'prior_tumor',
'peptic_ulcer',
'gvhd_proph',
'rheum_issue',
'sex_match',
'race_group',
'hepatic_mild',
'tce_div_match',
'donor_related',
'melphalan_dose',
'cardiac',
'pulm_moderate',
]

ord_cols= [
'year_hct',
'comorbidity_score',
'karnofsky_score',
'hla_nmdp_6',
'hla_low_res_6',
'hla_low_res_8',
'hla_low_res_10',
'hla_high_res_6',
'hla_high_res_10',
'hla_high_res_8',
'hla_match_a_low',
'hla_match_a_high',
'hla_match_b_low',
'hla_match_b_high',
'hla_match_c_low',
'hla_match_c_high',
'hla_match_dqb1_low',
'hla_match_dqb1_high',
'hla_match_drb1_low',
'hla_match_drb1_high',
]

num_cols= [
'donor_age',
'age_at_hct',
]


# Transformation
# dri_score ----------------------------------------------------------------------cat
mapping = {
    'Low': 'Low',
    'Intermediate': 'Intermediate',
    'High': 'High',
    'Very high': 'Very High',
    'Intermediate - TED AML case <missing cytogenetics': 'Intermediate TED AML case',
    'High - TED AML case <missing cytogenetics': 'High TED AML case',
    'N/A - non-malignant indication': 'Non Malignant',
    'N/A - pediatric': 'Pediatric',
    'N/A - disease not classifiable': 'Not Classifiable',
    'TBD cytogenetics': 'TBD cytogenetics',
    'Missing disease status': 'Missing disease status',
    np.nan: 'Unknown'
}
df['dri_score'] = df['dri_score'].map(mapping)

# psych_disturb ------------------------------------------------------------------cat
df['psych_disturb'] = df['psych_disturb'].fillna('Not done')

# cyto_score ---------------------------------------------------------------------cat
mapping = {
    'Favorable': 'Favorable',
    'Normal': 'Normal',
    'Intermediate': 'Intermediate',
    'Poor': 'Poor',
    'Other': 'Other',
    'TBD': 'Unknown',
    'Not tested': 'Unknown',
    np.nan: 'Unknown'
}
df['cyto_score'] = df['cyto_score'].map(mapping)

# diabetes ------------------------------------------------------------------------cat
df['diabetes'] = df['diabetes'].fillna('Not done')

# tbi_status ----------------------------------------------------------------------cat
mapping = {
    'No TBI': 'No TBI',
    'TBI + Cy +- Other': 'TBI with Cyclophosphamide',
    'TBI +- Other, >cGy': 'TBI High Dose',
    'TBI +- Other, <=cGy': 'TBI Low Dose',
    'TBI +- Other, unknown dose': 'TBI Unknown Dose',
    'TBI +- Other, -cGy, fractionated': 'TBI Fractionated',
    'TBI +- Other, -cGy, single': 'TBI Single',
    'TBI +- Other, -cGy, unknown dose': 'TBI cGy Unknown Dose'
}
df['tbi_status'] = df['tbi_status'].map(mapping)

# arrhythmia ----------------------------------------------------------------------cat
df['arrhythmia'] = df['arrhythmia'].fillna('Not done')

# graft_type ----------------------------------------------------------------------cat

# vent_hist -----------------------------------------------------------------------cat
df['vent_hist'] = df['vent_hist'].fillna('Unknown')

# renal_issue ---------------------------------------------------------------------cat
df['renal_issue'] = df['renal_issue'].fillna('Not done')

# pulm_severe ---------------------------------------------------------------------cat
df['pulm_severe'] = df['pulm_severe'].fillna('Not done')

# prim_disease_hct ----------------------------------------------------------------cat

# cmv_status ----------------------------------------------------------------------cat
mapping = {
    '+/+': 'PP',
    '+/-': 'PN',
    '-/+': 'NP',
    '-/-': 'NN',
    np.nan: 'Unknown'
}
df['cmv_status'] = df['cmv_status'].map(mapping)

# tce_imm_match ------------------------------------------------------------------cat
df['tce_imm_match'] = df['tce_imm_match'].fillna('Unknown')

# rituximab ----------------------------------------------------------------------cat
df['rituximab'] = df['rituximab'].fillna('Unknown')

# prod_type ----------------------------------------------------------------------cat

# cyto_score_detail --------------------------------------------------------------cat
df['cyto_score_detail'] = df['cyto_score_detail'].fillna('Unknown')

# conditioning_intensity ---------------------------------------------------------cat
mapping = {
    'MAC': 'MAC',
    'RIC': 'TIC',
    'NMA': 'NMA',
    'TBD': 'Unknown',
    'No drugs reported': 'Unknown',
    'N/A, F(pre-TED) not submitted': 'Unknown',
    np.nan: 'Unknown'
}
df['conditioning_intensity'] = df['conditioning_intensity'].map(mapping)

# ethnicity ----------------------------------------------------------------------cat
df['ethnicity'] = df['ethnicity'].fillna('Hispanic or Latino')

# obesity ------------------------------------------------------------------------cat
df['obesity'] = df['obesity'].fillna('Not done')

# mrd_hct ------------------------------------------------------------------------cat
df['mrd_hct'] = df['mrd_hct'].fillna('Unknown')

# in_vivo_tcd --------------------------------------------------------------------cat
df['in_vivo_tcd'] = df['in_vivo_tcd'].fillna('Unknown')

# tce_match ----------------------------------------------------------------------cat
df['tce_match'] = df['tce_match'].fillna('Unknown')

# hepatic_severe -----------------------------------------------------------------cat
df['hepatic_severe'] = df['hepatic_severe'].fillna('Not done')

# prior_tumor --------------------------------------------------------------------cat
df['prior_tumor'] = df['prior_tumor'].fillna('Unknown') # due to low score

# peptic_ulcer -------------------------------------------------------------------cat
df['peptic_ulcer'] = df['peptic_ulcer'].fillna('Not done')

# gvhd_proph ---------------------------------------------------------------------cat
mapping = {
    'FKalone': 'FKalone',
    'Other GVHD Prophylaxis': 'Other GVHD Prophylaxis',
    'Cyclophosphamide alone': 'Cyclophosphamide alone',
    'FK+ MMF +- others': 'FKp MMF pn others',
    'TDEPLETION +- other': 'TDEPLETION pn other',
    'CSA + MMF +- others(not FK)': 'CSA p MMF pn others not FK',
    'CSA + MTX +- others(not MMF,FK)': 'CSA p MTX pn others not MMF FK',
    'FK+ MTX +- others(not MMF)': 'FKp MTX pn others not MMF',
    'Cyclophosphamide +- others': 'Cyclophosphamide pn others',
    'CSA alone': 'CSA alone',
    'TDEPLETION alone': 'TDEPLETION alone',
    'No GvHD Prophylaxis': 'No GvHD Prophylaxis',
    'CDselect alone': 'CDselect alone',
    'CDselect +- other': 'CDselect pn other',
    'Parent Q = yes, but no agent': 'Other',
    'FK+- others(not MMF,MTX)': 'FKpn others not MMF MTX',
    'CSA +- others(not FK,MMF,MTX)': 'CSA pn others not FK MMF MTX',
    np.nan : 'Unknown'
}

df['gvhd_proph'] = df['gvhd_proph'].map(mapping)

# rheum_issue --------------------------------------------------------------------cat
df['rheum_issue'] = df['rheum_issue'].fillna('Not done')

# sex_match ----------------------------------------------------------------------cat
df['sex_match'] = df['sex_match'].fillna('Unknown')

# race_group ---------------------------------------------------------------------cat

# hepatic_mild -------------------------------------------------------------------cat
df['hepatic_mild'] = df['hepatic_mild'].fillna('Not done')

# tce_div_match ------------------------------------------------------------------cat
df['tce_div_match'] = df['tce_div_match'].fillna('Unknown')

# donor_related ------------------------------------------------------------------cat
df['donor_related'] = df['donor_related'].fillna('Unknown')

# melphalan_dose -----------------------------------------------------------------cat
df['melphalan_dose'] = df['melphalan_dose'].fillna('Unknown')

# cardiac ------------------------------------------------------------------------cat
df['cardiac'] = df['cardiac'].fillna('Not done')

# pulm_moderate ------------------------------------------------------------------cat
df['pulm_moderate'] = df['pulm_moderate'].fillna('Not done')

# hla_match_drb1_low -------------------------------------------------------------ord
#df['hla_match_drb1_low'] = df['hla_match_drb1_low'].fillna(-1)

# hla_match_drb1_high ------------------------------------------------------------ord
df['hla_match_drb1_high'] = df['hla_match_drb1_high'].fillna(-1)

# hla_match_a_low ----------------------------------------------------------------ord
#df['hla_match_a_low'] = df['hla_match_a_low'].fillna(-1)

# hla_match_a_high ---------------------------------------------------------------ord
#df['hla_match_a_high'] = df['hla_match_a_high'].fillna(-1)

# hla_match_b_low ----------------------------------------------------------------ord
#df['hla_match_b_low'] = df['hla_match_b_low'].fillna(-1)

# hla_match_b_high ---------------------------------------------------------------ord
#df['hla_match_b_high'] = df['hla_match_b_high'].fillna(-1)

# hla_match_c_low ----------------------------------------------------------------ord
#df['hla_match_c_low'] = df['hla_match_c_low'].fillna(-1)

# hla_match_c_high ---------------------------------------------------------------ord
df['hla_match_c_high'] = df['hla_match_c_high'].fillna(-1)

# hla_match_dqb1_high ------------------------------------------------------------ord
df['hla_match_dqb1_high'] = df['hla_match_dqb1_high'].fillna(-1)

# hla_match_dqb1_low -------------------------------------------------------------ord
#df['hla_match_dqb1_low'] = df['hla_match_dqb1_low'].fillna(-1)

# hla_high_res_6 -----------------------------------------------------------------ord
#df['hla_high_res_6'] = df['hla_high_res_6'].fillna(-1)

# hla_high_res_8 -----------------------------------------------------------------ord
#df['hla_high_res_8'] = df['hla_high_res_8'].fillna(-1)

# hla_high_res_10 ----------------------------------------------------------------ord
#df['hla_high_res_10'] = df['hla_high_res_10'].fillna(-1)

# hla_low_res_6 ------------------------------------------------------------------ord
#df['hla_low_res_6'] = df['hla_low_res_6'].fillna(-1)

# hla_low_res_8 ------------------------------------------------------------------ord
#df['hla_low_res_8'] = df['hla_low_res_8'].fillna(-1)

# hla_low_res_10 -----------------------------------------------------------------ord
#df['hla_low_res_10'] = df['hla_low_res_10'].fillna(-1)

# hla_nmdp_6 ---------------------------------------------------------------------ord
#df['hla_nmdp_6'] = df['hla_nmdp_6'].fillna(-1)

# comorbidity_score --------------------------------------------------------------ord
#df['comorbidity_score'] = df['comorbidity_score'].fillna(-1)

# karnofsky_score ----------------------------------------------------------------ord
df['karnofsky_score'] = df['karnofsky_score'].fillna(-1)

# year_hct -----------------------------------------------------------------------ord

# donor_age ----------------------------------------------------------------------num
#df['donor_age'] = df['donor_age'].fillna(-1)

# age_at_hct ---------------------------------------------------------------------num


## KFOLDs
df[cat_cols] = df[cat_cols].astype("category")

train = df.loc[df['src']=='trn']
test = df.loc[df['src']=='tst']

FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

preds = []
oofs = []



from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import xgboost as xgb

for target in ['naf','kmf','cox','gpt']:

    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    
    for i, (trn_index, val_index) in enumerate(kf.split(train)):
    
        x_train = train.loc[trn_index,feature_list].copy()
        y_train = train.loc[trn_index,target]
        x_valid = train.loc[val_index,feature_list].copy()
        y_valid = train.loc[val_index,target]
        x_test  = test[feature_list].copy()
    
        model_xgb = XGBRegressor(
            device="cuda",
            max_depth=3,  
            colsample_bytree=0.5,  
            subsample=0.8,  
            n_estimators=2000,  
            learning_rate=0.02,  
            enable_categorical=True,
            min_child_weight=80,
            # early_stopping_rounds=25,
            random_state=42,
        )
        model_xgb.fit(
            x_train, y_train,
            eval_set=[(x_valid, y_valid)],  
            verbose=0 
        )
    
        # INFER OOF
        oof[val_index] = model_xgb.predict(x_valid)
        
        # INFER TEST
        pred += model_xgb.predict(x_test)

    # COMPUTE AVERAGE TEST PREDS
    pred /= FOLDS
    train["prediction"] = oof
    m = score(train.copy())
    preds.append(pred)
    oofs.append(oof)
    print(f"Overall CV for XGBoost {target} =",m)



from catboost import CatBoostRegressor
import catboost as cb
for target in ['naf','kmf','cox','gpt']:

    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    
    for i, (trn_index, val_index) in enumerate(kf.split(train)):
    
        x_train = train.loc[trn_index,feature_list].copy()
        y_train = train.loc[trn_index,target]
        x_valid = train.loc[val_index,feature_list].copy()
        y_valid = train.loc[val_index,target]
        x_test  = test[feature_list].copy()
    
        model_cat = CatBoostRegressor(
            task_type="GPU",  
            learning_rate=0.1,    
            grow_policy='Lossguide',
            early_stopping_rounds=25,
        )
        model_cat.fit(x_train,y_train,
                  eval_set=(x_valid, y_valid),
                  cat_features=cat_cols,
                  verbose=0)
    
        # INFER OOF
        oof[val_index] = model_cat.predict(x_valid)
        # INFER TEST
        pred += model_cat.predict(x_test)

    # COMPUTE AVERAGE TEST PREDS
    pred /= FOLDS
    pred /= FOLDS
    train["prediction"] = oof
    m = score(train.copy())
    preds.append(pred)
    oofs.append(oof)
    print(f"Overall CV for Catboost {target} =",m)



from lightgbm import LGBMRegressor
import lightgbm as lgb
for target in ['naf','kmf','cox','gpt']:

    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    
    for i, (trn_index, val_index) in enumerate(kf.split(train)):
    
        x_train = train.loc[trn_index,feature_list].copy()
        y_train = train.loc[trn_index,target]
        x_valid = train.loc[val_index,feature_list].copy()
        y_valid = train.loc[val_index,target]
        x_test  = test[feature_list].copy()
    
        model_lgb = LGBMRegressor(
            device="gpu", 
            max_depth=3, 
            colsample_bytree=0.4,  
            subsample=0.8, 
            n_estimators=2500, 
            learning_rate=0.02, 
            objective="regression", 
            verbose=-1, 
            early_stopping_rounds=25,
        )
        model_lgb.fit(
            x_train, y_train,
            eval_set=[(x_valid, y_valid)],
        )
    
        # INFER OOF
        oof[val_index] = model_lgb.predict(x_valid)
        # INFER TEST
        pred += model_lgb.predict(x_test)

    # COMPUTE AVERAGE TEST PREDS
    pred /= FOLDS
    train["prediction"] = oof
    m = score(train.copy())
    preds.append(pred)
    oofs.append(oof)
    print(f"Overall CV for lgbm {target} =",m)



oof = np.zeros(len(train))
pred = np.zeros(len(test))

for i, (trn_index, val_index) in enumerate(kf.split(train)):

    x_train = train.loc[trn_index,feature_list].copy()
    y_train = train.loc[trn_index,'cox']
    x_valid = train.loc[val_index,feature_list].copy()
    y_valid = train.loc[val_index,'cox']
    x_test  = test[feature_list].copy()
        
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
        early_stopping_rounds=25,
    )
    model_xgb_cox.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=0  
    )
    
    # INFER OOF
    oof[val_index] = model_xgb_cox.predict(x_valid)
    # INFER TEST
    pred += model_xgb_cox.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred /= FOLDS
train["prediction"] = oof
m = score(train.copy())
preds.append(pred)
oofs.append(oof)
print(f"Overall CV for xgb_cox =",m)



from catboost import CatBoostRegressor 
import catboost as cb

oof = np.zeros(len(train))
pred = np.zeros(len(test))

for i, (trn_index, val_index) in enumerate(kf.split(train)):

    x_train = train.loc[trn_index,feature_list].copy()
    y_train = train.loc[trn_index,'cox']
    x_valid = train.loc[val_index,feature_list].copy()
    y_valid = train.loc[val_index,'cox']
    x_test  = test[feature_list].copy()
        
    model_cat_cox = CatBoostRegressor(
        loss_function="Cox",
        #task_type="GPU",   
        iterations=400,     
        learning_rate=0.1,  
        grow_policy='Lossguide',
        use_best_model=False,)
    
    model_cat_cox.fit(x_train,y_train,
              eval_set=(x_valid, y_valid),
              cat_features=cat_cols,
              verbose=0)
    
    # INFER OOF
    oof[val_index] = model_cat_cox.predict(x_valid)
    # INFER TEST
    pred += model_cat_cox.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred /= FOLDS
train["prediction"] = oof
m = score(train.copy())
preds.append(pred)
oofs.append(oof)
print(f"Overall CV for cat_cox =",m)


## Overall evaluaiton
from scipy.stats import rankdata 

train["prediction"] = np.sum([rankdata(arr) for arr in oofs], axis=0)
m = score(train.copy())
print(f"\n Overall rank Ensemble =",m)



#Overall CV for XGBoost naf = 0.6741848683168845
#Overall CV for XGBoost kmf = 0.6719987338405566
#Overall CV for XGBoost cox = 0.6208352699402188
#Overall CV for XGBoost gpt = 0.6700214118274613
#Overall CV for Catboost naf = 0.673004711036217
#Overall CV for Catboost kmf = 0.670712318066686
#Overall CV for Catboost cox = 0.6166792448927585
#Overall CV for Catboost gpt = 0.6660799757447542
#Overall CV for lgbm naf = 0.6728935330593759
#Overall CV for lgbm kmf = 0.6707411279428226
#Overall CV for lgbm cox = 0.618223880216823
#Overall CV for lgbm gpt = 0.6660242430409125
#Overall CV for xgb_cox = 0.6680620133381031
#Overall CV for cat_cox = 0.6695209077876626

# Define weights based on model performance
weights = [
    0.6741, 0.6719, 0.6208, 0.6700, # CatBoost weights
    0.6730, 0.6707, 0.6166, 0.6660, # LightGBM weights
    0.6728, 0.6707, 0.6182, 0.6660,  # Cox weights 
    0.6680,
    0.6695
]

weights = [
    5.0, 2.0, 0.2, 1.0, # CatBoost weights
    4.0, 1.0, 0.2, 0.7, # LightGBM weights
    3.0, 1.0, 0.2, 0.7,  # Cox weights 
    0.7,
    0.7
]

# Create ranked predictions
ranked_preds = np.array([rankdata(arr) for arr in oofs])
train["prediction"] = np.sum([w * p for w, p in zip(weights, ranked_preds)], axis=0)
m = score(train.copy())
print(f"\n Overall weighted rank Ensemble =",m)



ranked_preds = np.array([rankdata(arr) for arr in preds])
submission = test[["ID"]].copy()
submission["prediction"] = np.sum([w * p for w, p in zip(weights, ranked_preds)], axis=0)
submission.to_csv("submission.csv",index=False)

print("submission completed")




