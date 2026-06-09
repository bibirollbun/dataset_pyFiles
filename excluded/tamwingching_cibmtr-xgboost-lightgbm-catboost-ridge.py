import sys
sys.path.insert(1, '/kaggle/input/featurecreation')
from features_creation import *


import pandas as pd
from sklearn.model_selection import KFold


#!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0
!pip install /kaggle/input/autogradgammawhl/other/default/1/autograd_gamma-0.4.2-py2.py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


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
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))


from lifelines import KaplanMeierFitter, NelsonAalenFitter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin


train_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')


old_train_data = train_data.copy()
old_test_data = test_data.copy()


ID_col = 'ID'
y_train_kmf = train_data['KaplanMeier'] = KaplanMeierFitter().fit(train_data['efs_time'], train_data['efs']).survival_function_at_times(train_data['efs_time']).values


RMV = ["ID","efs","efs_time",'KaplanMeier']
FEATURES = [c for c in train_data.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


hct_ci_mapping = {
    "arrhythmia": {"No": 0, "Not done": 0, "Yes": 1},  
    "cardiac": {"No": 0, "Not done": 0, "Yes": 1}, 
    "diabetes": {"No": 0, "Not done": 0, "Yes": 1},  
    "hepatic_mild": {"No": 0, "Not done": 0, "Yes": 1},
    "hepatic_severe": {"No": 0, "Not done": 0, "Yes": 3},
    "psych_disturb": {"No": 0, "Not done": 0, "Yes": 1}, 
    "obesity": {"No": 0, "Not done": 0, "Yes": 1}, 
    "rheum_issue": {"No": 0, "Not done": 0, "Yes": 2},
    "peptic_ulcer": {"No": 0, "Not done": 0, "Yes": 2},  
    "renal_issue": {"No": 0, "Not done": 0, "Yes": 2}, 
    "prior_tumor": {"No": 0, "Not done": 0, "Yes": 3}, 
    "pulm_moderate": {"No": 0, "Not done": 0, "Yes": 2}, 
    "pulm_severe": {"No": 0, "Not done": 0, "Yes": 3},  
}
def calculate_hct_ci_score(row, mapping):
        """
        This function calculates the hct_ci score
    
        Args:
            row (pd.Series): Patient Clinical Data
            mapping (dict): HCT-CI score mapping
    
        Returns:
            int: HCT-CI score
        """
    
        score = 0
    
        if "hepatic_severe" in row and row["hepatic_severe"] == "Yes":
            score += mapping["hepatic_severe"]["Yes"]
        elif "hepatic_mild" in row and row["hepatic_mild"] == "Yes":
            score += mapping["hepatic_mild"]["Yes"]
        if "pulm_moderate" in row and row["pulm_moderate"] == "Yes":
            score += mapping["pulm_moderate"]["Yes"]
        elif "pulm_severe" in row and row["pulm_severe"] == "Yes":
            score += mapping["pulm_severe"]["Yes"]
    
        # Other Conditions
        for condition, mapping_values in mapping.items():
            if condition not in ["hepatic_mild", "hepatic_severe","pulm_moderate", "pulm_severe"] and condition in row:
                score += mapping_values.get(row[condition], 0)
    
        return score


def cat2num(df):
    df['conditioning_intensity'] = df['conditioning_intensity'].map({
    'NMA': 1, 
    'RIC': 2,
    'MAC': 3,
    'TBD': None,
    'No drugs reported': None,
    'N/A, F(pre-TED) not submitted': None})
    
    df['tbi_status'] = df['tbi_status'].map({
    'No TBI': 0, 
    'TBI +- Other, <=cGy': 1,
    'TBI +- Other, -cGy, fractionated': 2,
    'TBI + Cy +- Other': 3,
    'TBI +- Other, -cGy, single': 4,
    'TBI +- Other, >cGy': 5,
    'TBI +- Other, unknown dose': None})
    
    df['dri_score'] = df['dri_score'].map({
    'Low': 1, 
    'Intermediate': 2,
    'Intermediate - TED AML case <missing cytogenetics': 3,
    'High': 4,
    'High - TED AML case <missing cytogenetics': 5,
    'Very High': 6,
    'N/A - pediatric': -3,
    'N/A - non-malignant indication': -1,
    'TBD cytogenetics': -2,
    'N/A - disease not classifiable': -4,
    'Missing disease status': 0})
    
    df['cyto_score'] = df['cyto_score'].map({
    'Poor': 4,
    'Normal': 3,
    'Intermediate': 2,
    'Favorable': 1,
    'TBD': -1,
    'Other': -2,
    'Not tested': None})
    
    df['cyto_score_detail'] = df['cyto_score_detail'].map({
    'Poor': 3, 
    'Intermediate': 2,
    'Favorable': 1,
    'TBD': -1,
    'Not tested': None})
    
    return df


def fill_hla_combined_low(row):
    if np.isnan(row['hla_combined_low']): 
        components = [
            row['hla_match_drb1_low'], row['hla_match_dqb1_low'], 
            row['hla_match_a_low'], row['hla_match_b_low'], row['hla_match_c_low']
        ]
        if all([not np.isnan(x) for x in components]):
            return sum(components)
        else:
            if not np.isnan(row['hla_low_res_8']) and not np.isnan(row['hla_match_dqb1_low']):
                return row['hla_low_res_8'] + row['hla_match_dqb1_low']
            elif not np.isnan(row['hla_low_res_6']): 
                components_6 = [
                    row['hla_match_dqb1_low'], row['hla_match_c_low']
                ]
                if all([not np.isnan(x) for x in components_6]):
                    return row['hla_low_res_6'] + sum(components_6)
                else: 
                    return sum([x for x in components if not np.isnan(x)])
    return row['hla_combined_low'] 


def add_features(df):
    df["hct_ci_score"] = df.apply(lambda row: calculate_hct_ci_score(row, hct_ci_mapping), axis=1)
    df['donor_recipient_age_diff'] = abs(df['donor_age'] - df['age_at_hct'])
    df = cat2num(df)
    df['hla_combined_low'] = df['hla_low_res_10']
    df['hla_combined_low'] = df.apply(fill_hla_combined_low, axis=1)
    df['hla_match_ratio'] = (df['hla_high_res_8'] + df['hla_low_res_8']) / 16
    df['years_since_2000'] = df['year_hct'] - 2000
    df['null_count'] = df.isnull().sum(axis=1)
    df['ci_score_danger'] = df['hct_ci_score'].apply(lambda x: 2 if x >= 3 else 1 if x >= 1 else 0)
    return df

train_data = add_features(train_data)
test_data = add_features(test_data)


FEATURES += ["hct_ci_score", 'donor_recipient_age_diff', "hla_combined_low", "hla_match_ratio", 
             "years_since_2000", "null_count","ci_score_danger"]


train_data.head()


del train_data['year_hct']
del test_data['year_hct']


FEATURES = ['dri_score',
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
 'hct_ci_score',
 'donor_recipient_age_diff',
 'hla_combined_low',
 'hla_match_ratio',
 'years_since_2000',
 'null_count',
 'ci_score_danger']


numeric = ['age_at_hct','year_hct','donor_age','null_count','hla_match_ratio','years_since_2000','donor_recipient_age_diff']
#numeric = ['age_at_hct','donor_age','null_count','donor_recipient_age_diff']
CATS = []
for c in FEATURES:
    if c not in numeric:
        CATS.append(c)
        train_data[c] = train_data[c].fillna("NAN")
        test_data[c] = test_data[c].fillna("NAN")
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


combined = pd.concat([train_data,test_data],axis=0,ignore_index=True)
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

# for c in cat2num:
#     combined[c] = combined[c].astype("int32")

train_data = combined.iloc[:len(train_data)].copy()
test_data = combined.iloc[len(train_data):].reset_index(drop=True).copy()


ID_col = 'ID'
target = 'KaplanMeier'
del train_data['efs_time']
del train_data['efs']


del test_data['efs_time']
del test_data['efs']
del test_data['KaplanMeier']


train_data.shape


test_data.shape


train_cont = train_data[['age_at_hct','donor_age','null_count','hla_match_ratio','years_since_2000','donor_recipient_age_diff']]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_newcont = scaler.fit_transform(train_cont.values)
train_transform_cont = pd.DataFrame(train_newcont, index=train_cont.index, columns=train_cont.columns)


train_data['age_at_hct'] = train_transform_cont['age_at_hct']
train_data['donor_age'] = train_transform_cont['donor_age'] 
train_data['null_count']= train_transform_cont['null_count'] 
train_data['hla_match_ratio']= train_transform_cont['hla_match_ratio'] 
train_data['years_since_2000']= train_transform_cont['years_since_2000'] 
train_data['donor_recipient_age_diff']= train_transform_cont['donor_recipient_age_diff'] 


test_cont = test_data[['age_at_hct','donor_age','null_count','hla_match_ratio','years_since_2000','donor_recipient_age_diff']]
test_newcont = scaler.transform(test_cont.values)
test_transform_cont = pd.DataFrame(test_newcont, index=test_cont.index, columns=test_cont.columns)


test_data['age_at_hct'] = test_transform_cont['age_at_hct']
test_data['donor_age'] = test_transform_cont['donor_age'] 
test_data['null_count']= test_transform_cont['null_count'] 
test_data['hla_match_ratio']= test_transform_cont['hla_match_ratio'] 
test_data['years_since_2000']= test_transform_cont['years_since_2000'] 
test_data['donor_recipient_age_diff']= test_transform_cont['donor_recipient_age_diff'] 


train_data['donor_age'] = train_data['donor_age'].fillna(train_data['donor_age'].mean())
train_data['hla_match_ratio'] = train_data['hla_match_ratio'].fillna(train_data['hla_match_ratio'].mean())
train_data['donor_recipient_age_diff'] = train_data['donor_recipient_age_diff'].fillna(train_data['donor_recipient_age_diff'].mean())


test_data['donor_age'] = test_data['donor_age'].fillna(test_data['donor_age'].mean())
test_data['hla_match_ratio'] = test_data['hla_match_ratio'].fillna(test_data['hla_match_ratio'].mean())
test_data['donor_recipient_age_diff'] = test_data['donor_recipient_age_diff'].fillna(test_data['donor_recipient_age_diff'].mean())


X_train = train_data.drop(columns=[ID_col, target]).copy()
X_test = test_data.drop(columns=[ID_col]).copy()
y_train = train_data['KaplanMeier'].copy()


CATS = ['dri_score',
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
 'obesity',
 'mrd_hct',
 'in_vivo_tcd',
 'tce_match',
 'hla_match_a_high',
 'hepatic_severe',
 'prior_tumor',
 'hla_match_b_low',
 'peptic_ulcer',
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
 'hct_ci_score',
 'hla_combined_low',
 'ci_score_danger']


CATS


X_train.shape


X_test.shape


import lightgbm as lgb
from lightgbm import LGBMClassifier,LGBMRegressor


from sklearn.model_selection import KFold


train_data['y'] = y_train


from sklearn.model_selection import KFold
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_lgb = np.zeros(len(train_data))
pred_lgb = np.zeros(len(test_data))

for i, (train_index, test_index) in enumerate(kf.split(train_data)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train_data.loc[train_index,FEATURES].copy()
    y_train = train_data.loc[train_index,"y"]    
    x_valid = train_data.loc[test_index,FEATURES].copy()
    y_valid = train_data.loc[test_index,"y"]
    x_test = test_data[FEATURES].copy()

    model_lgb = LGBMRegressor(num_boost_round = 6000,verbose=-1,bagging_fraction=0.9496219531394654,bagging_freq=1,boosting_type= 'gbdt',colsample_bynode=0.49190952114458525,colsample_bytree=0.9276799260625302,feature_fraction=0.6237476443614289,importance_type='gain',lambda_l1=0.3307453595868796,lambda_l2=0.3748389968255914,learning_rate=0.014357619575271343,
max_depth=5,min_data_in_leaf=135,num_leaves=20)
   
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


y_true = old_train_data[["ID","efs","efs_time","race_group"]].copy()
y_pred = train_data[["ID"]].copy()
y_pred["prediction"] = oof_lgb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for LightGBM KaplanMeier =",m)


train_data["efs_time2"] = old_train_data.efs_time.copy()
train_data.loc[old_train_data.efs==0,"efs_time2"] *= -1


from sklearn.model_selection import KFold
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_lgb_cox = np.zeros(len(train_data))
pred_lgb_cox = np.zeros(len(test_data))

for i, (train_index, test_index) in enumerate(kf.split(train_data)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train_data.loc[train_index,FEATURES].copy()
    y_train = train_data.loc[train_index,"efs_time2"]    
    x_valid = train_data.loc[test_index,FEATURES].copy()
    y_valid = train_data.loc[test_index,"efs_time2"]
    x_test = test_data[FEATURES].copy()

    model_lgb_cox = LGBMRegressor(num_boost_round = 6000,verbose=-1,bagging_fraction=0.9496219531394654,bagging_freq=1,boosting_type= 'gbdt',colsample_bynode=0.49190952114458525,colsample_bytree=0.9276799260625302,feature_fraction=0.6237476443614289,importance_type='gain',lambda_l1=0.3307453595868796,lambda_l2=0.3748389968255914,learning_rate=0.014357619575271343,
max_depth=5,min_data_in_leaf=135,num_leaves=20)
   
    model_lgb_cox.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
    )
    
    # INFER OOF
    oof_lgb_cox[test_index] = model_lgb_cox.predict(x_valid)
    # INFER TEST
    pred_lgb_cox += model_lgb_cox.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_lgb_cox /= FOLDS


y_true = old_train_data[["ID","efs","efs_time","race_group"]].copy()
y_pred = train_data[["ID"]].copy()
y_pred["prediction"] = oof_lgb_cox
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for LightGBM Survival:Cox =",m)


from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
print("Using XGBoost version",xgb.__version__)


import time
start = time.time()

FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_xgb = np.zeros(len(train_data))
pred_xgb = np.zeros(len(test_data))

for i, (train_index, test_index) in enumerate(kf.split(train_data)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train_data.loc[train_index,FEATURES].copy()
    y_train = train_data.loc[train_index,"y"]
    x_valid = train_data.loc[test_index,FEATURES].copy()
    y_valid = train_data.loc[test_index,"y"]
    x_test = test_data[FEATURES].copy()

    model_xgb = XGBRegressor(num_boost_round =6000,enable_categorical=True,booster='gbtree',colsample_bynode=0.85,colsample_bytree=0.6,eta=0.01,reg_lambda=2.0,max_depth=7,min_child_weight=40,rate_drop=0.2,subsample=0.9)
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

end = time.time()
print(end - start)


y_true = old_train_data[["ID","efs","efs_time","race_group"]].copy()
y_pred = train_data[["ID"]].copy()
y_pred["prediction"] = oof_xgb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost KaplanMeier =",m)


import time
start = time.time()

FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_xgb_cox = np.zeros(len(train_data))
pred_xgb_cox = np.zeros(len(test_data))

for i, (train_index, test_index) in enumerate(kf.split(train_data)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train_data.loc[train_index,FEATURES].copy()
    y_train = train_data.loc[train_index,"efs_time2"]
    x_valid = train_data.loc[test_index,FEATURES].copy()
    y_valid = train_data.loc[test_index,"efs_time2"]
    x_test = test_data[FEATURES].copy()

    model_xgb_cox = XGBRegressor(num_boost_round =6000,enable_categorical=True,booster='gbtree',colsample_bynode=0.85,colsample_bytree=0.6,eta=0.01,reg_lambda=2.0,max_depth=7,min_child_weight=40,rate_drop=0.2,subsample=0.9)
    model_xgb_cox.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=500 
    )

    # INFER OOF
    oof_xgb_cox[test_index] = model_xgb_cox.predict(x_valid)
    # INFER TEST
    pred_xgb_cox += model_xgb_cox.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb_cox /= FOLDS

end = time.time()
print(end - start)


y_true = old_train_data[["ID","efs","efs_time","race_group"]].copy()
y_pred = train_data[["ID"]].copy()
y_pred["prediction"] = oof_xgb_cox
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost Survival:Cox =",m)


from catboost import CatBoostRegressor, CatBoostClassifier
import catboost as cb
print("Using CatBoost version",cb.__version__)


%%time
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_cat = np.zeros(len(train_data))
pred_cat = np.zeros(len(test_data))

for i, (train_index, test_index) in enumerate(kf.split(train_data)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train_data.loc[train_index,FEATURES].copy()
    y_train = train_data.loc[train_index,"y"]
    x_valid = train_data.loc[test_index,FEATURES].copy()
    y_valid = train_data.loc[test_index,"y"]
    x_test = test_data[FEATURES].copy()
 
    model_cat = CatBoostRegressor(
        bagging_temperature =  0.7954382875508031,
        border_count = 32.0,
        depth = 9.0,
        l2_leaf_reg = 5.656290326109749,
        learning_rate= 0.08337834510110963,    
        min_data_in_leaf = 39.01468374674604,
        grow_policy='Lossguide',
        random_strength = 0.6918202342914714,
        iterations = 3000.0
        #early_stopping_rounds=25,
    )
    model_cat.fit(x_train,y_train,
              eval_set=(x_valid, y_valid),
              cat_features=CATS,
              verbose=250)

    # INFER OOF
    oof_cat[test_index] = model_cat.predict(x_valid)
    # INFER TEST
    pred_cat += model_cat.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_cat /= FOLDS


y_true = old_train_data[["ID","efs","efs_time","race_group"]].copy()
y_pred = train_data[["ID"]].copy()
y_pred["prediction"] = oof_cat
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for CATBoost KaplanMeier =",m)


from sklearn.linear_model import Ridge
ridge = Ridge()
param_grid_ridge = {
    'alpha': [0.05, 0.1, 1, 3, 5, 10],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
}
ridge_cv = GridSearchCV(ridge, param_grid_ridge, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
ridge_cv.fit(X_train, train_data['y'])


%%time
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_rid = np.zeros(len(train_data))
pred_rid = np.zeros(len(test_data))

for i, (train_index, test_index) in enumerate(kf.split(train_data)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train_data.loc[train_index,FEATURES].copy()
    y_train = train_data.loc[train_index,"y"]
    x_valid = train_data.loc[test_index,FEATURES].copy()
    y_valid = train_data.loc[test_index,"y"]
    x_test = test_data[FEATURES].copy()
 
    model_rid = Ridge(
        alpha=10, solver= 'lsqr'
    )
    model_rid.fit(x_train,y_train)

    # INFER OOF
    oof_rid[test_index] = model_rid.predict(x_valid)
    # INFER TEST
    pred_rid += model_rid.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_rid /= FOLDS


y_true = old_train_data[["ID","efs","efs_time","race_group"]].copy()
y_pred = train_data[["ID"]].copy()
y_pred["prediction"] = oof_rid
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for Ridge KaplanMeier =",m)


#from scipy.stats import rankdata 

y_true = old_train_data[["ID","efs","efs_time","race_group"]].copy()
y_pred = train_data[["ID"]].copy()
y_pred["prediction"] = (oof_xgb+oof_cat+oof_lgb+0.6*oof_rid)/3.6
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for Ensemble =",m)


sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub.prediction = (pred_xgb + pred_lgb+pred_cat+0.6*pred_rid)/3.6
sub.to_csv("submission.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()

