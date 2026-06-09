import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from catboost import CatBoostRegressor
import optuna
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
submission = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')


train.head()


train.info()


cat_column = ['dri_score', 'psych_disturb', 'cyto_score', 'diabetes', 'tbi_status', 
                'arrhythmia', 'graft_type', 'vent_hist', 'renal_issue', 'pulm_severe',
                'prim_disease_hct', 'cmv_status', 'tce_imm_match', 'rituximab', 'prod_type',
                'cyto_score_detail', 'conditioning_intensity', 'ethnicity', 'obesity', 'mrd_hct',
                'in_vivo_tcd', 'tce_match', 'hepatic_severe', 'prior_tumor', 'peptic_ulcer', 
                'gvhd_proph','rheum_issue', 'sex_match', 'race_group', 'hepatic_mild',
              'tce_div_match', 'donor_related','melphalan_dose', 'cardiac', 'pulm_moderate']


num_column =  ['hla_match_c_high', 'hla_high_res_8', 'hla_low_res_6', 'hla_high_res_6',
               'hla_high_res_10', 'hla_match_dqb1_high', 'hla_nmdp_6', 'hla_match_c_low',
               'hla_match_drb1_low', 'hla_match_dqb1_low', 'year_hct', 'hla_match_a_high',
               'donor_age', 'hla_match_b_low', 'age_at_hct', 'hla_match_a_low', 'hla_match_b_high',
               'comorbidity_score', 'karnofsky_score', 'hla_low_res_8', 'hla_match_drb1_high', 
               'hla_low_res_10', 'efs', 'efs_time']


for c in cat_column:
    train[c] = train[c].fillna('None').astype('category')
    test[c] = test[c].fillna('None').astype('category')


j_ch = ',[]{}:"\\<'

for ch in j_ch:
    for c in num_column:
        if c in train.columns:
            train[c] = train[c].apply(lambda x: str(x).replace(ch, ''))
        if c in test.columns:
            test[c] = test[c].apply(lambda x: str(x).replace(ch, ''))


train.head()


label_encoder = LabelEncoder()

for c in cat_column:
    if c in train.columns:
        train[c] = label_encoder.fit_transform(train[c])


train.head()


X = train.drop(columns=['efs','efs_time'])
y = train['efs']


X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.2, random_state=42)


best_params = {
    'iterations': 1539,
    'learning_rate': 0.04,
    'depth': 5,
    'l2_leaf_reg': 6.937091055213791,
    'loss_function': 'RMSE',  
    'random_seed': 42,
    'verbose': 0,
}


model = CatBoostRegressor(**best_params)

model.fit(X, y)


le = LabelEncoder()

for c in cat_column:
    if c in test.columns:
        test[c] = le.fit_transform(test[c])


test.head()


pred = model.predict(test)


submission['prediction'] = pred
print(submission)


submission.to_csv('submission.csv', index=False)
print("File Saved!!")

