# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import missingno as msno
import statistics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression


train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
#train = train.dropna()
train.info()


train['efs'].head()


categorical_columns = train.select_dtypes(include='object').columns
print(categorical_columns)


train['dri_score_numeric'] = pd.factorize(train['dri_score'])[0]
train['psych_disturb_numeric'] = pd.factorize(train['psych_disturb'])[0]
train['cyto_score_numeric'] = pd.factorize(train['cyto_score'])[0]
train['diabetes_numeric'] = pd.factorize(train['diabetes'])[0]
train['tbi_status_numeric'] = pd.factorize(train['tbi_status'])[0]
train['arrhythmia_numeric'] = pd.factorize(train['arrhythmia'])[0]
train['graft_type_numeric'] = pd.factorize(train['graft_type'])[0]
train['vent_hist_numeric'] = pd.factorize(train['vent_hist'])[0]
train['renal_issue_numeric'] = pd.factorize(train['renal_issue'])[0]
train['pulm_severe_numeric'] = pd.factorize(train['pulm_severe'])[0]
train['prim_disease_hct_numeric'] = pd.factorize(train['prim_disease_hct'])[0]
train['cmv_status_numeric'] = pd.factorize(train['cmv_status'])[0]
train['tce_imm_match_numeric'] = pd.factorize(train['tce_imm_match'])[0]
train['rituximab_numeric'] = pd.factorize(train['rituximab'])[0]
train['prod_type_numeric'] = pd.factorize(train['cyto_score_detail'])[0]
train['conditioning_intensity_numeric'] = pd.factorize(train['conditioning_intensity'])[0]
train['ethnicity_numeric'] = pd.factorize(train['ethnicity'])[0]
train['obesity_numeric'] = pd.factorize(train['obesity'])[0]
train['mrd_hct_numeric'] = pd.factorize(train['mrd_hct'])[0]
train['in_vivo_tcd_numeric'] = pd.factorize(train['in_vivo_tcd'])[0]
train['tce_match_numeric'] = pd.factorize(train['tce_match'])[0]
train['hepatic_severe_numeric'] = pd.factorize(train['hepatic_severe'])[0]
train['prior_tumor_numeric'] = pd.factorize(train['prior_tumor'])[0]
train['peptic_ulcer_numeric'] = pd.factorize(train['peptic_ulcer'])[0]
train['gvhd_proph_numeric'] = pd.factorize(train['gvhd_proph'])[0]
train['rheum_issue_numeric'] = pd.factorize(train['rheum_issue'])[0]
train['sex_match_numeric'] = pd.factorize(train['sex_match'])[0]
train['race_group_numeric'] = pd.factorize(train['race_group'])[0]
train['hepatic_mild_numeric'] = pd.factorize(train['hepatic_mild'])[0]
train['tce_div_match_numeric'] = pd.factorize(train['tce_div_match'])[0]
train['donor_related_numeric'] = pd.factorize(train['donor_related'])[0]
train['melphalan_dose_numeric'] = pd.factorize(train['melphalan_dose'])[0]
train['cardiac_numeric'] = pd.factorize(train['cardiac'])[0]
train['pulm_moderate_numeric'] = pd.factorize(train['pulm_moderate'])[0]


train.select_dtypes(include=['int64','float64']).columns


y = train.efs
y2 = train.efs_time
features = ['hla_match_c_high', 'hla_high_res_8', 'hla_low_res_6',
       'hla_high_res_6', 'hla_high_res_10', 'hla_match_dqb1_high',
       'hla_nmdp_6', 'hla_match_c_low', 'hla_match_drb1_low',
       'hla_match_dqb1_low', 'year_hct', 'hla_match_a_high', 'donor_age',
       'hla_match_b_low', 'age_at_hct', 'hla_match_a_low', 'hla_match_b_high',
       'comorbidity_score', 'karnofsky_score', 'hla_low_res_8',
       'hla_match_drb1_high', 'hla_low_res_10', 'efs_time',
       'dri_score_numeric', 'psych_disturb_numeric', 'cyto_score_numeric',
       'diabetes_numeric', 'tbi_status_numeric', 'arrhythmia_numeric',
       'graft_type_numeric', 'vent_hist_numeric', 'renal_issue_numeric',
       'pulm_severe_numeric', 'prim_disease_hct_numeric', 'cmv_status_numeric',
       'tce_imm_match_numeric', 'rituximab_numeric', 'prod_type_numeric',
       'conditioning_intensity_numeric', 'ethnicity_numeric',
       'obesity_numeric', 'mrd_hct_numeric', 'in_vivo_tcd_numeric',
       'tce_match_numeric', 'hepatic_severe_numeric', 'prior_tumor_numeric',
       'peptic_ulcer_numeric', 'gvhd_proph_numeric', 'rheum_issue_numeric',
       'sex_match_numeric', 'race_group_numeric', 'hepatic_mild_numeric',
       'tce_div_match_numeric', 'donor_related_numeric',
       'melphalan_dose_numeric', 'cardiac_numeric', 'pulm_moderate_numeric']

features2 = ['ID', 'hla_match_c_high', 'hla_high_res_8', 'hla_low_res_6',
       'hla_high_res_6', 'hla_high_res_10', 'hla_match_dqb1_high',
       'hla_nmdp_6', 'hla_match_c_low', 'hla_match_drb1_low',
       'hla_match_dqb1_low', 'year_hct', 'hla_match_a_high', 'donor_age',
       'hla_match_b_low', 'age_at_hct', 'hla_match_a_low', 'hla_match_b_high',
       'comorbidity_score', 'karnofsky_score', 'hla_low_res_8',
       'hla_match_drb1_high', 'hla_low_res_10',
       'dri_score_numeric', 'psych_disturb_numeric', 'cyto_score_numeric',
       'diabetes_numeric', 'tbi_status_numeric', 'arrhythmia_numeric',
       'graft_type_numeric', 'vent_hist_numeric', 'renal_issue_numeric',
       'pulm_severe_numeric', 'prim_disease_hct_numeric', 'cmv_status_numeric',
       'tce_imm_match_numeric', 'rituximab_numeric', 'prod_type_numeric',
       'conditioning_intensity_numeric', 'ethnicity_numeric',
       'obesity_numeric', 'mrd_hct_numeric', 'in_vivo_tcd_numeric',
       'tce_match_numeric', 'hepatic_severe_numeric', 'prior_tumor_numeric',
       'peptic_ulcer_numeric', 'gvhd_proph_numeric', 'rheum_issue_numeric',
       'sex_match_numeric', 'race_group_numeric', 'hepatic_mild_numeric',
       'tce_div_match_numeric', 'donor_related_numeric',
       'melphalan_dose_numeric', 'cardiac_numeric', 'pulm_moderate_numeric']
X = train[features]
X2 = train[features2]


X = X.fillna(value = 0)
X2 = X2.fillna(value = 0)


X.info()


#looking for best fit for efs(target Variable)
feat_select = SelectKBest(f_classif, k='all')
feat_select.fit_transform(X, y)
feat_pvals = pd.DataFrame({'Feature' : X.columns, 'p_value' : feat_select.pvalues_}).sort_values('p_value') 
feat_pvals[feat_pvals['p_value'] < 0.05]


#Looking for best fit for efs_time(Secondary target variable)
feat_select2 = SelectKBest(f_classif, k='all')
feat_select2.fit_transform(X2, y2)
feat_pvals2 = pd.DataFrame({'Feature' : X2.columns, 'p_value' : feat_select2.pvalues_}).sort_values('p_value') 
feat_pvals2[feat_pvals2['p_value'] < 0.05]


feat_pvals['Feature'].values


feat_pvals2['Feature'].values


features = ['graft_type_numeric', 'efs_time', 'age_at_hct']
X = X[features]


features2 = ['graft_type_numeric', 'age_at_hct',
       'conditioning_intensity_numeric', 'karnofsky_score',
       'in_vivo_tcd_numeric', 'hepatic_severe_numeric', 'year_hct',
       'dri_score_numeric', 'donor_age', 'gvhd_proph_numeric',
       'tce_div_match_numeric', 'cmv_status_numeric', 'ethnicity_numeric',
       'tce_imm_match_numeric', 'cyto_score_numeric',
       'peptic_ulcer_numeric', 'hla_match_drb1_high', 'hla_high_res_8',
       'vent_hist_numeric', 'hla_nmdp_6', 'rheum_issue_numeric',
       'hla_high_res_6', 'race_group_numeric', 'mrd_hct_numeric',
       'hla_high_res_10', 'hla_match_a_high', 'hla_match_b_low',
       'hla_match_drb1_low', 'hla_low_res_8', 'hla_low_res_6',
       'hla_match_b_high', 'hla_low_res_10', 'hla_match_c_high',
       'hla_match_dqb1_low', 'hla_match_a_low', 'hla_match_c_low',
       'tce_match_numeric', 'hla_match_dqb1_high', 'rituximab_numeric',
       'prim_disease_hct_numeric', 'donor_related_numeric',
       'tbi_status_numeric', 'melphalan_dose_numeric',
       'hepatic_mild_numeric', 'prod_type_numeric', 'sex_match_numeric',
       'renal_issue_numeric', 'prior_tumor_numeric', 'arrhythmia_numeric',
       'diabetes_numeric', 'obesity_numeric', 'cardiac_numeric',
       'pulm_moderate_numeric', 'pulm_severe_numeric',
       'psych_disturb_numeric', 'comorbidity_score']
X2 = X2[features2]


#Model 1 - efs predict
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=2)


#Model 2 - efs_time predict
train_X2, val_X2, train_y2, val_y2 = train_test_split(X2, y2, random_state=2)


#Model 1 - efs predict
rf_model = RandomForestRegressor(random_state=2)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
print('The accuracy of the model is: ', rf_model.score(val_X, val_y)) 
print('The accuracy of the training model is: ', rf_model.score(train_X, train_y))


#Model 2 - efs_time predict
rf_model2 = RandomForestRegressor(random_state=2)
rf_model2.fit(train_X2, train_y2)
rf_val_predictions2 = rf_model2.predict(val_X2)
rf_val_mae2 = mean_absolute_error(rf_val_predictions2, val_y2)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae2))
print('The accuracy of the model is: ', rf_model2.score(val_X2, val_y2)) 
print('The accuracy of the training model is: ', rf_model2.score(train_X2, train_y2))


#Model 3 - Running a linear regression model too on just the efs predict features
linear_model = LinearRegression()
linear_model.fit(train_X, train_y)
linear_val_predictions = rf_model.predict(val_X)
linear_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(linear_val_mae))
print('The accuracy of the model is: ', linear_model.score(val_X, val_y)) 
print('The accuracy of the training model is: ', linear_model.score(train_X, train_y))


#Model 1 - efs predict
rf_model_on_full_data = RandomForestRegressor(random_state=2)

# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(X,y)


#Model 2 - efs_time predict
rf_model_on_full_data2 = RandomForestRegressor(random_state=2)

# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data2.fit(train_X2,train_y2)


test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')

test['dri_score_numeric'] = pd.factorize(test['dri_score'])[0]
test['psych_disturb_numeric'] = pd.factorize(test['psych_disturb'])[0]
test['cyto_score_numeric'] = pd.factorize(test['cyto_score'])[0]
test['diabetes_numeric'] = pd.factorize(test['diabetes'])[0]
test['tbi_status_numeric'] = pd.factorize(test['tbi_status'])[0]
test['arrhythmia_numeric'] = pd.factorize(test['arrhythmia'])[0]
test['graft_type_numeric'] = pd.factorize(test['graft_type'])[0]
test['vent_hist_numeric'] = pd.factorize(test['vent_hist'])[0]
test['renal_issue_numeric'] = pd.factorize(test['renal_issue'])[0]
test['pulm_severe_numeric'] = pd.factorize(test['pulm_severe'])[0]
test['prim_disease_hct_numeric'] = pd.factorize(test['prim_disease_hct'])[0]
test['cmv_status_numeric'] = pd.factorize(test['cmv_status'])[0]
test['tce_imm_match_numeric'] = pd.factorize(test['tce_imm_match'])[0]
test['rituximab_numeric'] = pd.factorize(test['rituximab'])[0]
test['prod_type_numeric'] = pd.factorize(test['cyto_score_detail'])[0]
test['conditioning_intensity_numeric'] = pd.factorize(test['conditioning_intensity'])[0]
test['ethnicity_numeric'] = pd.factorize(test['ethnicity'])[0]
test['obesity_numeric'] = pd.factorize(test['obesity'])[0]
test['mrd_hct_numeric'] = pd.factorize(test['mrd_hct'])[0]
test['in_vivo_tcd_numeric'] = pd.factorize(test['in_vivo_tcd'])[0]
test['tce_match_numeric'] = pd.factorize(test['tce_match'])[0]
test['hepatic_severe_numeric'] = pd.factorize(test['hepatic_severe'])[0]
test['prior_tumor_numeric'] = pd.factorize(test['prior_tumor'])[0]
test['peptic_ulcer_numeric'] = pd.factorize(test['peptic_ulcer'])[0]
test['gvhd_proph_numeric'] = pd.factorize(test['gvhd_proph'])[0]
test['rheum_issue_numeric'] = pd.factorize(test['rheum_issue'])[0]
test['sex_match_numeric'] = pd.factorize(test['sex_match'])[0]
test['race_group_numeric'] = pd.factorize(test['race_group'])[0]
test['hepatic_mild_numeric'] = pd.factorize(test['hepatic_mild'])[0]
test['tce_div_match_numeric'] = pd.factorize(test['tce_div_match'])[0]
test['donor_related_numeric'] = pd.factorize(test['donor_related'])[0]
test['melphalan_dose_numeric'] = pd.factorize(test['melphalan_dose'])[0]
test['cardiac_numeric'] = pd.factorize(test['cardiac'])[0]
test['pulm_moderate_numeric'] = pd.factorize(test['pulm_moderate'])[0]




#Model 2 - efs_time predict
test_X2 = test[features2].fillna(value = 0)

test_preds2 = rf_model_on_full_data2.predict(test_X2)





#Model 3 - Linear Regression
#Model 2 - efs_time predict
linear_model_on_full_data = LinearRegression()

# fit rf_model_on_full_data on all data from the training data
linear_model_on_full_data.fit(X,y)


#Model 1 RFR- efs predict
test['efs_time'] = test_preds2
test_X = test[features].fillna(value = 0)

test_preds = rf_model_on_full_data.predict(test_X)


#Model 3 LR- efs predict
test['efs_time'] = test_preds2
test_X = test[features].fillna(value = 0)

test_preds3 = linear_model_on_full_data.predict(test_X)


output = pd.DataFrame({'ID': test.ID,
                       'prediction': test_preds})
output2 = pd.DataFrame({'ID': test.ID,
                       'prediction': test_preds3})



#RandomForestRegressor Results - Score 0.506
output


#Linear Regression Results - Score 0.505
output2


output2.to_csv('submission.csv', index=False)

