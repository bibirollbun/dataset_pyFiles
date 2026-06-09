import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('darkgrid')
sns.set_theme('notebook')
palette = sns.color_palette("coolwarm")
from warnings import filterwarnings
filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")


train.sample(10)


number_of_rows = train.shape[0]
number_of_columns = train.shape[1]
print(f"number of rows: {number_of_rows}", f"number of columns: {number_of_columns}")


train.info()


train.isnull().sum() / train.shape[0] * 100


plt.figure(figsize=(12,9))
sns.barplot(y=train.columns, x=train.isnull().sum().values / train.shape[0] * 100, palette=palette)
for index, value in enumerate(train.isnull().sum().values / train.shape[0] * 100):
    plt.text(value, index, f'{value:.1f} %', ha='left',va='center', fontsize=6.5)
plt.legend(['Percentage of missing values'])
plt.xlabel('Percentage of missing values')
plt.ylabel('Features')
plt.title('Percentage of missing values in each feature')
plt.show()


train.describe()


train.efs.value_counts()


plt.figure(figsize=(12,9))
plt.pie(train.efs.value_counts().values, labels= train.efs.value_counts().index, colors=palette, autopct='%1.1f%%')
plt.title('Distribution of the target variable')
plt.show()


train.tce_match.unique()


train.tce_match.value_counts()


plt.figure(figsize=(12,9))
plt.pie(train.tce_match.value_counts().values, labels= train.tce_match.value_counts().index, colors=palette, autopct='%1.1f%%')
plt.title('Distribution of the tce_match variable')
plt.show()


train.mrd_hct.value_counts()


plt.figure(figsize=(12,9))
plt.pie(train.mrd_hct.value_counts().values, labels= train.mrd_hct.value_counts().index, colors=palette, autopct='%1.1f%%')
plt.title('Distribution of the mrd_hct variable')
plt.show()


train.cyto_score_detail.value_counts()


plt.figure(figsize=(12,9))
plt.pie(train.cyto_score_detail.value_counts().values, labels= train.cyto_score_detail.value_counts().index, colors=palette, autopct='%1.1f%%')
plt.title('Distribution of the cyto_score_detail variable')
plt.show()


train.tce_div_match.value_counts()


plt.figure(figsize=(12,9))
plt.pie(train.tce_div_match.value_counts().values, labels= train.tce_div_match.value_counts().index, colors=palette, autopct='%1.1f%%')
plt.title('Distribution of the tce_div_match variable')
plt.show()


train.tce_imm_match.value_counts()


plt.figure(figsize=(12,9))
plt.pie(train.tce_imm_match.value_counts().values, labels= train.tce_imm_match.value_counts().index, colors=palette, autopct='%1.1f%%')
plt.title('Distribution of the tce_imm_match variable')
plt.show()


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


RMV = ["ID","efs","efs_time"]
FEATURES = [c for c in train.columns if not c in RMV]
CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("No Data")
        test[c] = test[c].fillna("No Data")
    else:
        train[c] = train[c].fillna(-1)
        test[c] = test[c].fillna(-1)


useful_columns = []


cont_table = pd.crosstab(train.dri_score, train.efs)
cont_table


import scipy.stats as stats
stats.chi2_contingency(cont_table)


useful_columns.append('dri_score')


cont_table = pd.crosstab(train.efs, train.psych_disturb)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('psych_disturb')


cont_table = pd.crosstab(train.efs, train.cyto_score)
cont_table


stats.chi2_contingency(cont_table, correction=True)


cont_table = pd.crosstab(train.efs, train.diabetes)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('diabetes')


cont_table = pd.crosstab(train.efs, train.tbi_status)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('tbi_status')


cont_table = pd.crosstab(train.efs, train.arrhythmia)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('arrhythmia')


cont_table = pd.crosstab(train.efs, train.graft_type)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('graft_type')


cont_table = pd.crosstab(train.efs, train.vent_hist)
cont_table


stats.chi2_contingency(cont_table, correction=True)


cont_table = pd.crosstab(train.efs, train.renal_issue)
cont_table


stats.chi2_contingency(cont_table, correction=True)


cont_table = pd.crosstab(train.efs, train.pulm_severe)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('pulm_severe')


cont_table = pd.crosstab(train.efs, train.prim_disease_hct)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('prim_disease_hct')


cont_table = pd.crosstab(train.efs, train.cmv_status)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('cmv_status')


cont_table = pd.crosstab(train.efs, train.tce_imm_match)
cont_table


stats.chi2_contingency(cont_table, correction=True)


cont_table = pd.crosstab(train.efs, train.rituximab)
cont_table


stats.chi2_contingency(cont_table, correction=True)


cont_table = pd.crosstab(train.efs, train.prod_type)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('prod_type')


cont_table = pd.crosstab(train.efs, train.cyto_score_detail)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('cyto_score_detail')


cont_table = pd.crosstab(train.efs, train.conditioning_intensity)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('conditioning_intensity')


cont_table = pd.crosstab(train.efs, train.ethnicity)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('ethnicity')


cont_table = pd.crosstab(train.efs, train.obesity)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('obesity')


cont_table = pd.crosstab(train.efs, train.mrd_hct)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('mrd_hct')


cont_table = pd.crosstab(train.efs, train.in_vivo_tcd)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('in_vivo_tcd')


cont_table = pd.crosstab(train.efs, train.tce_match)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('tce_match')


cont_table = pd.crosstab(train.efs, train.hepatic_severe)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('hepatic_severe')


cont_table = pd.crosstab(train.efs, train.prior_tumor)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('prior_tumor')


cont_table = pd.crosstab(train.efs, train.peptic_ulcer)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('peptic_ulcer')


cont_table = pd.crosstab(train.efs, train.gvhd_proph)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('gvhd_proph')


cont_table = pd.crosstab(train.efs, train.rheum_issue)
cont_table


stats.chi2_contingency(cont_table, correction=True)


cont_table = pd.crosstab(train.efs, train.sex_match)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('sex_match')


cont_table = pd.crosstab(train.efs, train.race_group)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('race_group')


cont_table = pd.crosstab(train.efs, train.hepatic_mild)
cont_table


stats.chi2_contingency(cont_table, correction=True)


cont_table = pd.crosstab(train.efs, train.tce_div_match)
cont_table


stats.chi2_contingency(cont_table, correction=True)


cont_table = pd.crosstab(train.efs, train.donor_related)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('donor_related')


cont_table = pd.crosstab(train.efs, train.melphalan_dose)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('melphalan_dose')


cont_table = pd.crosstab(train.efs, train.cardiac)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('cardiac')


cont_table = pd.crosstab(train.efs, train.pulm_moderate)
cont_table


stats.chi2_contingency(cont_table, correction=True)


useful_columns.append('pulm_moderate')


stats.pointbiserialr(train.efs, train.hla_match_c_high)


useful_columns.append('hla_match_c_high')


stats.pointbiserialr(train.efs, train.hla_high_res_8)


useful_columns.append('hla_high_res_8')


stats.pointbiserialr(train.efs, train.hla_low_res_6)


useful_columns.append('hla_low_res_6')


stats.pointbiserialr(train.efs, train.hla_high_res_6)


useful_columns.append('hla_high_res_6')


stats.pointbiserialr(train.efs, train.hla_high_res_10)


useful_columns.append('hla_high_res_10')


stats.pointbiserialr(train.efs, train.hla_match_dqb1_high)


stats.pointbiserialr(train.efs, train.hla_nmdp_6)


useful_columns.append('hla_nmdp_6')


stats.pointbiserialr(train.efs, train.hla_match_c_low)


useful_columns.append('hla_match_c_low')


stats.pointbiserialr(train.efs, train.hla_match_drb1_low)


useful_columns.append('hla_match_drb1_low')


stats.pointbiserialr(train.efs, train.hla_match_dqb1_low)


useful_columns.append('hla_match_dqb1_low')


stats.pointbiserialr(train.efs, train.year_hct)


useful_columns.append('year_hct')


stats.pointbiserialr(train.efs, train.hla_match_a_high)


useful_columns.append('hla_match_a_high')


stats.pointbiserialr(train.efs, train.donor_age)


stats.pointbiserialr(train.efs, train.hla_match_b_low)


useful_columns.append('hla_match_b_low')


stats.pointbiserialr(train.efs, train.age_at_hct)


useful_columns.append('age_at_hct')


stats.pointbiserialr(train.efs, train.hla_match_a_low)


useful_columns.append('hla_match_a_low')


stats.pointbiserialr(train.efs, train.hla_match_b_high)


useful_columns.append('hla_match_b_high')


stats.pointbiserialr(train.efs, train.comorbidity_score)


useful_columns.append('comorbidity_score')


stats.pointbiserialr(train.efs, train.karnofsky_score)


useful_columns.append('karnofsky_score')


stats.pointbiserialr(train.efs, train.hla_low_res_8)


useful_columns.append('hla_low_res_8')


stats.pointbiserialr(train.efs, train.hla_match_drb1_high)


useful_columns.append('hla_match_drb1_high')


stats.pointbiserialr(train.efs, train.hla_low_res_10)


useful_columns.append('hla_low_res_10')


stats.pointbiserialr(train.efs, train.efs_time)


train_set = train[useful_columns]


train = pd.concat([train_set, train.efs, train.efs_time], axis=1)


train.isnull().sum()


from sklearn.preprocessing import LabelEncoder
categorical_columns = train.select_dtypes(include='object').columns.to_list()
encoder = LabelEncoder()
for column in categorical_columns:
    train[column] = encoder.fit_transform(train[column])



train.columns


test.columns


X_train = train.drop(['efs', 'efs_time'], axis=1)
T_train = train['efs_time']
E_train = train['efs']


X_train.isnull().sum().sum()


T_train.isnull().sum().sum()


E_train.isnull().sum().sum()


submission = test['ID']


test = test[useful_columns]


for column in categorical_columns:
    test[column] = encoder.fit_transform(test[column])


train.info()


train = train.astype('float64')


test = test.astype('float64')


import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier


X_train = train.drop(['efs', 'efs_time'], axis = 1)
T_train = train['efs_time']
E_train = train['efs']
X_test = test



xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train, E_train)

# Generate risk scores for training data and testing data
X_train_risk = xgb_model.predict_proba(X_train)[:, 1]
X_test_risk = xgb_model.predict_proba(X_test)[:, 1]
# Combine original features with the new risk scores for training data and testing data
train_combined = pd.concat([X_train, pd.Series(X_train_risk, name='risk_score')], axis=1)
test_combined = pd.concat([X_test, pd.Series(X_test_risk, name = 'risk_score')], axis= 1)


train_combined.head()


test_combined.head()


submission = pd.concat([submission, pd.Series(X_test_risk, name = 'prediction')], axis=1)
submission


submission.to_csv('submission.csv', index=False)




