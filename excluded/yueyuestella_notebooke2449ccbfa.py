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


import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns',None)  

data = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv")
test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
sample = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")


from sklearn.ensemble import StackingRegressor 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import RandomForestClassifier  
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error  
from sklearn.metrics import r2_score  
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split  
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import OrdinalEncoder 
from sklearn.preprocessing import StandardScaler 
from sklearn.impute import SimpleImputer 
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer 
import xgboost as XGB 
import lightgbm as lgb


from warnings import filterwarnings

filterwarnings('ignore')

train.head()


data.head(10)


train.sample(10)


sample.head(10)


train.describe()


train_corr = train.select_dtypes(include = ['int64','float64'],exclude = ['category'])
train_corr.corr()


for i in train_corr.columns:
    print(f'{i}:{train_corr[i].skew()}')


sns.histplot(train['efs_time'], bins=50)


train_category = train.select_dtypes(include=['object'],exclude=['int64','float64'])
train_category['efs']=train['efs']


plt.hist(train.loc[train.efs==1,"efs_time"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"efs_time"],bins=100,label="efs=0, Maybe Event")
plt.show()


train.isnull().sum()/len(train)*100


null_columns_n = train_corr.columns[train_corr.isnull().any()].tolist()
null_columns_n


null_columns_c = train_category.columns[train_category.isnull().any()].tolist()
null_columns_c


X=train.drop(['efs','ID', 'efs_time'],axis=1)
y=train['efs']
X_train,X_test,y_train,y_test= train_test_split(X,y,random_state=42,test_size=0.2)

test_=test.drop(['ID'],axis=1)


num=SimpleImputer(strategy='median')
cat=SimpleImputer(strategy='most_frequent')
ohe=OneHotEncoder(sparse_output=False,drop='first')


trans=ColumnTransformer([('n',num,['hla_match_c_high','hla_high_res_8','hla_low_res_6','hla_high_res_6','hla_high_res_10','hla_match_dqb1_high',
 'hla_nmdp_6','hla_match_c_low','hla_match_drb1_low','hla_match_dqb1_low','hla_match_a_high','donor_age','hla_match_b_low','hla_match_a_low','hla_match_b_high','comorbidity_score','karnofsky_score','hla_low_res_8','hla_match_drb1_high','hla_low_res_10']),
                          ('c',cat,['dri_score', 'psych_disturb', 'cyto_score', 'diabetes', 'arrhythmia', 'vent_hist', 'renal_issue', 'pulm_severe', 'cmv_status', 'tce_imm_match', 'rituximab', 'cyto_score_detail', 'conditioning_intensity', 'ethnicity', 'obesity',
                                    'mrd_hct', 'in_vivo_tcd', 'tce_match', 'hepatic_severe', 'prior_tumor', 'peptic_ulcer', 'gvhd_proph', 'rheum_issue', 'sex_match', 'hepatic_mild', 'tce_div_match', 'donor_related', 'melphalan_dose', 'cardiac', 
                                    'pulm_moderate'])],remainder='passthrough')


X_train_=trans.fit_transform(X_train)
X_test_=trans.transform(X_test)
test_pre=trans.transform(test_)


column_names = trans.get_feature_names_out()
column_names1 = [i.replace('n__','') for i in column_names]
column_names2 = [i.replace('c__','') for i in column_names1]
column_name = [i.replace('remainder__','') for i in column_names2]


X_train_n=pd.DataFrame(X_train_,columns=column_name)
X_test_n=pd.DataFrame(X_test_,columns=column_name)
test_pre_n=pd.DataFrame(test_pre,columns=column_name)


X_train_n['sex_match']


X_train_n[['Donor_Sex','Recipient_Sex']] = [x.split('-') for x in X_train_n['sex_match']]
X_train_n[['CMV_Don','CMV_rec',]] = [x.split('/') for x in X_train_n['cmv_status']]

X_test_n[['Donor_Sex','Recipient_Sex']] = [x.split('-') for x in X_test_n['sex_match']]
X_test_n[['CMV_Don','CMV_rec']] = [x.split('/') for x in X_test_n['cmv_status']]

test_pre_n[['Donor_Sex','Recipient_Sex']] = [x.split('-') for x in test_pre_n['sex_match']]
test_pre_n[['CMV_Don','CMV_rec',]] = [x.split('/') for x in test_pre_n['cmv_status']]


X_train_n=X_train_n.drop(['sex_match','cmv_status'],axis=1)
X_test_n=X_test_n.drop(['sex_match','cmv_status'],axis=1)
test_pre_n=test_pre_n.drop(['sex_match','cmv_status'],axis=1)


trans_f=ColumnTransformer(transformers=[('f',ohe,['psych_disturb',
      'vent_hist', 'pulm_severe','donor_related',
       'melphalan_dose', 'cardiac','dri_score', 'cyto_score', 'diabetes', 'tbi_status',
       'arrhythmia', 'graft_type', 'renal_issue',
       'prim_disease_hct','tce_imm_match','rituximab',
       'prod_type', 'cyto_score_detail', 'conditioning_intensity', 'ethnicity',
       'obesity', 'mrd_hct', 'in_vivo_tcd', 'tce_match', 'hepatic_severe',
       'prior_tumor', 'peptic_ulcer', 'gvhd_proph','rheum_issue',
       'race_group','hepatic_mild','tce_div_match','pulm_moderate','Donor_Sex','Recipient_Sex','CMV_Don','CMV_rec'])],remainder='passthrough')


X_train_final=trans_f.fit_transform(X_train_n)
X_test_final=trans_f.transform(X_test_n)
test_final = trans_f.transform(test_pre_n)

print(X_train_final.shape)
print(X_test_final.shape)


column_names_f = trans_f.get_feature_names_out()

column_names1_f = [i.replace('n__','') for i in column_names_f]

column_names2_f= [i.replace('c__','') for i in column_names1_f]

column_namef = [i.replace('remainder__','') for i in column_names2_f]


# import xgboost as XGB
# from sklearn.metrics import roc_auc_score

# XG_boost = XGB.XGBClassifier(
#     max_depth=6,
#     learning_rate=0.042,
#     n_estimators=300,
#     n_jobs=-1,
#     verbosity=0,
#     booster='dart',
#     random_state=42,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     gamma=0.1,
#     use_label_encoder=False, 
#     eval_metric='logloss'   
# )

# XG_boost.fit(X_train_final, y_train)

# predict_proba_xg = XG_boost.predict_proba(X_test_final)[:, 1] 
# print("XGBoost AUC:", auc)


train_data = lgb.Dataset(X_train_final, label=y_train)
test_data = lgb.Dataset(X_test_final, label=y_test)


params = {
    'max_depth': 10,            
    'learning_rate': 0.042,    
    'n_estimators': 400,        
    'boosting_type': 'dart', 
    'subsample': 0.8,          
    'colsample_bytree': 0.8,    
    'min_split_gain': 0.1,      
    'verbose': -1    
}

lgb_model = lgb.train(params, train_data)
predict_lgb = lgb_model.predict(X_test_final)
lgb_cv_scores = cross_val_score(
    lgb.LGBMClassifier(**params),
    X_train_final,
    y_train,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

print(f"each-AUC: {lgb_cv_scores}")
print(f"avg-AUC: {lgb_cv_scores.mean():.4f} (±{lgb_cv_scores.std()*2:.4f})")

auc = roc_auc_score(y_test, predict_lgb)
print("LightGBM AUC:", auc)


predict_xg=lgb_model.predict(test_final)
predict_xg = np.round(predict_xg, 1)
Final_pred=pd.DataFrame(predict_xg,columns=['prediction'])
Final_pred['ID'] = test['ID']                 
For_sub_l = Final_pred[['ID', 'prediction']]
For_sub_l.to_csv('submission.csv', index=False)




