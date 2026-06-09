## Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import r2_score,mean_squared_error

from sklearn.model_selection import KFold

from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor,GradientBoostingRegressor,RandomForestRegressor,VotingRegressor
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor


!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


pth_data_dict= "/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv"
pth_train="/kaggle/input/equity-post-HCT-survival-predictions/train.csv"
pth_test="/kaggle/input/equity-post-HCT-survival-predictions/test.csv"
pth_sample= "/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv"

df_train = pd.read_csv(pth_train)
df_test=pd.read_csv(pth_test)
df_desc=pd.read_csv(pth_data_dict)
df_sample = pd.read_csv(pth_sample)


class MinImputer(TransformerMixin):
    def fit(self,X,y=None):
        self.fill_value = [X[col].value_counts().index[-1] for col in X.columns]
        return self
    def transform(self, X):
        return np.where(X.isna(), self.fill_value, X)


class Preprocessing():
    def __init__(self):
        #Categorical
        self.not_done_cols = ['psych_disturb','diabetes','arrhythmia','renal_issue','pulm_severe','obesity','hepatic_severe','prior_tumor','peptic_ulcer','rheum_issue','hepatic_mild','cardiac','pulm_moderate']
        self.not_tested_cols = ['cyto_score','cyto_score_detail']
        self.min_columns = ['cmv_status','comorbidity_score', 'conditioning_intensity', 
                       'donor_related', 'dri_score', 'efs', 'efs_time', 'ethnicity', 'graft_type', 'gvhd_proph', 
                       'hla_high_res_10', 'hla_high_res_6', 'hla_high_res_8', 'hla_low_res_10', 'hla_low_res_6', 
                       'hla_low_res_8', 'hla_match_a_high', 'hla_match_a_low', 'hla_match_b_high', 'hla_match_b_low', 
                       'hla_match_c_high', 'hla_match_c_low', 'hla_match_dqb1_high', 'hla_match_dqb1_low', 'hla_match_drb1_high', 
                       'hla_match_drb1_low', 'hla_nmdp_6', 'in_vivo_tcd','karnofsky_score','melphalan_dose',
                       'mrd_hct', 'prim_disease_hct', 'prod_type', 'race_group', 'rituximab', 'sex_match', 'tbi_status',
                       'tce_div_match', 'tce_imm_match', 'tce_match', 'vent_hist','year_hct']
        self.min_columns_ordinal=['cmv_status','conditioning_intensity','donor_related','dri_score','ethnicity','graft_type','gvhd_proph','in_vivo_tcd','melphalan_dose','mrd_hct','prim_disease_hct','prod_type','race_group','rituximab','sex_match','tbi_status','tce_div_match','tce_imm_match','tce_match','vent_hist']
        self.min_columns_nominal = ['comorbidity_score','hla_high_res_10','hla_high_res_6','hla_high_res_8','hla_low_res_10','hla_low_res_6','hla_low_res_8','hla_match_a_high','hla_match_a_low','hla_match_b_high','hla_match_b_low','hla_match_c_high','hla_match_c_low','hla_match_dqb1_high','hla_match_dqb1_low','hla_match_drb1_high','hla_match_drb1_low','hla_nmdp_6','karnofsky_score','year_hct',]
        self.all_ordinal_columns = self.not_done_cols + self.not_tested_cols + self.min_columns_ordinal

        #Numerical
        self.numerical_columns = ['donor_age','age_at_hct']

        #Categorical Imputer
        self.imp_not_done = SimpleImputer(strategy='constant',fill_value='Not done')
        self.imp_not_tested = SimpleImputer(strategy='constant',fill_value='Not tested')
        self.imp_min_nominal = MinImputer()
        self.imp_min_ordinal = MinImputer()

        self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)

        #Numerical Imputer
        self.knn_imputer = KNNImputer(n_neighbors=5)
        
    def fit(self,X,y=None):
        self.imp_not_done.fit(X[self.not_done_cols])
        self.imp_not_tested.fit(X[self.not_tested_cols])
        self.imp_min_nominal.fit(X[self.min_columns_nominal])
        self.imp_min_ordinal.fit(X[self.min_columns_ordinal])
        self.ordinal_encoder.fit(X[self.all_ordinal_columns])
        
        self.knn_imputer.fit(X[self.numerical_columns])

    def transform(self,X,y=None):
        X[self.not_done_cols] = self.imp_not_done.transform(X[self.not_done_cols])
        X[self.not_tested_cols] = self.imp_not_tested.transform(X[self.not_tested_cols])
        X[self.min_columns_nominal] = self.imp_min_nominal.transform(X[self.min_columns_nominal])
        X[self.min_columns_ordinal] = self.imp_min_ordinal.transform(X[self.min_columns_ordinal])
        X[self.all_ordinal_columns] = self.ordinal_encoder.transform(X[self.all_ordinal_columns])
        X[self.numerical_columns] = self.knn_imputer.transform(X[self.numerical_columns])
        
        return X
    def fit_transform(self,X,y=None):
        self.fit(X,y)
        return self.transform(X,y)


transformer = Preprocessing()
df_transform = df_train.copy()
train = transformer.fit_transform(df_transform)
test = transformer.transform(df_test)
submission = test[['ID']]

train.drop(columns='ID',inplace=True)
test.drop(columns='ID',inplace=True)


from lifelines import KaplanMeierFitter
def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return y
train["y"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')



X = train.drop(columns=['efs','efs_time','y'])
y = train[['y']]
FEATURES=X.columns
x_train,x_test,y_train,y_test = train_test_split(train[FEATURES],train[['y']],test_size=0.2)



cat_params1={'task_type'           : "GPU",
       'eval_metric'         : "RMSE",
       'bagging_temperature' : 0.50,
       'iterations'          : 3096,
       'learning_rate'       : 0.08,
       'max_depth'           : 12,
       'l2_leaf_reg'         : 1.25,
       'min_data_in_leaf'    : 24,
       'random_strength'     : 0.25, 
       'verbose'             : 0,
      }
        
cat_params2={'task_type'           : "GPU",
       'eval_metric'         : "RMSE",
       'bagging_temperature' : 0.60,
       'iterations'          : 3096,
       'learning_rate'       : 0.08,
       'max_depth'           : 12,
       'l2_leaf_reg'         : 1.25,
       'min_data_in_leaf'    : 24,
       'random_strength'     : 0.20, 
       'max_bin'             :2048,
       'verbose'             : 0,
      }
models={
    'CatBoost2':CatBoostRegressor(**cat_params2),
    'LightGBM': LGBMRegressor(device='gpu',n_jobs=-1,verbose=-1),
    # 'CatBoost1':CatBoostRegressor(**cat_params1),
    'XGBoost' : XGBRegressor(eval_metric='logloss',tree_method='gpu_hist',verbosity=0),
}


voting_model = [(name,model) for name,model in models.items()]
base_weight = 0.25
tuned_weight = 0.50 / len(models)
weights = [base_weight] + [base_weight] + [tuned_weight]
weights = [0.7,0.5,0.3]
# weights = [0.7,0.5]

voting_reg = VotingRegressor(estimators = voting_model,weights=weights)
voting_reg.fit(x_train,y_train)
y_pred = voting_reg.predict(x_test)

#Accuracy
r2_test= r2_score(y_test,y_pred)
mse_test= mean_squared_error(y_test,y_pred)

print(f"Mean Square Error On Test : {mse_test:.4f}")
print(f"R2 Score On Test : {r2_test:.4f}")


submission['prediction'] = voting_reg.transform(test).mean(axis=1)
submission.to_csv("submission.csv",index=False)
print("Sub shape:",submission.shape)
submission.head()




