pip install /kaggle/input/cibmtr-whl-files-for-installation/scikit_survival-0.20.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl



!pip install /kaggle/input/cibmtr-whl-files-for-installation/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/cibmtr-whl-files-for-installation/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/cibmtr-whl-files-for-installation/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/cibmtr-whl-files-for-installation/formulaic-1.1.1-py3-none-any.whl
!pip install /kaggle/input/cibmtr-whl-files-for-installation/lifelines-0.30.0-py3-none-any.whl


import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
import numpy as np


test=pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")



data=pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
data.head()


data['age_comorbidity_interaction'] = data['age_at_hct'] * data['comorbidity_score']
test['age_comorbidity_interaction'] = test['age_at_hct'] * test['comorbidity_score']
baseline=2008
data["years_since_baseline"]=data["year_hct"]-baseline
data["years_since_baseline"]
test["years_since_baseline"]=test["year_hct"]-baseline
test["years_since_baseline"]



cat_data=data.select_dtypes(["object"])
number_data=data.select_dtypes(["int64","float64"])
cat_test=test.select_dtypes(["object"])
number_test=test.select_dtypes(["int64","float64"])


cat_imputer = SimpleImputer(strategy='most_frequent')
cat_data = pd.DataFrame(cat_imputer.fit_transform(cat_data), columns=cat_data.columns)

medians = number_data.median()
number_data = number_data.fillna(medians)




number_data.head()


cat_imputer = SimpleImputer(strategy='most_frequent')
cat_test = pd.DataFrame(cat_imputer.fit_transform(cat_test), columns=cat_test.columns)

medians = number_data.median()
number_test = number_test.fillna(medians)

number_test.head()


number_test.isnull().sum()


"""
scaler = StandardScaler()
number_data=number_data.drop("ID",axis=1)
number_test=number_test.drop("ID",axis=1)

number_data_scaled=scaler.fit_transform(number_data)
number_test_scaled=scaler.fit_transform(number_test)
"""


"""
number_data_scaled = pd.DataFrame(number_data_scaled, columns=number_data.columns)
number_test_scaled = pd.DataFrame(number_test_scaled, columns=number_test.columns)
"""


def outliers(data):
    for col in data.columns:
        q1=np.percentile(data[col],25,method="midpoint")
        q3=np.percentile(data[col],75,method="midpoint")
        IQR=q3-q1
        lower_bound=q1-1.5*IQR
        upper_bound=q3+1.5*IQR
        data[col] = np.clip(data[col], lower_bound, upper_bound)
    return data
number_data_scaled = outliers(number_data)
number_test_scaled = outliers(number_test)


data=pd.concat([number_data_scaled,cat_data],axis=1)
data.head()


test=pd.concat([number_test_scaled,cat_test],axis=1)
test.head()



def label_encode_datasets(train_df, test_df, categ_fields):
    train_encoded = data.copy()
    test_encoded = test.copy()
    le = LabelEncoder()
    
    for column in cat_data:
        print(f'Encoding: {column} ...')
        le.fit(train_encoded[column])
        
        train_encoded[column] = le.transform(train_encoded[column])
        if column in test_encoded.columns:
            test_encoded[column] = test_encoded[column].map(lambda s: le.transform([s])[0] if s in le.classes_ else None)
            test_encoded[column].fillna(-1, inplace=True)
            test_encoded[column] = test_encoded[column].astype(int)

    return train_encoded, test_encoded





remove_variables = data[['ID', 'efs', 'efs_time']]
features = [feat for feat in data if feat not in remove_variables]
categorical_features = [feat for feat in data[features] if data[feat].dtype == 'object']
numerical_features = [feat for feat in data[features] if feat not in categorical_features]



trn_encoded, tst_encoded = label_encode_datasets(data, test, categorical_features)


trn_encoded["efs"]=trn_encoded["efs"].abs()


trn_encoded['efs'] = (trn_encoded['efs']).astype(int)

print(trn_encoded['efs'].unique()) 



features=categorical_features+numerical_features




low_variance_cols = ['hla_match_c_high', 'hla_match_dqb1_high', 'hla_match_c_low', 'hla_match_dqb1_low', 'hla_match_a_high']
trn_encoded = trn_encoded.drop(columns=low_variance_cols)
tst_encoded = tst_encoded.drop(columns=low_variance_cols)




cpf = CoxPHFitter(penalizer=0.01)  
cpf.fit(trn_encoded, duration_col="efs_time", event_col="efs", show_progress=True)



tst_encoded.shape


trn_encoded.shape


preds = cpf.predict_partial_hazard(tst_encoded)
preds = preds.values


preds 




predictions_train = cpf.predict_partial_hazard(trn_encoded)

predictions_train




from lifelines.utils import concordance_index
true_times_train = trn_encoded['efs_time']
event_occurred_train = trn_encoded['efs']
c_index_train = concordance_index(true_times_train, -predictions_train.values, event_occurred_train)
c_index_train


sub = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv', index_col='ID')


sub['prediction'] = preds
sub.to_csv('submission.csv')


sub

