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


### training data


train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")


train


test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")


test


train.head(3)


data_dictionary = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')


data_dictionary


sample_submission = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')


sample_submission


### train data -->


train.head()


train.info()


train.shape


## checking is there any null value
train.isnull().sum()


train.describe()


train.columns


test.columns


train.shape


###### 'efs_time' is our target feature


## columns in train


train.columns


## ID


train['ID']


## here ID is unique for every row and every data and this is not important in prediction
train.drop('ID',axis=1,inplace=True)


## list of column we are dropping
col_to_drop = []
col_to_drop.append('ID')


col_to_drop


cat_col_list = [feature for feature in train.columns if train[feature].dtype=='O']
num_col_list = [feature for feature in train.columns if train[feature].dtype!='O']


num_col_list


cat_col_list


train[num_col_list]


## correlation

train[num_col_list].corr()


## importing required libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(18,8))
sns.heatmap(train[num_col_list].corr(),annot=True)
plt.show()


# with the following function we can select highly correlated features
# it will remove the first feature that is correlated with anything other feature

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold: # we are interested only in correlation positively high becuase -ve correlation is important
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


correlation(train[num_col_list], 0.7) ## taking threshold as 0.7


corr_features_list = correlation(train[num_col_list], 0.7)


corr_features_list


type(corr_features_list)


len(corr_features_list)


## so these are highly correlated features


## dropping these highly correlated features
train.drop(corr_features_list,axis=1,inplace=True)


col_to_drop = col_to_drop + list(corr_features_list)


col_to_drop


train.shape


## unique values in every categorical colum
for col in cat_col_list:
    print("Unique values in {}  ====>".format(col))
    print("There are total {} unique values in {}".format(train[col].nunique(), col))
    print(train[col].unique())
    print()


train.columns


## function to plot some graphs and print the dtype of feature
def show_details_and_graphs(df_col):
    col_dtype=train[df_col].dtype
    print("Data type:",col_dtype)
    print("Total null values:",train[df_col].isnull().sum())

    print()
    if(col_dtype=='O'):
        plt.figure(figsize=(15,6))

        plt.subplot(1,2,1)
        plt.title("Bar Plot for '{}'".format(df_col))
        plt.ylabel("Count")
        train['{}'.format(df_col)].value_counts().plot(kind='bar')

        plt.subplot(1,2,2)
        plt.title("Pie Chart for '{}'".format(df_col))
        train['{}'.format(df_col)].value_counts().plot(kind='pie', autopct='%.2f%%')
        plt.show()


    elif(col_dtype!='O'):
        plt.figure(figsize=(18,15))

        plt.subplot(2,2,1)
        plt.title("Histogram for '{}'".format(df_col))
        train['{}'.format(df_col)].plot(kind='hist')

        plt.subplot(2,2,2)
        plt.title("KDE plot for '{}'".format(df_col))
        train['{}'.format(df_col)].plot(kind='kde')

        plt.subplot(2,2,3)
        plt.title("Box Plot for '{}'".format(df_col))
        train['{}'.format(df_col)].plot(kind='box')

        plt.subplot(2,2,4)
        plt.title("Distplot for '{}'".format(df_col))
        sns.distplot(train['{}'.format(df_col)])

        plt.show()


    else:
        print("Datatype of feature is neither numeric not categorical...")


## function to find and print all the rows where outlier is present
def check_outlier(df_col):
    if train[df_col].dtype!='O':
        print("Feature Name : {}".format(df_col))
        df_col_mean = train[df_col].mean()
        df_col_std = train[df_col].std()

        df_col_lower_limit = df_col_mean - 3*df_col_std 
        df_col_upper_limit = df_col_mean + 3*df_col_std 

        print("Based on Z-Score test :")
        print()
        return train[(train[df_col]<df_col_lower_limit) | (train[df_col]>df_col_upper_limit)]
    else:
        print("This is a categorical Feature...")
    


train.head(4)


cat_ohe_cols = [] ## list of categorical columns which require ohe
cat_label_cols = [] ## list of categorical columns which require label encoding
cat_ordinal_cols = [] ## list of categorical columns which require ordinal encoding
num_cols = [] ## list of numerical columns


### dri_score


show_details_and_graphs('dri_score')


# plt.figure(figsize=(15,6))

# plt.subplot(1,2,1)
# plt.title("Bar Plot for 'dri_score'")
# plt.ylabel("Count")
# train['dri_score'].value_counts().plot(kind='bar')

# plt.subplot(1,2,2)
# plt.title("Pie Chart for 'dri_score'")
# train['dri_score'].value_counts().plot(kind='pie', autopct='%.2f%%')

# plt.show()


train['dri_score'].value_counts().sort_values(ascending=False)[:5]

## Top 5 occuring dri_score with highest frequency


dri_group = train.groupby('dri_score')


len(dri_group)


dri_group.size()


dri_group.size().sort_values(ascending=False)


dri_group.first()


dri_group[['efs_time']].mean().sort_values('efs_time',ascending=False) 

## Average time duration patient remains free of disease progression or recurrence after treatment based on 'dri_score' value


dri_group[['efs_time']].mean().sort_values('efs_time',ascending=False).index


# # Define the ordered categories
# categories = [[
#     'N/A - disease not classifiable', 
#     'N/A - non-malignant indication', 
#     'N/A - pediatric', 
#     'Missing disease status',
#     'TBD cytogenetics', 
#     'Low', 
#     'Intermediate - TED AML case <missing cytogenetics',
#     'Intermediate', 
#     'High - TED AML case <missing cytogenetics', 
#     'High', 
#     'Very high'
# ]]


cat_ordinal_cols.append('dri_score')


cat_ordinal_cols


pd.crosstab(train.dri_score,train.efs_time.mean())


##### psych_disturb


show_details_and_graphs('psych_disturb')


train.groupby('psych_disturb')[['efs_time']].mean().sort_values('efs_time',ascending=False) 


cat_ordinal_cols.append('psych_disturb')


# # Define the ordered categories based on efs_time
# categories = [['No', 'Not done', 'Yes']]  


train['psych_disturb'].isnull().sum()


## There are 2062 null values in 'psych_disturb' feature


#### cyto_score


show_details_and_graphs('cyto_score')


train['cyto_score'].unique()


# # Define the order of categories
# categories = [['Not tested', 'TBD', 'Other', 'Normal', 'Favorable', 'Intermediate', 'Poor']]


cat_ordinal_cols.append('cyto_score')


train.columns


#### diabetes


show_details_and_graphs('diabetes')


train.groupby('diabetes')[['efs_time']].mean().sort_values('efs_time',ascending=False) 
## Average efs_time is highest is Not done and lowest if Yes


# # Define the ordered categories based on decreasing efs_time
# categories = [['Not done', 'No', 'Yes']]


cat_ordinal_cols.append('diabetes')


### hla_match_c_high


show_details_and_graphs('hla_match_c_high')


check_outlier('hla_match_c_high')


## These rows seems to contain outliers based on Z-Score upper_limit and lower_limit (mean + 3*std) and (mean - 3*std)


print("Minimum value ",(train['hla_match_c_high']).min())
print("Maximum value ",(train['hla_match_c_high']).max())
print("Mean value ",(train['hla_match_c_high']).mean())
print("Standard Deviation ",(train['hla_match_c_high']).std())


train['hla_match_c_high'].describe()


hla_match_c_high_upper_limit = train['hla_match_c_high'].mean() + 3*train['hla_match_c_high'].std()
hla_match_c_high_lower_limit = train['hla_match_c_high'].mean() - 3*train['hla_match_c_high'].std()


hla_match_c_high_lower_limit,hla_match_c_high_upper_limit


train[train['hla_match_c_high']>hla_match_c_high_upper_limit]


train[train['hla_match_c_high']<hla_match_c_high_lower_limit]


### Here these 79 rows contains 'hla_match_c_high' value less than its lower limit


num_cols.append('hla_match_c_high')


train.columns


#### tbi_status


show_details_and_graphs('tbi_status')


## There is no any null values in this column


train['tbi_status'].value_counts()


train.groupby('tbi_status')[['efs_time']].mean().sort_values('efs_time',ascending=False) 


# categories = [[
#     'TBI +- Other, -cGy, fractionated',  # Highest efs_time (24.66)
#     'No TBI',                            # 23.88
#     'TBI + Cy +- Other',                 # 22.67
#     'TBI +- Other, -cGy, single',        # 21.75
#     'TBI +- Other, >cGy',                # 21.63
#     'TBI +- Other, unknown dose',        # 21.39
#     'TBI +- Other, <=cGy',               # 19.99
#     'TBI +- Other, -cGy, unknown dose'   # Lowest efs_time (18.23)
# ]]


cat_ordinal_cols.append('tbi_status')


### arrhythmia


train['arrhythmia'].dtype


show_details_and_graphs('arrhythmia')


train.groupby('arrhythmia')[['efs_time']].mean().sort_values('efs_time',ascending=False) 


# categories = [['Not done', 'No', 'Yes']]


cat_ordinal_cols.append('arrhythmia')


train['arrhythmia'].value_counts()


## function to check average efs_time with respect ro every category in every feature
def show_avg_efs_time(df_col):
    return train.groupby(df_col)[['efs_time']].mean().sort_values('efs_time',ascending=False) 


### graft_type


show_details_and_graphs('graft_type')


cat_label_cols.append('graft_type')


show_avg_efs_time('graft_type')


### vent_hist


show_details_and_graphs('vent_hist')


show_avg_efs_time('vent_hist')


cat_label_cols.append('vent_hist')


### renal_issue


show_details_and_graphs('renal_issue')


cat_ordinal_cols.append('renal_issue')


(train.isnull().mean()*100).sort_values(ascending=False)


## These features contains more than 40% null 

# ['tce_match',              
# 'mrd_hct',                   
# 'cyto_score_detail' ,        
# 'tce_div_match',             
# 'tce_imm_match']

## so let's drop these


col_to_drop += ['tce_match','mrd_hct','cyto_score_detail' ,'tce_div_match','tce_imm_match']


train.drop(columns=['tce_match','mrd_hct','cyto_score_detail' ,'tce_div_match','tce_imm_match'], inplace=True)


train.shape


train.columns


### pulm_severe


show_details_and_graphs('pulm_severe')


cat_ordinal_cols.append('pulm_severe')


# category = ['Not done','No','Yes']


## prim_disease_hct


show_details_and_graphs('prim_disease_hct')


train['prim_disease_hct'].value_counts()


cat_ohe_cols.append('prim_disease_hct')


### cmv_status


show_details_and_graphs('cmv_status')


train['cmv_status'].unique()


# categories = [['-/-', '-/+', '+/+', '+/-']]


cat_ordinal_cols.append('cmv_status')


### rituximab


train.columns


show_details_and_graphs('rituximab')


# category = ['No','Yes']


cat_label_cols.append('rituximab')


## hla_match_dqb1_low


show_details_and_graphs('hla_match_dqb1_low')


check_outlier('hla_match_dqb1_low')


num_cols.append('hla_match_dqb1_low')


#### prod_type


show_details_and_graphs('prod_type')


cat_label_cols.append('prod_type')


train.columns


### conditioning_intensity


show_details_and_graphs('conditioning_intensity')


show_avg_efs_time('conditioning_intensity')


# categories = [[
#     'MAC', 'N/A, F(pre-TED) not submitted', 'NMA', 'TBD',
#     'No drugs reported', 'RIC'
# ]]


cat_ordinal_cols.append('conditioning_intensity')


### ethnicity


show_details_and_graphs('ethnicity')


show_avg_efs_time('ethnicity')


cat_label_cols.append('ethnicity')


#### year_hct


show_details_and_graphs('year_hct')


check_outlier('year_hct')


num_cols.append('year_hct')


#### obesity


show_details_and_graphs('obesity')


# category = ['Not done','No','Yes']


cat_ordinal_cols.append('obesity')


#### in_vivo_tcd


show_details_and_graphs('in_vivo_tcd')


# category = ['No','Yes']


cat_label_cols.append('in_vivo_tcd')


#### hepatic_severe


show_details_and_graphs('hepatic_severe')


# category = ['Not done','No','Yes']


cat_ordinal_cols.append('hepatic_severe')


train.columns


### donor_age


show_details_and_graphs('donor_age')


check_outlier('donor_age')


num_cols.append('donor_age')


#### prior_tumor


show_details_and_graphs('prior_tumor')


# category = ['Not done','No','Yes']


cat_ordinal_cols.append('prior_tumor')


show_avg_efs_time('prior_tumor')


#### peptic_ulcer


show_details_and_graphs('peptic_ulcer')


# category = ['Not done','No','Yes']


cat_ordinal_cols.append('peptic_ulcer')


#### age_at_hct


show_details_and_graphs('age_at_hct')


check_outlier('age_at_hct')


num_cols.append('age_at_hct')


#### gvhd_proph


show_details_and_graphs('gvhd_proph')


train['gvhd_proph'].value_counts()


len(train['gvhd_proph'].value_counts())


## Label Encoding can be applied.


cat_label_cols.append('gvhd_proph')


##### rheum_issue


show_details_and_graphs('rheum_issue')


# category = ['Not done','No','Yes']


cat_ordinal_cols.append('rheum_issue')


##### sex_match


show_details_and_graphs('sex_match')


cat_ohe_cols.append('sex_match')


train.columns


#### race_group


show_details_and_graphs('race_group')


cat_ohe_cols.append('race_group')


#### comorbidity_score


show_details_and_graphs('comorbidity_score')


check_outlier('comorbidity_score')


num_cols.append('comorbidity_score')


##### karnofsky_score


show_details_and_graphs('karnofsky_score')


check_outlier('karnofsky_score')


num_cols.append('karnofsky_score')


#### hepatic_mild


show_details_and_graphs('hepatic_mild')


# categroy = ['Not done','No','Yes']


cat_ordinal_cols.append('hepatic_mild')


##### donor_related


show_details_and_graphs('donor_related')


show_avg_efs_time('donor_related')


cat_label_cols.append('donor_related')


#### melphalan_dose


show_details_and_graphs('melphalan_dose')


show_avg_efs_time('melphalan_dose')


cat_ohe_cols.append('melphalan_dose')


train.head(3)


##### cardiac


show_details_and_graphs('cardiac')


# category = ['Not done','No','Yes']
## Ordinal Encoding can be used


cat_ordinal_cols.append('cardiac')


##### pulm_moderate


show_details_and_graphs('pulm_moderate')


# category = ['Not done','No','Yes']


cat_ordinal_cols.append('pulm_moderate')


###### efs


show_details_and_graphs('efs')


check_outlier('efs')


###### efs_time


show_details_and_graphs('efs_time')


check_outlier('efs_time')


## Not appending efs and efs_time to any list because these are not required....


X = train.drop(columns=['efs','efs_time']) ## independent features
y = train['efs_time'] ## target feature


# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)


from sklearn.compose import ColumnTransformer


X_num_cols = [feature for feature in X.columns if X[feature].dtype!='O']
X_cat_cols = [feature for feature in X.columns if X[feature].dtype=='O']


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


from sklearn.pipeline import Pipeline


# One-Hot Encoding Pipeline
cat_ohe_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# Label Encoding Pipeline
cat_label_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('label_encoder', OrdinalEncoder()) ## applying Ordinal Encodin on these cat_label_cols 
])


cat_ordinal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal_encoder', OrdinalEncoder(categories=[
        # DRI Score
        ['N/A - disease not classifiable', 'N/A - non-malignant indication', 'N/A - pediatric',
         'Missing disease status', 'TBD cytogenetics', 'Low', 'Intermediate - TED AML case <missing cytogenetics',
         'Intermediate', 'High - TED AML case <missing cytogenetics', 'High', 'Very high'],
        
        # Features with ['No', 'Not done', 'Yes']
        ['No', 'Not done', 'Yes'],  # psych_disturb
        ['Not tested', 'TBD', 'Other', 'Normal', 'Favorable', 'Intermediate', 'Poor'],  # cyto_score
        ['No', 'Not done', 'Yes'],  # diabetes
        ['TBI +- Other, -cGy, fractionated', 'No TBI', 'TBI + Cy +- Other', 'TBI +- Other, -cGy, single',
         'TBI +- Other, >cGy', 'TBI +- Other, unknown dose', 'TBI +- Other, <=cGy', 'TBI +- Other, -cGy, unknown dose'],  # tbi_status
        ['No', 'Not done', 'Yes'],  # arrhythmia
        ['No', 'Not done', 'Yes'],  # renal_issue
        ['No', 'Not done', 'Yes'],  # pulm_severe
        ['-/-', '-/+', '+/+', '+/-'],  # cmv_status
        ['MAC', 'N/A, F(pre-TED) not submitted', 'NMA', 'TBD', 'No drugs reported', 'RIC'],  # conditioning_intensity
        ['No', 'Not done', 'Yes'],  # obesity
        ['No', 'Not done', 'Yes'],  # hepatic_severe
        ['No', 'Not done', 'Yes'],  # prior_tumor
        ['No', 'Not done', 'Yes'],  # peptic_ulcer
        ['No', 'Not done', 'Yes'],  # rheum_issue
        ['No', 'Not done', 'Yes'],  # hepatic_mild
        ['No', 'Not done', 'Yes'],  # cardiac
        ['No', 'Not done', 'Yes']   # pulm_moderate
    ]))
])


# Numerical Pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])


cat_ordinal_cols


cat_ordinal_cols.clear()


cat_ordinal_cols = [
    'dri_score',
    'psych_disturb',
    'cyto_score',
    'diabetes',
    'tbi_status',
    'arrhythmia',
    'renal_issue',
    'pulm_severe',
    'cmv_status',
    'conditioning_intensity',
    'obesity',
    'hepatic_severe',
    'prior_tumor',
    'peptic_ulcer',
    'rheum_issue',
    'hepatic_mild',
    'cardiac',
    'pulm_moderate'
]


len(cat_ordinal_cols)


# Column Transformer
transformer = ColumnTransformer(transformers=[
    ('cat_ohe', cat_ohe_pipeline, cat_ohe_cols),
    ('cat_label', cat_label_pipeline, cat_label_cols),
    ('cat_ordinal', cat_ordinal_pipeline, cat_ordinal_cols),
    ('num_pipeline', num_pipeline, num_cols),
], remainder='passthrough')  # Keeps other columns as they are


cat_label_cols


cat_ordinal_cols


len(cat_ordinal_cols)


cat_ohe_cols


num_cols


len(train.columns)


# X_train_trf = transformer.fit_transform(X_train)
# X_test_trf = transformer.transform(X_test)


# X_train_trf


# pd.DataFrame(X_train_trf)


# type(X_train_trf)


X_trf = transformer.fit_transform(X) ## using complete data


X_trf


## Model Training and Model Selection
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

import xgboost 
from xgboost import XGBRegressor

from lightgbm import LGBMRegressor


## Creating a function to evaluat model
def evaluate_model(true, predicted):
    mae=mean_absolute_error(true,predicted)
    mse=mean_squared_error(true,predicted)
    rmse=np.sqrt(mse)
    r2=r2_score(true,predicted)
    print("R2 Score:{:.4f}".format(r2))
    print("MSE:{:.4f}".format(mse))
    print("RMSE:{:.4f}".format(rmse))
    print("MAE:{:.4f}".format(mae))
    # ---------
    return 0


test


test.shape


test.isnull().sum()


col_to_drop


test.drop(columns=col_to_drop,inplace=True)


test


test.shape


test.isnull().sum()


### transforming the test 
test_trf = transformer.transform(test)


test_trf


pd.DataFrame(test_trf)


### sample submission file 


sample_submission


id_column = sample_submission['ID']


id_column


# ## Model training
# models={
#     "Linear_Regression":LinearRegression(),
#     "Lasso":Lasso(),
#     "Ridge":Ridge(),
#     "K_Neighbors_Regressor":KNeighborsRegressor()
# }

# for i in range(len(list(models))):
#     model_name = list(models.keys())[i]
#     model=list(models.values())[i]
#     model.fit(X_train_trf,y_train) ## Train Model 

#     ## Make Predictions
#     y_train_pred=model.predict(X_train_trf)
#     y_test_pred=model.predict(X_test_trf)

#     print(model_name,"=============>")
#     print("Evaluating Train Dataset")
#     evaluate_model(y_train,y_train_pred)
#     print(f"\n{'-'*50}\n")
#     print("Evaluating Test Dataset")
#     evaluate_model(y_test,y_test_pred)
#     print("="*50)
#     print("\n")

#     # ## prediction
#     # prediction = model.predict(test_trf)

#     # result = pd.DataFrame(
#     # {
#     #     'ID':id_column,
#     #     'prediction':prediction
#     # }
#     # )

#     # # Create folder if it doesn't exist
#     # folder_path = f"./{model_name}"
#     # os.makedirs(folder_path, exist_ok=True)

#     # # Save the CSV file inside the folder
#     # result.to_csv(f"{folder_path}/submission.csv", index=False)
#     # print(f"File saved as '{folder_path}/submission.csv'....")
#     # print()


# ## Model training
# models={
#     "Decision_Tree_Regressor":DecisionTreeRegressor(),
#     "Random_Forest":RandomForestRegressor(),
#     "AdaBoost":AdaBoostRegressor(),
#     "GradientBoost":GradientBoostingRegressor(),
#     "XGBRegressor":XGBRegressor(),
#     "LGBMRegressor":LGBMRegressor()
# }

# for i in range(len(list(models))):

#     model_name = list(models.keys())[i]
#     model=list(models.values())[i]
#     model.fit(X_train_trf,y_train) ## model training

#     ## Make Predictions
#     y_train_pred=model.predict(X_train_trf)
#     y_test_pred=model.predict(X_test_trf)

#     print(model_name,"=============>")
#     print("Evaluating Train Dataset")
#     evaluate_model(y_train,y_train_pred)
#     print(f"\n{'-'*50}\n")
#     print("Evaluating Test Dataset")
#     evaluate_model(y_test,y_test_pred)
#     print("="*50)
#     print("\n")

#     # ## prediction
#     # prediction = model.predict(test_trf)

#     # result = pd.DataFrame(
#     # {
#     #     'ID':id_column,
#     #     'prediction':prediction
#     # }
#     # )

#     # # Create folder if it doesn't exist
#     # folder_path = f"./{model_name}"
#     # os.makedirs(folder_path, exist_ok=True)

#     # # Save the CSV file inside the folder
#     # result.to_csv(f"{folder_path}/submission.csv", index=False)
#     # print(f"File saved as '{folder_path}/submission.csv'....")
#     # print()


## Prediction Using AdaBoost 
ada_boost = AdaBoostRegressor()

ada_boost.fit(X_trf,y) ## model training on complete data

# ## Make Predictions
# y_train_pred=ada_boost.predict(X_train_trf)
# y_test_pred=ada_boost.predict(X_test_trf)

# print("Evaluating Train Dataset")
# evaluate_model(y_train,y_train_pred)
# print(f"\n{'-'*50}\n")
# print("Evaluating Test Dataset")
# evaluate_model(y_test,y_test_pred)
# print("="*50)
# print("\n")

## prediction
prediction = ada_boost.predict(test_trf)

result = pd.DataFrame(
{
    'ID':id_column,
    'prediction':prediction
}
)

print("Model Used : AdaBoostRegressor")
result.to_csv("submission.csv", index=False)
print("File saved as submission.csv'....")


#### Used AdaBoost(better performance till now) trained on whole data 




