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


train=pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv',dtype={'diagnosed_diabetes':int,'family_history_diabetes':bool,'hypertension_history':bool,'cardiovascular_history':bool,'diagnosed_diabetes':bool},usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
test=pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
df_submission=pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


train.sample(5)


print(train.shape,test.shape)


train[['family_history_diabetes','hypertension_history','cardiovascular_history']].nunique()


train.info()


numeric_col= train.select_dtypes(exclude='object')
categorical_col=train.select_dtypes(include=['bool','object','category'])


categorical_col


for col in categorical_col.columns:
    print(categorical_col[col].value_counts(normalize=True)*100)
    print("\n")


numeric_col['alcohol_consumption_per_week'].value_counts(normalize=True)*100


numeric_col['family_history_diabetes'].value_counts(normalize=True)*100


numeric_col['hypertension_history'].value_counts(normalize=True)*100


numeric_col['cardiovascular_history'].value_counts(normalize=True)*100


train.describe()


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


for col in numeric_col.columns:
    plt.figure(figsize=(8,4))
    sns.boxplot(y=numeric_col[col])
    plt.title(f'Box Plot of {col}')


# Now i want to build a function to detect the outliers that i have in my independent features 
def outlier_summary(data):
    summary={}
    
    for col in data.columns:
        q1=data[col].quantile(0.25)
        q3=data[col].quantile(0.75)
        iqr= q3-q1
        lower_bound=q1-1.5*iqr
        upper_bound=q3+1.5*iqr
        outlier_mask= (data[col]<lower_bound) | (data[col]>upper_bound)
        outlier_count=int(outlier_mask.sum())
        outlier_percent=round((outlier_count/len(data))*100,2)

        summary[col]={
            'Outlier_percent':outlier_percent,
            'Outlier_count':outlier_count
        }
        
        

    return summary


outlier_summary(numeric_col[[ 'age', 'alcohol_consumption_per_week',
       'physical_activity_minutes_per_week', 'diet_score',
       'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
       'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
       'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
       'triglycerides']])


numeric_col.describe()


sns.histplot(x=train['diagnosed_diabetes'])


numeric_col.columns


for col in numeric_col:
    plt.figure(figsize=(8,6))
    sns.histplot(x=numeric_col[col])
    plt.title(f'Histogram of{col}')











plt.figure(figsize=(15,10))
sns.heatmap(numeric_col.corr(method='pearson'),annot=True,square=True,fmt='.2f',vmin=-1,vmax=1,cmap='coolwarm')
plt.title("Pearson correlation heatmap")


from sklearn.feature_selection import mutual_info_classif


X=numeric_col.drop(columns=['diagnosed_diabetes'])
y=numeric_col['diagnosed_diabetes']
mi_score=mutual_info_classif(X,y,random_state=43,)
mi_score




mi={'Numeric Columns': numeric_col.drop(columns=['diagnosed_diabetes']).columns,
       'MI Score': mi_score}
mi_df=pd.DataFrame(mi)


numeric_col.corr()


numeric_col[['age','diagnosed_diabetes']].corr()


# removing one of the two features that have the same affect or impact on the target variable
# OR
# removing one of the two multicollinear feature
corr_matrix = numeric_col.corr().abs()

upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

high_corr_features = [
    col for col in upper_triangle.columns
    if any(upper_triangle[col] > 0.85)
]

high_corr_features



mi_df


# Now, the feature selection will be based upon the value of MI,
# Two feature having the same correlation but different MI value than the one with high MI value will be selected.
mi_threshold=0.005
features_to_drop= mi_df.loc[mi_df['MI Score']< mi_threshold, 'Numeric Columns'].tolist()
features_to_drop


train.drop(columns=['cardiovascular_history'],inplace=True)


train.sample(2)


categorical_col.sample(2)


for col in categorical_col.columns:
    plt.figure(figsize=(8,6))
    plt.title(f'The Count Plot for {col}')
    sns.countplot(data=categorical_col,x=categorical_col[col])
    plt.show


sns.countplot(data=categorical_col,x='ethnicity',hue='diagnosed_diabetes')


for col in categorical_col.columns:
    plt.title(f'The Affect of {col} on the diagnosed_diabetes')
    sns.countplot(data=categorical_col,x=categorical_col[col],hue='diagnosed_diabetes')
    plt.show()


from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,LabelEncoder
from sklearn.feature_selection import chi2,SelectKBest


X1=categorical_col.drop(columns='diagnosed_diabetes')
Y1=categorical_col['diagnosed_diabetes']


X1


lb=LabelEncoder()
Y1_trans=lb.fit_transform(Y1)
X1_trans=X1.copy()
for i in range (X1.shape[1]):
    X1_trans.iloc[:,i]=lb.fit_transform(X1.iloc[:,i])


chi_score,p_value=chi2(X1_trans,Y1_trans)



chi_score


chi_df=pd.DataFrame({
    "Features":X1_trans.columns,
    "Chi_scores":chi_score,
    "P Value":p_value
}).sort_values(by='Chi_scores',ascending=False)
chi_df


significant_features=chi_df[chi_df['P Value']<0.05][["Features","P Value"]]
significant_features


categorical_col.drop(columns=['gender','ethnicity','smoking_status'],inplace=True)
categorical_col.shape


numeric_col.shape


train.shape


train.drop(columns=['gender','ethnicity','smoking_status'],inplace=True)



train.shape


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report,roc_auc_score


train.shape


train['age_risk']=train['age']*train['cholesterol_total']
train['metabolic_index'] = (train['bmi'] * train['systolic_bp']) / 100

test['age_risk']=test['age']*test['cholesterol_total']
test['metabolic_index'] = (test['bmi'] * test['systolic_bp']) / 100

X=train.drop(columns=['diagnosed_diabetes'])
y=train['diagnosed_diabetes']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=43)


pd.DataFrame(train.columns)


train.head(2)


trf1=ColumnTransformer([
    ('One Hot Enocder',OneHotEncoder(),[17,18,19]),
    ("Ordinal Encoder",OrdinalEncoder(),[15,16]),
    ("Standard Scalar",StandardScaler(),[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,20,21])
    
],remainder='passthrough')
lb.fit_transform(y)


rf=RandomForestClassifier(n_jobs=-1,class_weight='balanced',criterion='gini',max_features=None)
lgbm=LGBMClassifier(n_jobs=-1,class_weight='balanced',learning_rate=0.05,n_estimators=1000,num_leaves=31,metric='auc')
xgb=XGBClassifier(iterations=100,depth=11,learning_rate=0.2,eval_metric='auc')


pipe=Pipeline([
    ('trf1',trf1),
    ('rf',rf)
])


pipe.fit(X_train,y_train)


y_pred=pipe.predict_proba(X_test)[:,1]



roc_auc_score(y_test,y_pred)


pipe3=Pipeline([
    ('trf1',trf1),
    ('lgbm',lgbm)
])
pipe3.fit(X_train,y_train)
y_pred3=pipe3.predict_proba(X_test)[:,1]

roc_auc_score(y_test,y_pred3)


train.head(4)


pipe4=Pipeline([
    ('trf1',trf1),
    ('xgb',xgb)
])
pipe4.fit(X_train,y_train)
roc_auc_score(y_test,pipe4.predict_proba(X_test)[:,1])


test.shape


X_t=test.iloc[:,:27]



X_t


test_pred=pipe3.predict_proba(X_t)[:,1]



submission=pd.DataFrame({
    'id':test['id'],
    'diagnosed_diabetes':test_pred
})



submission.to_csv('submission.csv',index=False)
df=pd.read_csv('submission.csv')
df

