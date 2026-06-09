# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import roc_auc_score,RocCurveDisplay

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df=pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv') #importing the train set
df_test=pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv') #importing the test set


#displaying the first 5 rows
df.head()


#displaying the last 5 rows
df.tail()


#displaying the no of rows and no of columns in the train set
df.shape


#displaying the respective data types of all the features
df.dtypes


#displaying the list of column names
df.columns


#displays the number of duplicate rows
df.duplicated().sum()


#checking for null values
df.isna().sum()


#visualizing numerical features
df.hist(figsize=(15,15),bins=30)


#calculating correlation among numerical features
df.corr(numeric_only=True)


#visualizing for checking the outliers before treating them
df[['diet_score','sleep_hours_per_day','screen_time_hours_per_day','systolic_bp','waist_to_hip_ratio','hdl_cholesterol']].boxplot(figsize=(16,16),color='skyblue')
plt.title('Outlier analysis')
plt.show()


#replacing outliers with median value
cols=['diet_score','sleep_hours_per_day','screen_time_hours_per_day','systolic_bp','waist_to_hip_ratio','hdl_cholesterol']
for col in cols:
    Q3=df[col].quantile(0.75)
    Q1=df[col].quantile(0.25)
    
    IQR=Q3-Q1
    
    lower=Q1-1.5*IQR
    upper=Q3+1.5*IQR

    df[col]=np.where((df[col]<lower)|(df[col]>upper),df[col].median(),df[col])


#visualization for checking the outliers after treating them
df[cols].boxplot(figsize=(16,16),color='red')


#displaying the list of columns 
df.columns


#frequency distribution of the target variable
df['diagnosed_diabetes'].value_counts()


#visualizing numerical features after outlier treatment
df.hist(figsize=(16,16),bins=30)


#Adding categorical features to a list
categorical_features=[]

for column in df.columns:
    if df[column].dtype=='O':
        categorical_features.append(column)


#displaying categorical features again
categorical_features


#frequency distribution of gender variable
df['gender'].value_counts()


#visualizing gender with hue as target
plt.figure(figsize=(7,7))
sns.countplot(data=df,x=df['gender'],color='red',hue='diagnosed_diabetes')
plt.title('Analysis of gender feature')
plt.show()


#frequency distribution of ethnicity
df['ethnicity'].value_counts()


#visualizing ethnicity with hue as target
plt.figure(figsize=(7,7))
sns.countplot(data=df,x=df['ethnicity'],color='skyblue',hue='diagnosed_diabetes')
plt.title('Analysis of ethnicity feature')
plt.show()


#frequency distribution of education_level
df['education_level'].value_counts()


#visualizing education_level with hue as target
plt.figure(figsize=(7,7))
sns.countplot(data=df,x=df['education_level'],color='brown',hue='diagnosed_diabetes')
plt.title('Analysis of education_level feature')
plt.show()


#displaying list of categorical features again
categorical_features


#frequency distribution of income_level
df['income_level'].value_counts()


#visualizing income_level with hue as target
plt.figure(figsize=(7,7))
sns.countplot(data=df,x=df['income_level'],color='green',hue='diagnosed_diabetes')
plt.title('Analysis of income_level feature')
plt.show()


#frequency distribution of smoking_status
df['smoking_status'].value_counts()


#visualizing smoking_status with hue as target
plt.figure(figsize=(7,7))
sns.countplot(data=df,x=df['smoking_status'],hue='diagnosed_diabetes')
plt.title('Analysis of smoking_status feature')


#frequency distribution of employment_status
df['employment_status'].value_counts()


#visualizing employment_status with hue as target
plt.figure(figsize=(7,7))
sns.countplot(data=df,x=df['employment_status'],color='orange',hue='diagnosed_diabetes')
plt.title('Analysis of employment_status feature')
plt.show()


#visualizing categorical vs features pertaining to family history of illness
historical_illness=['family_history_diabetes','hypertension_history','cardiovascular_history']
hue_column='diagnosed_diabetes'

pairs=[(numerical,categorical) for numerical in historical_illness for categorical in categorical_features]

fig,axes=plt.subplots(9,2,figsize=(14,36))

axes=axes.flatten()

for ax,(numerical,categorical) in zip(axes,pairs):
    sns.barplot(x=categorical,y=numerical,data=df,ax=ax,hue=hue_column)
    ax.set_title(f'{numerical} vs {categorical}')
    ax.legend_.remove()
    
plt.tight_layout()
plt.show()



#displaying the list of columns
df.columns


#create a copy of df and store it in df_train
df_train=df.copy


#creating a new dataframe with the same columns which are available in dataframe df
df_train=pd.DataFrame(df,columns=df.columns)


#displaying first 5 rows
df_train.head()


#create a copy of df_test and store it in X_test
X_test=df_test.copy


#create a new dataframe using the same columns as in df_test
X_test=pd.DataFrame(df_test,columns=df_test.columns)


#display the first 5 rows
X_test.head()


#display the list of categorical features
categorical_features


#encoding the categorical features with LabelEncoder
le=LabelEncoder()

for colum in categorical_features:
    df_train[colum]=le.fit_transform(df_train[colum])
    X_test[colum]=le.transform(X_test[colum])


#display the first 5 rows again
df_train.head()


#display the first 5 rows of X_test again
X_test.head()


#calculating correlation among the features
correlation=df_train.corr()


#displaying the correlation
correlation


#visualizing the correlation with heatmap
plt.figure(figsize=(19,19))
sns.heatmap(correlation,annot=True,fmt='.3f',linewidths=0.2,linecolor='white',cmap='coolwarm')
plt.title('Visualization of correlation among features')
plt.tight_layout()
plt.show()


#creating X_train and y_train variables
X_train=df_train.drop('diagnosed_diabetes',axis=1)
y_train=df_train['diagnosed_diabetes']


#splitting the train set into X_tn,X_vl,y_tn,y_vl
X_tn,X_vl,y_tn,y_vl=train_test_split(X_train,y_train,test_size=0.2,random_state=42)


#creating a pipeline and param grid
pipeline = Pipeline([
    ('ss',StandardScaler()),
    ('model',LogisticRegression())
])
    
param_grid=[
    
    {
        'model':[LogisticRegression()],
        'model__C':[np.logspace(-3,2,10)],
        'model__solver':['liblinear','lbfgs']
        
    },

    {
        'model':[SVC()],
        'model__C':[np.logspace(-2,2,8)],
        'model__kernel':['rbf','sigmoid'],
        'model__gamma':['scale','auto']
        
    },

    {
        'model':[KNeighborsClassifier()],
        'model__n_neighbors':[range(1,51)],
        'model__weights':['uniform','distance']
        
    },

    {
        'model':[DecisionTreeClassifier()],
        'model__criterion':['gini','entropy'],
        'model__splitter':['best','random'],
        'model__max_depth':[10,15,20],
        'model__random_state':[42]
        
    },

    {
        'model':[RandomForestClassifier()],
        'model__n_estimators':[range(100,500,50)],
        'model__criterion':['gini','entropy'],
        'model__max_depth':[10,15],
        'model__random_state':[42]
        
    },

    {
        'model':[BernoulliNB()],
        'model__alpha':[range(1,4)]
        
    },

    {
        'model':[XGBClassifier()],
        'model__learning_rate': [np.arange(0.1,1.1,0.1)],
        'model__n_estimators':[range(100,151)],
        'model__max_depth':[10,15],
        'model__random_state':[42]
        
    },

    {
        'model':[HistGradientBoostingClassifier()],
        'model__loss':['log_loss','exponential'],
        'model__learning_rate': [np.arange(0.1,1.1,0.1)],
        'model__max_depth':[10,15],
        'model__random_state':[42]
        
    },

    {
        'model':[LGBMClassifier()],
        'model__num_leaves':[128],
        'model__learning_rate': [np.arange(0.1,1.1,0.1)],
        'model__n_estimators':[range(100,151)],
        'model__max_depth':[10,15],
        'model__random_state':[42]
        
            
        
    }

    
]
    
pipeline.fit(X_tn,y_tn)


#prediction with X_test
y_test_pred=pipeline.predict(X_test)


#displaying the prediction
y_test_pred


#hyper parameter tuning using ramdomized search cv with a pipeline
#Printing best cv roc auc score,best model and best parameters
for name in pipeline:
    random_s=RandomizedSearchCV(pipeline,param_grid,n_iter=10,cv=10,scoring='roc_auc',n_jobs=-1,random_state=42)
    random_s.fit(X_tn,y_tn)
    best_score=random_s.best_score_
    best_pipeline=random_s.best_estimator_
    best_parameters=random_s.best_params_

print('Best CV roc auc score is {:.4f}'.format(best_score))
print('Best model is {}'.format(random_s.best_params_['model']))
print('Best parameters are {}'.format(best_parameters))



#predicting the probabilities of the positive class in the validation set and printing the validation roc auc score
y_vl_prob=best_pipeline.predict_proba(X_vl)[:,1]
validation_roc_auc=roc_auc_score(y_vl,y_vl_prob)
print('Validation ROC AUC score is {:.4f}'.format(validation_roc_auc))


#visualizing the roc curve for the validation set on train.csv
plt.figure(figsize=(7,7))
RocCurveDisplay.from_estimator(best_pipeline,X_vl,y_vl)
plt.plot([0,1],[0,1],'k--',color='blue')
plt.title('ROC Curve(Validation)')
plt.show()


#predicting the probabilities of the positive class using the test set with the best pipeline
y_pred_proba=best_pipeline.predict_proba(X_test)[:,1]


#first 10 probability predictions of the positive class in the test set
y_pred_proba[0:10]


#creating a dataframe named submission
submission=pd.DataFrame({
    'id':X_test['id'],
    'diagnosed_diabetes':y_pred_proba
})


submission


submission.to_csv('submission.csv',index=False)
print('Submission saved successfully')




