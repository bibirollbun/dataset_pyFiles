import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


df=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv') #Load the Dataset


df.head()


df.info()


df.isna().sum() #Counts Null values for each column


df['job'].unique()


df['marital'].unique()


df['education'].unique()


df['default'].unique()


df['housing'].unique()


df['loan'].unique()


df['contact'].unique()


df['poutcome'].unique()


df['y'].unique()


Cat=df[df['y']==1].groupby(['job','marital','education']).count()


Cat['y'].head(15) #shows first 15 entries of the table


sns.countplot(y=df['job'],hue=df['y']) #Shows the count of Prediction Variable for each Job Type


sns.countplot(y=df['marital'],hue=df['y']) #Shows the count of Prediction Variable for each Marital Type


sns.countplot(y=df['education'],hue=df['y']) #Shows the count of Prediction Variable for each Education Type


job_mapping={'technician':11,'blue-collar':1,'student':2,'admin.':3,'management':4,
       'entrepreneur':5, 'self-employed':6, 'unknown':0, 'services':7, 'retired':8,
       'housemaid':9, 'unemployed':10}
marital_mapping={'married':1, 'single':0, 'divorced':2}
education_mapping={'secondary':2, 'primary':1, 'tertiary':3, 'unknown':0}
default_mapping={'no':0, 'yes':1}
housing_mapping={'no':0, 'yes':1}
loan_mapping={'no':0, 'yes':1}
contact_mapping={'cellular':1, 'unknown':0, 'telephone':2}
poutcome_mapping={'unknown':0, 'other':3, 'failure':2, 'success':1}
month_mapping={'aug':8,'jun':6,'may':5,'feb':2,'apr':4,'nov':11,'jul':7,'jan':1,'oct':10,
       'mar':3, 'sep':9, 'dec':12}
df['job']=df['job'].map(job_mapping)
df['marital']=df['marital'].map(marital_mapping)
df['education']=df['education'].map(education_mapping)
df['default']=df['default'].map(default_mapping)
df['housing']=df['housing'].map(housing_mapping)
df['loan']=df['loan'].map(loan_mapping)
df['contact']=df['contact'].map(contact_mapping)
df['poutcome']=df['poutcome'].map(poutcome_mapping)
df['month']=df['month'].map(month_mapping)


Q1=df[['job','marital','education','default','housing','loan','contact','poutcome','y']]
sns.heatmap(Q1.corr(),annot=True)


Q2=df[['duration','pdays','previous','balance','age','month','campaign','y']]
sns.heatmap(Q2.corr(),annot=True)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


Y=df['y'] #defining the Prediction Variable
X=df[['duration','pdays','previous','contact','loan','default']] #Defining the features for Model Testing based on Correlation Values 


X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=2529)



model = LogisticRegression()
model.fit(X_train, y_train)
y_pred_lr=model.predict(X_test)



classification_report(y_test,y_pred_lr)


confusion_matrix(y_test,y_pred_lr)


accuracy_score(y_test,y_pred_lr)*100


dt_model = DecisionTreeClassifier(criterion='gini',max_depth=10, min_samples_leaf=2, min_samples_split=10,random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt=dt_model.predict(X_test)


classification_report(y_test,y_pred_dt)


confusion_matrix(y_test,y_pred_dt)


accuracy_score(y_test,y_pred_dt)*100


xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_estimators=200,
    learning_rate=0.2
)
xgb_model.fit(X_train, y_train)
y_pred_xgb=xgb_model.predict(X_test)


classification_report(y_test,y_pred_xgb)


confusion_matrix(y_test,y_pred_xgb)


accuracy_score(y_test,y_pred_xgb)*100


cat_classifier = CatBoostClassifier(
    iterations=200,   
    learning_rate=0.2,
    loss_function='Logloss',
    eval_metric='Accuracy',
    random_seed=42,
    verbose=False
)
cat_classifier.fit(X_train, y_train)
y_pred_cat = cat_classifier.predict(X_test)
y_pred_proba_cat = cat_classifier.predict_proba(X_test)[:, 1]


classification_report(y_test,y_pred_cat)


confusion_matrix(y_test,y_pred_cat)


accuracy_score(y_test, y_pred_cat)*100




