import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler,OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


pd.options.display.max_columns=1000
pd.options.display.max_rows=None


df=pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df.head()


#drop duplicates (before dropping id)
df.drop_duplicates(keep='first',inplace=True)

#drop id
df.drop(columns='id',inplace=True)




#number of columns
print(len(df.columns))

#number of data points
print(len(df))


df.describe(include='all')


df.groupby('diagnosed_diabetes').describe(include='all').T


df.info()


for col in df.columns:
    print(f"Unique values in column: {col} are {np.sort(df[col].unique())}")
    print('='*150)


for col in df.select_dtypes('number'):
    df[col]=pd.to_numeric(df[col],downcast='integer')
    if df[col].dtype=='float':
        df[col]=pd.to_numeric(df[col],downcast='float')


df.info() #57% reduction in memory  usage


df.isna().sum() #no missings at all


diabetes_values=df['diagnosed_diabetes'].value_counts()
diabetes_values


diabetes_labels=['Diabetes','Non-diabetes']
plt.pie(diabetes_values,labels=diabetes_labels,autopct='%1.1f%%',explode=[0,0])
plt.ylabel('')
plt.show()


categorical_cols=[col for col in df.columns if df[col].dtype=='object']
for col in categorical_cols:
    counts=df.groupby([col,'diagnosed_diabetes']).size().unstack(fill_value=0)
    percentages=counts.div(counts.sum(axis=1),axis=0)
    percentages.plot(
    
        kind='bar',
        stacked=True
    )
    plt.xlabel(col)
    plt.ylabel('diagnosed_diabetes')
    plt.title(f"Diabetes Distribution per {col} categories")
    plt.show()
    print('='*100)



categorical_cols


def train_valid_test(X,y,valid,test):
    if not 0<valid<1 or not 0<test<1:#valid and test are the percentages of validation and test 
        raise ValueError('valid and test should be between 0 and 1')
    #first hold out the test set    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test,random_state=42, stratify=y)
    
    #then split to train and validation
    X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=(valid/(1-test)),random_state=42,stratify=y_train)
    
    return X_train,X_val,X_test,y_train,y_val,y_test


X=df.loc[:,df.columns!='diagnosed_diabetes']
y=df['diagnosed_diabetes']


X_train,X_val,X_test,y_train,y_val,y_test=train_valid_test(X,y,0.2,0.2)


numeric_features=[col for col in  X_train.columns if X_train[col].dtype!='object']
numeric_transformer=Pipeline(
    steps=[('scaler', StandardScaler())]
)

OHE_features=['gender','ethnicity','employment_status']
OHE_transformer=Pipeline(steps=[('OHE',OneHotEncoder(drop='first'))])

Ordinal_features=['smoking_status','income_level','education_level']
smoking_cat= ['Current', 'Former' ,'Never']
income_cat=['High','Upper-Middle','Middle','Lower-Middle','Low']
education_cat=['Postgraduate','Graduate','Highschool','No formal']
ordinal_encoder=OrdinalEncoder(categories=[smoking_cat,income_cat,education_cat])




column_transformer=ColumnTransformer(

    transformers=[
        ('num',numeric_transformer,numeric_features),
        ('OHE',OHE_transformer,OHE_features),
        ('ordinal',ordinal_encoder,Ordinal_features)
    ]
)


clf=Pipeline(
    steps=[
        ('preprocessor',column_transformer),
        ('classifier',RandomForestClassifier(random_state=42))
    ]
)


X_val_transformed = column_transformer.fit_transform(X_val)
X_test_transformed = column_transformer.fit_transform(X_test)




clf = Pipeline(
    steps=[
        ('preprocessor', column_transformer),
        ('classifier', XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='auc',
            random_state=42,
            early_stopping_rounds=10
        ))
    ]
)


clf.fit(X_train,y_train,
    classifier__eval_set=[(X_val_transformed, y_val)])


y_proba_val = clf.predict_proba(X_val)[:, 1]

y_proba_test = clf.predict_proba(X_test)[:, 1]



auc = roc_auc_score(y_val, y_proba_val)
print(f"AUC score over validation: {auc:.4f}")

auc = roc_auc_score(y_test, y_proba_test)
print(f"AUC score over test: {auc:.4f}")


kaggle_test=pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


submission=kaggle_test.drop(columns=kaggle_test.columns.difference(['id']))


submission['diagnosed_diabetes']=clf.predict_proba(kaggle_test.drop(columns='id'))[:, 1]


submission.to_csv('submission.csv',index=False)










