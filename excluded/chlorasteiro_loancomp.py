import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix,classification_report, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve,RandomizedSearchCV, GridSearchCV,train_test_split 



train=pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
df=train.copy()
df = df.drop('id',axis=1)


df.dtypes.value_counts()


df.isnull().sum()


df.head()


df['loan_paid_back'].value_counts(normalize=True)


numerical_col=['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']



sns.heatmap(df[numerical_col + ['loan_paid_back']].corr(),annot=True,fmt='.2f', cmap='Blues')



for col in numerical_col:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
    outlier_pct = (outliers / len(df)) * 100
    print(f"{col:25} {outliers:7,} outliers ({outlier_pct:5.2f}%)")




categorical_col = ['gender', 'marital_status', 'education_level', 
                   'employment_status', 'loan_purpose', 'grade_subgrade']


for col in categorical_col:
    crosstab = pd.crosstab(df[col], df['loan_paid_back'], normalize='index')
    print(f"\n{col}:")
    print(crosstab.round(2))
    





def encode_grade(df):
    
    df['grade']=df['grade_subgrade'].str[0]
    df=df.drop('grade_subgrade',axis=1)
    grade_map={
    'F':1,
    'E':2,
    'D':3,
    'C':4,
    'B':5, 
    'A':6
    }
    df['grade']=df['grade'].map(grade_map)
    return df

def encode_education(df):
    
    education_map={
    'Other':1,
    'High School':2,
    "Bachelor's":3,
    "Master's":4,
    "PhD":5
    }
    df['education_level']=df['education_level'].map(education_map)
    return df

def drop_useless_columns(df):
    
    df=df.drop(['marital_status', 'gender'],axis=1)
    
    return df

def encode_categorical(df):
    
    df=pd.get_dummies(df,columns=['employment_status','loan_purpose'],drop_first=True, dtype=int)
    return df

def preprocessing(df):
    df=df.copy()
    df= encode_grade(df)
    df=encode_education(df)
    df=encode_categorical(df)
    df=drop_useless_columns(df)

    X= df.drop('loan_paid_back',axis=1)
    y=df['loan_paid_back']

    print(y.value_counts())

    return X, y

X,y = preprocessing(df)


X_train, X_val, y_train, y_val  = train_test_split(X,y, test_size=0.2, random_state=42,stratify=y)


X_train.head()


X_train.describe()


model = CatBoostClassifier(
                iterations=1500,
                learning_rate=0.15,
                depth=4,
                class_weights = {0:3, 1:1} ,  
                eval_metric='AUC',
                l2_leaf_reg=9,
                random_seed=42,
                loss_function='Logloss',
                verbose=100
                    
        )
    
model.fit(X_train, y_train)



y_pred = model.predict(X_val)
y_pred_proba=model.predict_proba(X_val)[:,1]
score = roc_auc_score(y_val, y_pred_proba)
cm=confusion_matrix(y_val,y_pred)
importance = model.get_feature_importance()
importance_df= pd.DataFrame({
    'Feature': X_train.columns, 
    'Importance':importance
}).sort_values('Importance', ascending=False
)

print(classification_report(y_val, y_pred))
print(score)
print(cm)
print(importance_df.to_string(index=False))

    



def preprocessing_test(df):
    df = df.copy()
    df = df.drop('id', axis=1)
    df = encode_grade(df)
    df = encode_education(df)
    df = encode_categorical(df)
    df = drop_useless_columns(df)
    return df


test_ids = test['id']
X_test = preprocessing_test(test)

X_full, y_full = preprocessing(df)

model_final=  CatBoostClassifier(
                iterations=1500,
                learning_rate=0.15,
                depth=4,
                class_weights = {0:3, 1:1} ,  
                eval_metric='AUC',
                l2_leaf_reg=9,
                random_seed=42,
                loss_function='Logloss',
                verbose=100
                    
        )

model_final.fit(X_full, y_full)


y_pred_proba = model_final.predict_proba(X_test)[:, 1]


submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': y_pred_proba
})

submission.to_csv('/kaggle/working/submission.csv', index=False)





