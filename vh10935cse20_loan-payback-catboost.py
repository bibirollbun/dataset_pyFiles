import pandas as pd
import numpy as np


train=pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


train.head(3)


train.shape


train.dtypes


train.info()


train['gender'].unique()


train.isna().sum()


test.head(3)


test.shape


test.info()


test.isna().sum()


test.describe()


X=train.drop(columns=['id','loan_paid_back'],axis=1)
y=train['loan_paid_back']


cat_features=['education_level','employment_status','loan_purpose','grade_subgrade' ]


def enc(df):
    df['gender_enc']=df['gender'].map({'Female':'1','Male':'0','Other':'2'})
    df['marital_status_enc']=df['marital_status'].map({'Single':'0','Married':'1','Divorced':'2','Widowed':'3'})
    df.drop(columns=['gender', 'marital_status'], inplace=True)
    return df


test_id=test['id']
test_features=test.drop(columns='id',axis=1)


enc_X = enc(X.copy())
enc_test = enc(test_features.copy())


#Class Imbalance
count_0, count_1 = train['loan_paid_back'].value_counts()
scale_pos_weight = count_1 / count_0


from sklearn.model_selection import train_test_split 
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score


X_train,X_test,y_train,y_test=train_test_split(enc_X,y,test_size=0.2,random_state=42)


params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 7,
    'l2_leaf_reg': 3.0,
    'random_state': 42,
    'eval_metric': 'AUC',
    'scale_pos_weight':scale_pos_weight,
    'verbose': 100, # Set to 100 to see training progress
    'early_stopping_rounds': 50
}
model = CatBoostClassifier(
    **params, 
    cat_features=cat_features
)


model.fit(
    X_train, y_train, 
    eval_set=(X_test, y_test), 
    cat_features=cat_features
)


y_prob = model.predict_proba(X_test)[:, 1]


roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC AUC Score on Validation Set: {roc_auc:.4f}")


final=model.predict_proba(enc_test)[:,1]


submission=pd.DataFrame({'id':test_id,'loan_paid_back':final})
submission.to_csv('submission.csv',index=False)


submission.head(3)

