import pandas as pd
import numpy as np


train=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train.head(3)


train.info()


train.describe()


train.isna().sum()





test.head(3)


test.info()


test.describe()


test.isna().sum()


cat_cols=train.select_dtypes(include=['object']).columns
num_cols=train.select_dtypes(include=['int']).columns

print(f'Total Categorical Columns {len(cat_cols)}')
print(f'Total Numerical Columns {len(num_cols)}')


test_id=test['id']


X=train.drop(['y','month','pdays'],axis=1)
y=train['y']
cat_cols=cat_cols.drop('month')
test=test.drop(['month','pdays'],axis=1)


X=pd.concat([X,pd.get_dummies(X[cat_cols],drop_first=True)],axis=1)
X.drop(cat_cols,inplace=True,axis=1)


test=pd.concat([test,pd.get_dummies(test[cat_cols],drop_first=True)],axis=1)
test.drop(cat_cols,axis=1,inplace=True)


from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


cat=CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    verbose=100,
    random_state=42
)



cat.fit(X_train,y_train)


y_pred=cat.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


proba=cat.predict_proba(test)


submission=pd.DataFrame({
    'id':test_id,
    'probability':proba[:,1]
})
submission.to_csv('submission.csv', index=False)

