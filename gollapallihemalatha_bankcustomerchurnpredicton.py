import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


train=pd.read_csv("/kaggle/input/bank-customer-churn-prediction-2026/train.csv")
test=pd.read_csv("/kaggle/input/bank-customer-churn-prediction-2026/test.csv")
train.head()


train.info()


train.shape


train.describe().T.style.background_gradient(cmap='coolwarm',axis=1)


train.isnull().sum()


train.duplicated()


train['Exited'].value_counts()



X = train.drop('Exited', axis=1)
y = train['Exited']



X.select_dtypes(include='object').columns



X = pd.get_dummies(X, drop_first=True)



test=pd.read_csv("/kaggle/input/bank-customer-churn-prediction-2026/test.csv")
test = pd.get_dummies(test, drop_first=True)



X.shape,test.shape


X, test = X.align(test, join='left', axis=1, fill_value=0)



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)



from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)



from sklearn.metrics import accuracy_score

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
accuracy



test_pred = model.predict_proba(test)[:, 1]



submission = pd.read_csv('/kaggle/input/bank-customer-churn-prediction-2026/sample_submission.csv')
submission['Exited'] = test_pred



submission.to_csv('submission.csv', index=False)





