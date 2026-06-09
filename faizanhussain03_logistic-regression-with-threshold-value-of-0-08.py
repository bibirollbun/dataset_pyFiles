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


train=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train=train.drop(columns='id')
train.describe()


categorical_columns=[col for col in train.columns if train[col].dtypes=='object']
train[categorical_columns].isnull().sum()


from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder(sparse=False)
cat_encoded_column=encoder.fit_transform(train[categorical_columns])
cat_encoded_colname=encoder.get_feature_names_out(categorical_columns)
cat_encoded_column=pd.DataFrame(cat_encoded_column,columns=cat_encoded_colname)
cat_encoded_column


train=pd.concat([train.drop(columns=categorical_columns),cat_encoded_column],axis=1)
train


train


train.dtypes=='int64'
# as the first 8 columns are numerical but y is output so will take only first 7 columns
numerical_columns=list(train.columns[0:7])
print("Numerical col:",(numerical_columns))
print("Null values in \n",train[numerical_columns].isnull().sum())



from sklearn.preprocessing import StandardScaler
scaling=StandardScaler()
train[numerical_columns]=scaling.fit_transform(train[numerical_columns])


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(train.drop(columns=['y']),train.y,test_size=0.2,random_state=1)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics



def evaluate_model(model,X_test,y_test):
    y_pred=model.predict(X_test)
    
    acc=metrics.accuracy_score(y_test,y_pred)
    cm=metrics.confusion_matrix(y_test,y_pred)
    recall=metrics.recall_score(y_pred,y_test)
    y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    
    print("Accuracy",acc)
    print("confusion Matrix",cm)
    print("Recall",recall)
    print("auc",auc)
    print("best_threshold",threshold[np.argmax(tpr-fpr)])


lr=LogisticRegression()
lr.fit(X_train,y_train)


evaluate_model(lr,X_test,y_test)


# Now lets transform the test data for prediction and training on the whole train data
test_cat_col=encoder.transform(test[categorical_columns])
test_cat_col=pd.DataFrame(test_cat_col,columns=cat_encoded_colname)
test=pd.concat([test.drop(columns=categorical_columns),test_cat_col],axis=1)
test[numerical_columns]=scaling.transform(test[numerical_columns])


lr_model=LogisticRegression()
lr_model.fit(train.drop(columns=['y']),train.y)
# taking the threshold value as 0.08 instead of 0.5
lr_new_pred=[]

for i in lr.predict_proba(test.drop(columns=['id'])):
    if(i[1]>0.08):
        lr_new_pred.append(1)
    else:
        lr_new_pred.append(0)


sample_sub=pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
submission = pd.DataFrame({
    "id": sample_sub.id, 
    "y": lr_new_pred})
submission.to_csv("submission.csv", index=False)
print("Submission saved as submission.csv")


submission




