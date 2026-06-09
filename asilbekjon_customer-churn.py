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


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier


df=pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/train.csv')
df.head()


test=pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/test.csv')
test.drop(["Surname",'CustomerId'],axis=1,inplace=True)


##data cleaning
df.drop(['Surname','CustomerId','id'],axis=1,inplace=True)

#NaN values
df.isnull().sum()


df.describe()


df.info()


# categorical columns
fig, axes = plt.subplots(1,2 ,figsize=(15,5))

sns.countplot(x='Gender', hue='Exited', data=df, ax=axes[0])
axes[1].set_title("by gender")

sns.countplot(x='Geography', hue='Exited', data=df, ax=axes[1])
axes[0].set_title("by country")

plt.show()


fig, axes = plt.subplots(2, 3, figsize=(25, 10))



#CreditScore vs Exited
sns.histplot(x="CreditScore", data=df, bins=20, hue="Exited", ax=axes[0, 0])
axes[0, 0].set_title("Credit Score vs Exited")
axes[0, 0].set_xlabel("Credit Score")
axes[0, 0].set_ylabel("Customer")
axes[0, 0].legend(labels=["Churned (1)", "Not Churned (0)"])

# Age vs Exited
sns.histplot(x="Age", data=df, bins=20, hue="Exited", ax=axes[0, 1])
axes[0, 1].set_title("Customer's age")
axes[0, 1].set_xlabel("Age")
axes[0, 1].set_ylabel("Customer")
axes[0, 1].legend(labels=["Churned (1)", "Not Churned (0)"])

#Tenure vs Exited
sns.histplot(x="Tenure", data=df, bins=20, hue="Exited", ax=axes[1, 0])
axes[1, 0].set_title("Tenure vs Exited")
axes[1, 0].set_xlabel("Tenure")
axes[1, 0].set_ylabel("Customer")
axes[1, 0].legend(labels=["Churned (1)", "Not Churned (0)"])

#Balance vs Exited
sns.histplot(x="Balance", data=df, bins=20, hue="Exited",  ax=axes[1, 1])
axes[1, 1].set_title("Customer's balance")
axes[1, 1].set_xlabel("Balance")
axes[1, 1].set_ylabel("Customer")
axes[1, 1].legend(labels=["Churned (1)", "Not Churned (0)"])

#Estimated Salary vs Exited
sns.histplot(x="EstimatedSalary", data=df, bins=20, hue="Exited",  ax=axes[1, 2])
axes[1, 2].set_title("Customer's Salary")
axes[1, 2].set_xlabel("Salary")
axes[1, 2].set_ylabel("Customer")
axes[1, 2].legend(labels=["Churned (1)", "Not Churned (0)"])

#Number of products vs Exited
sns.histplot(x="NumOfProducts", data=df, bins=20, hue="Exited",  ax=axes[0, 2])
axes[0, 2].set_title("Number of products")
axes[0, 2].set_xlabel("Products")
axes[0, 2].set_ylabel("Customer")
axes[0, 2].legend(labels=["Churned (1)", "Not Churned (0)"])


import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*use_inf_as_na.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*length-1 tuple.*")




df.corrwith(df['Exited'], numeric_only=True).abs().sort_values(ascending=False)


df


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

categorical = ['Geography', 'Gender']
numerical = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical),
    ('cat', cat_pipeline, categorical)
])



X=df.drop('Exited',axis=1)
y=df['Exited']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=42,stratify=y)


X_train_prepared=preprocessor.fit_transform(X_train)
X_test_prepared=preprocessor.transform(X_test)


GB=GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
GB.fit(X_train_prepared, y_train)
y_pred= GB.predict(X_test_prepared)
y_proba = GB.predict_proba(X_test_prepared)[:, 1]
print('Model accuracy:',accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


conf_mat=confusion_matrix(y_test,y_pred)
sns.heatmap(conf_mat,annot=True,fmt='g')
plt.show()


fpr, tpr, thresholds = metrics.roc_curve(y_test, y_proba)
roc_auc = metrics.auc(fpr, tpr)

display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='GradientBoosting')
display.plot()
plt.title("AUC-ROC Curve (GB)")
plt.show()


test_id=test.id


test.drop('id',axis=1)


#test predict
test_prepared=preprocessor.transform(test)
test_proba=GB.predict_proba(test_prepared)
test_positive_proba = test_proba[:, 1]  
submission = pd.DataFrame({'id': test_id, 'Exited': test_positive_proba})
submission.to_csv('submission.csv', index=False)


submission

