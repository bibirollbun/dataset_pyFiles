# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import seaborn as sns
import matplotlib.pyplot as plt


train_data = pd.read_csv("/kaggle/input/playground-series-s4e10/train.csv")
test_data  = pd.read_csv("/kaggle/input/playground-series-s4e10/test.csv")


train_data.head()


train_data.info()


len(train_data["id"].unique())


train_data.describe()


categorical_features = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
numerical_features = ["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]


warnings.simplefilter(action='ignore', category=FutureWarning)
sns.pairplot(data=train_data.iloc[:,1:], hue="loan_status")
plt.show()


sns.countplot(data=train_data, x="loan_status")
plt.show()


unbalance_rate = len(train_data[train_data["loan_status"]==0]) / len(train_data[train_data["loan_status"]==1])
unbalance_rate


minor_class_rate = len(train_data[train_data["loan_status"]==1]) / len(train_data)
major_class_rate = len(train_data[train_data["loan_status"]==0])  / len(train_data)
print(f'minor class rate = {minor_class_rate}')
print(f'major class rate = {major_class_rate}')



fig, axes = plt.subplots(nrows=len(numerical_features)//2, ncols=len(numerical_features)//2, figsize=(15, 12))
axes = axes.flatten()


for i, feature in enumerate(numerical_features):
    sns.histplot(data=train_data, x=feature, hue="loan_status", bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {feature}")
    
plt.tight_layout()
plt.show()


train_data["loan_intent"].unique()


fig, axes = plt.subplots(nrows=len(categorical_features)//2, ncols=len(categorical_features)//2, figsize=(15, 12))
axes = axes.flatten()


for i, feature in enumerate(categorical_features):
    if feature == "loan_grade":
        sns.countplot(data=train_data, x=feature, hue="loan_status",ax=axes[i], order=["A", "B", "C", "D", "E", "F", "G"])
    else:
         sns.countplot(data=train_data, x=feature, hue="loan_status",ax=axes[i])
    axes[i].set_title(f"Distribution of {feature}")
    
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(nrows=len(categorical_features)//2, ncols=len(categorical_features)//2, figsize=(15, 12))
axes = axes.flatten()
order_ = train_data.index

for i, feature in enumerate(categorical_features):
    if feature == "loan_grade":
        sns.boxplot(data=train_data, y="loan_amnt" ,x=feature, hue="loan_status",ax=axes[i], order=["A", "B", "C", "D", "E", "F", "G"])
    else:
        sns.boxplot(data=train_data, y="loan_amnt" ,x=feature, hue="loan_status",ax=axes[i])
    axes[i].set_title(f"Distribution of {feature}")
    
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 16))
axes = axes.flatten()
df = pd.concat([train_data[numerical_features],train_data["loan_status"]], axis=1)
sns.heatmap(train_data[numerical_features].corr(), annot=True, ax=axes[0])
sns.heatmap(df.corr()[["loan_status"]].sort_values(by="loan_status", ascending=False), annot=True, ax=axes[1])
plt.tight_layout()
plt.show()



from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder


x_train, y_train = train_data[numerical_features], train_data.iloc[:,-1]
feat_labels = x_train.columns
print(x_train.info())
print(y_train.head())


forest = RandomForestClassifier(n_estimators=1000, random_state=1, n_jobs=-1)
forest.fit(x_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]


for f in range(x_train.shape[1]):
    print("%2d) %-*s %g" % (f+1, 30, feat_labels[indices[f]],
                           importances[indices[f]]))


plt.title('Numerical Feature Importance')
plt.bar(range(x_train.shape[1]), importances[indices], align='center')
plt.xticks(range(x_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.tight_layout()
plt.show()


X = train_data[categorical_features[1:]]
categories = [rows.unique().tolist() for col,rows in train_data[categorical_features[1:]].items()]
# ohe = OneHotEncoder(drop='first').fit(X)
ohe = OneHotEncoder().fit(X)



X2 = ohe.transform(X)
X2.toarray()


df = pd.DataFrame(X2.toarray())
categories2 = [elemento for sublista in ohe.categories_ for elemento in sublista]
df.columns = categories2
df


pd.DataFrame(ohe.inverse_transform(X2))


ohe.categories_


forest = RandomForestClassifier(n_estimators=1000, random_state=1, n_jobs=-1)
forest.fit(df, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]


for f in range(df.shape[1]):
    print("%2d) %-*s %g" % (f+1, 30, categories2[indices[f]],
                           importances[indices[f]]))


plt.title('Categorical Feature Importance')
plt.bar(range(df.shape[1]), importances[indices], align='center')
plt.xticks(range(df.shape[1]), df.columns[indices], rotation=90)
plt.xlim([-1, df.shape[1]])
plt.tight_layout()
plt.show()


dfx = pd.concat([x_train, df], axis=1)
dfx.info()


forest = RandomForestClassifier(n_estimators=1000, random_state=1, n_jobs=-1)
forest.fit(dfx, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]


for f in range(dfx.shape[1]):
    print("%2d) %-*s %g" % (f+1, 30, dfx.columns[indices[f]],
                           importances[indices[f]]))


plt.title('Feature Importance')
plt.bar(range(dfx.shape[1]), importances[indices], align='center')
plt.xticks(range(dfx.shape[1]), dfx.columns[indices], rotation=90)
plt.xlim([-1, dfx.shape[1]])
plt.tight_layout()
plt.show()


sns.pairplot(data=pd.concat([dfx, y_train], axis=1), hue="loan_status")
plt.show()




