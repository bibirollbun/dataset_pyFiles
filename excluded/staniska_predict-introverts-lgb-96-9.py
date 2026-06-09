import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


import pandas as pd

data = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
data.head()


data.info()


data.describe()


categorical_cols = data.select_dtypes(include=['object']).columns

plt.figure(figsize=(15,5))
for i, col in enumerate(categorical_cols):
    plt.subplot(1, len(categorical_cols), i+1)
    counts = data[col].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    plt.title(col)
plt.show()


data.isnull().sum()


data.select_dtypes(include=['object'])


null_counts = data.isnull().sum()
null_counts = null_counts[null_counts > 0]
null_cols = list(null_counts.to_dict().keys())
null_cols


imputers = {}


for col in [c for c in null_cols if data[c].dtype == 'object']:
    imputer = SimpleImputer(strategy='most_frequent')
    data[[col]] = imputer.fit_transform(data[[col]])
    imputers[col] = imputer
    


for col in [c for c in null_cols if data[c].dtype != 'object']:
    imputer = SimpleImputer(strategy='mean')
    data[[col]] = imputer.fit_transform(data[[col]])
    imputers[col] = imputer


data.isnull().sum()


encoders = {}


for col in data.select_dtypes(include=['object']):
    enc = LabelEncoder()
    data[col] = enc.fit_transform(data[col])
    encoders[col] = enc


X = data.drop(['id',"Personality"],axis=1)
y = data['Personality']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


lr = LogisticRegression()
lr.fit(X_train,y_train)


lr.score(X_test,y_test)


knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train,y_train)


knn.score(X_test,y_test)


svc = SVC()
svc.fit(X_train,y_train)


svc.score(X_test,y_test)


dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


dtree.score(X_test,y_test)


rf = RandomForestClassifier()
rf.fit(X_train,y_train)


rf.score(X_test,y_test)


xgb = XGBClassifier()
xgb.fit(X_train,y_train)


xgb.score(X_test,y_test)


cbm = CatBoostClassifier(verbose=0)
cbm.fit(X_train,y_train)


cbm.score(X_test,y_test)


lgb = LGBMClassifier()
lgb.fit(X_train,y_train)


lgb.score(X_test,y_test)


model = lgb


test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


test_data.isnull().sum()


test_data.info()


for col in list(imputers.keys()):
    test_data[[col]] = imputers[col].transform(test_data[[col]])


X_test1 = test_data.drop("id",axis=1)


X_test1.info()


for col in list(encoders.keys())[:-1]:
    X_test1[col] = encoders[col].transform(X_test1[[col]])


preds = model.predict(X_test1)


prediction_dataframe = pd.DataFrame({"id":[x for x in range(18524,18524+len(preds))],
                                    "Personality":list(encoders.values())[-1].inverse_transform(preds)})


prediction_dataframe


prediction_dataframe.to_csv("submission.csv",index=False)




