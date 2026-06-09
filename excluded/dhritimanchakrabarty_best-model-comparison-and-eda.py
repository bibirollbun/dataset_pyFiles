

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
import matplotlib.pyplot as plt
import seaborn as sns



train=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sub=pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
print(train.columns)


print(f"Train shape is {train.shape} and test shape is {test.shape} and subs is {sub.shape}")


train.head()


(train.isna().sum()/len(train))*100



missing_percent = (train.isna().sum() / len(train)) * 100
missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)


missing_df = missing_percent.reset_index()
missing_df.columns = ['Feature', 'MissingPercent']


plt.figure(figsize=(12, 6))
sns.barplot(data=missing_df, x='Feature', y='MissingPercent')


train.describe()


train.dtypes


train = train.drop('id', axis=1)
sns.heatmap(train.corr(numeric_only=True),annot=True,cmap='coolwarm')


y=train['Personality']
train.drop(['Personality'],inplace=True,axis=1)


def preprocess(df):
    categorical_cols=df.select_dtypes(include=['object','category']).columns.tolist()
    numerical_cols=df.select_dtypes(include=['float','int']).columns.tolist()
    for i in categorical_cols:
        df[i]=df[i].fillna(df[i].mode()[0])
    for i in numerical_cols:
        df[i]=df[i].fillna(df[i].mean())
    df=pd.get_dummies(df,drop_first=True)
    return df


train=preprocess(train)



test = test.drop('id', axis=1)



test=preprocess(test)



y_encoded = y.map({'Introvert': 0, 'Extrovert': 1})


y_encoded


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(train,y_encoded,test_size=0.15)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier



model_list = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "Decision Trees Classification": DecisionTreeClassifier(random_state=42),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
    "Support Vector Machines": SVC(random_state=42, probability=True),
    "Neural Network Binary Classification": MLPClassifier(random_state=42, max_iter=1000),
    "XG Boost": XGBClassifier(random_state=42, eval_metric='logloss'),
    "CatBoost Classifier": CatBoostClassifier(random_state=42, verbose=False),
    "LightGBM Classifier": LGBMClassifier(random_state=42, verbose=-1)
}


from sklearn.metrics import accuracy_score
val=0
best_model_name = "" 
for i,model in model_list.items():
    model.fit(X_train,y_train)
    y_preds=model.predict(X_test)
    score=accuracy_score(y_preds,y_test)*100
    if(score>val):
        val=score
        best_model_name=i
    print(f"The score of {i} is {score}")
    
print(f"\nBest model: {best_model_name} with accuracy: {val}")


model=GradientBoostingClassifier(random_state=42)
model.fit(X_train,y_train)


new_col=model.predict(test)



prediction_labels = pd.Series(new_col).map({0: 'Introvert', 1: 'Extrovert'})


prediction_labels


output = pd.DataFrame({'id': sub.id, 'Personality': prediction_labels})
output.to_csv("submission.csv",index=False)
print("Done")


output

