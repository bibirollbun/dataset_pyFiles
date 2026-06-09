import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv("/kaggle/input/playground-series-s4e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e2/test.csv")


train


len(train['NObeyesdad'].values)


train['NObeyesdad'].unique()


train['MTRANS'].unique()


import seaborn as sns
sns.countplot(x='NObeyesdad',data = train,dodge = 1)


train['Gender'] = train['Gender'].map({'Male':0,'Female':1})


sns.barplot(x = 'NObeyesdad', y = 'Gender', data = train)


sns.barplot(x = 'CAEC', y = 'Gender', data = train)


sns.barplot(x = 'CALC', y = 'Gender', data = train)


train['family_history_with_overweight'] = train['family_history_with_overweight'].map({'no':0,'yes':1})


train['FAVC'] = train['FAVC'].map({'no':0,'yes':1})


train['CAEC'] = train['CAEC'].map({'no':0,'Sometimes':1,'Frequently':2,'Always':3})


train['SMOKE'] = train['SMOKE'].map({'no':0,'yes':1})


train['SCC'] = train['SCC'].map({'no':0,'yes':1})
train['CALC'] = train['CALC'].map({'no':0,'Sometimes':1,'Frequently':2})


train['MTRANS'] = train['MTRANS'].map({'Public_Transportation':0, 'Automobile':1, 'Walking':2, 'Motorbike':3,'Bike':4})


train['NObeyesdad'] = train['NObeyesdad'].map({'Overweight_Level_II':0, 'Normal_Weight':1, 'Insufficient_Weight':2,
       'Obesity_Type_III':3, 'Obesity_Type_II':4, 'Overweight_Level_I':5,
       'Obesity_Type_I':6})


test['family_history_with_overweight'] = test['family_history_with_overweight'].map({'no':0,'yes':1})
test['FAVC'] = test['FAVC'].map({'no':0,'yes':1})
test['Gender'] = test['Gender'].map({'Male':0,'Female':1})
test['SCC'] = test['SCC'].map({'no':0,'yes':1})
test['CALC'] = test['CALC'].map({'no':0,'Sometimes':1,'Frequently':2})
test['SMOKE'] = test['SMOKE'].map({'no':0,'yes':1})
test['CAEC'] = test['CAEC'].map({'no':0,'Sometimes':1,'Frequently':2,'Always':3})
test['MTRANS'] = test['MTRANS'].map({'Public_Transportation':0, 'Automobile':1, 'Walking':2, 'Motorbike':3,'Bike':4})



train


test


Features = ['Gender','Age','Height','Weight','family_history_with_overweight','FAVC','FCVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS']


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


x_t = train[Features]
y_t = train['NObeyesdad']


x_train,x_val,y_train,y_val = train_test_split(x_t, y_t, test_size = 0.2,random_state = 42)


imputer =  SimpleImputer(strategy = 'median')


model = Pipeline([
 ("imputer",imputer),
 ("classifier",RandomForestClassifier())
])


# model.fit(x_train,y_train)
model.fit(train[Features],train['NObeyesdad']) #full train set might have better perf


# from sklearn.metrics import classification_report
# y_pre = model.predict(x_val)
# print(classification_report(y_pre,y_val))


sub = model.predict(test[Features])



submisson = pd.DataFrame({"id":test["id"],"NObeyesdad":sub})


submisson["NObeyesdad"] = submisson["NObeyesdad"].map({0:'Overweight_Level_II', 1:'Normal_Weight', 2:'Insufficient_Weight',
       3:'Obesity_Type_III', 4:'Obesity_Type_II', 5:'Overweight_Level_I',
       6:'Obesity_Type_I'})


submisson.to_csv("submisson.csv",index = False)

