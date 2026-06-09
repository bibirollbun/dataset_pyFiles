import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train_data=pd.read_csv("/kaggle/input/playground-series-s4e1/train.csv")
test_data=pd.read_csv("/kaggle/input/playground-series-s4e1/test.csv")


train_data.head()


test_data.head()



train_data.drop(["id","CustomerId","Surname","Age"],axis="columns",inplace=True)
test_data.drop(["id","CustomerId","Surname","Age"],axis="columns",inplace=True)
train_data.head()


train_data.info()


train_data.describe()


train_data.isna().sum()


train_data.shape


t=train_data.Exited.value_counts()
t


t[0]/len(train_data)*100


t[1]/len(train_data)*100


sns.set_theme(style='darkgrid',palette='Set1')
ax=sns.countplot(y="Exited",data=train_data)


def percentage_visual(c):
     p=train_data.groupby(c)['Exited'].value_counts().to_frame().rename({'count':'No of Exited'},axis=1).reset_index()
     p['Percentage of Exited']=p['No of Exited']/p['No of Exited'].sum()*100
     sns.barplot(x=c,y='Percentage of Exited',hue='Exited',data=p)


percentage_visual("Geography")


percentage_visual("Gender")


percentage_visual("Tenure")


percentage_visual("IsActiveMember")


percentage_visual("HasCrCard")


plt.figure(figsize=(9,4))
plt.title("KDE for NumOfProducts")
ax0=sns.kdeplot(train_data[train_data["Exited"]==1]["NumOfProducts"],label="Exited: Yes")
ax1=sns.kdeplot(train_data[train_data["Exited"]==0]["NumOfProducts"],label="Exited: No")


plt.figure(figsize=(9,4))
plt.title("KDE for EstimatedSalary")
ax0=sns.kdeplot(train_data[train_data["Exited"]==1]["EstimatedSalary"],label="Exited: Yes")
ax1=sns.kdeplot(train_data[train_data["Exited"]==0]["EstimatedSalary"],label="Exited: No")


X=train_data.drop(['Exited'],axis=1)
y=train_data.Exited
X


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
for c in ["Geography","Gender",]:
    encoder.fit(X[c])
    X[c]=encoder.transform(X[c])
    x_mapping=dict(zip(encoder.classes_,encoder.transform(encoder.classes_)))
    print(c,":",x_mapping)


X


from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
X["CreditScore"]=scale.fit_transform(X[["CreditScore"]])
X["Balance"]=scale.fit_transform(X[["Balance"]])
X["EstimatedSalary"]=scale.fit_transform(X[["EstimatedSalary"]])


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()



model.fit(X, y)


test_data


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
for c in ["Geography","Gender",]:
    encoder.fit(test_data[c])
    test_data[c]=encoder.transform(test_data[c])
    x_mapping=dict(zip(encoder.classes_,encoder.transform(encoder.classes_)))
    print(c,":",x_mapping)


from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
test_data["CreditScore"]=scale.fit_transform(test_data[["CreditScore"]])
test_data["Balance"]=scale.fit_transform(test_data[["Balance"]])
test_data["Balance"]=scale.fit_transform(test_data[["Balance"]])
test_data["EstimatedSalary"]=scale.fit_transform(test_data[["EstimatedSalary"]])


y_pred=model.predict(test_data)


submission = pd.read_csv('/kaggle/input/playground-series-s4e1/sample_submission.csv')
output = pd.DataFrame({'id':submission.id, 'Survived': y_pred})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")




