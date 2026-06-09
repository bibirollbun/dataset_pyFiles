import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  confusion_matrix, accuracy_score,recall_score,f1_score,precision_score
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')
average_type='macro'
import joblib


df=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
df.head()


df.info()


test=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')



df.drop(columns=['id'], inplace=True)




df.describe()


df.columns


df['Fertilizer Name'].value_counts()


num_col=df.select_dtypes(include='number')


num_col


corr=num_col.corr()


corr


sns.heatmap(corr,cmap='YlGnBu',annot=True)


x=df.drop(columns=['Fertilizer Name'])
y=df['Fertilizer Name']




labelencoder=LabelEncoder()
y=labelencoder.fit_transform(y)


x=pd.get_dummies(columns=['Soil Type','Crop Type'],drop_first=True,data=x)
test=pd.get_dummies(columns=['Soil Type','Crop Type'],drop_first=True,data=test)


test


x=x.astype(int)
test=test.astype(int)


scaler=StandardScaler()
scaler.fit(x[['Temparature','Humidity','Moisture','Nitrogen','Potassium','Phosphorous']])
x[['Temparature','Humidity','Moisture','Nitrogen','Potassium','Phosphorous']]=scaler.transform(x[['Temparature','Humidity','Moisture','Nitrogen','Potassium','Phosphorous']])
x.std()

scaler.fit(test[['Temparature','Humidity','Moisture','Nitrogen','Potassium','Phosphorous']])
test[['Temparature','Humidity','Moisture','Nitrogen','Potassium','Phosphorous']]=scaler.transform(test[['Temparature','Humidity','Moisture','Nitrogen','Potassium','Phosphorous']])


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


results=[]



logistic=LogisticRegression()
logistic.fit(x_train,y_train)
y_pred=logistic.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


decision_tree=DecisionTreeClassifier()
decision_tree.fit(x_train,y_train)
y_pred=decision_tree.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


naive_bayes=GaussianNB()
naive_bayes.fit(x_train,y_train)
y_pred=naive_bayes.predict(x_test)
print(accuracy_score(y_test,y_pred))


from lightgbm import LGBMClassifier

model = LGBMClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(accuracy_score(y_test,y_pred))


from xgboost import XGBClassifier

modelxg = XGBClassifier(tree_method='hist')  # or 'gpu_hist' if on GPU
modelxg.fit(x_train, y_train)
y_pred = modelxg.predict(x_test)
print(accuracy_score(y_test,y_pred))


prediction=modelxg.predict(test.drop(columns=['id']))


submission = pd.DataFrame({
    'id': test['id'], 
    'Fertilizer Name': prediction
})


submission.head()


submission.to_csv('submission.csv', index=False)




