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


train=pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')
sample=pd.read_csv('/kaggle/input/playground-series-s4e2/sample_submission.csv')


train.head()


test.head()


sample.head()


train["NObeyesdad"].unique()


train.drop(["id"],axis=1,inplace=True)
test.drop(["id"],axis=1,inplace=True)


train["Gender"]=train["Gender"].replace({"Male":1,"Female":0})
test["Gender"]=test["Gender"].replace({"Male":1,"Female":0})


mapping = {"yes":1,"no":0}
cols=["family_history_with_overweight","FAVC","SMOKE","SCC"]
train[cols] = train[cols].replace(mapping)
test[cols] = test[cols].replace(mapping)


train.head()


test["CALC"].unique()


train["MTRANS"].unique()


train["CALC"] = train["CALC"].replace({"Always":3,"Frequently":2,"Sometimes":1, "no":0})
test["CALC"] = test["CALC"].replace({"Always":3, "Frequently":2,"Sometimes":1, "no":0})

train["CAEC"] = train["CAEC"].replace({"Always":3,"Frequently":2,"Sometimes":1, "no":0})
test["CAEC"] = test["CAEC"].replace({"Always":3,"Frequently":2,"Sometimes":1, "no":0})


train= pd.get_dummies(train, columns=["MTRANS"], drop_first=True)
test= pd.get_dummies(test, columns=["MTRANS"], drop_first=True)


train.head()


test.head()


print("x DataFrame'inin veri tipleri (ölçeklendirme öncesi):")
print(train.dtypes) # x değişkenini oluşturduktan sonra bu kodu çalıştırın

print("\ntest DataFrame'inin veri tipleri (ölçeklendirme öncesi):")
print(test.dtypes) # test değişkenini One-Hot Encoding ve hizalamadan sonra bu kodu çalıştırın


x = pd.get_dummies(train.drop(['NObeyesdad'], axis = 1))
y = train['NObeyesdad']


from sklearn.preprocessing import LabelEncoder #tahmin edeceğimiz değeri sayısal hale getirdik
encoder = LabelEncoder()
y = encoder.fit_transform(y)


y


def fnc_classification_all_model(x,y):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split   
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    from sklearn.metrics import confusion_matrix,classification_report
              
    g=GaussianNB()
    b=BernoulliNB()
    D=DecisionTreeClassifier()
    R=RandomForestClassifier()
    Log=LogisticRegression()
    XGB=XGBClassifier()
    G=GradientBoostingClassifier()
      
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
    
    
    algos=[g,b,D,R,Log,XGB,G]
    algo_names=['GaussianNB','BernoulliNB','DecisionTreeClassifier','RandomForestClassifier','LogisticRegression','XGBClassifier','GradientBoostingClassifier']
    
    accuracy_scored=[]
    precision_scored=[]
    recall_scored=[]
    f1_scored=[]
       
   
    for item in algos:
        print(item)

        predict=item.fit(x_train,y_train).predict(x_test)
        
        
        accuracy_scored.append(accuracy_score(y_test,predict))
        precision_scored.append(precision_score(y_test,predict,average='macro'))
        recall_scored.append(recall_score(y_test,predict,average='macro'))
        f1_scored.append(f1_score(y_test,predict,average='macro'))

    result=pd.DataFrame(columns=['accuracy_score','f1_score','recall_score','precision_score'],index=algo_names)
    result['accuracy_score']=accuracy_scored
    result['f1_score']=f1_scored
    result['recall_score']=recall_scored
    result['precision_score']=precision_scored
    
    return result.sort_values('accuracy_score',ascending=False)


fnc_classification_all_model(x,y)


from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler = MinMaxScaler()#verileri birbirine kıyasla normalize ediyor 
x = scaler.fit_transform(x)
test = scaler.transform(test)


from sklearn.ensemble import GradientBoostingClassifier
G=GradientBoostingClassifier()


gmodel=G.fit(x,y)
predg=G.predict(test)


predg


sample.head()


predg= encoder.inverse_transform(predg) #etiketlerimiz 012 ama tahmin etmemiz gerekenler yazı bu kodla geri aldık 


predg


submission = pd.DataFrame({'id': sample['id'], 'NObeyesdad': predg})
submission.to_csv('submission.csv', index=False)
submission
#id sütnunu silmiştik kaggle sistem formatına uygun olması için tekrar sample dosyasından çekerek geri ekledik
#yükleyebilmek için tahmini csv dosyası olarak dışarı aktardık




