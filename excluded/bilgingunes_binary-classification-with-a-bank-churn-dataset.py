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


train=pd.read_csv('/kaggle/input/playground-series-s4e1/train.csv', nrows=110023)
test=pd.read_csv('/kaggle/input/playground-series-s4e1/test.csv', nrows=110023)
sample=pd.read_csv('/kaggle/input/playground-series-s4e1/sample_submission.csv', nrows=110023)


train.head()


test.head()


train.isnull().sum().sum()


train.shape, test.shape, sample.shape


train=train


train.drop(["CustomerId","id","Surname"],axis=1,inplace=True)
test.drop(["CustomerId","id","Surname"],axis=1,inplace=True)


import seaborn as sns
sns.countplot(train,x='Geography')


sns.countplot(train,x='Exited')


train= pd.get_dummies(train, columns=["Geography"], drop_first=True)
test= pd.get_dummies(test, columns=["Geography"], drop_first=True)


train["Gender"]=train["Gender"].replace({"Male":1,"Female":0})
test["Gender"]=test["Gender"].replace({"Male":1,"Female":0})


train.head()


import pandas as pd
from sklearn.utils import shuffle

# Exited kolonuna göre verileri ayır
class_0 = train[train["Exited"] == 0]
class_1 = train[train["Exited"] == 1]

# Hedef: azınlık sınıfını çoğunluk sınıfı kadar çoğaltmak
repeat_factor = len(class_0) // len(class_1)

# Azınlık sınıfını çoğalt
class_1_oversampled = pd.concat([class_1] * repeat_factor, ignore_index=True)

# Gerekirse birkaç tane daha örnek ekle (tam eşitlik için)
extra = len(class_0) - len(class_1_oversampled)
class_1_extra = class_1.sample(extra, replace=True, random_state=42)
class_1_oversampled = pd.concat([class_1_oversampled, class_1_extra], ignore_index=True)

# Dengeleme sonrası yeni veri seti
balanced_train = pd.concat([class_0, class_1_oversampled], ignore_index=True)

# Rastgele karıştır (önemli)
train = shuffle(balanced_train, random_state=42)

# Kontrol
print(train["Exited"].value_counts())




x = train.drop('Exited', axis=1)
y = train['Exited']


x.head()


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
from sklearn.preprocessing import StandardScaler,MinMaxScaler


def fnc_classification_all_model(x, y):
    from sklearn.naive_bayes import GaussianNB, BernoulliNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import pandas as pd

    g = GaussianNB()
    b = BernoulliNB()
    KN = KNeighborsClassifier()
    D = DecisionTreeClassifier()
    R = RandomForestClassifier()
    Log = LogisticRegression()
    XGB = XGBClassifier()
    G = GradientBoostingClassifier()
      
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
    
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    algos = [g, b, KN, D, R, Log, XGB, G]
    algo_names = ['GaussianNB', 'BernoulliNB', 'KNeighborsClassifier', 'DecisionTreeClassifier', 
                  'RandomForestClassifier', 'LogisticRegression', 'XGBClassifier', 'GradientBoostingClassifier']
    
    accuracy_scored = []
    precision_scored = []
    recall_scored = []
    f1_scored = []

    for item in algos:
        print(item)
        
        predict = item.fit(x_train, y_train).predict(x_test)
        
        accuracy_scored.append(accuracy_score(y_test, predict))
        precision_scored.append(precision_score(y_test, predict))
        recall_scored.append(recall_score(y_test, predict))
        f1_scored.append(f1_score(y_test, predict))

    result = pd.DataFrame(columns=['accuracy_score', 'f1_score', 'recall_score', 'precision_score'],
                          index=algo_names)
    result['accuracy_score'] = accuracy_scored
    result['f1_score'] = f1_scored
    result['recall_score'] = recall_scored
    result['precision_score'] = precision_scored
    
    return result.sort_values('accuracy_score', ascending=False)



fnc_classification_all_model(x,y)


scaler = MinMaxScaler()#verileri birbirine kıyasla normalize ediyor 
x = scaler.fit_transform(x)
test = scaler.transform(test)


from sklearn.ensemble import RandomForestClassifier
R = RandomForestClassifier()


rmodel=R.fit(x,y)
predr=R.predict(test)


predr


sample.head()


submission = pd.DataFrame({'id': sample['id'], 'Exited': predr})
submission.to_csv('submission.csv', index=False)
submission
#id sütnunu silmiştik kaggle sistem formatına uygun olması için tekrar sample dosyasından çekerek geri ekledik
#yükleyebilmek için tahmini csv dosyası olarak dışarı aktardık



























































































