import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split ,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from category_encoders import LeaveOneOutEncoder

import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from sklearn.metrics import r2_score,mean_squared_error


train_data=pd.read_csv(r'/kaggle/input/playground-series-s4e2/train.csv')
test_data=pd.read_csv(r'/kaggle/input/playground-series-s4e2/test.csv')


train_data.head(10)


train_data.info()


train_data.describe()


train_data.nunique()


train_data.isnull().sum()


train_data.duplicated().sum()


 train_data.select_dtypes(include='object').columns
 le= LabelEncoder()
 la= LabelEncoder()
 train_data['NObeyesdad']=la.fit_transform(train_data['NObeyesdad'])


def encoding(data):
    non_numeric= data.select_dtypes(include='object').columns
    for col in non_numeric:
        data[col]=le.fit_transform(data[col])
    return data


encoding(train_data)


train_data.info()


encoding(test_data)



non_numeric= test_data.select_dtypes(include='object').columns
for col in non_numeric:
    test_data[col]=le.fit_transform(test_data[col])



def scaling(data):
    sc=StandardScaler()
    data['Age']=sc.fit_transform(data[['Age']])
    data['Weight']=sc.fit_transform(data[['Weight']])
    return data


scaling(train_data)


scaling(test_data)


plt.figure(figsize=(16,8))
sns.heatmap(train_data.corr(),annot=True,fmt='.2f',linewidth=1)


x=train_data.drop(['id','NObeyesdad'],axis=1)
y=train_data['NObeyesdad']


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=21)


Accuracies=[]
def models(model):
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    acc=accuracy_score(y_pred,y_test)
    print('Accuracy = ',acc)
    Accuracies.append(acc)


model1=DecisionTreeClassifier()
models(model1)


model2=SVC()
models(model2)


model3=RandomForestClassifier()
models(model3)


model4=GradientBoostingClassifier()
models(model4)


model5=KNeighborsClassifier()
models(model5)


model6=GaussianNB()
models(model6)


model7=LogisticRegression()
models(model7)


model8=XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=300)
models(model8)


Algorithms=['Decision Tree','SVM','Random Forest','Gradient Boosting','KNeighbors','GaussianNB','Logistic Regression','XGBClassifier']


Algorithms_frame=pd.DataFrame({'Algorithms':Algorithms,'Accuracies': Accuracies})
Algorithms_frame


test_data1=test_data.drop(['id'],axis=1)
prex=model8.predict(test_data1)


submission=pd.DataFrame({"id":test_data['id'],"NObeyesdad":prex})
submission['NObeyesdad']=la.inverse_transform(prex)
submission.to_csv("submission.csv",index=False)

