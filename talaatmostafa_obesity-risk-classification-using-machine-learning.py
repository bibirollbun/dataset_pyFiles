import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix , accuracy_score , classification_report
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.preprocessing import LabelEncoder , StandardScaler

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv(r"/kaggle/input/playground-series-s4e2/train.csv")
train.head()


train.tail()


train.shape


train.columns


train.info()


train.isnull().sum()


train.describe()


train.duplicated().sum()


la = LabelEncoder()


obj = train.select_dtypes(exclude = ['int64' , 'float64'])


obj.columns


target_col = 'NObeyesdad'

feature_categorical_cols = [col for col in obj if col != target_col]

label_encoders = {}

for col in feature_categorical_cols:
    la = LabelEncoder()
    train[col] = la.fit_transform(train[col].astype(str))
    label_encoders[col] = la 

la_target = LabelEncoder()
train[target_col] = la_target.fit_transform(train[target_col].astype(str))
label_encoders[target_col] = la_target


train.head()


test = pd.read_csv(r"/kaggle/input/playground-series-s4e2/test.csv")
test.head()


obj_test = test.select_dtypes(include = 'object')


for col in obj_test.columns:
    test[col] = la.fit_transform(test[col])



scaler = StandardScaler()


scal1 = train[['Age']]
scal2 = train[['Weight']]

train['Age'] = scaler.fit_transform(scal1)
train['Weight'] = scaler.fit_transform(scal2)


scal1 = test[['Age']]
scal2 = test[['Weight']]

test['Age'] = scaler.fit_transform(scal1)
test['Weight'] = scaler.fit_transform(scal2)


test.head()





X = train.drop(['id','NObeyesdad'] , axis = 1)
y = train['NObeyesdad']


X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = .2 , random_state = 42)


model1 = LogisticRegression()
model2 = SVC()
model3 = RandomForestClassifier()
model4 = GradientBoostingClassifier(learning_rate = 0.1, max_depth = 3, n_estimators = 200)
model5 = GaussianNB()
model6 = DecisionTreeClassifier()


def pred(model):
    model.fit(X_train , y_train)
    pre = model.predict(X_test)
    print(classification_report(y_test , pre))


pred(model1)


pred(model2)


pred(model3)


pred(model4)


pred(model5)


pred(model6)


testx = test.drop('id' , axis = 1)


prex = model4.predict(testx)


final_predicted_labels = label_encoders[target_col].inverse_transform(prex)

submission = pd.DataFrame({'id': test['id'], "NObeyesdad": final_predicted_labels})


submission

