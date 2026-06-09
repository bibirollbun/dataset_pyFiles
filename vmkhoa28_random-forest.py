import pandas as pd
import numpy as np

test = pd.read_csv('/kaggle/input/customer-churn-prediction-2020/test.csv')
train = pd.read_csv('/kaggle/input/customer-churn-prediction-2020/train.csv')

pd.set_option('display.max_columns',None)

train.info()


test = test.drop(columns = ['id'])
test.info()


#Hashing Encoding State
import category_encoders as ce

h = ce.HashingEncoder(cols = 'state')
train = h.fit_transform(train)
test = h.fit_transform(test)
train.head()


test.head()


#area code onehotencoding
from sklearn.preprocessing import OneHotEncoder
ohe_area = OneHotEncoder()
ohe_area.fit(train[['area_code']])

encoded_values = ohe_area.transform(train[['area_code']])
train[ohe_area.categories_[0]] = encoded_values.toarray()
train = train.drop('area_code', axis = 1)

encoded_values = ohe_area.transform(test[['area_code']])
test[ohe_area.categories_[0]] = encoded_values.toarray()
test = test.drop('area_code', axis = 1)
train.head()


#replace no yes to 0 1
train.international_plan.replace(['no','yes'],[0,1],inplace = True)
train.voice_mail_plan.replace(['no','yes'],[0,1],inplace=True)
train.churn.replace(['no','yes'],[0,1],inplace = True)
test.international_plan.replace(['no','yes'],[0,1],inplace = True)
test.voice_mail_plan.replace(['no','yes'],[0,1],inplace = True)



train.head()


#split
from sklearn.model_selection import train_test_split

x = train.drop('churn', axis=1).values
y = train.churn.values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


#minmax scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print(classification_report(y_test, y_pred))

print('Accuracy: ')
print('{}'.format(accuracy_score(y_test, y_pred)))




