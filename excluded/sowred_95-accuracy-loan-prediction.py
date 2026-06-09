
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




train=pd.read_csv("/kaggle/input/playground-series-s4e10/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s4e10/test.csv")


train.head()


train.info()


train.isnull().sum()


train['id'].duplicated().sum()


train["loan_status"].value_counts()


train.describe()


plt.figure(figsize=(4,4))
ax=sns.countplot(x='loan_status',data=train, palette='viridis')
ax.bar_label(ax.containers[0])
plt.title("Loan Status Distribution")  
plt.xlabel("Loan Status (0 = Default, 1 = Paid Off)")
plt.show()



features = ['person_income', 'loan_amnt']
plt.figure(figsize=(15, 3))
for i, feature in enumerate(features):
  plt.subplot(1, 2, i+1)
  sns.histplot(data=train, x=feature, kde=False)



features = ['person_age', 'person_emp_length','loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

plt.figure(figsize=(16,18))
for i, feature in enumerate(features):
    plt.subplot(5, 1, i + 1)
    train[feature + '_str'] = train[feature].round(0).astype(str)  
    sns.countplot(x=feature + '_str', data=train, hue='loan_status', palette="viridis")
    plt.xticks(rotation=45)  
    plt.title(f'Countplot for {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')

plt.tight_layout()
plt.subplots_adjust(hspace=0.5)
plt.show()



from sklearn.preprocessing import  LabelEncoder
encoder=LabelEncoder()
train['person_home_ownership']=encoder.fit_transform(train['person_home_ownership'])
train['loan_intent']=encoder.fit_transform(train['loan_intent'])
train['loan_grade']=encoder.fit_transform(train['loan_grade'])
train['cb_person_default_on_file']=encoder.fit_transform(train['cb_person_default_on_file'])


train.head(3)


train.info()


test.head(3)


test.info()


from sklearn.preprocessing import  LabelEncoder
encoder=LabelEncoder()
test['person_home_ownership']=encoder.fit_transform(test['person_home_ownership'])
test['loan_intent']=encoder.fit_transform(test['loan_intent'])
test['loan_grade']=encoder.fit_transform(test['loan_grade'])
test['cb_person_default_on_file']=encoder.fit_transform(test['cb_person_default_on_file'])


pqr=test
pqr


from sklearn.model_selection import train_test_split
X = train.drop(['loan_status','person_age_str','person_emp_length_str','loan_int_rate_str',
                'loan_percent_income_str','cb_person_cred_hist_length_str'],axis=1)
y = train['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)


from sklearn.linear_model import LogisticRegression
lrg = LogisticRegression()
lrg.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, confusion_matrix
y_pred1 = lrg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred1)
print(f'Accuracy: {accuracy}')


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


from sklearn.ensemble import RandomForestClassifier
r=RandomForestClassifier()
r.fit(X_train, y_train)
y_p = r.predict(pqr)



f= pqr['id']
new_dataFrame = pd.DataFrame({'id':f, 'loan_status':y_p})
new_dataFrame.to_csv('submission.csv' , index=False) 
new_dataFrame

