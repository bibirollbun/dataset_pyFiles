!pip install imbalanced-learn==0.13.0 --quiet



!pip install scikit-learn==1.5.2 --upgrade --quiet


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


import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv(r'/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/train_dataset.csv')
test = pd.read_csv(r'/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/test_dataset_exam.csv')
train.head()


train['satisfaction'] = train['satisfaction'].map({'satisfied': 1,'neutral or dissatisfied': 0})

print(train['satisfaction'].value_counts())


test['satisfaction']=5
#test.head()


df = pd.concat([train, test],axis=0)
#df.head()


df['satisfaction'].value_counts()


df.info()


df.isnull().sum()


df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median(), inplace=True)
#df.isnull().sum()


df.drop(columns=['Unnamed: 0','id'],axis=1,inplace=True)


import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(6,4))
sns.countplot(x='Gender', hue='satisfaction', data=train)
plt.title("Gender vs Satisfaction")
plt.show()


plt.figure(figsize=(6,4))
sns.countplot(x='Class', hue='satisfaction', data=train)
plt.title("Class vs Satisfaction")
plt.show()


plt.figure(figsize=(8,4))
sns.histplot(train['Age'], bins=30, kde=True)
plt.title("Age Distribution of Passengers")
plt.show()



plt.figure(figsize=(8,4))
sns.histplot(train['Flight Distance'], bins=30, kde=True)
plt.title("Flight Distance Distribution")
plt.show()


service_cols = ['Inflight wifi service', 'Departure/Arrival time convenient',
                'Ease of Online booking', 'Gate location', 'Food and drink',
                'Online boarding', 'Seat comfort', 'Inflight entertainment',
                'On-board service', 'Leg room service', 'Baggage handling',
                'Checkin service', 'Inflight service', 'Cleanliness']

plt.figure(figsize=(10,6))
train[service_cols].mean().sort_values().plot(kind='barh', color='skyblue')
plt.title("Average Service Ratings")
plt.show()



plt.figure(figsize=(6,4))
sns.countplot(x='satisfaction', data=train) 
plt.title("Target Variable Distribution")
plt.show()


categorical_cols = train.select_dtypes(include=['object']).columns

for col in categorical_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(x=col, data=train)
    plt.title(f"Distribution of {col}")
    plt.show()


"""numeric_cols = train.select_dtypes(include=['number']).columns

for col in numeric_cols:
    plt.figure(figsize=(8,4))
    sns.boxplot(x=train[col], color='lightgreen')
    plt.title(f"Boxplot of {col}")
    plt.show()"""


df.describe()


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


cat=df.select_dtypes(include='object')
cat.head()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in cat.columns:
    cat[i]=le.fit_transform(cat[i])
cat.head()


cat_col=df.select_dtypes(include='object')
df=df.drop(cat_col.columns,axis=1)
df.head()


df=pd.concat([df,cat],axis=1)
df.head()


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()


test = df[df['satisfaction']==5]
train = df[df['satisfaction']!=5]



x = train.drop('satisfaction',axis=1)
y = train['satisfaction']


from imblearn.over_sampling import SMOTE
smote=SMOTE()
x,y=smote.fit_resample(x,y)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


x_train.shape,y_train.shape


x_test.shape,y_test.shape


from xgboost import XGBClassifier


xgb_model = XGBClassifier(n_estimators=200,learning_rate=0.1,max_depth=6,random_state=42,n_jobs=-1)

xgb_model.fit(x_train, y_train)


prediction=xgb_model.predict(x_test)


from sklearn.metrics import classification_report
print (classification_report(y_test,prediction))


xgb_model.score(x_train,y_train),xgb_model.score(x_test,y_test)


#test.head()


test.drop('satisfaction',axis=1,inplace=True)
test.head()


test_pred=xgb_model.predict(test)
test_pred=pd.DataFrame(test_pred)
test_pred


test_pred[0] = test_pred[0].map({1:'satisfied',0:'neutral or dissatisfied'})
test_pred.head()


test1=pd.read_csv(r'/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/test_dataset_exam.csv')

#test1.head()


submission = pd.DataFrame({'ID': test1['id'],'satisfaction': test_pred[0]  })


submission.to_csv("submission.csv", index=False)
submission.head()




