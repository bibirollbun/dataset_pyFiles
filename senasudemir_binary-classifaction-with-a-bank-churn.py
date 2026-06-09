import pandas as pd
pd.set_option('display.max_columns',100)
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import normalize, scale
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


df=pd.read_csv('/kaggle/input/playground-series-s4e1/train.csv')


df.head()


df.isnull().sum()


df.shape


df.info()


df['Geography'].unique()


df.describe().T


plt.figure(figsize=(12,6))
sns.histplot(df['CreditScore'], bins=30, kde=True, color='teal')
plt.title('Distribution of Credit Scores')
plt.show()

plt.figure(figsize=(12,6))
sns.histplot(df['Age'], bins=30, kde=True, color='purple')
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(12,6))
sns.histplot(df['Balance'], bins=30, kde=True, color='orange')
plt.title('Balance Distribution')
plt.show()


plt.figure(figsize=(10,5))
sns.countplot(x='Geography', data=df, palette='coolwarm')
plt.title('Customer Count by Geography')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x='Gender', data=df, palette='pastel')
plt.title('Customer Count by Gender')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x='Exited', data=df, palette='Set1')
plt.title('Churn Distribution (0 = Stayed, 1 = Left)')
plt.show()


plt.figure(figsize=(12,6))
sns.boxplot(x='Exited', y='Age', data=df, palette='coolwarm')
plt.title('Age vs. Churn')
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x='Exited', y='Balance', data=df, palette='pastel')
plt.title('Balance vs. Churn')
plt.show()

plt.figure(figsize=(12,6))
sns.violinplot(x='Exited', y='CreditScore', data=df, palette='muted')
plt.title('Credit Score Distribution for Churned & Non-Churned Customers')
plt.show()


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='viridis');


plt.figure(figsize=(10,5))
sns.countplot(x='NumOfProducts', hue='Exited', data=df, palette='coolwarm')
plt.title('Num of Products vs. Churn')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x='IsActiveMember', hue='Exited', data=df, palette='Set2')
plt.title('Active Members vs. Churn')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x='Geography', hue='Exited', data=df, palette='pastel')
plt.title('Churn by Geography')
plt.show()


def feature_engineering(df):
    df['Senior']=df['Age'].apply(lambda x:1 if x>=60 else 0)
    df['Active_by_CreditCard']=df['HasCrCard']*df['IsActiveMember']
    df['Products_Per_Tenure']=df['Tenure']/df['NumOfProducts']
    df['AgeCat']=np.round(df['Age']/20).astype('int').astype('category')


feature_engineering(df)


x=df.drop(['id','CustomerId','Surname','Exited'],axis=1)
y=df[['Exited']]


x=pd.get_dummies(x,drop_first=True)


def classification_algo(x, y, confusion_mtr=False, classification_rpt=False):
    g = GaussianNB()
    b = BernoulliNB()
    l = LogisticRegression()
    d = DecisionTreeClassifier()
    rf = RandomForestClassifier()
    h = GradientBoostingClassifier()
    k = KNeighborsClassifier()
    
    algos = [g, b, l, d, rf, h, k]
    algo_names = ['Gaussian NB', 'Bernoulli NB', 'Logistic Regression', 
                  'Decision Tree Classifier', 'Random Forest Classifier', 
                  'Gradient Boosting Classifier', 'KNeighbors Classifier']

    accuracy = []
    confusion = []
    classification = []
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Prepare a DataFrame to store results
    result = pd.DataFrame(columns=['Accuracy Score', 'Confusion Matrix', 'Classification Report'], 
                          index=algo_names)

    labels = sorted(y["Exited"].unique())

    for algo in algos:
        p = algo.fit(x_train, y_train).predict(x_test)
        accuracy.append(accuracy_score(y_test, p))
        confusion.append(confusion_matrix(y_test, p, labels=labels))
        classification.append(classification_report(y_test, p))

    # Store results
    result['Accuracy Score'] = accuracy
    result['Confusion Matrix'] = confusion
    result['Classification Report'] = classification

    # Sort results by accuracy
    r_table = result.sort_values('Accuracy Score', ascending=False)
    
    if confusion_mtr:
        for index, row in r_table.iterrows():
            confusion_mat = np.array(row['Confusion Matrix'])
            print(f"Confusion Matrix of {index}")
            plt.figure(figsize=(5, 4))
            sns.heatmap(confusion_mat, annot=True, fmt="d", 
                        xticklabels=labels, yticklabels=labels, cmap="Blues")
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.show()
    
    if classification_rpt:
        for index, row in r_table.iterrows():
            print(f"Classification Report of {index}:")
            print(row['Classification Report'])

    return r_table[['Accuracy Score']]



classification_algo(x,y,confusion_mtr=True,classification_rpt=True)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
h = GradientBoostingClassifier()
model=h.fit(x_train, y_train)


import joblib
joblib.dump(model, 'best_classification_model.pkl')


x=df.drop(['id','CustomerId','Surname','Exited'],axis=1)
y=df[['Exited']]


x=pd.get_dummies(x,drop_first=True)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
h = GradientBoostingClassifier()
model=h.fit(x_train, y_train)


df_test=pd.read_csv('/kaggle/input/playground-series-s4e1/test.csv')


submission=pd.DataFrame({
    'id':df_test['id']
})


feature_engineering(df_test)


df_test.head()


df_test=df_test.drop(['id','CustomerId','Surname'],axis=1)


df_test=pd.get_dummies(df_test,drop_first=True)


predicstions=model.predict(df_test)


submission['Exited']=predicstions


x=df.drop(['id','CustomerId','Surname','Exited'],axis=1)
y=df[['Exited']]


x=pd.get_dummies(x,drop_first=True)


x_train , x_test, y_train, y_test =train_test_split(x,y,test_size=.2,random_state=42)


model2=Sequential()
model2.add(Dense(8, activation='relu'))
model2.add(Dense(32,activation='relu')) 
model2.add(Dense(64,activation='relu')) 
model2.add(Dense(128,activation='relu'))
model2.add(Dense(64,activation='relu'))
model2.add(Dense(32,activation='relu'))
model2.add(Dense(2,activation='softmax'))
model2.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


x_train.shape,y_train.shape


history=model2.fit(x_train,y_train,epochs=50,validation_split=.20,verbose=1)


predictions=model2.predict(x_test)
predictions_labels = np.argmax(predictions, axis=1)
accuracy_score(predictions_labels,y_test)


predictions=model2.predict(df_test)


submission['Exited']=predictions


submission.head()

