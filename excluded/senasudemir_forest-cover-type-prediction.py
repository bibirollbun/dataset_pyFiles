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


df=pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')


df.head()


df.isnull().sum()


df.info()


df.shape


df.describe().T


df["Cover_Type"].unique()


plt.figure(figsize=(50,30))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='viridis');


sns.boxplot(x=df['Horizontal_Distance_To_Fire_Points'])


plt.figure(figsize=(8,5))
sns.countplot(x=df['Cover_Type'], palette='viridis')
plt.xlabel("Cover Type")
plt.ylabel("Count")
plt.title("Distribution of Cover Types")
plt.show()


sns.pairplot(df, hue="Cover_Type", vars=['Elevation', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology'])


plt.figure(figsize=(10,6))
sns.boxplot(x='Cover_Type', y='Elevation', data=df, palette="coolwarm")
plt.title("Elevation Distribution Across Cover Types");


plt.figure(figsize=(10,6))
sns.boxplot(x='Cover_Type', y='Slope', data=df, palette="coolwarm");


plt.figure(figsize=(10,6))
sns.boxplot(x='Cover_Type', y='Vertical_Distance_To_Hydrology', data=df, palette="coolwarm");


upper_bound=df.quantile(q=.97,numeric_only=True)
lower_bound = df.quantile(q=.03,numeric_only=True)


df = df[(df['Horizontal_Distance_To_Fire_Points'] <= upper_bound['Horizontal_Distance_To_Fire_Points'])]


df.shape


abs(df.corr(numeric_only=True)['Cover_Type'].sort_values(ascending=False))


x=df.drop(['Cover_Type','Id'],axis=1)
y=df[['Cover_Type']]


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

    labels = sorted(y["Cover_Type"].unique())

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
rf = RandomForestClassifier()
model1=rf.fit(x_train, y_train)


df_test=pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')


df_submission=pd.read_csv('/kaggle/input/forest-cover-type-prediction/sampleSubmission.csv')


submission=pd.DataFrame({
    'Id':df_test['Id']}
)


df_test.drop('Id',axis=1,inplace=True)


predictions=model1.predict(df_test)


submission['Cover_Type']=predictions



submission.to_csv("submission1.csv", index=False)


x=df.drop(['Cover_Type','Id'],axis=1)
y=df[['Cover_Type']]


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y['Cover_Type'])


x_train , x_test, y_train, y_test =train_test_split(x,y,test_size=.2,random_state=42)


model2=Sequential()
model2.add(Dense(8, activation='relu'))
model2.add(Dense(32,activation='relu')) 
model2.add(Dense(64,activation='relu')) 
model2.add(Dense(128,activation='relu'))
model2.add(Dense(64,activation='relu'))
model2.add(Dense(32,activation='relu'))
model2.add(Dense(7,activation='softmax'))
model2.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


x_train.shape,y_train.shape


history=model2.fit(x_train,y_train,epochs=200,validation_split=.20,verbose=0)


predictions=model2.predict(x_test)
predictions_labels = np.argmax(predictions, axis=1)
accuracy_score(predictions_labels,y_test)


df_test=pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')


df_test.head()


submission2=pd.DataFrame({
    'Id':df_test['Id']})


df_test.drop('Id',axis=1,inplace=True)


df_submission=pd.read_csv('/kaggle/input/forest-cover-type-prediction/sampleSubmission.csv')


predictions=model2.predict(df_test)
predictions_labels = np.argmax(predictions, axis=1)


predictions_labels


submission2['Cover_Type']=predictions_labels


submission2['Cover_Type']+=1


submission2.to_csv("submission2.csv", index=False)


submission2.head()

