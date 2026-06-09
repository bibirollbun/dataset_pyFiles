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


df=pd.read_csv('/kaggle/input/playground-series-s3e23/train.csv')


df.head()


df.shape


df.isnull().sum()


df.info()


plt.figure(figsize=(8,5))
sns.histplot(df['v(g)'], bins=20, kde=True, color='blue')
plt.title("Distribution of Code Complexity (v(g))")
plt.xlabel("Cyclomatic Complexity (v(g))")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(8,5))
sns.scatterplot(x=df['loc'], y=df['defects'].astype(int), alpha=0.7)
plt.title("Code Size vs. Defects")
plt.xlabel("Lines of Code (LOC)")
plt.ylabel("Defects (0 = No, 1 = Yes)")
plt.show()


plt.figure(figsize=(10,6))
sns.boxplot(x=df['defects'], y=df['v(g)'])
plt.title("Cyclomatic Complexity (v(g)) by Defect Status")
plt.xlabel("Defects (0 = No, 1 = Yes)")
plt.ylabel("Cyclomatic Complexity (v(g))")
plt.show()


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Code Metrics")
plt.show()


plt.figure(figsize=(8,5))
sns.barplot(x=['uniq_Op', 'uniq_Opnd'], y=[df['uniq_Op'].mean(), df['uniq_Opnd'].mean()], palette="viridis")
plt.title("Average Unique Operators & Operands")
plt.ylabel("Count")
plt.show()


x=df.drop(['defects','id'],axis=1)
y=df[['defects']]


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

    labels = sorted(y["defects"].unique())

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


h = GradientBoostingClassifier()
model1=h .fit(x, y)


df_test=pd.read_csv('/kaggle/input/playground-series-s3e23/test.csv')


df_test.head()


submission=pd.DataFrame({
    'id':df_test['id']}
)


df_test.drop('id',axis=1,inplace=True)


predictions=model1.predict(df_test)


predictions


submission['defects']=predictions



submission.to_csv("submission.csv", index=False)


x=df.drop(['defects','id'],axis=1)
y=df[['defects']]


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y['defects'])


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


history=model2.fit(x_train,y_train,epochs=50,validation_split=.20,verbose=1)


predictions=model2.predict(x_test)
predictions_labels = np.argmax(predictions, axis=1)
accuracy_score(predictions_labels,y_test)


predictions=model2.predict(df_test)


submission['defects']=predictions


submission.to_csv("submission.csv", index=False)

