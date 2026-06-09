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


df=pd.read_csv('/kaggle/input/ghouls-goblins-and-ghosts-boo/train.csv.zip')


df.head()


df.isnull().sum()


df.info()


df.shape


df.describe().T


df["type"].unique()


df["color"].unique()


sns.histplot(df['bone_length'], kde=True, color='blue', bins=20)
plt.title('Bone Length Distribution')
plt.show()


corr = df[['bone_length', 'rotting_flesh', 'hair_length', 'has_soul']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


sns.countplot(data=df, x='type', palette='viridis')
plt.title('Class Distribution')
plt.show()


sns.pairplot(df, hue='type', palette='Set1')
plt.show()


sns.boxplot(data=df, x='type', y='bone_length')
plt.title('Bone Length by Type')
plt.show()


def feature_engineering(df):
    df['hair_soul'] = df['hair_length'] * df['has_soul']
    df['hair_bone'] = df['hair_length'] * df['bone_length']


feature_engineering(df)


x=df.drop(['id','type'],axis=1)
y=df[['type']]


x=pd.get_dummies(x,drop_first=True)


import logging
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

def classification_algo(x, y, confusion_mtr=False, classification_rpt=False):
    logging.basicConfig(level=logging.ERROR)
    g = GaussianNB()
    b = BernoulliNB()
    l = LogisticRegression()
    d = DecisionTreeClassifier()
    rf = RandomForestClassifier()
    h = GradientBoostingClassifier()
    k = KNeighborsClassifier()
    lgbm = lgb.LGBMClassifier(verbose=-1)  
    
    algos = [g, b, l, d, rf, h, k, lgbm]
    algo_names = ['Gaussian NB', 'Bernoulli NB', 'Logistic Regression', 
                  'Decision Tree Classifier', 'Random Forest Classifier', 
                  'Gradient Boosting Classifier', 'KNeighbors Classifier', 
                  'LightGBM Classifier']

    accuracy = []
    confusion = []
    classification = []
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Prepare a DataFrame to store results
    result = pd.DataFrame(columns=['Accuracy Score', 'Confusion Matrix', 'Classification Report'], 
                          index=algo_names)

    labels = sorted(y["type"].unique())

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


df_test=pd.read_csv('/kaggle/input/ghouls-goblins-and-ghosts-boo/test.csv.zip')


df_test.head()


feature_engineering(df_test)


df_test.head()


submission=pd.DataFrame({
    'id':df_test['id']
})


df_test.drop('id',axis=1,inplace=True)


df_test=pd.get_dummies(df_test,drop_first=True)


predictions=model1.predict(df_test)


submission['type']=predictions


submission.to_csv('submission.csv',index=False)

