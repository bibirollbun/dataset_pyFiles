import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings

import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix,classification_report

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)


df=pd.read_csv('/kaggle/input/playground-series-s4e1/train.csv')


df.head()


df.shape


df.info()


df.isnull().sum()


plt.figure(figsize=(17,10))
corr = df.drop('id',axis=1).corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution of Customers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='CreditScore', y='Balance', hue='Exited', alpha=0.6)
plt.title('Credit Score vs. Balance')
plt.xlabel('Credit Score')
plt.ylabel('Balance')
plt.legend(title='Exited', loc='upper right')
plt.show()


plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Geography', hue='Exited')
plt.title('Number of Customers by Country')
plt.xlabel('Country')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.legend(title='Exited')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Exited', y='Balance')
plt.title('Account Balance by Exit Status')
plt.xlabel('Exited (0 = Stayed, 1 = Exited)')
plt.ylabel('Balance')
plt.xticks([0, 1], ['Stayed', 'Exited'])
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Gender', hue='Exited')
plt.title('Number of Customers by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Customers')
plt.legend(title='Exited')
plt.show()


plt.figure(figsize=(12, 10))
sns.pairplot(df, hue='Exited', vars=['Age', 'CreditScore', 'Balance'])
plt.title('Pair Plot of Key Features')
plt.show()


x=df.drop(['id','CustomerId','Surname','Exited'],axis=1)
y=df[['Exited']]


x=pd.get_dummies(x,drop_first=True)


def algo_test(x, y):
    models = [
        MultinomialNB(),
        BernoulliNB(),
        GaussianNB(),
        LogisticRegression(max_iter=1000),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        KNeighborsClassifier(),
        XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    ]

    algorithms = [
        "MultinomialNB", "BernoulliNB", "GaussianNB", "LogisticRegression",
        "DecisionTreeClassifier", "RandomForestClassifier",
        "AdaBoostClassifier", "GradientBoostingClassifier", "KNeighborsClassifier",
        "XGBClassifier"
    ]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    results = []

    for model, name in zip(models, algorithms):
        model.fit(x_train, y_train)
        predict = model.predict(x_test)

        cm = confusion_matrix(y_test, predict)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix for {name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

        report = classification_report(y_test, predict, output_dict=True)
        accuracy = report['accuracy']
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']

        metrics = {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score
        }
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='Accuracy', ascending=False)

    return results_df


algo_test(x, y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier()
model.fit(x_train, y_train)

predict = model.predict(x_test)

accuracy = accuracy_score(y_test, predict)
report = classification_report(y_test, predict)

print(f"Accuracy: {accuracy}, \nReport:\n {report}")


with open('bank_customer.pkl', 'wb') as file:
    pickle.dump(model, file)


test_df=pd.read_csv('/kaggle/input/playground-series-s4e1/test.csv')


test_df.head()


test_df.info()


null_values = test_df.isnull().sum()
null_values[null_values>0]


pred_x=test_df.drop(['id','CustomerId','Surname'],axis=1)


pred_x=pd.get_dummies(pred_x,drop_first=True)


predictions = model.predict(pred_x)


submision = pd.DataFrame()
submision['id'] = test_df['id']
submision['Exited'] = predictions


submision.head()


submision.to_csv('submission.csv', index=False)

