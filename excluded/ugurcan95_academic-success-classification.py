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
from sklearn.preprocessing import scale

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)


df=pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')


df.head()


df.shape


df.info()


df.isnull().sum()


plt.figure(figsize=(30,20))
corr = df.drop('id',axis=1).corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.3)
plt.title('Correlation Heatmap')
plt.show()


sns.countplot(x='Marital status', data=df)
plt.title('Count of Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='Admission grade', data=df)
plt.title('Admission Grade by Gender')
plt.xlabel('Gender')
plt.ylabel('Admission Grade')
plt.show()


plt.figure(figsize=(10, 6))
sns.violinplot(x='Marital status', y='Admission grade', data=df)
plt.title('Admission Grades by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Admission Grade')
plt.show()


mapping = {
    'Graduate': 1,
    'Dropout': 0,
    'Enrolled': 2
}

df['Target'] = df['Target'].map(mapping)


x=df.drop(['id','Target'],axis=1)
y=df[['Target']]


x_scaled=scale(x)
x=pd.DataFrame(x_scaled,columns=x.columns)


x.head()


def algo_test(x, y):
    models = [
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
        "BernoulliNB", "GaussianNB", "LogisticRegression",
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


with open('academic_success.pkl', 'wb') as file:
    pickle.dump(model, file)


test_df=pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')


test_df.head()


test_df.info()


null_values = test_df.isnull().sum()
null_values[null_values>0]


pred_x=test_df.drop(['id'],axis=1)


predictions = model.predict(pred_x)


submision = pd.DataFrame()
submision['id'] = test_df['id']
submision['Target'] = predictions


reverse_Target_mapping = {v: k for k, v in mapping.items()}
submision['Target'] = submision['Target'].map(reverse_Target_mapping)


submision.head()


submision.to_csv('submission.csv', index=False)

