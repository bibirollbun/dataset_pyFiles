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

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix,classification_report

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)


df=pd.read_csv('/kaggle/input/playground-series-s3e23/train.csv')


df.head()


df.shape


df.info()


df.isnull().sum()


plt.figure(figsize=(17,10))
corr = df.drop('id',axis=1).corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


for col in df.columns[1:]:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()


x=df.drop(['id','defects'],axis=1)
y=df[['defects']]


def algo_test(x, y):
    mnb = MultinomialNB()
    bnb = BernoulliNB()
    gnb = GaussianNB()
    lr = LogisticRegression()
    dtc = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    abc = AdaBoostClassifier()
    gbc = GradientBoostingClassifier()
    kn = KNeighborsClassifier()

    models = [mnb, bnb, gnb, lr, dtc, rfc, abc, gbc, kn]
    algorithms = [
        "MultinomialNB", "BernoulliNB", "GaussianNB", "LogisticRegression",
        "DecisionTreeClassifier", "RandomForestClassifier",
        "AdaBoostClassifier", "GradientBoostingClassifier", "KNeighborsClassifier"
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

        # Extract metrics from classification_report
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

    return results_df


algo_test(x, y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier()
model.fit(x_train, y_train)

predict = model.predict(x_test)

accuracy = accuracy_score(y_test, predict)
report = classification_report(y_test, predict)

print(f"Accuracy: {accuracy}, \nReport:\n {report}")


with open('defect_prediction.pkl', 'wb') as file:
    pickle.dump(model, file)


test_df=pd.read_csv('/kaggle/input/playground-series-s3e23/test.csv')


test_df.head()


pred_x=test_df.drop(['id'],axis=1)


predictions = model.predict(pred_x)


submision = pd.DataFrame()
submision['id'] = test_df['id']
submision['defects'] = predictions


submision.to_csv('submission.csv', index=False)

