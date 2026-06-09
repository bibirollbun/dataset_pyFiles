# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')


df.head()


# removing the unnecessery features
df.drop(columns=['id'],inplace=True)


df['rainfall'].value_counts().plot(kind='bar')


# Checking the distribution of input features
input_columns = df.drop(columns=['rainfall'])
for col in input_columns.columns:
    plt.figure(figsize=(6, 4))  # Optional: Adjust figure size
    sns.histplot(df[col], kde=True)  # kde=True adds a KDE curve
    plt.title(f'Distribution of {col}')
    plt.show()


df.corr()


# normalized the distribution using yeo-jhonson
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()


scaled_input_columns = pt.fit_transform(input_columns)


scaled_df = pd.DataFrame(scaled_input_columns,columns=input_columns.columns)


x = scaled_df
y = df['rainfall']


from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Dictionary of classifiers (excluding those affected by negative values)
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "GaussianNB": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Bagging": BaggingClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "MLP (Neural Network)": MLPClassifier(max_iter=500),
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0)
}

# Store results
results = []

# Run each classifier
for name, clf in tqdm(classifiers.items(), desc="Running models"):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results.append([name, acc, prec, f1])

# Convert results to DataFrame
df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "F1-score"])

# Sort by F1-score
df_results = df_results.sort_values(by="F1-score", ascending=False)

# Display results
print(df_results)



df_results




