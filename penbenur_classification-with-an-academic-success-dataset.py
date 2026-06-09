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
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


df=pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')


df.head()


df.shape


df.info()


df.describe()


df.tail()


df.corr(numeric_only=True)


df.isnull().sum()


mapping = {
    'Graduate': 0,
    'Enrolled': 1,
    'Dropout': 2
}
df['Target'] = df['Target'].map(mapping)


df.corr(numeric_only=True)


df=df.drop('id', axis=1)


# Drop columns with less effect on target
df=df.drop(columns=["Previous qualification", "Nacionality", "Mother's qualification", "Father's qualification", "Mother's occupation",	"Father's occupation", 
"Educational special needs", "International", "Unemployment rate",	"Inflation rate"], axis=1)


df.head()


def drop_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]


# Dropping outliers using IQR
df = drop_outliers_iqr(df)


# Calculate correlation matrix
correlation_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()


# Plot distribution of cover types
plt.figure(figsize=(10, 6))
df['Target'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of Target')
plt.xlabel('Target')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


# Bar plot of marital status by educational outcome
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Marital status', hue='Target')
plt.title('Marital Status vs. Educational Outcome')
plt.xlabel('Marital Status (0: Single, 1: Married)')
plt.ylabel('Count')
plt.legend(title='Target', labels=['Dropout', 'Enrolled', 'Graduate'])
plt.show()


# Distribution of the "Age at enrollment"
# Set the style of seaborn
sns.set(style="whitegrid")

# Create a histogram and a KDE plot
plt.figure(figsize=(10, 6))
sns.histplot(df['Age at enrollment'], bins=10, kde=True, color='blue', stat='density', alpha=0.5)

# Add titles and labels
plt.title('Distribution of Age at Enrollment')
plt.xlabel('Age at Enrollment')
plt.ylabel('Density')
plt.xlim(df['Age at enrollment'].min() - 1, df['Age at enrollment'].max() + 1)

# Show the plot
plt.show()


x=df.drop(['Target'],axis=1)
y=df[['Target']]


scaler = StandardScaler()
x=scaler.fit_transform(x)


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

    labels = sorted(y["Target"].unique())

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
gb = GradientBoostingClassifier()
model=gb.fit(x_train, y_train)


# save the model
import joblib
joblib.dump(model, 'model.pkl')


# save the scaler
joblib.dump(scaler, 'scaler.pkl')




