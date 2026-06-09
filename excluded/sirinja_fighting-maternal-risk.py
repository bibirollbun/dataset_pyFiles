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


train = pd.read_csv('/kaggle/input/mlolympiadbd2025/train.csv')
test = pd.read_csv('/kaggle/input/mlolympiadbd2025/test.csv')


train.info()


train.head(5)


train['RiskLevel'].unique()


train['RiskLevel'].value_counts('p')*100


train.describe()


import matplotlib.pyplot as plt
train.hist(figsize=(10,7))
plt.show()


import seaborn as sns

# Generate a correlation matrix for all numerical columns in the DataFrame
corr_matrix = train.drop('Usage',axis=1).corr()

# Create a heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='viridis')

# Set the title of the plot
plt.title('Correlation Matrix')

# Display the plot
plt.show()


# split data into features and target
X = train.drop(['Id','RiskLevel','Usage'], axis=1)
y = train['RiskLevel']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=10, stratify=y)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# train random forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
# Calculate the accuracy of the classifier
scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')

#fit the model
rf.fit(X, y)

# get feature importance scores
importance = rf.feature_importances_

# create a DataFrame to store feature importance scores
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importance})

# sort the features by importance score in descending order
feature_importance = pd.Series(rf.feature_importances_, index=X.columns)

# print the feature importance scores
print(feature_importance.sort_values(ascending=False))


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


# Define the models
models = [
    ('LR', LogisticRegression()),
    ('DT', DecisionTreeClassifier()),
    ('RF', RandomForestClassifier()),
    ('SVM', SVC()),
    ('Naive', GaussianNB()),
    ('KNN',KNeighborsClassifier(n_neighbors=2))
]

# Evaluate each model using cross-validation
results = []
names = []
for name, model in models:
    cv_results = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    
    print(f"{name}: Mean accuracy = {round(cv_results.mean(),3)}, Standard deviation = {round(cv_results.std(),3)}")   


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

clf = make_pipeline(StandardScaler(), RandomForestClassifier())


from sklearn.metrics import accuracy_score

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)
print('Final accuracy:', round(final_accuracy,3))


X_final = test.drop(['Id','Usage'],axis=1)


# pred = clf.predict(X_final)
# pred_df = pd.DataFrame({
#    "Id": test["Id"],
#    "RiskLevel": pd.Series(pred).map({
#        0: "Low Risk",
#        1: "Mid Risk",
#        2: "High Risk"
#    })
#})
## pred_df.to_csv('submission.csv',index=False)

