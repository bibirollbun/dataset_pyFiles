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
import numpy as np

train = '/kaggle/input/data-science-london-scikit-learn/train.csv'
test = '/kaggle/input/data-science-london-scikit-learn/test.csv'
train_lbl = '/kaggle/input/data-science-london-scikit-learn/trainLabels.csv'

# read the columns
column_df = pd.read_csv(train_lbl)

df = pd.read_csv(train)


column_df


df.head()


df = pd.concat([df, column_df],axis='columns')

df


columns = df.columns

print(columns)


import matplotlib.pyplot as plt
import seaborn as sns


fig, axes = plt.subplots()
plt.ylabel('Count')
plt.suptitle('Frequency of Target values')
axes = sns.barplot(x=df.groupby('1')['1'].count().sort_index().index, y=df.groupby('1')['1'].count().sort_index().values)
plt.show()


df.describe()


len(columns[:-1])

independent_columns = columns[:-1]
available_layout = [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8)]

def get_figure_axes(num,available_layout):
    
    if ( available_layout[len(available_layout)-1][0] * available_layout[len(available_layout)-1][1] ) < num:
        return None
    
    if ( available_layout[0][0] * available_layout[0][1] ) >= num:
        return available_layout[0]

    for i in range(len(available_layout)):
        if ( available_layout[i][0] * available_layout[i][1] ) >= num:
            return available_layout[i]
    
    return None

ax = get_figure_axes(len(independent_columns), available_layout)
print(ax)
fig, axe = plt.subplots(ax[0], ax[1], figsize=(18,18), sharex=True, sharey=True)

rows, cols = ax
def plot_histogram(df, m, n, columns, bins=20):
    count = 1
    for i in range(m):
        for j in range(n):
            if count > len(columns):
                break
    
            plt.subplot(rows, cols, count)
            df[independent_columns[count-1]].plot(kind='hist', edgecolor='black', bins=30)
            count += 1
    
    plt.suptitle('Distribution of Independent Variable Values', font='30')
    plt.tight_layout()
    plt.show()

plot_histogram(df, rows, cols, independent_columns)


# classification prob -> log
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

# split into X and y
X = df[independent_columns]
y = df[columns[-1]]

def stratified_train_test(X, y, model, cv=5):

    # create stratified Kfold
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    accuracy_scores = []

    # split train-test dataset
    # 2. iterate through the splits
    try:
        for train_idx, test_idx in kfold.split(X,y):
            X_train, X_val = X.iloc[train_idx, :], X.iloc[test_idx, :]
            y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]
    
            # 3. train the model
            model.fit(X_train, y_train)
    
            # 4. calculate the accuracy score
    
            # 5. add the accuracy to list of scores
            accuracy_scores.append(model.score(X_val, y_val))
    
        print('Accuracy over iterations: ', accuracy_scores)
        # 6. compute mean score
        print(sum(accuracy_scores)/len(accuracy_scores))
    except Exception as ex:
        print('Exception Thrown')
        print('Message:', ex)

stratified_train_test(X, y, LogisticRegression(),5)


from sklearn.model_selection import cross_val_score

def stratified_train_test2(X, y, model, cv=5):

    # create stratified Kfold
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    accuracy_scores = []

    # split train-test dataset
    # 2. iterate through the splits
    try:
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=kfold, n_jobs=-1)
        print(f"Scores: {scores}")
        print(f"Mean Scores: {np.mean(scores)}")

        return np.mean(scores)
    except Exception as ex:
        print('Exception Thrown')
        print('Message:', ex)
        return 0


mean_scores = []

for i in range(2, 11):
    print("Cross Validation", i)
    mean_scores.append(stratified_train_test2(X, y, LogisticRegression(),i))

print(f"Max Mean Score: {np.max(mean_scores)}")
print(f"Best performing CV: {np.argmax(mean_scores)}")


from xgboost import XGBClassifier

best_cv = np.argmax(mean_scores) + 2

clf = XGBClassifier()

stratified_train_test(X,y,clf,best_cv)


X_new = X[:3]
y_new = y[:3]


predictions = clf.predict(X_new)

print('Predictions: ', predictions)
print('Actual: ', y_new.values)


plt.suptitle('Correlation between independent variables')
sns.heatmap(X.corr(), annot=True)
plt.show()


pd.plotting.scatter_matrix(df, figsize=(20,20), grid=True, marker='o',)
plt.show()


X_test = pd.read_csv(test)


X_test.head()


X_test.index.values


predictions = clf.predict(X_test.values)

output = pd.DataFrame({'Id':X_test.index.values, 'Target':predictions})

output.to_csv('submissions.csv', index=False)




