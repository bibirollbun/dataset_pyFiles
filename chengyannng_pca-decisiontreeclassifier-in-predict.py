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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier



data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
data.head()


data.shape


data.info()


data.isna().sum()


data.duplicated().sum()


data['winddirection_x'] = round(np.cos(np.radians(data['winddirection'])), 4)
data['winddirection_y'] = round(np.sin(np.radians(data['winddirection'])), 4)
data = data.drop(columns=['winddirection'])


data.describe().T


corr_matrix = data.drop(columns=['id', 'day', 'rainfall']).corr()
corr_matrix


X = data.drop(columns=['id', 'rainfall'])
y = data['rainfall']


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
dtc = DecisionTreeClassifier(random_state=42)
model = dtc.fit(X_train, y_train)
y_pred = model.predict(X_test)


confusion_matrix(y_test, y_pred)


print(classification_report(y_test, y_pred))


train_acc = []
test_acc = []
max_depth_range = range(1, 10) 

for depth in max_depth_range:
    dtc = DecisionTreeClassifier(random_state=42)
    model = dtc.fit(X_train, y_train)
    
    train_acc.append(accuracy_score(y_train, model.predict(X_train)))
    test_acc.append(accuracy_score(y_test, model.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(max_depth_range, train_acc, marker='o', label="Train Accuracy", linestyle='-')
plt.plot(max_depth_range, test_acc, marker='s', label="Test Accuracy", linestyle='-')

plt.xlabel("Max Depth of Decision Tree")
plt.ylabel("Accuracy")
plt.title("Train & Test Accuracy Curve for Decision Tree")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.show()



param_grid = {
    'max_depth': [3, 5, 10, 15, 20], 
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    dtc, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)

grid_search.fit(X_train, y_train)

print("best_params:", grid_search.best_params_)
print("best_score:", grid_search.best_score_)

best_model = grid_search.best_estimator_

test_accuracy = best_model.score(X_test, y_test)
print("test_accuracy:", test_accuracy)



train_acc = []
test_acc = []
max_depth_range = range(1, 10)

for depth in max_depth_range:
    dtc = DecisionTreeClassifier(criterion='entropy',
                                 max_depth=5,
                                 min_samples_leaf=5,
                                 min_samples_split=2,
                                 random_state=42)
    model = dtc.fit(X_train, y_train)
    
    train_acc.append(accuracy_score(y_train, model.predict(X_train)))
    test_acc.append(accuracy_score(y_test, model.predict(X_test)))
plt.figure(figsize=(10, 6))
plt.plot(max_depth_range, train_acc, marker='o', label="Train Accuracy", linestyle='-')
plt.plot(max_depth_range, test_acc, marker='s', label="Test Accuracy", linestyle='-')

plt.xlabel("Max Depth of Decision Tree")
plt.ylabel("Accuracy")
plt.title("Train & Test Accuracy Curve for Decision Tree")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.show()



X = data.drop(columns=['id'])
y = data['rainfall']


scale = StandardScaler()
X_scaled = scale.fit_transform(X)


pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)


X_pca.shape


y.shape


print(pca.explained_variance_ratio_)


pca_df = pd.DataFrame(X_pca, index=data.index, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
plt.scatter(pca_df.PC1, pca_df.PC2, color=['r' if rain else 'b' for rain in data['rainfall']])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Scatter Plot: Red = Rain, Blue = No Rain")
plt.show()


X_train2, X_test2, y_train2, y_test2 = train_test_split(X_pca, y, random_state=42, test_size=0.2)
dtc2 = DecisionTreeClassifier(random_state=42)
model2 = dtc2.fit(X_train2, y_train2)
y_pred2 = model2.predict(X_test2)


confusion_matrix(y_test, y_pred)


print(classification_report(y_test, y_pred))


train_acc2 = []
test_acc2 = []
max_depth_range = range(1, 10) 

for depth in max_depth_range:
    dtc2 = DecisionTreeClassifier(random_state=42)
    model2 = dtc2.fit(X_train2, y_train2)
    
    train_acc2.append(accuracy_score(y_train2, model2.predict(X_train2)))
    test_acc2.append(accuracy_score(y_test2, model2.predict(X_test2)))

plt.figure(figsize=(10, 6))
plt.plot(max_depth_range, train_acc2, marker='o', label="Train Accuracy", linestyle='-')
plt.plot(max_depth_range, test_acc2, marker='s', label="Test Accuracy", linestyle='-')

plt.xlabel("Max Depth of Decision Tree")
plt.ylabel("Accuracy")
plt.title("Train & Test Accuracy Curve for Decision Tree")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.show()



param_grid = {
    'max_depth': [3, 5, 10, 15, 20], 
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid_search2 = GridSearchCV(
    dtc2, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)

grid_search2.fit(X_train2, y_train2)

print("best_params:", grid_search2.best_params_)
print("best_score:", grid_search2.best_score_)

best_model2 = grid_search2.best_estimator_

test_accuracy2 = best_model2.score(X_test2, y_test2)
print("test_accuracy:", test_accuracy2)



train_acc2 = []
test_acc2 = []
max_depth_range = range(1, 10) 

for depth in max_depth_range:
    dtc2 = DecisionTreeClassifier(criterion='gini',
                                 max_depth=5,
                                 min_samples_leaf=5,
                                 min_samples_split=2,
                                 random_state=42)
    model2 = dtc2.fit(X_train2, y_train2)
    
    train_acc2.append(accuracy_score(y_train2, model2.predict(X_train2)))
    test_acc2.append(accuracy_score(y_test2, model2.predict(X_test2)))

plt.figure(figsize=(10, 6))
plt.plot(max_depth_range, train_acc2, marker='o', label="Train Accuracy", linestyle='-')
plt.plot(max_depth_range, test_acc2, marker='s', label="Test Accuracy", linestyle='-')

plt.xlabel("Max Depth of Decision Tree")
plt.ylabel("Accuracy")
plt.title("Train & Test Accuracy Curve for Decision Tree")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.show()



test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
test_data.head()


test_data.info()


test_data[test_data['winddirection'].isna()]


wind_counts = test_data['winddirection'].value_counts()
plt.bar(wind_counts.index, wind_counts.values, color='b')
plt.show()


day_of_id_2707 = test_data.loc[test_data['id'] == 2707, 'day'].values[0]


day_wind_avg = test_data.loc[test_data['day'] == day_of_id_2707, 'winddirection'].mean()


test_data.loc[(test_data['id'] == 2707) & (test_data['winddirection'].isna()), 'winddirection'] = day_wind_avg


test_data['winddirection_x'] = round(np.cos(np.radians(test_data['winddirection'])), 4)
test_data['winddirection_y'] = round(np.sin(np.radians(test_data['winddirection'])), 4)
test_data = test_data.drop(columns=['winddirection'])


test_data.isna().sum()


test_data.shape


test_scaled = scale.fit_transform(test_data.drop(columns=['id']))


pca2 = PCA(n_components=3)
test_pca = pca2.fit_transform(test_scaled)


test_pca.shape


print(pca2.explained_variance_ratio_)


rainfall_probabilities = best_model2.predict_proba(test_pca)
rainfall_prob = rainfall_probabilities[:, 1]
rainfall_prob = np.round(rainfall_prob, 1)



result_df = pd.DataFrame({
    'id': test_data['id'],  
    'rainfall': rainfall_prob
})
result_df.head()


result_df.to_csv('submission.csv', index=False)

