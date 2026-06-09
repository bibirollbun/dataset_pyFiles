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


train_data = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
train_data.head()


test_data = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
test_data.head()


from sklearn.ensemble import RandomForestClassifier

y = train_data["rainfall"]

features = ["pressure", "temparature", "humidity", "cloud", "sunshine", "windspeed"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'id': test_data.id, 'Rainfall': predictions})
output.to_csv('submission.csv', index=False)

output_file = pd.read_csv("/kaggle/working/submission.csv")
output_file.head()




from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print("Accuracy is: ", accuracy)


from sklearn.metrics import classification_report

print(classification_report(y_val, y_pred))


from sklearn.metrics import roc_auc_score

y_prob = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_prob)

print("AUC-ROC:", auc)


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Rain", "Rain"], yticklabels=["No Rain", "Rain"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion matrix")
plt.show()


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=1), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)

print("Best parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

tree = model.estimators_[0]

plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=X.columns, class_names=["No Rain", "Rain"], filled=True, rounded=True, fontsize=8)
plt.show()


importances = model.feature_importances_

feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest Model")
plt.gca().invert_yaxis()
plt.show()

print(feature_importance_df)

