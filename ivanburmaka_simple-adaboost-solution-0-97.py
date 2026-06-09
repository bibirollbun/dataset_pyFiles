# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


X_raw = pd.read_csv('/kaggle/input/breast-cancer-detection/train.csv')


X_raw['diagnosis'].value_counts()


X_raw.describe()


X_raw = X_raw.drop('Unnamed: 32', axis=1)
X_raw = X_raw.drop('id', axis=1)


X_raw.dtypes


X_raw['diagnosis'].unique()


from sklearn.preprocessing import  OrdinalEncoder
enc = OrdinalEncoder()
X_raw['diagnosis_e'] = enc.fit_transform(X_raw[['diagnosis']])
X_raw[['diagnosis', 'diagnosis_e']].head()


corr = X_raw.select_dtypes(include=['float64', 'int64']).corr().round(2)
#px.imshow(corr, text_auto=True).show() #plotly not showing image on Kaggle
#So using seaborn
sns.heatmap(corr,  linewidth=.5)



X_train = X_raw.drop(['diagnosis', 'diagnosis_e'], axis=1)
y_train = X_raw['diagnosis']


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
clf = RandomForestClassifier(max_depth=2, random_state=42)
cv_results = cross_validate(clf, X_train, y_train, cv=4)
cv_results



from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [   150, 200, 300],
    'learning_rate': [2, 1, 0.5 ],
    'estimator__max_depth': [None, 1,  2, 3]
   # 'base_estimator__min_samples_split': [2, 5, 10]
}
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
ada_clf = AdaBoostClassifier(estimator=tree_clf, n_estimators=50, learning_rate=0.5, random_state=42)
grid_search = GridSearchCV(ada_clf, param_grid, cv=4, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
ada_clf = grid_search.best_estimator_
cv_results = cross_validate(ada_clf, X_train, y_train, cv=4)
cv_results


X_test_full = pd.read_csv('/kaggle/input/breast-cancer-detection/test.csv')
X_test = X_test_full.drop('Unnamed: 32', axis=1)
X_test = X_test.drop('id', axis=1)


ada_clf.fit(X_train, y_train)
pred = ada_clf.predict(X_test)
submission = pd.DataFrame({
    'id': X_test_full['id'],
    'diagnosis': pred
})
submission.to_csv('/kaggle/working/submission.csv', index=False)

