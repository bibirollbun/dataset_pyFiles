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
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")


df.info()


df.describe()


df.head().T


# @title Stage_fear vs Drained_after_socializing

plt.subplots(figsize=(8, 8))
df_2dhist = pd.DataFrame({
    x_label: grp['Drained_after_socializing'].value_counts()
    for x_label, grp in df.groupby('Stage_fear')
})
sns.heatmap(df_2dhist, cmap='viridis')
plt.xlabel('Stage_fear')
_ = plt.ylabel('Drained_after_socializing')


# @title Going_outside vs Friends_circle_size

df.plot(kind='scatter', x='Going_outside', y='Friends_circle_size', s=32, alpha=.8);


# @title Time_spent_Alone vs Social_event_attendance

df.plot(kind='scatter', x='Time_spent_Alone', y='Social_event_attendance', s=32, alpha=.8);


# @title Drained_after_socializing

df.groupby('Drained_after_socializing').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'));


# @title Stage_fear
df.groupby('Stage_fear').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'));


# @title Social_event_attendance
df['Social_event_attendance'].plot(kind='hist', bins=20, title='Social_event_attendance');


# @title Time_spent_Alone

df['Time_spent_Alone'].plot(kind='hist', bins=20, title='Time_spent_Alone');


df.isna().sum()


df.head().T


X = df.drop('Personality', axis = 1)
y = df['Personality']


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_col = X.select_dtypes(include = ['int64', 'float64']).columns
cat_col = X.select_dtypes(exclude = ['object', 'category']).columns

num_features = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('scaler', StandardScaler())
])

cat_features = Pipeline([
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_features, num_col),
    ('cat', cat_features, cat_col)
])


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model.fit(X_train, y_train)
y_preds = model.predict(X_test)


from sklearn.metrics import  accuracy_score
accuracy_score(y_test, y_preds)


from sklearn.model_selection import GridSearchCV
param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear']
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)


print("Best parameters: ", grid_search.best_params_)
print("Best CV accuracy: ", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_preds = best_model.predict(X_test)
print("Test accuracy: ", accuracy_score(y_test, y_preds))


df.head()


df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")



df_test.head()


test_preds = best_model.predict(df_test)


final = pd.DataFrame()
final['id'] = df_test.id
final['Personality'] = test_preds

print("Submission file is ready to upload")




