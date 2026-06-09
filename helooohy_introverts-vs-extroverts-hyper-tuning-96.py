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


df=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')


test=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


df.head()


test.head()


print("Train Data Prepare\n")

print(df.info())

print(df.isnull().sum)

print(df.duplicated().sum())

print(df.describe())


print("Test Data Preprare")


print(test.info())

print(test.isnull().sum)

print(test.duplicated().sum())

print(test.describe())



for col in df.select_dtypes(include='number').columns:
    plt.figure(figsize=(12,6))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Histogram of {col}')
    plt.show()


from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import GridSearchCV


num_col = df.select_dtypes(include='number').columns

imputer = SimpleImputer(strategy='median')

df[num_col] = imputer.fit_transform(df[num_col])


from sklearn.impute import SimpleImputer

cat_cols = df.select_dtypes(include='object').columns

imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = imputer.fit_transform(df[cat_cols])



num_col = test.select_dtypes(include='number').columns

imputer = SimpleImputer(strategy='median')

test[num_col] = imputer.fit_transform(test[num_col])


from sklearn.impute import SimpleImputer

cat_cols = test.select_dtypes(include='object').columns

imputer = SimpleImputer(strategy='most_frequent')
test[cat_cols] = imputer.fit_transform(test[cat_cols])



df.drop('id', axis=1, inplace=True)


 df.select_dtypes(include='object').columns


le=LabelEncoder()

df['Personality']=le.fit_transform(df['Personality'])


for col in df.select_dtypes(include='object').columns:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True).astype(int)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=[col])
    


for col in test.select_dtypes(include='object').columns:
    dummies = pd.get_dummies(test[col], prefix=col, drop_first=True).astype(int)
    test = pd.concat([test, dummies], axis=1)
    test = test.drop(columns=[col])


x=df.drop('Personality', axis=1)
y=df['Personality']

X_test=test.drop('id', axis=1)


x=df.drop('Personality', axis=1)
y=df['Personality']

X_test=test.drop('id', axis=1)


x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=42)


sc=StandardScaler()

x_train_sc=sc.fit_transform(x_train)
x_val_sc=sc.fit_transform(x_test)
x_test_sc=sc.fit_transform(X_test)


param_grid = {
    'n_estimators': [20, 60, 100,150],
    'max_depth': [None, 2, 8],
    'max_samples':[0.5, 0.75, 1.0],
    'max_features':[0.2, 0.6, 1.0],
    'bootstrap': [True, False]
}


rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid,
                           cv=2,
                           n_jobs=-1,
                           scoring='accur


grid_search.fit(x_train_sc, y_train)


print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_predd = best_model.predict(x_test_sc)


submission = pd.DataFrame({
    'id': test['id'],
    'Personality': y_pred
})

submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved with tuned model!")





