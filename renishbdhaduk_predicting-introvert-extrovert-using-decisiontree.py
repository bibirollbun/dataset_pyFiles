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

warnings.filterwarnings('ignore')


df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test= pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


df.head()


test.head()


sub.head()


df.info()


df.describe()


df.isnull().mean() * 100


test.isnull().mean() * 100 




df.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)


numeric_column = ['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']
categoric_column = ['Stage_fear','Drained_after_socializing']
target_column = 'Personality'


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(style='whitegrid')
df.head()


plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Personality', palette='pastel')
plt.title('Target Class Distribution: Personality Types')
plt.xticks(rotation=45)
plt.show()


for col in numeric_column:
    plt.figure(figsize=(8, 4))
    sns.kdeplot(data=df, x=col, hue='Personality', fill=True)
    plt.title(f'Distribution of {col} by Personality Type')
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(10,8))
sns.heatmap(df[numeric_column].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Numeric Features')
plt.show()


for col in numeric_column:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='Personality', y=col, data=df, palette='Set2')
    plt.title(f'Boxplot of {col} by Personality Type')
    plt.tight_layout()
    plt.show()


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler 
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score


# x_train,x_test,y_train,y_test = train_test_split(df.drop('Personality',axis=1),df['Personality'],test_size=0.3,random_state=42)

x_train = df.drop(columns=["Personality"])
y_train = df["Personality"]
x_test = test.copy()

num = Pipeline([
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])

cat = Pipeline([
    ('imputer',SimpleImputer(strategy = 'most_frequent')),
    ('encoding',OneHotEncoder(drop='first',sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num',num,numeric_column),
    ('cat',cat,categoric_column)
],remainder='passthrough')

pipe = Pipeline([
    ('preprocessor',preprocessor),
    ('model',DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42,min_samples_leaf =1,criterion = 'gini'))
])


pipe.fit(x_train,y_train)


y_pred= pipe.predict(x_test)


# accuracy_score(y_test,y_pred)


param_grid = {
    # Numerical imputer strategy
    'preprocessor__num__imputer__strategy': ['mean', 'median'],

    # Categorical imputer strategy
    'preprocessor__cat__imputer__strategy': ['most_frequent', 'constant'],

    # Decision Tree hyperparameters
    'model__max_depth': [3, 5, 10, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(pipe, param_grid, cv=10, n_jobs=-1, scoring="accuracy")
grid_search.fit(x_train, y_train)


y_pred_grid = grid_search.predict(x_test)


# accuracy_score(y_test,y_pred_grid)


print(f"Best params: {grid_search.best_params_}")
# print(f"\nTest accuracy: {accuracy_score * 100:.4f}")


submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
submission['Personality'] = y_pred_grid
submission.to_csv("final_submission.csv", index=False)



from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipe, x_train, y_train, cv=5, scoring='accuracy')
print("CV Accuracy:", scores.mean())




