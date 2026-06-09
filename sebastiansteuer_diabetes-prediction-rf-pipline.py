# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


train.head()


train.columns


train.info()


train.describe()


train = train.drop("id", axis=1)


num_features = train.select_dtypes(["int64", "float64"]).columns
cat_features = train.select_dtypes("object").columns


train_numeric = train[num_features]


plt.figure(figsize=(15,12))
sns.heatmap(train_numeric.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.show()


for cat in cat_features:
  print(f"Value counts for {cat}:")
  print(train[cat].value_counts())
  plt.figure(figsize=(10, 6))
  sns.countplot(data=train, x=cat, hue='diagnosed_diabetes', palette='viridis')
  plt.title(f'Distribution of Diagnosed Diabetes by {cat}')
  plt.xlabel(cat)
  plt.ylabel('Count')
  plt.xticks(rotation=45, ha='right')
  plt.legend(title='Diagnosed Diabetes')
  plt.tight_layout()
  plt.show()


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

cat_ord_features = ['education_level','income_level','smoking_status']

X = train.drop('diagnosed_diabetes', axis=1)
y= train['diagnosed_diabetes']

# Redefine num_features and cat_features based on X
num_features_X = X.select_dtypes(["int64", "float64"]).columns
cat_features_X = X.select_dtypes("object").columns
cat_oh_features = cat_features_X.difference(cat_ord_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features_X),
        ('cat_oh', OneHotEncoder(handle_unknown='ignore'), cat_oh_features),
        ('cat_ord_edu', OrdinalEncoder(categories=[["No formal","Highschool","Graduate","Postgraduate"]]), ['education_level']),
        ('cat_ord_inc', OrdinalEncoder(categories=[["Low","Lower-Middle","Middle","Upper-Middle","High"]]), ['income_level']),
        ('cat_ord_smo', OrdinalEncoder(categories=[["Never","Former","Current"]]), ['smoking_status'])
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=200, max_depth=10))
])

param_grid = {
    'model__n_estimators': [100,200,300],
    'model__max_depth': [10,20,None],
    'model__max_features': ['sqrt','log2'],
    'model__min_samples_split': [2,5]
}



pipeline.fit(X_train, y_train)
print('Accuracy of test: {:.6f}'.format(pipeline.score(X_test, y_test)))


X_test_final = test.drop(['id'], axis=1)

predictions_test = pipeline.predict_proba(X_test_final)

data = np.ndarray((len(test), 2))
data[:,0] = test['id']
data[:,1] = predictions_test[:,1]
submission = pd.DataFrame(data, columns=['id', 'diagnosed_diabetes'])
submission['id'] = submission['id'].astype(int)

path = "/kaggle/working/"
submission.to_csv(path + "submission.csv", index=False)

