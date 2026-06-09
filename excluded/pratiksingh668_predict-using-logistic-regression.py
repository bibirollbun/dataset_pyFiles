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


import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df.head()


test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
test.head()


y = df['Personality']
X = df.drop(['id', 'Personality'], axis=1)

# all from numerical data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y,train_size=0.8, test_size=0.2,random_state=0)



categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols=X.select_dtypes(include=['int','float']).columns.tolist()
print(categorical_cols,numerical_cols)


from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='mean')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = LogisticRegression()
# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_valid = le.transform(y_valid)


# Preprocessing of training data, fit model 
clf.fit(X_train_full, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(X_valid_full)

print('Accuracy:', accuracy_score(y_valid, preds))


final_predictions = clf.predict(test)
final_predictions


pred=le.inverse_transform(final_predictions)
pred


output = pd.DataFrame({'id': test.id, 'Personalities': pred})
output.to_csv('submission_Introvert.csv', index=False)
print("successfully saved!")


sub=pd.read_csv("submission_Introvert.csv")
sub

