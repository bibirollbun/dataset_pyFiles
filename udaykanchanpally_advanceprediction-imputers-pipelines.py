# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd

# data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
data = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv')


train 



test


data


train.tail()


val = train.shape[0]

end = val+data.shape[0]
print(val,end,data.shape[0])


ids = [i for i in range(val,end)]
print(len(ids))


data["id"] = ids


data


total_train = pd.concat([train, data], ignore_index=True)


total_train


test.info()


test.info()





X = total_train.drop('Personality', axis=1)
y = total_train['Personality']




train_numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
train_categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()



from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

Num_preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

Cat_preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder()),
    ('scaler', StandardScaler())
])

Combine_preprocessor = ColumnTransformer([
    ('num', Num_preprocessor, train_numerical_cols),
    ('cat', Cat_preprocessor, train_categorical_cols)
])

Prepipe = Pipeline([
    ('preprocessor', Combine_preprocessor)
])


pipe = make_pipeline(Prepipe, LogisticRegression(max_iter=1000))  # Adjust max_iter as needed
pipe




pipe.fit(X, y)


y_pred = pipe.predict(test)






submission = pd.DataFrame({
    'id': test['id'],          # if there is an ID column
    'target': y_pred
})
submission.to_csv('submission.csv', index=False)

