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


train=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
train.head()


test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


train=train.drop("id",axis=1)


train["Stage_fear"].unique()


train.isnull().sum()


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

y = train["Personality"]
x = train.drop("Personality", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

num_cols = list(x_train.select_dtypes(exclude="object").columns)
cat_cols = list(x_train.select_dtypes(include="object").columns)

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])


model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=150, max_depth=4))
])

model.fit(x_train, y_train)


print(model.score(x_train,y_train))
print(model.score(x_test,y_test))


pred=model.predict(test)


submission=pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
submission.head()


submission["Personality"]=pred

submission.to_csv("submission.csv", index=False)

