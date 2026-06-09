# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


df_train.head()


df_train.value_counts("Soil Type")


df_train.value_counts("Crop Type")


df_test.head()


df_train.isnull().sum()


X = df_train.drop("Fertilizer Name", axis = 1)
y = df_train["Fertilizer Name"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


X_train.info()


numerical_col_feature = [col for col in X_train.columns if X_train[col].dtype == 'int64']
categorical_col_feature = [col for col in X_train.columns if X_train[col].dtype == 'object']


num_pipeline_features = Pipeline([
    ('MMS', MinMaxScaler())
])
categories_pipeline_features = Pipeline([
    ('OHE', OneHotEncoder(sparse_output = False, handle_unknown = "ignore"))
])
categories_pipeline_features


preprocessing = ColumnTransformer([
    ("num_pipeline_features", num_pipeline_features, numerical_col_feature),
    ("categories_pipeline_features", categories_pipeline_features, categorical_col_feature)
])


selection_feature = SelectKBest(score_func = chi2, k=10)


model = LogisticRegression(random_state=42)


my_prediction = Pipeline([
    ("preprocessing", preprocessing),
    ("selection_feature", selection_feature),
    ("model", model)
])


X_train


y_train


le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)


my_prediction.fit(X_train, y_train_encoded)


scores = cross_val_score(my_prediction, X_train, y_train, cv=5, scoring="accuracy")
print("Accuracy:", scores.mean())


test_preds = my_prediction.predict(df_test)
print(test_preds)


test_preds = le.inverse_transform(test_preds)

submission = pd.DataFrame({
    "id": df_test["id"],
    "Fertilizer Name": test_preds
})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("submission.csv file is ready!")

