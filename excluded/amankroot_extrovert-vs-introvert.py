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


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import logging
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger().setLevel(logging.ERROR)
%matplotlib inline


df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv', index_col='id')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv', index_col='id')


X = df.drop('Personality', axis=1)
y = df['Personality']


# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)


# Identify column types
num_cols_mean = ['Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
num_col_median = ['Time_spent_Alone']
cat_cols = ['Stage_fear', 'Drained_after_socializing']


# Preprocessing pipelines
mean_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

median_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Combine all preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('mean_num', mean_pipeline, num_cols_mean),
    ('median_num', median_pipeline, num_col_median),
    ('cat', cat_pipeline, cat_cols)
])


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Fit model
pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))


scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy: %.4f ± %.4f" % (scores.mean(), scores.std()))


test_predictions = pipeline.predict(df_test)
decoded_preds = le.inverse_transform(test_predictions)


submission = pd.DataFrame({'Personality': decoded_preds}, index=df_test.index)
submission.to_csv('submission.csv')
submission.head()




