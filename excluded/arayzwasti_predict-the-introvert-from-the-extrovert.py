# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import traceback
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from xgboost import XGBClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


df_test.shape


X = df_train.drop('Personality', axis = 1)
y = df_train['Personality']

print(y)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2, random_state = 42)  


leEn = LabelEncoder()
y_train_encoded = leEn.fit_transform(y_train)
y_test_encoded = leEn.transform(y_test)


numerical_category = [col for col in X.columns if X[col].dtypes in ['int64', 'float64']]
categorical_categories = [col for col in X.columns if X[col].dtypes == 'object']


numerical_feature_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy= 'median')),
    ('MMS', MinMaxScaler())
])


categorical_feature_pipeline = Pipeline(steps = [
    ('impute', SimpleImputer(strategy= 'most_frequent')),
    ('OHE', OneHotEncoder(sparse =False))
])


preprocessing = ColumnTransformer([
        ("numerical_imputer_pipeline", numerical_feature_pipeline, numerical_category),
        ("categorical_imputer_pipeline", categorical_feature_pipeline, categorical_categories)
    ],remainder='passthrough')


select_features = SelectKBest(score_func = mutual_info_classif, k = 10)


model = XGBClassifier() 


my_pipeline = Pipeline([
    ("processor", preprocessing),
    ("select_features", select_features),
    ("model", model)
])
my_pipeline


my_pipeline.fit(X_train, y_train_encoded)


try:
    test_preds = my_pipeline.predict(df_test)

    # Create submission
    submission = pd.DataFrame({
        "id": df_test['id'],
        "Personality": leEn.inverse_transform(test_preds)
    })
    submission.to_csv("/kaggle/working/submission.csv", index=False)
    print("submitted successfully")
except Exception:
    print("Error during prediction or submission creation:")
    traceback.print_exc()
    raise

