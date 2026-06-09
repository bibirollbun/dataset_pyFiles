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


train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


for col in train_df.select_dtypes(include='float64').columns:
    train_df[col] = train_df[col].astype('float32')
for col in train_df.select_dtypes(include='int64').columns:
    train_df[col] = train_df[col].astype('int32')


rmv = ["Listening_Time_minutes"]
features = [c for c in train_df.columns if c not in rmv]
cats = [c for c in features if train_df[c].dtype == "object"]
nums = [c for c in features if c not in cats]

print(f"Features: {len(features)} (Categorical: {len(cats)} (Numerical: {len(nums)})")


train_df = train_df.copy()
test_df = test_df.copy()

print("Missing Values in train data:")
print(train_df.isnull().sum())

print("\nMissing Values in test data:")
print(test_df.isnull().sum())


train_df['Number_of_Ads'] = train_df['Number_of_Ads'].fillna(train_df['Number_of_Ads'].median())

train_df['Episode_Length_minutes'] = train_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median())
test_df['Episode_Length_minutes'] = test_df['Episode_Length_minutes'].fillna(test_df['Episode_Length_minutes'].median())

train_df['Guest_Popularity_percentage'] = train_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].median())
test_df['Guest_Popularity_percentage'] = test_df['Guest_Popularity_percentage'].fillna(test_df['Guest_Popularity_percentage'].median())


print(f"All the features: {features} \n")
print(f"Categorical columns: {cats} \n")
print(f"Numerical columns: {nums}")


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
import gc


gc.collect()

X_train, y_train = train_df[features], train_df[rmv]
X_test = test_df[features]

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder())
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, cats),
    ('num', numerical_transformer, nums)])

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1))])


model_pipeline.fit(X_train, y_train.values.ravel())


rmse_scorer = make_scorer(mean_squared_error, squared=False)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

gc.collect()

scores = cross_val_score(model_pipeline, X_train, y_train.values.ravel(), 
                         cv=kf, scoring=rmse_scorer, n_jobs=-1)

print("RMSE values for each fold:", scores)
print("Mean RMSE:", np.mean(scores))


test_preds_pipe = model_pipeline.predict(X_test)


sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
sub["Listening_Time_minutes"] = test_preds_pipe
sub.to_csv("submission.csv", index=False)

