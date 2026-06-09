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


TRAIN_PATH = r"/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT.csv"
TEST_PATH  = r"/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT_UNK.csv"


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import skew, kurtosis
import re
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


def clean_col_names(df):
    cols = df.columns
    new_cols = []
    for col in cols:
        new_col = re.sub(r'[^A-Za-z0-9_]+', '', col)
        new_cols.append(new_col)
    df.columns = new_cols
    return df

train_df = clean_col_names(train_df)
test_df = clean_col_names(test_df)
test_ids = test_df['LOCAL_IDENTIFIER']
train_df = train_df.rename(columns={'CORRUCYSTIC_DENSITY': 'target'})
train_df.dropna(subset=['target'], inplace=True)



all_df = pd.concat([train_df.drop('target', axis=1), test_df], axis=0)
categorical_features = all_df.select_dtypes(include=['object', 'category']).columns
numerical_features = all_df.select_dtypes(include=np.number).columns.drop('LOCAL_IDENTIFIER')

for col in numerical_features:
    all_df[col].fillna(all_df[col].median(), inplace=True)
for col in categorical_features:
    all_df[col].fillna('missing', inplace=True)


all_df['num_mean'] = all_df[numerical_features].mean(axis=1)
all_df['num_std'] = all_df[numerical_features].std(axis=1)
all_df['num_sum'] = all_df[numerical_features].sum(axis=1)
all_df['num_skew'] = all_df[numerical_features].skew(axis=1)
all_df['num_kurtosis'] = all_df[numerical_features].kurtosis(axis=1)


impfeatures = ['b6nl', '3Iy', '14WQ', 'ZZw3t', 'wnskR', '64']
impfeatures = [col for col in impfeatures if col in all_df.columns]
if len(impfeatures) > 1:
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly_features = poly.fit_transform(all_df[impfeatures])
    poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(impfeatures), index=all_df.index)
    all_df = pd.concat([all_df, poly_df], axis=1)


all_df = pd.get_dummies(all_df, columns=categorical_features, dummy_na=False)
X = all_df[:len(train_df)].drop('LOCAL_IDENTIFIER', axis=1)
y = train_df['target']
X_test = all_df[len(train_df):].drop('LOCAL_IDENTIFIER', axis=1)
X_aligned, X_test_aligned = X.align(X_test, join='inner', axis=1)



gbr_params = {
    'n_estimators': 1200,
    'learning_rate': 0.015,
    'max_depth': 5,
    'subsample': 0.75,
    'max_features': 'sqrt',
    'random_state': 42,
    'loss': 'squared_error',
    'verbose': 0
}



model = GradientBoostingRegressor(**gbr_params)
model.fit(X_aligned, y)
predictions = model.predict(X_test_aligned)


submission_df = pd.DataFrame({'LOCAL_IDENTIFIER': test_ids, 'CORRUCYSTIC_DENSITY': predictions})
submission_df.to_csv('submission.csv', index=False)
print(submission_df.head())

