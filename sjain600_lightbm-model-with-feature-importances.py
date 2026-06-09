# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
%load_ext cudf.pandas
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', 500)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')


train.shape


test.shape


train.sample(10)


train.isnull().sum()


test.isnull().sum()


train['Policy Start Date'] = pd.to_datetime(train['Policy Start Date'])
train['year'] = train['Policy Start Date'].dt.year.astype('float32')
train['month'] = train['Policy Start Date'].dt.month.astype('float32')
train['day'] = train['Policy Start Date'].dt.day.astype('float32')
train['day_of_week'] = train['Policy Start Date'].dt.day_of_week.astype('float32')
train['seconds'] = (train['Policy Start Date'].astype(int) // 10**9).astype('float32')
train.drop('Policy Start Date', axis=1, inplace=True)


test['Policy Start Date'] = pd.to_datetime(test['Policy Start Date'])
test['year'] = test['Policy Start Date'].dt.year.astype('float32')
test['month'] = test['Policy Start Date'].dt.month.astype('float32')
test['day'] = test['Policy Start Date'].dt.day.astype('float32')
test['day_of_week'] = test['Policy Start Date'].dt.day_of_week.astype('float32')
test['seconds'] = (test['Policy Start Date'].astype(int) // 10**9).astype('float32')
test.drop('Policy Start Date', axis=1, inplace=True)


train.info()


cat_cols = train.select_dtypes(include='object').columns
train[cat_cols] = train[cat_cols].astype('category')
test[cat_cols] = test[cat_cols].astype('category')


from sklearn.impute import SimpleImputer

simple = SimpleImputer(strategy='most_frequent')
cat_cols = train.select_dtypes(include='category').columns
train[cat_cols] = simple.fit_transform(train[cat_cols])
test[cat_cols] = simple.transform(test[cat_cols])


from cuml.preprocessing import TargetEncoder

te = TargetEncoder(n_folds=15, smooth=20, split_method='random', stat='mean', seed=340)
for col in cat_cols:
    te = TargetEncoder().fit(train[col], train['Premium Amount'])
    train[col] = te.transform(train[col])
    test[col] = te.transform(test[col])


simple = SimpleImputer(strategy='median')
num_cols = test.select_dtypes(exclude='object').columns
train[num_cols] = simple.fit_transform(train[num_cols])
test[num_cols] = simple.transform(test[num_cols])


features = ['Age', 'Gender', 'Annual Income', 'Marital Status',
       'Number of Dependents', 'Education Level', 'Occupation', 'Health Score',
       'Location', 'Policy Type', 'Previous Claims', 'Vehicle Age',
       'Credit Score', 'Insurance Duration', 'Customer Feedback',
       'Smoking Status', 'Exercise Frequency', 'Property Type']


def add_features(df):
    for i, col1 in enumerate(features):
        for col2 in (features[i+1:]):
            new_col = f'{col1}_{col2}'
            df[new_col] = train[col1] * train[col2]
    return df
    


sns.boxplot(data=train, y='Premium Amount', color='orange')


sns.histplot(x='Annual Income', data=train, bins=35)
plt.title('Distribution plot of Annual Income')


from sklearn.model_selection import KFold

X = train.copy()
y = X.pop('Premium Amount')
X = add_features(X)
y = np.log(y)

X_test = test.copy()
X_test = add_features(X_test)

kf = KFold(n_splits=10, random_state=340, shuffle=True)
for train_index, valid_index in kf.split(X, y):
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    


from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error

lgm = LGBMRegressor(
    n_estimators=300, 
    learning_rate=0.012057, 
    min_split_gain= 3, 
    max_depth=11, 
    reg_lambda = 0.70115,  
    random_state=340,
    feature_fraction=0.8,
    n_jobs= -1,
    device_type='gpu',
    verbose=-1
).fit(X_train, y_train)

preds_light = lgm.predict(X_valid)
print(f'RMSLE: {np.sqrt(mean_squared_log_error(preds_light, y_valid)):,.5f}')


importances = lgm.feature_importances_
columns = X.columns

threshold = np.quantile(importances, 0.5)
selected_features = columns[importances > threshold]
top_X = X[selected_features]
print(selected_features)





from sklearn.model_selection import cross_val_score

kf = KFold(n_splits=5, random_state=340, shuffle=True)
top_model_score = -1 * cross_val_score(lgm, top_X, y, cv=kf, scoring='neg_mean_squared_log_error').mean()

print(f'{np.sqrt(top_model_score)}')


top_X_train = X_train[selected_features]
top_X_test = X_test[selected_features]


lgm.fit(top_X_train, y_train)

test_preds = lgm.predict(top_X_test)
sub = pd.read_csv('/kaggle/input/playground-series-s4e12/sample_submission.csv')
sub['Premium Amount'] = np.exp(test_preds) - 1
sub.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
sub.head()

