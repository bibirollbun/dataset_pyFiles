# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/playground-series-s5e4'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings
warnings.filterwarnings('ignore')


!wget https://raw.githubusercontent.com/ezzaddeentru/used-cars-selling-price-estimating---case-study/refs/heads/main/reg_helper_functions.py


from reg_helper_functions import *


!wget https://raw.githubusercontent.com/ezzaddeentru/recipe-popularity-prediction/main/helper_functions.py



from helper_functions import *


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_path = '/kaggle/input/playground-series-s5e4/train.csv'
test_path = '/kaggle/input/playground-series-s5e4/test.csv'
sample_submission_path = '/kaggle/input/playground-series-s5e4/sample_submission.csv'


train_df = pd.read_csv(train_path)
train_df.head()


train_df.shape


test_df = pd.read_csv(test_path)
sub_df = pd.read_csv(sample_submission_path)
test_df.head()


test_df.shape


sub_df.head()


sub_df.shape


test_df.describe()


train_df.head(2)


train_df.info()


train_df.describe()


train_df.describe(include='object')


train_df.isna().sum()


plot_feature_distributions(train_df)


plot_numerical_features(train_df)


plot_numerical_features(train_df, 'stripplot')


!pip install ydata-profiling


from ydata_profiling import ProfileReport

profile = ProfileReport(train_df, title="Pandas Profiling Report")
profile.to_notebook_iframe()


print("\nCorrelation Matrix:")
correlation_matrix = train_df.select_dtypes(include=np.number).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


categorical_cols = train_df.select_dtypes(include='object').columns

for col in categorical_cols:
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=train_df, x=col, y='Listening_Time_minutes')
    plt.title(f'{col} vs Listening Time')
    plt.xticks(rotation=45)
    plt.show()



numerical_cols = train_df.select_dtypes(include=np.number).columns

for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=train_df, x=col, y='Listening_Time_minutes', alpha=0.2)
    plt.title(f'{col} vs Listening Time')
    plt.show()





df = train_df.copy()
df = df.drop('id', axis=1)

# Drop rows with any NaN values
df = df.dropna()

# Define a function to remove outliers using IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers from all numerical columns except the target
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
if 'Listening_Time_minutes' in numerical_cols:
    numerical_cols.remove('Listening_Time_minutes')

for col in numerical_cols:
    df = remove_outliers_iqr(df, col)

print("DataFrame after cleaning:")
print(f"New shape: {df.shape}")


train_df.shape


train_df.isna().sum()


df = df[df['Number_of_Ads'] <= 20]
df = df[df['Host_Popularity_percentage'] <= 100]
df = df[df['Guest_Popularity_percentage'] <= 100]
df = df[df['Episode_Length_minutes'] <= 200]


plot_numerical_features(df, 'stripplot')


df.shape


initial_cols = train_df.shape[1]
current_cols = df.shape[1]
deleted_cols = initial_cols - current_cols
percent_deleted_cols = (deleted_cols / initial_cols) * 100
print(f"Percentage of rows deleted: {percent_deleted_cols:.2f}%")



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import mean_squared_error


categorical_cols = train_df.select_dtypes(include='object').columns



# Label encode categorical features
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])


df.head(4)


X = df.drop('Listening_Time_minutes', axis=1)
y = df['Listening_Time_minutes']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_val.shape, y_train.shape, y_val.shape


train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val)


train_set


params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt'
}

lgb = lgb.train(params,
                train_set,
                valid_sets=[train_set, val_set],
                num_boost_round=1000)



y_train_pred_lgb = lgb.predict(X_train)
y_val_pred_lgb = lgb.predict(X_val)


lgb_metrics = regression_metrics_df(y_train, y_train_pred_lgb, y_val, y_val_pred_lgb, 'lgb')
lgb_metrics


plot_residuals(y_train, y_train_pred_lgb, 'LightGBM Training Residuals')


plot_residuals(y_val, y_val_pred_lgb, 'LightGBM Validation Residuals')


test_df.head()


X_test = test_df.drop('id', axis=1)

text_categorical_cols = X_test.select_dtypes(include='object').columns

for col in text_categorical_cols:
    le = LabelEncoder()
    X_test[col] = le.fit_transform(X_test[col])

X_test.shape


X_test.head()


y_test_pred_lgb = lgb.predict(X_test)
y_test_pred_lgb


sub_df['Listening_Time_minutes'] = y_test_pred_lgb
sub_df.head()


sub_df.to_csv('submission.csv', index=False)


pd.read_csv('/kaggle/working/submission.csv')




