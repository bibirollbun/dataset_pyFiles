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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category = FutureWarning)


train_df = pd.read_csv('/kaggle/input/predict-supercars-prices-2025/supercars_train.csv')
test_df = pd.read_csv('/kaggle/input/predict-supercars-prices-2025/supercars_test.csv')


train_df.sample(5)


train_df.info()


test_df.info()


train_df['damage_type']


#Filling Missing Values
train_df['damage_cost'] = train_df['damage_cost'].fillna(0)
test_df['damage_cost'] = test_df['damage_cost'].fillna(0)


train_df['damage_type'] = train_df['damage_type'].fillna('None')
test_df['damage_type'] = test_df['damage_type'].fillna('None')


train_df['damage_type'].value_counts()


train_df.duplicated().sum()


train_df.info()


train_df.drop(columns = ['id'],inplace = True)
ids = test_df['id'].copy()
test_df.drop(columns = ['id'],inplace = True)


cols = train_df.select_dtypes(include=['object']).columns
cols


for col in cols:
    plt.figure(figsize=(8, 4))
    train_df[col].value_counts(dropna=False).plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Value Counts of {col}', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


train_df['last_service_date'].value_counts()


train_df['last_service_date'] = pd.to_datetime(train_df['last_service_date'], errors='coerce')
test_df['last_service_date'] = pd.to_datetime(test_df['last_service_date'], errors='coerce')



train_df['last_service_year'] = train_df['last_service_date'].dt.year
train_df['last_service_month'] = train_df['last_service_date'].dt.month


test_df['last_service_year'] = test_df['last_service_date'].dt.year
test_df['last_service_month'] = test_df['last_service_date'].dt.month





train_df.drop(columns = ['last_service_date'], inplace = True)
test_df.drop(columns = ['last_service_date'], inplace = True)


from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()

for col in train_df.select_dtypes(include=['object']).columns:
    train_df[col] = encoder.fit_transform(train_df[[col]])


for col in test_df.select_dtypes(include=['object']).columns:
    test_df[col] = encoder.fit_transform(test_df[[col]])



train_df.select_dtypes(include = ['object']).columns


test_df.select_dtypes(include = ['object']).columns


train_df.columns


plt.figure(figsize=(20, 20))
corr = train_df.corr(numeric_only=True) 
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.show()


train_df.skew()


X = train_df.drop(columns = ['price'],axis = 1)
y = train_df['price']


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)


from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import xgboost as xgb

# Define pipeline
pipeline = Pipeline([
    ('var_thresh', VarianceThreshold(threshold=0.01)),  # Remove features with variance < 0.01
    ('model', xgb.XGBRegressor(n_estimators=200))
])

# Fit pipeline
pipeline.fit(x_train, y_train)

# Predict
y_pred = pipeline.predict(x_test)

# Evaluate
print(f'R2 Score: {r2_score(y_test, y_pred):.4f}')
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}')
print(f'Mean absolute Error: {mean_absolute_error(y_test, y_pred):.2f}')
print(f'Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}')




from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, StackingRegressor

# Define pipeline
pipeline = Pipeline([
    ('var_thresh', VarianceThreshold(threshold=0.01)),  # Remove features with variance < 0.01
    ('model', GradientBoostingRegressor(n_estimators=150))
])

# Fit pipeline
pipeline.fit(x_train, y_train)

# Predict
y_pred = pipeline.predict(x_test)

# Evaluate
print(f'R2 Score: {r2_score(y_test, y_pred):.4f}')
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}')
print(f'Mean absolute Error: {mean_absolute_error(y_test, y_pred):.2f}')
print(f'Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}')



gbr = GradientBoostingRegressor(n_estimators = 150)

bg = BaggingRegressor(estimator = gbr, n_estimators = 15)
bg.fit(x_train, y_train)

# Predict
y_pred = bg.predict(x_test)

# Evaluate
print(f'R2 Score: {r2_score(y_test, y_pred):.4f}')
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}')
print(f'Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}')


gbr = GradientBoostingRegressor(n_estimators = 150)
gbr.fit(x_train, y_train)
# Predict
y_pred = gbr.predict(x_test)

# Evaluate
print(f'R2 Score: {r2_score(y_test, y_pred):.4f}')
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}')
print(f'Mean absolute Error: {mean_absolute_error(y_test, y_pred):.2f}')
print(f'Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}')



y_pred = bg.predict(test_df)


y_pred


submission = pd.DataFrame({
    'id':ids,
    'target': y_pred
})


submission.head()


submission.to_csv('submissionc2.csv',index = False)

