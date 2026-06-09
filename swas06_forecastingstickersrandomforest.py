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


df_train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")


df_train.head(3),df_test.head(3)


df_train.shape,df_test.shape


df_train.info(),df_test.info()


df_train.isnull().sum(),df_test.isnull().sum()


df_train['num_sold'] = df_train['num_sold'].fillna(df_train['num_sold'].median())


df_train.isnull().sum(),


df_train['date'] = pd.to_datetime(df_train['date'])
df_train['year'] = df_train['date'].dt.year
df_train['month'] = df_train['date'].dt.month
df_train['day'] = df_train['date'].dt.day


df_train['year'].head(3)




import matplotlib.pyplot as plt
import seaborn as sns
features =['country', 'store', 'product']
plt.figure(figsize=(15, 12))

for i, feature in enumerate(features, 1):
    plt.subplot(4, 3, i)  # Adjust number of rows and columns based on the number of features
    sns.countplot(data=df_train, x=feature)
    plt.title(f'Distribution of {feature}')
    plt.xticks(rotation=45)  # Rotate x-axis labels if needed

plt.tight_layout() 
plt.show()



for feature in features:
    plt.figure(figsize=(7, 7))
    df_train[feature].value_counts().plot.pie(autopct='%1.1f%%', figsize=(7,7))
    plt.title(f'Proportions of {feature}')
    plt.ylabel('')
    plt.show()


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error

# Define features and target
X = df_train.drop(columns=['date', 'num_sold'])  # Remove original date column
y = df_train['num_sold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical and numerical features
categorical_features = ['country', 'store', 'product']
numerical_features = ['year', 'month', 'day']

# Preprocessing Pipelines
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# ColumnTransformer to Apply Transformations
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

# Regression Model Pipeline
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "CatBoost": CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, verbose=0, random_state=42),
    "XGBoost":XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    print(f"{name} MAPE: {mape:.4f}")


pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict on test data
y_pred = pipeline.predict(X_test)

# Calculate MAPE
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"Random Forest MAPE: {mape:.4f}")


df_test['date'] = pd.to_datetime(df_test['date'])
df_test['year'] = df_test['date'].dt.year
df_test['month'] = df_test['date'].dt.month
df_test['day'] = df_test['date'].dt.day


y_test_pred = pipeline.predict(df_test)
y_test_pred = np.ceil(y_test_pred)


submission_df = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
submission_df['num_sold'] = y_test_pred
submission_df.to_csv('submission.csv', index=False)
print(submission_df.head())

