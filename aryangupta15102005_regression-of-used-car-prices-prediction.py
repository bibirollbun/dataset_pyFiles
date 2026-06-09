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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import seaborn as sns


# Load data
train_data = pd.read_csv("/kaggle/input/playground-series-s4e9/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s4e9/test.csv")


train_data.info()


train_data.describe


def plot_histograms(df, columns):
    for col in columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()


# Plot histograms for numerical columns
numerical_columns = ['price', 'milage', 'model_year']
plot_histograms(train_data, numerical_columns)


train_data['log_price'] = np.log(train_data['price'])
plt.figure(figsize=(10, 5))
sns.histplot(train_data['log_price'], kde=True)
plt.title('Distribution of log(price)')
plt.show()


X = train_data.drop(['id', 'price', 'log_price'], axis=1)
y = train_data['log_price']
X_test = test_data.drop('id', axis=1)


#preprocessing steps
numeric_features = ['milage', 'model_year']
categorical_features = ['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# Create a pipeline with preprocessor and model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42))
])


# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Fit the model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_val)


# Evaluate the model
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
print(f'RMSE on validation set: {rmse}')


feature_importance = model.named_steps['regressor'].feature_importances_
feature_names = (numeric_features + 
                 [f"{feature}_{category}" for feature in categorical_features 
                  for category in X[feature].unique()])

importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
importance_df = importance_df.sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=importance_df)
plt.title('Top 20 Feature Importances')
plt.show()


# Make predictions on test set
test_predictions = np.exp(model.predict(X_test))


test_predictions 


# Prepare submission
submission = pd.DataFrame({'id': test_data['id'], 'price': test_predictions})
submission.to_csv('submission.csv', index=False)
print("Submission file created.")


submission

