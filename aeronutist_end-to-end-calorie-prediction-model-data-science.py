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


#Calories Burned Prediction Analysis

## 1. Importing Libraries



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/calories-of-workout/train.csv')
test_df = pd.read_csv('/kaggle/input/calories-of-workout/test.csv')
sample_submission = pd.read_csv('/kaggle/input/calories-of-workout/sample_submission.csv')  # If available


print("Train Data Shape:", train_df.shape)
print("Test Data Shape:", test_df.shape)
print("\nTrain Data Columns:", train_df.columns.tolist())
print("\nTrain Data Sample:")
display(train_df.head())
print("\nTest Data Sample:")
display(test_df.head())


print("\nTrain Data Statistics:")
display(train_df.describe())
print("\nTest Data Statistics:")
display(test_df.describe())


print("\nMissing Values in Train Data:")
print(train_df.isnull().sum())
print("\nMissing Values in Test Data:")
print(test_df.isnull().sum())


plt.figure(figsize=(10, 6))
sns.histplot(train_df['Calories'], kde=True)
plt.title('Distribution of Calories Burned')
plt.show()


# Select only numeric columns for correlation
numeric_df = train_df.select_dtypes(include=['float64', 'int64'])

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (Numeric Features Only)')
plt.show()


sns.pairplot(train_df[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']])
plt.show()


X = train_df.drop(['Calories', 'id'], axis=1)
y = train_df['Calories']


categorical_cols = ['Sex']
numerical_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

rf_model.fit(X_train, y_train)


xgb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        objective='reg:gamma',  # Ensures positive predictions
        random_state=42
    ))
])
xgb_model.fit(X_train, y_train)


def evaluate_model(model, X, y):
    predictions = model.predict(X)
    rmsle = np.sqrt(mean_squared_log_error(y, predictions))
    return rmsle


print("Random Forest RMSLE:")
print("Train:", evaluate_model(rf_model, X_train, y_train))
print("Validation:", evaluate_model(rf_model, X_val, y_val))

print("\nXGBoost RMSLE:")
print("Train:", evaluate_model(xgb_model, X_train, y_train))
print("Validation:", evaluate_model(xgb_model, X_val, y_val))


rf_regressor = rf_model.named_steps['regressor']
feature_names = numerical_cols + list(rf_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols))
importances = pd.Series(rf_regressor.feature_importances_, index=feature_names)
importances.sort_values().plot(kind='barh')
plt.title('Feature Importance')
plt.show()


X_test = test_df.drop(['id'], axis=1)


test_predictions = xgb_model.predict(X_test)


submission = pd.DataFrame({
    'id': test_df['id'],
    'Calories': test_predictions
})

submission.to_csv('calories_prediction_submission.csv', index=False)
print("\nSubmission file created successfully!")
#print()

