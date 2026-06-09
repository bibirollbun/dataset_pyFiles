df_test = pd.read_csv('/content/test (7).csv')
df_test_with_predictions = pd.merge(df_test, submission_df, on='id')
display(df_test_with_predictions.head())


import matplotlib.pyplot as plt

# Generate histogram of predicted prices in the test set
plt.figure(figsize=(8, 4))
plt.hist(df_test_with_predictions['price'], bins=50, edgecolor='black')
plt.xlabel('Predicted Price')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Prices (Test Set)')
plt.grid(axis='y', alpha=0.5)
plt.show()

# Generate histogram of actual prices in the training set
plt.figure(figsize=(8, 4))
plt.hist(df_train['price'], bins=50, edgecolor='black')
plt.xlabel('Actual Price')
plt.ylabel('Frequency')
plt.title('Distribution of Actual Prices (Training Set)')
plt.grid(axis='y', alpha=0.5)
plt.show()

# Display descriptive statistics for predicted and actual prices
display("Descriptive Statistics for Predicted Prices (Test Set):")
display(df_test_with_predictions['price'].describe())

display("Descriptive Statistics for Actual Prices (Training Set):")
display(df_train['price'].describe())


import matplotlib.pyplot as plt
import seaborn as sns

# Define numerical features
numerical_features = ['carat', 'depth', 'table', 'x', 'y', 'z']

# Plot scatter plots for numerical features vs predicted price
for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=df_test_with_predictions, x=feature, y='price', alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('Predicted Price')
    plt.title(f'Predicted Price vs {feature}')
    plt.show()

# Define categorical features
categorical_features = ['cut', 'color', 'clarity']

# Plot box plots for categorical features vs predicted price
for feature in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df_test_with_predictions, x=feature, y='price')
    plt.xlabel(feature)
    plt.ylabel('Predicted Price')
    plt.title(f'Predicted Price Distribution by {feature}')
    plt.show()


display(df_train.describe())
display(df_train.info())
display(df_train.isnull().sum())


import matplotlib.pyplot as plt

# Define numerical features
numerical_features = ['carat', 'depth', 'table', 'x', 'y', 'z', 'price']

# Plot histograms for each feature
for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    plt.hist(df_train[feature], bins=50, edgecolor='black')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {feature}')
    plt.grid(axis='y', alpha=0.5)  # Add grid for better readability
    plt.show()

# Plot box plots for each feature
for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    plt.boxplot(df_train[feature], vert=False)  # Horizontal box plot for better visualization
    plt.xlabel(feature)
    plt.title(f'Box Plot of {feature}')
    plt.grid(axis='x', alpha=0.5)  # Add grid for better readability
    plt.show()



import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of rows
num_rows = 30000

# Generate synthetic features
data = {
    'carat': np.random.uniform(0.1, 5.0, num_rows),
    'depth': np.random.uniform(40.0, 80.0, num_rows),
    'table': np.random.uniform(40.0, 100.0, num_rows),
    'x': np.random.uniform(1.0, 15.0, num_rows),
    'y': np.random.uniform(1.0, 15.0, num_rows),
    'z': np.random.uniform(1.0, 15.0, num_rows),
}

# Create DataFrame
df_synthetic = pd.DataFrame(data)

# Calculate synthetic price
df_synthetic['price'] = (
    1000 +  # Base price
    (df_synthetic['carat'] * 5000) +  # Carat contributes significantly to price
    (df_synthetic['x'] * df_synthetic['y'] * df_synthetic['z'] * 10) +  # Volume contributes to price
    np.random.normal(0, 1000, num_rows)  # Add noise
)

# Ensure price is positive and round to 2 decimal places
df_synthetic['price'] = df_synthetic['price'].abs().round(2)

# Save to CSV
df_synthetic.to_csv('synthetic_diamond_data.csv', index=False)

# Display the first 5 rows
display(df_synthetic.head())


df_test = pd.read_csv('/content/test (7).csv')

# Keep track of the original IDs for the submission file
original_ids = df_test['id']

# Apply the fitted preprocessor to the test data
X_test_processed = preprocessor.transform(df_test)

# Predict prices using the trained model
predictions = model.predict(X_test_processed)

display(predictions[:5])


submission_df = pd.DataFrame({'id': original_ids, 'price': predictions})
submission_df.to_csv('submission.csv', index=False)
display(submission_df.shape)


display(submission_df.shape)


scatter_features = ['carat', 'depth', 'table', 'x', 'y', 'z']

for feature in scatter_features:
    plt.figure(figsize=(8, 4))
    plt.scatter(df_train[feature], df_train['price'], alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('Price')
    plt.title(f'Scatter Plot of {feature} vs Price')
    plt.show()

categorical_features = ['cut', 'color', 'clarity']

for feature in categorical_features:
    display(df_train[feature].value_counts())


# Add your function here
def my_function(param1, param2):
  pass


display(df.isnull().sum())


display(df.describe())
display(df.info())


import pandas as pd

df = pd.read_csv('/content/test (7).csv')
display(df.head())


df_train = pd.read_csv('/content/train (7).csv')
display(df_train.head())


display(df_train.describe())
display(df_train.info())


numerical_features = ['carat', 'depth', 'table', 'x', 'y', 'z', 'price']

for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    plt.hist(df_train[feature], bins=50)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {feature}')
    plt.show()

for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    plt.boxplot(df_train[feature])
    plt.xlabel(feature)
    plt.title(f'Box Plot of {feature}')
    plt.show()


scatter_features = ['carat', 'depth', 'table', 'x', 'y', 'z']

for feature in scatter_features:
    plt.figure(figsize=(8, 4))
    plt.scatter(df_train[feature], df_train['price'], alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('Price')
    plt.title(f'Scatter Plot of {feature} vs Price')
    plt.show()

categorical_features = ['cut', 'color', 'clarity']

for feature in categorical_features:
    display(df_train[feature].value_counts())


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Handle outliers in x, y, z where values are 0
df_train = df_train[(df_train['x'] > 0) & (df_train['y'] > 0) & (df_train['z'] > 0)]

# Handle outliers using IQR for numerical features except price
numerical_features = ['carat', 'depth', 'table']
for feature in numerical_features:
    Q1 = df_train[feature].quantile(0.25)
    Q3 = df_train[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_train = df_train[(df_train[feature] >= lower_bound) & (df_train[feature] <= upper_bound)]

# Apply one-hot encoding to categorical features
categorical_features = ['cut', 'color', 'clarity']
one_hot = OneHotEncoder(handle_unknown='ignore')
transformer = ColumnTransformer([('one_hot', one_hot, categorical_features)], remainder='passthrough')

# Separate features (X) and target (y)
X = df_train.drop('price', axis=1)
y = df_train['price']

# Create a pipeline for preprocessing, including scaling
preprocessor = Pipeline(steps=[('transformer', transformer),
                               ('scaler', StandardScaler())])

X_processed = preprocessor.fit_transform(X)

display(X_processed.shape)
display(y.shape)


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)
model.fit(X_processed, y)


from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

# Perform cross-validation
cv_scores = cross_val_score(model, X_processed, y, scoring='r2', cv=5)
print("Cross-validation R² scores:", cv_scores)

# Calculate and print mean and standard deviation of cross-validation scores
print("Mean cross-validation R² score:", cv_scores.mean())
print("Standard deviation of cross-validation R² score:", cv_scores.std())

# Predict on training data and calculate R² score
y_pred = model.predict(X_processed)
train_r2_score = r2_score(y, y_pred)
print("Training R² score:", train_r2_score)


df_test = pd.read_csv('/content/test (7).csv')

# Keep track of the original IDs for the submission file
original_ids = df_test['id']

# Apply the fitted preprocessor to the test data
# The preprocessor was fitted on the training data and includes
# one-hot encoding for categorical features and scaling for numerical features.
X_test_processed = preprocessor.transform(df_test)

# Predict prices using the trained model
# The 'model' variable holds the trained RandomForestRegressor model.
predictions = model.predict(X_test_processed)

# Display the first 5 predictions
display(predictions[:5])


submission_df = pd.DataFrame({'id': original_ids, 'price': predictions})
submission_df.to_csv('submission.csv', index=False)


df_test = pd.read_csv('/content/test (7).csv')
df_test_with_predictions = pd.merge(df_test, submission_df, on='id')
display(df_test_with_predictions.head())


import matplotlib.pyplot as plt

# Generate histogram of predicted prices in the test set
plt.figure(figsize=(8, 4))
plt.hist(df_test_with_predictions['price'], bins=50, edgecolor='black')
plt.xlabel('Predicted Price')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Prices (Test Set)')
plt.grid(axis='y', alpha=0.5)
plt.show()

# Generate histogram of actual prices in the training set
plt.figure(figsize=(8, 4))
plt.hist(df_train['price'], bins=50, edgecolor='black')
plt.xlabel('Actual Price')
plt.ylabel('Frequency')
plt.title('Distribution of Actual Prices (Training Set)')
plt.grid(axis='y', alpha=0.5)
plt.show()

# Display descriptive statistics for predicted and actual prices
display("Descriptive Statistics for Predicted Prices (Test Set):")
display(df_test_with_predictions['price'].describe())

display("Descriptive Statistics for Actual Prices (Training Set):")
display(df_train['price'].describe())


import matplotlib.pyplot as plt
import seaborn as sns

# Define numerical features
numerical_features = ['carat', 'depth', 'table', 'x', 'y', 'z']

# Plot scatter plots for numerical features vs predicted price
for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=df_test_with_predictions, x=feature, y='price', alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('Predicted Price')
    plt.title(f'Predicted Price vs {feature}')
    plt.show()

# Define categorical features
categorical_features = ['cut', 'color', 'clarity']

# Plot box plots for categorical features vs predicted price
for feature in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df_test_with_predictions, x=feature, y='price')
    plt.xlabel(feature)
    plt.ylabel('Predicted Price')
    plt.title(f'Predicted Price Distribution by {feature}')
    plt.show()


# Load the test data again
df_test = pd.read_csv('/content/test (7).csv')

# Keep track of the original IDs for the submission file
original_ids = df_test['id']

# Apply the fitted preprocessor to the test data
# The preprocessor was fitted on the training data and includes
# one-hot encoding for categorical features and scaling for numerical features.
X_test_processed = preprocessor.transform(df_test)

# Predict prices using the trained XGBoost model
predictions_xgb = xgb_model.predict(X_test_processed)

# Create a submission DataFrame with 'id' and 'price' columns
submission_df_xgb = pd.DataFrame({'id': original_ids, 'price': predictions_xgb})

# Save the submission DataFrame to a CSV file
submission_df_xgb.to_csv('submission_xgb.csv', index=False)

# Display the shape of the submission file to confirm the number of rows
display(submission_df_xgb.shape)


# Install XGBoost if you haven't already
!pip install xgboost

import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

# Initialize and train the XGBoost Regressor model
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_processed, y)

# Perform cross-validation
cv_scores_xgb = cross_val_score(xgb_model, X_processed, y, scoring='r2', cv=5)
print("Cross-validation R² scores (XGBoost):", cv_scores_xgb)

# Calculate and print mean and standard deviation of cross-validation scores
print("Mean cross-validation R² score (XGBoost):", cv_scores_xgb.mean())
print("Standard deviation of cross-validation R² score (XGBoost):", cv_scores_xgb.std())

# Predict on training data and calculate R² score
y_pred_xgb = xgb_model.predict(X_processed)
train_r2_score_xgb = r2_score(y, y_pred_xgb)
print("Training R² score (XGBoost):", train_r2_score_xgb)

