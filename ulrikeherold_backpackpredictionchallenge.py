# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns



# Load the datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
train_extra_df = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')

# Display the first few rows of the training data
train_df.head()


# Summary statistics
train_df.describe()


# Check for missing values
train_df.isnull().sum()


# Visualize the distribution of the target variable
sns.histplot(train_df['Price'], kde=True)
plt.title('Distribution of Prices')
plt.show()


# Fill missing values
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.drop('Price')
train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].mean())
test_df[numeric_cols] = test_df[numeric_cols].fillna(test_df[numeric_cols].mean())

# Encode categorical variables
X = pd.get_dummies(train_df.drop(columns=['Price'], errors='ignore'))
test_df_encoded = pd.get_dummies(test_df)

# Align the columns of the test set with the training set
X, test_df_encoded = X.align(test_df_encoded, join='left', axis=1, fill_value=0)

# Separate target variable
y = train_df['Price'] if 'Price' in train_df.columns else pd.Series([])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_scaled = scaler.transform(test_df_encoded)


# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on the validation set
y_val_pred = model.predict(X_val_scaled)

# Calculate the mean squared error
mse = mean_squared_error(y_val, y_val_pred)
print(f'Mean Squared Error: {mse}')


# Predict on the test set
test_predictions = model.predict(test_scaled)

# Prepare the submission file
submission = pd.DataFrame({'id': test_df['id'].astype(str), 'Price': test_predictions})
submission.to_csv('submission.csv', index=False)
submission.head()

