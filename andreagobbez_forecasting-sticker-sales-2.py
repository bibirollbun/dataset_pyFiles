import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Load the data
train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")


# Inspect the first few rows
print(train.head())
print(test.head())

# Check for missing values
print(train.isnull().sum())
print(test.isnull().sum())

# Basic statistics
print(train.describe())


# Check rows with missing 'num_sold'
missing_data = train[train['num_sold'].isnull()]
print(missing_data.head())

# Check if missing values are concentrated in specific countries, stores, or products
print(missing_data['country'].value_counts())
print(missing_data['store'].value_counts())
print(missing_data['product'].value_counts())

# Check if missing values are concentrated in specific time periods
print(missing_data['date'].min(), missing_data['date'].max())


# Group by country, store, and product, then impute missing values using the median
train['num_sold'] = train.groupby(['country', 'store', 'product'])['num_sold'].transform(
    lambda x: x.fillna(x.median())
)

# If there are still missing values, fill with the overall median
train['num_sold'] = train['num_sold'].fillna(train['num_sold'].median())

# Check if there aren't missing values anymore
print(train.isnull().sum())


# Convert 'date' to datetime
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

# Plot sales over time
plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='num_sold', data=train)
plt.title('Sales Over Time')
plt.show()


# Sales by country
plt.figure(figsize=(10, 5))
sns.boxplot(x='country', y='num_sold', data=train)
plt.title('Sales by Country')
plt.show()

# Sales by store
plt.figure(figsize=(10, 5))
sns.boxplot(x='store', y='num_sold', data=train)
plt.title('Sales by Store')
plt.show()

# Sales by item
plt.figure(figsize=(10, 5))
sns.boxplot(x='product', y='num_sold', data=train)
plt.title('Sales by Item')
plt.show()


from statsmodels.tsa.seasonal import seasonal_decompose

# Aggregate sales by date
daily_sales = train.groupby('date')['num_sold'].sum()

# Decompose the time series
decomposition = seasonal_decompose(daily_sales, period=365)
decomposition.plot()
plt.show()


# Add a 'weekday' column
train['weekday'] = train['date'].dt.weekday
test['weekday'] = test['date'].dt.weekday

# Plot sales by weekday
plt.figure(figsize=(10, 5))
sns.boxplot(x='weekday', y='num_sold', data=train)
plt.title('Sales by Weekday')
plt.show()


train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day
train['week_of_year'] = train['date'].dt.isocalendar().week

test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test['week_of_year'] = test['date'].dt.isocalendar().week


# Example: Add a placeholder for holidays
train['is_holiday'] = train['date'].isin([...])  # Add actual holiday dates
test['is_holiday'] = test['date'].isin([...])    # Add actual holiday dates


# Example: Create a 7-day lag feature
train['lag_7'] = train.groupby(['country', 'store', 'product'])['num_sold'].shift(7)


train.info()
test.info()


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define categorical and numerical features
categorical_features = ['country', 'store', 'product']
numerical_features = ['weekday', 'year', 'month', 'day', 'week_of_year', 'is_holiday']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),  # One-hot encode categorical features
        ('num', StandardScaler(), numerical_features)  # Scale numerical features
    ])

# Apply preprocessing to train and test data
X_train = preprocessor.fit_transform(train)
y_train = train['num_sold']

X_test = preprocessor.transform(test)


# Add 'is_weekend' feature
train['is_weekend'] = train['weekday'].isin([5, 6]).astype(int)
test['is_weekend'] = test['weekday'].isin([5, 6]).astype(int)

# Add 'season' feature
train['season'] = (train['month'] % 12 + 3) // 3
test['season'] = (test['month'] % 12 + 3) // 3

# Update numerical features
numerical_features.extend(['is_weekend', 'season'])

# Reapply preprocessing with new features
X_train = preprocessor.fit_transform(train)
X_test = preprocessor.transform(test)


print(f"Number of categorical features (one-hot encoded): {X_train[:, :len(categorical_features)].shape[1]}")
print(f"Number of numerical features: {X_train[:, len(categorical_features):].shape[1]}")


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate

# Define inputs
input_cat = Input(shape=(len(categorical_features),), name='input_cat')
input_num = Input(shape=(X_train[:, len(categorical_features):].shape[1],), name='input_num')  # Updated shape


# Embedding layers for categorical features
embedding_layers = []
for i, col in enumerate(categorical_features):
    embedding_size = min(50, train[col].nunique() // 2)
    embedding_layers.append(Embedding(input_dim=train[col].nunique(), output_dim=embedding_size)(input_cat[:, i]))

# Flatten embeddings
flattened_embeddings = [Flatten()(layer) for layer in embedding_layers]

# Concatenate embeddings and numerical features
concat = Concatenate()(flattened_embeddings + [input_num])

# Dense layers
x = Dense(128, activation='relu')(concat)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='linear')(x)  # Linear activation for regression

# Define the model
model = Model(inputs=[input_cat, input_num], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(
    [X_train[:, :len(categorical_features)], X_train[:, len(categorical_features):]],
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)


# Predict on test data
predictions = model.predict([X_test[:, :len(categorical_features)], X_test[:, len(categorical_features):]])

# Flatten predictions to 1D array
predictions = predictions.flatten()


# Replace the 'num_sold' column with our predictions
sample_submission['num_sold'] = predictions

# Save the submission file
sample_submission.to_csv('submission.csv', index=False)

print(sample_submission.head())

