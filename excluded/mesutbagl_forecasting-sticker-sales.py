import numpy as np
import pandas as pd
import joblib
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split  # Importing train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score


train_df=pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


train_df.head()


test_df.head()


train_df.info()


train_df.shape


train_df.isnull().sum()


train_df = train_df.dropna().reset_index(drop=True)


train_df['country'].value_counts()


train_df['product'].value_counts()


train_df['store'].value_counts()


# Convert 'date' to datetime
train_df['date'] = pd.to_datetime(train_df['date'])

# Extracting features from the date
train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day


# Convert 'date' to datetime
test_df['date'] = pd.to_datetime(test_df['date'])

# Extracting features from the date
test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day


# Visualizing the relationships

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(train_df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Distribution of num_sold
plt.figure(figsize=(8, 5))
sns.histplot(train_df['num_sold'], bins=10, kde=True)
plt.title('Distribution of Number Sold')
plt.xlabel('Number Sold')
plt.ylabel('Frequency')
plt.show()


# Set Seaborn style
sns.set(style='whitegrid')

# Group by country and sum the sales
sales_by_country = train_df.groupby('country')['num_sold'].sum().reset_index()

# Create a figure with a specific size and style
plt.figure(figsize=(14, 8))
plt.rc('axes', titlesize=16, titleweight='bold')  # Customize title size and weight
plt.rc('axes', labelsize=14, labelweight='bold')  # Customize label size and weight
plt.rc('xtick', labelsize=12)  # Customize x-tick label size
plt.rc('ytick', labelsize=12)  # Customize y-tick label size

# Plot sales by country
sns.barplot(data=sales_by_country, x='country', y='num_sold', palette='viridis')
plt.title('Total Sales by Country', fontsize=18, fontweight='bold')
plt.xlabel('Country', fontsize=16, fontweight='bold')
plt.ylabel('Total Sales', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()


plt.figure(figsize = (12, 8))
ax = sns.barplot(data=train_df, x='country', y='num_sold', hue='store')


plt.figure(figsize = (12, 8))
ax = sns.barplot(data=train_df, x='store', y='num_sold', hue='country')


plt.figure(figsize = (12, 8))
ax = sns.barplot(data=train_df, x='country', y='num_sold', hue='product')


plt.figure(figsize = (12, 8))
ax = sns.barplot(data=train_df, x='store', y='num_sold', hue='product')


# Group by year and sum the sales
yearly_sales = train_df.groupby('year')['num_sold'].sum()

# Plot the yearly sales comparison
plt.figure(figsize=(12, 6))
yearly_sales.plot(kind='bar')
plt.title('Total Sales by Year')
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()


train_df.head()


# Set 'date' as the index
train_df.set_index('date', inplace=True)
train_df['dayofweek'] = train_df.index.dayofweek
train_df['quarter'] = train_df.index.quarter
train_df['week_of_year'] = train_df.index.isocalendar().week
train_df['day_of_month'] = train_df.index.day
train_df['is_weekend'] = train_df.index.dayofweek >= 5  # 5 for Saturday, 6 for Sunday


# Set 'date' as the index
test_df.set_index('date', inplace=True)
test_df['dayofweek'] = test_df.index.dayofweek
test_df['quarter'] = test_df.index.quarter
test_df['week_of_year'] = test_df.index.isocalendar().week
test_df['day_of_month'] = test_df.index.day
test_df['is_weekend'] = test_df.index.dayofweek >= 5  # 5 for Saturday, 6 for Sunday


X = train_df.drop(columns=['id','num_sold'], axis=1)  # Features
y = train_df['num_sold']  # Target


X.isnull().sum()


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the encoder (for one-hot encoding categorical columns)
encoder = ColumnTransformer(
    transformers=[
        ('country', OneHotEncoder(), ['country']),  # One-hot encoding for 'country'
        ('store', OneHotEncoder(), ['store']),      # One-hot encoding for 'store'
        ('product', OneHotEncoder(), ['product'])   # One-hot encoding for 'product'
    ],
    remainder='passthrough'  # Keep other columns as they are
)

# Fit and transform on the training data
X_train_encoded = encoder.fit_transform(X_train)

# Now, transform the test data using the same encoder
X_test_encoded = encoder.transform(X_test)


# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_encoded, y_train)


# Make predictions on the test set
y_pred = model.predict(X_test_encoded)


# Evaluate the model

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


train_df.head()


test_df.head()


test_id=test_df['id']


test_df=test_df.drop('id', axis=1)


# Define the encoder (for one-hot encoding categorical columns)
encoder = ColumnTransformer(
    transformers=[
        ('country', OneHotEncoder(), ['country']),  # One-hot encoding for 'country'
        ('store', OneHotEncoder(), ['store']),      # One-hot encoding for 'store'
        ('product', OneHotEncoder(), ['product'])   # One-hot encoding for 'product'
    ],
    remainder='passthrough'  # Keep other columns as they are
)

# Fit and transform on the training data
test_df_encoded = encoder.fit_transform(test_df)

# Now, transform the test data using the same encoder
test_df_encoded = encoder.transform(test_df)


# Make predictions on the test data
y_pred = model.predict(test_df_encoded)


y_pred


# Prepare submission file
submission_df = pd.DataFrame({
    'id': test_id,  # 'id' from the test set index
    'num_sold': y_pred  # Predictions
})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)


# Save the trained model
joblib.dump(model, 'model.pkl')

# Save the encoder
joblib.dump(encoder, 'encoder.pkl')





