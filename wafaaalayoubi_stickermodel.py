import pandas as pd


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


# Basic information about the datasets
print("Train Data Info:")
print(train.info())
print("\nTest Data Info:")
print(test.info())


# Display the first few rows
print("\nFirst few rows of the training data:")
print(train.head())

# Check for missing values
print("\nMissing values in training data:")
print(train.isnull().sum())

print("\nMissing values in test data:")
print(test.isnull().sum())


# Descriptive statistics of numerical columns
print("\nDescriptive statistics of the training data:")
print(train.describe())


# Unique values in categorical columns
categorical_columns = ['country', 'store', 'product']
for col in categorical_columns:
    print(f"\nUnique values in {col}:")
    print(train[col].unique())


# Analyze the target variable
print("\nTarget variable (num_sold) distribution:")
print(train['num_sold'].describe())


# Remove rows with missing values in the training set
train = train.dropna(subset=['num_sold'])

print(f"After removing missing values, training data shape: {train.shape}")


import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the distribution of num_sold
plt.figure(figsize=(10, 6))
sns.histplot(train['num_sold'], bins=50, kde=True, color='blue')
plt.title('Distribution of num_sold')
plt.xlabel('num_sold')
plt.ylabel('Frequency')
plt.show()

# Bar plots for num_sold by categorical variables
categorical_columns = ['country', 'store', 'product']
for col in categorical_columns:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=train, x=col, y='num_sold', ci=None)
    plt.title(f'Average num_sold by {col}')
    plt.xticks(rotation=45)
    plt.show()


# Convert date column to datetime
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

# Extract date features
for df in [train, test]:
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)  # 5=Saturday, 6=Sunday

# Check the updated training data
print(train.head())


import numpy as np

# Apply log transformation to num_sold
train['num_sold_log'] = np.log1p(train['num_sold'])

# Visualize the transformed distribution
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(train['num_sold_log'], bins=50, kde=True, color='green')
plt.title('Log-Transformed Distribution of num_sold')
plt.xlabel('num_sold_log')
plt.ylabel('Frequency')
plt.show()


from category_encoders import TargetEncoder

# Define categorical columns
categorical_cols = ['country', 'store', 'product']

# Apply target encoding
encoder = TargetEncoder(cols=categorical_cols)
train[categorical_cols] = encoder.fit_transform(train[categorical_cols], train['num_sold'])
test[categorical_cols] = encoder.transform(test[categorical_cols])


from sklearn.model_selection import train_test_split

X = train.drop(columns=['num_sold', 'num_sold_log', 'date', 'id'])  # Features
y = train['num_sold_log']  # Target (log-transformed)

# 80-20 train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# Define the parameter grid for XGBoost
param_dist = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [4, 6, 8, 10],
    'n_estimators': [500, 1000, 1500],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [0.5, 1, 1.5]
}

# Initialize the model
model = xgb.XGBRegressor(random_state=42)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=10,  # Number of random combinations to try
    scoring='neg_mean_absolute_error',  # Optimize for MAE
    cv=3,  # 3-fold cross-validation
    verbose=1,
    random_state=42,
    n_jobs=-1  # Use all cores
)

# Fit the model with RandomizedSearchCV
random_search.fit(X_train, y_train)

# Get the best model and its parameters
best_model = random_search.best_estimator_
print(f"Best Model Parameters: {random_search.best_params_}")

# Evaluate the model on the validation set
y_pred = best_model.predict(X_val)

# Calculate MAE and R² score
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
print(f"MAE: {mae}")
print(f"R²: {r2}")


# Retrain on the full dataset
final_model = best_model
final_model.fit(X, y)

# Predict on test set
X_test = test.drop(columns=['id', 'date'])  # Dropping 'id' and 'date' from test set
test_predictions = final_model.predict(X_test)

# Apply inverse log transformation and create submission file
submission = pd.DataFrame({'id': test['id'], 'num_sold': np.expm1(test_predictions)})
submission.to_csv('submission.csv', index=False)

