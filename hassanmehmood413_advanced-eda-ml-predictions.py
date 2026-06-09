import pandas as pd
import numpy as np

# For Data Visualizations 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# For Encoding 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# For Feature Engineering 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# For Model Training 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Metrics for regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# To save the model
import pickle


df_train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
df_sample = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')

df_train.head()


df_test.shape


df_train.shape


df_sample.shape


df_test.head()


# Check the shape of the data 
print("The shape of the data is ", df_train.shape)


df_train.info()


df_train.describe()


df_train = df_train.dropna()
df_test = df_test.dropna()


import pandas as pd

def format_date(df_train):
    # Ensure the 'date' column is a datetime object
    if 'date' in df_train.columns:
        df_train['date'] = pd.to_datetime(df_train['date'], errors='coerce')  # Convert to datetime
        if df_train['date'].isna().any():
            raise ValueError("The 'date' column contains invalid datetime values.")
    else:
        raise KeyError("The DataFrame does not have a 'date' column.")
    
    # Extract date-related components
    df_train['year'] = df_train['date'].dt.year
    df_train['month'] = df_train['date'].dt.month
    df_train['day'] = df_train['date'].dt.day
    df_train['dayOfYear'] = df_train['date'].dt.dayofyear
    df_train['weekday'] = df_train['date'].dt.weekday
    
    return df_train

df_train = format_date(df_train)
df_test = format_date(df_test)


df_train.head()


def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df_train['season'] = df_train['month'].apply(get_season)
df_test['season'] = df_test['month'].apply(get_season)


df_train.head()


df_test.head()


# Shape of data 
print(f"The shape of train data is " , df_train.shape)
print(f"The shape of test data is " , df_test.shape)



df_train['num_sold'] = np.log(df_train['num_sold'])



plt.figure(figsize=(28, 6))
df_train.groupby('date')['num_sold'].sum().plot(title='Total Sales Over Time', xlabel='Date', ylabel='Number of Products Sold')
plt.grid()
plt.show()


plt.figure(figsize=(20,5))
fig = sns.lineplot(data=df_train,x='country',y='num_sold',hue='year')
plt.title('Sales Trends by Country Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Products Sold')
plt.legend(title='Country')
plt.show()
plt.show()


plt.figure(figsize=(20,5))
fig = sns.lineplot(data=df_train,x='year',y='num_sold',hue='country')
plt.title('Different countries performed in terms of sales year-over-year.')
plt.xlabel('Year')
plt.ylabel('Number of Products Sold')
plt.legend(title='Country')
plt.show()


plt.figure(figsize=(20,5))
fig = sns.lineplot(data=df_train,x='month',y='num_sold',hue='product')
plt.title('Sales Trends by Product Year-Wise')
plt.xlabel('Year')
plt.ylabel('Number of Products Sold')
plt.legend(title='Product')
plt.show()


plt.figure(figsize=(20,5))
fig = sns.lineplot(data=df_train,x='year',y='num_sold',hue='store')
plt.title('Sales Trends by Stores')
plt.xlabel('Year')
plt.ylabel('Number of Products Sold')
plt.legend(title='Stores')
plt.show()


sns.histplot(data=df_train, x=df_train['num_sold'], bins=10, kde=False)


df_train.head()


df_test.head()



# Categorical columns to encode
categorical_columns = ['country', 'product', 'store','season']

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Encode each categorical column in both the train and test DataFrames
for col in categorical_columns:
    # Fit the encoder on the train data and transform both train and test data
    df_train[col] = label_encoder.fit_transform(df_train[col])
    df_test[col] = label_encoder.transform(df_test[col])


# Drop id and date column from dataset
df_train = df_train.drop(columns=['id','date'])
df_train.head()



df_test = df_test.drop(columns=['date'])
df_test.head()


# Calculate the correlation matrix
correlation_matrix = df_train.corr()

# Plot the heatmap
plt.figure(figsize=(15, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()


# Split df_train into X and y
X = df_train.drop('num_sold', axis=1)
y = df_train['num_sold']


# Train-test split (only for training and validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define models
models = [
    ('LinearRegression', LinearRegression()),
    ('DecisionTreeRegressor', DecisionTreeRegressor(random_state=42)),
    ('RandomForestRegressor', RandomForestRegressor(random_state=42)),
    ('KNeighborsRegressor', KNeighborsRegressor()),
    ('GradientBoostingRegressor', GradientBoostingRegressor(random_state=42)),
    ('XGBRegressor', XGBRegressor(random_state=42)),
    ('AdaBoostRegressor', AdaBoostRegressor(random_state=42)),
    ('CatBoostRegressor', CatBoostRegressor(random_state=42, verbose=0)),
]



import joblib
model_scores = []

# Train and evaluate each model
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    model_scores.append((name, mse, mae, r2, mape))
    
    print(f"{name} MSE: {mse:.2f}, MAE: {mae:.2f}, RÂ²: {r2:.2f}, MAPE: {mape:.2f}")
    print("-" * 50)

# Sort models by MAPE (lowest MAPE is best)
model_scores.sort(key=lambda x: x[4])

# Best model based on MAPE
best_model_name, best_mse, best_mae, best_r2, best_mape = model_scores[0]
print(f"Best model: {best_model_name} with MAPE: {best_mape:.2f}")

# Re-initialize and fit the best model
best_model = None
for name, model in models:
    if name == best_model_name:
        best_model = model
        break

best_model.fit(X_train, y_train)

# Save the best model
joblib.dump(best_model, 'best_model.joblib')
print("Best model saved successfully!")


# Assuming df_test is the full test set (with 'id' column)
# Preprocess df_test in the same way as df_train (e.g., dropping 'num_sold')
X_test_full = df_test.drop('num_sold', axis=1, errors='ignore')  # Use only features for prediction

# Make predictions on the entire df_test
y_test_pred = best_model.predict(X_test_full)

# Prepare the submission DataFrame
submission = pd.DataFrame({
    'id': df_test['id'],  # Use the entire 'id' column from df_test
    'num_sold': y_test_pred  # Predicted 'num_sold' values
})

# Save submission file
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")


