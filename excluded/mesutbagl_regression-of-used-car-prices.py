import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',None)
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree, ExtraTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import normalize, scale

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


train_df=pd.read_csv('/kaggle/input/playground-series-s4e9/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s4e9/test.csv')


train_df.head()


train_df['transmission'].value_counts()


# Remove rows with "â€“" and "not supported"
train_df = train_df[~train_df['transmission'].isin(['â€“', 'Variable', 'F', 'SCHEDULED FOR OR IN PRODUCTION'])]


# Remove rows with "â€“" and "not supported"
train_df = train_df[~train_df['fuel_type'].isin(['â€“', 'not supported'])]


train_df.info()


train_df.shape


train_df.describe()


train_df['clean_title'].value_counts()


train_df.isnull().sum()


train_df.corr(numeric_only=True)


# Distribution of Car Prices
plt.figure(figsize=(10, 6))
sns.histplot(train_df['price'], bins=30, kde=True)
plt.title('Distribution of Car Prices', fontsize=16)
plt.xlabel('Price', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid()
plt.show()


# Relationship between Mileage and Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='milage', y='price', data=train_df, alpha=0.6)
plt.title('Mileage vs Price', fontsize=16)
plt.xlabel('Mileage', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.grid()
plt.show()


# Box plot of Prices by Accident
plt.figure(figsize=(10, 6))
sns.boxplot(x='accident', y='price', data=train_df)
plt.title('Price Distribution by Accident', fontsize=16)
plt.xlabel('Accident', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.grid()
plt.show()


# Average Price by Model Year
plt.figure(figsize=(12, 6))
avg_price_by_make = train_df.groupby('model_year')['price'].mean().sort_values(ascending=False)
sns.barplot(x=avg_price_by_make.index, y=avg_price_by_make.values, palette='viridis')
plt.title('Average Price by Model Year', fontsize=16)
plt.xlabel('Model Year', fontsize=12)
plt.ylabel('Average Price', fontsize=12)
plt.xticks(rotation=45)
plt.grid()
plt.show()


# Create a box plot for visualizing outliers in the milage column
plt.figure(figsize=(10, 5))
sns.boxplot(data=train_df['milage'])
plt.title('Box Plot for Mileage')
plt.show()


# Checking missing values
train_df.isnull().sum()


# Filter rows where 'fuel_type' is missing
missing_fuel_type = train_df[train_df['fuel_type'].isnull()]

missing_fuel_type


train_df['fuel_type'].fillna('Electric', inplace=True)


train_df['accident'].fillna(train_df['accident'].mode()[0], inplace=True)


train_df['clean_title'].fillna('Unknown', inplace=True)


# Function to extract engine details
def extract_engine_details(engine_str):
    # Regex patterns
    hp_pattern = r'(\d+\.?\d*)HP'
    size_pattern = r'(\d+\.?\d*)L'
    cylinder_pattern = r'(\d+)\s+Cylinder'
    

    # Extracting values
    hp = re.search(hp_pattern, engine_str)
    size = re.search(size_pattern, engine_str)
    cylinder = re.search(cylinder_pattern, engine_str)
    

    return {
        'horsepower': float(hp.group(1)) if hp else None,
        'engine_size': float(size.group(1)) if size else None,
        'cylinders': int(cylinder.group(1)) if cylinder else None,
    }

# Apply the function to the engine column
engine_details = train_df['engine'].apply(extract_engine_details)

# Create new columns in the DataFrame
train_df = train_df.join(pd.DataFrame(engine_details.tolist()))


train_df.isnull().sum()


# Mean Imputation
train_df['horsepower'].fillna(train_df['horsepower'].median(), inplace=True)
train_df['engine_size'].fillna(train_df['engine_size'].median(), inplace=True)
train_df['cylinders'].fillna(train_df['cylinders'].median(), inplace=True)


# 3. simplify transmission column
train_df['transmission'] = train_df['transmission'].replace({
        r'(?i).*Dual Shift.*': 'Dual Shift',
        r'(?i).*(automatic|A/T).*': 'Automatic',
        r'(?i).*(manual|M/T).*': 'Manual',
        r'(?i).*CVT.*': 'CVT'
    }, regex=True)

train_df['transmission'] =train_df['transmission'].where(train_df['transmission'].isin(['Dual Shift', 'Automatic', 'Manual', 'CVT']),'Other')


# Create a box plot for visualizing outliers in the horsepower column
plt.figure(figsize=(10, 5))
sns.boxplot(data=train_df['horsepower'])
plt.title('Box Plot for Horsepower')
plt.show()


# Create a box plot for visualizing outliers in the cylinders column
plt.figure(figsize=(10, 5))
sns.boxplot(data=train_df['cylinders'])
plt.title('Box Plot for Cylinders')
plt.show()


# Create a box plot for visualizing outliers in the engine_size column
plt.figure(figsize=(10, 5))
sns.boxplot(data=train_df['engine_size'])
plt.title('Box Plot for Engine Size')
plt.show()


train_df['transmission'].nunique()


# Map 'yes' to 1 and 'unknown' to 0 in the 'clean_title' column
train_df['clean_title'] = train_df['clean_title'].map({'Yes': 1, 'Unknown': 0})


# Map 'yes' to 1 and 'unknown' to 0 in the 'accident' column
train_df['accident'] = train_df['accident'].map({'None reported': 0, 'At least 1 accident or damage reported': 1})


train_df.head()


# Step 1: Calculate Q1 and Q3
Q1 = train_df['price'].quantile(0.25)
Q3 = train_df['price'].quantile(0.75)

# Step 2: Calculate IQR
IQR = Q3 - Q1

# Step 3: Determine outlier thresholds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 4: Drop outliers
train_df = train_df[(train_df['price'] >= lower_bound) & (train_df['price'] <= upper_bound)]


Q1 = train_df['milage'].quantile(0.25)
Q3 = train_df['milage'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

train_df = train_df[(train_df['milage'] >= lower_bound) & (train_df['milage'] <= upper_bound)]


Q1 = train_df['horsepower'].quantile(0.25)
Q3 = train_df['horsepower'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

train_df = train_df[(train_df['horsepower'] >= lower_bound) & (train_df['horsepower'] <= upper_bound)]


Q1 = train_df['engine_size'].quantile(0.25)
Q3 = train_df['engine_size'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

train_df = train_df[(train_df['engine_size'] >= lower_bound) & (train_df['engine_size'] <= upper_bound)]


train_df = train_df.drop(columns=['id','engine','clean_title']) 
# with the help of correlation matrix, we figured out that the clean_title has no significant effect on the price


# Create a correlation matrix to decide which features mostly effect the price
correlation_matrix = train_df.corr(numeric_only=True)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create the heatmap
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})

# Title and show the plot
plt.title('Heatmap of Correlation Matrix')
plt.show()


train_df.head()


# Splitting the data into training and testing sets
X = train_df.drop(['price'], axis=1)  # Features
y = train_df['price']                  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define categorical features
categorical_features = ['brand', 'model', 'fuel_type','transmission']
numeric_features = ['model_year', 'milage', 'accident','horsepower', 'engine_size', 'cylinders']  

# Create the column transformer with handle_unknown set to 'ignore'
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ])

# Create a pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Fit the model
model_pipeline.fit(X_train, y_train)

# Predict using the same pipeline
y_pred = model_pipeline.predict(X_test)


train_df.head()


# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
r2 = r2_score(y_test, y_pred)

# Print the results
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'RÂ² Score: {r2:.2f}')


# Create a new pipeline with Random Forest

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Use the same preprocessor as before
    ('scaler', StandardScaler(with_mean=False)),  # Set with_mean=False
    ('model', RandomForestRegressor(n_estimators=100))
])

# Fit the model
rf_pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = rf_pipeline.predict(X_test)
rf_rmse = mean_squared_error(y_test, y_pred_rf, squared=False)
print(f'Random Forest RMSE: {rf_rmse:.2f}')



# Save the pipeline to a file
with open('rf_pipeline.pkl', 'wb') as file:
    pickle.dump(rf_pipeline, file)


r2 = r2_score(y_test, y_pred_rf)
print(f'Random Forest RÂ² Score: {r2:.2f}')


test_df=pd.read_csv('/kaggle/input/playground-series-s4e9/test.csv')


test_df.head()


# Apply the function to the engine column
engine_details = test_df['engine'].apply(extract_engine_details)

# Create new columns in the DataFrame
test_df = test_df.join(pd.DataFrame(engine_details.tolist()))


# Mean Imputation
test_df['horsepower'].fillna(test_df['horsepower'].median(), inplace=True)
test_df['engine_size'].fillna(test_df['engine_size'].median(), inplace=True)
test_df['cylinders'].fillna(test_df['cylinders'].median(), inplace=True)


test_df.isnull().sum()


test_df['fuel_type'].fillna('Electric', inplace=True)


test_df['accident'].fillna(test_df['accident'].mode()[0], inplace=True)


# Remove rows with "â€“" and "not supported"
test_df = test_df[~test_df['fuel_type'].isin(['â€“', 'not supported'])]


# Remove rows with "â€“" and "not supported"
test_df = test_df[~test_df['transmission'].isin(['â€“', 'Variable', 'F', 'SCHEDULED FOR OR IN PRODUCTION'])]


test_df['clean_title'].fillna('Unknown', inplace=True)


test_df.head()


train_df.head()


# Map 'yes' to 1 and 'unknown' to 0 in the 'accident' column
test_df['accident'] = test_df['accident'].map({'None reported': 0, 'At least 1 accident or damage reported': 1})


# 3. simplify transmission column
test_df['transmission'] = test_df['transmission'].replace({
        r'(?i).*Dual Shift.*': 'Dual Shift',
        r'(?i).*(automatic|A/T).*': 'Automatic',
        r'(?i).*(manual|M/T).*': 'Manual',
        r'(?i).*CVT.*': 'CVT'
    }, regex=True)

test_df['transmission'] =test_df['transmission'].where(test_df['transmission'].isin(['Dual Shift', 'Automatic', 'Manual', 'CVT']),'Other')


test_df=test_df.drop(columns=['engine','clean_title'],axis=1)


# Define categorical features
categorical_features = ['brand', 'model', 'fuel_type','transmission']
numeric_features = ['model_year', 'milage', 'accident','horsepower', 'engine_size', 'cylinders']  

# Create the column transformer with handle_unknown set to 'ignore'
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ])


test_df.isnull().sum()


# Prepare test data for predictions
submission = pd.DataFrame({'id': test_df['id']})
test_df.drop('id', axis=1, inplace=True)


# Make predictions on the test set
predictions = rf_pipeline.predict(test_df)


predictions


print(predictions.shape)


# Add predictions to submission DataFrame
submission['price'] = predictions

# Save submission
submission.to_csv('submission.csv', index=False)


# Scale the Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dense(24, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1)  # Output layer
])


# Compile
model.compile(optimizer="adam", loss="mse", metrics=["mae"])


# Train
history = model.fit(
    X_train_scaled, y_train,
    epochs=200, batch_size=16,
    validation_data=(X_test_scaled, y_test),
    verbose=1
)


# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate RÂ² score
r2 = r2_score(y_test, y_pred)

print(f'RÂ² Score: {r2}')




