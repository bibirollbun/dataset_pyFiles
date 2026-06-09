import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import holidays
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import KNNImputer
import xgboost as xgb
import lightgbm as lgb
import catboost as cb


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/playground-series-s5e1'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")


train_df.shape


train_df.head()


train_df.isnull().sum()


train_df['date'] = pd.to_datetime(train_df['date'])


train_df = train_df.dropna()


train_df['Year'] = train_df['date'].dt.year
train_df['Month'] = train_df['date'].dt.month
train_df['Day'] = train_df['date'].dt.day
train_df['Weekday'] = train_df['date'].dt.day_name()

train_df


# Define target variable
target = 'num_sold'

# Loop through each column
for column in train_df.columns:
    if column != target and column != 'Day':  # Skip the target column itself
        plt.figure(figsize=(8, 4))
        
        # Check if the column is numerical or categorical
        if train_df[column].dtype in ['int64', 'float64']:
            # Scatter plot for numerical columns
            sns.scatterplot(data=train_df, x=column, y=target)
            plt.title(f'{column} vs {target}')
        else:
            # Box plot for categorical columns
            sns.boxplot(data=train_df, x=column, y=target)
            plt.title(f'{column} vs {target}')
        
        plt.xlabel(column)
        plt.ylabel(target)
        plt.show()


sample = train_df.sample(n = 221259, random_state= 42)


sample


sample['Year'].max()


# Create holiday mappings
holiday_mappings = {
    'Canada': holidays.CA(years=range(2010, 2025)),
    'Singapore': holidays.SG(years=range(2010, 2025)),
    'Norway': holidays.NO(years=range(2010, 2025)),
    'Finland': holidays.FI(years=range(2010, 2025)),
    'Kenya': holidays.KE(years=range(2010, 2025)),
    'Italy': holidays.IT(years=range(2010, 2025))
}



# Function to check if a date is a holiday
def is_holiday(date, country):
    country_holidays = holiday_mappings.get(country, None)
    if country_holidays:
        # Convert pandas datetime to Python's date object
        return 1 if date.date() in country_holidays else 0
        print(country_holidays)



# Apply the function to create the 'is_holiday' column
sample['is_holiday'] = sample.apply(lambda row: is_holiday(row['date'], row['country']), axis=1)


# Function to calculate days until the next holiday
def days_until_next_holiday(date, country):
    country_holidays = holiday_mappings.get(country, None)
    if country_holidays:
        # Convert 'date' to a datetime.date object if it's a Timestamp
        date = date.date() if isinstance(date, pd.Timestamp) else date
        # Get the future holidays (already datetime.date, no need to call .date())
        future_holidays = [d for d in country_holidays if d >= date]
        
        if future_holidays:
            next_holiday = min(future_holidays)
            return (next_holiday - date).days
    return None  # Return None if no future holidays or unsupported country

# Apply the function to create a new column
sample['days_until_holiday'] = sample.apply(lambda row: days_until_next_holiday(row['date'], row['country']), axis=1)


# Function to calculate days after the holiday
def days_after_holiday(date, country):
    country_holidays = holiday_mappings.get(country, None)
    if country_holidays:
        # Convert 'date' to a datetime.date object if it's a Timestamp
        date = date.date() if isinstance(date, pd.Timestamp) else date
        # Get the past holidays (already datetime.date, no need to call .date())
        past_holidays = [d for d in country_holidays if d <= date]
        
        if past_holidays:
            last_holiday = max(past_holidays)
            return (date - last_holiday).days
    return None  # Return None if no past holidays or unsupported country

sample['days_after_holiday'] = sample.apply(lambda row: days_after_holiday(row['date'], row['country']), axis=1)


sample


print(sample['is_holiday'].value_counts())


# Generate a unique mapping for each country
country_mapping = {country: idx + 1 for idx, country in enumerate(sorted(sample['country'].unique()))}

# Apply the mapping to the 'Country' column
sample['country'] = sample['country'].map(country_mapping)


# Generate a unique mapping for each store 
store_mapping = {store: idx + 1 for idx, store in enumerate(sorted(sample['store'].unique()))}

# Apply the mapping to the 'Store' column
sample['store'] = sample['store'].map(store_mapping)


# Generate a unique mapping for each product
product_mapping = {product: idx + 1 for idx, product in enumerate(sorted(sample['product'].unique()))}

# Apply the mapping to the 'Product' column
sample['product'] = sample['product'].map(product_mapping)


# Generate a unique mapping for each weekday 
weekday_mapping = {weekday: idx + 1 for idx, weekday in enumerate(sorted(sample['Weekday'].unique()))}

# Apply the mapping to the 'Weekday' column
sample['Weekday'] = sample['Weekday'].map(weekday_mapping)


sample


#(modifies the sample DataFrame)
sample.drop(columns=['id', 'date'], inplace=True)


X = sample[['country', 'store', 'product', 'Year','Month','Weekday', 'days_until_holiday', 'days_after_holiday']]
y = sample['num_sold']


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Apply Standardization
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)


# Calculate mutual information for the training data
mi_scores = mutual_info_regression(X_train, y_train, random_state=42)

# Create a DataFrame for better visualization
mi_df = pd.DataFrame({"Feature": X_train.columns, "MI Score": mi_scores})
mi_df = mi_df.sort_values(by="MI Score", ascending=False)

print(mi_df)

# Select top features based on a threshold or top-k
threshold = 0.001  # Adjust this threshold
selected_features = mi_df[mi_df["MI Score"] > threshold]["Feature"]
print("Selected Features:", selected_features.tolist())


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100


models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"{name} - MAPE: {mape:.2f}%")


# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)

# Train the model 
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')
print(f"MAPE: {mape:.2f}%")


# 3. **CatBoost**
#cat_reg = cb.CatBoostRegressor(iterations=100, depth=10, learning_rate=0.1, loss_function='RMSE', verbose=0)
#cat_reg.fit(X_train, y_train)
#model = cb.CatBoostRegressor(loss_function='MAPE')

#grid = {
   # 'learning_rate': [0.01, 0.05, 0.1],
  #  'depth': [6, 8, 10],
 #   'iterations': [500, 1000]
#}

#grid_result = model.grid_search(grid, X=X_train, y=y_train)
# Predictions
#cat_pred = model.predict(X_test)
#cat_mse = mean_squared_error(y_test, cat_pred)
#cat_r2 = r2_score(y_test, cat_pred)
#mape = mean_absolute_percentage_error(y_test, cat_pred)
# Display best parameters
#print("Best Parameters:", grid_result['params'])
#print(f'CatBoost Mean Squared Error: {cat_mse}')
#print(f'CatBoost R^2 Score: {cat_r2}')
#print(f"MAPE: {mape:.2f}%")


# Define custom MAPE scorer
def custom_mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

mape_scorer = make_scorer(custom_mape, greater_is_better=False)  # Minimize MAPE


original_test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")


# Convert to datetime format
original_test_df['date'] = pd.to_datetime(original_test_df['date'])


original_test_df['Year'] = original_test_df['date'].dt.year
original_test_df['Month'] = original_test_df['date'].dt.month
original_test_df['Day'] = original_test_df['date'].dt.day
original_test_df['Weekday'] = original_test_df['date'].dt.day_name()
original_test_df


# Apply the function to create the 'is_holiday' column
original_test_df['is_holiday'] = original_test_df.apply(lambda row: is_holiday(row['date'], row['country']), axis=1)

# Apply the function to create a new column
original_test_df['days_until_holiday'] = original_test_df.apply(lambda row: days_until_next_holiday(row['date'], row['country']), axis=1)
original_test_df['days_after_holiday'] = original_test_df.apply(lambda row: days_after_holiday(row['date'], row['country']), axis=1)


# Generate a unique mapping for each country
country_mapping = {country: idx + 1 for idx, country in enumerate(sorted(original_test_df['country'].unique()))}

# Apply the mapping to the 'Country' column
original_test_df['country'] = original_test_df['country'].map(country_mapping)


# Generate a unique mapping for each store 
store_mapping = {store: idx + 1 for idx, store in enumerate(sorted(original_test_df['store'].unique()))}

# Apply the mapping to the 'Store' column
original_test_df['store'] = original_test_df['store'].map(store_mapping)


# Generate a unique mapping for each product
product_mapping = {product: idx + 1 for idx, product in enumerate(sorted(original_test_df['product'].unique()))}

# Apply the mapping to the 'Product' column
original_test_df['product'] = original_test_df['product'].map(product_mapping)


# Generate a unique mapping for each weekday 
weekday_mapping = {weekday: idx + 1 for idx, weekday in enumerate(sorted(original_test_df['Weekday'].unique()))}

# Apply the mapping to the 'Weekday' column
original_test_df['Weekday'] = original_test_df['Weekday'].map(weekday_mapping)


original_test_df


original_test_df = original_test_df.drop(columns=['id', 'date', 'Day', 'is_holiday'])


X_test = original_test_df


# Predict values
y_pred = rf_model.predict(X_test)

# Add predictions to test data
original_test_df["num_sold"] = y_pred


original_test_df


original_test_df["num_sold"] = y_pred.round(0).astype(int)


# Step 2: Add the ID column back (assuming you have the original test dataset with 'id')
final_test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")  # Reload original test dataset
original_test_df["id"] = final_test_df["id"]


submit_df = original_test_df[["id", "num_sold"]]


submit_df


# Step 4: Save the final DataFrame to a CSV file (if needed)
submit_df.to_csv("final_predictions4.csv", index=False)

