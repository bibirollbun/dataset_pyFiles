# import package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error


# Step 1: Load the data
# The training and test datasets are loaded with the 'date' column parsed as a datetime object.
# Parsing dates is crucial for time-series analysis as it allows us to extract time-related features like year, month, etc.
train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', parse_dates=['date'])
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', parse_dates=['date'])

# GDP data is loaded separately to incorporate external economic indicators, specifically GDP per capita.
gdp_data = pd.read_csv('/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv')

# print(train.head)
# print(test.head)
# print(gdp_data.head)


# Step 2: Feature engineering for train and test datasets
# Time-related features such as year, month, day of the week, and whether the day is a weekend are added.
# These features help capture temporal patterns in sticker sales.
for df in [train, test]:
    df['Year'] = df['date'].dt.year  # Extract the year from the date
    df['month'] = df['date'].dt.month  # Extract the month from the date
    df['day_of_week'] = df['date'].dt.dayofweek  # Extract the day of the week (0=Monday, 6=Sunday)
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)  # Mark weekends


# Step 3: Process GDP data
# The GDP dataset is in wide format (columns for each year). We transform it into a long format
# where each row represents a country, year, and its corresponding GDP per capita.
gdp_long = gdp_data.melt(id_vars=['Country Name'], var_name='Year', value_name='GDP per Capita')

# Convert the 'Year' column to numeric to ensure it aligns with the 'Year' in train and test datasets.
gdp_long['Year'] = pd.to_numeric(gdp_long['Year'], errors='coerce')

# Drop rows with missing or invalid GDP values.
gdp_long = gdp_long.dropna(subset=['Year', 'GDP per Capita'])


# Step 4: Merge GDP data with train and test datasets
# The GDP per capita is merged into the train and test datasets based on the country and year.
# This adds an external economic factor that may influence sticker sales.
train = train.merge(gdp_long, left_on=['country', 'Year'], right_on=['Country Name', 'Year'], how='left')
test = test.merge(gdp_long, left_on=['country', 'Year'], right_on=['Country Name', 'Year'], how='left')

# Drop the redundant 'Country Name' column after merging.
train = train.drop(columns=['Country Name'])
test = test.drop(columns=['Country Name'])


# Step 5: Handle missing target values in the training data
# Rows with missing 'num_sold' values are removed since they cannot be used for supervised learning.
train = train.dropna(subset=['num_sold'])


# Step 6: Define features and target variable
# We select features that capture temporal patterns and economic indicators for modeling.
features = ['month', 'day_of_week', 'is_weekend', 'GDP per Capita']
X = train[features]  # Features used for training
y = train['num_sold']  # Target variable representing the number of stickers sold


# Step 7: Split the data into training and validation sets
# A train-validation split is performed to evaluate the model on unseen data before testing.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 8: Train the Ridge regression model
# Ridge regression is chosen for its ability to handle multicollinearity and provide regularization.
# Regularization helps prevent overfitting, especially with correlated features.
ridge_model = Ridge(alpha=1.0)  # Regularization strength is set to 1.0.
ridge_model.fit(X_train, y_train)  # Train the model on the training dataset


# Step 9: Validate the model
# Predictions are made on the validation set, and the performance is evaluated using MAPE.
# MAPE (Mean Absolute Percentage Error) measures the prediction error as a percentage, making it intuitive.
y_pred = ridge_model.predict(X_val)
mape = mean_absolute_percentage_error(y_val, y_pred)
print(f"Validation MAPE: {mape}")


# Step 10: Make predictions on the test dataset
# The same features used for training are applied to the test dataset for predictions.
test['num_sold'] = ridge_model.predict(test[features])


# Step 11: Prepare the submission file
# The submission file includes the 'id' and the predicted 'num_sold' for each row in the test dataset.
submission = test[['id', 'num_sold']]
submission.to_csv('submission.csv', index=False)


# Step 12: Visualize validation results
# A scatter plot is used to compare the predicted values against the actual values from the validation set.
# This helps visually assess the model's accuracy and identify potential patterns or outliers.
plt.figure(figsize=(12, 6))
plt.scatter(y_val, y_pred, alpha=0.3)
plt.title("Validation Predictions vs Actual Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.grid(True)
plt.show()

print("Solution ready: submission.csv has been generated.")

