# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift-Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Import pandas for data manipulation and warnings to manage notifications
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# Load the store information dataset
store_df = pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')
store_df.head()


# Load the training dataset, which contains the historical sales data
raw_df = pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv')
raw_df.head()


# Merge the training data with the store information using a left merge on the 'Store' ID
merged_df = raw_df.merge(store_df, how='left', on='Store')
merged_df


# Load the test dataset
test_df = pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv')
# Merge the test data with the store information, just as we did with the training data
merged_test_df= test_df.merge(store_df, how='left', on='Store')
merged_test_df.head()


# Define a function to extract date components from the 'Date' column
def split_date(df):
    # Convert 'Date' column to datetime objects
    df['Date'] = pd.to_datetime(df['Date'])
    # Extract year, month, day, and week of the year
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['WeekOfYear'] = df.Date.dt.isocalendar().week


# Apply the function to both the training and test DataFrames
split_date(merged_df)
split_date(merged_test_df)


# Check the sales values for rows where the store is closed
merged_df[merged_df.Open == 0].Sales.value_counts()
# As expected, when the store is closed, sales are always zero.


# We will train our model only on the data where stores were open.
# We create a copy to avoid SettingWithCopyWarning.
merged_df = merged_df[merged_df.Open == 1].copy()


# Define a function to calculate the duration of competition in months
def comp_months(df):
    # Calculate the number of months since the competition opened
    df['CompetitionOpen'] = 12 * (df.Year - df.CompetitionOpenSinceYear) + (df.Month - df.CompetitionOpenSinceMonth)
    # If the competition opened in the future, set the duration to 0. Fill any NaNs with 0.
    df['CompetitionOpen'] = df['CompetitionOpen'].map(lambda x: 0 if x < 0 else x).fillna(0)


# Apply the competition feature engineering to both datasets
comp_months(merged_df)
comp_months(merged_test_df)


# Display the newly created 'CompetitionOpen' feature
merged_df.CompetitionOpen


# Transpose the DataFrame to view all columns for a few rows
merged_df.head().T


# Helper function to check if the current month is a Promo2 month
def check_promo_month(row):
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',      
                 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    try:
        # Get the list of promo months (e.g., ['Jan', 'Apr', ...])
        months = (row['PromoInterval'] or '').split(',')
        # Check if the store has Promo2 and if the current month is in the list
        if row['Promo2Open'] and month2str[row['Month']] in months:
            return 1
        else:
            return 0
    except Exception:
        return 0

# Main function to create promotion-related columns
def promo_cols(df):
    # Calculate months since Promo2 was active
    df['Promo2Open'] = 12 * (df.Year - df.Promo2SinceYear) + (df.WeekOfYear - df.Promo2SinceWeek)*7/30.5
    # Set to 0 if negative and fill NaNs. Only keep the value if Promo2 is active for the store.
    df['Promo2Open'] = df['Promo2Open'].map(lambda x: 0 if x < 0 else x).fillna(0) * df['Promo2']
    
    # Check if the current month is a promotion month
    df['IsPromo2Month'] = df.apply(check_promo_month, axis=1) * df['Promo2']


# Apply the promotion feature engineering to both datasets
promo_cols(merged_df)
promo_cols(merged_test_df)


# Display the DataFrame with all the new features
merged_df


# List all column names for reference
merged_df.columns


# Define the columns we will use as inputs (features) and the target variable
input_cols = ['Store', 'DayOfWeek', 'Promo',
        'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
        'CompetitionDistance', 'Promo2', 'Year', 'Month', 'Day',
        'WeekOfYear', 'CompetitionOpen', 'Promo2Open', 'IsPromo2Month']
target_col = 'Sales'


# Create the input and target DataFrames for the training set
inputs = merged_df[input_cols].copy()
targets = merged_df[target_col].copy()


# Create the input DataFrame for the test set
test_inputs = merged_test_df[input_cols].copy()


# Separate columns into numerical and categorical types
categorical_cols = ['DayOfWeek', 'StateHoliday', 'StoreType', 'Assortment']
numeric_cols = list(set(input_cols) - set(categorical_cols))


# Check for missing values in the numerical columns
inputs[numeric_cols].isna().sum()


# Get the maximum competition distance
max_dist = inputs.CompetitionDistance.max()
# Fill missing values with a value larger than any existing distance
inputs['CompetitionDistance'].fillna(max_dist*2, inplace=True)
test_inputs['CompetitionDistance'].fillna(max_dist*2, inplace=True)


# Import the MinMaxScaler
from sklearn.preprocessing import MinMaxScaler


# Initialize the scaler
scaler = MinMaxScaler()
# Fit the scaler on the training data's numeric columns
scaler.fit(inputs[numeric_cols])


# Transform the numeric columns in both the training and test sets
inputs[numeric_cols] = scaler.transform(inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])


# Import the OneHotEncoder
from sklearn.preprocessing import OneHotEncoder


# Ensure categorical columns are of string type for the encoder
inputs[categorical_cols] = inputs[categorical_cols].astype(str)
test_inputs[categorical_cols] = test_inputs[categorical_cols].astype(str)


# Initialize the encoder. sparse=False returns a numpy array. handle_unknown='ignore' prevents errors on unseen test data categories.
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
# Fit the encoder on the training data's categorical columns
encoder.fit(inputs[categorical_cols])


# Get the names of the new encoded columns
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))


# Transform the categorical columns and add the new encoded columns to our DataFrames
inputs[encoded_cols] = encoder.transform(inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])


# Create the final training and test sets with all processed features
X = inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]


# Import the XGBRegressor model
from xgboost import XGBRegressor
# Initialize the model with some basic parameters
model = XGBRegressor(random_state=42, n_jobs=-1, n_estimators=20, max_depth=4)


# Train the model on the entire training dataset
model.fit(X, targets)


# Make predictions on the training data to see how well it fits
preds = model.predict(X)


# Import the mean_squared_error metric
from sklearn.metrics import mean_squared_error

# Define a function to calculate RMSE
def rmse(a, b):
    return mean_squared_error(a,b, squared=False)


# Calculate the RMSE on the training predictions
# Note: This is not a validation score and is likely overly optimistic.
rmse(preds, targets)


# Create a DataFrame of feature importances
importance_df = pd.DataFrame({
    'features': X.columns,
    'importance' : model.feature_importances_
}).sort_values('importance', ascending=False)


# Display the top 10 most important features
importance_df.head(10)


# Visualize the feature importances
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.barplot(data=importance_df.head(10), x='importance', y='features')
plt.title('Top 10 Feature Importances')
plt.show()


# Import KFold from scikit-learn
from sklearn.model_selection import KFold


# Define a function to train a model and return its train/validation RMSEs
def train_evaluate(X_train, train_targets, X_val, val_targets, **params):
    model = XGBRegressor(random_state=42, n_jobs=-1, **params)
    model.fit(X_train, train_targets)
    train_rmse = rmse(model.predict(X_train), train_targets)
    val_rmse = rmse(model.predict(X_val), val_targets)
    return model, train_rmse, val_rmse


# Initialize 5-fold cross-validation
kfold = KFold(n_splits=5)


# Loop through the folds, train a model for each, and store them
models = []
for train_idxs, val_idxs in kfold.split(X):
    X_train, train_targets = X.iloc[train_idxs], targets.iloc[train_idxs]
    X_val, val_targets = X.iloc[val_idxs], targets.iloc[val_idxs]
    
    model, train_rmse, val_rmse = train_evaluate(X_train, 
                                                 train_targets, 
                                                 X_val, 
                                                 val_targets, 
                                                 max_depth=5, 
                                                 n_estimators=20)
    
    models.append(model)
    print(f'TrainRMSE: {train_rmse:.4f}, ValRMSE: {val_rmse:.4f}')


# Define a function to average the predictions from a list of models
def predict_avg(models, inputs):
    return np.mean([model.predict(inputs) for model in models], axis=0)


# Make predictions on the training set using the ensembled models
preds = predict_avg(models, X)
preds


# Create a simple train-validation split for quick hyperparameter testing
from sklearn.model_selection import train_test_split
X_train, X_val, train_targets, val_targets = train_test_split(X, targets, test_size=0.1, random_state=42)


# Define a function to quickly test different hyperparameters
def test_params(**params):
    model = XGBRegressor(n_jobs=-1, random_state=42, **params)
    model.fit(X_train, train_targets)
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_rmse = rmse(train_pred, train_targets)
    val_rmse = rmse(val_pred, val_targets)
    print(f'Parameters: {params}')
    print(f'Train RMSE: {train_rmse:.4f}, Validation RMSE: {val_rmse:.4f}\n')


# Tuning n_estimators
test_params(n_estimators=50)
test_params(n_estimators=120)
test_params(n_estimators=240)
test_params(n_estimators=300)


# Tuning max_depth
test_params(max_depth=7)
test_params(max_depth=10)
test_params(max_depth=15)


# Tuning learning_rate
test_params(n_estimators=50, learning_rate=0.5)
test_params(n_estimators=100, learning_rate=0.7)


# Initialize the final model with tuned hyperparameters
# Note: These parameters are a starting point and can be further optimized.
model = XGBRegressor(n_jobs=-1, 
                     random_state=42, 
                     n_estimators=1000, 
                     learning_rate=0.2, 
                     max_depth=10, 
                     subsample=0.9, 
                     colsample_bytree=0.7)


# Train the final model on all available training data
model.fit(X, targets)


# Make predictions on the test data
test_preds = model.predict(X_test)


# Load the sample submission file to get the correct format
submission_df = pd.read_csv('/kaggle/input/rossmann-store-sales/sample_submission.csv')
submission_df


# Assign our predictions to the 'Sales' column
submission_df['Sales'] = test_preds


# Check for missing values in the 'Open' column of the original test data
test_df.Open.isna().sum()


# Multiply our predictions by the 'Open' column.
# If a store is open (1), the prediction remains. If closed (0), it becomes 0.
# We fill any potential NaNs in 'Open' with 1 (assuming open if not specified).
submission_df['Sales'] = submission_df['Sales'] * test_df.Open.fillna(1.0)


# Ensure the final sales are integers and handle the 'Open' column correctly
submission_df['Sales'] = test_preds * test_df['Open'].fillna(1).astype('int')


# Save the submission file
submission_df.to_csv('submission.csv', index=None)
print("Submission file created successfully!")

