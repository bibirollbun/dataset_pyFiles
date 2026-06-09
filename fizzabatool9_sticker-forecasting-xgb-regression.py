import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")  
test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")




train_df = train_df.dropna()  
test_df = test_df.dropna()  


for col in train_df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])


for col in test_df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    test_df[col] = le.fit_transform(test_df[col])




Q1 = np.percentile(train_df['num_sold'], 25)
Q3 = np.percentile(train_df['num_sold'], 75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = train_df['num_sold'][(train_df['num_sold'] < lower_bound) | (train_df['num_sold'] > upper_bound)]

print(f"Outliers: {outliers}")
outliers_sum = np.sum(outliers)
df = train_df

df['num_sold_log'] = np.log1p(df['num_sold'])  



df['date'] = pd.to_datetime(df['date'])


df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
df['day_of_year'] = df['date'].dt.dayofyear  # Helps capture annual seasonality
df['week_of_year'] = df['date'].dt.isocalendar().week
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days







# Features and target variables
X_train = df.drop(columns=['num_sold','date','num_sold_log', 'id'])
y_train = df['num_sold_log']

# Train-test split (time series: no shuffle)
# Split the data chronologically (the test set comes after the training set)
train_size = int(len(X_train) * 0.8)
X_train, X_val = X_train[:train_size], X_train[train_size:]
y_train, y_val = y_train[:train_size], y_train[train_size:]


X_train


from xgboost import XGBRegressor

# Initialize and train the XGBoost model
model = XGBRegressor()
model.fit(X_train, y_train)

# Make predictions on the test set (using only features, no lag)
test_predictions = model.predict(X_val)


from sklearn.metrics import mean_squared_error


rmse = np.sqrt(mean_squared_error(y_val, test_predictions))

# Print the predictions for the test set
rmse


# test data preprocessing

test_df['date'] = pd.to_datetime(test_df['date'])


test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day_of_week'] = test_df['date'].dt.dayofweek  # Monday=0, Sunday=6
test_df['day_of_year'] = test_df['date'].dt.dayofyear  # Helps capture annual seasonality
test_df['week_of_year'] = test_df['date'].dt.isocalendar().week
test_df['days_since_start'] = (test_df['date'] - test_df['date'].min()).dt.days

test_df = test_df.drop(columns =['date', 'id'])


num_sold = model.predict(test_df)



num_sold = np.expm1(num_sold)

submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
submission['num_sold'] = num_sold
submission['num_sold'] = submission['num_sold'].astype(int)

submission


submission.to_csv("output_file.csv", index=False)

