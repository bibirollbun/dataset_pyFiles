import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")  
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv") 


train.describe()


train =train.dropna()



# Iterate through each column and print the number of unique values
for col in train.columns:
    print(f"Column '{col}' has {train[col].nunique()} unique values.")


import matplotlib.pyplot as plt
import seaborn as sns

plt.hist(train["store"])


plt.hist(train["product"])


for i in train['product'].unique():
    sns.histplot(data=train[train['product'] == i], x='num_sold')


from sklearn.preprocessing import LabelEncoder

for col in train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])

for col in test.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    test[col] = le.fit_transform(test[col])



# Extract date features 

df = train
df['date'] = pd.to_datetime(df['date'])


df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
df['day_of_year'] = df['date'].dt.dayofyear  # Helps capture annual seasonality
df['week_of_year'] = df['date'].dt.isocalendar().week
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days


# check for any outliers in num_sold & eliminate them by taking log

#cCalculate Q1 (25th percentile), Q3 (75th percentile), and IQR
Q1 = np.percentile(train['num_sold'], 25)
Q3 = np.percentile(train['num_sold'], 75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = train['num_sold'][(train['num_sold'] < lower_bound) | (train['num_sold'] > upper_bound)]

print(f"Outliers: {outliers}")
outliers_sum = np.sum(outliers)

df['num_sold_log'] = np.log1p(df['num_sold'])  # Log transformation


# Using K-means for searching any patters or clusters for independent features

from sklearn.cluster import KMeans

# Select features for clustering
features_for_clustering = df[[ 'days_since_start', 'month', 'day_of_week', 
                              'country', 'store', 'product']]

kmeans = KMeans(random_state=42)
df['cluster'] = kmeans.fit_predict(features_for_clustering)

print(df[['num_sold', 'cluster']].head())


from xgboost import XGBRegressor

# Drop the original 'date' column (since we've extracted useful features)
X_train = df.drop(columns=['date','num_sold', 'num_sold_log'])
y_train = df['num_sold_log']


# Train-test split (time series: no shuffle)
# Split the data chronologically (the test set comes after the training set)
train_size = int(len(X_train) * 0.8)
X_train, X_val = X_train[:train_size], X_train[train_size:]
y_train, y_val = y_train[:train_size], y_train[train_size:]

# Initialize and train the XGBoost model
model = XGBRegressor()
model.fit(X_train, y_train)

# Make predictions on the test set (using only features, no lag)
test_predictions = model.predict(X_val)

from sklearn.metrics import mean_squared_error


rmse = np.sqrt(mean_squared_error(y_val, test_predictions))

# Print the predictions for the test set
rmse


# pre processing for test data

df = test
df['date'] = pd.to_datetime(df['date'])


df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
df['day_of_year'] = df['date'].dt.dayofyear  # Helps capture annual seasonality
df['week_of_year'] = df['date'].dt.isocalendar().week
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

# Select features for clustering
features_for_clustering = df[[ 'days_since_start', 'month', 'day_of_week', 
                              'country', 'store', 'product']]

kmeans = KMeans(random_state=42)
df['cluster'] = kmeans.fit_predict(features_for_clustering)

test = df.drop(columns = ['date'])




num_sold = model.predict(test)
num_sold = np.expm1(num_sold)

submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
submission['num_sold'] = num_sold
submission['num_sold'] = submission['num_sold'].astype(int)

submission


submission.to_csv("output_file.csv", index=False)




