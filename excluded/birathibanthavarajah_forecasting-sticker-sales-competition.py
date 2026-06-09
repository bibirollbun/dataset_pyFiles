import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.simplefilter(action='ignore')
import seaborn as sns

# SNS styling via themes
sns.set_theme(style='darkgrid', palette='Set2')


df_train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")



df_train.head()


df_train.info()


df_train.shape


df_train.isnull().sum()


df_train = df_train.dropna(subset=['num_sold'])


df_train.isnull().sum()


df_train.shape


# Iterate through all columns and print unique values
for column in df_train.columns:
    print(f"Unique values in column '{column}':")
    print(df_train[column].unique())
    print("~"*50)


import seaborn as sns
import matplotlib.pyplot as plt 
for column in df_train[['product']]:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=column, data=df_train)
    plt.show()


import seaborn as sns
import matplotlib.pyplot as plt 
for column in df_train[['country']]:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=column, data=df_train)
    plt.show()


df_train['date'] = pd.to_datetime(df_train['date'])


df_train['year'] = df_train['date'].dt.year
df_train['month'] = df_train['date'].dt.month
df_train['day_of_week'] = df_train['date'].dt.dayofweek
df_train['is_weekend'] = df_train['day_of_week'].isin([5, 6])  # 5 = Saturday, 6 = Sunday


df_train.head()


df_train = pd.get_dummies(df_train, columns=['country', 'store', 'product'], drop_first=True)


df_train


import matplotlib.pyplot as plt

df_train_copy = df_train.copy()
# 1. Monthly Sales Analysis
monthly_sales = df_train_copy.groupby(df_train_copy['date'].dt.month)['num_sold'].sum()

plt.figure(figsize=(12, 6))
monthly_sales.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Total Stickers Sold by Month', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Total Stickers Sold')
plt.xticks(rotation=0)
plt.grid(True)
plt.show()

# 2. Quarterly Sales Analysis
df_train_copy['quarter'] = df_train_copy['date'].dt.to_period('Q')  # Adding a quarter column
quarterly_sales = df_train_copy.groupby('quarter')['num_sold'].sum()

plt.figure(figsize=(12, 6))
quarterly_sales.plot(kind='bar', color='orange', edgecolor='black')
plt.title('Total Stickers Sold by Quarter', fontsize=14)
plt.xlabel('Quarter')
plt.ylabel('Total Stickers Sold')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# 3. Day of the Week Sales Analysis
df_train_copy['day_of_week'] = df_train_copy['date'].dt.dayofweek
day_of_week_sales = df_train_copy.groupby('day_of_week')['num_sold'].sum()

# Plotting the sales by day of the week
plt.figure(figsize=(12, 6))
day_of_week_sales.plot(kind='bar', color='green', edgecolor='black')
plt.title('Total Stickers Sold by Day of the Week', fontsize=14)
plt.xlabel('Day of the Week')
plt.ylabel('Total Stickers Sold')
plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
plt.grid(True)
plt.show()



import matplotlib.pyplot as plt

df_train_plot = df_train.copy()

# 1. Distribution of num_sold
plt.figure(figsize=(10, 6))
plt.hist(df_train_plot['num_sold'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Number of Stickers Sold')
plt.xlabel('Number of Stickers Sold')
plt.ylabel('Frequency')
plt.show()

# 2. Sales Trends Over Time (using a smaller sample if needed to avoid overplotting)
df_train_plot['date'] = pd.to_datetime(df_train_plot['date'])
df_train_plot.set_index('date', inplace=True)

# Plot the sales trend for the entire dataset (resampling to monthly frequency)
monthly_sales = df_train_plot['num_sold'].resample('M').sum()

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales, color='orange', lw=2)
plt.title('Sales Trends Over Time (Monthly)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Total Stickers Sold')
plt.grid(True)
plt.show()



df_train


# Calculate the IQR (Interquartile Range)
Q1 = df_train['num_sold'].quantile(0.25)
Q3 = df_train['num_sold'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for detecting outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out the outliers from the data
df_train_no_outliers = df_train[(df_train['num_sold'] >= lower_bound) & (df_train['num_sold'] <= upper_bound)]

# Show the shape of the dataset before and after removing outliers
print(f"Original dataset shape: {df_train.shape}")
print(f"Dataset shape after outlier removal: {df_train_no_outliers.shape}")



from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Scaling the 'num_sold' data (standardization)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_train_no_outliers[['num_sold']])

# Applying KMeans clustering with a specified number of clusters (e.g., 4 clusters)
kmeans = KMeans(n_clusters=4, random_state=42)
df_train_no_outliers['sales_cluster'] = kmeans.fit_predict(df_scaled)

# Now, you can analyze the clusters
print(df_train_no_outliers.groupby('sales_cluster')['num_sold'].describe())

# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(df_train_no_outliers['date'], df_train_no_outliers['num_sold'], c=df_train_no_outliers['sales_cluster'], cmap='viridis', s=10)
plt.title('Sales Clusters')
plt.xlabel('Date')
plt.ylabel('Number of Stickers Sold')
plt.show()



plt.figure(figsize=(12, 6))
for cluster in df_train_no_outliers['sales_cluster'].unique():
    cluster_data = df_train_no_outliers[df_train_no_outliers['sales_cluster'] == cluster]
    plt.plot(cluster_data['date'], cluster_data['num_sold'], label=f'Cluster {cluster}')

plt.title('Sales Over Time by Cluster')
plt.xlabel('Date')
plt.ylabel('Number of Stickers Sold')
plt.legend()
plt.show()



df_train_no_outliers


# Step 1: Prepare the Data

# We'll keep the necessary features and drop the target variable for X (features)
X = df_train_no_outliers.drop(columns=['num_sold', 'date', 'id', 'sales_cluster'])  # Drop target and non-numeric features
y = df_train_no_outliers['num_sold']  # Target variable


X.columns


X


# Step 2: Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Step 3: Model Training (Random Forest or XGBoost)

from sklearn.ensemble import RandomForestRegressor

# Initialize the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
rf_model.fit(X_train, y_train)

# Step 4: Evaluate the Model
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Predict on the test data
y_pred = rf_model.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")


# Load the test set
df_test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


df_test.columns


df_test


df_test.isnull().sum()


# Preprocessing the test set in the same way as the train set
df_test['date'] = pd.to_datetime(df_test['date'])
df_test['year'] = df_test['date'].dt.year
df_test['month'] = df_test['date'].dt.month
df_test['day_of_week'] = df_test['date'].dt.dayofweek
df_test['is_weekend'] = df_test['day_of_week'].isin([5, 6])


# One-hot encoding the categorical columns
df_test = pd.get_dummies(df_test, columns=['country', 'store', 'product'], drop_first=True)

# Make sure to add the 'sales_cluster' feature from the model
# df_test['sales_cluster'] = kmeans.predict(scaler.transform(df_test[['num_sold']]))  # If applicable



# # 2. Prepare features for prediction (drop 'id' and 'date' columns)
# X_test = df_test.drop(columns=['date', 'id'])

# # 3. Make predictions
# y_pred_test = rf_model.predict(X_test)

# # 4. Prepare the submission DataFrame
# submission = pd.DataFrame({
#     'id': df_test['id'],
#     'num_sold': y_pred_test
# })

# # Save the submission file
# submission.to_csv('/kaggle/working/submission.csv', index=False)


# Features and target
X = df_train_no_outliers.drop(columns=['id', 'date', 'num_sold', "sales_cluster"])
y = df_train_no_outliers['num_sold']

# Split into training and validation sets (chronological split)
train_size = int(0.8 * len(df_train))  # 80% for training, 20% for validation
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]



#del X, y


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Train the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions and evaluate
y_pred_lr = lr.predict(X_val)
mae_lr = mean_absolute_error(y_val, y_pred_lr)
print(f"Linear Regression MAE: {mae_lr}")



from sklearn.ensemble import RandomForestRegressor

# Train the model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions and evaluate
y_pred_rf = rf.predict(X_val)
mae_rf = mean_absolute_error(y_val, y_pred_rf)
print(f"Random Forest MAE: {mae_rf}")



# For Random Forest
feature_importances_rf = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Random Forest Feature Importances:")
print(feature_importances_rf)



import xgboost as xgb

# Train the model
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xg_reg.fit(X_train, y_train)

# Make predictions and evaluate
y_pred_xg = xg_reg.predict(X_val)
mae_xg = mean_absolute_error(y_val, y_pred_xg)
print(f"XGBoost MAE: {mae_xg}")



# For XGBoost
feature_importances_xg = pd.Series(xg_reg.feature_importances_, index=X.columns).sort_values(ascending=False)
print("XGBoost Feature Importances:")
print(feature_importances_xg)



from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# RandomizedSearchCV for Random Forest
param_dist_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create RandomizedSearchCV object
random_search_rf = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=42),
                                      param_distributions=param_dist_rf,
                                      n_iter=10, cv=3, n_jobs=-1, verbose=2,
                                      random_state=42, scoring='neg_mean_absolute_error')

random_search_rf.fit(X_train, y_train)

# Get the best parameters and model
print("Best parameters for Random Forest:", random_search_rf.best_params_)
best_rf_model = random_search_rf.best_estimator_

# Evaluate the tuned model
y_pred_rf_tuned = best_rf_model.predict(X_val)
mae_rf_tuned = mean_absolute_error(y_val, y_pred_rf_tuned)
print(f"Tuned Random Forest MAE: {mae_rf_tuned}")



from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# RandomizedSearchCV for XGBoost
param_dist_xg = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 6, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Create RandomizedSearchCV object
random_search_xg = RandomizedSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
                                      param_distributions=param_dist_xg,
                                      n_iter=10, cv=3, n_jobs=-1, verbose=2,
                                      random_state=42, scoring='neg_mean_absolute_error')

# Fit the model
random_search_xg.fit(X_train, y_train)

# Get the best parameters and model
print("Best parameters for XGBoost:", random_search_xg.best_params_)
best_xg_model = random_search_xg.best_estimator_

# Evaluate the tuned model
y_pred_xg_tuned = best_xg_model.predict(X_val)
mae_xg_tuned = mean_absolute_error(y_val, y_pred_xg_tuned)
print(f"Tuned XGBoost MAE: {mae_xg_tuned}")



from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

# Define the base models and meta-model
base_learners = [
    ('random_forest', best_rf_model),
    ('xgboost', best_xg_model)
]

# Meta-model (We can use a simple Linear Regression as the meta-model)
meta_model = LinearRegression()

# Create the Stacking Regressor
stacking_model = StackingRegressor(estimators=base_learners, final_estimator=meta_model)

# Train the stacking model
stacking_model.fit(X_train, y_train)

# Evaluate the stacking model
y_pred_stacked = stacking_model.predict(X_val)
mae_stacked = mean_absolute_error(y_val, y_pred_stacked)
print(f"Stacked Model MAE: {mae_stacked}")



# Prepare the test data
X_test = pd.get_dummies(df_test.drop(columns=['id', 'date']), drop_first=True)

# Make predictions using the stacked model
test_predictions = stacking_model.predict(X_test)

# Prepare the submission DataFrame
submission = pd.DataFrame({'id': df_test['id'], 'num_sold': test_predictions})

# Save to CSV for submission
submission.to_csv('submission.csv', index=False)


