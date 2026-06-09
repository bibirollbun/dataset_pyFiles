import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


train_data=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
test_data=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')


train_data.shape


# load first few rowa
train_data.head()


test_data.head()


# Display basic information about the datase
train_data.info()

# Summary statistics
train_data.describe()


#check for missing values
train_data.isnull().sum()



# 1. Distribution of item prices
plt.figure(figsize=(10,6))
sns.histplot(train_data['item_price'], bins=50, kde=True)
plt.title('Distribution of Item Prices')
plt.xlabel('Item Price')
plt.ylabel('Frequency')
plt.show()


# 2. Distribution of item counts
plt.figure(figsize=(10,6))
sns.histplot(train_data['item_cnt_day'], bins=50, kde=True)
plt.title('Distribution of Item Counts')
plt.xlabel('Item Count')
plt.ylabel('Frequency')
plt.show()


# 3. Sales over time
train_data['date'] = pd.to_datetime(train_data['date'], format='%d.%m.%Y')
daily_sales = train_data.groupby('date')['item_cnt_day'].sum().reset_index()

plt.figure(figsize=(12,6))
sns.lineplot(data=daily_sales, x='date', y='item_cnt_day')
plt.title('Total Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Item Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 4. Sales by shop
shop_sales = train_data.groupby('shop_id')['item_cnt_day'].sum().reset_index()

plt.figure(figsize=(12,6))
sns.barplot(data=shop_sales, x='shop_id', y='item_cnt_day')
plt.title('Total Sales by Shop')
plt.xlabel('Shop ID')
plt.ylabel('Total Item Count')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# 11. Sales distribution by shop (pie chart)
shop_sales = train_data.groupby('shop_id')['item_cnt_day'].sum().reset_index()
plt.figure(figsize=(10, 10))
plt.pie(shop_sales['item_cnt_day'], labels=shop_sales['shop_id'], autopct='%1.1f%%', startangle=140)
plt.title('Sales Distribution by Shop')
plt.axis('equal')
plt.show()


# 5. Sales by item
item_sales = train_data.groupby('item_id')['item_cnt_day'].sum().reset_index()

plt.figure(figsize=(12,6))
sns.barplot(data=item_sales.sort_values('item_cnt_day', ascending=False).head(20), x='item_id', y='item_cnt_day')
plt.title('Top 20 Items by Sales')
plt.xlabel('Item ID')
plt.ylabel('Total Item Count')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Convert 'Date' to datetime format
train_data['date'] = pd.to_datetime(train_data['date'],format='%d.%m.%Y')

plt.figure(figsize=(10,6))
sns.lineplot(data=train_data, x='date', y='item_price', marker='o')
plt.title('Item Price Over Time')
plt.xlabel('Date')
plt.ylabel('Item Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 8. Correlation heatmap
plt.figure(figsize=(12,10))
correlation_matrix = train_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# 9. Monthly sales heatmap
monthly_sales_pivot = train_data.pivot_table(index=train_data['date'].dt.year, columns=train_data['date'].dt.month, values='item_cnt_day', aggfunc='sum')

plt.figure(figsize=(12,8))
sns.heatmap(monthly_sales_pivot, annot=True, fmt='g', cmap='viridis')
plt.title('Monthly Sales Heatmap')
plt.xlabel('Month')
plt.ylabel('Year')
plt.show()


train_data['year'] = train_data['date'].dt.year
train_data['month'] = train_data['date'].dt.strftime('%B')

# Define the desired sequence of months
month_sequence = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']

# Group by year and month, then calculate total sales
sales_yearly_monthly = train_data.groupby(['year', 'month'])['item_cnt_day'].sum().unstack()

# Reindex the columns to ensure the months are in the correct order
sales_yearly_monthly = sales_yearly_monthly.reindex(columns=month_sequence)

# Plot sales by month and by year
plt.figure(figsize=(12, 6))
sns.lineplot(data=sales_yearly_monthly.T)  # Transpose for correct plotting
plt.title('Total Sales of Item Count By Year and Month')
plt.xlabel('Month')
plt.ylabel('Item Count')
plt.legend(title='Year')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
from scipy import stats


# # Extract additional features from 'Date'
# train_data['Year'] = train_data['date'].dt.year
# train_data['Month'] =train_data['date'].dt.month
# train_data['Day'] = train_data['date'].dt.day
train_data['Day_of_Week'] =train_data['date'].dt.day_name()


train_data.head()


# Remove rows where any column contains negative values
train_data= train_data[train_data['item_cnt_day'] >= 0]


# Encode cetegorical data
categorical_cols = ['Day_of_Week']
# Apply Label Encoding
for col in categorical_cols:
    le=LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])


# Assuming df is your DataFrame
z_scores = stats.zscore(train_data.select_dtypes(include=['float64', 'int64']))  # Only for numeric columns
abs_z_scores = abs(z_scores)

# Set a threshold (commonly 3 or -3)
threshold = 2
train_data = train_data[(abs_z_scores < threshold).all(axis=1)]  # Remove rows where any column has outlier


train_data.shape


# Handle missing values
train_data.fillna(0, inplace=True)


# Rename the column
train_data.rename(columns={'item_cnt_day': 'Sales'}, inplace=True)

# Save the DataFrame to a CSV file
train_data.to_csv('Processed_data.csv', index=False)


# Load the processed data
import pandas as pd
processed_data = pd.read_csv('/kaggle/working/Processed_data.csv')
processed_data.head()



# Features Distribution
processed_data.shape
# Plot the distribution of each feature
processed_data.hist(bins=50, figsize=(20, 15))
plt.tight_layout()
plt.title("Features Distribution")
plt.show()


# List of numerical features
numerical_features = processed_data.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Remove the target column from the list of numerical features
numerical_features.remove('Sales')


# Plot box plots for each numerical feature against the target class 'Sales'
plt.figure(figsize=(20, 15))
for i, feature in enumerate(numerical_features):
    plt.subplot(len(numerical_features) // 3 + 1, 3, i + 1)
    sns.boxplot(x='Sales', y=feature, data=processed_data)
    plt.title(f'Box plot of {feature} vs Sales')
    plt.legend([feature], loc='upper right')
    plt.tight_layout()

plt.show()


# # Create pair plots
# plt.figure(figsize=(20, 15))
# sns.pairplot(processed_data, 
#              diag_kind='kde',  # kernel density estimate for diagonal plots
#              plot_kws={'alpha': 0.6},  # transparency of scatter plots
#              diag_kws={'alpha': 0.6})  # transparency of diagonal plots
# plt.suptitle("Pair Plots of Features", y=1.02)
# plt.show()

# For a subset of features if the plot is too large
# Select important features including the target
# features_subset = ['Sales', 'item_price', 'shop_id', 'item_id']
# sns.pairplot(processed_data[features_subset], 
#              diag_kind='kde',
#              plot_kws={'alpha': 0.6},
#              diag_kws={'alpha': 0.6})
# plt.suptitle("Pair Plots of Key Features", y=1.02)
# plt.show()


from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

# Prepare features and target
X = processed_data[numerical_features]
y = processed_data['Sales']

# Apply ANOVA F-test
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X, y)

# Get scores and p-values
feature_scores = pd.DataFrame({
    'Feature': numerical_features,
    'Score': selector.scores_,
    'P_value': selector.pvalues_
})

# Sort by score in descending order
feature_scores = feature_scores.sort_values('Score', ascending=False)

# Visualize feature importance scores
plt.figure(figsize=(12, 6))
sns.barplot(x='Score', y='Feature', data=feature_scores)
plt.title('Feature Importance Scores (ANOVA F-test)')
plt.xlabel('F-Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# Print feature scores
print("\nFeature Scores:")
print(feature_scores)


# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

num_features = ['item_id', 'item_price', 'year', 'Day_of_Week']
# Select all features except the target variable 'Sales'
X = processed_data[num_features]  # Dropping less significant columns
y = processed_data['Sales']  # Target variable

# Proceed with the encoded features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Evaluate performance
mse_lr = mean_squared_error(y_test, y_pred_lr)
# Mean Absolute Error (MAE)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Linear Regression MSE: {mse_lr}")
print(f"Linear Regression MAE: {mae_lr}")
print(f"Linear Regression R2 Score: {r2_lr}")


# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate performance
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)
print(f"Random Forest Regression MSE: {mse_rf}")
print(f"Linear Regression MAE: {mae_rf}")
print(f"Random Forest Regression MAPE: {mape_rf}")
print(f"Random Forest Regression R2 Score: {r2_rf}")


import matplotlib.pyplot as plt

# Scatter plot for Random Forest predictions
plt.scatter(y_test, y_pred_rf, alpha=0.3)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs. Predicted Sales (Random Forest)')
plt.show()


## Random Forest Feature Importance
# Get feature importances from the Random Forest model
importances = rf_model.feature_importances_
feature_names = X.columns
forest_importances = pd.Series(importances, index=feature_names)

# Sort the importances
sorted_importances = forest_importances.sort_values(ascending=False)

# Plot the feature importances
plt.figure(figsize=(8, 6))
sorted_importances.plot(kind='bar')
plt.title('Feature Importances from Random Forest Model')
plt.ylabel('Importance')
plt.show()


# K-NN Regression
from sklearn.neighbors import KNeighborsRegressor

knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# Evaluate performance
mse_knn = mean_squared_error(y_test, y_pred_knn)
mae_knn = mean_absolute_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)
print(f"K-NN Regression MSE: {mse_knn}")
print(f"K-NN Regression MAE: {mae_knn}")
print(f"K-NN Regression R2 Score: {r2_knn}")


# XGBoost Regression
from xgboost import XGBRegressor

xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate performance
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"XGBoost Regression MSE: {mse_xgb}")
print(f"XGBoost Regression MSE: {mae_xgb}")
print(f"XGBoost Regression R2 Score: {r2_xgb}")


# Long Short-Term Memory (LSTM)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Split the scaled data
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# Reshape input to be 3D [samples, timesteps, features]
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model
lstm_model.fit(X_train_lstm, y_train_scaled, epochs=10, batch_size=32, verbose=2)

# Make predictions
y_pred_lstm_scaled = lstm_model.predict(X_test_lstm)
y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)

# Evaluate performance
mse_lstm = mean_squared_error(y_test_actual, y_pred_lstm)
mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
r2_lstm = r2_score(y_test_actual, y_pred_lstm)
print(f"LSTM Regression MSE: {mse_lstm}")
print(f"LSTM Regression MSE: {mae_lstm}")
print(f"LSTM Regression R2 Score: {r2_lstm}")


# Visualize the results
import matplotlib.pyplot as plt

models = ['Linear Regression', 'Random Forest', 'K-NN', 'XGBoost', 'LSTM']
mse_scores = [mse_lr, mse_rf, mse_knn, mse_xgb, mse_lstm]
mae_scores = [mae_lr, mae_rf, mae_knn, mae_xgb, mae_lstm]
r2_scores = [r2_lr, r2_rf, r2_knn, r2_xgb, r2_lstm]


# Create a DataFrame
results_df = pd.DataFrame({
    'Model': models,
    'MSE': mse_scores,
    'MAE': mae_scores,
    'R2 Score': r2_scores
})


# Display the table
print(results_df)

# Plot MSE comparison
plt.figure(figsize=(10, 5))
plt.bar(models, mse_scores, color='skyblue')
plt.title('Model Comparison - Mean Squared Error')
plt.ylabel('MSE')
plt.show()

# Plot R2 Score comparison
plt.figure(figsize=(10, 5))
plt.bar(models, r2_scores, color='lightgreen')
plt.title('Model Comparison - R2 Score')
plt.ylabel('R2 Score')
plt.show()

