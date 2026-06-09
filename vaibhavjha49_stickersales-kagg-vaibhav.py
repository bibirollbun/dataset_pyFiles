! pip install kaggle


from ast import Index
import pandas as pd

df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")


df.head(5)


# prompt: find how many na values are there in the num_sold column in df

df['num_sold'].isna().sum()


df['num_sold'].sum()


# Calculate the mean of 'num_sold', ignoring NaN values
mean_num_sold = df['num_sold'].mean()

# Fill NaN values in 'num_sold' with the calculated mean
df['num_sold'].fillna(mean_num_sold, inplace=True)

# Verify the changes (optional)
print(df['num_sold'].isna().sum())  # Should print 0



df['num_sold'].isna().sum()


df.head(5)


df['num_sold'] = df['num_sold'].round(2)


df.head(5)


df['product'].value_counts()


df['store'].value_counts()


df['country'].value_counts()


# prompt: generate a graph for total num_sold by each country in the df dataframe above

import matplotlib.pyplot as plt
import seaborn as sns

# Group by country and sum num_sold
country_sales = df.groupby('country')['num_sold'].sum()

# Create the plot
plt.figure(figsize=(12, 6))
sns.barplot(x=country_sales.index, y=country_sales.values)
plt.xlabel('Country')
plt.ylabel('Total num_sold')
plt.title('Total num_sold by Country')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()


df.groupby('country')['num_sold'].sum()


# prompt: generate a time series graph for total num_sold by each month in the df dataframe above

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'date' column exists and is in datetime format
# If not, convert it first: df['date'] = pd.to_datetime(df['date'])

# Convert 'date' column to datetime objects if it's not already
df['date'] = pd.to_datetime(df['date'])

# Extract year and month
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Group by year and month, then sum num_sold
monthly_sales = df.groupby(['year', 'month'])['num_sold'].sum().reset_index()

# Create a datetime index for plotting
monthly_sales['date'] = pd.to_datetime(monthly_sales[['year', 'month']].assign(DAY=1))
monthly_sales = monthly_sales.set_index('date')

# Plot the time series
plt.figure(figsize=(12, 6))
sns.lineplot(x=monthly_sales.index, y=monthly_sales['num_sold'])
plt.xlabel('Date')
plt.ylabel('Total num_sold')
plt.title('Total num_sold by Month')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


df.head(5)


# prompt: plot a graph to identify num_sold group it by product

import matplotlib.pyplot as plt
import seaborn as sns

# Group by product and sum num_sold
product_sales = df.groupby('product')['num_sold'].sum()

# Create the plot
plt.figure(figsize=(12, 6))
sns.barplot(x=product_sales.index, y=product_sales.values)
plt.xlabel('Product')
plt.ylabel('Total num_sold')
plt.title('Total num_sold by Product')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()


product_sales = df.groupby('store')['num_sold'].sum()

# Create the plot
plt.figure(figsize=(12, 6))
sns.barplot(x=product_sales.index, y=product_sales.values)
plt.xlabel('store')
plt.ylabel('Total num_sold')
plt.title('Total num_sold by store')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()


# prompt: use time series analysis to create a predictive model to forecast num_sold in the df dataframe

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Prepare the time series data
# Assuming 'monthly_sales' DataFrame from the previous code
# with 'date' as datetime index and 'num_sold' as the values
# If you have a different DataFrame or data format, adapt this part
train_data = monthly_sales['num_sold']


# Split the data into training and testing sets (e.g., 80/20 split)
train_size = int(len(train_data) * 0.8)
train, test = train_data[:train_size], train_data[train_size:]

# Fit the ARIMA model (adjust (p, d, q) order as needed)
model = ARIMA(train, order=(5,1,0)) # Example order, tune for optimal performance
model_fit = model.fit()

# Make predictions on the test set
predictions = model_fit.predict(start=len(train), end=len(train_data)-1)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(test, predictions))
print(f'Root Mean Squared Error: {rmse}')


# Forecast future values (e.g., the next 6 months)
forecast_steps = 6
forecast = model_fit.forecast(steps=forecast_steps)

# Print the forecast
print(f'Forecast for the next {forecast_steps} months:')
print(forecast)

# Plot the forecast
plt.figure(figsize=(12,6))
plt.plot(train_data.index, train_data, label='Actual')
plt.plot(test.index, predictions, label='Predictions')
plt.plot(pd.date_range(start=train_data.index[-1], periods=forecast_steps+1)[1:], forecast, label='Forecast')
plt.xlabel('Date')
plt.ylabel('Num Sold')
plt.title('Time Series Forecast')
plt.legend()
plt.show()


# prompt: create a df_copy dataframe from df dataframe and label encode store, product and country columns

from sklearn.preprocessing import LabelEncoder

# Create a copy of the DataFrame
df_copy = df.copy()

# Initialize LabelEncoder
le = LabelEncoder()

# Fit and transform the 'store', 'product', and 'country' columns
for col in ['store', 'product', 'country']:
    df_copy[col] = le.fit_transform(df_copy[col])

df_copy.head(5)


product_sales = df_copy.groupby('store')['num_sold'].sum()

# Create the plot
plt.figure(figsize=(12, 6))
sns.barplot(x=product_sales.index, y=product_sales.values)
plt.xlabel('store')
plt.ylabel('Total num_sold')
plt.title('Total num_sold by store')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()


product_sales = df_copy.groupby('product')['num_sold'].sum()

# Create the plot
plt.figure(figsize=(12, 6))
sns.barplot(x=product_sales.index, y=product_sales.values)
plt.xlabel('Product')
plt.ylabel('Total num_sold')
plt.title('Total num_sold by Product')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()


country_sales = df_copy.groupby('country')['num_sold'].sum()

# Create the plot
plt.figure(figsize=(12, 6))
sns.barplot(x=country_sales.index, y=country_sales.values)
plt.xlabel('Country')
plt.ylabel('Total num_sold')
plt.title('Total num_sold by Country')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()


# prompt: use df_copy to create a correlation matrix with target variable as num_sold and plot it in a heatmap

# Calculate the correlation matrix
correlation_matrix = df_copy.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# prompt: use ensemble learning method to predict num_sold with features - date, country, store and product

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# ... (Your existing code for data loading and preprocessing) ...

# Feature Engineering (if needed) - Example:
# You might create features like day of the week, week of the year, etc.

# Prepare data for modeling
X = df_copy[['year','store', 'product', 'country']]
y = df_copy['num_sold']


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train multiple models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42) # Adjust parameters as needed
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42) # Adjust parameters as needed

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Make predictions using both models
rf_predictions = rf_model.predict(X_test)
gb_predictions = gb_model.predict(X_test)

# Ensemble predictions (simple average)
ensemble_predictions = (rf_predictions + gb_predictions) / 2


# Example prediction for a new data point
# new_data_point = pd.DataFrame({'date': [pd.to_datetime('2024-01-15').toordinal()],
#                               'store': [0], 'product': [1], 'country': [2]})

# ensemble_prediction = (rf_model.predict(new_data_point) + gb_model.predict(new_data_point)) / 2
# print(f"Prediction for new data point: {ensemble_prediction}")


# prompt: plot a confusion matrix and determine the precision, recall, accuracy and f1 score for the above model

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming 'ensemble_predictions' and 'y_test' are defined from your previous code

# Convert predictions to classes (e.g., using a threshold)
threshold = np.mean(y_test) # Example threshold - you may need to adjust this
predicted_classes = (ensemble_predictions >= threshold).astype(int)
actual_classes = (y_test >= threshold).astype(int)

# Calculate the confusion matrix
cm = confusion_matrix(actual_classes, predicted_classes)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Calculate precision, recall, F1-score, and accuracy
print(classification_report(actual_classes, predicted_classes))


df_test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")

df_test.head(5)


# prompt: use the above created ensemble model on the df_test dataframe and predict num_sold

# Assuming df_test is already loaded and preprocessed similarly to df_copy

# Initialize LabelEncoder (if not already initialized)
le = LabelEncoder()

# Fit and transform the 'store', 'product', and 'country' columns in df_test
for col in ['store', 'product', 'country']:
    df_test[col] = le.fit_transform(df_test[col])

# Convert 'date' to ordinal if necessary
df_test['date'] = pd.to_datetime(df_test['date'])
df_test['year'] = df_test['date'].dt.year


# Prepare the features for prediction
X_test_final = df_test[['year','store', 'product', 'country']]

# Make predictions using the trained models
rf_predictions_final = rf_model.predict(X_test_final)
gb_predictions_final = gb_model.predict(X_test_final)

# Ensemble the predictions
ensemble_predictions_final = (rf_predictions_final + gb_predictions_final) / 2

# Add the predictions to the df_test dataframe
df_test['num_sold'] = ensemble_predictions_final

# Display the predictions
print(df_test[['date', 'store', 'product', 'country', 'num_sold']].head())


# prompt: create a dataframe "sample_submission" with columns "id" and "num_sold" from df_test dataframe and round off the num_sold to 0 decimal places

# Create the sample submission DataFrame
sample_submission = pd.DataFrame({'id': df_test['id'], 'num_sold': df_test['num_sold'].astype(int)})

# Display the first few rows of the sample submission
print(sample_submission.head())








# prompt: count the number of rows in sample_Submission dataframe

print(len(sample_submission))


# prompt: use random forest and xgboost methods to create a model using df_copy dataframe to predict num_sold using features - year, country, store
# Prepare data for modeling
X = df_copy[['year','month','country']]
y = df_copy['num_sold']


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Initialize and train the XGBoost model
import xgboost as xgb
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42) # Adjust parameters as needed
xgb_model.fit(X_train, y_train)

# Make predictions using both models
rf_predictions = rf_model.predict(X_test)
xgb_predictions = xgb_model.predict(X_test)

# Ensemble predictions (simple average)
ensemble_predictions = (rf_predictions + xgb_predictions) / 2


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# Assuming 'ensemble_predictions' and 'y_test' are defined from your previous code

# Convert predictions to classes (e.g., using a threshold)
threshold = np.mean(y_test) # Example threshold - you may need to adjust this
predicted_classes = (ensemble_predictions >= threshold).astype(int)
actual_classes = (y_test >= threshold).astype(int)

# Calculate the confusion matrix
cm = confusion_matrix(actual_classes, predicted_classes)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Calculate precision, recall, F1-score, and accuracy
print(classification_report(actual_classes, predicted_classes))

accuracy = accuracy_score(actual_classes, predicted_classes)
precision = precision_score(actual_classes, predicted_classes)
recall = recall_score(actual_classes, predicted_classes)
f1 = f1_score(actual_classes, predicted_classes)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")


# prompt: use ensemble_predictions above to predict num_sold in the df_test dataframe 

df_test['month'] = df_test['date'].dt.month

# Prepare the features for prediction
X_test_final = df_test[['year','month','country']]

# Make predictions using the trained models
rf_predictions_final = rf_model.predict(X_test_final)
xgb_predictions_final = xgb_model.predict(X_test_final)

# Ensemble the predictions
ensemble_predictions_final = (rf_predictions_final + xgb_predictions_final) / 2

# Add the predictions to the df_test dataframe
df_test['num_sold_xgb'] = ensemble_predictions_final

# Display the predictions
print(df_test[['date', 'store', 'country', 'num_sold_xgb']].head())


# prompt: create a new dataframe submission_xgb with columns "id" and "num_sold_xgb" from df_test dataframe. Then export it as csv without the index 

# Create the submission DataFrame
submission_xgb = pd.DataFrame({'id': df_test['id'], 'num_sold_xgb': df_test['num_sold_xgb'].astype(int)})

submission_xgb.shape




