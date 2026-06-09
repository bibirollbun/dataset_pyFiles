import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# prediction for the next 365 days(periods = 365)
data = {
    'date': pd.date_range(start='2023-01-01', periods=365, freq='D'),
    'sales': np.random.randint(100, 500, size=365)  # Random sales data
}
df = pd.DataFrame(data)

# Add a numeric "day" feature for the regression model
df['day'] = (df['date'] - df['date'].min()).dt.days

# Split the data into train and test sets
X = df[['day']]  # Feature: days since start
y = df['sales']  # Target: sales
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# using linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future sales
future_days = pd.DataFrame({'day': range(df['day'].max() + 1, df['day'].max() + 366)})  # Predict next 30 days
future_sales = model.predict(future_days)

# Combine actual and predicted data for bar graph
future_dates = pd.date_range(start=df['date'].max() + pd.Timedelta(days=1), periods=365, freq='D')
future_df = pd.DataFrame({'date': future_dates, 'sales': future_sales})
combined_df = pd.concat([df[['date', 'sales']], future_df])

# Plot bar graph
plt.figure(figsize=(12, 6))
plt.bar(df['date'], df['sales'], label="Actual Sales", alpha=0.7, color='blue')
plt.bar(future_df['date'], future_df['sales'], label="Predicted Future Sales", alpha=0.7, color='orange')
plt.axvline(x=df['date'].max(), color='green', linestyle='--', label="Today")
plt.title("Sales Forecasting")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()



# Create a combined DataFrame with both actual and predicted sales
df_combined = pd.concat([df[['sales']], future_df[['sales']]])

# Create an 'id' column for both actual and predicted data
df_combined['id'] = df_combined.index

# Create the final submission file with 'id' and 'sales'
submission_df = df_combined[['id', 'sales']]

# Save the submission file
submission_df.to_csv('sales_predictions_submission.csv', index=False)

# Preview the first few rows of the submission file
print(submission_df.head())


