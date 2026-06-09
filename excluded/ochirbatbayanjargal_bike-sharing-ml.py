import pandas as pd
bikes = pd.read_csv("../input/bike-sharing-demand/train.csv")
print(bikes.head())


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

bikes = pd.read_csv("../input/bike-sharing-demand/train.csv")
print("First 5 rows:")
print(bikes.head())
print("\nShape:", bikes.shape)  # ~10k rows, 12 cols
print("\nMissing values:")
print(bikes.isnull().sum())  # Should be 0—clean dataset!


# Convert datetime and extract features
bikes["datetime"] = pd.to_datetime(bikes["datetime"])
bikes["hour"] = bikes["datetime"].dt.hour
bikes["day"] = bikes["datetime"].dt.day
bikes["month"] = bikes["datetime"].dt.month

# Drop datetime and registered/casual (we predict total count)
bikes = bikes.drop(columns=["datetime", "casual", "registered"])
print(bikes.head())


# Hourly counts
plt.figure(figsize=(10, 6))
sns.boxplot(x="hour", y="count", data=bikes)
plt.title("Bike Rentals by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Rental Count")
plt.show()

# Monthly trend
plt.figure(figsize=(10, 6))
sns.barplot(x="month", y="count", data=bikes)
plt.title("Bike Rentals by Month")
plt.show()


corr_matrix = bikes.corr()
print("Correlation with count:")
print(corr_matrix["count"])

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Feature Correlations")
plt.show()


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Features and target
X = bikes.drop(columns=["count"])
y = bikes["count"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
print("Feature importances:", dict(zip(X.columns, rf.feature_importances_)))


plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title("Actual vs Predicted Bike Rentals")
plt.xlabel("Actual Count")
plt.ylabel("Predicted Count")
plt.show()


print(f"Mean bike rentals: {bikes['count'].mean():.1f}")
print(f"Model RMSE: {rmse:.1f}")
print("Key insight: Hour and temp drive rentals—Random Forest captures non-linear trends.")

