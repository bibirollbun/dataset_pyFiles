import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error, r2_score


df=pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')


df.head()


df.shape


df.info()


#rentals across seasons
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.boxplot(x="season", y="count", data=df, palette="pastel")
plt.title("Rental Distribution Across Seasons")
plt.xlabel("Season (1=Spring, 2=Summer, 3=Fall, 4=Winter)")
plt.ylabel("Count")
plt.show()


#correlation heatmap
df['datetime'] = pd.to_datetime(df['datetime'])
plt.figure(figsize=(10, 8))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


#temperature vs total rentals by season
plt.figure(figsize=(12, 6))
sns.scatterplot(x="temp", y="count", hue="season", palette="viridis", data=df)
plt.title("Temperature vs Total Rentals by Season")
plt.xlabel("Temperature (Celsius)")
plt.ylabel("Count")
plt.show()


#bike rentals over time
df['date'] = df['datetime'].dt.date
time_series = df.groupby('date')['count'].sum()
plt.figure(figsize=(10, 6))
plt.plot(time_series.index, time_series.values, marker='o', linestyle='-', color='blue')
plt.title('Bike Rentals Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Rentals')
plt.grid(True)
plt.show()


# Extracting features from the datetime column
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day
df['month'] = df['datetime'].dt.month
df['weekday'] = df['datetime'].dt.weekday


#features and target
x = df.drop(['count','datetime','registered','date'], axis=1)
y = df['count']

#x = pd.get_dummies(x, drop_first=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#initializing models
models = {"Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)}


#training and evaluating each model
results = []
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = np.maximum(0, y_pred)
    r2 = r2_score(y_test, y_pred)
    msle = mean_squared_log_error(y_test, y_pred)
    rmsle = np.sqrt(msle)
    results.append({"Model": name, "R² Score": r2, "RMSLE": rmsle})
results_df = pd.DataFrame(results).sort_values(by="RMSLE")
results_df


#fitting the Random Forest model
best_model = models['Random Forest']
best_model.fit(x_train, y_train)

#predictions
y_pred = best_model.predict(x_test)
y_pred = np.maximum(0, y_pred)
r2 = r2_score(y_test, y_pred)
msle = mean_squared_log_error(y_test, y_pred)
rmsle = np.sqrt(msle)

print(f"R² Score: {r2}")
print(f"RMSLE: {rmsle}")


#actual vs predicted
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Actual vs Predicted Rentals")
plt.xlabel("Actual Rentals")
plt.ylabel("Predicted Rentals")
plt.grid(True)
plt.show()


#residual distribution
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.title("Residual Distribution")
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


#feature importance
feature_importances = best_model.feature_importances_
features = x.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.grid(True)
plt.show()

