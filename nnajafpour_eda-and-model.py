from pandas import read_csv

df = read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df.head()



df.info()



df.isnull().sum()



df.describe()



import matplotlib.pyplot as plt
import seaborn as sns

# Select only numeric features for correlation analysis
numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(), cmap="coolwarm")
plt.title("Feature Correlation")



sns.histplot(df["accident_risk"], kde=True)


# Sort features by correlation with target to identify most predictive ones
target_corr = numeric_df.corr()["accident_risk"].sort_values(ascending=False)
target_corr



target_corr.drop("accident_risk").plot(kind="bar", figsize=(10,4))
plt.title("Feature Correlation with Accident Risk")
plt.ylabel("Correlation")



from pandas import get_dummies

y = df["accident_risk"]
X = df.drop("accident_risk", axis=1)

# Convert categorical variables to dummy/indicator variables
X = get_dummies(X)
X.head()



from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

X_train.head()



from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()

# Train Linear Regression model
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_valid)

# Plot actual vs predicted for visual assessment
plt.figure(figsize=(8,6))
plt.scatter(y_valid, y_pred, alpha=0.5)
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--')
plt.xlabel("Actual Accident Risk")
plt.ylabel("Predicted Accident Risk")
plt.title("Actual vs Predicted (Linear Regression)")
plt.show()



from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Calculate validation metrics
mae = mean_absolute_error(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_valid, y_pred)

print(f"Linear Regression MAE: {mae:,.4f}")
print(f"Linear Regression MSE: {mse:,.4f}")
print(f"Linear Regression RMSE: {rmse:,.4f}")
print(f"Linear Regression R2: {r2:,.4f}")



from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)

# Train Random Forest model
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_valid)
print(y_pred[:5])



import pandas as pd

# Extract feature importances from Random Forest
feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': rf_model.feature_importances_}).sort_values(by='importance', ascending=False)
feature_importances.head()


# Plot top 20 important features for clarity
top_features = feature_importances.head(20)
plt.figure(figsize=(10,6))
plt.bar(top_features['feature'], top_features['importance'])
plt.xticks(rotation=90)
plt.ylabel("Importance")
plt.title("Feature Importances (Random Forest)")
plt.show()


# Calculate validation metrics
mae = mean_absolute_error(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_valid, y_pred)

print(f"Random Forest MAE: {mae:,.4f}")
print(f"Random Forest MSE: {mse:,.4f}")
print(f"Random Forest RMSE: {rmse:,.4f}")
print(f"Random Forest R2: {r2:,.4f}")



from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=300)

# Train XGBoost model
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_valid)
print(y_pred[:5])


# Extract feature importances from XGBoost
feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': xgb_model.feature_importances_}).sort_values(by='importance', ascending=False)
feature_importances.head()


# Plot top 20 important features for clarity
top_features = feature_importances.head(20)
plt.figure(figsize=(10,6))
plt.bar(top_features['feature'], top_features['importance'])
plt.xticks(rotation=90)
plt.ylabel("Importance")
plt.title("Feature Importances (XGBoost Regressor)")
plt.show()


# Calculate validation metrics
mae = mean_absolute_error(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_valid, y_pred)

print(f"XGBoost Regressor MAE: {mae:,.4f}")
print(f"XGBoost Regressor MSE: {mse:,.4f}")
print(f"XGBoost Regressor RMSE: {rmse:,.4f}")
print(f"XGBoost Regressor R2: {r2:,.4f}")



from pandas import DataFrame
test_df = read_csv("/kaggle/input/playground-series-s5e10/test.csv")

X_test = get_dummies(test_df)

X_train, X_test = X_train.align(X_test, join='left', axis=1)

y_pred = xgb_model.predict(X_test)

submission = DataFrame({
    "id": test_df["id"],
    "accident_risk": y_pred
})

submission.to_csv("submission.csv", index=False)

