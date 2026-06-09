from statistics import linear_regression

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor


df_train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


print(f"Total number of rows, columns: {df_train.shape}")
print("First few rows of Train Dataset:")
df_train.head()


print(f"Total number of rows, columns: {df_test.shape}")
print("First few rows of Test Dataset:")
df_test.head()


df_train.info()


# Encoding features of object type to int

# Initialize Encoder
encoder = LabelEncoder()

# List of categories
object_features = ["road_type", "lighting", "weather", "time_of_day"]

# Loop through each categorical column and encode both train & test
for col in object_features:
    # Fit on train and transform both train & test
    df_train[col] = encoder.fit_transform(df_train[col])
    df_test[col] = encoder.transform(df_test[col])


# Remove id column
df_train = df_train.drop(columns = ["id"])


df_train.head()


X = df_train.drop(columns = ["accident_risk"])
y = df_train["accident_risk"]

X_train, X_test_1, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=316)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test_1)


# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE for linear regression model: {rmse}")


coef = np.ravel(model.coef_)  # ensure 1D
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': coef
}).sort_values('Coefficient', ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(data=coef_df, x='Coefficient', y='Feature')
plt.title('Linear Regression Coefficients')
plt.tight_layout()
plt.show()


X_2 = df_train[["curvature", 'lighting', 'weather', 'num_reported_accidents', 'speed_limit']]
X_train_2, X_test_2, y_train, y_test = train_test_split(X_2, y, test_size=0.3, random_state=316)


model2 = HistGradientBoostingRegressor(
        learning_rate=0.1,
        max_depth=None,        # lets the model choose
        max_iter=300,          # trees
        early_stopping=True,
        random_state=42
    )
model2.fit(X_train_2, y_train)
y_pred_2 = model2.predict(X_test_2)

rmse_2 = np.sqrt(mean_squared_error(y_test, y_pred_2))
print(f"RMSE for HistGradientBoosting model: {rmse_2}")


X_test = df_test[["curvature", 'lighting', 'weather', 'num_reported_accidents', 'speed_limit']]
y_pred_3 = model2.predict(X_test)
sample_submission["accident_risk"] = y_pred_3
sample_submission.to_csv('/kaggle/working/submission.csv', index=False)
print(sample_submission.head())










