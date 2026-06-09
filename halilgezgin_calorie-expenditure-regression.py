import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_squared_log_error
)


data = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv", index_col="id")
data.head()


df = data.copy()


df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
df.head()


X = df.drop('Calories', axis=1)
y = df['Calories']

print(X.head())

scaler = StandardScaler()
X = scaler.fit_transform(X)

poly = PolynomialFeatures(degree=4, include_bias=False)
X = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


model = PoissonRegressor(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
# MAE
mae = mean_absolute_error(y_test, y_pred)

# MSE
mse = mean_squared_error(y_test, y_pred)

# RMSE
rmse = np.sqrt(mse)

# MAPE (manuel hesaplanır)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# R2 Score
r2 = r2_score(y_test, y_pred)

# RMSLE (log(0) sorununa karşı sıfır içermeyen değerlerde çalıştırılır)
if np.all(y_test >= -1) and np.all(y_pred >= -1):
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
else:
    rmsle = None  # Negatif değer varsa hesaplanmaz

# Sonuçları yazdır
print("Regresyon Metrikleri Karşılaştırması")
print("-" * 40)
print(f"MAE   : {mae:.4f}")
print(f"MSE   : {mse:.4f}")
print(f"RMSE  : {rmse:.4f}")
print(f"MAPE  : {mape:.2f}%")
print(f"R²    : {r2:.4f}")
print(f"RMSLE : {rmsle:.4f}" if rmsle is not None else "RMSLE : Geçersiz (negatif değer içeriyor)")


test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv", index_col="id")
test.head()


test = pd.get_dummies(test, columns=['Sex'], drop_first=True)
test.head()


test.shape


scaler = StandardScaler()
test = scaler.fit_transform(test)

poly = PolynomialFeatures(degree=4, include_bias=False)
test = poly.fit_transform(test)


pred = model.predict(test)

start_id = 750000
ids = np.arange(start_id, start_id + len(pred))

df_pred = pd.DataFrame({
    "id": ids,
    "Calories": pred
})
df_pred


df_pred.to_csv("calorie_expenditure_result.csv", index=False)




