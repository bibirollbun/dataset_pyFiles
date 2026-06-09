import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, PoissonRegressor, GammaRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_squared_log_error
)


data = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv", index_col="id")
data.head()


data.info()


data.Sex.unique()


sns.countplot(data=data, x="Sex")


num_col = data.select_dtypes(include='number').columns.tolist()

plt.figure(figsize=(15, 10))

for i, col in enumerate(num_col, 1):
    plt.subplot(3, 3, i)
    sns.histplot(data[col], kde=True)

plt.tight_layout()
plt.show()


sns.heatmap(data[num_col].corr(), annot=True, vmin=-1, vmax=1, fmt=".3f", linewidth=.5)
plt.show()


plt.figure(figsize=(15, 10))

for i, col in enumerate(num_col, 1):
    plt.subplot(3, 3, i)
    sns.scatterplot(x=col, y='Calories', data=data)

    # Regresyon doğrusunu hesapla
    x = data[col]
    y = data['Calories']
    mask = ~x.isna() & ~y.isna()
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    x_vals = np.array([x.min(), x.max()])
    y_vals = intercept + slope * x_vals

    plt.plot(x_vals, y_vals, color='orange')

plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 10))

for i, col in enumerate(num_col, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data[col], orient="h")
    plt.title(col)

plt.tight_layout()
plt.show()


df = data.copy()


num_col = data.select_dtypes(include='number').columns.tolist()

for col in num_col:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    if ((df[col] < lower_bound) | (df[col] > upper_bound)).any():
        df[col].clip(lower=lower_bound, upper=upper_bound, inplace=True)


df['BMI'] = df['Weight'] / (df['Height']/100)**2
df.drop(['Height', 'Weight'], axis=1, inplace=True)


df['Activity_Intensity'] = df['Heart_Rate'] * df['Body_Temp'] * df['Duration']
df.drop(['Duration', 'Heart_Rate', 'Body_Temp'], axis=1, inplace=True)


df['Age_Group'] = pd.cut(df['Age'], bins=[0, 20, 35, 50, 65, 100], labels=['Teen', 'Young_Adult', 'Adult', 'Middle_Age', 'Senior'])
df = pd.get_dummies(df, columns=['Age_Group'], drop_first=True)
df.drop("Age", axis=1, inplace=True)


df.head()


df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
df.head()


X = df.drop('Calories', axis=1)
y = df['Calories']

scaler = StandardScaler()
X = scaler.fit_transform(X)

poly = PolynomialFeatures(degree=3, include_bias=False)
X = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


model = PoissonRegressor()
model.fit(X_train, y_train)


model.score(X_test, y_test)


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


np.all(y_test >= -1) and np.all(y_pred >= -1)


(y_pred < 0).sum()


(y_test < 0).sum()







