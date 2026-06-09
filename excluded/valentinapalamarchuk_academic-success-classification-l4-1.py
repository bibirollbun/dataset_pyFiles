import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import PolynomialFeatures


train = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')


print(train.head())
print(train.describe())
print(train.info())


X = train.drop(columns=['id', 'Target'])
y = train['Target']


le = LabelEncoder()
y = le.fit_transform(y)


X = pd.get_dummies(X, drop_first=True)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)


X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)


model = Ridge(alpha=1.0)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cv_scores = cross_val_score(model, X_poly, y, cv=5, scoring='r2')

print(f'MSE: {mse:.4f}')
print(f'R^2 Score: {r2:.4f}')
print(f'Cross-Validation R² Score: {cv_scores.mean():.4f}')


plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ідеальне вгадування (y=x)')
plt.xlabel('Фактичні значення')
plt.ylabel('Передбачені значення')
plt.title(f'Фактичні vs Передбачені значення (R² = {r2:.4f})')
plt.legend()
plt.grid(True)
plt.show()


residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=30, kde=True, color='purple')
plt.axvline(0, color='red', linestyle='--', label='Нульова помилка')
plt.xlabel('Залишки (фактичне - передбачене)')
plt.ylabel('Частота')
plt.title('Гістограма залишків: чи нормальний розподіл помилок?')
plt.legend()
plt.show()


plt.figure(figsize=(7, 6))
stats.probplot(residuals, dist='norm', plot=plt)
plt.title('Графік залишків (перевірка нормальності)', fontsize=16)
plt.xlabel('Теоретичні квантилі', fontsize=14)  
plt.ylabel('Емпіричні квантилі', fontsize=14) 
plt.grid(True)
plt.show()

