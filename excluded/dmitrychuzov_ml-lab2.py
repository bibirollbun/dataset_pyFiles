import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv', nrows=100000)


data = data.dropna()
data = data[(data['fare_amount'] > 0) & (data['fare_amount'] < 500)] 

data['trip_distance'] = np.sqrt((data['pickup_longitude'] - data['dropoff_longitude'])**2 +
                                (data['pickup_latitude'] - data['dropoff_latitude'])**2)

# Видалення некоректних рядків
data = data[data['trip_distance'] > 0]

# Визначення змінних X та y
X = data[['trip_distance']]
y = data['fare_amount']

# Розбиття на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створення та навчання моделі регресії
model = LinearRegression()
model.fit(X_train, y_train)

# Прогнозування
y_pred = model.predict(X_test)

# Оцінка моделі
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Побудова графіку
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Data', alpha=0.5)
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.title('Linear Regression: Trip Distance vs Fare Amount')
plt.xlabel('Trip Distance')
plt.ylabel('Fare Amount')
plt.legend()
plt.grid(True)
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Вибір основних змінних для моделі
features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
target = 'fare_amount'

# Розділення даних на ознаки та цільову змінну
X = data[features]
y = data[target]

# Розділення на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створення та тренування моделі лінійної регресії
model = LinearRegression()
model.fit(X_train, y_train)

# Оцінка моделі
y_pred = model.predict(X_test)

# Розрахунок метрик
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)

# Виведення коефіцієнтів моделі
print("Коефіцієнти моделі:", model.coef_)
print("Вільний член:", model.intercept_)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# Вибір основних змінних для моделі
features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
target = 'fare_amount'

# Розділення даних на ознаки та цільову змінну
X = data[features]
y = data[target]

# Розділення на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Функція для тренування та оцінки моделей
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model: {model.__class__.__name__}")
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)
    print("Коефіцієнти моделі:", model.coef_)
    print("Вільний член:", model.intercept_)
    print("-" * 50)

# Моделі регуляризації
lasso = Lasso(alpha=0.1, random_state=42)
ridge = Ridge(alpha=1.0, random_state=42)
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)

# Тренування та оцінка моделей
train_and_evaluate(lasso, X_train, X_test, y_train, y_test)
train_and_evaluate(ridge, X_train, X_test, y_train, y_test)
train_and_evaluate(elastic_net, X_train, X_test, y_train, y_test)



from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Вибір основних змінних для моделі
features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
target = 'fare_amount'


# Розділення даних на ознаки та цільову змінну
X = data[features]
y = data[target]

# Розділення на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормалізація даних
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Побудова нейронної мережі
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # Вихідний шар для регресії
])

# Компіляція моделі
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

# Тренування моделі
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Оцінка моделі
loss, mse = model.evaluate(X_test, y_test, verbose=0)
y_pred = model.predict(X_test)

# Обчислення R^2 вручну
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)


