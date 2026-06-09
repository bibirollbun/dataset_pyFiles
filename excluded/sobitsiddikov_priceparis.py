import pandas as pd

# Загрузка данных
train_data = pd.read_csv('/kaggle/input/playground-series-s3e6/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s3e6/test.csv')

# Просмотр первых строк
train_data.head(), test_data.head()



# Проверка структуры данных
train_data.info()
train_data.describe()

# Проверка на пропущенные значения
train_data.isnull().sum()



from sklearn.preprocessing import StandardScaler

# Стандартизация числовых признаков
numerical_columns = train_data.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])



from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Разделим данные на обучающие и валидационные выборки
X = train_data.drop(columns=['id', 'price'])
y = train_data['price']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Модель случайного леса
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказания на валидационной выборке
y_pred = model.predict(X_val)

# Оценка модели
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f'RMSE: {rmse}')



test_X = test_data.drop(columns=['id'])
test_predictions = model.predict(test_X)

# Создание файла для отправки
submission = pd.DataFrame({'id': test_data['id'], 'price': test_predictions})
submission.to_csv('submission.csv', index=False)


