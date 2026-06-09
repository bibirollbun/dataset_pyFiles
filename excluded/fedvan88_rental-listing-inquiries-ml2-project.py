import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import re
from collections import Counter
import itertools
from sklearn.preprocessing import MinMaxScaler


df = pd.read_json("/kaggle/input/two-sigma-connect-rental-listing-inquiries/train.json.zip")


lower_bound = df['price'].quantile(0.01)
upper_bound = df['price'].quantile(0.99)

df_filtered = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)].copy()

print(f"Размер данных до фильтрации: {df.shape[0]} строк")
print(f"Размер данных после фильтрации: {df_filtered.shape[0]} строк")


level_mapping = {'low': 0, 'medium': 1, 'high': 2}

df_filtered['interest_level_encoded'] = df_filtered['interest_level'].map(level_mapping)


df_filtered


X = df_filtered[['bathrooms', 'bedrooms']]
y = df_filtered['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


poly_features = PolynomialFeatures(degree=10, include_bias=False)

X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Проверка количества новых признаков
print(f"Количество признаков до преобразования: {X_train.shape[1]}")
print(f"Количество признаков после преобразования: {X_train_poly.shape[1]}")


# Создание пустого DataFrame для метрик MAE
result_MAE = pd.DataFrame(columns=['model', 'train', 'test'])

# Создание пустого DataFrame для метрик RMSE
result_RMSE = pd.DataFrame(columns=['model', 'train', 'test'])

# Создание пустого DataFrame для метрик R2
result_R2 = pd.DataFrame(columns=['model', 'train', 'test'])



linear_regression_model = LinearRegression()

# Обучение модели на обучающих данных
linear_regression_model.fit(X_train_poly, y_train)

# Прогнозы на обучающих данных
y_train_pred = linear_regression_model.predict(X_train_poly)

# Прогнозы на тестовых данных
y_test_pred = linear_regression_model.predict(X_test_poly)

# Сохранение результатов в виде новых столбцов в данных
df_train = X_train.copy()
df_test = X_test.copy()

df_train['true_price'] = y_train
df_train['predicted_price'] = y_train_pred

df_test['true_price'] = y_test
df_test['predicted_price'] = y_test_pred

print("Результаты на обучающем наборе:")
print(df_train.head())

print("\nРезультаты на тестовом наборе:")
print(df_test.head())


# Вычисление MAE на обучающих данных
mae_train = mean_absolute_error(y_train, y_train_pred)

# Вычисление MAE на тестовых данных
mae_test = mean_absolute_error(y_test, y_test_pred)

result_MAE.loc[len(result_MAE)] = ['linear_regression', mae_train, mae_test]

print(f"MAE на обучающем наборе: {mae_train:.2f}")
print(f"MAE на тестовом наборе: {mae_test:.2f}")


# Вычисление MSE на обучающих и тестовых данных
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

# Вычисление RMSE (корень из MSE)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

result_RMSE.loc[len(result_RMSE)] = ['linear_regression', rmse_train, rmse_test]

print(f"RMSE на обучающем наборе: {rmse_train:.2f}")
print(f"RMSE на тестовом наборе: {rmse_test:.2f}")


# Метрика R²
def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# Вычисление r2 на обучающих данных
r2_train = calculate_r2(y_train, y_train_pred)

# Вычисление r2 на тестовых данных
r2_test = calculate_r2(y_test, y_test_pred)

result_R2.loc[len(result_R2)] = ['linear_regression', r2_train, r2_test]


print('MAE\n', result_MAE)
print('-----------------------------------------------')
print('RMSE\n',result_RMSE)
print('-----------------------------------------------')
print('R2\n',result_R2)


decision_tree_model = DecisionTreeRegressor(random_state=21)
decision_tree_model.fit(X_train_poly, y_train)

# Прогнозы на обучающих данных
y_train_pred_dt = decision_tree_model.predict(X_train_poly)

# Прогнозы на тестовых данных
y_test_pred_dt = decision_tree_model.predict(X_test_poly)

# Сохранение результатов в новые столбцы DataFrame
df_train['predicted_price_dt'] = y_train_pred_dt
df_test['predicted_price_dt'] = y_test_pred_dt

print("Прогнозы дерева решений на обучающем наборе:")
print(df_train.head())


mae_train_dt = mean_absolute_error(y_train, y_train_pred_dt)
mae_test_dt = mean_absolute_error(y_test, y_test_pred_dt)

print(f"MAE на обучающем наборе (Дерево решений): {mae_train_dt:.2f}")
print(f"MAE на тестовом наборе (Дерево решений): {mae_test_dt:.2f}")


mse_train_dt = mean_squared_error(y_train, y_train_pred_dt)
mse_test_dt = mean_squared_error(y_test, y_test_pred_dt)

# Вычисление RMSE
rmse_train_dt = np.sqrt(mse_train_dt)
rmse_test_dt = np.sqrt(mse_test_dt)

print(f"RMSE на обучающем наборе (Дерево решений): {rmse_train_dt:.2f}")
print(f"RMSE на тестовом наборе (Дерево решений): {rmse_test_dt:.2f}")


# Вычисление r2 на обучающих данных
r2_train_dt = calculate_r2(y_train, y_train_pred_dt)

# Вычисление r2 на тестовых данных
r2_test_dt = calculate_r2(y_test, y_test_pred_dt)



# Добавление результатов в DataFrame result_MAE
result_MAE.loc[len(result_MAE)] = ['decision_tree', mae_train_dt, mae_test_dt]
# Добавление результатов в DataFrame result_RMSE
result_RMSE.loc[len(result_RMSE)] = ['decision_tree', rmse_train_dt, rmse_test_dt]
# Добавление результатов в DataFrame result_R2
result_R2.loc[len(result_R2)] = ['decision_tree', r2_train_dt, r2_test_dt]

print('MAE\n', result_MAE)
print('-----------------------------------------------')
print('RMSE\n',result_RMSE)
print('-----------------------------------------------')
print('R2\n',result_R2)


# Рассчитайте среднее и медиану на обучающем наборе
mean_price = y_train.mean()
median_price = y_train.median()

# Создайте столбцы с этими значениями как прогнозы
y_train_pred_mean = [mean_price] * len(y_train)
y_test_pred_mean = [mean_price] * len(y_test)

y_train_pred_median = [median_price] * len(y_train)
y_test_pred_median = [median_price] * len(y_test)

print(f"Среднее значение цены (обучение): {mean_price:.2f}")
print(f"Медиана цены (обучение): {median_price:.2f}")


# MAE для наивной модели на основе среднего
mae_train_mean = mean_absolute_error(y_train, y_train_pred_mean)
mae_test_mean = mean_absolute_error(y_test, y_test_pred_mean)

# MAE для наивной модели на основе медианы
mae_train_median = mean_absolute_error(y_train, y_train_pred_median)
mae_test_median = mean_absolute_error(y_test, y_test_pred_median)

result_MAE.loc[len(result_MAE)] = ['naive_mean', mae_train_mean, mae_test_mean]
result_MAE.loc[len(result_MAE)] = ['naive_median', mae_train_median, mae_test_median]

print(f"MAE (Наивное среднее): train={mae_train_mean:.2f}, test={mae_test_mean:.2f}")
print(f"MAE (Наивная медиана): train={mae_train_median:.2f}, test={mae_test_median:.2f}")


# RMSE для наивной модели на основе среднего
rmse_train_mean = np.sqrt(mean_squared_error(y_train, y_train_pred_mean))
rmse_test_mean = np.sqrt(mean_squared_error(y_test, y_test_pred_mean))

# RMSE для наивной модели на основе медианы
rmse_train_median = np.sqrt(mean_squared_error(y_train, y_train_pred_median))
rmse_test_median = np.sqrt(mean_squared_error(y_test, y_test_pred_median))

result_RMSE.loc[len(result_RMSE)] = ['naive_mean', rmse_train_mean, rmse_test_mean]
result_RMSE.loc[len(result_RMSE)] = ['naive_median', rmse_train_median, rmse_test_median]

print(f"RMSE (Наивное среднее): train={rmse_train_mean:.2f}, test={rmse_test_mean:.2f}")
print(f"RMSE (Наивная медиана): train={rmse_train_median:.2f}, test={rmse_test_median:.2f}")


# Вычисление r2 на обучающих данных
r2_train_mean = calculate_r2(y_train, y_train_pred_mean)

# Вычисление r2 на тестовых данных
r2_test_mean = calculate_r2(y_test, y_test_pred_mean)

result_R2.loc[len(result_R2)] = ['naive_mean', r2_train_mean, r2_test_mean]

# Вычисление r2 на обучающих данных
r2_train_median = calculate_r2(y_train, y_train_pred_median)

# Вычисление r2 на тестовых данных
r2_test_median = calculate_r2(y_test, y_test_pred_median)

result_R2.loc[len(result_R2)] = ['naive_median', r2_train_median, r2_test_median]


print('MAE\n', result_MAE)
print('-----------------------------------------------')
print('RMSE\n',result_RMSE)
print('-----------------------------------------------')
print('R2\n',result_R2)


# Список для хранения всех признаков из набора данных
all_features_list = []

# Очищаем столбец 'features'
df_filtered['features'] = df_filtered['features'].apply(lambda features_list: [re.sub(r'\[|\'|\"|\]|\s', '', f) for f in features_list])

# Создаем объединенный список всех очищенных признаков
all_features_list = list(itertools.chain.from_iterable(df_filtered['features']))

# Находим количество уникальных признаков
unique_features_count = len(set(all_features_list))

print(f"Общее количество уникальных признаков: {unique_features_count}")
print(df_filtered['features'].head())


# Подсчет топ-20 наиболее популярных признаков
feature_counts = Counter(all_features_list)
top_20_features = [feature for feature, count in feature_counts.most_common(20)]

print("Топ-20 самых популярных признаков:")
print(top_20_features)


# Создание 20 новых признаков на основе топ-20
for feature in top_20_features:
    df_filtered[feature] = df_filtered['features'].apply(lambda x: 1 if feature in x else 0)

# Список всех фич для модели
feature_list = top_20_features + ['bathrooms', 'bedrooms']

# Отображение созданных фич
print("\nСозданные фичи:")
print(df_filtered[feature_list].head())

print(f"\nОбщее количество фич для модели: {len(feature_list)}")
print(f"Список фич: {feature_list}")


df_filtered[['features'] + feature_list]


X = df_filtered[feature_list].values
y = df_filtered['price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)


np.random.seed(21)


class LinearRegressionCustom:
    def __init__(self, method='sgd', learning_rate=0.01, n_iterations=1000):
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        if self.method == 'sgd':
            indices = np.arange(n_samples)
            for _ in range(self.n_iterations):
                np.random.shuffle(indices)  # порядок всегда одинаковый при фиксированном сидe
                for i in indices:
                    y_pred = np.dot(X[i], self.weights) + self.bias
                    error = y_pred - y[i]
                    self.weights -= self.learning_rate * 2 * error * X[i]
                    self.bias -= self.learning_rate * 2 * error

        elif self.method == 'non_stochastic':
            for _ in range(self.n_iterations):
                y_pred = np.dot(X, self.weights) + self.bias
                error = y_pred - y
                dw = (2 / n_samples) * np.dot(X.T, error)
                db = (2 / n_samples) * np.sum(error)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

        elif self.method == 'analytical':
            X_b = np.c_[np.ones((n_samples, 1)), X]
            theta_best = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
            self.bias = theta_best[0]
            self.weights = theta_best[1:]

        else:
            raise ValueError("method должен быть 'sgd', 'non_stochastic' или 'analytical'")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias



# Создание и обучение моделей
custom_sgd_model = LinearRegressionCustom(method='sgd', learning_rate=0.01, n_iterations=1000)
custom_sgd_model.fit(X_train, y_train)

custom_ols_model = LinearRegressionCustom(method='analytical')
custom_ols_model.fit(X_train, y_train)

custom_nsgd_model = LinearRegressionCustom(method='non_stochastic')
custom_nsgd_model.fit(X_train, y_train)

sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)

# Предсказания
sgd_train_pred = custom_sgd_model.predict(X_train)
sgd_test_pred = custom_sgd_model.predict(X_test)

ols_train_pred = custom_ols_model.predict(X_train)
ols_test_pred = custom_ols_model.predict(X_test)

nsgd_train_pred = custom_nsgd_model.predict(X_train)
nsgd_test_pred = custom_nsgd_model.predict(X_test)

sklearn_train_pred = sklearn_model.predict(X_train)
sklearn_test_pred = sklearn_model.predict(X_test)

result_MAE.loc[len(result_MAE)] = ['Custom SGD', mean_absolute_error(y_train, sgd_train_pred), mean_absolute_error(y_test, sgd_test_pred)]
result_RMSE.loc[len(result_RMSE)] = ['Custom SGD', mean_squared_error(y_train, sgd_train_pred), mean_squared_error(y_test, sgd_test_pred)]
result_R2.loc[len(result_R2)] = ['Custom SGD', calculate_r2(y_train, sgd_train_pred), calculate_r2(y_test, sgd_test_pred)]

result_MAE.loc[len(result_MAE)] = ['Custom Analytical', mean_absolute_error(y_train, ols_train_pred), mean_absolute_error(y_test, ols_test_pred)]
result_RMSE.loc[len(result_RMSE)] = ['Custom Analytical', mean_squared_error(y_train, ols_train_pred), mean_squared_error(y_test, ols_test_pred)]
result_R2.loc[len(result_R2)] = ['Custom Analytical', calculate_r2(y_train, ols_train_pred), calculate_r2(y_test, ols_test_pred)]

result_MAE.loc[len(result_MAE)] = ['Custom non stochastic', mean_absolute_error(y_train, nsgd_train_pred), mean_absolute_error(y_test, nsgd_test_pred)]
result_RMSE.loc[len(result_RMSE)] = ['Custom non stochastic', mean_squared_error(y_train, nsgd_train_pred), mean_squared_error(y_test, nsgd_test_pred)]
result_R2.loc[len(result_R2)] = ['Custom non stochastic', calculate_r2(y_train, nsgd_train_pred), calculate_r2(y_test, nsgd_test_pred)]

result_MAE.loc[len(result_MAE)] = ['Sklearn linreg', mean_absolute_error(y_train, sklearn_train_pred), mean_absolute_error(y_test, sklearn_test_pred)]
result_RMSE.loc[len(result_RMSE)] = ['Sklearn linreg', mean_squared_error(y_train, sklearn_train_pred), mean_squared_error(y_test, sklearn_test_pred)]
result_R2.loc[len(result_R2)] = ['Sklearn linreg', calculate_r2(y_train, sklearn_train_pred), calculate_r2(y_test, sklearn_test_pred)]

# Вывод результатов в виде DataFrame
print('MAE\n', result_MAE)
print('-----------------------------------------------')
print('RMSE\n',result_RMSE)
print('-----------------------------------------------')
print('R2\n',result_R2)


class RegularizedLinearRegression:
    def __init__(self, method='ridge', learning_rate=0.01, n_iterations=2000, alpha=1.0, l1_ratio=0.5):
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Обновление градиента в зависимости от метода регуляризации
            if self.method == 'ridge':
                # Пакетный градиентный спуск для Ridge
                dw = (2/n_samples) * np.dot(X.T, (y_predicted - y)) + 2 * self.alpha * self.weights
            elif self.method == 'lasso':
                # Пакетный градиентный спуск для Lasso
                dw = (2/n_samples) * np.dot(X.T, (y_predicted - y)) + self.alpha * np.sign(self.weights)
            elif self.method == 'elasticnet':
                # Пакетный градиентный спуск для ElasticNet
                l1_reg = self.alpha * self.l1_ratio * np.sign(self.weights)
                l2_reg = self.alpha * (1 - self.l1_ratio) * 2 * self.weights
                dw = (2/n_samples) * np.dot(X.T, (y_predicted - y)) + l1_reg + l2_reg
            else:
                raise ValueError("Method not supported. Choose from 'ridge', 'lasso', 'elasticnet'.")
            
            db = (2/n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# Создание и обучение моделей

custom_ridge = RegularizedLinearRegression(method='ridge', learning_rate=0.01, n_iterations=2000, alpha=0.1)
custom_ridge.fit(X_train, y_train)

sklearn_ridge = Ridge(alpha=0.1)
sklearn_ridge.fit(X_train, y_train)

custom_lasso = RegularizedLinearRegression(method='lasso', learning_rate=0.01, n_iterations=2000, alpha=0.1)
custom_lasso.fit(X_train, y_train)

sklearn_lasso = Lasso(alpha=0.1)
sklearn_lasso.fit(X_train, y_train)

custom_elasticnet = RegularizedLinearRegression(method='elasticnet', learning_rate=0.01, n_iterations=2000, alpha=0.1, l1_ratio=0.5)
custom_elasticnet.fit(X_train, y_train)

sklearn_elasticnet = ElasticNet(alpha=0.1, l1_ratio=0.5)
sklearn_elasticnet.fit(X_train, y_train)

# Предсказания

custom_ridge_train_pred = custom_ridge.predict(X_train)
custom_ridge_test_pred = custom_ridge.predict(X_test)

sklearn_ridge_train_pred = sklearn_ridge.predict(X_train)
sklearn_ridge_test_pred = sklearn_ridge.predict(X_test)

custom_lasso_train_pred = custom_lasso.predict(X_train)
custom_lasso_test_pred = custom_lasso.predict(X_test)

sklearn_lasso_train_pred = sklearn_lasso.predict(X_train)
sklearn_lasso_test_pred = sklearn_lasso.predict(X_test)

custom_elasticnet_train_pred = custom_elasticnet.predict(X_train)
custom_elasticnet_test_pred = custom_elasticnet.predict(X_test)

sklearn_elasticnet_train_pred = sklearn_elasticnet.predict(X_train)
sklearn_elasticnet_test_pred = sklearn_elasticnet.predict(X_test)

# Добавление результатов в таблицу метрик

result_MAE.loc[len(result_MAE)] = ['Custom Ridge', mean_absolute_error(y_train, custom_ridge_train_pred), mean_absolute_error(y_test, custom_ridge_test_pred)]
result_RMSE.loc[len(result_RMSE)] = ['Custom Ridge', np.sqrt(mean_squared_error(y_train, custom_ridge_train_pred)), np.sqrt(mean_squared_error(y_test, custom_ridge_test_pred))]
result_R2.loc[len(result_R2)] = ['Custom Ridge', calculate_r2(y_train, custom_ridge_train_pred), calculate_r2(y_test, custom_ridge_test_pred)]

result_MAE.loc[len(result_MAE)] = ['Sklearn Ridge', mean_absolute_error(y_train, sklearn_ridge_train_pred), mean_absolute_error(y_test, sklearn_ridge_test_pred)]
result_RMSE.loc[len(result_RMSE)] = ['Sklearn Ridge', np.sqrt(mean_squared_error(y_train, sklearn_ridge_train_pred)), np.sqrt(mean_squared_error(y_test, sklearn_ridge_test_pred))]
result_R2.loc[len(result_R2)] = ['Sklearn Ridge', r2_score(y_train, sklearn_ridge_train_pred), r2_score(y_test, sklearn_ridge_test_pred)]

result_MAE.loc[len(result_MAE)] = ['Custom Lasso', mean_absolute_error(y_train, custom_lasso_train_pred), mean_absolute_error(y_test, custom_lasso_test_pred)]
result_RMSE.loc[len(result_RMSE)] = ['Custom Lasso', np.sqrt(mean_squared_error(y_train, custom_lasso_train_pred)), np.sqrt(mean_squared_error(y_test, custom_lasso_test_pred))]
result_R2.loc[len(result_R2)] = ['Custom Lasso', calculate_r2(y_train, custom_lasso_train_pred), calculate_r2(y_test, custom_lasso_test_pred)]

result_MAE.loc[len(result_MAE)] = ['Sklearn Lasso', mean_absolute_error(y_train, sklearn_lasso_train_pred), mean_absolute_error(y_test, sklearn_lasso_test_pred)]
result_RMSE.loc[len(result_RMSE)] = ['Sklearn Lasso', np.sqrt(mean_squared_error(y_train, sklearn_lasso_train_pred)), np.sqrt(mean_squared_error(y_test, sklearn_lasso_test_pred))]
result_R2.loc[len(result_R2)] = ['Sklearn Lasso', r2_score(y_train, sklearn_lasso_train_pred), r2_score(y_test, sklearn_lasso_test_pred)]

result_MAE.loc[len(result_MAE)] = ['Custom ElasticNet', mean_absolute_error(y_train, custom_elasticnet_train_pred), mean_absolute_error(y_test, custom_elasticnet_test_pred)]
result_RMSE.loc[len(result_RMSE)] = ['Custom ElasticNet', np.sqrt(mean_squared_error(y_train, custom_elasticnet_train_pred)), np.sqrt(mean_squared_error(y_test, custom_elasticnet_test_pred))]
result_R2.loc[len(result_R2)] = ['Custom ElasticNet', calculate_r2(y_train, custom_elasticnet_train_pred), calculate_r2(y_test, custom_elasticnet_test_pred)]

result_MAE.loc[len(result_MAE)] = ['Sklearn ElasticNet', mean_absolute_error(y_train, sklearn_elasticnet_train_pred), mean_absolute_error(y_test, sklearn_elasticnet_test_pred)]
result_RMSE.loc[len(result_RMSE)] = ['Sklearn ElasticNet', np.sqrt(mean_squared_error(y_train, sklearn_elasticnet_train_pred)), np.sqrt(mean_squared_error(y_test, sklearn_elasticnet_test_pred))]
result_R2.loc[len(result_R2)] = ['Sklearn ElasticNet', r2_score(y_train, sklearn_elasticnet_train_pred), r2_score(y_test, sklearn_elasticnet_test_pred)]

print('MAE\n', result_MAE)
print('-----------------------------------------------')
print('RMSE\n', result_RMSE)
print('-----------------------------------------------')
print('R2\n', result_R2)


class CustomMinMaxScaler:
    def fit(self, X):
        self.min_vals = np.min(X, axis=0)
        self.max_vals = np.max(X, axis=0)
        return self

    def transform(self, X):
        return (X - self.min_vals) / (self.max_vals - self.min_vals)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# Инициализация и применение кастомного MinMaxScaler
custom_scaler = CustomMinMaxScaler()
X_train_scaled_custom = custom_scaler.fit_transform(X_train)
X_test_scaled_custom = custom_scaler.transform(X_test)

# Инициализация и применение sklearn MinMaxScaler
sklearn_scaler = MinMaxScaler()
X_train_scaled_sklearn = sklearn_scaler.fit_transform(X_train)
X_test_scaled_sklearn = sklearn_scaler.transform(X_test)


# Сравнение результатов
print("Сравнение MinMax-нормализации (обучающая выборка):")
print(np.allclose(X_train_scaled_custom, X_train_scaled_sklearn))

print("Сравнение MinMax-нормализации (тестовая выборка):")
print(np.allclose(X_test_scaled_custom, X_test_scaled_sklearn))

# Вывод результатов
print("\nКастомная нормализация (train):")
print(X_train_scaled_custom)
print("\nSklearn нормализация (train):")
print(X_train_scaled_sklearn)


class CustomStandardScaler:
    def fit(self, X):
        self.mean_vals = np.mean(X, axis=0)
        self.std_vals = np.std(X, axis=0)
        # Обработка случая, когда стандартное отклонение равно 0
        self.std_vals[self.std_vals == 0] = 1
        return self

    def transform(self, X):
        return (X - self.mean_vals) / self.std_vals

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# Инициализация и применение кастомного StandardScaler
custom_scaler_std = CustomStandardScaler()
X_train_std_custom = custom_scaler_std.fit_transform(X_train)
X_test_std_custom = custom_scaler_std.transform(X_test)

# Инициализация и применение sklearn StandardScaler
from sklearn.preprocessing import StandardScaler
sklearn_scaler_std = StandardScaler()
X_train_std_sklearn = sklearn_scaler_std.fit_transform(X_train)
X_test_std_sklearn = sklearn_scaler_std.transform(X_test)


# Сравнение результатов
print("Сравнение StandardScaler-нормализации (обучающая выборка):")
print(np.allclose(X_train_std_custom, X_train_std_sklearn))

print("Сравнение StandardScaler-нормализации (тестовая выборка):")
print(np.allclose(X_test_std_custom, X_test_std_sklearn))

# Вывод результатов
print("\nКастомная нормализация (train):")
print(X_train_std_custom)
print("\nSklearn нормализация (train):")
print(X_train_std_sklearn)


# Создание и обучение моделей Linear Regression (аналитическое решение)
custom_ols_model_minmax = LinearRegressionCustom(method='analytical')
custom_ols_model_minmax.fit(X_train_scaled_custom, y_train)

sklearn_model_minmax = LinearRegression()
sklearn_model_minmax.fit(X_train_scaled_custom, y_train)

# Предсказания
ols_train_pred_minmax = custom_ols_model_minmax.predict(X_train_scaled_custom)
ols_test_pred_minmax = custom_ols_model_minmax.predict(X_test_scaled_custom)

sklearn_train_pred_minmax = sklearn_model_minmax.predict(X_train_scaled_custom)
sklearn_test_pred_minmax = sklearn_model_minmax.predict(X_test_scaled_custom)

result_MAE.loc[len(result_MAE)] = ['Custom Analytical_minmax', mean_absolute_error(y_train, ols_train_pred_minmax), mean_absolute_error(y_test, ols_test_pred_minmax)]
result_RMSE.loc[len(result_RMSE)] = ['Custom Analytical_minmax', mean_squared_error(y_train, ols_train_pred_minmax), mean_squared_error(y_test, ols_test_pred_minmax)]
result_R2.loc[len(result_R2)] = ['Custom Analytical_minmax', calculate_r2(y_train, ols_train_pred_minmax), calculate_r2(y_test, ols_test_pred_minmax)]

result_MAE.loc[len(result_MAE)] = ['Sklearn linreg_minmax', mean_absolute_error(y_train, sklearn_train_pred_minmax), mean_absolute_error(y_test, sklearn_test_pred_minmax)]
result_RMSE.loc[len(result_RMSE)] = ['Sklearn linreg_minmax', mean_squared_error(y_train, sklearn_train_pred_minmax), mean_squared_error(y_test, sklearn_test_pred_minmax)]
result_R2.loc[len(result_R2)] = ['Sklearn linreg_minmax', calculate_r2(y_train, sklearn_train_pred_minmax), calculate_r2(y_test, sklearn_test_pred_minmax)]

# Вывод результатов в виде DataFrame
print('MAE\n', result_MAE)
print('-----------------------------------------------')
print('RMSE\n',result_RMSE)
print('-----------------------------------------------')
print('R2\n',result_R2)


# Создание и обучение моделей с нормализованными данными
custom_ridge = RegularizedLinearRegression(method='ridge', learning_rate=0.01, n_iterations=2000, alpha=0.1)
custom_ridge.fit(X_train_scaled_custom, y_train)

sklearn_ridge = Ridge(alpha=0.1)
sklearn_ridge.fit(X_train_scaled_custom, y_train)

custom_lasso = RegularizedLinearRegression(method='lasso', learning_rate=0.01, n_iterations=2000, alpha=0.1)
custom_lasso.fit(X_train_scaled_custom, y_train)

sklearn_lasso = Lasso(alpha=0.1)
sklearn_lasso.fit(X_train_scaled_custom, y_train)

custom_elasticnet = RegularizedLinearRegression(method='elasticnet', learning_rate=0.01, n_iterations=2000, alpha=0.1, l1_ratio=0.5)
custom_elasticnet.fit(X_train_scaled_custom, y_train)

sklearn_elasticnet = ElasticNet(alpha=0.1, l1_ratio=0.5)
sklearn_elasticnet.fit(X_train_scaled_custom, y_train)

# Предсказания
custom_ridge_train_pred = custom_ridge.predict(X_train_scaled_custom)
custom_ridge_test_pred = custom_ridge.predict(X_test_scaled_custom)

sklearn_ridge_train_pred = sklearn_ridge.predict(X_train_scaled_custom)
sklearn_ridge_test_pred = sklearn_ridge.predict(X_test_scaled_custom)

custom_lasso_train_pred = custom_lasso.predict(X_train_scaled_custom)
custom_lasso_test_pred = custom_lasso.predict(X_test_scaled_custom)

sklearn_lasso_train_pred = sklearn_lasso.predict(X_train_scaled_custom)
sklearn_lasso_test_pred = sklearn_lasso.predict(X_test_scaled_custom)

custom_elasticnet_train_pred = custom_elasticnet.predict(X_train_scaled_custom)
custom_elasticnet_test_pred = custom_elasticnet.predict(X_test_scaled_custom)

sklearn_elasticnet_train_pred = sklearn_elasticnet.predict(X_train_scaled_custom)
sklearn_elasticnet_test_pred = sklearn_elasticnet.predict(X_test_scaled_custom)

# Добавление результатов в таблицу метрик
result_MAE.loc[len(result_MAE)] = ['Custom Ridge_minmax', mean_absolute_error(y_train, custom_ridge_train_pred), mean_absolute_error(y_test, custom_ridge_test_pred)]
result_RMSE.loc[len(result_RMSE)] = ['Custom Ridge_minmax', np.sqrt(mean_squared_error(y_train, custom_ridge_train_pred)), np.sqrt(mean_squared_error(y_test, custom_ridge_test_pred))]
result_R2.loc[len(result_R2)] = ['Custom Ridge_minmax', calculate_r2(y_train, custom_ridge_train_pred), calculate_r2(y_test, custom_ridge_test_pred)]

result_MAE.loc[len(result_MAE)] = ['Sklearn Ridge_minmax', mean_absolute_error(y_train, sklearn_ridge_train_pred), mean_absolute_error(y_test, sklearn_ridge_test_pred)]
result_RMSE.loc[len(result_RMSE)] = ['Sklearn Ridge_minmax', np.sqrt(mean_squared_error(y_train, sklearn_ridge_train_pred)), np.sqrt(mean_squared_error(y_test, sklearn_ridge_test_pred))]
result_R2.loc[len(result_R2)] = ['Sklearn Ridge_minmax', r2_score(y_train, sklearn_ridge_train_pred), r2_score(y_test, sklearn_ridge_test_pred)]

result_MAE.loc[len(result_MAE)] = ['Custom Lasso_minmax', mean_absolute_error(y_train, custom_lasso_train_pred), mean_absolute_error(y_test, custom_lasso_test_pred)]
result_RMSE.loc[len(result_RMSE)] = ['Custom Lasso_minmax', np.sqrt(mean_squared_error(y_train, custom_lasso_train_pred)), np.sqrt(mean_squared_error(y_test, custom_lasso_test_pred))]
result_R2.loc[len(result_R2)] = ['Custom Lasso_minmax', calculate_r2(y_train, custom_lasso_train_pred), calculate_r2(y_test, custom_lasso_test_pred)]

result_MAE.loc[len(result_MAE)] = ['Sklearn Lasso_minmax', mean_absolute_error(y_train, sklearn_lasso_train_pred), mean_absolute_error(y_test, sklearn_lasso_test_pred)]
result_RMSE.loc[len(result_RMSE)] = ['Sklearn Lasso_minmax', np.sqrt(mean_squared_error(y_train, sklearn_lasso_train_pred)), np.sqrt(mean_squared_error(y_test, sklearn_lasso_test_pred))]
result_R2.loc[len(result_R2)] = ['Sklearn Lasso_minmax', r2_score(y_train, sklearn_lasso_train_pred), r2_score(y_test, sklearn_lasso_test_pred)]

result_MAE.loc[len(result_MAE)] = ['Custom ElasticNet_minmax', mean_absolute_error(y_train, custom_elasticnet_train_pred), mean_absolute_error(y_test, custom_elasticnet_test_pred)]
result_RMSE.loc[len(result_RMSE)] = ['Custom ElasticNet_minmax', np.sqrt(mean_squared_error(y_train, custom_elasticnet_train_pred)), np.sqrt(mean_squared_error(y_test, custom_elasticnet_test_pred))]
result_R2.loc[len(result_R2)] = ['Custom ElasticNet_minmax', calculate_r2(y_train, custom_elasticnet_train_pred), calculate_r2(y_test, custom_elasticnet_test_pred)]

result_MAE.loc[len(result_MAE)] = ['Sklearn ElasticNet_minmax', mean_absolute_error(y_train, sklearn_elasticnet_train_pred), mean_absolute_error(y_test, sklearn_elasticnet_test_pred)]
result_RMSE.loc[len(result_RMSE)] = ['Sklearn ElasticNet_minmax', np.sqrt(mean_squared_error(y_train, sklearn_elasticnet_train_pred)), np.sqrt(mean_squared_error(y_test, sklearn_elasticnet_test_pred))]
result_R2.loc[len(result_R2)] = ['Sklearn ElasticNet_minmax', r2_score(y_train, sklearn_elasticnet_train_pred), r2_score(y_test, sklearn_elasticnet_test_pred)]

print('MAE\n', result_MAE)
print('-----------------------------------------------')
print('RMSE\n', result_RMSE)
print('-----------------------------------------------')
print('R2\n', result_R2)


# Создание и обучение моделей Linear Regression (аналитическое решение)
custom_ols_model_stdscaler = LinearRegressionCustom(method='analytical')
custom_ols_model_stdscaler.fit(X_train_std_custom, y_train)

sklearn_model_stdscaler = LinearRegression()
sklearn_model_stdscaler.fit(X_train_std_custom, y_train)

# Предсказания
ols_train_pred_stdscaler = custom_ols_model_stdscaler.predict(X_train_std_custom)
ols_test_pred_stdscaler = custom_ols_model_stdscaler.predict(X_test_std_custom)

sklearn_train_pred_stdscaler = sklearn_model_stdscaler.predict(X_train_std_custom)
sklearn_test_pred_stdscaler = sklearn_model_stdscaler.predict(X_test_std_custom)

result_MAE.loc[len(result_MAE)] = ['Custom Analytical_stdscaler', mean_absolute_error(y_train, ols_train_pred_stdscaler), mean_absolute_error(y_test, ols_test_pred_stdscaler)]
result_RMSE.loc[len(result_RMSE)] = ['Custom Analytical_stdscaler', mean_squared_error(y_train, ols_train_pred_stdscaler), mean_squared_error(y_test, ols_test_pred_stdscaler)]
result_R2.loc[len(result_R2)] = ['Custom Analytical_stdscaler', calculate_r2(y_train, ols_train_pred_stdscaler), calculate_r2(y_test, ols_test_pred_stdscaler)]

result_MAE.loc[len(result_MAE)] = ['Sklearn linreg_stdscaler', mean_absolute_error(y_train, sklearn_train_pred_stdscaler), mean_absolute_error(y_test, sklearn_test_pred_stdscaler)]
result_RMSE.loc[len(result_RMSE)] = ['Sklearn linreg_stdscaler', mean_squared_error(y_train, sklearn_train_pred_stdscaler), mean_squared_error(y_test, sklearn_test_pred_stdscaler)]
result_R2.loc[len(result_R2)] = ['Sklearn linreg_stdscaler', calculate_r2(y_train, sklearn_train_pred_stdscaler), calculate_r2(y_test, sklearn_test_pred_stdscaler)]

# Создание и обучение моделей с нормализованными данными
custom_ridge = RegularizedLinearRegression(method='ridge', learning_rate=0.01, n_iterations=2000, alpha=0.1)
custom_ridge.fit(X_train_std_custom, y_train)

sklearn_ridge = Ridge(alpha=0.1)
sklearn_ridge.fit(X_train_std_custom, y_train)

custom_lasso = RegularizedLinearRegression(method='lasso', learning_rate=0.01, n_iterations=2000, alpha=0.1)
custom_lasso.fit(X_train_std_custom, y_train)

sklearn_lasso = Lasso(alpha=0.1)
sklearn_lasso.fit(X_train_std_custom, y_train)

custom_elasticnet = RegularizedLinearRegression(method='elasticnet', learning_rate=0.01, n_iterations=2000, alpha=0.1, l1_ratio=0.5)
custom_elasticnet.fit(X_train_std_custom, y_train)

sklearn_elasticnet = ElasticNet(alpha=0.1, l1_ratio=0.5)
sklearn_elasticnet.fit(X_train_std_custom, y_train)

# Предсказания
custom_ridge_train_pred = custom_ridge.predict(X_train_std_custom)
custom_ridge_test_pred = custom_ridge.predict(X_test_std_custom)

sklearn_ridge_train_pred = sklearn_ridge.predict(X_train_std_custom)
sklearn_ridge_test_pred = sklearn_ridge.predict(X_test_std_custom)

custom_lasso_train_pred = custom_lasso.predict(X_train_std_custom)
custom_lasso_test_pred = custom_lasso.predict(X_test_std_custom)

sklearn_lasso_train_pred = sklearn_lasso.predict(X_train_std_custom)
sklearn_lasso_test_pred = sklearn_lasso.predict(X_test_std_custom)

custom_elasticnet_train_pred = custom_elasticnet.predict(X_train_std_custom)
custom_elasticnet_test_pred = custom_elasticnet.predict(X_test_std_custom)

sklearn_elasticnet_train_pred = sklearn_elasticnet.predict(X_train_std_custom)
sklearn_elasticnet_test_pred = sklearn_elasticnet.predict(X_test_std_custom)

# Добавление результатов в таблицу метрик
result_MAE.loc[len(result_MAE)] = ['Custom Ridge_stdscaler', mean_absolute_error(y_train, custom_ridge_train_pred), mean_absolute_error(y_test, custom_ridge_test_pred)]
result_RMSE.loc[len(result_RMSE)] = ['Custom Ridge_stdscaler', np.sqrt(mean_squared_error(y_train, custom_ridge_train_pred)), np.sqrt(mean_squared_error(y_test, custom_ridge_test_pred))]
result_R2.loc[len(result_R2)] = ['Custom Ridge_stdscaler', calculate_r2(y_train, custom_ridge_train_pred), calculate_r2(y_test, custom_ridge_test_pred)]

result_MAE.loc[len(result_MAE)] = ['Sklearn Ridge_stdscaler', mean_absolute_error(y_train, sklearn_ridge_train_pred), mean_absolute_error(y_test, sklearn_ridge_test_pred)]
result_RMSE.loc[len(result_RMSE)] = ['Sklearn Ridge_stdscaler', np.sqrt(mean_squared_error(y_train, sklearn_ridge_train_pred)), np.sqrt(mean_squared_error(y_test, sklearn_ridge_test_pred))]
result_R2.loc[len(result_R2)] = ['Sklearn Ridge_stdscaler', r2_score(y_train, sklearn_ridge_train_pred), r2_score(y_test, sklearn_ridge_test_pred)]

result_MAE.loc[len(result_MAE)] = ['Custom Lasso_stdscaler', mean_absolute_error(y_train, custom_lasso_train_pred), mean_absolute_error(y_test, custom_lasso_test_pred)]
result_RMSE.loc[len(result_RMSE)] = ['Custom Lasso_stdscaler', np.sqrt(mean_squared_error(y_train, custom_lasso_train_pred)), np.sqrt(mean_squared_error(y_test, custom_lasso_test_pred))]
result_R2.loc[len(result_R2)] = ['Custom Lasso_stdscaler', calculate_r2(y_train, custom_lasso_train_pred), calculate_r2(y_test, custom_lasso_test_pred)]

result_MAE.loc[len(result_MAE)] = ['Sklearn Lasso_stdscaler', mean_absolute_error(y_train, sklearn_lasso_train_pred), mean_absolute_error(y_test, sklearn_lasso_test_pred)]
result_RMSE.loc[len(result_RMSE)] = ['Sklearn Lasso_stdscaler', np.sqrt(mean_squared_error(y_train, sklearn_lasso_train_pred)), np.sqrt(mean_squared_error(y_test, sklearn_lasso_test_pred))]
result_R2.loc[len(result_R2)] = ['Sklearn Lasso_stdscaler', r2_score(y_train, sklearn_lasso_train_pred), r2_score(y_test, sklearn_lasso_test_pred)]

result_MAE.loc[len(result_MAE)] = ['Custom ElasticNet_stdscaler', mean_absolute_error(y_train, custom_elasticnet_train_pred), mean_absolute_error(y_test, custom_elasticnet_test_pred)]
result_RMSE.loc[len(result_RMSE)] = ['Custom ElasticNet_stdscaler', np.sqrt(mean_squared_error(y_train, custom_elasticnet_train_pred)), np.sqrt(mean_squared_error(y_test, custom_elasticnet_test_pred))]
result_R2.loc[len(result_R2)] = ['Custom ElasticNet_stdscaler', calculate_r2(y_train, custom_elasticnet_train_pred), calculate_r2(y_test, custom_elasticnet_test_pred)]

result_MAE.loc[len(result_MAE)] = ['Sklearn ElasticNet_stdscaler', mean_absolute_error(y_train, sklearn_elasticnet_train_pred), mean_absolute_error(y_test, sklearn_elasticnet_test_pred)]
result_RMSE.loc[len(result_RMSE)] = ['Sklearn ElasticNet_stdscaler', np.sqrt(mean_squared_error(y_train, sklearn_elasticnet_train_pred)), np.sqrt(mean_squared_error(y_test, sklearn_elasticnet_test_pred))]
result_R2.loc[len(result_R2)] = ['Sklearn ElasticNet_stdscaler', r2_score(y_train, sklearn_elasticnet_train_pred), r2_score(y_test, sklearn_elasticnet_test_pred)]


print('MAE\n', result_MAE)
print('-----------------------------------------------')
print('RMSE\n', result_RMSE)
print('-----------------------------------------------')
print('R2\n', result_R2)


features = ['bathrooms', 'bedrooms', 'interest_level_encoded']
X_poly_base = df_filtered[features]

X_train_poly_base, X_test_poly_base, y_train, y_test = train_test_split(
    X_poly_base, df_filtered['price'], test_size=0.2, random_state=42
)

poly_features = PolynomialFeatures(degree=10, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train_poly_base)
X_test_poly = poly_features.transform(X_test_poly_base)

# Находим константные столбцы в обучающей выборке
constant_cols = np.where(X_train_poly.std(axis=0) == 0)[0]
# Удаляем эти столбцы из обеих выборок
X_train_poly = np.delete(X_train_poly, constant_cols, axis=1)
X_test_poly = np.delete(X_test_poly, constant_cols, axis=1)

# Scale the polynomial features using StandardScaler
scaler = StandardScaler()
X_train_poly_scaled = scaler.fit_transform(X_train_poly)
X_test_poly_scaled = scaler.transform(X_test_poly)



# Linear Regression (overfitted model)
# Custom Analytical
custom_ols_poly = LinearRegressionCustom(method='analytical')
custom_ols_poly.fit(X_train_poly_scaled, y_train)
ols_train_pred_poly = custom_ols_poly.predict(X_train_poly_scaled)
ols_test_pred_poly = custom_ols_poly.predict(X_test_poly_scaled)
result_MAE.loc[len(result_MAE)] = ['Custom Analytical_poly', mean_absolute_error(y_train, ols_train_pred_poly), mean_absolute_error(y_test, ols_test_pred_poly)]
result_RMSE.loc[len(result_RMSE)] = ['Custom Analytical_poly', mean_squared_error(y_train, ols_train_pred_poly), mean_squared_error(y_test, ols_test_pred_poly)]
result_R2.loc[len(result_R2)] = ['Custom Analytical_poly', calculate_r2(y_train, ols_train_pred_poly), calculate_r2(y_test, ols_test_pred_poly)]

# Sklearn Linear Regression
sklearn_linreg_poly = LinearRegression()
sklearn_linreg_poly.fit(X_train_poly_scaled, y_train)
linreg_train_pred_poly = sklearn_linreg_poly.predict(X_train_poly_scaled)
linreg_test_pred_poly = sklearn_linreg_poly.predict(X_test_poly_scaled)
result_MAE.loc[len(result_MAE)] = ['Sklearn linreg_poly', mean_absolute_error(y_train, linreg_train_pred_poly), mean_absolute_error(y_test, linreg_test_pred_poly)]
result_RMSE.loc[len(result_RMSE)] = ['Sklearn linreg_poly', mean_squared_error(y_train, linreg_train_pred_poly), mean_squared_error(y_test, linreg_test_pred_poly)]
result_R2.loc[len(result_R2)] = ['Sklearn linreg_poly', r2_score(y_train, linreg_train_pred_poly), r2_score(y_test, linreg_test_pred_poly)]


# Regularized models
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
learning_rate_for_poly = 1e-6 # Уменьшаем learning_rate для полиномиальных признаков

for alpha in alphas:
    # Custom Ridge
    custom_ridge_poly = RegularizedLinearRegression(method='ridge', learning_rate=learning_rate_for_poly, n_iterations=2000, alpha=alpha)
    custom_ridge_poly.fit(X_train_poly_scaled, y_train)
    custom_ridge_train_pred = custom_ridge_poly.predict(X_train_poly_scaled)
    custom_ridge_test_pred = custom_ridge_poly.predict(X_test_poly_scaled)
    result_MAE.loc[len(result_MAE)] = [f'Custom Ridge_poly_alpha={alpha}', mean_absolute_error(y_train, custom_ridge_train_pred), mean_absolute_error(y_test, custom_ridge_test_pred)]
    result_RMSE.loc[len(result_RMSE)] = [f'Custom Ridge_poly_alpha={alpha}', np.sqrt(mean_squared_error(y_train, custom_ridge_train_pred)), np.sqrt(mean_squared_error(y_test, custom_ridge_test_pred))]
    result_R2.loc[len(result_R2)] = [f'Custom Ridge_poly_alpha={alpha}', calculate_r2(y_train, custom_ridge_train_pred), calculate_r2(y_test, custom_ridge_test_pred)]

    # Sklearn Ridge
    sklearn_ridge_poly = Ridge(alpha=alpha)
    sklearn_ridge_poly.fit(X_train_poly_scaled, y_train)
    sklearn_ridge_train_pred = sklearn_ridge_poly.predict(X_train_poly_scaled)
    sklearn_ridge_test_pred = sklearn_ridge_poly.predict(X_test_poly_scaled)
    result_MAE.loc[len(result_MAE)] = [f'Sklearn Ridge_poly_alpha={alpha}', mean_absolute_error(y_train, sklearn_ridge_train_pred), mean_absolute_error(y_test, sklearn_ridge_test_pred)]
    result_RMSE.loc[len(result_RMSE)] = [f'Sklearn Ridge_poly_alpha={alpha}', np.sqrt(mean_squared_error(y_train, sklearn_ridge_train_pred)), np.sqrt(mean_squared_error(y_test, sklearn_ridge_test_pred))]
    result_R2.loc[len(result_R2)] = [f'Sklearn Ridge_poly_alpha={alpha}', r2_score(y_train, sklearn_ridge_train_pred), r2_score(y_test, sklearn_ridge_test_pred)]

    # Custom Lasso
    custom_lasso_poly = RegularizedLinearRegression(method='lasso', learning_rate=learning_rate_for_poly, n_iterations=2000, alpha=alpha)
    custom_lasso_poly.fit(X_train_poly_scaled, y_train)
    custom_lasso_train_pred = custom_lasso_poly.predict(X_train_poly_scaled)
    custom_lasso_test_pred = custom_lasso_poly.predict(X_test_poly_scaled)
    result_MAE.loc[len(result_MAE)] = [f'Custom Lasso_poly_alpha={alpha}', mean_absolute_error(y_train, custom_lasso_train_pred), mean_absolute_error(y_test, custom_lasso_test_pred)]
    result_RMSE.loc[len(result_RMSE)] = [f'Custom Lasso_poly_alpha={alpha}', np.sqrt(mean_squared_error(y_train, custom_lasso_train_pred)), np.sqrt(mean_squared_error(y_test, custom_lasso_test_pred))]
    result_R2.loc[len(result_R2)] = [f'Custom Lasso_poly_alpha={alpha}', calculate_r2(y_train, custom_lasso_train_pred), calculate_r2(y_test, custom_lasso_test_pred)]

    # Sklearn Lasso
    sklearn_lasso_poly = Lasso(alpha=alpha, max_iter=2000)
    sklearn_lasso_poly.fit(X_train_poly_scaled, y_train)
    sklearn_lasso_train_pred = sklearn_lasso_poly.predict(X_train_poly_scaled)
    sklearn_lasso_test_pred = sklearn_lasso_poly.predict(X_test_poly_scaled)
    result_MAE.loc[len(result_MAE)] = [f'Sklearn Lasso_poly_alpha={alpha}', mean_absolute_error(y_train, sklearn_lasso_train_pred), mean_absolute_error(y_test, sklearn_lasso_test_pred)]
    result_RMSE.loc[len(result_RMSE)] = [f'Sklearn Lasso_poly_alpha={alpha}', np.sqrt(mean_squared_error(y_train, sklearn_lasso_train_pred)), np.sqrt(mean_squared_error(y_test, sklearn_lasso_test_pred))]
    result_R2.loc[len(result_R2)] = [f'Sklearn Lasso_poly_alpha={alpha}', r2_score(y_train, sklearn_lasso_train_pred), r2_score(y_test, sklearn_lasso_test_pred)]

    # Custom ElasticNet
    custom_elasticnet_poly = RegularizedLinearRegression(method='elasticnet', learning_rate=learning_rate_for_poly, n_iterations=2000, alpha=alpha, l1_ratio=0.5)
    custom_elasticnet_poly.fit(X_train_poly_scaled, y_train)
    custom_elasticnet_train_pred = custom_elasticnet_poly.predict(X_train_poly_scaled)
    custom_elasticnet_test_pred = custom_elasticnet_poly.predict(X_test_poly_scaled)
    result_MAE.loc[len(result_MAE)] = [f'Custom ElasticNet_poly_alpha={alpha}', mean_absolute_error(y_train, custom_elasticnet_train_pred), mean_absolute_error(y_test, custom_elasticnet_test_pred)]
    result_RMSE.loc[len(result_RMSE)] = [f'Custom ElasticNet_poly_alpha={alpha}', np.sqrt(mean_squared_error(y_train, custom_elasticnet_train_pred)), np.sqrt(mean_squared_error(y_test, custom_elasticnet_test_pred))]
    result_R2.loc[len(result_R2)] = [f'Custom ElasticNet_poly_alpha={alpha}', calculate_r2(y_train, custom_elasticnet_train_pred), calculate_r2(y_test, custom_elasticnet_test_pred)]

    # Sklearn ElasticNet
    sklearn_elasticnet_poly = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=2000)
    sklearn_elasticnet_poly.fit(X_train_poly_scaled, y_train)
    sklearn_elasticnet_train_pred = sklearn_elasticnet_poly.predict(X_train_poly_scaled)
    sklearn_elasticnet_test_pred = sklearn_elasticnet_poly.predict(X_test_poly_scaled)
    result_MAE.loc[len(result_MAE)] = [f'Sklearn ElasticNet_poly_alpha={alpha}', mean_absolute_error(y_train, sklearn_elasticnet_train_pred), mean_absolute_error(y_test, sklearn_elasticnet_test_pred)]
    result_RMSE.loc[len(result_RMSE)] = [f'Sklearn ElasticNet_poly_alpha={alpha}', np.sqrt(mean_squared_error(y_train, sklearn_elasticnet_train_pred)), np.sqrt(mean_squared_error(y_test, sklearn_elasticnet_test_pred))]
    result_R2.loc[len(result_R2)] = [f'Sklearn ElasticNet_poly_alpha={alpha}', r2_score(y_train, sklearn_elasticnet_train_pred), r2_score(y_test, sklearn_elasticnet_test_pred)]

print('MAE\n', result_MAE)
print('-----------------------------------------------')
print('RMSE\n',result_RMSE)
print('-----------------------------------------------')
print('R2\n',result_R2)


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', None)


print('MAE\n', result_MAE)
print('-----------------------------------------------')
print('RMSE\n',result_RMSE)
print('-----------------------------------------------')
print('R2\n',result_R2)


print('MAE\n', result_MAE)
print('-----------------------------------------------')
print('RMSE\n',result_RMSE)
print('-----------------------------------------------')
print('R2\n',result_R2)

