import numpy as np  # Библиотека для линейной алгебры
import pandas as pd  # Библиотека для обработки данных и работы с CSV файлами
import os
import lightgbm as lgb  # Модель LightGBM
from sklearn.ensemble import RandomForestRegressor  # Модель случайного леса
from sklearn.model_selection import cross_val_score  # Кросс-валидация для оценки модели

# Настройка вывода: показать все строки в DataFrame
pd.set_option('display.max_rows', None)

# Чтение файлов из каталога /kaggle/input
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

path_train = '/kaggle/input/try-to-calculate-math-expectation/train.csv'
data_train = pd.read_csv(path_train, sep=";")
print("Размер обучающего набора:", data_train.shape)
print(data_train.to_string(index=False))

path_test = '/kaggle/input/try-to-calculate-math-expectation/test.csv'
data_test = pd.read_csv(path_test, sep=";")
print("Размер тестового набора:", data_test.shape)
print(data_test.to_string(index=False))

path_sample = '/kaggle/input/try-to-calculate-math-expectation/sample_submission.csv'
data_sample = pd.read_csv(path_sample)
print("Полные данные файла отправки:")
print(data_sample.to_string(index=False))

data_train.columns = data_train.columns.str.strip()
data_test.columns = data_test.columns.str.strip()

y_train = data_train.iloc[:, 1:17].mean(axis=1)

selected_cols = ["value 2", "value 3", "value 7", "value 8", "value 9", "value 10"]
print("Выбранные столбцы для признаков:", selected_cols)

for col in selected_cols:
    if data_train[col].isnull().any():
        mean_val = data_train[col].mean()
        data_train[col].fillna(mean_val, inplace=True)
    if data_test[col].isnull().any():
        mean_val = data_test[col].mean()
        data_test[col].fillna(mean_val, inplace=True)

def create_features(df, cols):
    features = pd.DataFrame()
    features['mean'] = df[cols].mean(axis=1)
    features['sum'] = df[cols].sum(axis=1)
    features['min'] = df[cols].min(axis=1)
    features['max'] = df[cols].max(axis=1)
    features['std'] = df[cols].std(axis=1)
    features['median'] = df[cols].median(axis=1)
    for col in cols:
        features[col] = df[col]
    features['range'] = features['max'] - features['min']
    return features

X_train = create_features(data_train, selected_cols)
print("Примеры признаков обучающего набора:")
print(X_train.to_string(index=False))

X_test = create_features(data_test, selected_cols)
print("Примеры признаков тестового набора:")
print(X_test.to_string(index=False))

lgb_model = lgb.LGBMRegressor(
    random_state=42,
    n_estimators=1000,
    min_split_gain=0.0,
    min_child_samples=1,
    max_depth=15,
    verbose=-1
)
lgb_cv_scores = cross_val_score(lgb_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("Средняя отрицательная MSE LightGBM:", lgb_cv_scores.mean())

rf_model = RandomForestRegressor(random_state=42, n_estimators=200)
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("Средняя отрицательная MSE Случайного Леса:", rf_cv_scores.mean())

lgb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

lgb_preds = lgb_model.predict(X_test)
rf_preds = rf_model.predict(X_test)
ensemble_preds = (lgb_preds + rf_preds) / 2
ensemble_preds = np.round(ensemble_preds, 2)

result = pd.DataFrame({
    'id': data_test['ID'],
    'target_feature': ensemble_preds
})
print("Полные данные результатов отправки:")
print(result.to_string(index=False))

result.to_csv('submission.csv', index=False)

path_my_sample = '/kaggle/working/submission.csv'
data_my_sample = pd.read_csv(path_my_sample)
print("Размер итогового файла submission.csv:", data_my_sample.shape)
print("Полные данные итогового файла submission.csv:")
print(data_my_sample.to_string(index=False)) 


