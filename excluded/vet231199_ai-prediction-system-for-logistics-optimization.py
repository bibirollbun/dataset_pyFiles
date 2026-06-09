import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

# 1. Загрузка данных
train_df = pd.read_csv('/kaggle/input/ctai-ctd-hackathon/train.csv')

# 2. Очистка и подготовка данных
# Удаляем ненужные или неподдерживаемые столбцы
columns_to_drop = [
    'PROJECTNUMBER', 'invoiceDate', 'ItemDescription', 
    'MasterItemNo', 'UOM', 'ExtendedQuantity', 
    'PriceUOM', 'CONSTRUCTION_START_DATE', 'SUBSTANTIAL_COMPLETION_DATE'
]
train_df = train_df.drop(columns=columns_to_drop, errors='ignore')

# Обработка числовых признаков
for col in ['invoiceTotal', 'UnitPrice', 'ExtendedPrice', 'QtyShipped']:
    if col in train_df.columns:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)

# Обработка категориальных признаков
categorical_cols = ['PROJECT_CITY', 'STATE', 'PROJECT_COUNTRY', 'CORE_MARKET', 'PROJECT_TYPE']
for col in categorical_cols:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype(str)
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        joblib.dump(le, f'le_{col}.pkl')  # сохраняем лейберы для предсказаний

# 3. Формируем признаки и целевые переменные
X = train_df.drop(columns=['id', 'invoiceId', 'QtyShipped', 'UnitPrice'], errors='ignore')
y_qty = train_df['QtyShipped']
y_price = train_df['UnitPrice']

# 4. Сохраняем список признаков
features_used_in_training = X.columns
joblib.dump(features_used_in_training, 'features_used_in_training.pkl')

# 5. Разделение данных
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y_qty, test_size=0.2, random_state=42
)

# Проверка размеров
print(f"Размер X_train: {X_train.shape}")
print(f"Размер y_train (QtyShipped): {y_train.shape}")
print(f"Размер y_price: {y_price.shape}")

# 6. Обучение моделей
model_qty = lgb.LGBMRegressor()
model_qty.fit(X_train, y_train)

model_price = lgb.LGBMRegressor()
model_price.fit(X_train, y_price.loc[X_train.index])  # убедитесь, что размеры совпадают

# 7. Создаём папку 'models', если её нет
os.makedirs('models', exist_ok=True)

# 8. Сохраняем модели
joblib.dump(model_qty, 'models/model_qty.pkl')
joblib.dump(model_price, 'models/model_price.pkl')

print("Модели обучены и успешно сохранены.")


import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score  

# Пути к файлам
model_path = '/kaggle/working/models/model_qty.pkl'
features_path = '/kaggle/working/features_used_in_training.pkl'
test_csv_path = '/kaggle/input/ctai-ctd-hackathon/test.csv'

# Загрузка модели и признаков
try:
    model_qty = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Ошибка загрузки модели: {e}")

try:
    features_used_in_training = joblib.load(features_path)
except Exception as e:
    raise RuntimeError(f"Ошибка загрузки признаков: {e}")

# Загрузка тестовых данных
try:
    test_df = pd.read_csv(test_csv_path)
except Exception as e:
    raise RuntimeError(f"Ошибка загрузки тестового файла: {e}")

# Название колонок
id_col = 'id'

categorical_cols = ['PROJECT_CITY', 'STATE', 'PROJECT_COUNTRY', 'CORE_MARKET', 'PROJECT_TYPE']
for col in categorical_cols:
    le_path = f'le_{col}.pkl'
    if os.path.exists(le_path):
        le = joblib.load(le_path)
        test_df[col] = test_df[col].astype(str)
        # Обработка значений, которых нет в лейбл-классе
        test_df[col] = test_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        test_df[col] = le.transform(test_df[col])
    else:
        print(f"Лейбл-энкодер для {col} не найден по пути {le_path}. Пропускаем обработку этого признака.")

numeric_cols = ['MW', 'SIZE_BUILDINGSIZE', 'NUMFLOORS', 'NUMROOMS', 'NUMBEDS', 'invoiceTotal', 'ExtendedPrice', 'REVISED_ESTIMATE']
for col in numeric_cols:
    if col in test_df.columns:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
        test_df[col] = test_df[col].fillna(0)
    else:
        test_df[col] = 0

missing_features = [f for f in features_used_in_training if f not in test_df.columns]
if missing_features:
    print(f"Отсутствующие признаки в тестовых данных: {missing_features}")
    for feat in missing_features:
        test_df[feat] = 0

X_test = test_df[features_used_in_training]

print("Типы данных в X_test:\n", X_test.dtypes)
print("Первые строки X_test:\n", X_test.head())

X_test = X_test.fillna(0)
X_test = X_test.replace([float('inf'), -float('inf')], 0)

print("Количество NaN после обработки:", X_test.isnull().sum().sum())
print("Количество inf после обработки:", ((X_test == float('inf')) | (X_test == -float('inf'))).sum().sum())

# Предсказание
try:
    test_df['QtyShipped'] = model_qty.predict(X_test)
except Exception as e:
    print("Ошибка при предсказании:", e)
    raise

if 'TARGET' in test_df.columns:
    y_true = test_df['TARGET']
    y_pred = test_df['QtyShipped']
    try:
        score = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {score}")
    except Exception as e:
        print(f"Ошибка при вычислении метрики: {e}")
else:
    print("В тестовых данных отсутствует истинная целевая переменная 'TARGET'. Метрику вычислить нельзя.")

if 'PROJECTNUMBER' in test_df.columns:
    test_df['MasterItemNo'] = test_df['PROJECTNUMBER']
else:
    test_df['MasterItemNo'] = 'N/A'

final_submission = pd.DataFrame({
    'id': test_df[id_col],
    'MasterItemNo': test_df['MasterItemNo'],
    'QtyShipped': test_df['QtyShipped']
})

final_submission['QtyShipped'] = final_submission['QtyShipped'].round().astype(int)

final_submission['QtyShipped'] = final_submission['QtyShipped'].clip(lower=0)

assert final_submission['QtyShipped'].ge(0).all(), "В файле есть отрицательные значения."

# Сохраняем в файл
final_submission.to_csv('submission.csv', index=False)
print("Файл submission.csv успешно создан.")

