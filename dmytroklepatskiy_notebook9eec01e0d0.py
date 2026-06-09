# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gc


!unzip /kaggle/input/santander-product-recommendation/train_ver2.csv.zip
!unzip /kaggle/input/santander-product-recommendation/test_ver2.csv.zip


def reduce_mem_usage(df):
    """
    Замість ручного визначення типу, ця функція перевіряє кожен стовпець
    і перетворює його на найменший тип даних (int8, float32 тощо), не пошкоджуючи дані.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Базове використання пам’яті: {start_mem:.2f} МБ')

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            # Зменшити цілі числа (int64 -> int8/16/32)
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            # Зменшити дробові числа (float64 -> float32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32) # float16 іноді викликає проблеми, 32 є безпечним
                else:
                    df[col] = df[col].astype(np.float32)
        else:
            # Перетворити об'єктні (рядкові) стовпці на 'category' (значно прискорює роботу)
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Кінцеве використання пам’яті: {end_mem:.2f} МБ ({100 * (start_mem - end_mem) / start_mem:.1f}% економії)')
    return df


train_data = pd.read_csv('/kaggle/working/train_ver2.csv', nrows=500000)
test_data = pd.read_csv('/kaggle/working/test_ver2.csv')


drop_cols = ['fecha_alta', 'ult_fec_cli_1t', 'tipodom', 'cod_prov', 'conyuemp', 'fecha_dato']
train_data.drop(columns=drop_cols, errors='ignore', inplace=True)
test_data.drop(columns=drop_cols, errors='ignore', inplace=True)


train_data.head()


train_data.info()


numeric_cols = train_data.select_dtypes(include=['number']).columns
train_data[numeric_cols] = train_data[numeric_cols].fillna(-1)


cat_cols = train_data.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    # Кодуємо Train і Test разом, щоб значення збігалися
    combined = pd.concat([train_data[col], test_data[col]], axis=0).astype('category')
    train_data[col] = combined[:len(train_data)].cat.codes
    test_data[col] = combined[len(train_data):].cat.codes


train_data = reduce_mem_usage(train_data)


# Це стовпці, які є в train, але немає в test, і починаються на "ind_"
target_cols = [col for col in train_data.columns if col not in test_data.columns and col.startswith('ind_')]

print(f"Кількість цільових продуктів: {len(target_cols)}")


X = train_data.drop(target_cols + ['ncodpers'], axis=1)
y = train_data[target_cols]
test_ids = test_data['ncodpers']
X_test = test_data.drop(['ncodpers'], axis=1)

print(f"Кількість ознак для навчання: {X.shape[1]}")
print(f"Кількість продуктів для прогнозу: {len(target_cols)}")


gc.collect()


print(f"Колонки: {X.columns.tolist()}")


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Запустіть цей код перед початком циклу навчання
print("--- Перевірка нечислових стовпців у X та X_test ---")

# Знайдемо стовпці типу 'object' (string/текст) у тренувальному наборі
object_cols_X = X.select_dtypes(include='object').columns
print(f"Категоріальні стовпці у X: {list(object_cols_X)}")

# Знайдемо стовпці типу 'object' у тестовому наборі
object_cols_X_test = X_test.select_dtypes(include='object').columns
print(f"Категоріальні стовпці у X_test: {list(object_cols_X_test)}")


from sklearn.preprocessing import LabelEncoder
import pandas as pd # Переконайтеся, що pandas імпортовано

# 1. Об'єднуємо набори для узгодженого кодування (як ви вже робили)
combined = pd.concat([X, X_test], ignore_index=True)

# 2. Список категоріальних стовпців, які потрібно закодувати
# Оскільки ми вже обробили більшість числових стовпців, ми фокусуємося на 'object'
categorical_cols_to_encode = combined.select_dtypes(include='object').columns

for col in categorical_cols_to_encode:
    
    # --- ВИПРАВЛЕННЯ ПОМИЛКИ: ПЕРЕТВОРЕННЯ НА STR ---
    # Заповнюємо пропуски і одразу перетворюємо весь стовпець на рядки.
    combined[col] = combined[col].fillna('MISSING').astype(str)
    
    # Виконуємо кодування
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])

# 3. Розділяємо назад
X_fixed = combined.iloc[:len(X)].copy()
X_test_fixed = combined.iloc[len(X):].copy()

# Тепер використовуємо X_fixed та X_test_fixed для навчання
# Примітка: використовуйте .copy(), щоб уникнути SettingWithCopyWarning


from sklearn.ensemble import RandomForestClassifier

probs_dict = {}

# Для швидкості я встановив низьку кількість дерев (n_estimators) та обмежену глибину (max_depth).
model = RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42)

# 1. ЕКСПЛІЦИТНА ПЕРЕВІРКА НА NaN (рекомендовано)
print("Кількість NaN у X_fixed перед навчанням:", X_fixed.isnull().sum().sum())
print("Кількість NaN у X_test_fixed перед прогнозуванням:", X_test_fixed.isnull().sum().sum())

# 2. ФІКСАЦІЯ: Заповнення всіх залишкових NaN значенням -1
# Це гарантує, що модель отримає лише числові значення.

X_final = X_fixed.fillna(-1)
X_test_final = X_test_fixed.fillna(-1)

print("Кількість NaN у X_final після обробки:", X_final.isnull().sum().sum())
print("Кількість NaN у X_test_final після обробки:", X_test_final.isnull().sum().sum())

for col in target_cols:
    y_current = y[col]
    
    model.fit(X_final, y_current) 
    
    preds = model.predict_proba(X_test_final)[:, 1]
    
    probs_dict[col] = preds
    print(f"Продукт {col} - модель навчена.")


print("Всі прогнози зроблено!")


for col in X.columns:
    # Якщо цей стовпець існує в тестовому наборі і його тип 'object' (текст)
    if col in X_test.columns and X_test[col].dtype == 'object':
        print(f"Виправлення: {col}")
        # Перетворює ' NA' або неправильні тексти в NaN, потім заповнює -1
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(-1)


print("Відсутні значення заповнюються -1...")
X_test = X_test.fillna(-1)


if X_test.isnull().values.any():
    print("УВАГА: Все ще є порожні значення!")
else:
    print("Дані чисті, порожніх значень немає.")


# Перетворюємо словник прогнозів у DataFrame
preds_df = pd.DataFrame(probs_dict)

# Функція для вибору топ-7 продуктів
def get_top7_products(row):
    # Сортуємо значення в рядку від найбільшого до найменшого і беремо індекси (назви колонок)
    top_cols = row.sort_values(ascending=False).head(7).index.tolist()
    # Об'єднуємо назви в рядок через пробіл
    return " ".join(top_cols)

print("Формування фінальних рекомендацій (це може зайняти хвилину)...")

# Застосовуємо функцію до кожного рядка (клієнта)
final_recommendations = preds_df.apply(get_top7_products, axis=1)

# Створюємо фінальний датафрейм
submission = pd.DataFrame({
    'ncodpers': test_ids,
    'added_products': final_recommendations
})

submission.head()


submission.to_csv('submission_fixed.csv', index=False)
print("Файл 'submission_fixed.csv' успішно збережено!")


submission

