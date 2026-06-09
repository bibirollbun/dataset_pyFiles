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


pd.set_option('display.max_columns', None)


!ls /kaggle/input/fall-ml-6-mipt-2025


df_sample = pd.read_csv("/kaggle/input/fall-ml-6-mipt-2025/sample.csv")
df_train = pd.read_csv("/kaggle/input/fall-ml-6-mipt-2025/train.csv")
df_test = pd.read_csv("/kaggle/input/fall-ml-6-mipt-2025/test.csv")


df_sample


df_train["TARGET"].unique()


df_train


df_train.columns


cols_to_drop = ["SSN", "Клиент_Инфо"]

df_train = df_train.drop(columns=cols_to_drop, errors="ignore")
df_test = df_test.drop(columns=cols_to_drop, errors="ignore")


import re

def convert_age_to_months(age_str):
    """
    Преобразует строку '33 Years and 2 Months' в число месяцев.
    """
    if pd.isna(age_str):
        return None
    
    # Находим число лет
    years_match = re.search(r"(\d+)\s*Years?", age_str)
    years = int(years_match.group(1)) if years_match else 0
    
    # Находим число месяцев
    months_match = re.search(r"(\d+)\s*Months?", age_str)
    months = int(months_match.group(1)) if months_match else 0
    
    return years * 12 + months

# Применяем к df_train и df_test
df_train["Возраст_кредитной_истории"] = df_train["Возраст_кредитной_истории"].apply(convert_age_to_months)
df_test["Возраст_кредитной_истории"] = df_test["Возраст_кредитной_истории"].apply(convert_age_to_months)

# Проверка
df_train[["Возраст_кредитной_истории"]].head()



# Список числовых столбцов (можно расширить по необходимости)
numeric_cols = [
    "Колво_отсроченных_платежей", "Месячный_баланс", "Задержка_платежа_дни",
    "Годовой_доход", "Сумма_ежемесячных_выплат", "Процентная_ставка", "Колво_займов",
    "Колво_банковских_счетов", "Оставшийся_долг", "Коэффициент_использования_кредита",
    "Колво_кредитных_карт", "Колво_кредитных_запросов", "Месячная_зарплата",
    "Сумма_инвестиций", "Изменение_кредитного_лимита"
]

# Преобразуем тип данных
for col in numeric_cols:
    df_train[col] = pd.to_numeric(df_train[col], errors="coerce")
    df_test[col] = pd.to_numeric(df_test[col], errors="coerce")

# Проверка
print(df_train[numeric_cols].dtypes)
print(df_test[numeric_cols].dtypes)



df_train["TARGET"].value_counts()


df_train.isna().sum()[df_train.isna().sum() > 0]


import numpy as np

cols_to_fill = [
    "Колво_отсроченных_платежей",
    "Месячный_баланс",
    "Тип_кредита",
    "Колво_кредитных_запросов",
    "Месячная_зарплата",
    "Возраст_кредитной_истории",
    "Сумма_инвестиций",
    "Годовой_доход",
    "Колво_займов",
    "Оставшийся_долг",
    "Изменение_кредитного_лимита"
]

fill_dict = {}  # сюда сохраним медианы и "Unknown"

for col in cols_to_fill:
    if np.issubdtype(df_train[col].dtype, np.number):
        value = df_train[col].median()
    else:
        value = "Unknown"

    fill_dict[col] = value

# вывод словаря
print(fill_dict)
df_train = df_train.fillna(fill_dict)


df_train.isna().sum()[df_train.isna().sum() > 0]


df_test = df_test.fillna(fill_dict)


import pandas as pd

stats_list = []

cols = list(df_train.columns)
cols.remove("ID_клиента")

for col in cols:
    data = df_train[col]
    if data.dtype != object:
        stats = {
            "Столбец": col,
            "Тип": data.dtype,
            "Среднее": data.mean() if pd.api.types.is_numeric_dtype(data) else None,
            "Медиана": data.median() if pd.api.types.is_numeric_dtype(data) else None,
            "Минимум": data.min(),
            "Максимум": data.max(),
            "Стандартное отклонение": data.std() if pd.api.types.is_numeric_dtype(data) else None,
            "Уникальные значения": data.nunique()
        }
        stats_list.append(stats)

# Превращаем в DataFrame для наглядного вывода
stats_df = pd.DataFrame(stats_list)
pd.set_option('display.max_columns', None)
stats_df






import pandas as pd

cat_cols = df_train.select_dtypes(include=["object"]).columns.tolist()
if "ID_клиента" in cat_cols:
    cat_cols.remove("ID_клиента")  # исключаем ID

cat_stats = []

for col in cat_cols:
    nunique = df_train[col].nunique()
    perc_unique = nunique / len(df_train) * 100
    cat_stats.append({
        "Столбец": col,
        "Уникальные значения": nunique,
        "% уникальных значений": round(perc_unique, 2)
    })

cat_stats_df = pd.DataFrame(cat_stats)
pd.set_option('display.max_columns', None)
cat_stats_df.sort_values("% уникальных значений", ascending=False, inplace=True)
cat_stats_df



from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Целевая переменная
target = "TARGET"

# Создаём LabelEncoder и обучаем на df_train
le_target = LabelEncoder()
le_target.fit(df_train[target])

# Словарь прямого и обратного преобразования
target_mapping = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))
reverse_mapping = dict(zip(le_target.transform(le_target.classes_), le_target.classes_))

# Функции для преобразования
def target_to_num(value):
    return target_mapping.get(value, -1)  # неизвестные категории -> -1

def num_to_target(value):
    return reverse_mapping.get(value, "Unknown")  # неизвестные числа -> "Unknown"

# Применяем к df_train
df_train["TARGET"] = df_train[target].apply(target_to_num)

# Разделяем данные
# Сначала отделяем 20% в test
train_full, test = train_test_split(df_train, test_size=0.2, random_state=42, stratify=df_train["TARGET"])

# Потом из оставшегося выделим часть для таргет-энкодинга (например 25%)
train, enc = train_test_split(train_full, test_size=0.25, random_state=42, stratify=train_full["TARGET"])

print("Размеры наборов:")
print("train:", train.shape)
print("test:", test.shape)
print("enc:", enc.shape)

# Проверка
print("Mapping TARGET -> число:", target_mapping)
print("Mapping число -> TARGET:", reverse_mapping)



import json

# Целевая переменная
target = "TARGET"

# Категориальные признаки (кроме ID_клиента и ID_записи и TARGET)
cat_cols = enc.select_dtypes(include=["object"]).columns.tolist()
for col_to_remove in ["ID_клиента", "ID_записи", "TARGET"]:
    if col_to_remove in cat_cols:
        cat_cols.remove(col_to_remove)

# Словарь для хранения всех TE маппингов
target_encoders = {}
# Вычисляем глобальное среднее таргета (можно использовать числовое преобразование TARGET, если нужно)
# Для простоты примем, что TARGET уже числовой. Если нет, можно использовать LabelEncoder для TARGET перед TE.
global_mean = enc[target].apply(lambda x: 0 if x is None else 0).mean()  # placeholder, заменим на числовой TARGET при необходимости

# Обучаем таргет-энкодер на выборке enc
for col in cat_cols:
    # Среднее TARGET по каждой категории
    target_mean = enc.groupby(col)[target].mean().to_dict()  # если TARGET числовой
    # Добавляем константу для неизвестных категорий
    target_mean["__UNKNOWN__"] = global_mean
    # Сохраняем маппинг в словарь
    target_encoders[col] = target_mean

# Сохраняем таргет-энкодер в JSON
with open("target_encoder_cols.json", "w", encoding="utf-8") as f:
    json.dump(target_encoders, f, ensure_ascii=False, indent=4)

print("Таргет-энкодер для категориальных признаков сохранён в target_encoder_cols.json с константой для неизвестных категорий")



import json

# Загружаем таргет-энкодер из JSON
with open("target_encoder_cols.json", "r", encoding="utf-8") as f:
    target_encoders = json.load(f)

# Применяем TE к train
for col, mapping in target_encoders.items():
    # map заменяет категории на их значения TE
    # если категория неизвестна, используем __UNKNOWN__
    train[f"{col}_TE"] = train[col].map(lambda x: mapping.get(x, mapping["__UNKNOWN__"]))

# Проверка
train[[f"{col}_TE" for col in target_encoders.keys()]].head()



import json

# Загружаем target-энкодеры
with open("target_encoder_cols.json", "r", encoding="utf-8") as f:
    target_encoders = json.load(f)

# Применяем TE к df_test
for col, mapping in target_encoders.items():
    df_test[f"{col}_TE"] = df_test[col].map(lambda x: mapping.get(x, mapping["__UNKNOWN__"]))

# Проверка
df_test[[f"{col}_TE" for col in target_encoders.keys()]].head()



num_cols = train.select_dtypes(include=["number"]).columns.tolist()
sorted(num_cols)





import seaborn as sns
import matplotlib.pyplot as plt

# Выбираем только числовые столбцы
num_cols = train.select_dtypes(include=["number"]).columns.tolist()

# Корреляционная матрица
corr_matrix = train[num_cols].corr()

# Выводим матрицу
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Корреляционная матрица числовых признаков (train)")
plt.show()



import json

# Загружаем таргет-энкодер из JSON
with open("target_encoder_cols.json", "r", encoding="utf-8") as f:
    target_encoders = json.load(f)

# Применяем TE к test
for col, mapping in target_encoders.items():
    # map заменяет категории на их значения TE
    # если категория неизвестна, используем __UNKNOWN__
    test[f"{col}_TE"] = test[col].map(lambda x: mapping.get(x, mapping["__UNKNOWN__"]))

# Проверка
test[[f"{col}_TE" for col in target_encoders.keys()]].head()



# Оставляем только числовые колонки
numeric_cols = train.select_dtypes(include=["int64", "float64"]).columns.tolist()

# В train оставляем числовые + целевую переменную
train = train[numeric_cols]

# В test оставляем только числовые колонки, которые есть в train
test = test[[c for c in numeric_cols if c in test.columns]]



train





import lightgbm as lgb
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import joblib
import pandas as pd
import numpy as np

# Целевая переменная
target = "TARGET"
features = [c for c in train.columns if c != target]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

# Objective для Optuna
def objective(trial):
    param = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "subsample_freq": trial.suggest_int("bagging_freq", 1, 10),
        "colsample_bytree": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("lambda_l1", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("lambda_l2", 0.0, 5.0),
        "objective": "multiclass",
        "num_class": len(y_train.unique()),
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []

    for train_idx, valid_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        model = lgb.LGBMClassifier(**param, n_estimators=500)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        f1_scores.append(f1_score(y_val, preds, average='macro'))

    return np.mean(f1_scores)

# Запуск Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("Лучшие параметры:", study.best_params)
print("Лучший F1-score (macro):", study.best_value)

# Финальная модель
best_params = study.best_params
best_params.update({
    "objective": "multiclass",
    "num_class": len(y_train.unique()),
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1
})

final_model = lgb.LGBMClassifier(**best_params, n_estimators=500)
final_model.fit(X_train, y_train)

# Сохраняем модель
joblib.dump(final_model, "lgbm_model_optuna.pkl")

# Предсказания на тесте
y_pred = final_model.predict(X_test)
f1 = f1_score(y_test, y_pred, average="macro")
print("F1-macro на тесте:", f1)



# Предсказания на тесте
y_pred = final_model.predict(X_test)
f1 = f1_score(y_test, y_pred, average="macro")
print("F1-macro на тесте:", f1)



df_test


df_test[["ID_записи"] + final_model.booster_.feature_name()]





df_sample




