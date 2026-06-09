pip install pymatgen


# 1. Импорт библиотек
import pandas as pd
import numpy as np
import h2o  # <--- Новая библиотека
from h2o.automl import H2OAutoML
import re
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tqdm.notebook import tqdm # Для красивого progress bar
import optuna
from pymatgen.core import Composition
import matplotlib.pyplot as plt
import seaborn as sns


# 2. Загрузка и подготовка данных (как в нашем лучшем решении)
train_prop = pd.read_csv('/kaggle/input/critical-temperature-of-superconductors/train.csv')
formula_train = pd.read_csv('/kaggle/input/critical-temperature-of-superconductors/formula_train.csv')
test_prop = pd.read_csv('/kaggle/input/critical-temperature-of-superconductors/test.csv')
formula_test = pd.read_csv('/kaggle/input/critical-temperature-of-superconductors/formula_test.csv')


train_df = pd.concat([train_prop, formula_train], axis=1)
train_df = train_df.loc[:,~train_df.columns.duplicated()]
test_df = pd.concat([test_prop, formula_test], axis=1)
test_df = test_df.loc[:,~test_df.columns.duplicated()]

import re
def count_elements(formula):
    elements = re.findall(r'[A-Z][a-z]*', formula)
    return len(elements)
train_df['number_of_elements'] = train_df['material'].apply(count_elements)
test_df['number_of_elements'] = test_df['material'].apply(count_elements)

# 3. Инициализация H2O
# Запускаем H2O кластер. nthreads = -1 означает использовать все доступные ядра CPU.
# max_mem_size определяет, сколько памяти может использовать H2O.
h2o.init(nthreads=-1, max_mem_size='8g')

# 4. Конвертация данных в H2OFrame
print("\nКонвертируем данные в H2OFrame...")
# H2O требует, чтобы имена столбцов не содержали спецсимволов, но у нас с этим все в порядке.
train_h2o = h2o.H2OFrame(train_df)
test_h2o = h2o.H2OFrame(test_df)

# Определяем признаки (x) и целевую переменную (y)
features = [col for col in train_df.columns if col not in ['critical_temp', 'material']]
target = 'critical_temp'

# 5. Настройка и запуск H2O AutoML
print("\nЗапускаем H2O AutoML...")
aml = H2OAutoML(
    max_runtime_secs=30000,  # <--- ВРЕМЯ РАБОТЫ В СЕКУНДАХ. 3600 = 1 час. Чем дольше, тем лучше.
    nfolds=10,              # Количество фолдов для кросс-валидации
    sort_metric='RMSE',     # Метрика для ранжирования моделей (эквивалентно MSE)
    seed=42,                # Для воспроизводимости
    project_name="superconductors_automl"
)

# Запускаем обучение
aml.train(x=features, y=target, training_frame=train_h2o)

# 6. Просмотр результатов
print("\nAutoML завершен. Лидерборд моделей:")
lb = aml.leaderboard
# Выводим весь лидерборд
print(lb.head(rows=lb.nrows))

# 7. Предсказание с помощью лучшей модели
print("\nДелаем предсказания с помощью лучшей модели...")
best_model = aml.leader
preds = best_model.predict(test_h2o)

# 8. Создание файла для отправки
# Конвертируем предсказания обратно в pandas DataFrame
preds_df = preds.as_data_frame()

submission_df = pd.DataFrame({
    'index': test_df.index,
    'critical_temp': preds_df['predict'].values
})

# Отрицательные предсказания не имеют физического смысла
submission_df.loc[submission_df['critical_temp'] < 0, 'critical_temp'] = 0
submission_df.to_csv('submission_h2o_automl.csv', index=False)

print("\nФайл 'submission_h2o_automl.csv' успешно создан.")
print(submission_df.head())

# 9. (Опционально) Остановка кластера H2O
# h2o.shutdown()

