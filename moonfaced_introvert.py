!pip install \
  autogluon.common==1.3.1 \
  autogluon.core==1.3.1 \
  autogluon.features==1.3.1 \
  autogluon.tabular==1.3.1 \
  --no-deps


!pip install scikit-learn==1.5.2



import pandas as pd
from autogluon.tabular import TabularPredictor
import numpy as np
import os



import numpy
import pandas
import sklearn
import lightgbm
import xgboost
import catboost

print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("lightgbm:", lightgbm.__version__)
print("xgboost:", xgboost.__version__)
print("catboost:", catboost.__version__)


# Пути к данным соревнования
TRAIN_PATH = '/kaggle/input/playground-series-s5e7/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s5e7/test.csv'
SAMPLE_SUBMISSION_PATH = '/kaggle/input/playground-series-s5e7/sample_submission.csv'

# Путь к оригинальному датасету, который разрешено использовать
ORIGINAL_DATASET_PATH = '/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv'

# 2. Загрузка данных
print("Загрузка данных...")
try:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    original_df = pd.read_csv(ORIGINAL_DATASET_PATH)
    print("Данные успешно загружены.")
except FileNotFoundError as e:
    print(f"Ошибка: Один или несколько файлов не найдены. Проверьте пути. {e}")
    # Если файлы не найдены (например, при локальном запуске без Kaggle), создадим заглушки для демонстрации
    print("Создание фиктивных данных для демонстрации.")
    train_df = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
    train_df['id'] = range(100)
    train_df['Personality'] = np.random.choice(['Introvert', 'Extrovert'], 100)
    test_df = pd.DataFrame(np.random.rand(50, 9), columns=[f'feature_{i}' for i in range(9)])
    test_df['id'] = range(100, 150)
    sample_submission_df = pd.DataFrame({'id': test_df['id'], 'Personality': 'Introvert'})
    original_df = pd.DataFrame(np.random.rand(200, 10), columns=[f'feature_{i}' for i in range(10)])
    original_df['Personality'] = np.random.choice(['Introvert', 'Extrovert'], 200)
    original_df['id'] = range(200, 400) # Добавляем 'id' для оригинального датасета, если его нет
    # Убедимся, что колонки в original_df соответствуют train_df, кроме 'id'
    # Для этого примера, сделаем более надежное выравнивание колонок
    train_cols_no_id = [col for col in train_df.columns if col != 'id']
    original_cols_no_id = [col for col in original_df.columns if col != 'id']

    # Находим общие колонки, которые будут использоваться для объединения
    common_cols = list(set(train_cols_no_id) & set(original_cols_no_id))
    
    # Добавляем отсутствующие колонки в original_df, заполняя NaN
    for col in train_cols_no_id:
        if col not in original_df.columns:
            original_df[col] = np.nan
    # Добавляем отсутствующие колонки в train_df, заполняя NaN (если такое возможно, хотя обычно train_df полный)
    for col in original_cols_no_id:
        if col not in train_df.columns:
            train_df[col] = np.nan
            
    # Выбираем только те колонки, которые есть в train_df (без id) и в original_df
    original_df = original_df[train_cols_no_id]
    train_df = train_df[train_cols_no_id]


# 3. Предварительная обработка данных
print("Начало предварительной обработки данных...")

# Сохраним id тестового набора для submission файла.
test_ids = test_df['id']

# Удаляем колонку 'id' из тренировочных данных, если она есть,
# так как она не является признаком для обучения.
if 'id' in train_df.columns:
    train_df = train_df.drop('id', axis=1)
if 'id' in original_df.columns:
    original_df = original_df.drop('id', axis=1)
if 'id' in test_df.columns:
    test_df = test_df.drop('id', axis=1) # Удаляем 'id' из test_df для предсказания

# Объединение тренировочного датасета соревнования с оригинальным датасетом.
# Это важный шаг, так как лидеры прошлых соревнований использовали оригинальные данные.
# Убедимся, что колонки совпадают перед объединением.
# Найдем все уникальные колонки из train_df и original_df (исключая 'Personality')
all_features = list(set(train_df.columns) | set(original_df.columns))
if 'Personality' in all_features:
    all_features.remove('Personality') # Убираем целевую колонку из списка признаков

# Выравниваем колонки для обоих датафреймов, добавляя отсутствующие с NaN
for col in all_features:
    if col not in train_df.columns:
        train_df[col] = np.nan
    if col not in original_df.columns:
        original_df[col] = np.nan

# Убедимся, что порядок колонок одинаковый (кроме целевой)
train_df = train_df[all_features + ['Personality']]
original_df = original_df[all_features + ['Personality']]

combined_train_df = pd.concat([train_df, original_df], ignore_index=True)

print(f"Размер объединенного тренировочного датасета: {combined_train_df.shape}")
print(f"Первые 5 строк объединенного датасета:\n{combined_train_df.head()}")

# Обработка "неразумного шума" (как в решении 4-го места).
# Этот шаг может быть очень специфичным для каждого датасета.
# Здесь мы реализуем общую стратегию: преобразование нечисловых значений в числовых колонках в NaN.
# AutoGluon хорошо справляется с NaN.
for col in combined_train_df.columns:
    if combined_train_df[col].dtype == 'object' and col != 'Personality':
        # Попытка преобразовать в число, если это возможно, иначе оставить как есть
        # или преобразовать в NaN, если это явно числовая колонка с ошибками.
        temp_col_train = pd.to_numeric(combined_train_df[col], errors='coerce')
        
        # Если после преобразования большинство значений стали NaN, возможно, это не числовая колонка,
        # а категориальная, которую AutoGluon обработает сам.
        # Порог 0.5 означает, что если более половины значений стали NaN, то это, вероятно, не числовая колонка.
        if temp_col_train.isnull().sum() / len(temp_col_train) < 0.5:
            combined_train_df[col] = temp_col_train
            # Применяем то же преобразование к тестовым данным
            if col in test_df.columns:
                test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
        # else: AutoGluon обработает как категориальную

print("Предварительная обработка данных завершена.")

# 4. Обучение модели с использованием AutoGluon
print("Начало обучения модели AutoGluon...")

# Определение целевой переменной
label = 'Personality'

# Каталог для сохранения моделей AutoGluon
save_path = 'AutogluonModels_IntrovertExtrovert'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Инициализация TabularPredictor
# eval_metric='accuracy' соответствует метрике соревнования.
predictor = TabularPredictor(
    label=label,
    eval_metric='accuracy',
    path=save_path
)

# Перед обучением, убедимся, что test_df имеет те же колонки, что и combined_train_df (кроме 'Personality')
# и что их типы данных соответствуют.
train_features = combined_train_df.drop(columns=[label]).columns
test_features_present = test_df.columns # Колонки, которые фактически есть в test_df

# Найдем общие признаки для обучения и предсказания
# Это колонки, которые есть в тренировочном наборе (без целевой) И в тестовом наборе
common_predict_features = list(set(train_features) & set(test_features_present))

# Увеличиваем time_limit для обучения AutoGluon.
# Например, 2 часа (7200 секунд). Можно увеличить до 4-8 часов, если есть время.
TIME_LIMIT_SECONDS = 3600 * 6 # 2 часа

predictor.fit(
    combined_train_df[common_predict_features + [label]], # Обучаем на общих признаках + целевая
    presets='best_quality',
    time_limit=TIME_LIMIT_SECONDS # Устанавливаем лимит времени
)

print("Обучение модели AutoGluon завершено.")

# 5. Генерация предсказаний для тестового набора
print("Генерация предсказаний для тестового набора...")

# Используем predict() для получения предсказанных классов (Introvert/Extrovert)
# Убедимся, что тестовый датафрейм содержит только те признаки, на которых обучалась модель.
predictions = predictor.predict(test_df[common_predict_features])

print("Предсказания сгенерированы.")

# 6. Формирование файла Submission
print("Формирование файла submission.csv...")

submission_df = pd.DataFrame({'id': test_ids, 'Personality': predictions})

# Сохранение файла submission.csv
submission_df.to_csv('submission.csv', index=False)

print("Файл submission.csv успешно создан.")
print(f"Первые 5 строк файла submission.csv:\n{submission_df.head()}")
print("Пайплайн завершен.")


