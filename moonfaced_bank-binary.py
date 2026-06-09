!pip install \
  autogluon.common==1.3.1 \
  autogluon.core==1.3.1 \
  autogluon.features==1.3.1 \
  autogluon.tabular==1.3.1 \
  --no-deps



!pip install scikit-learn==1.5.2


!pip install ray==2.40.0


import pandas as pd
from autogluon.tabular import TabularPredictor
import numpy as np
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer # Для обработки NaN перед генерацией признаков


# --- 1. Определение путей к датасетам ---
# Пути к данным соревнования. Измените, если имена файлов отличаются.
TRAIN_PATH = '/kaggle/input/playground-series-s5e8/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s5e8/test.csv'
SAMPLE_SUBMISSION_PATH = '/kaggle/input/playground-series-s5e8/sample_submission.csv'

ORIGINAL_DATASET_PATH = '/kaggle/input/bank-marketing-dataset-full/bank-full.csv'

# --- 1. Определение путей к датасетам ---
# Пути к данным соревнования
TRAIN_PATH = '/kaggle/input/playground-series-s5e8/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s5e8/test.csv'
SAMPLE_SUBMISSION_PATH = '/kaggle/input/playground-series-s5e8/sample_submission.csv'

# Путь к оригинальному датасету
ORIGINAL_DATASET_PATH = '/kaggle/input/bank-marketing-dataset-full/bank-full.csv'


# --- 2. Загрузка данных ---
print("Загрузка данных...")
# Предполагаем, что все CSV используют ';' как разделитель, как это часто бывает с этими датасетами
train_df = pd.read_csv(TRAIN_PATH, delimiter=';')
test_df = pd.read_csv(TEST_PATH, delimiter=';')
sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
original_df = pd.read_csv(ORIGINAL_DATASET_PATH, delimiter=';')
print("Данные успешно загружены.")


# --- 3. Предварительная обработка данных ---
print("Начало предварительной обработки данных...")

# Имя колонки, которую нужно предсказать
label = 'y'

# Сохраняем id тестового набора для submission файла.
# Всегда берем ID из sample_submission_df, так как test_df может не содержать колонку 'id'.
test_ids = sample_submission_df['id']

# Удаляем колонку 'id' из тренировочных и тестовых данных, если она есть,
# так как она не является признаком для обучения.
if 'id' in train_df.columns:
    train_df = train_df.drop('id', axis=1)
# Важно: test_df может не иметь 'id', поэтому проверяем перед удалением
if 'id' in test_df.columns:
    test_df = test_df.drop('id', axis=1)
if 'id' in original_df.columns:
    original_df = original_df.drop('id', axis=1)


# Объединяем тренировочный датасет соревнования с оригинальным датасетом.
# Шаг 1: Находим все уникальные колонки (признаки + целевая) между train_df и original_df.
all_unique_cols = list(set(train_df.columns) | set(original_df.columns))

# Шаг 2: Выравниваем train_df и original_df по этим колонкам, заполняя отсутствующие NaN.
# Это гарантирует, что оба DataFrame имеют одинаковый набор колонок перед конкатенацией.
train_df = train_df.reindex(columns=all_unique_cols, fill_value=np.nan)
original_df = original_df.reindex(columns=all_unique_cols, fill_value=np.nan)

# Шаг 3: Объединяем DataFrame.
combined_train_df = pd.concat([train_df, original_df], ignore_index=True)

# --- НОВОЕ ИСПРАВЛЕНИЕ: Удаляем строки с NaN в целевой колонке 'y' ---
# Это критично для sklearn.model_selection.train_test_split, который не может работать с NaN в y.
initial_rows = combined_train_df.shape[0]
combined_train_df.dropna(subset=[label], inplace=True)
rows_after_drop = combined_train_df.shape[0]
if initial_rows > rows_after_drop:
    print(f"Удалено {initial_rows - rows_after_drop} строк из-за NaN в целевой колонке '{label}'.")


# Выравнивание колонок тестового набора относительно тренировочного.
# Шаг 1: Определяем все колонки-признаки, которые есть в объединенном тренировочном наборе.
train_features_cols = [col for col in combined_train_df.columns if col != label]

# Шаг 2: Выравниваем test_df по этим признакам, заполняя отсутствующие NaN.
test_df = test_df.reindex(columns=train_features_cols, fill_value=np.nan)


# Обработка "неразумного шума" и преобразование типов данных
# Заменяем 'unknown' на NaN и приводим к нижнему регистру для категориальных признаков.
# Теперь более надежно проверяем, является ли колонка строковой, прежде чем применять .str accessor.
for col in combined_train_df.columns:
    # Проверяем, что колонка имеет тип 'object' (строковый)
    # и что она не является целевой колонкой
    if combined_train_df[col].dtype == 'object' and col != label:
        # Дополнительная проверка, что колонка действительно содержит строковые значения,
        # а не, например, числа, которые были прочитаны как object из-за NaN.
        if combined_train_df[col].apply(lambda x: isinstance(x, str)).any():
            combined_train_df[col] = combined_train_df[col].str.lower()
            combined_train_df[col] = combined_train_df[col].replace('unknown', pd.NA)
            
            # Применяем то же к тестовому набору
            if col in test_df.columns:
                if test_df[col].apply(lambda x: isinstance(x, str)).any():
                    test_df[col] = test_df[col].str.lower()
                    test_df[col] = test_df[col].replace('unknown', pd.NA)

print(f"Размер объединенного тренировочного датасета после предобработки: {combined_train_df.shape}")
print(f"Размер тестового датасета после предобработки: {test_df.shape}")
print("Предварительная обработка данных завершена.")


# --- 4. Обучение модели с использованием AutoGluon ---
print("Начало обучения модели AutoGluon...")

# Каталог для сохранения моделей AutoGluon
save_path = 'AutogluonModels_BankDeposit_Prob'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Инициализация TabularPredictor
# eval_metric='roc_auc' - хорошая метрика для предсказания вероятностей
predictor = TabularPredictor(
    label=label,
    eval_metric='roc_auc',
    path=save_path
)

# Определяем финальные признаки для обучения и предсказания
# Это все колонки из тренировочного набора, кроме целевой
train_features_final = [col for col in combined_train_df.columns if col != label]

# Устанавливаем лимит времени на обучение.
# Для начала, 1 час - это хороший выбор. Можно увеличить до 2-4 часов, если позволяет время.
TIME_LIMIT_SECONDS = 3600 * 8

print(f"Запуск обучения AutoGluon. Лимит времени: {TIME_LIMIT_SECONDS / 3600} часа.")

predictor.fit(
    combined_train_df[train_features_final + [label]], # Обучаем на всех признаках из train_features_final
    presets='best_quality', # 'best_quality' - отличная отправная точка для Kaggle
    time_limit=TIME_LIMIT_SECONDS
)

print("Обучение модели AutoGluon завершено.")


# --- 5. Генерация предсказаний для тестового набора ---
print("Генерация предсказаний для тестового набора (вероятности)...")

# Используем predict_proba() для получения вероятностей
# Для бинарной классификации predict_proba() возвращает 2 колонки: вероятность класса 0 и класса 1.
# Нам нужна вероятность положительного класса ('yes').
# AutoGluon автоматически определяет, какой класс является положительным.
# Обычно это класс, который идет в алфавитном порядке последним (т.е. 'yes' после 'no').
# Или можно явно указать positive_class при инициализации Predictor.
predictions_proba = predictor.predict_proba(test_df[train_features_final]) # Используем те же признаки, что и для обучения

# Если 'yes' - это положительный класс, то берем вторую колонку (индекс 1)
# Убедимся в этом, проверив mapping: predictor.class_labels
# Например, если predictor.class_labels = ['no', 'yes'], то 'yes' - это индекс 1
positive_class_index = predictor.class_labels.index('yes') if 'yes' in predictor.class_labels else 1
final_probabilities = predictions_proba.iloc[:, positive_class_index]


print("Вероятности предсказаний сгенерированы.")


# --- 6. Формирование файла Submission ---
print("Формирование файла submission.csv...")

submission_df = pd.DataFrame({'id': test_ids, label: final_probabilities})

# Сохранение файла submission.csv
submission_df.to_csv('submission.csv', index=False)

print("Файл submission.csv успешно создан.")
print(f"Первые 5 строк файла submission.csv:\n{submission_df.head()}")
print("Пайплайн завершен.")


# # --- 1. Определение путей и загрузка данных ---
# # Эти пути должны совпадать с теми, которые вы использовали для обучения AutoGluon.
# # Важно: для загрузки модели вам не нужны тренировочные данные, но для предсказания
# # на тестовом наборе нужен сам тестовый набор.
# TEST_PATH = '/kaggle/input/playground-series-s5e8/test.csv'
# SAMPLE_SUBMISSION_PATH = '/kaggle/input/playground-series-s5e8/sample_submission.csv'

# # Путь, куда AutoGluon сохранил свои модели
# # Убедитесь, что это тот же 'save_path', который вы использовали при обучении.
# # В вашем случае это 'AutogluonModels_BankDeposit_Prob'
# MODEL_SAVE_PATH = 'AutogluonModels_BankDeposit_Prob'

# print("Загрузка тестовых данных и Sample Submission...")
# try:
#     test_df = pd.read_csv(TEST_PATH, delimiter=';')
#     sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
#     print("Тестовые данные и Sample Submission успешно загружены.")
# except FileNotFoundError as e:
#     print(f"Ошибка: Не найдены тестовые данные или Sample Submission. Проверьте пути. {e}")
#     exit()

# # Сохраняем id тестового набора для submission файла
# test_ids = sample_submission_df['id']

# # Удаляем колонку 'id' из тестового DataFrame, если она есть
# if 'id' in test_df.columns:
#     test_df = test_df.drop('id', axis=1)

# # --- 2. Загрузка основного объекта Predictor ---
# # Это загружает весь обученный Predictor, включая все его модели и метаданные.
# print(f"\nЗагрузка обученного Predictor из '{MODEL_SAVE_PATH}'...")
# try:
#     predictor = TabularPredictor.load(MODEL_SAVE_PATH)
#     print("Predictor успешно загружен.")
# except Exception as e:
#     print(f"Ошибка при загрузке Predictor. Убедитесь, что путь '{MODEL_SAVE_PATH}' корректен и содержит обученную модель. {e}")
#     exit()

# # --- 3. Просмотр обученных моделей и их производительности ---
# # Метод leaderboard() показывает производительность каждой отдельной модели и ансамблей.
# print("\n--- Таблица лидеров обученных моделей ---")
# leaderboard = predictor.leaderboard(silent=True) # silent=True для более чистого вывода
# print(leaderboard)

# # --- 4. Выбор и загрузка конкретной модели ---
# # Вы можете выбрать модель по имени (из колонки 'model' в leaderboard)
# # Например, 'LightGBM_BAG_L1' или 'XGBoost_BAG_L2'
# # Для примера, давайте попробуем загрузить лучшую базовую модель (не ансамбль)
# # из таблицы лидеров.
# best_base_model_name = leaderboard[~leaderboard['model'].str.contains('WeightedEnsemble')]['model'].iloc[0]

# print(f"\nВыбрана лучшая базовая модель: '{best_base_model_name}'")

# # Загружаем выбранную модель
# print(f"Загрузка модели '{best_base_model_name}'...")
# try:
#     individual_model = predictor.load_model(best_base_model_name)
#     print(f"Модель '{best_base_model_name}' успешно загружена.")
# except Exception as e:
#     print(f"Ошибка при загрузке индивидуальной модели. Убедитесь, что имя модели '{best_base_model_name}' корректно. {e}")
#     exit()

# # --- 5. Подготовка тестовых данных для предсказания отдельной моделью ---
# # Важно: тестовые данные должны иметь те же признаки и тот же порядок,
# # что и данные, на которых обучалась модель.
# # Predictor хранит информацию о признаках, поэтому мы можем использовать его.
# train_features_cols = predictor.features() # Получаем список признаков, которые использовал Predictor

# # Выравниваем тестовый DataFrame по этим признакам
# for col in train_features_cols:
#     if col not in test_df.columns:
#         test_df[col] = np.nan # Добавляем отсутствующие признаки с NaN

# test_df_aligned = test_df[train_features_cols] # Выбираем только нужные признаки и их порядок

# # --- 6. Генерация предсказаний с помощью отдельной модели ---
# print(f"\nГенерация предсказаний с помощью модели '{best_base_model_name}'...")

# # predict_proba() для получения вероятностей
# predictions_proba_individual = individual_model.predict_proba(test_df_aligned)

# # Нам нужна вероятность положительного класса ('yes').
# # Predictor хранит информацию о mapping классов.
# label = predictor.label # Получаем имя целевой колонки
# positive_class_index = predictor.class_labels.index('yes') if 'yes' in predictor.class_labels else 1
# final_probabilities_individual = predictions_proba_individual.iloc[:, positive_class_index]

# print("Предсказания сгенерированы отдельной моделью.")


# # --- 7. Формирование файла Submission для отдельной модели ---
# print("Формирование файла submission_individual_model.csv...")

# submission_df_individual = pd.DataFrame({'id': test_ids, label: final_probabilities_individual})

# # Сохраняем файл submission.csv
# submission_df_individual.to_csv('submission_individual_model.csv', index=False)

# print("Файл submission_individual_model.csv успешно создан.")
# print(f"Первые 5 строк файла submission_individual_model.csv:\n{submission_df_individual.head()}")
# print("\nПроцесс завершен.")


