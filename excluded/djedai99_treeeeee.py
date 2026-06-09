import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
train_data = pd.read_csv("/kaggle/input/forest-cover-type-prediction/train.csv")
test_data = pd.read_csv("/kaggle/input/forest-cover-type-prediction/test.csv")

print("Размер обучающей выборки:", train_data.shape)
print("Размер тестовой выборки:", test_data.shape)

# Первичный осмотр данных
print("\nПервые 5 строк обучающей выборки:")
print(train_data.head())

# Распределение целевой переменной
print("\nРаспределение классов целевой переменной:")
print(train_data['Cover_Type'].value_counts())

# Сохранение идентификаторов для финальной отправки
test_ids = test_data['Id'].copy()

# Подготовка признаков и целевой переменной
target = train_data['Cover_Type']
features = train_data.drop('Cover_Type', axis=1)

# Проверка на пропущенные значения
print("\nПропущенные значения в обучающей выборке:", features.isna().sum().sum())
print("Пропущенные значения в тестовой выборке:", test_data.isna().sum().sum())

def prepare_features(dataframe):
    """Функция для создания и преобразования признаков"""
    df = dataframe.copy()
    
    # Создание новых признаков на основе тени
    df['Hillshade_Total'] = df['Hillshade_9am'] + df['Hillshade_Noon'] + df['Hillshade_3pm']
    df['Hillshade_Mean'] = df['Hillshade_Total'] / 3.0
    
    # Удаление исходных признаков тени
    shade_columns = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Hillshade_Total']
    df = df.drop(shade_columns, axis=1)
    
    # Удаление идентификатора
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    
    return df

# Применение преобразований к данным
train_features = prepare_features(features)
test_features = prepare_features(test_data)

print("\nРазмеры после преобразований:")
print("Обучающие признаки:", train_features.shape)
print("Тестовые признаки:", test_features.shape)

# Анализ корреляций
print("\nАнализ корреляций между признаками:")
correlation_matrix = train_features.corr()

# Стилизация таблицы корреляций
def enhance_correlation_display(corr_data):
    """Улучшенное отображение матрицы корреляций"""
    cmap_colors = sns.diverging_palette(5, 250, as_cmap=True)
    
    formatting_rules = [
        dict(selector="th", props=[("font-size", "8pt")]),
        dict(selector="td", props=[('padding', "0.1em 0.2em")]),
        dict(selector="th:hover", props=[("font-size", "10pt")]),
        dict(selector="tr:hover td:hover", 
             props=[('max-width', '150px'), ('font-size', '10pt')])
    ]
    
    return (corr_data.style.background_gradient(cmap_colors, axis=1)
            .format(precision=2)
            .set_properties(**{'max-width': '70px', 'font-size': '9pt'})
            .set_caption("Матрица корреляций признаков")
            .set_table_styles(formatting_rules))

# Отображение корреляций
correlation_display = enhance_correlation_display(correlation_matrix)
display(correlation_display)

# Преобразование целевой переменной
label_encoder = LabelEncoder()
encoded_target = label_encoder.fit_transform(target)

# Разделение данных для валидации
X_train, X_val, y_train, y_val = train_test_split(
    train_features, encoded_target, 
    test_size=0.2, 
    random_state=42, 
    stratify=encoded_target
)

print(f"\nРазмеры выборок:")
print(f"Обучающая: {X_train.shape}, Валидационная: {X_val.shape}")

# Определение моделей и их гиперпараметров
models_config = {
    'XGBoost': {
        'estimator': XGBClassifier(random_state=42, eval_metric="mlogloss", verbosity=0),
        'parameters': {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0]
        }
    },
    'RandomForest': {
        'estimator': RandomForestClassifier(random_state=42),
        'parameters': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'LogisticRegression': {
        'estimator': LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial'),
        'parameters': {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'sag']
        }
    },
    'GradientBoosting': {
        'estimator': GradientBoostingClassifier(random_state=42),
        'parameters': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
    }
}



# Использование предопределенной модели 
final_model = RandomForestClassifier(
    max_depth=30,
    n_estimators=300,
    random_state=42
)

print("\nОбучение финальной модели...")
final_model.fit(train_features, encoded_target)

# Предсказание на валидационной выборке
val_predictions = final_model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)
print(f"Точность на валидационной выборке: {val_accuracy:.4f}")

# Предсказание на тестовых данных
print("\nГенерация предсказаний для тестовой выборки...")
test_predictions = final_model.predict(test_features)
test_predictions_decoded = label_encoder.inverse_transform(test_predictions)

# Создание файла для отправки
submission_result = pd.DataFrame({
    'Id': test_ids,
    'Cover_Type': test_predictions_decoded
})

submission_result.to_csv('submission.csv', index=False)
print("\nФайл 'submission.csv' успешно сохранен!")
print(f"Количество предсказаний: {len(submission_result)}")

# Анализ распределения предсказаний
print("\nРаспределение предсказанных классов:")
prediction_distribution = pd.Series(test_predictions_decoded).value_counts().sort_index()
print(prediction_distribution)

