import pandas as pd  
import numpy as np   
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


train_data = pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')  
test_data = pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')

# данные
print("Тренировочные данные размер:", train_data.shape)
print("Тестовые данные размер:", test_data.shape)
print("\nПервые 5 строк тренировочных данных:")
print(train_data.head())
print("\nРаспределение классов:")
print(train_data['Cover_Type'].value_counts().sort_index())

# Разделение на признаки и целевую переменную
X_train = train_data.drop(['Id', 'Cover_Type'], axis=1)  
y_train = train_data['Cover_Type']
X_test = test_data.drop('Id', axis=1)
test_ids = test_data['Id']

# Анализ признаков
print(f"\nКоличество признаков: {X_train.shape[1]}")
print(f"Типы признаков:\n{X_train.dtypes.value_counts()}")


numeric_features = ['Elevation', 'Aspect', 'Slope', 
                   'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
                   'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 
                   'Hillshade_Noon', 'Hillshade_3pm', 
                   'Horizontal_Distance_To_Fire_Points']

# Бинарные признаки (wilderness areas и soil types)
binary_features = [col for col in X_train.columns if col not in numeric_features]
print(f"\nЧисловых признаков: {len(numeric_features)}")
print(f"Бинарных признаков: {len(binary_features)}")

# Предобработка данных - попробуем разные подходы
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Масштабируем только числовые признаки, бинарные оставляем как есть
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

# Разделим данные для валидации
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
)


print("\n" + "="*50)
print("1. Обучение RandomForest с улучшенными параметрами")
print("="*50)

rf_model = RandomForestClassifier(
    n_estimators=300,           # Увеличим количество деревьев
    max_depth=25,               # Ограничим глубину для предотвращения переобучения
    min_samples_split=8,        # Увеличим минимальное количество образцов для разделения
    min_samples_leaf=3,         # Увеличим минимальное количество образцов в листе
    max_features='sqrt',        # Используем sqrt от общего количества признаков
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'     # Учитываем несбалансированность классов
)

# Кросс-валидация
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Кросс-валидация (5 фолдов): {cv_scores}")
print(f"Средняя точность: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Обучение и проверка на валидационной выборке
rf_model.fit(X_train_split, y_train_split)
val_predictions = rf_model.predict(X_val_split)
val_accuracy = accuracy_score(y_val_split, val_predictions)
print(f"Точность на валидационной выборке: {val_accuracy:.4f}")


print("\n" + "="*50)
print("2. Обучение GradientBoosting")
print("="*50)

gb_model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    subsample=0.8
)

# Быстрая проверка на части данных
gb_model.fit(X_train_split[:10000], y_train_split[:10000])
gb_val_predictions = gb_model.predict(X_val_split)
gb_accuracy = accuracy_score(y_val_split, gb_val_predictions)
print(f"Точность GradientBoosting на валидации: {gb_accuracy:.4f}")


print("\n" + "="*50)
print("3. Предсказания на тестовых данных")
print("="*50)

# Обучаем лучшую модель на всех тренировочных данных
print("Обучение финальной модели RandomForest на всех данных...")
rf_model.fit(X_train_scaled, y_train)

# Предсказание на тестовых данных
test_predictions = rf_model.predict(X_test_scaled)

# Анализ распределения предсказаний
prediction_counts = pd.Series(test_predictions).value_counts().sort_index()
print(f"\nРаспределение предсказанных классов:")
for class_type, count in prediction_counts.items():
    print(f"Класс {class_type}: {count} ({count/len(test_predictions)*100:.1f}%)")

# Сравним с распределением в тренировочных данных
train_counts = train_data['Cover_Type'].value_counts().sort_index()
print(f"\nРаспределение в тренировочных данных:")
for class_type, count in train_counts.items():
    print(f"Класс {class_type}: {count} ({count/len(train_data)*100:.1f}%)")

# Важность признаков
feature_importance = pd.DataFrame({
    'Признак': X_train.columns,
    'Важность': rf_model.feature_importances_
}).sort_values('Важность', ascending=False)

print("\n" + "="*50)
print("ТОП-15 ВАЖНЫХ ПРИЗНАКОВ")
print("="*50)
print(feature_importance.head(15).to_string(index=False))


print("\n" + "="*50)
print("ГРАФИК ВАЖНОСТИ ПРИЗНАКОВ (текстовый)")
print("="*50)

top_features = feature_importance.head(10)
max_importance = top_features['Важность'].max()

for _, row in top_features.iterrows():
    bar_length = int(row['Важность'] / max_importance * 50)
    print(f"{row['Признак']:40} | {'█' * bar_length} {row['Важность']:.4f}")

# Создание submission файла
submission = pd.DataFrame({
    'Id': test_ids,
    'Cover_Type': test_predictions
})

# Сохранение результатов
submission.to_csv('submission.csv', index=False)

# Проверка submission файла
print("\n" + "="*50)
print("ИНФОРМАЦИЯ О SUBMISSION ФАЙЛЕ")
print("="*50)
print(f"Имя файла: forest_cover_predictions.csv")
print(f"Количество предсказаний: {len(submission)}")
print(f"Уникальные предсказанные классы: {sorted(submission['Cover_Type'].unique())}")

# Примеры предсказаний
print("\n" + "="*50)
print("ПРИМЕРЫ ПРЕДСКАЗАНИЙ")
print("="*50)
print("Первые 10 строк submission файла:")
print(submission.head(10))

print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ И РЕКОМЕНДАЦИИ")
print("="*50)
print(f"""
Итоговая модель: RandomForestClassifier
Конфигурация:
- Количество деревьев: {rf_model.n_estimators}
- Максимальная глубина: {rf_model.max_depth}
- Использованные признаки: {rf_model.max_features}

Достигнутая точность:
- Кросс-валидация: {cv_scores.mean():.4f}
- Валидационная выборка: {val_accuracy:.4f}
""")

