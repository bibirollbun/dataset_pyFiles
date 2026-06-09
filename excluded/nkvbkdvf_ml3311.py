import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')


train_data = pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')
test_data = pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')

print(f"Размер тренировочных данных: {train_data.shape}")
print(f"Размер тестовых данных: {test_data.shape}")

cover_dist = train_data['Cover_Type'].value_counts().sort_index()
print("Распределение целевой переменной:")
print(cover_dist)


X = train_data.drop('Cover_Type', axis=1)
y = train_data['Cover_Type']

if 'Id' in X.columns:
    train_ids = X['Id']
    X = X.drop('Id', axis=1)

test_ids = test_data['Id'] if 'Id' in test_data.columns else test_data.index
X_test = test_data.drop('Id', axis=1) if 'Id' in test_data.columns else test_data.copy()

print(f"Исходные признаки: {X.shape[1]}")


def create_features(df):
    df = df.copy()

    df['Distance_To_Hydrology'] = np.sqrt(
        df['Horizontal_Distance_To_Hydrology']**2 +
        df['Vertical_Distance_To_Hydrology']**2
    )

    df['Hillshade_Mean'] = (df['Hillshade_9am'] + df['Hillshade_Noon'] + df['Hillshade_3pm']) / 3

    df['Hillshade_Range'] = df[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].max(axis=1) - \
                           df[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].min(axis=1)

    df['Fire_Road_Ratio'] = df['Horizontal_Distance_To_Fire_Points'] / (df['Horizontal_Distance_To_Roadways'] + 1)
    df['Hydrology_Road_Ratio'] = df['Horizontal_Distance_To_Hydrology'] / (df['Horizontal_Distance_To_Roadways'] + 1)

    df['Elevation_Above_Hydrology'] = df['Elevation'] - df['Vertical_Distance_To_Hydrology']

    df['Elevation_Slope'] = df['Elevation'] * df['Slope']

    df['Elevation_Squared'] = df['Elevation'] ** 2
    df['Slope_Squared'] = df['Slope'] ** 2

    df['Log_Distance_To_Hydrology'] = np.log1p(df['Horizontal_Distance_To_Hydrology'])
    df['Log_Distance_To_Roadways'] = np.log1p(df['Horizontal_Distance_To_Roadways'])
    df['Log_Distance_To_Fire_Points'] = np.log1p(df['Horizontal_Distance_To_Fire_Points'])

    df['Aspect_North'] = ((df['Aspect'] >= 315) | (df['Aspect'] <= 45)).astype(int)
    df['Aspect_South'] = ((df['Aspect'] >= 135) & (df['Aspect'] <= 225)).astype(int)

    wilderness_cols = [col for col in df.columns if 'Wilderness_Area' in col]
    soil_cols = [col for col in df.columns if 'Soil_Type' in col]

    df['Dominant_Wilderness'] = df[wilderness_cols].idxmax(axis=1)

    df['Dominant_Soil'] = df[soil_cols].idxmax(axis=1)

    df['Wilderness_Count'] = df[wilderness_cols].sum(axis=1)
    df['Soil_Count'] = df[soil_cols].sum(axis=1)

    return df

print("Создание новых признаков...")
X_extended = create_features(X)
X_test_extended = create_features(X_test)

print(f"Признаки после feature engineering: {X_extended.shape[1]}")
print(f"Новых признаков создано: {X_extended.shape[1] - X.shape[1]}")


categorical_cols = ['Dominant_Wilderness', 'Dominant_Soil']
X_final = pd.get_dummies(X_extended, columns=categorical_cols, drop_first=True)
X_test_final = pd.get_dummies(X_test_extended, columns=categorical_cols, drop_first=True)

common_cols = X_final.columns.intersection(X_test_final.columns)
X_final = X_final[common_cols]
X_test_final = X_test_final[common_cols]

print(f"Финальное количество признаков: {X_final.shape[1]}")


X_train, X_val, y_train, y_val = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Обучающая выборка: {X_train.shape}")
print(f"Валидационная выборка: {X_val.shape}")


et_model_basic = ExtraTreesClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("Обучение базовой модели с новыми признаками...")
start_time = time.time()
et_model_basic.fit(X_train, y_train)
training_time_basic = time.time() - start_time

y_pred_basic = et_model_basic.predict(X_val)
accuracy_basic = accuracy_score(y_val, y_pred_basic)

print(f"Точность базовой модели: {accuracy_basic:.4f}")
print(f"Время обучения: {training_time_basic:.2f} сек")


param_grid_fast = {
    'n_estimators': [100, 150],
    'max_depth': [None, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt']
}

et_model_fast = ExtraTreesClassifier(random_state=42, n_jobs=-1)

grid_search_fast = GridSearchCV(
    et_model_fast,
    param_grid_fast,
    cv=2,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("Настройка гиперпараметров...")
start_time = time.time()
grid_search_fast.fit(X_train, y_train)
grid_search_time = time.time() - start_time

print(f"Лучшие параметры: {grid_search_fast.best_params_}")

best_et_model = grid_search_fast.best_estimator_
y_pred_tuned = best_et_model.predict(X_val)
accuracy_tuned = accuracy_score(y_val, y_pred_tuned)

print(f"Точность настроенной модели: {accuracy_tuned:.4f}")
print(f"Улучшение точности: {accuracy_tuned - accuracy_basic:+.4f}")


feature_importance = pd.DataFrame({
    'feature': X_final.columns,
    'importance': best_et_model.feature_importances_
}).sort_values('importance', ascending=False)

print("15 важных признаков:")
print(feature_importance.head(15))

plt.figure(figsize=(12, 10))
top_features = feature_importance.head(15)

bars = plt.barh(top_features['feature'], top_features['importance'], color='lightgreen')
plt.xlabel('Важность признака')
plt.title('15 важных признаков (с новыми признаками)')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)

for bar, importance in zip(bars, top_features['importance']):
    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f'{importance:.4f}', ha='left', va='center')

plt.tight_layout()
plt.show()


new_features_in_top = []
for feature in top_features['feature']:
    if any(keyword in feature for keyword in ['Distance_To_Hydrology', 'Hillshade_Mean', 'Hillshade_Range',
                                            'Fire_Road_Ratio', 'Elevation_Above_Hydrology', 'Elevation_Slope',
                                            'Log_', 'Aspect_', 'Dominant_']):
        new_features_in_top.append(feature)

print(f"Новые признаки в топ-15: {len(new_features_in_top)}")
for feature in new_features_in_top:
    importance = top_features[top_features['feature'] == feature]['importance'].values[0]
    print(f"  {feature}: {importance:.4f}")


cm_tuned = confusion_matrix(y_val, y_pred_tuned)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_tuned, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(1, 8), yticklabels=range(1, 8))
plt.title(f'Матрица ошибок - Точность: {accuracy_tuned:.4f}')
plt.ylabel('Истинные значения')
plt.xlabel('Предсказанные значения')
plt.show()

print(classification_report(y_val, y_pred_tuned, digits=4))


print("Создание предсказаний для тестовых данных...")
start_time = time.time()
test_predictions = best_et_model.predict(X_test_final)
prediction_time = time.time() - start_time

pred_distribution = pd.Series(test_predictions).value_counts().sort_index()

print("Распределение предсказаний:")
for cover_type, count in pred_distribution.items():
    percentage = count / len(test_predictions) * 100
    print(f"Тип {cover_type}: {count:6d} ({percentage:5.1f}%)")


submission = pd.DataFrame({
    'Id': test_ids,
    'Cover_Type': test_predictions
})

filename = 'submissions.csv'
submission.to_csv(filename, index=False)

print(f"Файл для отправки создан: {filename}")
print(f"Количество строк в файле: {len(submission)}")
print("Первые 5 строк файла:")
print(submission.head())

