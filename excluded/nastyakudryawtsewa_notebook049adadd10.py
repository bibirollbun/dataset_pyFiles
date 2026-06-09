!pip install catboost -q
!pip install imblearn -q


import numpy as np #для матричных вычислений
import pandas as pd #для анализа и предобработки данных
import matplotlib.pyplot as plt #для визуализации
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier


df_train = pd.read_csv('/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/train_sessions.csv')

df_train.head()


# удаляем стобцы, в которых пропусков больше 50%
for col in df_train.columns:
    if df_train[col].isnull().sum() / df_train.shape[0] > 0.5:
        df_train.drop(col, axis=1, inplace=True)

# удаляем стобцы с одним уникальным значением
for col in df_train.columns:
    if df_train[col].nunique() == 1:
        df_train.drop(col, axis=1, inplace=True)


# Преобразуем тип datetime в отдельные признаки для тренировчных данных
for i in range(1, 11):
    df_train['time' + str(i)] = pd.to_datetime(df_train['time' + str(i)])

for i in range(1, 11):
    df_train['hour' + str(i)] = df_train['time' + str(i)].dt.hour
    df_train['day_of_week' + str(i)] = df_train['time' + str(i)].dt.dayofweek
    df_train['day' + str(i)] = df_train['time' + str(i)].dt.day
    df_train['month' + str(i)] = df_train['time' + str(i)].dt.month
    df_train['year' + str(i)] = df_train['time' + str(i)].dt.year

# Избавимся от исходного преобразованного признака
valid_df_train = df_train.drop(['time' + str(i) for i in range(1, 11)], axis=1)


# Заполним пропущенные значения
valid_df_train.fillna(0, inplace=True)


df_test = pd.read_csv('/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/test_sessions.csv')

df_test.head()


# удаляем стобцы, в которых пропусков больше 50%
for col in df_test.columns:
    if df_test[col].isnull().sum() / df_test.shape[0] > 0.5:
        df_test.drop(col, axis=1, inplace=True)

# удаляем стобцы с одним уникальным значением
for col in df_test.columns:
    if df_test[col].nunique() == 1:
        df_test.drop(col, axis=1, inplace=True)


# Преобразуем тип datetime в отдельные признаки для тестовых данных
for i in range(1, 11):
    df_test['time' + str(i)] = pd.to_datetime(df_test['time' + str(i)])

for i in range(1, 11):
    df_test['hour' + str(i)] = df_test['time' + str(i)].dt.hour
    df_test['day_of_week' + str(i)] = df_test['time' + str(i)].dt.dayofweek
    df_test['day' + str(i)] = df_test['time' + str(i)].dt.day
    df_test['month' + str(i)] = df_test['time' + str(i)].dt.month
    df_test['year' + str(i)] = df_test['time' + str(i)].dt.year

# Избавимся от исходного преобразованного признака
valid_df_test = df_test.drop(['time' + str(i) for i in range(1, 11)], axis=1)


# Заполним пропущенные значения
valid_df_test.fillna(0, inplace=True)


# Выделение признака и целевой переменной. Разделение на тренировочную и тестовую выборки
X = valid_df_train.drop('target', axis = 1)
y = valid_df_train['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Стандартизация. Обучение скалера на тренировочных данных и преобразование.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Применим метод обработки несбалансированных наборов данных
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)


# Обучаем модель 1 (случайный лес)
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    criterion='gini',
    random_state=42
)
rf_model.fit(X_train_resampled, y_train_resampled)

# Предсказание
y_test_pred = rf_model.predict(X_test_scaled)

# Оценка модели
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Accuracy (Random Forest): {test_accuracy:.3f}')
roc_auc_rf = roc_auc_score(y_test, y_test_pred)
print(f'ROC AUC score: {roc_auc_rf:.3f}')


# Обучим модель 2 (CatBoost)
catboost_model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=3,
    verbose=0
 )
catboost_model.fit(X_train_resampled, y_train_resampled)

# Предсказание
y_test_pred = catboost_model.predict(X_test_scaled)

# Оценка модели
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Accuracy (CatBoost): {test_accuracy:.3f}')
roc_auc_catboost = roc_auc_score(y_test, y_test_pred)
print(f'ROC AUC score: {roc_auc_catboost:.3f}')


# ROC-кривая
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')


# Стандартизация
valid_df_test_scaler = scaler.transform(valid_df_test)


# Предсказание
predictions = rf_model.predict(valid_df_test_scaler)


predictions_series = pd.Series(predictions)
predictions_series.value_counts()


df_sample = pd.read_csv('/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/sample_submission.csv')
df_sample.head()


# Подготовка файла
submission = pd.DataFrame({
    'session_id': df_sample['session_id'],
    'target': predictions
})
submission.to_csv('submission.csv', index=False)




