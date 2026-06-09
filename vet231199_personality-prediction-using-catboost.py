import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool

# ================================
# 1. Загрузка данных
# ================================
# Загрузка обучающего, тестового датасета и файла sample_submission
train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

# ================================
# 2. Предобработка данных
# ================================
# Удаление 'id' из признаков и выделение целевой переменной
X = train_df.drop(columns=['id', 'Personality'])
y = train_df['Personality']

# Проверка распределения классов
print(y.value_counts())

# Разделение данных на обучающую и валидационную выборки
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Определение категориальных признаков
categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

# ================================
# 3. Обработка пропусков в категориальных признаках
# ================================
# Заполнение NaN в категориальных признаках строкой 'missing'
for col in categorical_features:
    X_train[col] = X_train[col].fillna('missing')
    X_valid[col] = X_valid[col].fillna('missing')
    test_df[col] = test_df[col].fillna('missing')

# Обработка тестовых данных
X_test = test_df.drop(columns=['id'])
for col in categorical_features:
    X_test[col] = X_test[col].fillna('missing')

# ================================
# 4. Создание Pool объектов для CatBoost
# ================================
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
valid_pool = Pool(X_valid, y_valid, cat_features=categorical_features)
test_pool = Pool(X_test, cat_features=categorical_features)

# ================================
# 5. Обучение модели
# ================================
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    eval_metric='Accuracy',
    random_seed=42,
    verbose=100
)

model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

# ================================
# 6. Предсказания для тестового набора
# ================================
preds = model.predict(test_pool)

# ================================
# 7. Сохранение файла для сдачи
# ================================
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': preds
})

submission.to_csv('submission.csv', index=False)

print("Предсказания сохранены в 'submission.csv'")

