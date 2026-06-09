import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# Загрузка данных
train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

# Разделение признаков и целевой переменной
X = train_df.drop(['id', 'y'], axis=1)
y = train_df['y']
X_test = test_df.drop('id', axis=1)

# Обработка категориальных признаков
categorical_cols = X.select_dtypes(include='object').columns
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])  # Используем один и тот же encoder

# Обработка пропущенных значений
X.fillna(-999, inplace=True)
X_test.fillna(-999, inplace=True)

# Разделение для обучения и валидации
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Создаем датасеты для LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.005,
    'num_leaves': 64,
    'max_depth': 12,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'seed': 42
}

# Используем callbacks для ранней остановки
callbacks = [
    lgb.early_stopping(stopping_rounds=100),
    lgb.log_evaluation(period=100)
]

model = lgb.train(
    params,
    train_data,
    num_boost_round=2000,
    valid_sets=[train_data, valid_data],
    valid_names=['train', 'valid'],
    callbacks=callbacks
)

# Предсказание вероятностей
probas = model.predict(X_test, num_iteration=model.best_iteration)

# Создаем файл сабмита
submission = pd.DataFrame({
    'id': test_df['id'],
    'y': probas
})

# Сохраняем
submission.to_csv('submission.csv', index=False)
print("Submission файл сохранён как submission.csv")

