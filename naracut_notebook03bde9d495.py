import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

pd.set_option("display.max_columns", 30)


# 1. Загрузка данных
train_df = pd.read_csv("/kaggle/input/playground-series-s3e24/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s3e24/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s3e24/sample_submission.csv")


# 2. Предобработка
train_df = train_df.drop(columns=["id"])
test_df_id = test_df["id"]
test_df = test_df.drop(columns=["id"])


# 3. Разделение на train/val
X = train_df.drop(columns=["smoking"])
y = train_df["smoking"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# 4. Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_df_scaled = scaler.transform(test_df)


# 5. Модель LightGBM
model = lgb.LGBMClassifier(
    objective="binary", # Цель модели: бинарная классификация (0/1). LightGBM будет предсказывать вероятности принадлежности к классу "курящий".
    metric="auc", # ROC-AUC используется как метрика для оценки качества модели. Чем ближе значение к 1, тем лучше модель.
    learning_rate=0.05, # Контролирует "шаг" обучения: насколько сильно каждое новое дерево корректирует предыдущие ошибки. Значение 0.05 — баланс между скоростью обучения и точностью.
    num_leaves=63, # Количество листьев в каждом дереве. Больше листьев = более сложные деревья, но риск переобучения. Значение 63 — стандартный выбор для бинарной классификации.
    max_depth=8, # Максимальная глубина дерева (сколько раз можно "разделить" данные). 8 уровней — компромисс между сложностью и скоростью.
    n_estimators=200, # Количество деревьев в ансамбле. 200 деревьев достаточно для хорошей точности, но не слишком медленно.
    subsample=0.8, # На каждой итерации обучения используется 80% случайных строк из обучающего датасета. Уменьшает переобучение.
    colsample_bytree=0.8, # На каждой итерации используются 80% случайных признаков для построения дерева. Добавляет случайность, улучшая обобщаемость.
    reg_alpha=0.1, #L1 (reg_alpha) и L2 (reg_lambda) регуляризация . Пенализируют слишком сложные модели, снижая риск переобучения.
    reg_lambda=0.1,
    random_state=42, # Фиксирует случайность для воспроизводимости результатов.
    n_jobs=-1 # Использует все CPU-ядра для ускорения обучения.
)



# 6. Обучение с early stopping и логированием
model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    eval_metric="auc",
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(50)  # Лог каждые 50 итераций
    ]
)


# 7. Проверка обучения
if hasattr(model, "booster_"):
    print("Модель обучена успешно!")
    best_iter = model.best_iteration_
else:
    print("Ошибка: модель не обучилась")


# 8. Предсказание
y_pred_test = model.predict_proba(test_df_scaled, num_iteration=best_iter)[:, 1]


# 9. Формирование сабмита
submission = pd.DataFrame({
    "id": test_df_id,
    "smoking": y_pred_test
})
submission.to_csv("submission.csv", index=False)


# 10. Проверка примера предсказаний
print("Примеры предсказаний:")
print(submission.head())

