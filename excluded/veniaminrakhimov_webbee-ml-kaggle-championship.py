import pandas as pd
import numpy as np
import time
from catboost import CatBoostClassifier
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score
)
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin

# 1. Загрузка данных
train_data = pd.read_csv("/kaggle/input/client-churn-rate/train.csv")
test_data = pd.read_csv("/kaggle/input/client-churn-rate/test.csv")

# 2. Предобработка
cat_columns = train_data.select_dtypes(include=["object"]).columns

# Создаем новый признак, если есть необходимые столбцы
if "Balance" in train_data.columns and "EstimatedSalary" in train_data.columns:
    train_data["BalanceSalaryRatio"] = train_data["Balance"] / (train_data["EstimatedSalary"] + 1)
    test_data["BalanceSalaryRatio"] = test_data["Balance"] / (test_data["EstimatedSalary"] + 1)

# Определяем столбцы, которые исключаются из признакового набора
drop_cols = ["Exited"]
if "id" in train_data.columns:
    drop_cols.insert(0, "id")

X = train_data.drop(columns=drop_cols)
y = train_data["Exited"]

# Определяем категориальные признаки - для CatBoost нужны индексы столбцов
cat_features_names = [col for col in cat_columns if col in X.columns]
cat_features_indices = [list(X.columns).index(col) for col in cat_features_names]

# 3. Разделение данных для RandomizedSearchCV с eval_set
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Определяем начальную модель CatBoost с ранней остановкой и использованием GPU
cb_clf = CatBoostClassifier(
    loss_function="Logloss",
    iterations=200,
    task_type="GPU",
    devices='0',
    random_seed=42,
    verbose=0,
    early_stopping_rounds=30
)

# 5. Параметры для RandomizedSearchCV
param_dist = {
    "learning_rate": [0.01, 0.05, 0.1],
    "depth": [6, 8, 10],
    "l2_leaf_reg": [1, 3, 5],
    "bagging_temperature": [0, 1, 5]
}

random_search = RandomizedSearchCV(
    estimator=cb_clf,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring="roc_auc",
    n_jobs=1,  # При использовании GPU лучше использовать один поток
    verbose=2,
    random_state=42
)

# 6. Запуск RandomizedSearchCV с передачей параметров cat_features и eval_set для ранней остановки
random_search.fit(
    X_train, y_train,
    cat_features=cat_features_indices,
    eval_set=(X_val, y_val)
)

print("\nЛучшие параметры:", random_search.best_params_)
print("Лучшая ROC AUC по кросс-валидации:", random_search.best_score_)

# 7. Обучение финальной модели с найденными лучшими параметрами
best_params = random_search.best_params_

final_model = CatBoostClassifier(
    loss_function="Logloss",
    iterations=200,
    task_type="GPU",
    devices='0',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=30,
    **best_params
)

X_full_train, X_early, y_full_train, y_early = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

final_model.fit(
    X_full_train, y_full_train,
    eval_set=(X_early, y_early),
    cat_features=cat_features_indices,
    early_stopping_rounds=30
)

# 8. Обёртка для корректной передачи cat_features в кросс-валидацию с реализацией decision_function и classes_
class CatBoostWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, **params):
        self.params = params
        self.model = None
        
    def fit(self, X, y):
        self.model = CatBoostClassifier(**self.params)
        self.model.fit(
            X, y,
            cat_features=cat_features_indices,
            verbose=0,
            early_stopping_rounds=30
        )
        # Сохраняем классы для корректной работы scorer'а
        self.classes_ = self.model.classes_
        return self
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def decision_function(self, X):
        # Для roc_auc можно возвращать вероятность положительного класса
        return self.model.predict_proba(X)[:, 1]

# 9. Дополнительная 10-кратная кросс-валидация с использованием обёртки
wrapped_model = CatBoostWrapper(
    loss_function="Logloss",
    iterations=200,
    task_type="GPU",
    devices='0',
    random_seed=42,
    **best_params
)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    wrapped_model,
    X, y,
    cv=cv,
    scoring='roc_auc',
    n_jobs=1,
    verbose=2
)

print("CV ROC AUC: Среднее = {:.4f}, Стандартное отклонение = {:.4f}".format(
    np.mean(cv_scores), np.std(cv_scores)
))

# 10. Предсказания и сохранение сабмишена
if "id" in test_data.columns:
    test_ids = test_data["id"]
    X_test = test_data.drop("id", axis=1)
else:
    test_ids = np.arange(len(test_data))
    X_test = test_data.copy()

test_preds = final_model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "id": test_ids,
    "Exited": test_preds
})

submission.to_csv("submission.csv", index=False)
print("\nФайл 'submission.csv' успешно создан!")


