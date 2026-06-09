import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)  # показывает все столбцы
pd.set_option('display.width', None)     # предотвращает перенос строк


# Загрузка обучающей выборки
train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
# Загрузка тестовой выборки
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
# Загрузка оригинального датасета
original_df = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')


def analyze_diabetes_datasets(train_df, test_df, original_df):
    """
    Анализ трёх датасетов для соревнования по диабету.
    Выводит:
    - форму и размер каждого датасета;
    - список числовых и категориальных признаков;
    - статистику по числовым признакам;
    - уникальные значения и частоты для категориальных признаков.
    """
    datasets = {
        'Обучающая выборка (train)': train_df,
        'Тестовая выборка (test)': test_df,
        'Оригинальный датасет': original_df
    }
    
    for name, df in datasets.items():
        print(f"\n{'='*60}")
        print(f"{name.upper()}")
        print(f"{'='*60}")
        print(f"Форма: {df.shape} (строк: {len(df)}, столбцов: {len(df.columns)})")
        
        # Числовые признаки
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"\n1. Числовые признаки ({len(num_cols)}):")
        if num_cols:
            desc = df[num_cols].describe().T
            for col in num_cols:
                print(f"  {col}:")
                print(f"    count: {desc.loc[col, 'count']:6.0f}")
                print(f"    mean:  {desc.loc[col, 'mean']:8.3f}")
                print(f"    std:   {desc.loc[col, 'std']:8.3f}")
                print(f"    min:   {desc.loc[col, 'min']:8.3f}")
                print(f"    25%:   {desc.loc[col, '25%']:8.3f}")
                print(f"    50%:   {desc.loc[col, '50%']:8.3f}")
                print(f"    75%:   {desc.loc[col, '75%']:8.3f}")
                print(f"    max:   {desc.loc[col, 'max']:8.3f}")
        else:
            print("  Нет числовых признаков")

        # Категориальные признаки
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"\n2. Категориальные признаки ({len(cat_cols)}):")
        if cat_cols:
            for col in cat_cols:
                # Уникальные значения и их частоты
                value_counts = df[col].value_counts(dropna=False)
                unique_vals = value_counts.size
                vals = sorted(df[col].dropna().unique())
                
                print(f"  {col} → {unique_vals} уникальных значений:")
                print(f"    Значения: {vals}")
                
                # Выводим частоты и доли
                print("    Частоты:")
                total = len(df[col])
                for val, cnt in value_counts.items():
                    share = cnt / total
                    print(f"      {val}: {cnt} ({share:.1%})")
        else:
            print("  Нет категориальных признаков")
        
        print()  # Пустая строка между датасетами



# Вызов функции
analyze_diabetes_datasets(train_df, test_df, original_df)






import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 1. Загрузка данных
train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
original_df = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')

# 2. Подготовка данных
# Удаляем ID из обучающей выборки
train_df = train_df.drop('id', axis=1)

# Сохраняем ID из тестовой выборки для сабмишена
test_ids = test_df['id']
X_test = test_df.drop('id', axis=1)

# Определяем признаки и целевую переменную
X = train_df.drop('diagnosed_diabetes', axis=1)
y = train_df['diagnosed_diabetes']

# 3. Разбиение на train/val (для проверки качества)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Определение категориальных признаков
cat_features = [
    'gender', 'ethnicity', 'education_level',
    'income_level', 'smoking_status', 'employment_status'
]

# 5. Настройка и обучение CatBoost с оптимизированными параметрами
cat_params = {
    'iterations': 1017,
    'learning_rate': 0.07, 
    'depth': 6,
    'l2_leaf_reg': 5.92,  
    'random_strength': 0.75,  
    'bagging_temperature': 0.46, 
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': 100,
    'random_seed': 42,
    'allow_writing_files': False,
    'task_type': 'CPU',
    'devices': '0',
    'bootstrap_type': 'Bayesian'
}

model = CatBoostClassifier(**cat_params, cat_features=cat_features)

# Обучение модели
model.fit(X_train, y_train, eval_set=(X_val, y_val))

# 6. Предсказания на валидации (проверка качества)
y_val_pred_proba = model.predict_proba(X_val)[:, 1]
auc_score = roc_auc_score(y_val, y_val_pred_proba)
print(f"AUC на валидации: {auc_score:.4f}")

# 7. Предсказания на тестовой выборке
y_test_pred_proba = model.predict_proba(X_test)[:, 1]

# 8. Формирование и сохранение сабмишена
submission = pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': y_test_pred_proba
})
submission.to_csv('submission_cat.csv', index=False)

print("Сабмишен сохранён как 'submission_cat.csv'")
print(f"Форма сабмишена: {submission.shape}")
print("\nПервые 5 строк:")
print(submission.head())



print("Min probability:", submission["diagnosed_diabetes"].min())
print("Max probability:", submission["diagnosed_diabetes"].max())
print("Below 0:", (submission["diagnosed_diabetes"] < 0).sum())
print("Above 1:", (submission["diagnosed_diabetes"] > 1).sum())









import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping


# 1. Загрузка и подготовка данных
train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

train_df = train_df.drop('id', axis=1)
test_ids = test_df['id']
X_test = test_df.drop('id', axis=1)
X = train_df.drop('diagnosed_diabetes', axis=1)
y = train_df['diagnosed_diabetes']

# 2. Определение категориальных признаков
cat_features = [
    'gender', 'ethnicity', 'education_level',
    'income_level', 'smoking_status', 'employment_status'
]

# 3. Преобразование в тип 'category' для train и test
for col in cat_features:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# 4. Настройка параметров LightGBM
lgb_params = {
    'n_estimators': 1017,
    'learning_rate': 0.07,
    'max_depth': 6,
    'num_leaves': 31,
    'reg_lambda': 5.92,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'bagging_freq': 1,
    'objective': 'binary',
    'metric': 'auc',
    'random_state': 42,
    'device': 'cpu'
}

# 5. Кросс‑валидация
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}/{n_splits}")
    
    # Разбиение данных по индексам
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx]

    # Обучение модели
    model = LGBMClassifier(**lgb_params)
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        categorical_feature=cat_features,  # Указываем категориальные признаки
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
    )

    # OOF предсказания (вне выборки)
    oof_preds[val_idx] = model.predict_proba(X_val_fold)[:, 1]
    # Предсказания на тесте (усредняем по фолдам)
    test_preds += model.predict_proba(X_test)[:, 1] / n_splits

# 6. Оценка качества на кросс‑валидации
print(f"CV AUC (LightGBM): {roc_auc_score(y, oof_preds):.4f}")

# 7. Формирование сабмишена
submission = pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': test_preds
})
submission.to_csv('submission.csv', index=False)
print("Сабмишен сохранён как 'submission.csv'")











