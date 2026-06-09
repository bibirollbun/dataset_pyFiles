import pandas as pd
import numpy as np



train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
# Загрузка дополнительного датасета loan_dataset_20000.csv
loan_dataset = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')


import pandas as pd
import numpy as np
from scipy.stats import skew

def analyze_loan_datasets_extended(train_df, test_df, loan_df):
    """
    Расширенный анализ трёх датасетов: train, test, loan_dataset.
    Выводит:
    - форму датасетов;
    - числовые признаки: describe(), пропуски, выбросы (IQR), skewness;
    - категориальные признаки: уникальные значения, пропуски, частота категорий.
    """
    datasets = {
        'train': train_df,
        'test': test_df,
        'loan_dataset': loan_df
    }
    
    for name, df in datasets.items():
        print(f"{'-'*50}\n{name.upper()}\n{'-'*50}")
        print(f"Форма датасета: {df.shape}")
        print(f"Количество строк: {len(df)}")
        print(f"Количество столбцов: {len(df.columns)}")

        # Пропуски во всём датасете
        total_missing = df.isnull().sum().sum()
        print(f"\nПропуски в датасете: {total_missing} значений")

        # Разделяем признаки
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        print(f"\n1. Числовые признаки ({len(num_cols)}):")
        if num_cols:
            desc = df[num_cols].describe().T
            for col in num_cols:
                print(f"   {col}:")
                print(f"      count: {desc.loc[col, 'count']:.0f}")
                print(f"      mean:  {desc.loc[col, 'mean']:.3f}")
                print(f"      std:   {desc.loc[col, 'std']:.3f}")
                print(f"      min:   {desc.loc[col, 'min']:.3f}")
                print(f"      25%:   {desc.loc[col, '25%']:.3f}")
                print(f"      50%:   {desc.loc[col, '50%']:.3f}")
                print(f"      75%:   {desc.loc[col, '75%']:.3f}")
                print(f"      max:   {desc.loc[col, 'max']:.3f}")

                # Пропуски в колонке
                missing = df[col].isnull().sum()
                print(f"      пропуски: {missing} ({missing/len(df)*100:.1f}%)")

                # Выбросы через IQR
                Q1 = desc.loc[col, '25%']
                Q3 = desc.loc[col, '75%']
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                print(f"      выбросы (IQR×1.5): {outliers} ({outliers/len(df)*100:.1f}%)")


                # Асимметрия
                skewness = skew(df[col].dropna())
                print(f"      асимметрия (skew): {skewness:.3f}")
        else:
            print("   Нет числовых признаков")

        print(f"\n2. Категориальные признаки ({len(cat_cols)}):")
        for col in cat_cols:
            unique_vals = df[col].nunique()
            unique_list = sorted(df[col].dropna().unique())
            missing = df[col].isnull().sum()
            
            print(f"   {col} → {unique_vals} уникальных значений:")
            print(f"         пропуски: {missing} ({missing/len(df)*100:.1f}%)")
            print(f"         {unique_list}")

# Запуск анализа
analyze_loan_datasets_extended(train, test, loan_dataset)




import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def preprocess_loan_data(train_df, test_df):
    """
    Предобработка данных для задачи Predicting Loan Payback.
    
    Действия:
    - Удаление id
    - Логарифмирование annual_income
    - One-Hot Encoding категориальных признаков
    - Target Encoding grade_subgrade
    
    Возвращает:
    - X_train, y_train, X_test, test_ids
    """
    # Копии датасетов
    train = train_df.copy()
    test = test_df.copy()
    
    # Сохранение id для сабмишена
    test_ids = test['id'].values
    
    # Удаление id
    train.drop('id', axis=1, inplace=True)
    test.drop('id', axis=1, inplace=True)

    # 1. Логарифмирование annual_income
    for df in [train, test]:
        df['log_annual_income'] = np.log1p(df['annual_income'])
        df.drop('annual_income', axis=1, inplace=True)

    # 2. One-Hot Encoding
    ohe_cols = [
        'gender', 'marital_status', 'education_level',
        'employment_status', 'loan_purpose'
    ]
    
    ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
    ohe.fit(train[ohe_cols])
    
    train_ohe = ohe.transform(train[ohe_cols])
    test_ohe = ohe.transform(test[ohe_cols])
    
    
    ohe_feature_names = ohe.get_feature_names_out(ohe_cols)

    # 3. Target Encoding grade_subgrade
    # Среднее loan_paid_back по категориям на train
    te_map = train.groupby('grade_subgrade')['loan_paid_back'].mean().to_dict()
    global_mean = train['loan_paid_back'].mean()  # для неизвестных категорий

    # Применяем без inplace
    train['grade_subgrade_te'] = train['grade_subgrade'].map(te_map)
    train['grade_subgrade_te'] = train['grade_subgrade_te'].fillna(global_mean)
    
    test['grade_subgrade_te'] = test['grade_subgrade'].map(te_map)
    test['grade_subgrade_te'] = test['grade_subgrade_te'].fillna(global_mean)

    train.drop('grade_subgrade', axis=1, inplace=True)
    test.drop('grade_subgrade', axis=1, inplace=True)

    # 4. Сборка финальных датасетов
    num_cols = [
        'debt_to_income_ratio', 'credit_score', 'loan_amount',
        'interest_rate', 'log_annual_income', 'grade_subgrade_te'
    ]

    X_train = pd.concat([
        train[num_cols],
        pd.DataFrame(train_ohe, columns=ohe_feature_names, index=train.index)
    ], axis=1)

    X_test = pd.concat([
        test[num_cols],
        pd.DataFrame(test_ohe, columns=ohe_feature_names, index=test.index)
    ], axis=1)

    y_train = train['loan_paid_back'].copy()

    return X_train, y_train, X_test, test_ids


# Пример вызова
X_train, y_train, X_test, test_ids = preprocess_loan_data(train, test)

# Вывод признаков
print("Признаки в модели:")
for col in X_train.columns:
    print(f"- {col}")







import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

# 1. Разделение на train/val (для валидации)
X_train_full = X_train  # Ваш предобработанный X_train из предыдущей функции
y_train_full = y_train  # Ваша целевая переменная


# Делим train на train + validation (20% на валидацию)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full  # сохраняем баланс классов
)

# 2. Создание DMatrix (оптимальный формат для XGBoost)
dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval = xgb.DMatrix(X_val, label=y_val)


# 3. Параметры модели
params = {
    'objective': 'binary:logistic',  # бинарная классификация
    'eval_metric': 'auc',           # метрика — ROC-AUC
    'max_depth': 6,                 # глубина деревьев
    'learning_rate': 0.1,        # шаг обучения
    'subsample': 0.8,             # доля объектов для каждого дерева
    'colsample_bytree': 0.8,    # доля признаков для каждого дерева
    'gamma': 0,                   # регуляризация (минимальная редукция потери для сплит)
    'reg_lambda': 1,             # L2-регуляризация
    'seed': 42
}

# 4. Обучение модели
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,          # максимальное число деревьев
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=50,     # остановка, если val-AUC не растёт 50 итераций
    verbose_eval=False              # не выводить лог обучения
)

# 5. Предсказания на валидации
y_val_pred = model.predict(dval)  # вероятности [0..1]


# 6. Подсчёт ROC-AUC
val_auc = roc_auc_score(y_val, y_val_pred)
print(f"Валидационный ROC-AUC: {val_auc:.4f}")


# 7. Предсказания на test (для сабмишена)
dtest = xgb.DMatrix(X_test)  # X_test — ваш предобработанный test
test_preds = model.predict(dtest)


# 8. Сохранение сабмишена
# 8. Сохранение сабмишена
submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': test_preds
})
submission.to_csv('submission_lgb.csv', index=False)
print("Сабмишен сохранён!")



import pandas as pd
import numpy as np

def preprocess_loan_data_for_catboost(train_df, test_df):
    """
    Предобработка данных для задачи Predicting Loan Payback с фокусом на CatBoost.

    Действия:
    - Удаление id
    - Логарифмирование annual_income (сохраняем как дополнительный признак)
    - Все остальные признаки — в сыром виде

    Возвращает:
    - X_train, y_train, X_test, test_ids, cat_features
    """
    # Копии датасетов
    train = train_df.copy()
    test = test_df.copy()

    # Сохранение id для сабмишена
    test_ids = test['id'].values

    # Удаление 'id' без inplace
    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)

    # 1. Логарифмирование annual_income без inplace — создаём новый столбец
    train['log_annual_income'] = np.log1p(train['annual_income'])
    test['log_annual_income'] = np.log1p(test['annual_income'])

    # Удаляем исходный annual_income, оставляем только логарифмированную версию
    train = train.drop('annual_income', axis=1)
    test = test.drop('annual_income', axis=1)

    # 2. Список категориальных признаков для CatBoost
    cat_features = [
        'gender', 'marital_status', 'education_level',
        'employment_status', 'loan_purpose', 'grade_subgrade'
    ]

    # 3. Сборка финальных датасетов — все признаки в сыром виде, кроме annual_income
    X_train = train.drop('loan_paid_back', axis=1)  # все признаки кроме целевой
    X_test = test  # все признаки

    y_train = train['loan_paid_back'].copy()

    return X_train, y_train, X_test, test_ids, cat_features



from catboost import CatBoostClassifier

# Вызов функции предобработки
X_train, y_train, X_test, test_ids, cat_features = preprocess_loan_data_for_catboost(train, test)

# Обучение модели — CatBoost сам обработает категориальные признаки
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    verbose=100,
    random_seed=42,
    cat_features=cat_features  # указываем, какие столбцы — категориальные
)

model.fit(X_train, y_train)

# Предсказание вероятностей
predictions = model.predict_proba(X_test)[:, 1]

# Создание сабмишена
submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': predictions
})
submission.to_csv('submission.csv', index=False)











