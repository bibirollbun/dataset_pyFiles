# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ==============================================================================
# 1. ИМПОРТ БИБЛИОТЕК И НАСТРОЙКИ
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
import re

pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
plt.style.use('ggplot')
warnings.filterwarnings('ignore', category=UserWarning)

print("Библиотеки успешно импортированы.")

# ==============================================================================
# 2. ЗАГРУЗКА ДАННЫХ
# ==============================================================================
train_df = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/train.csv')
test_df = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/test.csv')
test_ids = test_df['id']
print("Данные успешно загружены.")

# ==============================================================================
# 3. ПРОДВИНУТОЕ КОНСТРУИРОВАНИЕ ПРИЗНАКОВ (FEATURE ENGINEERING)
# ==============================================================================

print("\n--- Создаем новые признаки ---")
combined_df = pd.concat([train_df.drop(['Exited'], axis=1), test_df], ignore_index=True)

def engineer_surname_features(df):
    df_new = df.copy()
    df_new['Surname_Len'] = df_new['Surname'].str.len()
    df_new['Surname_Has_Apostrophe'] = df_new['Surname'].apply(lambda x: 1 if "'" in x else 0)
    df_new['Surname_Initial'] = df_new['Surname'].str[0]
    # Итальянские суффиксы
    italian_suffixes = ['o', 'i', 'a', 'ese', 'anti', 'ini']
    df_new['Surname_Is_Italian_like'] = df_new['Surname'].apply(
        lambda x: 1 if any(x.endswith(s) for s in italian_suffixes) else 0
    )
    # Африканские префиксы
    african_prefixes = ['Chukwu', 'Nwa', 'Onyeka', 'Ugo', 'Ebe', 'Ilo', 'Ogue']
    df_new['Surname_Is_African_like'] = df_new['Surname'].apply(
        lambda x: 1 if any(x.startswith(p) for p in african_prefixes) else 0
    )
    # Редкая фамилия
    surname_counts = df_new['Surname'].value_counts().to_dict()
    df_new['Surname_Rare'] = df_new['Surname'].map(lambda x: 1 if surname_counts[x] == 1 else 0)
    # Количество гласных/согласных
    df_new['Surname_Vowel_Count'] = df_new['Surname'].str.count(r'[AEIOUYaeiouyАЕЁИОУЫЭЮЯаеёиоуыэюя]')
    df_new['Surname_Consonant_Count'] = df_new['Surname'].str.count(r'[B-DF-HJ-NP-TV-Zb-df-hj-np-tv-zБ-Яб-я]', flags=re.IGNORECASE)
    # Флаг двойной буквы подряд
    df_new['Surname_Has_Double_Letter'] = df_new['Surname'].str.contains(r'(.)\1').astype(int)
    # Количество людей с фамилией
    surname_popularity = df_new['Surname'].map(surname_counts)
    df_new['Surname_Popularity'] = surname_popularity
    # Есть "родственники" в стране (минус сам клиент)
    df_new['Relatives_in_Country'] = df_new.groupby(['Surname', 'Geography'])['Surname'].transform('count') - 1
    # Славянский суффикс
    slavic_suffixes = ('ov', 'ev', 'skiy', 'ska', 'ich')
    df_new['Surname_Is_Slavic'] = df_new['Surname'].str.lower().str.endswith(slavic_suffixes).astype(int)
    # Признак редкой буквы
    df_new['Surname_Rare_Char'] = df_new['Surname'].str.contains(r'[qxzwñ]', flags=re.IGNORECASE).astype(int)
    # Длина фамилии бинами
    df_new['Surname_Len_Bin'] = pd.cut(df_new['Surname_Len'], [0, 5, 8, 12, 40], labels=['short', 'medium', 'long', 'verylong'])

    # Уже есть Surname_Is_Slavic, Surname_Rare, Gender, Geography

    df_new['Is_Slavic_Female'] = (df_new['Surname_Is_Slavic'] == 1) & (df_new['Gender'] == 'Female')
    df_new['Is_Slavic_Female'] = df_new['Is_Slavic_Female'].astype(int)

    df_new['Rare_Surname_Male'] = (df_new['Surname_Rare'] == 1) & (df_new['Gender'] == 'Male')
    df_new['Rare_Surname_Male'] = df_new['Rare_Surname_Male'].astype(int)

    df_new['Slavic_Surname_in_Germany'] = (df_new['Surname_Is_Slavic'] == 1) & (df_new['Geography'] == 'Germany')
    df_new['Slavic_Surname_in_Germany'] = df_new['Slavic_Surname_in_Germany'].astype(int)

    df_new['Italian_Surname_in_Spain'] = (df_new['Surname_Is_Italian_like'] == 1) & (df_new['Geography'] == 'Spain')
    df_new['Italian_Surname_in_Spain'] = df_new['Italian_Surname_in_Spain'].astype(int)

    df_new['Slavic_Female_in_Germany'] = (
        (df_new['Surname_Is_Slavic'] == 1) &
        (df_new['Gender'] == 'Female') &
        (df_new['Geography'] == 'Germany')
    ).astype(int)

    # Взаимодействие: редкая фамилия и высокая зарплата
    if 'EstimatedSalary' in df_new:
        salary_q75 = df_new['EstimatedSalary'].quantile(0.75)
        df_new['RareSurname_HighSalary'] = ((df_new['Surname_Rare'] == 1) & (df_new['EstimatedSalary'] > salary_q75)).astype(int)
    return df_new

def engineer_other_features(df):
    df_new = df.copy()
    df_new['BalanceSalaryRatio'] = df_new['Balance'] / df_new['EstimatedSalary'].replace(0, 1e-6)
    df_new['TenureByAge'] = df_new['Tenure'] / df_new['Age'].replace(0, 1e-6)
    df_new['CreditScoreByAge'] = df_new['CreditScore'] / df_new['Age'].replace(0, 1e-6)
    df_new['BalanceToProducts'] = df_new['Balance'] / df_new['NumOfProducts'].replace(0, 1e-6)
    df_new['Age_squared'] = df_new['Age'] ** 2
    df_new['CreditScore_squared'] = df_new['CreditScore'] ** 2
    df_new['IsZeroBalance'] = (df_new['Balance'] == 0).astype(int)
    df_new['IsSenior'] = (df_new['Age'] >= 60).astype(int)
    geo_balance_mean = df_new.groupby('Geography')['Balance'].transform('mean')
    geo_salary_mean = df_new.groupby('Geography')['EstimatedSalary'].transform('mean')
    df_new['BalanceMinusCountryMean'] = df_new['Balance'] - geo_balance_mean
    df_new['SalaryMinusCountryMean'] = df_new['EstimatedSalary'] - geo_salary_mean
    df_new['Age_Quartile'] = pd.qcut(df_new['Age'], 4, labels=False, duplicates='drop')
    df_new['Balance_Quartile'] = pd.qcut(df_new['Balance'], 4, labels=False, duplicates='drop')
    df_new['Is_Young_HighBalance'] = ((df_new['Age'] < 30) & (df_new['Balance'] > df_new['Balance'].median())).astype(int)
    df_new['Is_Senior_LowSalary'] = ((df_new['Age'] >= 60) & (df_new['EstimatedSalary'] < df_new['EstimatedSalary'].quantile(0.25))).astype(int)
    df_new['Germany_Female'] = ((df_new['Geography'] == 'Germany') & (df_new['Gender'] == 'Female')).astype(int)
    
    for col in ['Geography', 'Gender']:
        for agg_type in ['mean', 'std', 'min', 'max']:
            agg_col = 'CreditScore'
            new_col_name = f'{agg_col}_by_{col}_{agg_type}'
            agg_stats = train_df.groupby(col)[agg_col].agg(agg_type).to_dict()
            df_new[new_col_name] = df_new[col].map(agg_stats)
    return df_new

# Применяем функции
combined_df = engineer_surname_features(combined_df)
combined_df = engineer_other_features(combined_df)
print("Все признаки, включая новые по фамилии, созданы.")

# ==============================================================================
# 4. ФИНАЛЬНАЯ ПОДГОТОВКА ДАННЫХ
# ==============================================================================
print("\n--- Финальная подготовка данных ---")
drop_cols = ['id', 'CustomerId', 'Surname']
combined_df = combined_df.drop(columns=drop_cols)

# Категориальные признаки
label_encoder = LabelEncoder()
combined_df['Surname_Initial'] = label_encoder.fit_transform(combined_df['Surname_Initial'])
if 'Surname_Len_Bin' in combined_df:
    combined_df['Surname_Len_Bin'] = combined_df['Surname_Len_Bin'].astype(str)
    combined_df = pd.get_dummies(combined_df, columns=['Surname_Len_Bin'], drop_first=True, dtype=int)
combined_df = pd.get_dummies(combined_df, columns=['Geography', 'Gender'], drop_first=True, dtype=int)

# Разделяем обратно
X = combined_df.iloc[:len(train_df)].copy()
X_test = combined_df.iloc[len(train_df):].copy()
y = train_df['Exited']

print(f"Количество признаков до отбора: {X.shape[1]}")

# ==============================================================================
# 5. ОТБОР ПРИЗНАКОВ (FEATURE SELECTION)
# ==============================================================================
print("\n--- Проводим отбор признаков ---")
preliminary_model = lgb.LGBMClassifier(random_state=42)
preliminary_model.fit(X, y)
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': preliminary_model.feature_importances_
})
zero_importance_features = feature_importance_df[feature_importance_df['importance'] == 0]['feature'].tolist()

if zero_importance_features:
    print(f"Найдено {len(zero_importance_features)} признаков с нулевой важностью:")
    print(zero_importance_features)
    X.drop(columns=zero_importance_features, inplace=True)
    X_test.drop(columns=zero_importance_features, inplace=True)
    print(f"\nКоличество признаков после отбора: {X.shape[1]}")
else:
    print("Признаков с нулевой важностью не найдено. Используем все признаки.")

# ==============================================================================
# 6. ПОДБОР ГИПЕРПАРАМЕТРОВ С OPTUNA (на очищенных данных)
# ==============================================================================
print("\n--- Подбираем оптимальные гиперпараметры для LightGBM ---")

def objective(trial, X_data, y_data):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'is_unbalance': True,
        'random_state': 1337
    }
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)
    auc_scores = []

    for train_index, val_index in kf.split(X_data, y_data):
        X_train, X_val = X_data.iloc[train_index], X_data.iloc[val_index]
        y_train, y_val = y_data.iloc[train_index], y_data.iloc[val_index]
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='auc',
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        preds = model.predict_proba(X_val)[:, 1]
        auc_scores.append(roc_auc_score(y_val, preds))
    return np.mean(auc_scores)

study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, X, y), n_trials=50, timeout=900)

print(f"\nЛучшее значение AUC на кросс-валидации: {study.best_value:.5f}")
print("Лучшие гиперпараметры:")
best_params = study.best_params
print(best_params)

# ==============================================================================
# 7. ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ И ПРЕДСКАЗАНИЕ
# ==============================================================================
print("\n--- Обучаем финальную модель ---")
final_params = best_params
final_params['objective'] = 'binary'
final_params['metric'] = 'auc'
final_params['random_state'] = 42
final_params['is_unbalance'] = True
final_params['verbosity'] = -1

model = lgb.LGBMClassifier(**final_params)
model.fit(X, y)
print("Финальная модель обучена.")
predictions = model.predict_proba(X_test)[:, 1]

# ==============================================================================
# 8. ФОРМИРОВАНИЕ ФАЙЛА РЕШЕНИЯ
# ==============================================================================
submission_df = pd.DataFrame({'id': test_ids, 'Exited': predictions})
submission_df.to_csv('submission_3.csv', index=False)
print("\n--- Файл 'submission.csv' успешно создан ---")
print(submission_df.head())

# ==============================================================================
# 9. АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ
# ==============================================================================
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nТоп-25 самых важных признаков:")
print(feature_importance_df.head(25))

plt.figure(figsize=(12, 10))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(25), palette='plasma')
plt.title('Топ-25 самых важных признаков')
plt.xlabel('Важность')
plt.ylabel('Признак')
plt.tight_layout()
plt.show()


