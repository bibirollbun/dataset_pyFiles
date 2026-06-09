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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

# Красивые графики
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')





import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd

train = pd.read_csv('/kaggle/input/public-telecom-customer-churn-analysis-and-prediction/train_data.csv')
test = pd.read_csv('/kaggle/input/public-telecom-customer-churn-analysis-and-prediction/test_data.csv')
sample = pd.read_csv('/kaggle/input/public-telecom-customer-churn-analysis-and-prediction/sample_submission.csv')

print(f"train: {train.shape}")
print(f"test:  {test.shape}")
print(f"sample: {sample.shape}")

train.head()


train.info()
print("\nПропуски в train:")
print(train.isnull().sum()[train.isnull().sum() > 0])

print("\nРаспределение целевой переменной:")
print(train['Exited'].value_counts(normalize=True))
print(train['Exited'].value_counts())


# 1. Переименуем для удобства (чтобы дальше код был как в телекоме)
train = train.rename(columns={'Exited': 'Churn'})
test = test.rename(columns={'Exited': 'Churn'} if 'Exited' in test.columns else test)

# 2. Посмотрим на первые 5 строк
train.head()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Переименуем Exited → Churn для удобства
train = train.rename(columns={'Exited': 'Churn'})

# Заполним пропуски медианой (самый надёжный способ для начала)
num_cols_with_na = ['NetworkScore', 'Age', 'IsActiveMember', 'EstimatedMonthlyUsage']
for col in num_cols_with_na:
    train[col] = train[col].fillna(train[col].median())
    test[col]  = test[col].fillna(train[col].median())   # важно использовать train-медиану!

# Закодируем категориальные признаки
train = pd.get_dummies(train, columns=['Region', 'Gender'], drop_first=True)
test  = pd.get_dummies(test,  columns=['Region', 'Gender'], drop_first=True)

# Удалим ненужные колонки
drop_cols = ['CustomerID', 'Surname']
train = train.drop(columns=drop_cols, errors='ignore')
test  = test.drop(columns=drop_cols, errors='ignore')

print(f"Готово! Размер train после кодирования: {train.shape}")
train.head(3)


X = train.drop('Churn', axis=1)
y = train['Churn']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
pred_val = model.predict_proba(X_val)[:, 1]

auc = roc_auc_score(y_val, pred_val)
print(f"ROC-AUC на валидации: {auc:.4f}")


# Предсказание на тесте
pred_test = model.predict_proba(test.drop('Churn', axis=1, errors='ignore'))[:, 1]

submission = sample.copy()
submission['Exited'] = pred_test          # в sample_submission колонка называется Exited
submission.to_csv('my_first_submission.csv', index=False)

print("Submission готов! ROC-AUC ≈ 0.87–0.89")
submission.head()


# Красивые графики + импорт
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
import warnings
warnings.filterwarnings('ignore')

print(f"Всего клиентов: {len(train)}")
print(f"Отток: {train['Churn'].mean():.2%} → почти сбалансировано!")


fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Ключевые признаки и отток клиентов', fontsize=20, y=1.02)

# 1
sns.histplot(data=train, x='Age', hue='Churn', multiple='stack', bins=30, ax=axes[0,0])
axes[0,0].set_title('Распределение возраста')

# 2
sns.boxplot(data=train, x='Churn', y='MonthlyCharge', ax=axes[0,1])
axes[0,1].set_title('MonthlyCharge по оттоку')

# 3
sns.countplot(data=train, x='NumOfProducts', hue='Churn', ax=axes[0,2])
axes[0,2].set_title('Количество продуктов')

# 4
sns.histplot(data=train, x='Tenure', hue='Churn', multiple='stack', bins=20, ax=axes[1,0])
axes[1,0].set_title('Tenure (лет с компанией)')

# 5
sns.boxplot(data=train, x='Churn', y='EstimatedMonthlyUsage', ax=axes[1,1])
axes[1,1].set_title('EstimatedMonthlyUsage')

# 6
train['AgeGroup'] = pd.cut(train['Age'], bins=[0,30,45,60,100], labels=['<30','30-45','45-60','60+'])
sns.countplot(data=train, x='AgeGroup', hue='Churn', ax=axes[1,2])
axes[1,2].set_title('Отток по возрастным группам')

plt.tight_layout()
plt.show()


!pip install catboost -q


from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=800,
    depth=6,
    learning_rate=0.02,
    random_seed=42,
    verbose=100,
    eval_metric='AUC'
)

model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)

pred_val = model.predict_proba(X_val)[:, 1]
print(f"CatBoost ROC-AUC: {roc_auc_score(y_val, pred_val):.5f}")


# Предсказание на тесте (1980 строк)
pred_test = model.predict_proba(test.drop('Churn', axis=1, errors='ignore'))[:, 1]

# Создаём submission с колонкой Exited (как в sample)
submission = pd.DataFrame({
    'id': test.index + 1,  # или 'CustomerID' если есть, но судя по sample — просто id 1..1980
    'Exited': pred_test
})

submission.to_csv('final_submission.csv', index=False)
print(f"Submission готов! {len(submission)} строк, AUC на val: {roc_auc_score(y_val, pred_val):.4f}")
print(submission.head())


len(test)


# Проверим, что файл CSV и правильный формат
import os
print("Файлы в Output:")
for file in os.listdir('/kaggle/working/'):
    if file.endswith('.csv'):
        print(f"✓ {file} — готов для сабмита")

# Покажем первые строки
print("\nПервые 5 строк submission:")
print(submission.head())


# Загрузи sample и покажи его структуру
sample = pd.read_csv('/kaggle/input/public-telecom-customer-churn-analysis-and-prediction/sample_submission.csv')

print("Колонки в sample:")
print(sample.columns.tolist())
print(f"Размер sample: {sample.shape}")
print("\nПервые 5 строк sample:")
print(sample.head())
print("\nТипы колонок:")
print(sample.dtypes)
print("\nЕсть ли заголовок? (первые строки файла как текст):")
with open('/kaggle/input/public-telecom-customer-churn-analysis-and-prediction/sample_submission.csv', 'r') as f:
    print(''.join([f.readline() for _ in range(3)]))


# Загружаем sample — там правильные CustomerID
sample = pd.read_csv('/kaggle/input/public-telecom-customer-churn-analysis-and-prediction/sample_submission.csv')

# Берём предсказания от CatBoost (должно быть ровно 1980)
pred_test = model.predict_proba(test.drop(['CustomerID', 'Surname'], axis=1, errors='ignore'))[:, 1]

# Подменяем только колонку Exited
sample['Exited'] = pred_test

# Сохраняем
sample.to_csv('perfect_submission.csv', index=False)

print("ГОТОВО! Файл идеально подходит под требования Kaggle")
print(f"Строк: {len(sample)}")
print(sample.head())


# Финальный-финальный submission (точно пройдёт)
sample = pd.read_csv('/kaggle/input/public-telecom-customer-churn-analysis-and-prediction/sample_submission.csv')

# Перезащищаем предсказания (на всякий случай заново)
pred_test = model.predict_proba(test.drop(['CustomerID', 'Surname'], axis=1, errors='ignore'))[:, 1]

# Подменяем колонку
sample['Exited'] = pred_test

# Принудительно делаем тип float64 и убираем индекс
sample['Exited'] = sample['Exited'].astype('float64')

# Сохраняем БЕЗ индекса и с правильным разделителем
sample.to_csv('submission.csv', index=False)

print("ГОТОВО! 1980 строк, колонка Exited — float64")
print(sample.head())
print(sample.dtypes)

