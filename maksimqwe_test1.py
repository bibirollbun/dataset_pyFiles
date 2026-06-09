import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from category_encoders import TargetEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from catboost import CatBoostClassifier, Pool
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/chocolate-rating-prediction-ai-edu/chocolate_train.csv')
df_test = pd.read_csv('/kaggle/input/chocolate-rating-prediction-ai-edu/chocolate_test_new.csv')


df.head(2)


df['Cocoa Percent'] = df['Cocoa Percent'].apply(lambda x: x[0:-1])
df['Cocoa Percent'] = df['Cocoa Percent'].astype(float)


df['Bean Type'] = df['Bean Type'].fillna('\xa0')


df['Company Location'].unique()


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(df[df['Bean Type'] !=  '\xa0']['Bean Type'])


df['Broad Bean Origin'] = df['Broad Bean Origin'].fillna('Madagascar')


model_1 = CatBoostClassifier(verbose = 0
                            ,cat_features = ['Company' ,'Specific Bean Origin', 'Broad Bean Origin', 'Company Location'])
model_1.fit(df[df['Bean Type'] !=  '\xa0'].drop(columns = ['Rating','Bean Type']), target_encoded)


df_test['Cocoa Percent'] = df_test['Cocoa Percent'].apply(lambda x: x[0:-1])
df_test['Cocoa Percent'] = df_test['Cocoa Percent'].astype(float)


y_bean = model_1.predict(df.drop(columns = ['Rating','Bean Type']))
y_bean_test = model_1.predict(df_test.drop(columns = ['Bean Type']))


df['Bean Type'] = y_bean
df_test['Bean Type'] = y_bean_test


import optuna
from sklearn.model_selection import cross_val_score
# Функция для оптимизации
def objective(trial):
    # Определение гиперпараметров для оптимизации
    params = {
        'iterations': trial.suggest_int('iterations', 100, 300),  # Количество итераций
        'depth': trial.suggest_int('depth', 3, 4, 5),  # Глубина деревьев
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),  # Скорость обучения
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 4, 8),  # Коэффициент L2-регуляризации
        'loss_function': trial.suggest_categorical('loss_function', ['RMSE', 'MAE']),  # Функция потерь
        'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 20),  # Ранняя остановка
        'border_count': trial.suggest_categorical('border_count', [32, 128]),  # Количество границ
        'random_strength': trial.suggest_float('random_strength', 0.1, 10),  # Степень случайности
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),  # Интенсивность бэггинга
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise']),  # Стратегия роста
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 15),  # Минимальное количество объектов в листе
        'rsm': trial.suggest_float('rsm', 0.8, 1.0),  # Доля признаков для каждого дерева
        'random_seed': 42  # Фиксированный seed для воспроизводимости
    }

    # Создание модели
    model = CatBoostRegressor(verbose=0, 
                              cat_features = ['Company' , 'Broad Bean Origin','Specific Bean Origin',  'Company Location', 'Bean Type'], 
                              **params)

    # Оценка модели с использованием кросс-валидации
    score = cross_val_score(model,df.drop(columns = ['Rating']), df.Rating, scoring='r2', cv=3).mean()

    return score

# Создание исследования Optuna
study = optuna.create_study(direction='maximize')  # Максимизация R^2
study.optimize(objective, n_trials=70)  # Количество испытаний (trials)

# Лучшие параметры
print(f'Лучшие параметры: {study.best_params}')
print(f'Лучшее значение R^2: {study.best_value}')


model = CatBoostRegressor(iterations=186, 
                          depth=4,
                          learning_rate = 0.137, 
                          l2_leaf_reg = 6.8,
                          loss_function = 'RMSE',
                          early_stopping_rounds = 16,
                          border_count = 32,
                          random_strength = 5.14,
                          grow_policy = 'Depthwise',
                          min_data_in_leaf = 10,
                          rsm = 0.87,
                          #bootstrap_type = 'MVS',
                          leaf_estimation_iterations = 6,
                          fold_permutation_block = 138,
                          bagging_temperature = 0.137,
                          verbose=0, 
                          cat_features = ['Company' , 'Broad Bean Origin','Specific Bean Origin',  'Company Location', 'Bean Type'])
model.fit(df.drop(columns = ['Rating']), df.Rating)


y_pred = model.predict(df_test)


y_test = pd.read_csv('/kaggle/input/chocolate-rating-prediction-ai-edu/choco_sample_submission.csv')

y_test['Rating'] = y_pred


y_test


y_test.to_csv('submission6.csv', index=False)


os.environ['KAGGLE_USERNAME'] = 'maksimqwe'
os.environ['KAGGLE_KEY'] = 'c3a7ae9351871c1dc6b7680bf4846c60'

# Проверка
print(os.environ['KAGGLE_USERNAME'])
print(os.environ['KAGGLE_KEY'])


!kaggle competitions submit -c chocolate-rating-prediction-ai-edu -f submission6.csv -m "For test"


# Получение важности признаков
feature_importance = model.get_feature_importance()

# Создание DataFrame для удобного вывода
importance_df = pd.DataFrame({
    'Feature': df_test.columns,
    'Importance': feature_importance
})

# Сортировка по важности
importance_df = importance_df.sort_values(by='Importance', ascending=False)

importance_df


df


import matplotlib.pyplot as plt
import seaborn as sns

# Визуализация
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Важность признаков (CatBoost)')
plt.xlabel('Важность')
plt.ylabel('Признак')
plt.show()




