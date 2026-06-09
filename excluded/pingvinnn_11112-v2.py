import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb


train = pd.read_csv('/kaggle/input/neoai-2025-tricy-table-data/train_tables.csv')
test = pd.read_csv('/kaggle/input/neoai-2025-tricy-table-data/test_tables.csv')


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

data=train.copy()
# Обработка пропусков и кодирование категориальных признаков
features = data.drop(['day', 'hour', 'minute'], axis=1)
features = pd.get_dummies(features)

def train_and_predict_reg(target_column):
    # Разделение данных
    train_data = data[~data[target_column].isnull()]
    test_data = data[data[target_column].isnull()]

    X_train = train_data.drop([target_column], axis=1)
    y_train = train_data[target_column]
    X_test = test_data.drop([target_column], axis=1)
    
    # Выравнивание колонок
    X_train, X_test = X_train.align(X_test, join='left', axis=1)

    # Импутация пропусков в признаках
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Обучение модели
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Предсказание и заполнение пропусков
    predictions = model.predict(X_test)
    data.loc[data[target_column].isnull(), target_column] = predictions


def train_and_predict_cls(target_column):
    # Разделение данных
    train_data = data[~data[target_column].isnull()]
    test_data = data[data[target_column].isnull()]

    X_train = train_data.drop([target_column], axis=1)
    y_train = train_data[target_column]
    X_test = test_data.drop([target_column], axis=1)
    
    # Выравнивание колонок
    X_train, X_test = X_train.align(X_test, join='left', axis=1)

    # Импутация пропусков в признаках
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Обучение модели
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Предсказание и заполнение пропусков
    predictions = model.predict(X_test)
    data.loc[data[target_column].isnull(), target_column] = predictions



data = train.copy().drop(columns='target')
for column in ['hour', 'minute']:
    train_and_predict_reg(column)
train_and_predict_cls('day')
train = data


data = test.copy()
for column in ['hour', 'minute']:
    train_and_predict_reg(column)
train_and_predict_cls('day')
test = data


train


def fill_missing_with_neighbors_avg(df):
    # Создаем копию DataFrame чтобы избежать изменений в оригинале
    df_filled = df.copy()

    # Проходим по каждому столбцу
    for column in df.columns:
        # Получаем индексы пропущенных значений
        nan_indices = df[column][df[column].isna()].index

        # Обрабатываем каждое пропущенное значение
        for idx in nan_indices:
            # Находим ближайшего непустого соседа сверху
            upper_value = None
            for i in range(idx - 1, -1, -1):
                if pd.notna(df[column].iloc[i]):
                    upper_value = df[column].iloc[i]
                    break

            # Находим ближайшего непустого соседа снизу
            lower_value = None
            for i in range(idx + 1, len(df)):
                if pd.notna(df[column].iloc[i]):
                    lower_value = df[column].iloc[i]
                    break

            # Вычисляем среднее между найденными значениями
            if upper_value is not None and lower_value is not None:
                df_filled.at[idx, column] = (upper_value + lower_value) / 2
            elif upper_value is not None:
                df_filled.at[idx, column] = upper_value
            elif lower_value is not None:
                df_filled.at[idx, column] = lower_value

    return df_filled

# Заполнение пропущенных значений
df_filled_train = fill_missing_with_neighbors_avg(train.sort_values(by=['day', 'hour', 'minute']))

print(df_filled_train)


# Заполнение пропущенных значений
df_filled_test = fill_missing_with_neighbors_avg(test.sort_values(by=['day', 'hour', 'minute']))

print(df_filled_test)


'''for i in ['feat_0', 'feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6',
       'feat_7', 'feat_8', 'minute']:
    train[i] = train[i].fillna(np.mean(train[i]))
    test[i] = test[i].fillna(np.mean(train[i]))'''


'''missing_mask = train['day'].isnull()
random_days = np.random.choice([1, 2], size=missing_mask.sum())
train.loc[missing_mask, 'day'] = random_days'''


'''missing_mask = test['day'].isnull()
random_days = np.random.choice([1, 2], size=missing_mask.sum())
test.loc[missing_mask, 'day'] = random_days'''


'''median_hours = test.groupby('day')['hour'].median()

def fill_hour(row):
    if pd.isna(row['hour']):
        return median_hours[row['day']]
    else:
        return row['hour']
test['hour'] = test.apply(fill_hour, axis=1)
test'''


'''median_hours = train.groupby('day')['hour'].median()
train['hour'] = train.apply(fill_hour, axis=1)
train'''


def transform_data(df):
    features = ['feat_0', 'feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8']
    
    # Calculate mean of features
    df['mean_of_features'] = df[features].mean(axis=1)
    
    # Calculate mean by day
    mean_by_day = df.groupby('day')[features].mean().add_prefix('mean_day_')
    df = df.join(mean_by_day, on='day')
    
    # Calculate square root of features
    for feature in features:
        sqrt = f'sqrt_{feature}'
        df[sqrt] = np.sqrt(df[feature])
        diff = f'diff_{feature}'
        df[diff] = df[feature] - df[feature].shift(1)
        df[f'log_{feature}'] = np.log(df[feature])
        
    
    return df


train = df_filled_train.copy()
test = df_filled_test.copy()


for_target = pd.read_csv('/kaggle/input/neoai-2025-tricy-table-data/train_tables.csv')
for_id = pd.read_csv('/kaggle/input/neoai-2025-tricy-table-data/test_tables.csv')


train = train.sort_index()
train['target'] = for_target['target']
test = test.sort_index()
test['id'] = for_id['id']
train


train['source'] = 'train'
test['source'] = 'test'

train['id'] = None
test['target'] = None

big_data = pd.concat([train, test], ignore_index=True)
big_data.sort_values(by=['day', 'hour', 'minute'], inplace=True)


train.sort_values(by=['day', 'hour', 'minute'])


big_data = transform_data(big_data)


# Условие для тренировочного набора: наличие столбца 'target'
train_data = big_data[big_data['target'].notna()]

# Условие для тестового набора: наличие столбца 'id' и отсутствие 'target'
test_data = big_data[big_data['id'].notna()]

# Убедитесь, что в train отсутствует 'id' и в test отсутствует 'target'
train_data = train_data.drop(columns=['id'], errors='ignore')
test_data = test_data.drop(columns=['target'], errors='ignore')

# Вывод размеров полученных наборов данных для проверки
print(f'Train data shape: {train_data.shape}')
print(f'Test data shape: {test_data.shape}')


train_data['target'] = np.log(train_data['target'])


def clf_train(train, test, target, weight_col, id_col, name_file = 'sub.csv', func_inv = None):

    param = {
    'learning_rate': 0.1,
    'num_leaves': 48,
    'lambda_l1' : 1,
    'lambda_l2' : 1,
    'min_data_in_leaf' : 100,
    'objective': 'mae',
    'verbosity':-1,
    }
    
    predict_test = np.zeros(len(test))

    tr = lgb.Dataset(train, target, weight=weight_col)
    bst = lgb.train(param, tr, num_boost_round=500)
    predict_test = bst.predict(test)
    if func_inv:
        predict_test = func_inv(predict_test)
    sub = pd.DataFrame()
    sub['id'] = id_col
    sub['target'] = predict_test
    sub.to_csv(name_file, index = None)


def func_inv(x):
    
    return np.exp(x)


drop_cols = ['target', 'source']
train_cols = [c for c in train_data.columns if c not in drop_cols]
weight = np.ones(len(train))


target = train_data['target']
id_col = test_data['id'].tolist()
train = train_data[train_cols]
test = test_data[train_cols]
weight_col = weight
name_file = 'sub.csv'





train_data = train_data.drop(5696)
train_data.sample(frac=1)


test_sub = clf_train(train[train_cols], test[train_cols], target, weight, id_col, 'submission_itog.csv', func_inv = func_inv)




