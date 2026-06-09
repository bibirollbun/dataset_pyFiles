import numpy as np
import pandas as pd


train_data = pd.read_csv('/kaggle/input/santander-customer-satisfaction/train.csv')
test_data = pd.read_csv('/kaggle/input/santander-customer-satisfaction/test.csv')


train_data


# сразу отметим, что в тестовой выборке данных даже больше, чем в обучающей
train_data.shape, test_data.shape


# обратим внимание также, что огромное число признаков имеют нулевой 75-й перцентиль
train_data.describe()


# посмотрим на большие перцентили
train_data.describe(percentiles=[0.9, 0.95, 0.99])


# очень похоже на выбросы, посмотрим, как дела с тестовой выборкой
test_data.describe(percentiles=[0.9, 0.95, 0.99])


# с тестовой выборкой ситуация такая же, так что потенциальные выбросы удалять не стоит
# для начала просто определим, какие признаки страдают такой штукой

description = train_data.describe(percentiles=[0.9, 0.95, 0.99])
weird_features_train = description.loc['99%'] == 0
weird_features_train = weird_features_train[weird_features_train].index.tolist()
len(weird_features_train)


description_test = test_data.describe(percentiles=[0.9, 0.95, 0.99])
weird_features_test = description_test.loc['99%'] == 0
weird_features_test = weird_features_test[weird_features_test].index.tolist()
len(weird_features_test)


set(weird_features_train) - set(weird_features_test)


# итак, есть три признака, которые не являются выбросами в тестовой выборке

test_data['ind_var13_largo'].describe(), train_data['ind_var13_largo'].describe()


test_data['num_var13_largo'].describe(), train_data['num_var13_largo'].describe()


test_data['saldo_var13_largo'].describe(), train_data['saldo_var13_largo'].describe()


# теперь отлогарифмируем такие выбросы, чтобы они не вносили излишнюю погрешность в данные
# информацию о них мы, тем не менее, не потеряем

train_data[weird_features_train] = np.log1p(train_data[weird_features_train])
test_data[weird_features_train] = np.log1p(test_data[weird_features_train])


train_data.describe()


# также посмотрим, есть ли скоррелированные признаки в датасете и избавимся от них
# для этого построим корреляционную матрицу
corr_matrix = train_data.corr()


# и найдём сильно скоррелированные признаки

high_correlation_pairs = []

for column in corr_matrix.columns:
    for index in corr_matrix.index:
        if column != index:
            if abs(corr_matrix.loc[index, column]) > 0.99:
                high_correlation_pairs.append((index, column))

len(high_correlation_pairs)


# пар больше, чем столбцов, поэтому вариант "удалить по одному из двух" не сработает
# поступим иначе, используем только данные выше главной диагонали

corr_matrix = corr_matrix.abs()

# здесь я попросила чатгпт помочь мне оформить это красиво, описав ему, что именно я хочу получить
upper_part = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool) # k=1, чтобы не учитывать корреляцию с самим собой
)

high_correlation = [column for column in upper_part.columns if any(upper_part[column] > 0.99)]
len(high_correlation)


train_data = train_data.drop(columns=high_correlation)
test_data = test_data.drop(columns=high_correlation)


train_data.describe()


# меня ещё смущают очень большие значения, их тоже хочется отнормировать
# найдём нужные столбцы (var38 я увидела сама, а остальные надо поискать ещё)

description_train = train_data.describe()
large_features_train = description_train.loc['mean'] > 10000
large_features_train = large_features_train[large_features_train].index.tolist()
len(large_features_train)


description_test = test_data.describe()
large_features_test = description_test.loc['mean'] > 10000
large_features_test = large_features_test[large_features_test].index.tolist()
len(large_features_test)


large_features_train, large_features_test


train_data['saldo_var30'].describe(), test_data['saldo_var30'].describe()


# здесь наблюдается очень много странностей: огромное стандартное отклонение,
# очень маленькие и очень большие значения, квантили тоже дают странное значение
# используем StandardScaler

from sklearn.preprocessing import StandardScaler

scaler_standard = StandardScaler()
train_data['saldo_var30'] = scaler_standard.fit_transform(train_data[['saldo_var30']])


test_data['saldo_var30'] = scaler_standard.fit_transform(test_data[['saldo_var30']])


train_data['var38'].describe(), test_data['var38'].describe()


# поступаем аналогично

train_data['var38'] = scaler_standard.fit_transform(train_data[['var38']])
test_data['var38'] = scaler_standard.fit_transform(test_data[['var38']])


# с базовой предобработкой мы разобрались
# напоследок убедимся, что у нас нет NULL-значений (с этого надо было начинать, наверное...)

train_data.isnull().sum().sum(), test_data.isnull().sum().sum()


# а также проверим наличие +inf или -inf значений

(train_data == np.inf).sum().sum(), (train_data == -np.inf).sum().sum()


names = train_data.columns[(train_data == -np.inf).any()].tolist()
names


# супер, избавимся от маленьких значений

train_data = train_data.replace(-np.inf, np.nan)
train_data = train_data.fillna(train_data.mean())

test_data = test_data.replace(-np.inf, np.nan)
test_data = test_data.fillna(test_data.mean())


# переходим к CatBoost (сразу оговоримся, что валидационную выборку делать не будем,
# так как тестовая выборка и без того больше обучающей; постараемся просто взять меньше шагов,
# чтобы модель не переобучалась)

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score # как в соревновании


# выбрасываем ID и переносим таргет в отдельную переменную
train_data_y = train_data['TARGET']
train_data = train_data.drop(columns=['ID', 'TARGET'])


# в тестовой выборке удаляем ID (их сохраняем отдельно для сохранения результата)
test_data_ID = test_data['ID']
test_data = test_data.drop(columns=['ID'])


# посмотрим, есть ли у нас категориальные признаки
train_data.select_dtypes('object').columns


model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.001,
    depth=8,
    l2_leaf_reg=4,
    eval_metric='AUC',
    custom_metric=['AUC']
)

model.fit(train_data, train_data_y, verbose=100)


test_y = model.predict_proba(test_data)
test_y = test_y[:, 1]


# собираем итоговый submission.csv

result = pd.DataFrame({"ID": test_data_ID, "TARGET": test_y})
result.to_csv('submission.csv', index=False)

