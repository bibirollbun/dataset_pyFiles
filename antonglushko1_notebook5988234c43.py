import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm_notebook, tqdm
import warnings
import random
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

warnings.simplefilter('ignore')
warnings.filterwarnings("ignore")

seed = 927
random.seed(seed)
np.random.seed(seed)


def evaluate_models(models, model_names, X, Y) -> pd.DataFrame:

    """
    Функция для оценки списка моделей по метрикам MAE, MSE, RMSE, R² и MAPE.

    Параметры:
    models (list): Список обученных моделей
    model_names (list): Список названий моделей
    X (numpy.ndarray): Набор данных для предсказания, shape (n_samples, n_features)
    Y (numpy.ndarray): Истинные ответы, shape (n_samples,)
    
    Возвращает:
    pandas.DataFrame: Таблица с метриками для каждой модели
    """

    if len(models) != len(model_names):
        raise ValueError("Списки моделей и названий моделей должны быть одинаковой длины")

    results = []
    for model, name in zip(models, model_names):
        y_pred = model.predict(X)
        
        # Вычисление метрик регрессии
        mae = mean_absolute_error(Y, y_pred)
        mse = mean_squared_error(Y, y_pred)
        rmse = np.sqrt(mse)  # Корень из среднеквадратичной ошибки
        r2 = r2_score(Y, y_pred)
        
        # Средняя абсолютная процентная ошибка (MAPE)
        mape = np.mean(np.abs((Y - y_pred) / Y)) * 100 if np.all(Y != 0) else None

        results.append({
            'Model': name,
            'MAE': round(mae, 4),
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'R²': round(r2, 4),
            'MAPE': round(mape, 4)
        }) 

    results_df = pd.DataFrame(results)

    return results_df


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """ Проходит по всем столбцам DataFrame и изменяет тип данных
        для уменьшения использования памяти.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype.name


        if col_type not in ['object', 'category', 'datetime64[ns, UTC]']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


data = pd.read_csv('/kaggle/input/london-house-price-prediction-advanced-techniques/train.csv')
data


data.isnull().sum()


test = pd.read_csv('/kaggle/input/london-house-price-prediction-advanced-techniques/test.csv')
test


test.isnull().sum()


dt = []
for i in test.isnull().sum().items():
    if 0 < i[-1] <= len(test) * 0.5:
        if test[i[0]].dtype == object:
            test[i[0]] = test[i[0]].fillna(test[i[0]].mode()[0])
        else:
            test[i[0]] = test[i[0]].fillna(test[i[0]].median())
    if i[-1] > len(test) * 0.5:
        dt.append(i[0])
for i in data.isnull().sum().items():
    if 0 < i[-1] <= len(data) * 0.5:
        if data[i[0]].dtype == object:
            data[i[0]] = data[i[0]].fillna(test[i[0]].mode()[0])
        else:
            data[i[0]] = data[i[0]].fillna(test[i[0]].median())
    if i[-1] > len(data) * 0.5:
        dt.append(i[0])
data = data.drop(columns=dt)
del dt


data


data.isnull().sum()


X_train, X_valid, Y_train, Y_valid = train_test_split(
    data.drop(columns=['ID', 'price']),
    np.log10(data['price']),
    test_size=0.1,
    shuffle=True,
    random_state=seed
)


X_train.shape, X_valid.shape


model = CatBoostRegressor(
    iterations=4096*4,
    learning_rate=0.08,
    l2_leaf_reg=0.4,
    depth=8,
    task_type='GPU',
    bagging_temperature=0.5,
    border_count=128,
    use_best_model=True,
    random_state=seed,
    verbose=100
)
model.fit(
    X_train,
    Y_train,
    eval_set=(X_valid, Y_valid),
    cat_features=['postcode', 'country', 'outcode', 'tenure', 'propertyType', 'currentEnergyRating'],
    text_features=['fullAddress'],
    early_stopping_rounds=128
)


evaluate_models(
    [model],
    ['CatBoostRegressor'],
    X_valid, Y_valid
)


sub = pd.read_csv('/kaggle/input/london-house-price-prediction-advanced-techniques/sample_submission.csv')
sub


preds = model.predict(test[X_train.columns.tolist()])


sub['price'] = [10 ** i for i in preds]


sub


sub.to_csv('submission_cat.csv', index=False)




