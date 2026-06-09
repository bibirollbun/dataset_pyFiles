from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn import metrics
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from scipy import stats

from statsmodels.tsa.deterministic import DeterministicProcess
from learntools.time_series.utils import plot_periodogram, seasonal_plot

import requests
import holidays

np.random.seed(42)
simplefilter("ignore")

plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)
%config InlineBackend.figure_format = 'retina'


data_dir = Path("/kaggle/input/playground-series-s5e1")
train_raw = pd.read_csv(data_dir / "train.csv", parse_dates=["date"])
test_raw = pd.read_csv(data_dir / "test.csv", parse_dates=["date"])
sample_raw = pd.read_csv(data_dir / "sample_submission.csv")

train_raw


def optimize_types(my_df) -> pd.DataFrame:
    data = my_df.copy()

    # в category
    categorical_columns = ['country', 'store', 'product']
    for col in categorical_columns:
        data[col] = data[col].astype('category')

    # уменьшаем занимаемый размер
    try:
        data['num_sold'] = data['num_sold'].astype('float32')
    except:
        pass
    
    return data


def show_memory_optimization(original_data: pd.DataFrame, optimized_data: pd.DataFrame):
    original_memory = original_data.memory_usage(deep=True).sum() / (1024 ** 2)  # MB
    optimized_memory = optimized_data.memory_usage(deep=True).sum() / (1024 ** 2)  # MB
    memory_reduction = original_memory - optimized_memory
    reduction_percentage = (memory_reduction / original_memory) * 100
    
    print(f"Original Memory Used: {original_memory:.2f} MB")
    print(f"Optimized Memory Used: {optimized_memory:.2f} MB")
    print(f"Memory reduction: {memory_reduction:.2f} MB ({reduction_percentage:.2f}%)")
    
    return {
        "original_memory_mb": original_memory,
        "optimized_memory_mb": optimized_memory,
        "memory_reduction_mb": memory_reduction,
        "reduction_percentage": reduction_percentage,
    }


train_df = train_raw.dropna()
test_df = test_raw.copy()


df_raw = pd.concat([train_df, test_df], axis=0)
df = optimize_types(df_raw)
show_memory_optimization(df_raw, df)


class CFG:
    years_train = train_df.date.dt.year.unique()
    years_test = test_df.date.dt.year.unique()    
    years = np.concatenate((train_df.date.dt.year.unique(), test_df.date.dt.year.unique()))
    
    validation_year = 2016
    
    countries = train_df['country'].unique()
    stores = train_df['store'].unique()
    products = train_df['product'].unique()

    alpha3 = {'Finland': 'FIN', 'Canada': 'CAN', 'Italy': 'IT', 'Kenya': 'KEN', 'Singapore': 'SGP', 'Norway': 'NOR'}
    fft_filter_width = 8

    countries_2l = {'Finland': 'FI', 'Canada': 'CA', 'Italy': 'IT', 'Kenya': 'KE', 'Singapore': 'SG', 'Norway': 'NO'}
    holiday_response_len = 10


train_start_date = dt.datetime(2010, 1, 1)
train_end_date = dt.datetime(2016, 12, 31)
test_start_date = dt.datetime(2017, 1, 1)
test_end_date = dt.datetime(2019, 12, 31)


# Базовые признаки из даты
df['year'] = df['date'].dt.year
df['weekday'] = df['date'].dt.weekday
df['dayofyear'] = df['date'].dt.dayofyear
df['daynum'] = (df.date - df.date.iloc[0]).dt.days
df['weeknum'] = df['daynum'] // 7
df['month'] = df.date.dt.month


def get_season_onehot(month):
    if month in [12, 1, 2]:
        return [1, 0, 0, 0]  # Winter
    elif month in [3, 4, 5]:
        return [0, 1, 0, 0]  # Spring
    elif month in [6, 7, 8]:
        return [0, 0, 1, 0]  # Summer
    else:
        return [0, 0, 0, 1]  # Autumn

# seasons_onehot = df['month'].apply(get_season_onehot).tolist()
# df[['winter', 'spring', 'summer', 'autumn']] = pd.DataFrame(seasons_onehot, index=df.index)


# Лаг
# df['rolling_mean_week'] = df['num_sold'].rolling(window=7).mean()
# df['rolling_mean_month'] = df['num_sold'].rolling(window=30).mean()
# df['rolling_sum_week'] = df['num_sold'].rolling(window=7).sum()
# df['rolling_sum_month'] = df['num_sold'].rolling(window=30).sum()


# Тригонометрические преобразования дат

# Нормальное число дней в году
daysinyear = (df.groupby('year').id.count() / len(CFG.countries) / len(CFG.stores) / len(CFG.products)).rename('daysinyear').astype(int).to_frame()
df = df.join(daysinyear, on='year', how='left')

# Нормальный номер дня в году: 
# Для годичного цикла: от 0 до 1 (0 - 1 января, 1 - 31 декабря)
# Для двухгодичного цикла: от 0 до 2 (0 - 1 января четного года, 2 - 31 декабря нечетного)
# Примечание: если датасет начинается с нечетного года, то во 2 строке надо еще прибавить 1
df['partofyear'] = (df['dayofyear'] - 1) / df['daysinyear']
df['partof2year'] = df['partofyear'] + df['year'] % 2 

# Просто записываем названия столбцов
CFG.sincoscol = [f'sin t', f'cos t', f'sin t/2', f'cos t/2']
CFG.sincoscol2 = [f'sin 2t', f'cos 2t', f'sin t', f'cos t', f'sin t/2', f'cos t/2']

# Простейшая тригонометрия - вспомни уроки с Лейсан
df['sin 4t'] = np.sin(8 * np.pi * df['partofyear']) # полный цикл занимает 1/4 года - период: 3 месяца
df['cos 4t'] = np.cos(8 * np.pi * df['partofyear']) # полный цикл занимает 1/4 года - период: 3 месяца
df['sin 3t'] = np.sin(6 * np.pi * df['partofyear']) # полный цикл занимает 1/3 года - период: 4 месяца
df['cos 3t'] = np.cos(6 * np.pi * df['partofyear']) # полный цикл занимает 1/3 года - период: 4 месяца
df['sin 2t'] = np.sin(4 * np.pi * df['partofyear']) # полный цикл занимает 1/2 года - период: 6 месяцев
df['cos 2t'] = np.cos(4 * np.pi * df['partofyear']) # полный цикл занимает 1/2 года - период: 6 месяцев
df['sin t'] = np.sin(2 * np.pi * df['partofyear'])  # полный цикл занимает 1 год
df['cos t'] = np.cos(2 * np.pi * df['partofyear'])  # полный цикл занимает 1 год
df['sin t/2'] = np.sin(np.pi * df['partof2year'])   # полный цикл занимает 2 года
df['cos t/2'] = np.cos(np.pi * df['partof2year'])   # полный цикл занимает 2 года
df.drop(['daysinyear', 'partofyear', 'partof2year'], axis=1, inplace=True)


# Тут просто - получаем ВВП на душу населения для данной страны и года
def get_gdp_per_capita(country, year):
    url="https://api.worldbank.org/v2/country/{0}/indicator/NY.GDP.PCAP.CD?date={1}&format=json".format(CFG.alpha3[country],year)
    response = requests.get(url).json()
    return response[1][0]['value']

# Двумерный массив с данными о ВВП на душу населения
# Строки - страны, столбцы - года, на пересечении - соответствующий ВВП 
gdp = np.array([[
    get_gdp_per_capita(country, year) 
        for year in CFG.years] 
                for country in CFG.countries
               ])
gdp_df = pd.DataFrame(gdp, index=train_df.country.unique(), columns=CFG.years)

df['gdp_factor'] = None
for year in CFG.years:
    for country in CFG.countries:
        df.loc[ # читать как "для всех строк с текущим годом и страной, столбец gdp_factor"
            (df.country == country) & (df.year == year), 
            'gdp_factor'
        ] = gdp_df.loc[country, year] # попробовать работать тоже с нормальными значениями


# Мы не учитываем Канаду и Кению потому что они содержат пропуски
df_no_can_ken = df[~df.country.isin(('Canada', 'Kenya'))]

# Берем среднее значение продаж в день и записываем в 'store_factor'
store_df = df_no_can_ken.groupby('store').num_sold.mean().rename('store_factor').to_frame()
df = df.drop('store_factor', axis=1, errors='ignore').join(store_df, on='store', how='left')


# Мы не учитываем Канаду и Кению потому что они содержат пропуски
df_no_can_ken = df[~df.country.isin(('Canada', 'Kenya'))]

# Берем сумму продаж за день и записываем в 'num_sold_total'
total = df_no_can_ken.groupby(by='date').num_sold.sum().rename('num_sold_total')

# Теперь каждая строка знает общий объем продаж в этот день
df_no_can_ken = df_no_can_ken.join(total, on='date', how='left') 

# Доля для данной комбинации store/product/country от общей суммы продаж
df_no_can_ken['num_sold_ratio'] = df_no_can_ken['num_sold'] / df_no_can_ken['num_sold_total']

plt.figure(figsize=(24, 6))
df['product_factor'] = None
for product in CFG.products:
    # Продажи только данного продукта в этот день
    df_no_can_ken_date = df_no_can_ken[
        (df_no_can_ken['product'] == product) & 
        (df_no_can_ken['date'] <= train_end_date)
    ].groupby(by='date')

    # Среднее значение sin/cos в этот день для данного продукта
    x = df_no_can_ken_date[CFG.sincoscol].mean().to_numpy() 

    # Общая доля продаж в этот день для данного продукта
    y = df_no_can_ken_date.num_sold_ratio.sum().to_numpy()

    # Обучаем модель для прогнозирования общей доли продаж этого продукта в этот день
    reg = Ridge()
    reg.fit(x, y)

    # Для каждой строки (тест + трен) записывает предсказание общей доли продаж 
    # этого продукта в этот день
    df.loc[
        (df['product'] == product), 
        'product_factor'
    ] = reg.predict(df.loc[(df['product'] == product), CFG.sincoscol].to_numpy())

    p = reg.predict(x)
    plt.plot(y, 'b') # синие - истинные
    plt.plot(p, 'r') # красные - предсказанные
    
plt.show();


df['holiday'] = 0
for country in CFG.countries:
    
    # Найти все праздничные дни для этой страны за 2010-2019
    days = [str(day) for day in 
            holidays.CountryHoliday(CFG.countries_2l[country], years=CFG.years)]
    
    # Отметить праздники единичкой
    df.loc[(df.country==country) & (df.date.isin(days)), 'holiday'] = 1

# Таблица следующего вида:
# строки - (weeknum, country), столбцы - weekday,
# на пересечении - сумма продаж 
num_sold_per_week_country_weekday = df.groupby(['weeknum', 'country', 'weekday'])['num_sold'].sum().reset_index().pivot(index=['weeknum', 'country'], columns='weekday')

# Доля продаж относительно общенедельной суммы продаж
ratio_sold_per_week_country_weekday = num_sold_per_week_country_weekday.apply(lambda row: row/sum(row), axis=1).reset_index()

# Медианное (среднее) значение продаж в этот день недели для этой страны
ratio_weekday = pd.DataFrame(columns=CFG.countries, data=[[0, ]*len(CFG.countries)]*7)
for n, country in enumerate(CFG.countries):
    for d in range(7):
        dt = ratio_sold_per_week_country_weekday.loc[
            ratio_sold_per_week_country_weekday.country == country, 
            ('num_sold', d)
        ][:-60] # пока что не берем в расчет новый год
        ratio_weekday.loc[d, country] = dt.median()

# Медианное (среднее) значение продаж в этот день недели в общем для всех стран
ratio_weekday_mean = ratio_weekday.mean(axis=1)
ratio_weekday['mean'] = ratio_weekday_mean

# Теперь все строки знают о средних продажах по всему миру в этот день недели
df['weekday_factor'] = df.weekday.map(ratio_weekday_mean)

# Финальное преобразование всех факторов
# (помним, про то, что линейная регрессия не знает таких сложных функций)
df['ratio'] = df['gdp_factor'] * df['product_factor'] * df['store_factor'] * df['weekday_factor']

# Нормализация суммы продаж - записываем под один знаменатель, "под одну черту всех"
df['total'] = df['num_sold'] / df['ratio']


# Отметим некоторое кол-во дней после праздника
df_holidays = df.copy()
df_holidays['holiday_response'] = 0
for country in CFG.countries:
    for holiday, _ in holidays.CountryHoliday(CFG.countries_2l[country], years=CFG.years).items():
        df_holidays.loc[
            (df_holidays.country==country) & 
             df_holidays.date.isin(
                 pd.date_range(holiday, periods=CFG.holiday_response_len)), 
            'holiday_response'
        ] = 1

# Для каждой страны в data записываем медианные (средние) нормальные продажи в будни
fig = plt.figure(figsize=(24,6))
data = pd.DataFrame()
for n, country in enumerate(CFG.countries):
    dt = df_holidays[
        (df_holidays.country==country) & (df_holidays.holiday_response == 0)
    ].groupby(['dayofyear']).total.median()
    data[country]= dt
    plt.plot(dt, label=country)

# Медианные нормальные продажи в будни по всему миру
data['median'] = data.median(axis=1) 

# X - просто номер, y - медианные нормальные продажи в будни по всему миру
x = data.index.to_numpy()
y = data['median'].to_numpy()

# Лямбда-функция Фурье: 
# Для каждого дня высчитывает значение sin/cos для периода 365 дней
fourier = lambda t: np.array([np.sin(2*np.pi/365*t), np.cos(2*np.pi/365*t)])

# Модель учится находить связь между днем в году и медианными продажами
year_ratio = Ridge(alpha=0.01).fit(fourier(x).T, y.T).predict(fourier(np.arange(1, 366)).T)

# Для високосных лет, добавим еще один день с первым попавшимся признаком
# Теперь каждая строка знает медианный уровень продаж для данного дня года
year_ratio = np.append(year_ratio, year_ratio[-1])
df['dayofyear_factor'] = df.dayofyear.map(dict(zip(np.arange(1, 367), year_ratio)))

# Финальное преобразование всех факторов
# (помним, про то, что линейная регрессия не знает таких сложных функций)
df['ratio'] = df['gdp_factor'] * df['product_factor'] * df['store_factor'] * df['weekday_factor'] * df['dayofyear_factor']

# Нормализация суммы продаж - записываем под один знаменатель, "под одну черту всех"
df['total'] = df['num_sold'] / df['ratio']

plt.plot(year_ratio, 'k', linewidth=4)
plt.legend();


# Этот код просто визуализирует тренд на больших периодах
# Сам автор предлагает использовать это для докрутки моделек Фурье
fig = plt.figure(figsize=(24,6))
data = pd.DataFrame()
for n, country in enumerate(CFG.countries):
    dt = df_holidays[(df_holidays['date'] <= train_end_date) & (df_holidays['country']==country) & (df_holidays['holiday_response'] == 0)].groupby(['date']).total.median()
    data[country]= dt
    plt.plot(dt, label=country)
data['median'] = data.median(axis=1)


CFG.sincoscol2 = ['sin 4t', 'cos 4t', 'sin 3t', 'cos 3t', 'sin 2t', 'cos 2t', 'sin t', 'cos t', 'sin t/2', 'cos t/2']

# Linear regression on fourier series
dfsc = df[df['date'] <= train_end_date].groupby('date')[CFG.sincoscol2].mean()#.to_numpy()
dfsc['median'] = data['median']

x = dfsc[~pd.isna(dfsc['median'])][CFG.sincoscol2].to_numpy()
y = dfsc[~pd.isna(dfsc['median'])]['median'].to_numpy()

reg = Ridge(alpha=0.01, fit_intercept=True)
reg.fit(x, y)

fig = plt.figure(figsize=(24,6))
plt.plot(y, 'k')
plt.plot(reg.predict(x), 'r')

df['sincos_factor'] = reg.intercept_ + (df[CFG.sincoscol2] * reg.coef_).sum(axis=1)


# Финальное преобразование всех факторов
# (помним, про то, что линейная регрессия не знает таких сложных функций)
df['ratio'] = df['gdp_factor'] * df['product_factor'] * df['store_factor'] * df['weekday_factor'] * df['sincos_factor']

# Нормализация суммы продаж - записываем под один знаменатель, "под одну черту всех"
df['total'] = df['num_sold'] / df['ratio']

fig = plt.figure(figsize=(24,6))
for c in CFG.countries:
    df_p = df[(df.country == c) & (df['product'] == 'Kaggle')].groupby('date').total.sum().to_numpy()
    plt.plot(df_p, label=c)

plt.legend();


# Вводим доп.признак - показатель страны
# Чтобы обрабатывать Кенийцев отдельно
country_factor = df[(df['product'] == 'Kaggle')].groupby('country').total.sum().rename('country_factor')
country_factor = country_factor / country_factor.median()
df = df.join(country_factor, on='country', how='left')


# Вводим новый признак - относится ли запись к новогоднему периоду
df['is_ny_holiday'] = 0
df['is_ny_holiday'] = df['date'].apply(lambda x: 1 if (x.month >= 11 and x.day >= 1) or (x.month == 1 and x.day <= 31) else 0)
ny_holiday_df = df[(df['is_ny_holiday'] == 1) & (df['year'] <= 2016)].copy()

# Таблица следующего вида:
# строки - (year, country), столбцы - new year holiday day,
# на пересечении - сумма продаж 
ny_holiday_df['day_period'] = ny_holiday_df['date'].dt.strftime('%m-%d')
grouped_df = ny_holiday_df.groupby(['year', 'country', 'day_period'])['num_sold'].sum().reset_index()
pivoted_df = grouped_df.pivot(index=['year', 'country'], columns='day_period', values='num_sold')

# Доля продаж относительно общего новогоднего периода
relative_sales_df = pivoted_df.div(pivoted_df.sum(axis=1), axis=0)

# Медианное (среднее) значение продаж в этот день для этой страны
median_sales_per_country = relative_sales_df.groupby('country').median()

# Медианное (среднее) значение продаж в этот день в общем для всех стран
median_sales_worldwide = median_sales_per_country.median(axis=0)

# Функция возвращает минимальное значение, если она не относится к новогоднему периоду
# и медианную относительную сумму продаж по всему миру в этот день, если относится  
def calculate_newyear_factor(row, median_sales_worldwide):
    if row['is_ny_holiday'] == 1:
        day_period = row['date'].strftime('%m-%d')
        return median_sales_worldwide.get(day_period, 1)
    return median_sales_worldwide.min()

df['newyear_factor'] = df.apply(calculate_newyear_factor, axis=1, args=(median_sales_worldwide,))






df['ratio'] = df['gdp_factor'] * df['product_factor'] * df['store_factor'] * df['weekday_factor'] * df['sincos_factor'] * df['country_factor'] * df['newyear_factor']
# df['ratio'] = df['gdp_factor'] * df['product_factor'] * df['store_factor'] * df['weekday_factor'] * df['sincos_factor'] * df['country_factor']
df['total'] = df['num_sold'] / df['ratio']

plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['total'], marker='o', linestyle='-', label='Total over time')
plt.xlabel('Date')
plt.ylabel('Total')
plt.title('Dependency of Total on Date')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)  # Поворот меток на оси X для читаемости
plt.tight_layout()  # Подгонка элементов графика
plt.show()


const_factor = df['total'].median() * 1.06

df['prediction'] = const_factor * df['ratio']
df_true = df[(df['date'] <= train_end_date) & (~pd.isna(df['num_sold']))]
df_pred = df[(df['date'] <= train_end_date) & (~pd.isna(df['num_sold']))]

mape_train = mean_absolute_percentage_error(df_true['num_sold'], df_pred['prediction'])

print(f'{mape_train=}')


submission_number = 2
submission = df[(df['date'] > train_end_date)][['id', 'prediction']].rename(columns={'prediction': 'num_sold'})
submission.to_csv(f'submission_{submission_number}.csv', index=False)


alphas = np.logspace(-3, 3, 50)

ols_model = LinearRegression(fit_intercept=False)
l2_model = RidgeCV(alphas=alphas, fit_intercept=False)
l1_model = LassoCV(alphas=None, random_state=42, max_iter=1000000, fit_intercept=False)
elastic_model = ElasticNetCV(
    alphas=None, 
    l1_ratio=[.1, .5, .7, .9, 1],
    random_state=42, 
    max_iter=1000000,
    fit_intercept=False
)


def FPD_and_visualize(model_name, model, train_df, test_df, features, target):
    tscv = TimeSeriesSplit(n_splits=5)
    source = train_df[features]
    target_series = train_df[target]

    mape_scores = []
    mse_scores = []
    mae_scores = []
    rmse_scores = []
    models = []

    fold = 1
    for train_index, valid_index in tscv.split(train_df[features]):
        X_train, X_valid = source.iloc[train_index], source.iloc[valid_index]
        y_train, y_valid = target_series.iloc[train_index], target_series.iloc[valid_index]

        model.fit(X_train, y_train)
        models.append(model)
        y_pred = model.predict(X_valid)

        mape = mean_absolute_percentage_error(y_valid, y_pred)
        mse = mean_squared_error(y_valid, y_pred)
        mae = mean_absolute_error(y_valid, y_pred)
        rmse = np.sqrt(mse)

        print(f"Fold {fold}:\nMAPE: {mape:.5%}\nMSE: {mse:.3f}\nMAE: {mae:.3f}\nRMSE: {rmse:.3f}\n")

        mape_scores.append(mape)
        mse_scores.append(mse)
        mae_scores.append(mae)
        rmse_scores.append(rmse)

        print("-" * 50)
        fold += 1

    best_index = np.argmin(mape_scores)
    best_model = models[best_index]
    best_mape = mape_scores[best_index]

    print(f"Best model ({model_name} fold {best_index + 1}) with MAPE = {best_mape:.2%}")

    coef = pd.DataFrame(best_model.coef_, columns=['coef'])
    coef['column'] = features
    coef = coef.set_index('column')
    coef = coef[coef['coef'].abs() > 1e-5]  # Не показывать около нулевые коэф-ты
    coef_sorted = coef.reindex(coef['coef'].abs().sort_values(ascending=False).index)

    # countries = result['country'].unique()
    # stores = result['store'].unique()
    # products = result['product'].unique()

    # for country in countries:
    #     result_country = result[result['country'] == country]
    #     n_rows = len(stores)
    #     n_cols = len(products)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Coefficients plot
    axes[0].bar(coef_sorted.index, coef_sorted['coef'], color='skyblue')
    axes[0].set_ylabel('Coefficient Value')
    axes[0].set_title(f'{model_name} Coefficients')
    axes[0].tick_params(axis='x', rotation=45)

    #     # Forecast visualization
    #     fig_forecast, axes_forecast = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), sharex=True, sharey=True)
    #     fig_forecast.suptitle(country, fontsize=16)

    #     for i, store in enumerate(stores):
    #         for j, product in enumerate(products):
    #             ax = axes_forecast[i, j]
    #             view = result_country[(result_country['store'] == store) & (result_country['product'] == product)]

    #             if view.empty:
    #                 ax.text(0.5, 0.5, "No Data", horizontalalignment='center', verticalalignment='center')
    #                 ax.set_title(f"{store} | {product}")
    #                 ax.set_xlabel("time_no")
    #                 ax.set_ylabel("num_sold")
    #                 continue

    #             ax.plot(view['time_no'], view['num_sold'], color='red', label='Actual')
    #             ax.plot(view['time_no'], view[target], color='blue', label='Predicted')
    #             ax.set_title(f"{store} | {product}")
    #             ax.set_xlabel("time_no")
    #             ax.set_ylabel("num_sold")
    #             ax.set_xlim(view['time_no'].min(), view['time_no'].max())

    #             if i == 0 and j == 0:
    #                 ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    #     plt.tight_layout()
    #     plt.show()

    return best_model


