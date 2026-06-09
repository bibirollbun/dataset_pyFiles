import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error as mse
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error as mse
seed = 0
random.seed(seed)
np.random.seed(seed)


ts = pd.read_csv("/kaggle/input/rossmann-store-sales/train.csv")


ts


train = ts[ts['Store'] == 1]



train


train.info()


train['Date'] = pd.to_datetime(train['Date'])


train.set_index('Date', inplace=True)


train.info()


train = train[::-1].asfreq(pd.infer_freq(train[::-1].index))


train


train.loc[train[train['Open'] == 0].index, 'Sales'] = np.nan


train


train['Sales'] = train['Sales'].fillna(method='ffill')


train.info()


train


train = train.dropna()


train


train, test = train.loc[:'2015-07-20'], train.loc['2015-07-21':]


train.shape[0], test.shape[0]


plt.figure(figsize=(20,10))
plt.plot(train['Sales'])


# "Выбросы" соответствуют рождественским закупкам
train[train['Sales'] >= 9000]


def test_stationarity(timeseries):
    adf_result = adfuller(timeseries)
    print('ADF Statistic: %f' % adf_result[0])
    print('p-value: %f' % adf_result[1])
    kpss_stat, p_value, _, critical_values = kpss(timeseries, regression='c')
    print('\nKPSS Statistic: %f' % kpss_stat)
    print('p-value: %f' % p_value)


#Тест ADF имеет нулевую гипотезу, что ряд нестационарен. КПСС наоборот — стационарен
#При p_value < 0.05 отклоняем нулевую гипотезу.
#Для ADF p_value = 0 => Отклоняем H0 => ряд стационарен
#Для ADF p_value = 0.1 => Принимаем H0 => ряд стационарен
test_stationarity(train['Sales'])


#Попробуем определить сезонность данных с помощью автокорреляционной функции
#Для этого продифференцируем ряд (текущий - предыдущий уровень ряда)
#Затем построим для полученного ряда ACF
trend = train['Sales'].diff().dropna()


np.mean(trend)


plt.figure(figsize=(10,10))
plot_acf(trend, lags=100, ax=plt.subplot(111), zero=False)
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.grid()
plt.show()


#Из графика выше видно, что сезонность равна 14 дням.
#Также не видно никакого возрастающего тренда и увеличения сезонности поэтому смотрим аддитивную модель
dec_add = seasonal_decompose(train['Sales'], model='additive', period = 14)


trend_add = dec_add.trend
seasonal_add = dec_add.seasonal
residuals_add = dec_add.resid


plt.figure(figsize=(20,16))
plt.subplot(411)
plt.plot(train['Sales'])
plt.title('Исходный ряд')
plt.subplot(412)
plt.plot(trend_add)
plt.title('Тренд')
plt.subplot(413)
plt.plot(seasonal_add)
plt.title('Сезонность')
plt.subplot(414)
plt.scatter(x = residuals_add.index, y = residuals_add)
plt.title('Остатки')
plt.subplot(414)
plt.axhline(y=0, c='r', lw=3, ls='--')


sns.histplot(residuals_add, kde=True, bins=int(1+3.332*np.log10(train.shape[0])))


#Перейдём к определению параметров для SARIMAX-модели


plt.figure(figsize=(20,5))
plot_acf(train['Sales'], lags=40, ax=plt.subplot(121), zero=False)
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.grid()
plot_pacf(train['Sales'], lags=40, ax=plt.subplot(122), zero=False)
plt.title('Partial Autocorrelation Function (PACF)')
plt.xlabel('Lag')
plt.ylabel('PACF')
plt.grid()
plt.show()


p,d,q = 3,0,5


S = 14


#В декомпозиции явно видно, что сезонность имеет стабильный паттерн, поэтому D>=1
#Правилов выбора D: d+D<=2
D = 1


#Правило выбора P и Q следующее: P>=1 при PACF(lag=S) >0 и выше уровня важности else 0
#Q аналогично. При этом P+Q<=2
P, Q = 1, 1


def sarima(p,d,q,P,D,Q,S, exog_train=None, exog_test=None):
    model = SARIMAX(endog = train['Sales'], exog = exog_train, order=(p,d,q), seasonal_order = (P,D,Q,S))
    model_fit = model.fit()
    print(model_fit.summary())
    forecast_on_test = model_fit.forecast(steps=len(test), exog = exog_test)
    rmse = np.sqrt(mse(test['Sales'], forecast_on_test))
    print(f'AIC = {model_fit.aic}\nRMSE = {rmse}')
    plt.figure(figsize = (20, 10))
    plt.plot(train['Sales'], label='Train_Actual')
    plt.plot(model_fit.predict(), label = 'Model_predict')
    plt.legend()
    # forecast_index = pd.date_range(start=train.index[-1] + \
    #                                pd.DateOffset(days=1), periods=len(test), freq='D')
    plt.show()
    plt.plot(test['Sales'], label='Test_Actual')
    plt.plot(forecast_on_test, label = 'Model_predict')
    plt.legend()
    plt.show()
    model_fit.plot_diagnostics(figsize=(14,7))
    plt.show()


sarima(p,d,q,P,D,Q,S,train.drop(columns=['Sales','StateHoliday','Store']),
                                test.drop(columns=['Sales','StateHoliday','Store']))


sarima(p,1,q,P,D,Q,S)


sarima(p,d,q,1,D,0,S)


#Можно поперебирать параметры, чтобы AIC и MSE были минимальны
#Также те параметры, где столбец P > |z| равен 0 можно попробовать убрать из модели


#Также нужно попробовать переписать обучение модели на окна размером 30 (930 данных хорошо делятся на 30)

