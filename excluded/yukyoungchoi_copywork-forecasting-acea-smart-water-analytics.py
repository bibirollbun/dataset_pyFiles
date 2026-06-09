import os
import numpy as np
import pandas as pd
import matplotlib as mpl 
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


data_description_table = "/kaggle/input/acea-water-prediction/datasets_description.xlsx"
excel_data = pd.read_excel(data_description_table, sheet_name=None) #딕셔너리로 읽어오게 됨

# print(excel_data.keys()) # 내가 불러온 엑셀 파일의 탭 이름
# print(excel_data.values()) # 내가 불러온 엑셀 파일 개별 탭에 저장된 시트 데이터 (DF로 변환해야 함)

# 각 DataFrame을 개별 변수로 할당
for sheet_name, df in excel_data.items():
    globals()[sheet_name] = df

display(Datasets_Description. head())
display(Datasets_Feature_Description.head())
display(Dataset_Feature_Value.head())

## 이런 류의 지침 파일은 내가 직접 열어서 확인하는 게 제일 나은 듯... ㅎ


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import missingno as msno    # It is the library for plotting the missing number values in each column
# 결측치를 분석할 때, isnull로 직접 계산하는 방법도 있지만 missingno를 사용하면 더욱 빠르다는 점.
import os
import warnings
warnings.filterwarnings('ignore')  # This removes all the warnings from the output

from datetime import datetime, date

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# 대수층 

Petrignano = pd.read_csv("/kaggle/input/acea-water-prediction/Aquifer_Petrignano.csv")
print(Petrignano.shape)
Petrignano.head()


Petrignano.isnull().sum()


# Here we are dropping the previous the keeping nearly 10 years of data
Petrignano = Petrignano[Petrignano.Rainfall_Bastia_Umbra.notna()].reset_index(drop = True)

# notna() returns True is value is not null
# when a new dataframe is created after filtering the null values 
# it contains the column index with previous indexes
# reset_index(drop = True) helps in dropping that extra index column from dataframe

Petrignano = Petrignano.drop(['Depth_to_Groundwater_P24', 'Temperature_Petrignano'], axis = 1)
Petrignano.columns = ['Date', 'Rainfall', 'Depth_to_groundwater', 'Temperature', 'Volume', 'Hydrometry']
Petrignano['Date'] = pd.to_datetime(Petrignano['Date'], format = '%d/%m/%Y')

print(Petrignano.shape)
Petrignano.head()


features = Petrignano.drop(['Depth_to_groundwater'], axis = 1)
target = [Petrignano['Depth_to_groundwater']]

display(features)
display(target)


fig, ax = plt.subplots(5, 1, figsize=(10, 10))

# .columns: 제거된 데이터프레임의 열 이름들을 가져온다.
# enumerate(...): 열 이름들과 그 인덱스(i)를 함께 반복합니다.
# for i, col in ...: 각 열 이름(col)과 그 인덱스(i)에 대해 반복문을 실행
# 인덱스가 먼저 나온다는 점 주의하기!!!!!

for i, col in enumerate(Petrignano.drop(['Date'], axis = 1).columns):
    
    sns.lineplot(x = Petrignano['Date'], y = Petrignano[col].fillna(method = 'ffill'), ax = ax[i], color = 'green')
    
    # fillna() function fills the NaN values using ffill method
    # ffill method replace NaN with last valid observation

    ax[i].set_ylabel(col, fontsize = 7)
    
ax[4].set_xlabel('Date')

plt.tight_layout()

# 강수량을 보면 일정 주기에 따라 상승/하강하는 패턴이 나타난다.
# 이를 더 자세히 보기 위해서는 x = 월, y = 월별 집계 데이터에 대한 다중 그래프를 사용할 수 있을 듯
# GroundWater에서는 2013년 경 급격하게 감소한 것이 보인다.
# 이때 무슨 일이 있었을까? 특이점이 나타나는 구간 위주로 분석할 수도 있겠다.

# 2015년 말, Hydrometry가 급격하게 감소한 구간이 나타났다. (0)


sns.pairplot(Petrignano)


## Petrignano['Date'].shift(1)은 'Date' 열의 각 값을 한 행씩 아래로 이동시킴. 따라서 첫 번째 행은 Na
## Petrignano['Date'] - Petrignano['Date'].shift(1)은 현재 행의 날짜에서 바로 이전 행의 날짜를 뺄셈
## 결과적으로 'Interval' 열은 연속적인 날짜 사이의 시간 간격을 나타냄
## 예를 들어, 두 번째 행의 'Interval' 값은 두 번째 날짜에서 첫 번째 날짜를 뺀 시간 차이

Petrignano = Petrignano.sort_values(by='Date')

Petrignano['Interval'] = Petrignano['Date'] - Petrignano['Date'].shift(1)
# shift() function shifts the index by one

print('Sum of all interval between dates: ', Petrignano['Interval'].sum())
print('Count number of rows: ', Petrignano['Interval'].count())

Petrignano = Petrignano.drop(['Interval'], axis=1)

print('Conclusion: It is clear that sum of intervals is equal to the number of rows which means dates are in chronological order.')


Petrignano.isnull().sum()


Petrignano['Volume'] = Petrignano['Volume'].replace(0, np.nan)
Petrignano['Hydrometry'] = Petrignano['Hydrometry'].replace(0, np.nan)

msno.matrix(Petrignano)


"""
첫 번째 서브플롯: Petrignano['Volume'].fillna(0)은 결측치를 0으로 채움
두 번째 서브플롯: Petrignano['Volume'].fillna(mean_val)은 결측치를 평균값으로 채움
세 번째 서브플롯: Petrignano['Volume'].ffill()은 결측치를 이전 값으로 채움(forward fill).
네 번째 서브플롯: Petrignano['Volume'].interpolate()는 결측치를 선형 보간으로 채움

"""


fig, ax = plt.subplots (4,1, figsize = (15, 12))

sns.lineplot(x = Petrignano['Date'], y = Petrignano['Volume'].fillna(0), ax = ax[0], color = 'green', label = 'modified')

# Petrignano['Volume'].isna().cumsum()는 결측치가 나타날 때마다 다른 색상으로 구분하여 표시함
sns.lineplot(x = Petrignano['Date'], y = Petrignano['Volume'], hue=Petrignano["Volume"].isna().cumsum(), ax = ax[0], 
             palette=["blue"]*sum(Petrignano["Volume"].isna()), label = 'original', legend = False)

# hue argument can be used to put the separate sections in separate buckets. 
# Though it is faster and easy to read but an outlier in the Petrignano which is 
# surrounded by None will not be drawn on the chart.
# palette color is set to blue for every section/bucket.


### 만약 .cumsum() 없이 단순히 hue=Petrignano["Volume"].isna()를 사용하면
## 연속되지 않은 단일 NaN 값은 별도의 그룹으로 인식되지 않고 그래프에서 누락될 수 있음
#.cumsum()을 사용하면 이러한 단일 NaN 값도 그룹에 포함되어 그래프에 표시됨

ax[0].set_title('Fill Nan with 0', fontsize = 14)
ax[0].set_ylabel(ylabel = 'Volume', fontsize = 14)

mean_val = Petrignano['Volume'].mean()
sns.lineplot(x = Petrignano['Date'], y = Petrignano['Volume'].fillna(mean_val), ax = ax[1], color = 'green', label = 'modified')
sns.lineplot(x = Petrignano['Date'], y = Petrignano['Volume'], hue=Petrignano["Volume"].isna().cumsum(), ax = ax[1], 
             palette=["blue"]*sum(Petrignano["Volume"].isna()), label = 'original', legend = False)
ax[1].set_title(f'Fill Nan with mean: {mean_val}', fontsize = 14)
ax[1].set_ylabel(ylabel = 'Volume', fontsize = 14)


sns.lineplot(x = Petrignano['Date'], y = Petrignano['Volume'].ffill(), ax = ax[2], color = 'green', label = 'modified')
sns.lineplot(x = Petrignano['Date'], y = Petrignano['Volume'], hue=Petrignano["Volume"].isna().cumsum(), ax = ax[2], 
             palette=["blue"]*sum(Petrignano["Volume"].isna()), label = 'original', legend = False)
ax[2].set_title('Fill Nan using ffill', fontsize = 14)
ax[2].set_ylabel(ylabel = 'Volume', fontsize = 14)


sns.lineplot(x = Petrignano['Date'], y = Petrignano['Volume'].interpolate(), ax = ax[3], color = 'green', label = 'modified')
sns.lineplot(x = Petrignano['Date'], y = Petrignano['Volume'], hue=Petrignano["Volume"].isna().cumsum(), ax = ax[3], 
             palette=["blue"]*sum(Petrignano["Volume"].isna()), label = 'original', legend = False)
ax[3].set_title('Fill Nan using linear interplotation', fontsize = 14)
ax[3].set_ylabel(ylabel = 'Volume', fontsize = 14)

for i in range(4):
    ax[i].set_xlim([date(2019, 5, 1), date(2019, 10, 1)])
    
plt.tight_layout()  
# tight_layout automatically adjusts subplot params so that the subplot(s) fits in to the figure area. 
plt.show()


# interpolate() 함수를 사용하여 결측치(NaN)를 선형 보간으로 채우기
# 선형 보간은 결측치 앞뒤의 값을 사용하여 그 사이를 선형적으로 추정하는 방식

"""
[선형 보간의 작동 방식]
결측치 주변 값 찾기: 결측치를 기준으로 바로 앞과 바로 뒤의 유효한 데이터 값을 찾음
직선으로 연결: 찾은 두 값을 직선으로 연결함
결측치 추정: 결측치의 위치에 해당하는 직선 위의 값을 결측치의 추정값으로 사용함
"""

Petrignano['Volume'] = Petrignano['Volume'].interpolate()
Petrignano['Hydrometry'] = Petrignano['Hydrometry'].interpolate()
Petrignano['Depth_to_groundwater'] = Petrignano['Depth_to_groundwater'].interpolate()


"""resample() 함수의 작동 방식

시간 기반 그룹화: resample()은 시간 기반 인덱스 또는 날짜/시간 열을 사용하여 데이터를 그룹화
빈도 설정: 원하는 재표본추출 빈도를 지정.
    예를 들어, 일별 데이터에서 월별 데이터로 변경하거나,
    분별 데이터에서 시간별 데이터로 변경할 수 있음
집계 함수 적용: 각 그룹에 대해 집계 함수(예: 평균, 합계, 개수)를 적용하여 재표본추출된 데이터를 생성

"""

fig, ax= plt.subplots(nrows = 3, ncols = 2, figsize = (16, 12))

# Downsampling Volume
sns.lineplot(x = Petrignano['Date'], y = Petrignano['Volume'], color = 'green', ax = ax[0, 0])
ax[0, 0].set_title('Volume', fontsize = 14)

resampled_week = Petrignano[['Date', 'Volume']].resample('7D', on='Date').sum().reset_index()
sns.lineplot(x = resampled_week['Date'], y = resampled_week['Volume'], color = 'green', ax = ax[1, 0])
ax[0, 0].set_title('Weekly Volume', fontsize = 14)

resampled_month = Petrignano[['Date', 'Volume']].resample('M', on='Date').sum().reset_index()
sns.lineplot(x = resampled_month['Date'], y = resampled_month['Volume'], color = 'green', ax = ax[2, 0])
ax[0, 0].set_title('Monthly Volume', fontsize = 14)

for i in range(3):
    ax[i, 0].set_xlim([date(2009, 1, 1), date(2020, 6, 30)])
    
# Downsampling Temperature
sns.lineplot(x = Petrignano['Date'], y = Petrignano['Temperature'], color = 'green', ax = ax[0, 1])
ax[0, 0].set_title('Temperature', fontsize = 14)

resampled_week = Petrignano[['Date', 'Temperature']].resample('7D', on='Date').sum().reset_index()
sns.lineplot(x = resampled_week['Date'], y = resampled_week['Temperature'], color = 'green', ax = ax[1, 1])
ax[0, 0].set_title('Weekly Temperature', fontsize = 14)

resampled_month = Petrignano[['Date', 'Temperature']].resample('M', on='Date').sum().reset_index()
sns.lineplot(x = resampled_month['Date'], y = resampled_month['Temperature'], color = 'green', ax = ax[2, 1])
ax[0, 0].set_title('Monthly Temperature', fontsize = 14)

for i in range(3):
    ax[i, 0].set_xlim([date(2009, 1, 1), date(2020, 6, 30)])


# Weekly downsampling can smooth data and help in analysis
Petrignano = Petrignano[['Date', 'Rainfall', 'Depth_to_groundwater', 
             'Temperature', 'Volume', 'Hydrometry']].resample('7D', on='Date').sum().reset_index()


# Checking stationarity using Visual method
rolling_window = 52                 # our data is sampled weekly and a year has 52 week
fig, ax = plt.subplots(nrows = 4, ncols=1, figsize = (15, 12))

sns.lineplot(x = Petrignano['Date'], y = Petrignano['Volume'], ax = ax[0], color = 'darkgreen')
sns.lineplot(x = Petrignano['Date'], y = Petrignano['Volume'].rolling(rolling_window).mean(), ax = ax[0], 
             color = 'orange', label = 'Rolling mean')
sns.lineplot(x = Petrignano['Date'], y = Petrignano['Volume'].rolling(rolling_window).std(), ax = ax[0], 
             color = 'blue', label = 'Rolling variance')
ax[0].set_title('Volume', fontsize = 14)
ax[0].set_ylabel('Volume')

sns.lineplot(x = Petrignano['Date'], y = Petrignano['Rainfall'], ax = ax[1], color = 'darkgreen')
sns.lineplot(x = Petrignano['Date'], y = Petrignano['Rainfall'].rolling(rolling_window).mean(), ax = ax[1], 
             color = 'orange', label = 'Rolling mean')
sns.lineplot(x = Petrignano['Date'], y = Petrignano['Rainfall'].rolling(rolling_window).std(), ax = ax[1], 
             color = 'blue', label = 'Rolling variance')
ax[0].set_title('Rainfall', fontsize = 14)
ax[0].set_ylabel('Rainfall')

sns.lineplot(x = Petrignano['Date'], y = Petrignano['Temperature'], ax = ax[2], color = 'darkgreen')
sns.lineplot(x = Petrignano['Date'], y = Petrignano['Temperature'].rolling(rolling_window).mean(), ax = ax[2], 
             color = 'orange', label = 'Rolling mean')
sns.lineplot(x = Petrignano['Date'], y = Petrignano['Temperature'].rolling(rolling_window).std(), ax = ax[2], 
             color = 'blue', label = 'Rolling variance')
ax[0].set_title('Temperature', fontsize = 14)
ax[0].set_ylabel('Temperature')

sns.lineplot(x = Petrignano['Date'], y = Petrignano['Hydrometry'], ax = ax[3], color = 'darkgreen')
sns.lineplot(x = Petrignano['Date'], y = Petrignano['Hydrometry'].rolling(rolling_window).mean(), ax = ax[3], 
             color = 'orange', label = 'Rolling mean')
sns.lineplot(x = Petrignano['Date'], y = Petrignano['Hydrometry'].rolling(rolling_window).std(), ax = ax[3], 
             color = 'blue', label = 'Rolling variance')
ax[0].set_title('Hydrometry', fontsize = 14)
ax[0].set_ylabel('Hydrometry')

for i in range(4):
    ax[i].set_xlim([date(2009, 1, 1), date(2020, 6, 30)])


from statsmodels.tsa.stattools import adfuller

# Input to adfuller function is a series
adfuller(Petrignano['Rainfall'].values)


fig, ax = plt.subplots(nrows = 5, ncols = 1, figsize = (15, 12))

def visualize_adf_results(series, title, ax):
    result = adfuller(series)
    significance_level = 0.05
    adf_stat = result[0]
    p_value = result[1]
    critical_val_5 = result[4]['5%']
    
    if (p_value < significance_level) and (adf_stat < critical_val_5):
        linecolor = 'orange'
    else:
        linecolor = 'green'
        
    sns.lineplot(x = Petrignano['Date'], y = series, ax = ax, color = linecolor)
    ax.set_title(f'ADF Statistics: {adf_stat:0.3f}, p-value: {p_value:0.3f}, Critical Values 5%: {critical_val_5:0.3f}', fontsize = 14)
    ax.set_ylabel(title)
    
for i, title in enumerate(Petrignano.drop(['Date'],axis=1).columns):
    visualize_adf_results(Petrignano[title], title, ax[i])
    
plt.tight_layout()
plt.show()


# Applying log transform
# For negative values it will be Nan
# log_series = np.log(abs(Petrignano['Depth_to_groundwater']))

# fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 6))
# visualize_adf_results(log_series, 'Transformed Depth_to_groundwater', ax[0])
# sns.distplot(log_series, ax = ax[1])

# Conclusion: This transformation cannot make Petrignano stationary


# If you want to apply differencing
# Applying first order differencing
Petrignano['Depth_to_groundwater'] = np.append([0], np.diff(Petrignano['Depth_to_groundwater']))

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 6))
visualize_adf_results(Petrignano['Depth_to_groundwater'], 'Transformed Depth_to_groundwater', ax[0])
sns.distplot(Petrignano['Depth_to_groundwater'], ax = ax[1])


### Petrignano 데이터프레임의 시계열 데이터를 추세(trend)와 계절성(seasonal)으로 분해

from statsmodels.tsa.seasonal import seasonal_decompose

core_columns = Petrignano.drop(['Date'], axis = 1).columns

# period = 52인 이유? week로 집계했기 때문
# model = 'additive'는 데이터를 추세, 계절성, 잔차의 합으로 분해

for col in core_columns:
    decompose = seasonal_decompose(Petrignano[col], period = 52, model = 'additive', extrapolate_trend = 'freq')
    
    # It is a method to decompose a time series into a trend component, multiple seasonal components, and a residual component.
    # The seasonal decomposition is a method used in time series analysis to represent a time series as a sum 
    # (or, sometimes, a product) of three components – the linear trend, the periodic (seasonal) component, and random residuals. 
    
    Petrignano[f'{col}_trend'] = decompose.trend
    Petrignano[f'{col}_seasonal'] = decompose.seasonal


fig, ax = plt.subplots(nrows = 5, ncols = 4, sharex = True, figsize  = (22, 8))

def plot_seasonal_decompose(column, i):
    result = seasonal_decompose(Petrignano[column], period = 52, model = 'additive', extrapolate_trend = 'freq')
    
    ax[0, 0].set_title('Observed', fontsize = 14)
    result.observed.plot(ax = ax[i, 0], legend = False, color = 'blue')
    ax[i, 0].set_ylabel(column, fontsize = 10)
    
    ax[0, 1].set_title('Trend', fontsize = 14)
    result.trend.plot(ax = ax[i, 1], legend = False, color = 'green')
    
    ax[0, 2].set_title('Seasonal', fontsize = 14)
    result.seasonal.plot(ax = ax[i, 2], legend = False, color = 'orange')
    
    ax[0, 3].set_title('Residual', fontsize = 14)
    result.resid.plot(ax = ax[i, 3], legend = False, color = 'violet')
    

for i, col in enumerate(core_columns):
    plot_seasonal_decompose(col, i)


# Individual decomposition of features

f, ax = plt.subplots(ncols=1, nrows=1, figsize=(16, 8))

result = seasonal_decompose(Petrignano['Rainfall'], period = 52, model = 'additive', extrapolate_trend = 'freq')

sns.lineplot(x=Petrignano['Date'], y=result.seasonal, color='violet')
ax.set_xlim([date(2017, 9, 30), date(2020, 6, 30)])


corr_matrix = Petrignano[core_columns].corr()

sns.heatmap(corr_matrix, annot = True, vmin = -1, vmax = 1, cmap = 'coolwarm_r')
plt.title('Correlation Matrix', fontsize = 14)


# Using pandas
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(Petrignano['Depth_to_groundwater'])


# Using statmodels
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (16,8))

plot_acf(Petrignano['Depth_to_groundwater'], lags = 100, ax = ax[0])
plot_pacf(Petrignano['Depth_to_groundwater'], lags = 100, ax = ax[1])

plt.show()


features = ['Rainfall', 'Temperature', 'Volume', 'Hydrometry']
target = ['Depth_to_groundwater']

train_size = int(0.85 * len(Petrignano)) 

multivariate_Petrignano = Petrignano[['Date'] + target + features].copy()
multivariate_Petrignano.columns = ['ds', 'y'] + features
# Columns named ds i.e. Date and y i.e. Target
# Forecasting models themselves identify ds, y as date and Target value respectively from Petrignano.

train = multivariate_Petrignano.iloc[:train_size, :]

# Splitting Petrignano into training and validation dataset
X_train = pd.DataFrame(multivariate_Petrignano.iloc[:train_size, [0, 2, 3, 4, 5]])
Y_train = pd.DataFrame(multivariate_Petrignano.iloc[:train_size, 1])

X_valid = pd.DataFrame(multivariate_Petrignano.iloc[train_size: , [0, 2, 3, 4, 5]])
Y_valid = pd.DataFrame(multivariate_Petrignano.iloc[train_size: , 1])

train.head()


from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from colorama import Fore

# Training model
model = Prophet()
for col in features:
    model.add_regressor(col)
    
# Fit model
model.fit(train)

# Predict on validation set
y_pred = model.predict(X_valid)

# Calculate error
mae = mean_absolute_error(Y_valid, y_pred['yhat'])
rmse = np.sqrt(mean_squared_error(Y_valid, y_pred['yhat']))

print(Fore.GREEN + f'MAE: {mae}')
print(Fore.BLUE + f'RMSE: {rmse}')


fig, ax = plt.subplots(1, figsize = (12, 6))

model.plot(y_pred, ax = ax)
sns.lineplot(x = X_valid['ds'], y = Y_valid['y'], ax = ax, color = 'orange', label = 'Groud truth')

ax.set_title(f'Prediction\nMAE: {mae} RMSE: {rmse}', fontsize = 14)
ax.set_xlabel('Date', fontsize = 14)
ax.set_ylabel('Depth to groundwater', fontsize = 14)

plt.show()

