# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings("ignore")

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")


print("train shape :", train.shape)
print("test shape :", test.shape)


display(train.head())


print("before train isna")
print(train.isna().sum(), "\n")

before_row = len(train)
train = train.dropna().reset_index(drop = True)

print("after train isna")
print(train.isna().sum(), "\n")
print("drop train row :", before_row - len(train))


train.info()


train["date"] = pd.to_datetime(train["date"])
train["date_year"] = train["date"].dt.year
train["date_month"] = train["date"].dt.month
train["date_day"] = train["date"].dt.day
train["date_dayofweek"] = train["date"].dt.dayofweek
train["date_dayofyear"] = train["date"].dt.dayofyear
train["date_weekofyear"] = train["date"].dt.isocalendar().week
train["season"] = train["date"].dt.month % 12 // 3 + 1 # 1: Winter, 2: Spring, 3: Summer, 4: Fall
base_date = pd.to_datetime("2010-01-01")
train["date_since_base"] = (train["date"] - base_date).dt.days
train["num_sold"] = train["num_sold"].astype("int")


test["date"] = pd.to_datetime(test["date"])
test["date_year"] = test["date"].dt.year
test["date_month"] = test["date"].dt.month
test["date_day"] = test["date"].dt.day
test["date_dayofweek"] = test["date"].dt.dayofweek
test["date_dayofyear"] = test["date"].dt.dayofyear
test["date_weekofyear"] = test["date"].dt.isocalendar().week
test["season"] = test["date"].dt.month % 12 // 3 + 1 # 1: Winter, 2: Spring, 3: Summer, 4: Fall
base_date = pd.to_datetime("2010-01-01")
test["date_since_base"] = (test["date"] - base_date).dt.days


print("train object", "\n")
object_col = train.select_dtypes(include = ["object"]).columns.tolist()
print("object columns :", object_col, "\n")

for col in object_col:
    unique_values = train[col].unique()
    value_mapping = {value : i + 1 for i, value in enumerate(unique_values)}
    
    print(f"Unique values in column '{col}' :\n{train[col].unique()}\n")
    print(f"Mapping for column '{col}' :\n{value_mapping}\n")
    
    train[col] = train[col].map(value_mapping)

print("---------------------------------------------------------------------------------------------", "\n")

print("test object", "\n")
object_col = test.select_dtypes(include = ["object"]).columns.tolist()
print("object columns :", object_col, "\n")

for col in object_col:
    unique_values = test[col].unique()
    value_mapping = {value : i + 1 for i, value in enumerate(unique_values)}
    
    print(f"Unique values in column '{col}' :\n{test[col].unique()}\n")
    print(f"Mapping for column '{col}' :\n{value_mapping}\n")
    
    test[col] = test[col].map(value_mapping)


train.info()


test.info()


display(train.head())


display(test.head())


train = train.sort_values(by = "date")
train.set_index("date", inplace = True)

test = test.sort_values(by = "date")
test.set_index("date", inplace = True)


# 타겟 변수의 분포 확인
sns.histplot(train["num_sold"], bins=30, kde=True)
plt.title("Distribution of num_sold")
plt.show()


#IQR 방식으로 이상치 탐색
Q1 = train["num_sold"].quantile(0.25)
Q3 = train["num_sold"].quantile(0.75)

IQR = Q3 - Q1

outliers = train[(train["num_sold"] < (Q1 - 1.5 * IQR)) | (train["num_sold"] > (Q3 + 1.5 * IQR))]
print(f"Number of outliers : {len(outliers)}")


#시간에 따른 패턴 분석 (주기성 확인)
weekly_avg = train.resample("W")["num_sold"].mean()
monthly_avg = train.resample("M")["num_sold"].mean()

plt.figure(figsize=(12, 6))
plt.plot(weekly_avg, marker='o')
plt.title("Weekly Average Sales")
plt.xlabel("Date")
plt.ylabel("Average Sales")

plt.figure(figsize=(12, 6))
plt.plot(monthly_avg, marker="o")
plt.title("Monthly Average Sales")
plt.xlabel("Date")
plt.ylabel("Average Sales")

plt.show()


#주차, 월, 요일, 계절에 따른 패턴 분석
weekofyear_avg = train.groupby("date_weekofyear")["num_sold"].mean()
monthofyear_avg = train.groupby("date_month")["num_sold"].mean()
dayofweek_avg = train.groupby("date_dayofweek")["num_sold"].mean()
season_avg = train.groupby("season")["num_sold"].mean()

plt.figure(figsize = (8, 4))
sns.barplot(x = weekofyear_avg.index, y = weekofyear_avg.values)
plt.title("Average Sales by Week of Year")
plt.xlabel("Week of Year (1 = first week, 53 = last week)")
plt.ylabel("Average Sales")

plt.figure(figsize = (8, 4))
sns.barplot(x = monthofyear_avg.index, y = monthofyear_avg.values)
plt.title("Average Sales by Month of Year")
plt.xlabel("Month of Year (1 = january, 12 = december)")
plt.ylabel("Average Sales")

plt.figure(figsize = (8, 4))
sns.boxplot(x = "date_month", y = "num_sold", data = train)
plt.title("Monthly Sales Distribution")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.figure(figsize = (8, 4))
sns.barplot(x = dayofweek_avg.index, y = dayofweek_avg.values)
plt.title("Average Sales by Day of Week")
plt.xlabel("Day of Week (0=Monday, 6=Sunday)")
plt.ylabel("Average Sales")


plt.figure(figsize = (8, 4))
sns.barplot(x = season_avg.index, y = season_avg.values)
plt.title("Average Sales by Season")
plt.xlabel("Season (1 = spring, 4 = winter)")
plt.ylabel("Average Sales")

plt.show()


#계절성 분해
result = seasonal_decompose(train["num_sold"], model = "additive", period=365)
result.plot()
plt.show()


#국가, 가게, 제품별 평균 판매량
country_avg = train.groupby("country")["num_sold"].mean()
store_avg = train.groupby("store")["num_sold"].mean()
product_avg = train.groupby("product")["num_sold"].mean()

plt.figure(figsize = (8, 4))
sns.barplot(x = country_avg.index, y = country_avg.values)
plt.title("Average Sales by Country")
plt.xlabel("Country")
plt.ylabel("Average Sales")

plt.figure(figsize = (8, 4))
sns.barplot(x = store_avg.index, y = store_avg.values)
plt.title("Average Sales by Store")
plt.xlabel("Store")
plt.ylabel("Average Sales")

plt.figure(figsize = (8, 4))
sns.barplot(x = product_avg.index, y = product_avg.values)
plt.title("Average Sales by Product")
plt.xlabel("Product")
plt.ylabel("Average Sales")
plt.show()


#시계열 자기 상관성 분석
plot_acf(train['num_sold'], lags=50, alpha = 0.05)
plot_pacf(train['num_sold'], lags=50, alpha = 0.05)
plt.show()

